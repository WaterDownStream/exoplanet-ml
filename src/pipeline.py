"""Unified end-to-end inference pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import ProbabilityCalibrator
from .classifier import AstrophysicalClassifier, ClassificationResult
from .data_loader import DataLoader
from .detrending import Detrender
from .feature_extraction import FEATURE_NAMES, FeatureExtractor, build_model_input
from .models import resample_flux
from .mono_features import MonoFeatureExtractor
from .mono_search import MonoTransitSearch
from .parameter_estimation import ParameterEstimator, ParameterReport
from .phase_folding import PhaseFolder
from .preprocessing import Preprocessor
from .quality import QualityAssessment, QualityReport
from .transit_search import BLSResult, TransitSearch
from .utils import LightCurve, get_logger

logger = get_logger()


@dataclass
class PipelineResult:
    """Complete pipeline output for one light curve."""

    lightcurve: LightCurve
    quality: QualityReport
    bls: BLSResult | None
    mono: object | None
    features: dict[str, float]
    feature_vector: np.ndarray
    feature_masks: np.ndarray
    is_mono: bool = False
    raw_lightcurve: LightCurve | None = None
    cleaned_lightcurve: LightCurve | None = None
    classification: ClassificationResult | None = None
    parameters: ParameterReport | None = None
    final_confidence: float = 0.0
    false_positive_diagnosis: dict | None = None
    calibrated_probabilities: dict[str, float] = field(default_factory=dict)
    confidence_components: dict[str, float | str | bool | None] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    flux_resampled: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


class Pipeline:
    """Orchestrate load -> quality -> preprocess -> detrend -> search -> features."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.loader = DataLoader()
        self.preprocessor = Preprocessor(config["preprocessing"])
        self.quality = QualityAssessment(config.get("quality", {}))
        self.detrender = Detrender(config["detrending"])
        search_cfg = dict(config["bls"])
        search_cfg["tls_enabled"] = config.get("tls", {}).get("enabled", False)
        self.transit_search = TransitSearch(search_cfg)
        self.mono_search = MonoTransitSearch(config.get("mono_search", {}))
        feat_cfg = dict(config["features"])
        feat_cfg["phase_folding"] = config.get("phase_folding", {})
        self.feature_extractor = FeatureExtractor(feat_cfg)
        self.mono_features = MonoFeatureExtractor()
        self.classifier_helper = AstrophysicalClassifier(config)
        param_cfg = dict(config.get("parameter_estimation", {}))
        param_cfg.setdefault("seed", config.get("seed", 42))
        self.param_estimator = ParameterEstimator(param_cfg)
        self.calibrator = ProbabilityCalibrator(
            config.get("calibration", {}).get("method", "isotonic")
        )
        self.resample_len = config.get("features", {}).get("resample_length", 1024)

    def run(
        self,
        path: str | Path,
        model=None,
        scaler=None,
        imputer=None,
        selector=None,
        is_hybrid: bool = False,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> PipelineResult:
        """Execute full pipeline on one file."""
        lc_raw = self.loader.load(path)
        return self.run_lightcurve(
            lc_raw,
            source_file=str(path),
            model=model,
            scaler=scaler,
            imputer=imputer,
            selector=selector,
            is_hybrid=is_hybrid,
            calibrator=calibrator,
        )

    def run_lightcurve(
        self,
        lc_raw: LightCurve,
        source_file: str | Path = "memory",
        model=None,
        scaler=None,
        imputer=None,
        selector=None,
        is_hybrid: bool = False,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> PipelineResult:
        """Execute the pipeline on an already loaded light curve."""
        meta = {
            key: value
            for key, value in lc_raw.meta.items()
            if key != "quality_flags"
        }
        meta["source_file"] = str(source_file)
        meta["tic_id"] = meta.get("tic_id", meta.get("targetid", Path(source_file).stem))

        try:
            lc = self.preprocessor.run(lc_raw)
        except ValueError as exc:
            quality = QualityReport(
                score=0.0,
                passed=False,
                warnings=[str(exc)],
                metrics={"n_points": float(len(lc_raw.time))},
                rejection_reason="preprocessing failed",
            )
            return PipelineResult(
                lightcurve=lc_raw,
                quality=quality,
                bls=None,
                mono=None,
                features={},
                feature_vector=np.full(len(FEATURE_NAMES), np.nan),
                feature_masks=np.zeros(len(FEATURE_NAMES)),
                raw_lightcurve=lc_raw,
                metadata=meta,
                warnings=[f"preprocessing failed: {exc}"],
            )

        quality = self.quality.run(lc)
        if not quality.passed:
            return PipelineResult(
                lightcurve=lc,
                quality=quality,
                bls=None,
                mono=None,
                features={},
                feature_vector=np.full(len(FEATURE_NAMES), np.nan),
                feature_masks=np.zeros(len(FEATURE_NAMES)),
                is_mono=False,
                raw_lightcurve=lc_raw,
                cleaned_lightcurve=lc,
                metadata=meta,
                warnings=[quality.rejection_reason or "quality check failed"],
            )

        lc_det = self.detrender.run(lc)
        bls = self.transit_search.run(lc_det)
        mono = self.mono_search.run(lc_det)

        is_mono = bool(mono.detected and not bls.detected)
        feats = self.feature_extractor.extract(
            lc_det,
            bls,
            quality.metrics,
            is_mono=is_mono,
            mono_result=mono,
        )
        vec = self.feature_extractor.vectorize(feats, apply_weights=False)
        masks = self.feature_extractor.masks(feats)
        flux_rs = resample_flux(lc_det.flux, self.resample_len)

        result = PipelineResult(
            lightcurve=lc_det,
            quality=quality,
            bls=bls,
            mono=mono,
            features=feats,
            feature_vector=vec,
            feature_masks=masks,
            is_mono=is_mono,
            raw_lightcurve=lc_raw,
            cleaned_lightcurve=lc,
            metadata=meta,
            flux_resampled=flux_rs,
        )
        if not bls.detected and not mono.detected:
            result.warnings.append("no statistically significant transit candidate")
        result.metadata["candidate_type"] = (
            "mono" if is_mono else "periodic" if bls.detected else "none"
        )

        if model is not None and scaler is not None:
            result = self._predict(
                result, model, scaler, imputer, selector, is_hybrid, calibrator
            )

        return result

    def _predict(
        self,
        result: PipelineResult,
        model,
        scaler,
        imputer,
        selector,
        is_hybrid: bool,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> PipelineResult:
        expected_dim = getattr(
            imputer,
            "n_features_in_",
            len(FEATURE_NAMES) * 2
            if self.config.get("features", {}).get("use_masks", True)
            else len(FEATURE_NAMES),
        )
        use_masks = expected_dim == len(FEATURE_NAMES) * 2
        x_vec = build_model_input(
            result.feature_vector,
            result.feature_masks,
            use_masks=use_masks,
        )
        if len(x_vec) != expected_dim:
            raise ValueError(
                f"Model expects {expected_dim} inputs but pipeline produced {len(x_vec)}"
            )
        x = x_vec.reshape(1, -1)
        if imputer is not None:
            x = imputer.transform(x)
        else:
            if np.any(~np.isfinite(x)):
                logger.warning(
                    "Legacy model bundle has no imputer; replacing missing values with zero"
                )
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if selector is not None:
            x = selector.transform(x)
        scaler_dim = getattr(scaler, "n_features_in_", x.shape[1])
        if x.shape[1] != scaler_dim:
            raise ValueError(
                f"Scaler expects {scaler_dim} selected inputs but received {x.shape[1]}"
            )
        x_scaled = scaler.transform(x)

        epistemic_std = float("nan")
        if is_hybrid:
            import torch
            device = next(model.parameters()).device
            with torch.inference_mode():
                flux = torch.tensor(
                    result.flux_resampled.reshape(1, -1),
                    dtype=torch.float32,
                    device=device,
                )
                xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)
                logits = model(xt, flux)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            epistemic_std = self._mc_dropout_uncertainty(
                model, xt, flux, int(self.config.get("inference", {}).get("mc_dropout_samples", 10))
            )
        elif self.config["model"]["type"] == "neural_net":
            import torch
            device = next(model.parameters()).device
            with torch.inference_mode():
                xt = torch.tensor(x_scaled, dtype=torch.float32, device=device)
                logits = model(xt)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            epistemic_std = self._mc_dropout_uncertainty(
                model, xt, None, int(self.config.get("inference", {}).get("mc_dropout_samples", 10))
            )
        else:
            raw = np.asarray(model.predict_proba(x_scaled)[0], dtype=float)
            model_classes = np.asarray(
                getattr(model, "classes_", np.arange(len(raw))), dtype=int
            )
            proba = np.zeros(len(self.classifier_helper.class_names), dtype=float)
            if np.any((model_classes < 0) | (model_classes >= len(proba))):
                raise ValueError("Saved model class IDs do not match model metadata")
            proba[model_classes] = raw
            proba /= np.sum(proba)
            epistemic_std = self._tree_ensemble_uncertainty(
                model, x_scaled, int(np.argmax(proba)), len(proba)
            )

        if self.config.get("calibration", {}).get("enabled", True):
            active_calibrator = calibrator or self.calibrator
            cal = active_calibrator.calibrate(proba.reshape(1, -1))[0]
        else:
            cal = proba

        classification = self.classifier_helper.classify_from_proba(cal)

        planet_prob = classification.transit_gate_prob
        if result.is_mono and result.mono is not None:
            params = self.param_estimator.estimate_mono(
                result.mono,
                stellar=result.lightcurve.meta.get("stellar", {}),
                planet_prob=planet_prob,
            )
        elif result.bls is not None and result.bls.detected:
            params = self.param_estimator.estimate(
                result.lightcurve, result.bls, planet_prob=planet_prob
            )
        else:
            params = self.param_estimator._unavailable(
                "no statistically significant transit candidate"
            )
        final_conf = self._final_confidence(
            float(cal[classification.class_id]),
            result,
            classification,
            params,
        )

        result.classification = classification
        result.parameters = params
        result.final_confidence = final_conf
        result.confidence_components = {
            "reported_probability": final_conf,
            "calibrated": bool(
                calibrator is not None and getattr(calibrator, "is_fitted", False)
            ),
            "quality_score": float(result.quality.score),
            "template_agreement": float(self._template_agreement(result)),
            "parameter_uncertainty_score": float(
                self._uncertainty_score(classification, params)
            ),
            "probability_margin": float(classification.probability_margin),
            "normalized_entropy": float(classification.normalized_entropy),
            "epistemic_probability_std": (
                float(epistemic_std) if np.isfinite(epistemic_std) else None
            ),
        }
        result.calibrated_probabilities = classification.probabilities
        result.false_positive_diagnosis = self.classifier_helper.diagnose_false_positive(
            classification
        )
        result.warnings.extend(result.quality.warnings)
        result.warnings.extend(params.warnings)
        if classification.abstained:
            result.warnings.append(
                f"classifier abstained: {classification.abstention_reason}"
            )
        return result

    @staticmethod
    def _tree_ensemble_uncertainty(
        model, X: np.ndarray, class_id: int, n_classes: int
    ) -> float:
        estimators = getattr(model, "estimators_", None)
        if estimators is None or len(estimators) < 2:
            return float("nan")
        if not hasattr(estimators[0], "predict_proba"):
            return float("nan")
        probabilities = []
        for estimator in estimators:
            raw = np.asarray(estimator.predict_proba(X)[0], dtype=float)
            classes = np.asarray(estimator.classes_, dtype=int)
            aligned = np.zeros(n_classes, dtype=float)
            aligned[classes] = raw
            probabilities.append(aligned[class_id])
        return float(np.std(probabilities, ddof=1))

    @staticmethod
    def _mc_dropout_uncertainty(model, features, flux, samples: int) -> float:
        if samples < 2:
            return float("nan")
        import torch

        dropout_modules = [
            module
            for module in model.modules()
            if isinstance(
                module,
                (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d),
            )
        ]
        if not dropout_modules:
            return float("nan")
        for module in dropout_modules:
            module.train()
        predictions = []
        with torch.inference_mode():
            for _ in range(samples):
                logits = model(features, flux) if flux is not None else model(features)
                predictions.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
        model.eval()
        matrix = np.asarray(predictions)
        predicted_class = int(np.argmax(np.mean(matrix, axis=0)))
        return float(np.std(matrix[:, predicted_class], ddof=1))

    def _final_confidence(
        self,
        classifier_probability: float,
        result: PipelineResult,
        classification: ClassificationResult,
        params: ParameterReport,
    ) -> float:
        """Return the calibrated class probability without ad-hoc score mixing.

        Quality, template agreement, and parameter uncertainty are reported as
        separate evidence.  Averaging them into a probability destroys
        calibration and can make a weak classifier look more certain.
        """
        del result, params
        if classification.abstained:
            return 0.0
        return float(np.clip(classifier_probability, 0.0, 1.0))

    @staticmethod
    def _template_agreement(result: PipelineResult) -> float:
        scores: list[float] = []
        template_score = result.features.get("template_planet_score")
        if template_score is not None and np.isfinite(template_score):
            scores.append(float(np.clip(template_score, 0.0, 1.0)))
        if result.bls is not None:
            power = max(float(result.bls.best_power), 0.0)
            scores.append(power / (1.0 + power))
        if result.mono is not None and getattr(result.mono, "detected", False):
            best = getattr(result.mono, "best", None)
            if best is not None:
                scores.append(float(np.clip(best.morphology_score, 0.0, 1.0)))
        return float(np.mean(scores)) if scores else 0.5

    @staticmethod
    def _uncertainty_score(
        classification: ClassificationResult,
        params: ParameterReport,
    ) -> float:
        if not classification.transit_like:
            return 1.0
        rel_values: list[float] = []
        for estimate in (params.period, params.duration, params.depth):
            if (
                estimate.available
                and estimate.value not in (None, 0)
                and estimate.uncertainty is not None
                and np.isfinite(estimate.uncertainty)
            ):
                rel_values.append(abs(float(estimate.uncertainty) / float(estimate.value)))
        if not rel_values:
            return 0.5
        return float(1.0 / (1.0 + np.median(rel_values)))
