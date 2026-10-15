"""Run the full detection pipeline on a single light-curve file."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from src.pipeline import Pipeline
from src.feature_extraction import build_model_input
from src.phase_folding import PhaseFolder
from src.reporting import ReportGenerator
from src.utils import ensure_dirs, get_logger, load_config, set_seed
from src.visualizer import Visualizer


def load_model_bundle(config: dict) -> dict:
    """Load saved model, scaler, metadata, and optional calibrator."""
    d = Path(config["paths"]["models_dir"])
    scaler = joblib.load(d / "scaler.joblib")
    imputer = joblib.load(d / "imputer.joblib") if (d / "imputer.joblib").exists() else None
    selector = joblib.load(d / "selector.joblib") if (d / "selector.joblib").exists() else None
    meta = joblib.load(d / "model_meta.joblib") if (d / "model_meta.joblib").exists() else {}
    if meta and int(meta.get("artifact_version", 1)) < 2:
        raise RuntimeError(
            "Saved model artifacts predate the leakage-safe scientific pipeline; "
            "rebuild the dataset and retrain before inference"
        )
    configured_feature_version = str(config.get("dataset", {}).get("version", ""))
    if meta.get("feature_version") and str(meta["feature_version"]) != configured_feature_version:
        raise RuntimeError(
            f"Model feature version {meta['feature_version']} does not match configured "
            f"version {configured_feature_version}; rebuild and retrain"
        )
    mtype = meta.get("type", config["model"]["type"])
    n_classes = meta.get("n_classes", len(config.get("classes", {}).get("astro_names", [])))
    input_dim = int(meta.get("input_dim") or getattr(scaler, "n_features_in_", 0))
    saved_model_config = meta.get("model_config", config["model"])
    if input_dim <= 0:
        from src.feature_extraction import FEATURE_NAMES

        input_dim = len(FEATURE_NAMES)
    calibrator = (
        joblib.load(d / "calibrator.joblib")
        if config.get("calibration", {}).get("enabled", True)
        and (d / "calibrator.joblib").exists()
        else None
    )

    if mtype == "hybrid_tcn":
        import torch
        from src.models import HybridTCNClassifier

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridTCNClassifier(input_dim, n_classes, saved_model_config).to(device)
        model.load_state_dict(
            torch.load(d / "hybrid_tcn.pt", map_location=device, weights_only=True)
        )
        model.eval()
        return {
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "selector": selector,
            "is_hybrid": True,
            "type": mtype,
            "meta": meta,
            "calibrator": calibrator,
        }

    if mtype == "neural_net":
        import torch
        from src.models import TransitNet

        c = saved_model_config["neural_net"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransitNet(
            input_dim, list(c["hidden_layers"]), c["dropout"], n_classes
        ).to(device)
        model.load_state_dict(
            torch.load(d / "nn_model.pt", map_location=device, weights_only=True)
        )
        model.eval()
        return {
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "selector": selector,
            "is_hybrid": False,
            "type": mtype,
            "meta": meta,
            "calibrator": calibrator,
        }

    model = joblib.load(d / f"{mtype}.joblib")
    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "selector": selector,
        "is_hybrid": False,
        "type": mtype,
        "meta": meta,
        "calibrator": calibrator,
    }


def load_model(config: dict):
    """Backward-compatible loader returning ``(model, scaler, is_hybrid)``."""
    bundle = load_model_bundle(config)
    return bundle["model"], bundle["scaler"], bundle["is_hybrid"]


def _top_feature_values(features: dict[str, float], n: int = 10) -> list[tuple[str, float]]:
    finite = [
        (name, float(value))
        for name, value in features.items()
        if np.isfinite(value)
    ]
    return sorted(finite, key=lambda item: abs(item[1]), reverse=True)[:n]


def _explanation_payload(result, calibrator, local_explanation: dict | None = None) -> dict:
    features = result.features or {}
    params = result.parameters
    posterior = {}
    if params is not None:
        posterior = {
            "credible_intervals": params.credible_intervals,
            "warnings": params.warnings,
        }

    template = {
        "template_planet_score": _finite_or_none(features.get("template_planet_score")),
        "template_residual_rms": _finite_or_none(features.get("template_residual_rms")),
        "harmonic_score": _finite_or_none(features.get("period_harmonic_score")),
    }
    confidence = {
        "final_confidence": result.final_confidence,
        "quality_score": result.quality.score,
        "calibration": (
            calibrator.method
            if calibrator is not None and getattr(calibrator, "is_fitted", False)
            else "uncalibrated"
        ),
        "is_mono_transit": result.is_mono,
        "classifier_entropy": (
            result.classification.entropy if result.classification is not None else None
        ),
        "transit_gate_probability": (
            result.classification.transit_gate_prob
            if result.classification is not None else None
        ),
    }
    return {
        "top_engineered_features": _top_feature_values(features, 15),
        "local_model_explanation": local_explanation or {
            "method": "unavailable",
            "contributions": [],
        },
        "template_similarity": template,
        "confidence_explanation": confidence,
        "posterior_summary": posterior,
        "alternative_hypotheses": (
            result.classification.top_classes[1:]
            if result.classification is not None else []
        ),
        "false_positive_evidence": (
            (result.false_positive_diagnosis or {}).get("evidence", [])
            if result.false_positive_diagnosis else []
        ),
        "missing_capabilities": _runtime_explanation_limits(result),
    }


def _runtime_explanation_limits(result) -> list[str]:
    limits = []
    if result.classification is None:
        limits.append("model explanations unavailable because no trained model was loaded")
    if result.flux_resampled is None:
        limits.append("TCN saliency unavailable because no raw-flux branch input was produced")
    return limits


def _finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _local_feature_explanation(bundle: dict | None, result) -> dict:
    """Compute per-prediction tree SHAP or gradient-times-input attribution."""
    if bundle is None or result.classification is None:
        return {"method": "unavailable", "contributions": []}
    meta = bundle.get("meta", {})
    input_names = list(meta.get("input_names", []))
    use_masks = bool(meta.get("use_masks", True))
    raw = build_model_input(
        result.feature_vector, result.feature_masks, use_masks=use_masks
    ).reshape(1, -1)
    imputer = bundle.get("imputer")
    if imputer is not None:
        raw = imputer.transform(raw)
    else:
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    selector = bundle.get("selector")
    if selector is not None:
        raw = selector.transform(raw)
    scaled = bundle["scaler"].transform(raw)
    if not input_names:
        input_names = [f"feature_{index}" for index in range(scaled.shape[1])]

    model = bundle["model"]
    class_id = int(result.classification.class_id)
    temporal_saliency = []
    try:
        if bundle["type"] in ("neural_net", "hybrid_tcn"):
            import torch

            device = next(model.parameters()).device
            model.zero_grad(set_to_none=True)
            features = torch.as_tensor(
                scaled, dtype=torch.float32, device=device
            ).requires_grad_(True)
            if bundle["type"] == "hybrid_tcn":
                temporal = torch.as_tensor(
                    result.flux_resampled.reshape(1, -1),
                    dtype=torch.float32,
                    device=device,
                ).requires_grad_(True)
                logits = model(features, temporal)
            else:
                temporal = None
                logits = model(features)
            logits[0, class_id].backward()
            contribution = (
                features.grad.detach().cpu().numpy()[0]
                * features.detach().cpu().numpy()[0]
            )
            method = "gradient_times_input"
            if temporal is not None and temporal.grad is not None:
                saliency = np.abs(
                    temporal.grad.detach().cpu().numpy()[0]
                    * temporal.detach().cpu().numpy()[0]
                )
                for index in np.argsort(saliency)[-10:][::-1]:
                    temporal_saliency.append(
                        {
                            "normalized_position": float(index / max(len(saliency) - 1, 1)),
                            "saliency": float(saliency[index]),
                        }
                    )
        else:
            import shap

            values = shap.TreeExplainer(model).shap_values(scaled)
            contribution = _select_shap_vector(
                values, class_id, scaled.shape[1], len(bundle["meta"].get("class_names", []))
            )
            method = "tree_shap"
    except Exception as exc:  # explanation failure must not suppress the prediction
        return {
            "method": "unavailable",
            "contributions": [],
            "reason": str(exc),
        }

    aggregated: dict[str, float] = {}
    for name, value in zip(input_names, contribution):
        base = str(name).removeprefix("mask__")
        aggregated[base] = aggregated.get(base, 0.0) + float(value)
    ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:15]
    contributions = [
        {
            "feature": name,
            "contribution": value,
            "value": _finite_or_none(result.features.get(name)),
            "available": bool(
                result.feature_masks[list(result.features).index(name)]
            ) if name in result.features else None,
        }
        for name, value in ranked
    ]
    return {
        "method": method,
        "target_class": result.classification.class_name,
        "contributions": contributions,
        "temporal_saliency": temporal_saliency,
    }


def _select_shap_vector(values, class_id: int, n_features: int, n_classes: int) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[class_id], dtype=float).reshape(-1, n_features)[0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape == (1, n_features):
        return array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        if array.shape[1] == n_features:
            return array[0, :, class_id]
        if array.shape[2] == n_features:
            return array[0, class_id, :]
    if array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == n_features:
        return array[class_id, 0, :]
    raise ValueError(
        f"Unsupported SHAP shape {array.shape} for {n_features} features and {n_classes} classes"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exoplanet inference")
    parser.add_argument("lightcurve")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger(config.get("log_level", "INFO"))
    set_seed(config["seed"])
    ensure_dirs(
        config["paths"]["processed_dir"],
        config["paths"].get("figures_dir", "data/processed/figures"),
        config["paths"].get("reports_dir", "data/reports"),
    )

    bundle = None
    model, scaler, imputer, selector, is_hybrid, calibrator = (
        None,
        None,
        None,
        None,
        config["model"]["type"] == "hybrid_tcn",
        None,
    )
    model_dir = Path(config["paths"]["models_dir"])
    if (model_dir / "scaler.joblib").exists():
        bundle = load_model_bundle(config)
        config["model"]["type"] = bundle["type"]
        if bundle["meta"].get("class_names"):
            config.setdefault("classes", {})["astro_names"] = bundle["meta"]["class_names"]
        elif bundle["meta"].get("n_classes") == 2:
            config.setdefault("classes", {})["astro_names"] = ["unknown", "planet"]
        if bundle["meta"].get("planet_threshold") is not None:
            threshold = float(bundle["meta"]["planet_threshold"])
            config.setdefault("inference", {})["planet_threshold"] = threshold
            config.setdefault("parameter_estimation", {})["planet_threshold"] = threshold
        model = bundle["model"]
        scaler = bundle["scaler"]
        imputer = bundle["imputer"]
        selector = bundle["selector"]
        is_hybrid = bundle["is_hybrid"]
        calibrator = bundle["calibrator"]

    pipeline = Pipeline(config)
    result = pipeline.run(
        args.lightcurve,
        model=model,
        scaler=scaler,
        imputer=imputer,
        selector=selector,
        is_hybrid=is_hybrid,
        calibrator=calibrator,
    )
    local_explanation = _local_feature_explanation(bundle, result)

    if not result.quality.passed:
        logger.warning(
            "Quality check failed: %s (score %.2f)",
            result.quality.rejection_reason,
            result.quality.score,
        )
        if config.get("inference", {}).get("generate_report", True):
            payload = {
                "target_id": result.metadata.get("tic_id", Path(args.lightcurve).stem),
                "predicted_class": "unavailable",
                "final_confidence": 0.0,
                "class_probabilities": {},
                "quality_score": result.quality.score,
                "quality_metrics": result.quality.metrics,
                "warnings": result.warnings,
                "processing_metadata": result.metadata,
                "failure_reason": result.quality.rejection_reason,
            }
            ReportGenerator(config).generate(payload["target_id"], payload)
        return

    bls = result.bls
    duration_hours = bls.best_duration * 24.0 if bls else 0.0

    if result.classification:
        cls = result.classification
        logger.info("Predicted class: %s", cls.class_name)
        logger.info("Confidence: %.1f%%", result.final_confidence * 100)
        logger.info("Class probabilities:")
        for name, prob in cls.probabilities.items():
            logger.info("  %-25s %.1f%%", name, prob * 100)
    else:
        logger.info("No trained model found; feature extraction only")
        logger.info("BLS period: %.2f days", bls.best_period)
        logger.info("Transit depth: %.4f", bls.best_depth)
        logger.info("Transit duration: %.1f hours", duration_hours)

    if result.mono and result.mono.detected:
        logger.info("Mono-transit SNR: %.1f", result.mono.best.snr)

    figures_dir = Path(config["paths"].get("figures_dir", "data/processed/figures"))
    viz = Visualizer(output_dir=str(figures_dir))
    fig_paths = {}
    if result.raw_lightcurve is not None:
        viz.light_curve(result.raw_lightcurve, "Raw light curve", "raw_light_curve.png")
        fig_paths["raw_light_curve"] = figures_dir / "raw_light_curve.png"
    if result.cleaned_lightcurve is not None:
        viz.light_curve(result.cleaned_lightcurve, "Cleaned light curve", "cleaned_light_curve.png")
        fig_paths["cleaned_light_curve"] = figures_dir / "cleaned_light_curve.png"
    viz.light_curve(result.lightcurve, "Detrended light curve", "detrended_light_curve.png")
    fig_paths["detrended_light_curve"] = figures_dir / "detrended_light_curve.png"
    if bls:
        viz.periodogram(bls.periods, bls.power, bls.best_period)
        fig_paths["periodogram"] = figures_dir / "periodogram.png"
        folder = PhaseFolder(config["phase_folding"])
        ph, fl = folder.fold(result.lightcurve, bls.best_period, bls.best_t0)
        ctr, binned = folder.bin(ph, fl)
        viz.phase_fold(ph, fl, ctr, binned)
        fig_paths["phase_fold"] = figures_dir / "phase_fold.png"
        model_flux = np.interp(ph, ctr, np.nan_to_num(binned, nan=np.nanmedian(fl)))
        viz.residuals(ph, fl - model_flux)
        fig_paths["residuals"] = figures_dir / "residuals.png"
    contribution_rows = local_explanation.get("contributions", [])
    if contribution_rows:
        viz.importance(
            [row["contribution"] for row in contribution_rows],
            [row["feature"] for row in contribution_rows],
            name="prediction_feature_contributions.png",
            top_n=15,
            aggregate_masks=False,
        )
        fig_paths["prediction_feature_contributions"] = (
            figures_dir / "prediction_feature_contributions.png"
        )

    if config.get("inference", {}).get("generate_report", True):
        payload = {
            "target_id": result.metadata.get("tic_id", Path(args.lightcurve).stem),
            "predicted_class": result.classification.class_name if result.classification else "unknown",
            "final_confidence": result.final_confidence,
            "class_probabilities": result.classification.probabilities if result.classification else {},
            "alternative_classes": result.classification.top_classes[1:] if result.classification else [],
            "false_positive_diagnosis": result.false_positive_diagnosis,
            "quality_score": result.quality.score,
            "quality_metrics": result.quality.metrics,
            "warnings": result.warnings,
            "processing_metadata": result.metadata,
            "calibration": (
                calibrator.method
                if calibrator is not None and getattr(calibrator, "is_fitted", False)
                else "uncalibrated"
            ),
            "is_calibrated": bool(
                calibrator is not None and getattr(calibrator, "is_fitted", False)
            ),
            "confidence_components": result.confidence_components,
            "features": {k: float(v) for k, v in result.features.items() if np.isfinite(v)},
            "top_feature_values": _top_feature_values(result.features),
            "explanation": _explanation_payload(
                result, calibrator, local_explanation
            ),
        }
        if result.parameters:
            payload["parameters"] = asdict(result.parameters)
        if bls:
            payload["bls"] = {
                "period": bls.best_period,
                "depth": bls.best_depth,
                "duration": bls.best_duration,
                "power": bls.best_power,
            }
        reporter = ReportGenerator(config)
        paths = reporter.generate(payload["target_id"], payload, fig_paths)
        logger.info("Report: %s", paths.get("json", paths.get("html")))

    logger.info("Inference complete")


if __name__ == "__main__":
    main()
