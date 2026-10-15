"""Leakage-free evaluation, calibration diagnostics, and explainability."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from .calibration import ProbabilityCalibrator
from .utils import get_logger

logger = get_logger()


class Evaluator:
    """Evaluate a fixed model and an already-fitted optional calibrator."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_type = config["model"]["type"]
        self.is_nn = self.model_type in ("neural_net", "hybrid_tcn")
        self.class_names = list(config.get("classes", {}).get("astro_names", []))
        self.n_classes = len(self.class_names) or 2
        self.positive_classes = self._positive_class_indices()

    def predict_proba(
        self, model, X: np.ndarray, flux: np.ndarray | None = None
    ) -> np.ndarray:
        """Return a complete, normalized class matrix for any model family."""
        if self.model_type == "hybrid_tcn":
            if flux is None:
                raise ValueError("Hybrid prediction requires temporal flux input")
            import torch

            device = next(model.parameters()).device
            with torch.inference_mode():
                logits = model(
                    torch.as_tensor(X, dtype=torch.float32, device=device),
                    torch.as_tensor(flux, dtype=torch.float32, device=device),
                )
                return torch.softmax(logits, dim=-1).cpu().numpy()

        if self.model_type == "neural_net":
            import torch

            device = next(model.parameters()).device
            with torch.inference_mode():
                logits = model(
                    torch.as_tensor(X, dtype=torch.float32, device=device)
                )
                return torch.softmax(logits, dim=-1).cpu().numpy()

        if not hasattr(model, "predict_proba"):
            raise TypeError("Model does not expose predict_proba")
        raw = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])), dtype=int)
        if np.any((classes < 0) | (classes >= self.n_classes)):
            raise ValueError(
                f"Model class IDs {classes.tolist()} do not match {self.n_classes} active classes"
            )
        output = np.zeros((len(X), self.n_classes), dtype=float)
        output[:, classes] = raw
        return _renormalize(output)

    def evaluate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        flux: np.ndarray | None = None,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> dict:
        """Compute classification and calibration metrics without fitting anything."""
        y = np.asarray(y, dtype=int)
        raw_proba = self.predict_proba(model, X, flux)
        proba = calibrator.calibrate(raw_proba) if calibrator is not None else raw_proba
        prediction = np.argmax(proba, axis=1)
        labels = np.arange(self.n_classes)
        per_precision = precision_score(
            y, prediction, labels=labels, average=None, zero_division=0
        )
        per_recall = recall_score(
            y, prediction, labels=labels, average=None, zero_division=0
        )
        per_f1 = f1_score(
            y, prediction, labels=labels, average=None, zero_division=0
        )
        support = np.bincount(y, minlength=self.n_classes)
        matrix = confusion_matrix(y, prediction, labels=labels)

        metrics = {
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_precision": float(
                precision_score(y, prediction, labels=labels, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y, prediction, labels=labels, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y, prediction, labels=labels, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y, prediction, labels=labels, average="weighted", zero_division=0)
            ),
            "per_class_precision": per_precision.tolist(),
            "per_class_recall": per_recall.tolist(),
            "per_class_f1": per_f1.tolist(),
            "per_class_support": support.tolist(),
            "class_names": self.class_names,
            "matthews_correlation": float(matthews_corrcoef(y, prediction)),
            "confusion_matrix": matrix.tolist(),
            "negative_log_likelihood": _safe_log_loss(y, proba, labels),
            "mean_predictive_entropy": float(
                np.mean(-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1))
            ),
            "calibration_applied": calibrator is not None,
            "proba": proba,
            "raw_proba": raw_proba,
        }
        if self.n_classes > 1:
            metrics["top_2_accuracy"] = _safe_top_k(
                y, proba, min(2, self.n_classes), labels
            )
        if self.n_classes > 2:
            metrics["top_3_accuracy"] = _safe_top_k(
                y, proba, min(3, self.n_classes), labels
            )
            metrics["macro_roc_auc_ovr"] = _safe_multiclass_auc(y, proba, labels)

        if self.positive_classes:
            metrics.update(self._positive_metrics(y, proba))
        else:
            metrics.update(
                {
                    "positive_class": "unavailable: no active planet class",
                    "positive_class_indices": [],
                    "decision_threshold": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "specificity": float("nan"),
                    "f1": float("nan"),
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                }
            )

        diagnostic = ProbabilityCalibrator("identity")
        bins = int(self.config.get("evaluation", {}).get("calibration_bins", 10))
        metrics["ece"] = diagnostic.expected_calibration_error(proba, y, bins)
        metrics["mce"] = diagnostic.maximum_calibration_error(proba, y, bins)
        metrics["brier_score"] = diagnostic.brier_score(proba, y)
        metrics["raw_ece"] = diagnostic.expected_calibration_error(raw_proba, y, bins)
        metrics["raw_brier_score"] = diagnostic.brier_score(raw_proba, y)
        logger.info(
            "Eval | balanced_acc=%.3f macro_f1=%.3f ece=%.3f",
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["ece"],
        )
        return metrics

    def shap_importance(self, model, X: np.ndarray, feature_names):
        if self.is_nn:
            logger.info("Tree SHAP unavailable for neural model; using permutation importance")
            return None
        try:
            import shap

            values = shap.TreeExplainer(model).shap_values(X)
            if isinstance(values, list):
                stacked = np.stack([np.asarray(value) for value in values], axis=0)
                importance = np.mean(np.abs(stacked), axis=(0, 1))
                return importance, feature_names
            values = np.asarray(values)
            if values.ndim == 3 and values.shape[1] == X.shape[1]:
                importance = np.mean(np.abs(values), axis=(0, 2))
            elif values.ndim == 3 and values.shape[2] == X.shape[1]:
                importance = np.mean(np.abs(values), axis=(0, 1))
            elif values.ndim == 2:
                importance = np.mean(np.abs(values), axis=0)
            else:
                return None
            return importance, feature_names
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.warning("SHAP failed: %s", exc)
            return None

    def permutation_importance(
        self, model, X, y, feature_names, flux=None, repeats: int = 3
    ):
        base = _base_score(y, self.predict_proba(model, X, flux))
        rng = np.random.default_rng(self.config.get("seed", 42))
        importances = np.zeros(X.shape[1], dtype=float)
        for feature_index in range(X.shape[1]):
            drops = []
            for _ in range(max(int(repeats), 1)):
                shuffled = X.copy()
                shuffled[:, feature_index] = rng.permutation(
                    shuffled[:, feature_index]
                )
                drops.append(
                    base
                    - _base_score(
                        y, self.predict_proba(model, shuffled, flux)
                    )
                )
            importances[feature_index] = float(np.mean(drops))
        return importances, feature_names

    def _positive_class_indices(self) -> list[int]:
        indices = [
            index
            for index, name in enumerate(self.class_names)
            if name in ("planet", "confirmed_planet", "planet_candidate")
        ]
        return indices

    def _positive_metrics(self, y: np.ndarray, proba: np.ndarray) -> dict:
        truth = np.isin(y, self.positive_classes).astype(int)
        probability = np.sum(proba[:, self.positive_classes], axis=1)
        threshold = float(self.config.get("inference", {}).get("planet_threshold", 0.5))
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        return {
            "positive_class": "planet_or_candidate",
            "positive_class_indices": self.positive_classes,
            "decision_threshold": threshold,
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "roc_auc": _safe_auc(roc_auc_score, truth, probability),
            "pr_auc": _safe_auc(average_precision_score, truth, probability),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }


def _renormalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    row_sum = np.sum(values, axis=1, keepdims=True)
    invalid = row_sum[:, 0] <= 0
    if np.any(invalid):
        values[invalid] = 1.0 / values.shape[1]
        row_sum = np.sum(values, axis=1, keepdims=True)
    return values / row_sum


def _safe_auc(function, y, probability):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(function(y, probability))


def _safe_multiclass_auc(y, proba, labels):
    if len(np.unique(y)) < len(labels):
        return float("nan")
    try:
        return float(
            roc_auc_score(y, proba, labels=labels, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def _safe_top_k(y, proba, k, labels):
    try:
        return float(top_k_accuracy_score(y, proba, k=k, labels=labels))
    except ValueError:
        return float("nan")


def _safe_log_loss(y, proba, labels):
    try:
        return float(log_loss(y, proba, labels=labels))
    except ValueError:
        return float("nan")


def _base_score(y, proba):
    prediction = np.argmax(proba, axis=1)
    return float(f1_score(y, prediction, average="macro", zero_division=0))
