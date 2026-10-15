"""Held-out probability calibration and calibration diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .utils import get_logger

logger = get_logger()


class ProbabilityCalibrator:
    """Calibrate a complete binary or multiclass probability vector.

    Calibration must be fitted on data that were not used to fit model weights.
    Temperature scaling is the default for multiclass and small calibration
    sets because it has one parameter.  Non-parametric isotonic calibration is
    rejected when any class lacks adequate support.
    """

    def __init__(
        self,
        method: str = "temperature",
        min_samples_per_class: int = 10,
    ) -> None:
        self.method = str(method).lower()
        self.min_samples_per_class = max(int(min_samples_per_class), 2)
        self._calibrator = None
        self.temperature = 1.0
        self.n_classes: int | None = None
        self.is_fitted = False

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        proba = _probability_matrix(proba)
        y = np.asarray(y, dtype=int)
        if len(proba) != len(y) or len(y) == 0:
            raise ValueError("Calibration probabilities and labels must have equal non-zero length")
        self.n_classes = int(proba.shape[1])
        if np.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("Calibration labels fall outside the probability columns")
        counts = np.bincount(y, minlength=self.n_classes)
        if np.any(counts == 0):
            raise ValueError(
                f"Calibration set does not contain every active class: counts={counts.tolist()}"
            )

        method = self.method
        if method == "auto":
            method = "temperature" if self.n_classes > 2 else "platt"
            self.method = method
        if method == "isotonic" and np.any(counts < self.min_samples_per_class):
            raise ValueError(
                "Isotonic calibration is too flexible for this calibration set; "
                f"need at least {self.min_samples_per_class} samples per class"
            )
        if method not in ("identity", "none", "isotonic") and np.any(
            counts < self.min_samples_per_class
        ):
            raise ValueError(
                "Calibration set is under-supported; "
                f"need at least {self.min_samples_per_class} samples per active class, "
                f"observed {counts.tolist()}"
            )

        if method == "isotonic":
            self._calibrator = []
            for class_id in range(self.n_classes):
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(proba[:, class_id], (y == class_id).astype(float))
                self._calibrator.append(calibrator)
        elif method == "platt":
            self._calibrator = LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=0,
            )
            self._calibrator.fit(np.log(np.clip(proba, 1e-12, 1.0)), y)
        elif method == "temperature":
            self.temperature = _fit_temperature(proba, y)
        elif method in ("identity", "none"):
            pass
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        self.is_fitted = True
        return self

    def calibrate(self, proba: np.ndarray) -> np.ndarray:
        matrix = _probability_matrix(proba)
        if not self.is_fitted:
            return matrix
        if self.n_classes is not None and matrix.shape[1] != self.n_classes:
            raise ValueError(
                f"Calibrator expects {self.n_classes} classes, received {matrix.shape[1]}"
            )

        if self.method == "isotonic" and self._calibrator:
            output = np.column_stack(
                [model.predict(matrix[:, index]) for index, model in enumerate(self._calibrator)]
            )
            return _renormalize(output)
        if self.method == "platt" and self._calibrator is not None:
            output = self._calibrator.predict_proba(
                np.log(np.clip(matrix, 1e-12, 1.0))
            )
            aligned = np.zeros_like(matrix)
            aligned[:, np.asarray(self._calibrator.classes_, dtype=int)] = output
            return _renormalize(aligned)
        if self.method == "temperature":
            logits = np.log(np.clip(matrix, 1e-12, 1.0)) / self.temperature
            logits -= np.max(logits, axis=1, keepdims=True)
            return _renormalize(np.exp(logits))
        return matrix

    def expected_calibration_error(
        self, proba: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> float:
        confidence, prediction = _confidence_and_prediction(proba)
        y = np.asarray(y)
        bins = np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)
        ece = 0.0
        for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            selected = (confidence >= lower) & (
                confidence <= upper if index == len(bins) - 2 else confidence < upper
            )
            if np.any(selected):
                accuracy = np.mean(y[selected] == prediction[selected])
                ece += np.mean(selected) * abs(accuracy - np.mean(confidence[selected]))
        return float(ece)

    def maximum_calibration_error(
        self, proba: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> float:
        confidence, prediction = _confidence_and_prediction(proba)
        y = np.asarray(y)
        bins = np.linspace(0.0, 1.0, max(int(n_bins), 2) + 1)
        gaps = []
        for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            selected = (confidence >= lower) & (
                confidence <= upper if index == len(bins) - 2 else confidence < upper
            )
            if np.any(selected):
                gaps.append(
                    abs(
                        np.mean(y[selected] == prediction[selected])
                        - np.mean(confidence[selected])
                    )
                )
        return float(max(gaps, default=0.0))

    @staticmethod
    def brier_score(proba: np.ndarray, y: np.ndarray) -> float:
        matrix = _probability_matrix(proba)
        y = np.asarray(y, dtype=int)
        one_hot = np.zeros_like(matrix)
        valid = (y >= 0) & (y < matrix.shape[1])
        one_hot[np.arange(len(y))[valid], y[valid]] = 1.0
        return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(proba, 1e-12, 1.0))

    def negative_log_likelihood(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        scaled = logits / temperature
        scaled -= np.max(scaled, axis=1, keepdims=True)
        probabilities = _renormalize(np.exp(scaled))
        return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0))))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = float(np.exp(result.x)) if result.success else 1.0
    return float(np.clip(temperature, 0.05, 20.0))


def _probability_matrix(proba: np.ndarray) -> np.ndarray:
    values = np.asarray(proba, dtype=float)
    if values.ndim == 1:
        values = np.column_stack((1.0 - values, values))
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Probabilities must be a binary vector or a 2-D class matrix")
    if np.any(~np.isfinite(values)):
        raise ValueError("Probabilities contain NaN or infinity")
    return _renormalize(np.clip(values, 0.0, 1.0))


def _renormalize(values: np.ndarray) -> np.ndarray:
    row_sum = np.sum(values, axis=1, keepdims=True)
    invalid = row_sum[:, 0] <= 0
    if np.any(invalid):
        values = values.copy()
        values[invalid] = 1.0 / values.shape[1]
        row_sum = np.sum(values, axis=1, keepdims=True)
    return values / row_sum


def _confidence_and_prediction(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _probability_matrix(proba)
    return np.max(matrix, axis=1), np.argmax(matrix, axis=1)
