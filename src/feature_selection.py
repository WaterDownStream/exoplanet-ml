"""Deterministic train-only feature filtering."""
from __future__ import annotations

import numpy as np


class StableFeatureSelector:
    """Remove constant and nearly duplicate columns in deterministic order.

    Earlier columns win correlation ties, which intentionally prioritizes the
    core scientific features before extended diagnostics and availability masks.
    """

    def __init__(
        self,
        enabled: bool = True,
        variance_threshold: float = 1e-12,
        correlation_threshold: float = 0.995,
        minimum_features: int = 10,
    ) -> None:
        self.enabled = bool(enabled)
        self.variance_threshold = max(float(variance_threshold), 0.0)
        self.correlation_threshold = float(correlation_threshold)
        self.minimum_features = max(int(minimum_features), 1)
        self.selected_indices_: np.ndarray | None = None
        self.input_names_: list[str] = []
        self.selected_names_: list[str] = []
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("Feature selection expects a 2-D matrix")
        self.n_features_in_ = int(values.shape[1])
        self.input_names_ = list(feature_names or [f"feature_{i}" for i in range(values.shape[1])])
        if len(self.input_names_) != values.shape[1]:
            raise ValueError("feature_names length does not match X")
        if not self.enabled:
            selected = np.arange(values.shape[1], dtype=int)
        else:
            variance = np.var(values, axis=0)
            candidates = np.flatnonzero(
                np.isfinite(variance) & (variance > self.variance_threshold)
            )
            if len(candidates) < self.minimum_features:
                candidates = np.argsort(np.nan_to_num(variance, nan=-np.inf))[::-1][
                    : min(self.minimum_features, values.shape[1])
                ]
                candidates = np.sort(candidates)
            selected = self._correlation_filter(values, candidates)
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.selected_names_ = [self.input_names_[index] for index in self.selected_indices_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None or self.n_features_in_ is None:
            raise RuntimeError("Feature selector has not been fitted")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature selector expects {self.n_features_in_} columns, received {values.shape}"
            )
        return values[:, self.selected_indices_]

    def fit_transform(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def _correlation_filter(
        self, values: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        threshold = self.correlation_threshold
        if not 0.0 < threshold < 1.0 or len(candidates) < 2:
            return candidates
        subset = values[:, candidates]
        centered = subset - np.mean(subset, axis=0)
        scale = np.sqrt(np.sum(centered**2, axis=0))
        normalized = centered / np.maximum(scale, np.finfo(float).tiny)
        correlation = np.abs(normalized.T @ normalized)
        keep_positions: list[int] = []
        for position in range(len(candidates)):
            if not keep_positions or np.all(
                correlation[position, keep_positions] < threshold
            ):
                keep_positions.append(position)
        if len(keep_positions) < self.minimum_features:
            keep_positions = list(range(min(self.minimum_features, len(candidates))))
        return candidates[np.asarray(keep_positions, dtype=int)]
