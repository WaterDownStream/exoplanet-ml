"""Mono-transit feature extraction with fixed dimensionality."""
from __future__ import annotations

import numpy as np
from .mono_search import MonoSearchResult
from .utils import LightCurve

MONO_FEATURE_NAMES = [
    "mono_depth",
    "mono_duration",
    "mono_snr",
    "mono_morphology",
    "mono_ingress_time",
    "mono_egress_time",
    "mono_symmetry",
    "mono_curvature",
    "mono_slope",
    "mono_local_noise",
    "mono_baseline_stability",
    "mono_signal_isolation",
    "mono_template_match",
    "mono_period_prior",
    "mono_radius_range_low",
    "mono_radius_range_high",
]


class MonoFeatureExtractor:
    """Extract mono-transit features; periodic slots filled with NaN + masks."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}

    def extract(
        self, lc: LightCurve, mono: MonoSearchResult
    ) -> dict[str, float]:
        """Return named mono-transit features."""
        flux = lc.flux
        if not mono.detected or mono.best is None:
            return {k: float("nan") for k in MONO_FEATURE_NAMES}

        b = mono.best
        window = flux[b.window_start : b.window_end]
        ingress, egress = _ingress_egress(window, lc.time[b.window_start : b.window_end])
        radius_low, radius_high = _radius_ratio_interval(b.depth, b.depth_err)
        local_noise = b.window_noise if np.isfinite(b.window_noise) else _robust_sigma(flux)

        feats = {
            "mono_depth": b.depth,
            "mono_duration": b.duration,
            "mono_snr": b.snr,
            "mono_morphology": b.morphology_score,
            "mono_ingress_time": ingress,
            "mono_egress_time": egress,
            "mono_symmetry": _symmetry(window, b.depth),
            "mono_curvature": _curvature(window, b.depth),
            "mono_slope": _slope(window, b.depth),
            "mono_local_noise": local_noise,
            "mono_baseline_stability": 1.0 / (1.0 + _robust_sigma(flux) / max(b.depth, 1e-12)),
            "mono_signal_isolation": _isolation(flux, b.window_start, b.window_end),
            "mono_template_match": b.morphology_score,
            "mono_period_prior": float("nan"),
            # Legacy names: these are radius-ratio bounds, not absolute radii.
            "mono_radius_range_low": radius_low,
            "mono_radius_range_high": radius_high,
        }
        return feats

    def vectorize(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([feats[k] for k in MONO_FEATURE_NAMES], dtype=float)

    def masks(self, feats: dict[str, float]) -> np.ndarray:
        """1 = available, 0 = missing."""
        return np.array(
            [0.0 if np.isnan(feats[k]) else 1.0 for k in MONO_FEATURE_NAMES],
            dtype=float,
        )


def _ingress_egress(window: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    if len(window) < 3:
        return float("nan"), float("nan")
    bottom = np.argmin(window)
    half = (window.max() + window.min()) / 2
    left = np.where(window[:bottom] > half)[0]
    right = np.where(window[bottom:] > half)[0]
    ingress = float(time[bottom] - time[left[-1]]) if len(left) else float("nan")
    egress = float(time[bottom + right[0]] - time[bottom]) if len(right) else float("nan")
    return ingress, egress


def _symmetry(window: np.ndarray, depth: float) -> float:
    if len(window) < 4 or depth <= 0:
        return float("nan")
    mid = len(window) // 2
    left = window[:mid]
    right = window[mid:][::-1]
    n = min(len(left), len(right))
    if n == 0:
        return float("nan")
    mismatch = np.median(np.abs(left[:n] - right[:n])) / depth
    return float(np.clip(1.0 - mismatch, 0.0, 1.0))


def _curvature(window: np.ndarray, depth: float) -> float:
    if len(window) < 5 or depth <= 0:
        return float("nan")
    d2 = np.diff(window, n=2)
    return float(np.median(np.abs(d2)) / depth)


def _slope(window: np.ndarray, depth: float) -> float:
    if len(window) < 2 or depth <= 0:
        return float("nan")
    coordinate = np.linspace(-1.0, 1.0, len(window))
    return float(np.polyfit(coordinate, window / depth, 1)[0])


def _isolation(flux: np.ndarray, start: int, end: int) -> float:
    oot = np.concatenate([flux[:start], flux[end:]])
    if len(oot) < 2:
        return 0.0
    in_transit = flux[start:end]
    return float(
        (np.median(oot) - np.min(in_transit))
        / max(_robust_sigma(oot), 1e-12)
    )


def _radius_ratio_interval(depth: float, depth_err: float, sigma: float = 1.96) -> tuple[float, float]:
    """Approximate dark-companion radius-ratio interval from depth uncertainty."""
    if depth <= 0 or not np.isfinite(depth_err) or depth_err < 0:
        return float("nan"), float("nan")
    low = np.sqrt(max(depth - sigma * depth_err, 0.0))
    high = np.sqrt(max(depth + sigma * depth_err, 0.0))
    return float(low), float(high)


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    return float(sigma if sigma > 0 else np.std(values))
