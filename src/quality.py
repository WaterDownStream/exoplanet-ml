"""Observation quality assessment with explicit, unitless diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import LightCurve, get_logger

logger = get_logger()

_MAD_TO_SIGMA = 1.4826


@dataclass
class QualityReport:
    score: float
    passed: bool
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    rejection_reason: str | None = None


class QualityAssessment:
    """Assess whether a cleaned light curve supports a meaningful search.

    The score is a search-reliability summary, not a probability that the data
    contain a planet.  Hard cadence, finite-data, and coverage constraints are
    evaluated separately so a weighted score cannot hide a fatal defect.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.weights = cfg.get("weights", {})

    def run(self, lc: LightCurve) -> QualityReport:
        time = np.asarray(lc.time, dtype=float)
        flux = np.asarray(lc.flux, dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        raw_n = max(int(lc.meta.get("raw_n_points", len(flux))), 1)
        raw_nonfinite = float(lc.meta.get("raw_nonfinite_fraction", np.mean(~finite)))
        n = int(np.count_nonzero(finite))
        minimum_points = int(self.cfg.get("minimum_points", 300))

        if n < 2:
            return QualityReport(
                score=0.0,
                passed=False,
                warnings=["fewer than two finite cadences"],
                metrics={"n_points": float(n), "raw_n_points": float(raw_n)},
                rejection_reason="insufficient finite cadence",
            )

        time_f, flux_f = time[finite], flux[finite]
        order = np.argsort(time_f, kind="mergesort")
        time_f, flux_f = time_f[order], flux_f[order]
        dt = np.diff(time_f)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        cadence = float(np.median(positive_dt)) if len(positive_dt) else float("nan")
        span = float(time_f[-1] - time_f[0])
        gap_threshold = float(self.cfg.get("gap_tolerance", 5.0)) * cadence
        gap_mask = positive_dt > gap_threshold if np.isfinite(cadence) else np.zeros_like(positive_dt, dtype=bool)
        gap_count = int(np.count_nonzero(gap_mask))
        gap_fraction = float(np.mean(gap_mask)) if len(gap_mask) else 0.0

        expected = span / cadence + 1.0 if cadence > 0 else float(n)
        duty_cycle = float(np.clip(n / max(expected, 1.0), 0.0, 1.0))
        retained_fraction = float(np.clip(n / raw_n, 0.0, 1.0))
        quality_removed = float(lc.meta.get("quality_removed_fraction", 0.0))

        median = float(np.median(flux_f))
        sigma = _robust_sigma(flux_f)
        relative_noise = sigma / max(abs(median), np.finfo(float).eps)
        outlier_fraction = float(
            np.mean(np.abs(flux_f - median) > 5.0 * sigma)
        ) if sigma > 0 else 0.0
        stationarity = _stationarity_score(flux_f)
        longest_segment = _longest_segment(time_f, gap_threshold)
        longest_fraction = longest_segment / span if span > 0 else 0.0
        cadence_regularity = _cadence_regularity(positive_dt, cadence)
        noise_score = float(1.0 / (1.0 + relative_noise / max(float(self.cfg.get("noise_scale", 0.01)), 1e-9)))

        metrics = {
            "n_points": float(n),
            "raw_n_points": float(raw_n),
            "observation_length": span,
            "cadence": cadence,
            "cadence_gaps": gap_fraction,
            "gap_count": float(gap_count),
            "missing_fraction": raw_nonfinite,
            "nan_fraction": raw_nonfinite,
            "retained_fraction": retained_fraction,
            "duty_cycle": duty_cycle,
            "quality_flag_fraction": quality_removed,
            "median_quality_flag": float("nan"),
            "outlier_fraction": outlier_fraction,
            "noise_estimate": relative_noise,
            "crowding_metric": _safe_meta_float(lc.meta, "crowding"),
            "estimated_contamination": _safe_meta_float(lc.meta, "contamination"),
            "flux_stability": noise_score,
            "signal_stationarity": stationarity,
            "longest_segment": longest_segment,
            "longest_segment_fraction": longest_fraction,
            "coverage_score": duty_cycle,
            "completeness": duty_cycle,
            "cadence_regularity": cadence_regularity,
        }

        components = {
            "cadence": cadence_regularity,
            "missing": 1.0 - raw_nonfinite,
            "noise": noise_score,
            "gaps": duty_cycle,
            "stability": stationarity,
        }
        weights = {
            "cadence": 0.15,
            "missing": 0.20,
            "noise": 0.25,
            "gaps": 0.15,
            "stability": 0.25,
            **self.weights,
        }
        denominator = sum(max(float(weights[name]), 0.0) for name in components) or 1.0
        score = sum(
            max(float(weights[name]), 0.0) * float(np.clip(value, 0.0, 1.0))
            for name, value in components.items()
        ) / denominator
        score = float(np.clip(score, 0.0, 1.0))

        warnings: list[str] = []
        hard_failures: list[str] = []
        if n < minimum_points:
            hard_failures.append("insufficient cadence")
        if raw_nonfinite > float(self.cfg.get("max_nan_fraction", 0.5)):
            hard_failures.append("high non-finite fraction")
        if duty_cycle < float(self.cfg.get("min_coverage", 0.5)):
            hard_failures.append("low temporal coverage")
        if outlier_fraction > float(self.cfg.get("max_outlier_fraction", 0.15)):
            warnings.append("high residual outlier fraction")
        if quality_removed > float(self.cfg.get("max_quality_flag_fraction", 0.5)):
            warnings.append("many cadences rejected by TESS quality flags")
        if stationarity < float(self.cfg.get("min_stationarity", 0.5)):
            warnings.append("non-stationary photometric noise")

        passed = not hard_failures and score >= float(self.cfg.get("reject_threshold", 0.3))
        rejection = "; ".join(hard_failures) if hard_failures else (
            None if passed else "quality below threshold"
        )
        warnings = hard_failures + warnings
        logger.info("Quality score=%.3f passed=%s", score, passed)
        return QualityReport(score, passed, warnings, metrics, rejection)


def _longest_segment(time: np.ndarray, gap_threshold: float) -> float:
    if len(time) < 2 or not np.isfinite(gap_threshold) or gap_threshold <= 0:
        return 0.0
    breaks = np.flatnonzero(np.diff(time) > gap_threshold) + 1
    edges = np.concatenate(([0], breaks, [len(time)]))
    lengths = [time[b - 1] - time[a] for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]
    return float(max(lengths, default=0.0))


def _stationarity_score(flux: np.ndarray) -> float:
    if len(flux) < 40:
        return 1.0
    chunks = np.array_split(flux, min(8, max(2, len(flux) // 100)))
    sigmas = np.array([_robust_sigma(chunk) for chunk in chunks], dtype=float)
    sigmas = sigmas[np.isfinite(sigmas) & (sigmas >= 0)]
    if len(sigmas) < 2:
        return 1.0
    coefficient = _robust_sigma(sigmas) / max(float(np.median(sigmas)), 1e-12)
    return float(1.0 / (1.0 + coefficient))


def _cadence_regularity(dt: np.ndarray, cadence: float) -> float:
    if len(dt) == 0 or not np.isfinite(cadence) or cadence <= 0:
        return 0.0
    relative = np.abs(dt - cadence) / cadence
    regular = relative[relative < 4.0]
    if len(regular) == 0:
        return 0.0
    return float(1.0 / (1.0 + np.median(regular)))


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    median = float(np.median(values))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(values))
    return sigma if np.isfinite(sigma) else 0.0


def _safe_meta_float(meta: dict, key: str) -> float:
    value = meta.get(key, np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")
