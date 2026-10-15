"""Time-aware, multi-scale search for isolated transit-like events."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .utils import LightCurve, get_logger

logger = get_logger()

_MAD_TO_SIGMA = 1.4826


@dataclass
class MonoCandidate:
    """One isolated dip measured against a local out-of-event baseline."""

    depth: float
    duration: float
    t0: float
    snr: float
    morphology_score: float
    window_start: int
    window_end: int
    depth_err: float = float("nan")
    window_noise: float = float("nan")
    coverage: float = float("nan")
    n_points: int = 0


@dataclass
class MonoSearchResult:
    candidates: list[MonoCandidate]
    best: MonoCandidate | None
    detected: bool


class MonoTransitSearch:
    """Vectorized box matched filter for non-periodic transit candidates.

    Unlike a point-minimum detector, this search tests the mean flux over a
    range of physical durations.  Its noise scale is estimated from the full
    distribution of same-duration windows, so red noise and cadence-correlated
    variability are reflected in the reported S/N.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def run(self, lc: LightCurve) -> MonoSearchResult:
        time = np.asarray(lc.time, dtype=float)
        flux = np.asarray(lc.flux, dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        n = len(flux)
        if n < max(int(self.cfg.get("minimum_points", 50)), 20):
            return MonoSearchResult([], None, False)
        if np.any(np.diff(time) <= 0):
            order = np.argsort(time, kind="mergesort")
            time, flux = time[order], flux[order]

        cadence = _median_cadence(time)
        span = float(time[-1] - time[0])
        if cadence <= 0 or span <= 0:
            return MonoSearchResult([], None, False)

        c = self.cfg
        min_duration = max(float(c.get("min_duration", 0.02)), 2.0 * cadence)
        max_duration = min(float(c.get("max_duration", 0.5)), 0.25 * span)
        if max_duration < min_duration:
            return MonoSearchResult([], None, False)
        n_durations = max(int(c.get("n_durations", 18)), 1)
        durations = (
            np.geomspace(min_duration, max_duration, n_durations)
            if max_duration > min_duration
            else np.array([min_duration])
        )
        step_days = max(float(c.get("window_step", cadence)), cadence)
        stride = max(1, int(round(step_days / cadence)))
        starts = np.arange(0, n - 2, stride, dtype=int)
        prefix = np.concatenate(([0.0], np.cumsum(flux, dtype=np.float64)))

        raw_candidates: list[MonoCandidate] = []
        min_points = max(int(c.get("min_points_in_event", 3)), 3)
        min_coverage = float(np.clip(c.get("min_window_coverage", 0.6), 0.1, 1.0))
        threshold = float(c.get("snr_threshold", 6.0))

        for duration in durations:
            ends = np.searchsorted(time, time[starts] + duration, side="right")
            counts = ends - starts
            safe_end = np.clip(ends - 1, 0, n - 1)
            actual_span = time[safe_end] - time[starts] + cadence
            coverage = np.clip(counts * cadence / max(duration, cadence), 0.0, 1.0)
            valid = (
                (counts >= min_points)
                & (ends <= n)
                & (actual_span >= 0.5 * duration)
                & (coverage >= min_coverage)
            )
            if np.count_nonzero(valid) < 10:
                continue

            means = np.full(len(starts), np.nan, dtype=float)
            means[valid] = (prefix[ends[valid]] - prefix[starts[valid]]) / counts[valid]
            center = float(np.nanmedian(means[valid]))
            window_sigma = _robust_sigma(means[valid])
            if not np.isfinite(window_sigma) or window_sigma <= 0:
                continue
            snr_series = (center - means) / window_sigma
            min_distance = max(1, int(round(0.75 * duration / (stride * cadence))))
            peaks, properties = find_peaks(
                np.nan_to_num(snr_series, nan=-np.inf),
                height=max(0.6 * threshold, 2.5),
                distance=min_distance,
            )
            for peak, preliminary_snr in zip(peaks, properties.get("peak_heights", [])):
                if not valid[peak] or not np.isfinite(preliminary_snr):
                    continue
                candidate = _measure_candidate(
                    time,
                    flux,
                    int(starts[peak]),
                    int(ends[peak]),
                    float(duration),
                    cadence,
                    window_sigma,
                    float(coverage[peak]),
                )
                if candidate is None:
                    continue
                if not (float(c.get("min_depth", 5e-4)) <= candidate.depth <= float(c.get("max_depth", 0.15))):
                    continue
                if candidate.snr < threshold:
                    continue
                if candidate.morphology_score < float(c.get("morphology_threshold", 0.3)):
                    continue
                raw_candidates.append(candidate)

        candidates = _deduplicate_candidates(
            raw_candidates,
            max_candidates=max(int(c.get("max_candidates", 20)), 1),
        )
        if not candidates:
            logger.info("No mono-transit candidates found")
            return MonoSearchResult([], None, False)

        best = candidates[0]
        logger.info(
            "Mono candidate: depth=%.6f +/- %.6f, dur=%.4f d, SNR=%.2f, morphology=%.2f",
            best.depth,
            best.depth_err,
            best.duration,
            best.snr,
            best.morphology_score,
        )
        return MonoSearchResult(candidates, best, True)


def _measure_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    start: int,
    end: int,
    trial_duration: float,
    cadence: float,
    empirical_window_sigma: float,
    coverage: float,
) -> MonoCandidate | None:
    if end - start < 3:
        return None
    left = np.searchsorted(time, time[start] - 2.0 * trial_duration, side="left")
    right = np.searchsorted(time, time[end - 1] + 2.0 * trial_duration, side="right")
    oot = np.concatenate((flux[left:start], flux[end:right]))
    event_flux = flux[start:end]
    event_time = time[start:end]
    if len(oot) < max(6, len(event_flux) // 2):
        return None

    baseline = float(np.median(oot))
    event_level = float(np.median(event_flux))
    depth = baseline - event_level
    if not np.isfinite(depth) or depth <= 0:
        return None
    local_sigma = _robust_sigma(oot)
    if not np.isfinite(local_sigma) or local_sigma <= 0:
        return None

    median_factor = np.sqrt(np.pi / 2.0)
    white_depth_err = local_sigma * median_factor * np.sqrt(
        1.0 / len(event_flux) + 1.0 / len(oot)
    )
    depth_err = max(float(white_depth_err), float(empirical_window_sigma))
    snr = depth / depth_err if depth_err > 0 else 0.0

    deficit = np.clip(baseline - event_flux, 0.0, None)
    if np.sum(deficit) > 0:
        t0 = float(np.sum(event_time * deficit) / np.sum(deficit))
    else:
        t0 = float(0.5 * (event_time[0] + event_time[-1]))
    half_depth = baseline - 0.5 * depth
    below = np.flatnonzero(event_flux <= half_depth)
    duration = (
        float(event_time[below[-1]] - event_time[below[0]] + cadence)
        if len(below) >= 2
        else float(trial_duration)
    )
    duration = float(np.clip(duration, 2.0 * cadence, 1.5 * trial_duration))
    morphology = _morphology_score(event_time, event_flux, baseline, depth, t0)
    return MonoCandidate(
        depth=float(depth),
        duration=duration,
        t0=t0,
        snr=float(snr),
        morphology_score=morphology,
        window_start=start,
        window_end=end,
        depth_err=depth_err,
        window_noise=float(empirical_window_sigma),
        coverage=float(coverage),
        n_points=int(len(event_flux)),
    )


def _morphology_score(
    time: np.ndarray,
    flux: np.ndarray,
    baseline: float,
    depth: float,
    t0: float,
) -> float:
    """Measure coherent, symmetric, baseline-recovering dip morphology."""
    if len(flux) < 5 or depth <= 0 or time[-1] <= time[0]:
        return 0.0
    grid = np.linspace(time[0], time[-1], 31)
    profile = (baseline - np.interp(grid, time, flux)) / depth
    profile = np.clip(profile, -1.0, 2.0)
    symmetry = 1.0 - np.median(np.abs(profile - profile[::-1]))
    edge_recovery = 1.0 - 0.5 * (abs(profile[0]) + abs(profile[-1]))
    central = profile[len(profile) // 3 : 2 * len(profile) // 3]
    central_coherence = 1.0 - float(np.std(central))
    center_offset = abs(t0 - 0.5 * (time[0] + time[-1])) / (time[-1] - time[0])
    centering = 1.0 - 2.0 * center_offset
    return float(
        np.clip(
            0.35 * symmetry
            + 0.25 * edge_recovery
            + 0.25 * central_coherence
            + 0.15 * centering,
            0.0,
            1.0,
        )
    )


def _deduplicate_candidates(
    candidates: list[MonoCandidate],
    max_candidates: int,
) -> list[MonoCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: candidate.snr * (0.5 + 0.5 * candidate.morphology_score),
        reverse=True,
    )
    selected: list[MonoCandidate] = []
    for candidate in ranked:
        if any(
            abs(candidate.t0 - existing.t0)
            <= 0.5 * max(candidate.duration, existing.duration)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    median = float(np.median(values))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(values - median)))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(values))
    return sigma


def _median_cadence(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 0.0
