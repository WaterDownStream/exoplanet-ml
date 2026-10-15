"""Periodic transit searches with physically constrained BLS/TLS backends."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .utils import LightCurve, get_logger

logger = get_logger()

_MAD_TO_SIGMA = 1.4826


@dataclass
class BLSResult:
    """Best-fit periodic candidate and diagnostics.

    ``best_power`` is the BLS depth signal-to-noise ratio when the Astropy BLS
    backend is used.  It is not a false-alarm probability.
    """

    best_period: float
    best_duration: float
    best_depth: float
    best_t0: float
    best_power: float
    periods: np.ndarray
    power: np.ndarray
    depth_err: float = float("nan")
    depth_snr: float = float("nan")
    sde: float = float("nan")
    n_transits: int = 0
    detected: bool = False
    secondary_peak_power: float = float("nan")
    period_uncertainty: float = float("nan")


class TransitSearch:
    """Interchangeable, conservative periodic transit-search wrapper.

    The BLS grid is uniform in frequency because the independent frequency
    spacing is set by the observing baseline.  Each trial duration is evaluated
    only at periods for which its fractional duration is physically plausible.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.algorithm = str(cfg.get("algorithm", "bls")).lower()

    def run(self, lc: LightCurve) -> BLSResult:
        if self.algorithm == "bls":
            return self._run_bls(lc)
        if self.algorithm == "tls":
            return self._run_tls(lc)
        raise ValueError(f"Unknown transit-search algorithm: {self.algorithm}")

    def _run_bls(self, lc: LightCurve) -> BLSResult:
        """Run Astropy BLS using depth S/N as the search objective."""
        from astropy.timeseries import BoxLeastSquares

        time = np.asarray(lc.time, dtype=float)
        flux = np.asarray(lc.flux, dtype=float)
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        if len(time) < 20:
            raise ValueError("BLS requires at least 20 finite samples")
        if np.any(np.diff(time) <= 0):
            order = np.argsort(time, kind="mergesort")
            time, flux = time[order], flux[order]

        span = float(time[-1] - time[0])
        if not np.isfinite(span) or span <= 0:
            raise ValueError("BLS requires a positive observing baseline")

        c = self.cfg
        period_min = max(float(c.get("period_min", 0.5)), 2.0 * _median_cadence(time))
        period_max = min(float(c.get("period_max", 20.0)), span)
        if period_max <= period_min:
            raise ValueError(
                f"No valid BLS period range for baseline {span:.4g} d: "
                f"[{period_min:.4g}, {period_max:.4g}]"
            )

        n_periods = max(int(c.get("n_periods", 10_000)), 128)
        frequencies = np.linspace(1.0 / period_max, 1.0 / period_min, n_periods)
        periods = np.sort(1.0 / frequencies)

        duration_min = max(float(c.get("duration_min", 0.02)), _median_cadence(time))
        duration_max = min(float(c.get("duration_max", 0.3)), 0.5 * period_max)
        n_durations = max(int(c.get("n_durations", 10)), 1)
        durations = (
            np.geomspace(duration_min, duration_max, n_durations)
            if duration_max > duration_min
            else np.array([duration_min])
        )
        max_fraction = float(np.clip(c.get("max_duration_fraction", 0.2), 0.01, 0.45))
        oversample = max(int(c.get("oversample", 10)), 1)

        errors = _safe_err_values(lc.flux_error[finite], flux)
        bls = BoxLeastSquares(time, flux, dy=errors)
        shape = periods.shape
        best_power_by_period = np.full(shape, -np.inf)
        best_depth_by_period = np.full(shape, np.nan)
        best_depth_err_by_period = np.full(shape, np.nan)
        best_duration_by_period = np.full(shape, np.nan)
        best_t0_by_period = np.full(shape, np.nan)

        period_bin_count = max(int(c.get("period_bins", 8)), 1)
        period_edges = np.geomspace(period_min, period_max, period_bin_count + 1)
        for bin_index, (lower, upper) in enumerate(
            zip(period_edges[:-1], period_edges[1:])
        ):
            valid_period = (periods >= lower) & (
                periods <= upper if bin_index == period_bin_count - 1 else periods < upper
            )
            allowed_durations = durations[durations <= max_fraction * lower]
            if not np.any(valid_period) or len(allowed_durations) == 0:
                continue
            trial_periods = periods[valid_period]
            try:
                result = bls.power(
                    trial_periods,
                    allowed_durations,
                    objective="snr",
                    method="fast",
                    oversample=oversample,
                )
            except (TypeError, ValueError):
                result = bls.power(
                    trial_periods,
                    allowed_durations,
                    objective="snr",
                    method="slow",
                    oversample=oversample,
                )

            trial_power = np.asarray(result.power, dtype=float)
            target_idx = np.flatnonzero(valid_period)
            improve = np.isfinite(trial_power)
            if not np.any(improve):
                continue
            update_idx = target_idx[improve]
            source_idx = np.flatnonzero(improve)
            best_power_by_period[update_idx] = trial_power[source_idx]
            best_depth_by_period[update_idx] = np.asarray(result.depth)[source_idx]
            best_depth_err_by_period[update_idx] = np.asarray(result.depth_err)[source_idx]
            best_duration_by_period[update_idx] = np.asarray(result.duration)[source_idx]
            best_t0_by_period[update_idx] = np.asarray(result.transit_time)[source_idx]

        finite_power = np.isfinite(best_power_by_period)
        if not np.any(finite_power):
            raise RuntimeError("BLS produced no finite periodogram values")
        power = np.where(finite_power, best_power_by_period, np.nan)
        best_index = int(np.nanargmax(power))
        best_period = float(periods[best_index])
        best_duration = float(best_duration_by_period[best_index])
        best_depth = max(float(best_depth_by_period[best_index]), 0.0)
        best_t0 = float(best_t0_by_period[best_index])
        depth_err = float(best_depth_err_by_period[best_index])
        depth_snr = float(power[best_index])
        sde = _robust_sde(power, best_index)
        n_transits = _count_sampled_transits(
            time,
            best_period,
            best_duration,
            best_t0,
            min_points=max(int(c.get("min_points_per_transit", 2)), 1),
        )
        secondary_power = _secondary_peak_power(periods, power, best_index)
        period_uncertainty = _local_peak_uncertainty(periods, power, best_index)

        min_transits = max(int(c.get("minimum_transits", 2)), 2)
        min_snr = float(c.get("min_snr", 6.0))
        min_sde = float(c.get("min_sde", 5.0))
        detected = bool(
            best_depth > 0
            and np.isfinite(depth_snr)
            and depth_snr >= min_snr
            and np.isfinite(sde)
            and sde >= min_sde
            and n_transits >= min_transits
        )

        output = BLSResult(
            best_period=best_period,
            best_duration=best_duration,
            best_depth=best_depth,
            best_t0=best_t0,
            best_power=depth_snr,
            periods=periods,
            power=power,
            depth_err=depth_err,
            depth_snr=depth_snr,
            sde=sde,
            n_transits=n_transits,
            detected=detected,
            secondary_peak_power=secondary_power,
            period_uncertainty=period_uncertainty,
        )
        logger.info(
            "BLS best: P=%.5f d, dur=%.4f d, depth=%.6f, SNR=%.2f, "
            "SDE=%.2f, transits=%d, detected=%s",
            best_period,
            best_duration,
            best_depth,
            depth_snr,
            sde,
            n_transits,
            detected,
        )
        return output

    def _run_tls(self, lc: LightCurve) -> BLSResult:
        """Optional Transit Least Squares backend with BLS-compatible output."""
        if not self.cfg.get("tls_enabled", False):
            raise RuntimeError("TLS backend requested but tls_enabled is false")
        try:
            from transitleastsquares import transitleastsquares
        except ImportError as exc:
            raise RuntimeError("TLS backend requires transitleastsquares") from exc

        model = transitleastsquares(lc.time, lc.flux)
        result = model.power(
            period_min=self.cfg["period_min"],
            period_max=self.cfg["period_max"],
        )
        periods = np.asarray(result.periods, dtype=float)
        power = np.asarray(result.power, dtype=float)
        idx = int(np.nanargmax(power))
        period = float(result.period)
        duration = float(getattr(result, "duration", np.nan))
        t0 = float(getattr(result, "T0", np.nan))
        n_transits = _count_sampled_transits(lc.time, period, duration, t0, 2)
        snr = float(getattr(result, "snr", np.nan))
        sde = float(getattr(result, "SDE", np.nan))
        detected = bool(
            np.isfinite(snr)
            and snr >= float(self.cfg.get("min_snr", 6.0))
            and np.isfinite(sde)
            and sde >= float(self.cfg.get("min_sde", 5.0))
            and n_transits >= max(int(self.cfg.get("minimum_transits", 2)), 2)
        )
        return BLSResult(
            best_period=period,
            best_duration=duration,
            best_depth=float(getattr(result, "depth", np.nan)),
            best_t0=t0,
            best_power=snr,
            periods=periods,
            power=power,
            depth_snr=snr,
            sde=sde,
            n_transits=n_transits,
            detected=detected,
            secondary_peak_power=_secondary_peak_power(periods, power, idx),
            period_uncertainty=_local_peak_uncertainty(periods, power, idx),
        )


def _safe_err_values(errors: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Return finite positive per-cadence errors without discarding good values."""
    errors = np.asarray(errors, dtype=float).copy()
    valid = np.isfinite(errors) & (errors > 0)
    if np.any(valid):
        fill = float(np.median(errors[valid]))
        errors[~valid] = fill
        return np.maximum(errors, np.finfo(float).tiny)
    diff = np.diff(np.asarray(flux, dtype=float))
    sigma = _MAD_TO_SIGMA * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(flux))
    return np.full_like(flux, sigma if np.isfinite(sigma) and sigma > 0 else 1e-6)


def _safe_err(lc: LightCurve) -> np.ndarray:
    """Backward-compatible helper used by older callers/tests."""
    return _safe_err_values(lc.flux_error, lc.flux)


def _median_cadence(time: np.ndarray) -> float:
    dt = np.diff(np.asarray(time, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 1e-6


def _robust_sde(power: np.ndarray, best_index: int) -> float:
    finite = np.asarray(power, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 5:
        return float("nan")
    median = float(np.median(finite))
    scale = _MAD_TO_SIGMA * float(np.median(np.abs(finite - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float((power[best_index] - median) / scale)


def _count_sampled_transits(
    time: np.ndarray,
    period: float,
    duration: float,
    t0: float,
    min_points: int,
) -> int:
    if period <= 0 or duration <= 0 or not np.isfinite(t0):
        return 0
    time = np.asarray(time, dtype=float)
    cycle = np.rint((time - t0) / period).astype(np.int64)
    center = t0 + cycle * period
    in_transit = np.abs(time - center) <= 0.5 * duration
    if not np.any(in_transit):
        return 0
    _, counts = np.unique(cycle[in_transit], return_counts=True)
    return int(np.count_nonzero(counts >= min_points))


def _secondary_peak_power(periods: np.ndarray, power: np.ndarray, best_index: int) -> float:
    finite_power = np.nan_to_num(np.asarray(power, dtype=float), nan=-np.inf)
    peaks, _ = find_peaks(finite_power)
    if len(peaks) == 0:
        return float("nan")
    best_period = float(periods[best_index])
    grid_width = abs(float(periods[min(best_index + 1, len(periods) - 1)]) - best_period)
    exclusion = max(3.0 * grid_width, 0.005 * best_period)
    peaks = peaks[np.abs(np.asarray(periods)[peaks] - best_period) > exclusion]
    if len(peaks) == 0:
        return float("nan")
    return float(np.max(finite_power[peaks]))


def _local_peak_uncertainty(periods: np.ndarray, power: np.ndarray, best_index: int) -> float:
    """Approximate local peak width; this is not a posterior uncertainty."""
    values = np.asarray(power, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 5:
        return float("nan")
    baseline = float(np.median(finite))
    threshold = baseline + 0.5 * (float(values[best_index]) - baseline)
    left = best_index
    while left > 0 and np.isfinite(values[left - 1]) and values[left - 1] >= threshold:
        left -= 1
    right = best_index
    while right < len(values) - 1 and np.isfinite(values[right + 1]) and values[right + 1] >= threshold:
        right += 1
    if right == left:
        if len(periods) < 2:
            return float("nan")
        return float(np.median(np.diff(periods)))
    return float(0.5 * (periods[right] - periods[left]))
