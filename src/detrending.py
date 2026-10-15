"""Gap-aware removal of stellar variability and instrumental trends."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .utils import LightCurve, get_logger

logger = get_logger()

_MAD_TO_SIGMA = 1.4826


class Detrender:
    """Flatten a light curve while masking likely negative-going transits.

    Trends are estimated independently inside contiguous observing segments so
    filters never interpolate across TESS downlink gaps.  The default window is
    specified in days; this gives consistent behavior across 20 s, 2 min, 10 min,
    and 30 min cadence products.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.method = str(cfg.get("method", "savgol")).lower()
        self.break_tolerance = float(cfg.get("break_tolerance", 5.0))
        self.transit_mask_sigma = float(cfg.get("transit_mask_sigma", 3.0))
        self.niters = max(int(cfg.get("niters", 3)), 1)

    def run(self, lc: LightCurve) -> LightCurve:
        lc = lc.copy()
        if len(lc.flux) < 3:
            return lc
        if np.any(np.diff(lc.time) <= 0):
            raise ValueError("Detrending requires strictly increasing timestamps")

        trend = np.full_like(lc.flux, np.nan, dtype=float)
        for start, stop in _contiguous_segments(lc.time, self.break_tolerance):
            segment_time = lc.time[start:stop]
            segment_flux = lc.flux[start:stop]
            if self.method == "moving_median":
                segment_trend = self._moving_median(segment_time, segment_flux)
            elif self.method == "savgol":
                segment_trend = self._savgol(segment_time, segment_flux)
            elif self.method == "gp":
                segment_trend = self._gp(segment_time, segment_flux)
            elif self.method == "pca":
                logger.warning(
                    "Single-light-curve 'pca' mode is an SSA compatibility mode; "
                    "mission cotrending basis vectors are preferred"
                )
                segment_trend = self._ssa_trend(segment_time, segment_flux)
            else:
                raise ValueError(f"Unknown detrending method: {self.method}")
            trend[start:stop] = segment_trend

        scale = float(np.nanmedian(np.abs(lc.flux)))
        floor = max(scale * 1e-8, np.finfo(float).tiny)
        valid_trend = np.isfinite(trend) & (np.abs(trend) > floor)
        if not np.any(valid_trend):
            raise ValueError("Detrending produced no finite non-zero trend")
        if not np.all(valid_trend):
            good = np.flatnonzero(valid_trend)
            trend[~valid_trend] = np.interp(
                np.flatnonzero(~valid_trend), good, trend[good]
            )

        lc.flux = lc.flux / trend
        lc.flux_error = lc.flux_error / np.abs(trend)
        finite = np.isfinite(lc.time) & np.isfinite(lc.flux)
        lc.time = lc.time[finite]
        lc.flux = lc.flux[finite]
        lc.flux_error = lc.flux_error[finite]
        logger.info("Detrended using gap-aware %s (%d segments)", self.method, len(_contiguous_segments(lc.time, self.break_tolerance)))
        return lc

    def _moving_median(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        cfg = self.cfg.get("moving_median", {})
        window = _window_samples(time, cfg, default_samples=101)
        if window >= len(flux):
            return np.full_like(flux, np.median(flux))
        filled = _mask_and_interpolate_transits(
            flux,
            window,
            self.transit_mask_sigma,
            self.niters,
        )
        return median_filter(filled, size=window, mode="nearest")

    def _savgol(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        cfg = self.cfg.get("savgol", {})
        polyorder = max(int(cfg.get("polyorder", 2)), 0)
        if len(flux) <= polyorder + 2:
            return np.full_like(flux, np.median(flux))
        window = _window_samples(time, cfg, default_samples=101)
        window = min(window, len(flux) if len(flux) % 2 else len(flux) - 1)
        minimum = polyorder + 2
        if minimum % 2 == 0:
            minimum += 1
        window = max(window, minimum)
        if window > len(flux):
            return np.full_like(flux, np.median(flux))

        filled = _mask_and_interpolate_transits(
            flux,
            window,
            self.transit_mask_sigma,
            self.niters,
        )
        return savgol_filter(filled, window, polyorder, mode="interp")

    def _gp(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Compatibility GP backend, capped to avoid cubic scaling."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

        cfg = self.cfg.get("gp", {})
        length_scale = float(cfg.get("length_scale", 1.0))
        kernels = {
            "rbf": RBF(length_scale=length_scale),
            "matern": Matern(length_scale=length_scale, nu=1.5),
            "rationalquadratic": RationalQuadratic(length_scale=length_scale),
        }
        kernel_name = str(cfg.get("kernel", "matern")).lower()
        if kernel_name not in kernels:
            raise ValueError(f"Unknown GP kernel: {kernel_name}")

        max_points = max(int(cfg.get("max_fit_points", 750)), 100)
        idx = np.unique(np.linspace(0, len(time) - 1, min(len(time), max_points)).astype(int))
        fit_flux = _mask_and_interpolate_transits(
            flux,
            _window_samples(time, cfg, default_samples=101),
            self.transit_mask_sigma,
            self.niters,
        )
        gp = GaussianProcessRegressor(
            kernel=kernels[kernel_name],
            alpha=max(float(cfg.get("alpha", 1e-3)), np.finfo(float).eps),
            normalize_y=True,
            optimizer=None,
        )
        gp.fit(time[idx, None], fit_flux[idx])
        return np.asarray(gp.predict(time[:, None]), dtype=float)

    def _ssa_trend(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Low-rank singular-spectrum trend retained for ``pca`` compatibility."""
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.decomposition import PCA

        cfg = self.cfg.get("pca", {})
        n_components = max(int(cfg.get("n_components", 5)), 1)
        window = _window_samples(time, cfg, default_samples=max(2 * n_components + 1, 51))
        window = min(window, max(len(flux) // 4, 3))
        window = window if window % 2 else window - 1
        if window < 3 or len(flux) < window + 2:
            return np.full_like(flux, np.median(flux))
        filled = _mask_and_interpolate_transits(
            flux, window, self.transit_mask_sigma, self.niters
        )
        trajectory = sliding_window_view(filled, window)
        components = min(n_components, trajectory.shape[0], trajectory.shape[1])
        pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
        reconstruction = pca.inverse_transform(pca.fit_transform(trajectory))
        trend = np.zeros_like(filled)
        counts = np.zeros_like(filled)
        for offset in range(window):
            values = reconstruction[:, offset]
            trend[offset : offset + len(values)] += values
            counts[offset : offset + len(values)] += 1.0
        return trend / np.maximum(counts, 1.0)


def _window_samples(time: np.ndarray, cfg: dict, default_samples: int) -> int:
    cadence = _median_cadence(time)
    window_days = cfg.get("window_length_days", cfg.get("window_days"))
    if window_days is not None and cadence > 0:
        window = int(np.ceil(float(window_days) / cadence))
    else:
        window = int(cfg.get("window_length", cfg.get("window", default_samples)))
    window = max(window, 3)
    return window if window % 2 else window + 1


def _mask_and_interpolate_transits(
    flux: np.ndarray,
    window: int,
    sigma_threshold: float,
    niters: int,
) -> np.ndarray:
    """Prevent negative outliers from pulling the trend through transits."""
    filled = np.asarray(flux, dtype=float).copy()
    x = np.arange(len(filled))
    for _ in range(max(niters, 1)):
        size = min(window, len(filled) if len(filled) % 2 else len(filled) - 1)
        if size < 3:
            break
        baseline = median_filter(filled, size=size, mode="nearest")
        residual = flux - baseline
        sigma = _robust_sigma(residual)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        protected = residual < -abs(sigma_threshold) * sigma
        if not np.any(protected):
            break
        good = np.flatnonzero(~protected & np.isfinite(flux))
        if len(good) < 2:
            break
        updated = np.asarray(flux, dtype=float).copy()
        updated[protected] = np.interp(x[protected], x[good], flux[good])
        if np.allclose(updated, filled, equal_nan=True):
            break
        filled = updated
    return filled


def _contiguous_segments(time: np.ndarray, break_tolerance: float) -> list[tuple[int, int]]:
    if len(time) == 0:
        return []
    cadence = _median_cadence(time)
    if cadence <= 0:
        return [(0, len(time))]
    breaks = np.flatnonzero(np.diff(time) > max(break_tolerance, 1.0) * cadence) + 1
    edges = np.concatenate(([0], breaks, [len(time)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _median_cadence(time: np.ndarray) -> float:
    dt = np.diff(np.asarray(time, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 0.0


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    center = float(np.median(values))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(values - center)))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(values))
    return sigma
