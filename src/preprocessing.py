"""Cleaning: NaN removal, sorting, robust (transit-safe) clipping, normalization.

Physics background
------------------
A transit is a brief, one-sided *dip* in flux. Symmetric outlier rejection
(``|F - median| > k*sigma``) treats a deep transit as an outlier and removes
exactly the points we want to detect. Cleaning therefore has to be asymmetric:
aggressively reject positive spikes (cosmic rays, flares, momentum dumps) while
protecting negative excursions (real transits). Robust scale is estimated with
the median absolute deviation (MAD), which -- unlike the standard deviation --
is not inflated by the very outliers we are trying to reject.
"""
from __future__ import annotations

import numpy as np

from .utils import LightCurve, get_logger

logger = get_logger()

_MAD_TO_SIGMA = 1.4826  # scale factor making MAD a consistent estimator of sigma
_TESS_QUALITY_BITMASKS = {
    "none": 0,
    "default": 17087,
    "hard": 24319,
    "hardest": 65535,
}


class Preprocessor:
    """Apply standard, transit-safe light-curve cleaning steps.

    Mathematical formula
    --------------------
    Normalization: ``F_norm = F / median(F)``.
    Robust scale: ``sigma_robust = 1.4826 * median(|F - median(F)|)``.
    Asymmetric clip keeps points with
    ``-sigma_lower * sigma_robust <= F - median <= sigma_upper * sigma_robust``.
    """

    def __init__(self, cfg: dict) -> None:
        base_sigma = float(cfg.get("sigma", 5.0))
        # Upper (positive spikes) clipped tightly; lower (transits) protected.
        self.sigma_upper = float(cfg.get("sigma_upper", base_sigma))
        lower = cfg.get("sigma_lower")
        self.sigma_lower = float(lower) if lower is not None else float("inf")
        self.iters = int(cfg.get("sigma_iters", 5))
        self.remove_nans = bool(cfg.get("remove_nans", True))
        self.normalize = bool(cfg.get("normalize", True))
        self.robust_scale = bool(cfg.get("robust_scale", True))
        self.quality_bitmask = cfg.get("quality_bitmask", "default")

    def run(self, lc: LightCurve) -> LightCurve:
        """Execute the full preprocessing chain.

        Parameters
        ----------
        lc : LightCurve

        Returns
        -------
        LightCurve
            Cleaned copy.
        """
        lc = lc.copy()
        lc.meta["raw_n_points"] = int(len(lc.time))
        lc.meta["raw_nonfinite_fraction"] = float(
            np.mean(~(np.isfinite(lc.time) & np.isfinite(lc.flux)))
        ) if len(lc.time) else 1.0
        lc = self._apply_quality_mask(lc)
        if self.remove_nans:
            lc = self._drop_nonfinite(lc)
        lc = self._deduplicate_times(lc)
        lc = self._sort_by_time(lc)
        if self.normalize:
            lc = self._normalize(lc)
        lc = self._sigma_clip(lc)
        logger.info("Preprocessing complete: %d points remain", len(lc.time))
        return lc

    def _apply_quality_mask(self, lc: LightCurve) -> LightCurve:
        quality = lc.meta.get("quality_flags")
        if quality is None or self.quality_bitmask in (None, False, "none"):
            return lc
        quality = np.asarray(quality)
        if len(quality) != len(lc.time):
            logger.warning("Ignoring quality flags with mismatched length")
            return lc
        try:
            bitmask = (
                _TESS_QUALITY_BITMASKS[self.quality_bitmask.lower()]
                if isinstance(self.quality_bitmask, str)
                else int(self.quality_bitmask)
            )
            if bitmask < 0:
                raise ValueError("bitmask cannot be negative")
            finite = np.isfinite(quality) & (quality >= 0)
            integer_quality = np.zeros(len(quality), dtype=np.uint64)
            integer_quality[finite] = quality[finite].astype(np.uint64)
            mask = finite & ((integer_quality & np.uint64(bitmask)) == 0)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            logger.warning(
                "Could not apply TESS quality bitmask %r (%s); requiring QUALITY=0",
                self.quality_bitmask,
                exc,
            )
            mask = quality == 0
        removed = int(np.count_nonzero(~mask))
        lc.meta["quality_removed_fraction"] = removed / max(len(mask), 1)
        if removed:
            logger.info("Removed %d cadences using TESS quality flags", removed)
        return self._apply_mask(lc, np.asarray(mask, dtype=bool))

    def _drop_nonfinite(self, lc: LightCurve) -> LightCurve:
        # Reject non-finite time or flux (NaN, +/-inf). Errors may be NaN and
        # are tolerated -- downstream code falls back to photometric scatter.
        mask = np.isfinite(lc.time) & np.isfinite(lc.flux)
        return self._apply_mask(lc, mask)

    @staticmethod
    def _deduplicate_times(lc: LightCurve) -> LightCurve:
        """Collapse exact duplicate timestamps (seen when stitching sectors)."""
        if len(lc.time) < 2:
            return lc
        order = np.argsort(lc.time, kind="mergesort")
        t_sorted = lc.time[order]
        keep_sorted = np.ones(len(t_sorted), dtype=bool)
        keep_sorted[1:] = np.diff(t_sorted) > 0
        keep = np.zeros(len(lc.time), dtype=bool)
        keep[order[keep_sorted]] = True
        if keep.all():
            return lc
        logger.info("Removed %d duplicate timestamps", int((~keep).sum()))
        return Preprocessor._apply_mask(lc, keep)

    @staticmethod
    def _sort_by_time(lc: LightCurve) -> LightCurve:
        order = np.argsort(lc.time, kind="mergesort")
        lc.time = lc.time[order]
        lc.flux = lc.flux[order]
        lc.flux_error = lc.flux_error[order]
        return lc

    @staticmethod
    def _normalize(lc: LightCurve) -> LightCurve:
        med = np.nanmedian(lc.flux)
        if med == 0 or not np.isfinite(med):
            raise ValueError("Median flux is zero/invalid; cannot normalize")
        lc.flux = lc.flux / med
        lc.flux_error = lc.flux_error / abs(med)
        return lc

    def _sigma_clip(self, lc: LightCurve) -> LightCurve:
        """Iterative, transit-safe outlier rejection.

        Uses MAD-based robust scale and asymmetric thresholds so that real
        transits (negative excursions) survive while positive spikes are
        removed.
        """
        flux = lc.flux
        mask = np.isfinite(flux)
        if not mask.any():
            return self._apply_mask(lc, mask)
        for _ in range(max(1, self.iters)):
            sample = flux[mask]
            med = float(np.median(sample))
            if self.robust_scale:
                scale = _MAD_TO_SIGMA * float(np.median(np.abs(sample - med)))
                if scale <= 0:  # degenerate (constant) segment -> fall back
                    scale = float(np.std(sample))
            else:
                scale = float(np.std(sample))
            if scale <= 0:
                break
            resid = flux - med
            new_mask = (
                (resid <= self.sigma_upper * scale)
                & (resid >= -self.sigma_lower * scale)
                & np.isfinite(flux)
            )
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask
        return self._apply_mask(lc, mask)

    @staticmethod
    def _apply_mask(lc: LightCurve, mask: np.ndarray) -> LightCurve:
        original_length = len(lc.time)
        lc.time = lc.time[mask]
        lc.flux = lc.flux[mask]
        lc.flux_error = lc.flux_error[mask]
        quality = lc.meta.get("quality_flags")
        if quality is not None and len(np.asarray(quality)) == original_length:
            lc.meta["quality_flags"] = np.asarray(quality)[mask]
        return lc
