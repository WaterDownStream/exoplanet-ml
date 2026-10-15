"""Light-curve augmentation for training."""
from __future__ import annotations

import hashlib

import numpy as np

from ..utils import LightCurve, get_logger

logger = get_logger()


class LightCurveAugmenter:
    """Apply configurable augmentations without altering class labels."""

    def __init__(self, cfg: dict, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def maybe_augment(
        self,
        lc: LightCurve,
        probability: float = 0.2,
        key: str | None = None,
    ) -> LightCurve:
        """Return augmented copy with given probability."""
        rng = self._rng_for_key(key) if key is not None else self.rng
        if rng.random() > probability:
            return lc
        return self.augment(lc, rng=rng)

    def augment(
        self, lc: LightCurve, rng: np.random.Generator | None = None
    ) -> LightCurve:
        rng = rng or self.rng
        time = lc.time.copy()
        flux = lc.flux.copy()
        flux_error = lc.flux_error.copy()
        meta = dict(lc.meta)
        quality = meta.get("quality_flags")
        quality = (
            np.asarray(quality).copy()
            if quality is not None and len(np.asarray(quality)) == len(flux)
            else None
        )
        c = self.cfg

        if c.get("gaussian_noise_std", 0):
            relative_std = float(c["gaussian_noise_std"])
            flux_scale = max(abs(float(np.nanmedian(flux))), 1e-12)
            added_sigma = relative_std * flux_scale
            flux += rng.normal(0.0, added_sigma, size=flux.shape)
            finite_error = np.isfinite(flux_error) & (flux_error >= 0)
            flux_error[finite_error] = np.sqrt(
                flux_error[finite_error] ** 2 + added_sigma**2
            )

        if c.get("poisson", False):
            scale = max(abs(float(np.nanmedian(flux))), 1e-12)
            counts = max(float(c.get("poisson_reference_counts", 1e6)), 1.0)
            rate = np.clip(flux / scale * counts, 0.0, None)
            flux = rng.poisson(rate) / counts * scale

        mask_frac = c.get("cadence_mask_fraction", 0)
        if mask_frac > 0:
            n_mask = int(len(flux) * mask_frac)
            if n_mask > 0:
                idx = rng.choice(len(flux), n_mask, replace=False)
                flux[idx] = np.nan

        shift = c.get("baseline_shift", 0)
        if shift:
            relative = np.linspace(-0.5, 0.5, len(flux))
            flux += rng.uniform(-shift, shift) * np.nanmedian(flux) * relative

        time_shift = c.get("time_shift", 0)
        if time_shift:
            time += rng.uniform(-time_shift, time_shift)

        cadence_jitter = c.get("cadence_jitter_std", 0)
        if cadence_jitter:
            time += rng.normal(0, cadence_jitter, size=time.shape)
            order = np.argsort(time)
            time, flux, flux_error = time[order], flux[order], flux_error[order]
            if quality is not None:
                quality = quality[order]

        flux_scale_std = c.get("flux_scale_std", 0)
        if flux_scale_std:
            factor = rng.normal(1.0, flux_scale_std)
            flux *= factor
            flux_error *= abs(factor)

        crop_fraction = c.get("random_crop_fraction", 0)
        if 0 < crop_fraction < 1 and len(flux) > 10:
            keep = max(10, int(len(flux) * (1 - crop_fraction)))
            start = int(rng.integers(0, max(1, len(flux) - keep + 1)))
            stop = start + keep
            time = time[start:stop]
            flux = flux[start:stop]
            flux_error = flux_error[start:stop]
            if quality is not None:
                quality = quality[start:stop]

        valid = np.isfinite(time) & np.isfinite(flux)
        if quality is not None:
            meta["quality_flags"] = quality[valid]
        return LightCurve(time[valid], flux[valid], flux_error[valid], meta)

    def _rng_for_key(self, key: str) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
        derived_seed = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(derived_seed)
