"""Fold a light curve at a candidate period and optionally bin it."""
from __future__ import annotations

import numpy as np

from .utils import LightCurve, get_logger

logger = get_logger()


class PhaseFolder:
    """Phase-fold flux for transit visualization and feature extraction.

    Mathematical formula
    --------------------
    ``phi = ((t - t0) / P) mod 1``, shifted so transit sits near phi = 0.5.
    """

    def __init__(self, cfg: dict) -> None:
        self.n_bins = int(cfg.get("n_bins", 200))

    def fold(
        self, lc: LightCurve, period: float, t0: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted phase and flux arrays.

        Parameters
        ----------
        lc : LightCurve
        period : float
            Orbital period in days.
        t0 : float
            Reference transit epoch.

        Returns
        -------
        (phase, flux) : tuple of np.ndarray
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        phase = ((lc.time - t0) / period) % 1.0
        phase = (phase + 0.5) % 1.0  # center transit
        order = np.argsort(phase)
        return phase[order], lc.flux[order]

    def bin(
        self, phase: np.ndarray, flux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin folded flux into ``n_bins`` equal-width phase bins.

        Returns
        -------
        (bin_centers, binned_flux)
        """
        edges = np.linspace(0, 1, self.n_bins + 1)
        idx = np.clip(np.digitize(phase, edges) - 1, 0, self.n_bins - 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binned = np.full(self.n_bins, np.nan)
        for b in range(self.n_bins):
            sel = idx == b
            if np.any(sel):
                binned[b] = np.median(flux[sel])
        return centers, binned
