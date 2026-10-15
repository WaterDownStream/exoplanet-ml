"""Astrophysically qualified transit-parameter estimates and uncertainties."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transit_search import BLSResult
from .utils import LightCurve, get_logger

logger = get_logger()

_G = 6.67430e-11
_M_SUN_KG = 1.98847e30
_R_SUN_M = 6.957e8
_R_SUN_AU = 0.00465047
_R_SUN_TO_EARTH = 109.076
_DAY_S = 86400.0
_T_SUN = 5772.0
_MAD_TO_SIGMA = 1.4826


@dataclass
class ParameterEstimate:
    """One estimate with provenance and assumptions."""

    value: float | None
    uncertainty: float | None
    unit: str
    available: bool = True
    method: str = ""
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ParameterReport:
    period: ParameterEstimate
    duration: ParameterEstimate
    depth: ParameterEstimate
    epoch: ParameterEstimate
    radius_ratio: ParameterEstimate
    planet_radius: ParameterEstimate
    impact_parameter: ParameterEstimate
    semi_major_axis: ParameterEstimate
    equilibrium_temperature: ParameterEstimate
    transit_probability: ParameterEstimate
    planet_insolation: ParameterEstimate
    credible_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ParameterEstimator:
    """Estimate parameters only for sufficiently planet-like candidates.

    Periodic estimates assume a circular orbit for geometry.  Mono-transit
    intervals use explicit impact-parameter and eccentricity priors and are
    therefore prior-dependent rather than direct measurements.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.n_boot = max(int(cfg.get("bootstrap_samples", 200)), 32)
        self.seed = int(cfg.get("seed", 42))

    def estimate(
        self,
        lc: LightCurve,
        bls: BLSResult,
        stellar: dict | None = None,
        planet_prob: float = 0.0,
    ) -> ParameterReport:
        if planet_prob < float(self.cfg.get("planet_threshold", 0.5)):
            return self.unavailable("planet probability below threshold")
        if not bls.detected:
            return self.unavailable("periodic candidate does not meet detection thresholds")

        stellar = stellar or lc.meta.get("stellar", {})
        r_star, r_star_err = _stellar_value(stellar, "radius")
        m_star, m_star_err = _stellar_value(stellar, "mass")
        t_eff, t_eff_err = _stellar_value(stellar, "teff")

        period = float(bls.best_period)
        duration = float(bls.best_duration)
        depth = max(float(bls.best_depth), 0.0)
        epoch = float(bls.best_t0)
        period_err = self._period_uncertainty(lc, bls)
        duration_err = self._duration_uncertainty(lc, bls)
        depth_err = self._depth_uncertainty(lc, bls)
        epoch_err = self._epoch_uncertainty(lc, bls)

        period_samples = self._normal_samples(period, period_err, self.seed + 1, lower=0.0)
        depth_samples = self._normal_samples(depth, depth_err, self.seed + 2, lower=0.0)
        radius_ratio = np.sqrt(depth) if depth > 0 else None
        radius_ratio_err = (
            0.5 * depth_err / radius_ratio
            if radius_ratio is not None and depth_err is not None
            else None
        )

        semi_major_axis, semi_major_axis_err = self._semi_major_axis(
            period,
            period_err,
            m_star,
            m_star_err,
        )
        planet_radius, planet_radius_err = self._planet_radius(
            radius_ratio, radius_ratio_err, r_star, r_star_err
        )
        impact, impact_err = self._impact_parameter(
            period,
            period_err,
            duration,
            duration_err,
            radius_ratio,
            radius_ratio_err,
            r_star,
            r_star_err,
            semi_major_axis,
            semi_major_axis_err,
        )
        equilibrium_temperature, equilibrium_temperature_err = self._equilibrium_temperature(
            t_eff, t_eff_err, r_star, r_star_err, semi_major_axis, semi_major_axis_err
        )
        transit_probability, transit_probability_err = self._transit_probability(
            r_star,
            r_star_err,
            semi_major_axis,
            semi_major_axis_err,
            radius_ratio,
            radius_ratio_err,
        )
        insolation, insolation_err = self._insolation(
            t_eff, t_eff_err, r_star, r_star_err, semi_major_axis, semi_major_axis_err
        )

        warnings = self._stellar_warnings(
            r_star, r_star_err, m_star, m_star_err, t_eff, t_eff_err
        )
        warnings.extend(
            [
                "period uncertainty is a local BLS/timing approximation, not a posterior",
                "radius ratio assumes an undiluted small-planet transit",
                "impact parameter assumes a circular orbit",
                "equilibrium temperature assumes full heat redistribution",
            ]
        )
        credible = self._credible_intervals(
            {"period": period_samples, "depth": depth_samples}
        )

        return ParameterReport(
            period=ParameterEstimate(
                period, period_err, "days", method="BLS local peak and timing precision"
            ),
            duration=ParameterEstimate(
                duration, duration_err, "days", method="BLS box duration"
            ),
            depth=ParameterEstimate(
                depth, depth_err, "fraction", method="BLS weighted box depth"
            ),
            epoch=ParameterEstimate(
                epoch, epoch_err, "BTJD or input time unit", method="BLS transit epoch"
            ),
            radius_ratio=ParameterEstimate(
                radius_ratio,
                radius_ratio_err,
                "Rp/Rstar",
                available=radius_ratio is not None,
                method="sqrt(depth)",
                assumptions=["small planet", "no dilution", "weak limb-darkening bias"],
            ),
            planet_radius=ParameterEstimate(
                planet_radius,
                planet_radius_err,
                "R_earth",
                available=planet_radius is not None,
                method="radius ratio times stellar radius",
            ),
            impact_parameter=ParameterEstimate(
                impact,
                impact_err,
                "",
                available=impact is not None,
                method="exact circular T14 geometry",
                assumptions=["circular orbit", "fixed box duration approximates T14"],
            ),
            semi_major_axis=ParameterEstimate(
                semi_major_axis,
                semi_major_axis_err,
                "AU",
                available=semi_major_axis is not None,
                method="Kepler's third law",
                assumptions=["planet mass negligible relative to star"],
            ),
            equilibrium_temperature=ParameterEstimate(
                equilibrium_temperature,
                equilibrium_temperature_err,
                "K",
                available=equilibrium_temperature is not None,
                method="radiative equilibrium",
                assumptions=[
                    f"Bond albedo={float(self.cfg.get('bond_albedo', 0.0)):.3g}",
                    "full heat redistribution",
                ],
            ),
            transit_probability=ParameterEstimate(
                transit_probability,
                transit_probability_err,
                "fraction",
                available=transit_probability is not None,
                method="circular geometric probability",
            ),
            planet_insolation=ParameterEstimate(
                insolation,
                insolation_err,
                "S_earth",
                available=insolation is not None,
                method="Stefan-Boltzmann luminosity scaling",
            ),
            credible_intervals=credible,
            warnings=warnings,
        )

    def estimate_mono(
        self,
        mono,
        stellar: dict | None = None,
        planet_prob: float = 0.0,
    ) -> ParameterReport:
        if planet_prob < float(self.cfg.get("planet_threshold", 0.5)):
            return self.unavailable("planet probability below threshold")
        if mono is None or not getattr(mono, "detected", False) or mono.best is None:
            return self.unavailable("no mono-transit candidate")

        stellar = stellar or {}
        r_star, r_star_err = _stellar_value(stellar, "radius")
        m_star, m_star_err = _stellar_value(stellar, "mass")
        t_eff, t_eff_err = _stellar_value(stellar, "teff")
        best = mono.best
        depth = max(float(best.depth), 0.0)
        depth_err = _positive_or_none(getattr(best, "depth_err", None))
        duration = float(best.duration)
        epoch = float(best.t0)
        radius_ratio = np.sqrt(depth) if depth > 0 else None
        radius_ratio_err = (
            0.5 * depth_err / radius_ratio
            if radius_ratio is not None and depth_err is not None
            else None
        )

        period_samples = self._mono_period_samples(
            duration,
            depth,
            r_star,
            r_star_err,
            m_star,
            m_star_err,
        )
        period = float(np.median(period_samples)) if len(period_samples) else None
        period_err = (
            float(0.5 * (np.quantile(period_samples, 0.84) - np.quantile(period_samples, 0.16)))
            if len(period_samples)
            else None
        )
        semi_major_axis, semi_major_axis_err = self._semi_major_axis(
            period, period_err, m_star, m_star_err
        ) if period is not None else (None, None)
        planet_radius, planet_radius_err = self._planet_radius(
            radius_ratio, radius_ratio_err, r_star, r_star_err
        )
        equilibrium_temperature, equilibrium_temperature_err = self._equilibrium_temperature(
            t_eff, t_eff_err, r_star, r_star_err, semi_major_axis, semi_major_axis_err
        )
        transit_probability, transit_probability_err = self._transit_probability(
            r_star,
            r_star_err,
            semi_major_axis,
            semi_major_axis_err,
            radius_ratio,
            radius_ratio_err,
        )
        insolation, insolation_err = self._insolation(
            t_eff, t_eff_err, r_star, r_star_err, semi_major_axis, semi_major_axis_err
        )

        warnings = [
            "mono-transit period is prior-dependent, not directly measured",
            "mono period uses a transiting-impact prior and configured eccentricity prior",
            "mono period range is truncated at configured physical bounds",
            "radius ratio assumes an undiluted small-planet transit",
        ]
        warnings.extend(
            self._stellar_warnings(
                r_star, r_star_err, m_star, m_star_err, t_eff, t_eff_err
            )
        )
        credible = self._credible_intervals({"period": period_samples})
        observable_method = "red-noise-aware mono matched filter"
        return ParameterReport(
            period=ParameterEstimate(
                period,
                period_err,
                "days",
                available=period is not None,
                method="Monte Carlo duration prior",
                assumptions=["impact parameter uniform conditional on transit", "eccentricity prior"],
            ),
            duration=ParameterEstimate(duration, None, "days", method=observable_method),
            depth=ParameterEstimate(depth, depth_err, "fraction", method=observable_method),
            epoch=ParameterEstimate(epoch, None, "BTJD or input time unit", method=observable_method),
            radius_ratio=ParameterEstimate(
                radius_ratio,
                radius_ratio_err,
                "Rp/Rstar",
                available=radius_ratio is not None,
                method="sqrt(depth)",
            ),
            planet_radius=ParameterEstimate(
                planet_radius,
                planet_radius_err,
                "R_earth",
                available=planet_radius is not None,
                method="radius ratio times stellar radius",
            ),
            impact_parameter=ParameterEstimate(None, None, "", available=False),
            semi_major_axis=ParameterEstimate(
                semi_major_axis,
                semi_major_axis_err,
                "AU",
                available=semi_major_axis is not None,
                method="Kepler's third law applied to period prior",
            ),
            equilibrium_temperature=ParameterEstimate(
                equilibrium_temperature,
                equilibrium_temperature_err,
                "K",
                available=equilibrium_temperature is not None,
                method="radiative equilibrium from period prior",
            ),
            transit_probability=ParameterEstimate(
                transit_probability,
                transit_probability_err,
                "fraction",
                available=transit_probability is not None,
                method="geometric probability from period prior",
            ),
            planet_insolation=ParameterEstimate(
                insolation,
                insolation_err,
                "S_earth",
                available=insolation is not None,
                method="stellar luminosity and period prior",
            ),
            credible_intervals=credible,
            warnings=warnings,
        )

    def _period_uncertainty(self, lc: LightCurve, bls: BLSResult) -> float | None:
        cadence = self._median_cadence(lc)
        grid = np.diff(np.asarray(bls.periods, dtype=float))
        grid = grid[np.isfinite(grid) & (grid > 0)]
        grid_sigma = float(np.median(grid) / np.sqrt(12.0)) if len(grid) else 0.0
        hwhm_sigma = (
            float(bls.period_uncertainty / np.sqrt(2.0 * np.log(2.0)))
            if np.isfinite(bls.period_uncertainty) and bls.period_uncertainty > 0
            else 0.0
        )
        n_transits = max(int(bls.n_transits), 1)
        snr_total = max(float(bls.depth_snr), 1.0) if np.isfinite(bls.depth_snr) else 1.0
        per_event_snr = snr_total / np.sqrt(n_transits)
        timing_sigma = max(
            0.5 * cadence if cadence is not None else 0.0,
            bls.best_duration / max(per_event_snr, 1.0),
        )
        regression_sigma = (
            timing_sigma * np.sqrt(12.0 / (n_transits * (n_transits**2 - 1.0)))
            if n_transits >= 2
            else 0.0
        )
        value = max(grid_sigma, hwhm_sigma, regression_sigma)
        return float(value) if np.isfinite(value) and value > 0 else None

    def _depth_uncertainty(self, lc: LightCurve, bls: BLSResult) -> float | None:
        finite = np.isfinite(lc.time) & np.isfinite(lc.flux)
        in_transit = self._transit_mask(lc.time, bls) & finite
        out_transit = (~in_transit) & finite
        candidates = []
        if np.isfinite(bls.depth_err) and bls.depth_err > 0:
            candidates.append(float(bls.depth_err))
        if np.count_nonzero(in_transit) >= 2 and np.count_nonzero(out_transit) >= 2:
            sigma = _robust_sigma(lc.flux[out_transit])
            white = sigma * np.sqrt(
                1.0 / np.count_nonzero(in_transit) + 1.0 / np.count_nonzero(out_transit)
            )
            if np.isfinite(white) and white > 0:
                candidates.append(float(white))
            event_depths = self._event_depths(lc, bls)
            if len(event_depths) > 1:
                red = float(np.std(event_depths, ddof=1) / np.sqrt(len(event_depths)))
                if np.isfinite(red) and red > 0:
                    candidates.append(red)
        return max(candidates) if candidates else None

    def _duration_uncertainty(self, lc: LightCurve, bls: BLSResult) -> float | None:
        if bls.best_duration <= 0:
            return None
        cadence = self._median_cadence(lc) or 0.0
        snr = max(float(bls.depth_snr), 1.0) if np.isfinite(bls.depth_snr) else 1.0
        return float(max(cadence, bls.best_duration / snr))

    def _epoch_uncertainty(self, lc: LightCurve, bls: BLSResult) -> float | None:
        cadence = self._median_cadence(lc) or 0.0
        snr = max(float(bls.depth_snr), 1.0) if np.isfinite(bls.depth_snr) else 1.0
        value = max(0.5 * cadence, 0.5 * bls.best_duration / snr)
        return float(value) if value > 0 else None

    @staticmethod
    def _transit_mask(time: np.ndarray, bls: BLSResult) -> np.ndarray:
        if bls.best_period <= 0 or bls.best_duration <= 0:
            return np.zeros_like(time, dtype=bool)
        cycle = np.rint((np.asarray(time) - bls.best_t0) / bls.best_period)
        center = bls.best_t0 + cycle * bls.best_period
        return np.abs(np.asarray(time) - center) <= 0.5 * bls.best_duration

    def _event_depths(self, lc: LightCurve, bls: BLSResult) -> np.ndarray:
        mask = self._transit_mask(lc.time, bls)
        if not np.any(mask):
            return np.array([])
        cycles = np.rint((lc.time - bls.best_t0) / bls.best_period).astype(int)
        baseline = float(np.median(lc.flux[~mask])) if np.any(~mask) else float(np.median(lc.flux))
        values = []
        for cycle in np.unique(cycles[mask]):
            selected = mask & (cycles == cycle)
            if np.count_nonzero(selected) >= 2:
                values.append(baseline - float(np.median(lc.flux[selected])))
        return np.asarray(values, dtype=float)

    @staticmethod
    def _median_cadence(lc: LightCurve) -> float | None:
        time = np.sort(np.asarray(lc.time, dtype=float)[np.isfinite(lc.time)])
        if len(time) < 2:
            return None
        dt = np.diff(time)
        dt = dt[dt > 0]
        return float(np.median(dt)) if len(dt) else None

    @staticmethod
    def _semi_major_axis(
        period: float | None,
        period_err: float | None,
        m_star: float | None,
        m_star_err: float | None,
    ) -> tuple[float | None, float | None]:
        if period is None or m_star is None or period <= 0 or m_star <= 0:
            return None, None
        m_solar = m_star / _M_SUN_KG if m_star > 100 else m_star
        m_err_solar = (
            m_star_err / _M_SUN_KG if m_star_err is not None and m_star > 100 else m_star_err
        )
        value = ((period / 365.25) ** 2 * m_solar) ** (1.0 / 3.0)
        relative_terms = []
        if period_err is not None:
            relative_terms.append((2.0 * period_err / (3.0 * period)) ** 2)
        if m_err_solar is not None:
            relative_terms.append((m_err_solar / (3.0 * m_solar)) ** 2)
        uncertainty = value * np.sqrt(sum(relative_terms)) if relative_terms else None
        return float(value), float(uncertainty) if uncertainty is not None else None

    @staticmethod
    def _planet_radius(
        radius_ratio: float | None,
        radius_ratio_err: float | None,
        r_star: float | None,
        r_star_err: float | None,
    ) -> tuple[float | None, float | None]:
        if radius_ratio is None or r_star is None or radius_ratio <= 0 or r_star <= 0:
            return None, None
        value = radius_ratio * r_star * _R_SUN_TO_EARTH
        relative_terms = []
        if radius_ratio_err is not None:
            relative_terms.append((radius_ratio_err / radius_ratio) ** 2)
        if r_star_err is not None:
            relative_terms.append((r_star_err / r_star) ** 2)
        uncertainty = value * np.sqrt(sum(relative_terms)) if relative_terms else None
        return float(value), float(uncertainty) if uncertainty is not None else None

    def _impact_parameter(
        self,
        period: float,
        period_err: float | None,
        duration: float,
        duration_err: float | None,
        radius_ratio: float | None,
        radius_ratio_err: float | None,
        r_star: float | None,
        r_star_err: float | None,
        a_au: float | None,
        a_err: float | None,
    ) -> tuple[float | None, float | None]:
        value = self._impact_from_geometry(period, duration, radius_ratio, r_star, a_au)
        if value is None:
            return None, None
        uncertainties = (period_err, duration_err, radius_ratio_err, r_star_err, a_err)
        if not any(error is not None and error > 0 for error in uncertainties):
            return value, None
        rng = np.random.default_rng(self.seed + 3)
        samples = []
        for _ in range(min(self.n_boot, 512)):
            sample = self._impact_from_geometry(
                _draw_positive(rng, period, period_err),
                _draw_positive(rng, duration, duration_err),
                _draw_positive(rng, radius_ratio, radius_ratio_err),
                _draw_positive(rng, r_star, r_star_err),
                _draw_positive(rng, a_au, a_err),
            )
            if sample is not None:
                samples.append(sample)
        return value, float(np.std(samples, ddof=1)) if len(samples) > 1 else None

    @staticmethod
    def _impact_from_geometry(
        period: float | None,
        duration: float | None,
        radius_ratio: float | None,
        r_star: float | None,
        a_au: float | None,
    ) -> float | None:
        if any(value is None for value in (period, duration, radius_ratio, r_star, a_au)):
            return None
        if period <= 0 or duration <= 0 or radius_ratio < 0 or r_star <= 0 or a_au <= 0:
            return None
        scaled_a = a_au / (r_star * _R_SUN_AU)
        sine = np.sin(np.pi * duration / period)
        denominator = 1.0 - sine**2
        numerator = (1.0 + radius_ratio) ** 2 - (sine * scaled_a) ** 2
        if denominator <= 0 or numerator < 0:
            return None
        impact = np.sqrt(numerator / denominator)
        if not np.isfinite(impact) or impact > 1.0 + radius_ratio:
            return None
        return float(impact)

    def _equilibrium_temperature(
        self,
        t_eff: float | None,
        t_eff_err: float | None,
        r_star: float | None,
        r_star_err: float | None,
        a_au: float | None,
        a_err: float | None,
    ) -> tuple[float | None, float | None]:
        if t_eff is None or r_star is None or a_au is None or min(t_eff, r_star, a_au) <= 0:
            return None, None
        albedo = float(np.clip(self.cfg.get("bond_albedo", 0.0), 0.0, 0.99))
        value = t_eff * np.sqrt(r_star * _R_SUN_AU / (2.0 * a_au)) * (1.0 - albedo) ** 0.25
        relative = _relative_uncertainty(
            (t_eff, t_eff_err, 1.0),
            (r_star, r_star_err, 0.5),
            (a_au, a_err, -0.5),
        )
        return float(value), float(value * relative) if relative is not None else None

    @staticmethod
    def _transit_probability(
        r_star: float | None,
        r_star_err: float | None,
        a_au: float | None,
        a_err: float | None,
        radius_ratio: float | None,
        radius_ratio_err: float | None,
    ) -> tuple[float | None, float | None]:
        if r_star is None or a_au is None or r_star <= 0 or a_au <= 0:
            return None, None
        ratio = max(radius_ratio or 0.0, 0.0)
        value = float(np.clip(r_star * _R_SUN_AU * (1.0 + ratio) / a_au, 0.0, 1.0))
        relative = _relative_uncertainty(
            (r_star, r_star_err, 1.0),
            (a_au, a_err, -1.0),
            (1.0 + ratio, radius_ratio_err, 1.0),
        )
        uncertainty = min(value * relative, 1.0) if relative is not None else None
        return value, float(uncertainty) if uncertainty is not None else None

    @staticmethod
    def _insolation(
        t_eff: float | None,
        t_eff_err: float | None,
        r_star: float | None,
        r_star_err: float | None,
        a_au: float | None,
        a_err: float | None,
    ) -> tuple[float | None, float | None]:
        if t_eff is None or r_star is None or a_au is None or min(t_eff, r_star, a_au) <= 0:
            return None, None
        value = (r_star / a_au) ** 2 * (t_eff / _T_SUN) ** 4
        relative = _relative_uncertainty(
            (r_star, r_star_err, 2.0),
            (a_au, a_err, -2.0),
            (t_eff, t_eff_err, 4.0),
        )
        return float(value), float(value * relative) if relative is not None else None

    def _mono_period_samples(
        self,
        duration_days: float,
        depth: float,
        r_star: float | None,
        r_star_err: float | None,
        m_star: float | None,
        m_star_err: float | None,
    ) -> np.ndarray:
        if duration_days <= 0 or r_star is None or m_star is None or r_star <= 0 or m_star <= 0:
            return np.array([])
        count = max(int(self.cfg.get("mono_prior_samples", 5000)), 1000)
        rng = np.random.default_rng(self.seed + 11)
        r_samples = _draw_positive_array(rng, r_star, r_star_err, count) * _R_SUN_M
        m_solar = m_star / _M_SUN_KG if m_star > 100 else m_star
        m_err_solar = m_star_err / _M_SUN_KG if m_star_err is not None and m_star > 100 else m_star_err
        m_samples = _draw_positive_array(rng, m_solar, m_err_solar, count) * _M_SUN_KG
        radius_ratio = np.sqrt(max(depth, 0.0))
        impact = rng.uniform(0.0, max(1.0 + radius_ratio - 1e-6, 1e-6), count)
        chord = np.sqrt(np.maximum((1.0 + radius_ratio) ** 2 - impact**2, 1e-12))

        if bool(self.cfg.get("mono_eccentricity_prior", True)):
            alpha = float(self.cfg.get("eccentricity_beta_alpha", 0.867))
            beta = float(self.cfg.get("eccentricity_beta_beta", 3.03))
            eccentricity = np.clip(rng.beta(alpha, beta, count), 0.0, 0.95)
            omega = rng.uniform(0.0, 2.0 * np.pi, count)
            velocity_factor = (1.0 + eccentricity * np.sin(omega)) / np.sqrt(1.0 - eccentricity**2)
        else:
            velocity_factor = np.ones(count)

        duration_s = duration_days * _DAY_S
        circular_period_s = (
            np.pi * _G * m_samples * duration_s**3 / (4.0 * r_samples**3 * chord**3)
        )
        period_days = circular_period_s * velocity_factor**3 / _DAY_S
        minimum = max(float(self.cfg.get("mono_period_min_days", 0.5)), 2.0 * duration_days)
        maximum = float(self.cfg.get("mono_period_max_days", 3650.0))
        valid = np.isfinite(period_days) & (period_days >= minimum) & (period_days <= maximum)
        return np.asarray(period_days[valid], dtype=float)

    def _normal_samples(
        self,
        value: float,
        uncertainty: float | None,
        seed: int,
        lower: float | None = None,
    ) -> np.ndarray:
        if uncertainty is None or uncertainty <= 0:
            return np.array([float(value)])
        rng = np.random.default_rng(seed)
        samples = rng.normal(value, uncertainty, self.n_boot)
        if lower is not None:
            samples = samples[samples > lower]
        return samples if len(samples) else np.array([float(value)])

    def _credible_intervals(
        self, samples_by_name: dict[str, np.ndarray]
    ) -> dict[str, tuple[float, float]]:
        output = {}
        for name, samples in samples_by_name.items():
            samples = np.asarray(samples, dtype=float)
            samples = samples[np.isfinite(samples)]
            if not len(samples):
                continue
            for level in self.cfg.get("credible_levels", [0.68, 0.95]):
                level = float(level)
                if not 0 < level < 1:
                    continue
                tail = 0.5 * (1.0 - level)
                output[f"{name}_{int(round(level * 100))}"] = (
                    float(np.quantile(samples, tail)),
                    float(np.quantile(samples, 1.0 - tail)),
                )
        return output

    @staticmethod
    def _stellar_warnings(
        r_star,
        r_star_err,
        m_star,
        m_star_err,
        t_eff,
        t_eff_err,
    ) -> list[str]:
        warnings = []
        for label, value in (
            ("stellar radius", r_star),
            ("stellar mass", m_star),
            ("stellar effective temperature", t_eff),
        ):
            if value is None:
                warnings.append(f"{label} unavailable")
        if any(
            value is not None and error is None
            for value, error in (
                (r_star, r_star_err),
                (m_star, m_star_err),
                (t_eff, t_eff_err),
            )
        ):
            warnings.append(
                "some stellar-input uncertainties are unavailable; derived errors are lower bounds"
            )
        return warnings

    @staticmethod
    def unavailable(reason: str) -> ParameterReport:
        def missing() -> ParameterEstimate:
            return ParameterEstimate(None, None, "", available=False)

        return ParameterReport(
            period=missing(),
            duration=missing(),
            depth=missing(),
            epoch=missing(),
            radius_ratio=missing(),
            planet_radius=missing(),
            impact_parameter=missing(),
            semi_major_axis=missing(),
            equilibrium_temperature=missing(),
            transit_probability=missing(),
            planet_insolation=missing(),
            warnings=[reason],
        )

    _unavailable = unavailable


def _stellar_value(stellar: dict, name: str) -> tuple[float | None, float | None]:
    value = _positive_or_none(stellar.get(name))
    error = _positive_or_none(
        stellar.get(f"{name}_err", stellar.get(f"{name}_uncertainty"))
    )
    return value, error


def _positive_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


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


def _relative_uncertainty(*terms: tuple[float | None, float | None, float]) -> float | None:
    contributions = []
    for value, uncertainty, exponent in terms:
        if value is not None and value > 0 and uncertainty is not None and uncertainty >= 0:
            contributions.append((exponent * uncertainty / value) ** 2)
    return float(np.sqrt(sum(contributions))) if contributions else None


def _draw_positive(
    rng: np.random.Generator,
    value: float | None,
    uncertainty: float | None,
) -> float | None:
    if value is None:
        return None
    if uncertainty is None or uncertainty <= 0:
        return float(value)
    for _ in range(20):
        sample = float(rng.normal(value, uncertainty))
        if sample > 0:
            return sample
    return float(value)


def _draw_positive_array(
    rng: np.random.Generator,
    value: float,
    uncertainty: float | None,
    count: int,
) -> np.ndarray:
    if uncertainty is None or uncertainty <= 0:
        return np.full(count, value, dtype=float)
    samples = rng.normal(value, uncertainty, count)
    invalid = samples <= 0
    for _ in range(10):
        if not np.any(invalid):
            break
        samples[invalid] = rng.normal(value, uncertainty, np.count_nonzero(invalid))
        invalid = samples <= 0
    samples[invalid] = value
    return samples
