"""Scientific regression tests for search, detrending, and geometry."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.detrending import Detrender
from src.feature_extraction import FeatureExtractor
from src.mono_search import MonoTransitSearch
from src.mono_features import _radius_ratio_interval
from src.parameter_estimation import ParameterEstimator
from src.transit_search import TransitSearch
from src.utils import LightCurve, load_config


def _box_lightcurve(
    period: float = 3.0,
    depth: float = 0.01,
    duration: float = 0.12,
    seed: int = 3,
) -> LightCurve:
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 27.0, 0.01)
    trend = 1.0 + 0.004 * np.sin(2.0 * np.pi * time / 6.5)
    phase_time = ((time - 1.2 + 0.5 * period) % period) - 0.5 * period
    flux = trend.copy()
    flux[np.abs(phase_time) <= 0.5 * duration] -= depth
    flux += rng.normal(0.0, 6e-4, len(time))
    return LightCurve(time, flux, np.full_like(time, 6e-4), {})


def test_gap_aware_detrending_preserves_transit_depth():
    lc = _box_lightcurve()
    detrended = Detrender(
        {
            "method": "savgol",
            "break_tolerance": 5.0,
            "transit_mask_sigma": 3.0,
            "niters": 3,
            "savgol": {"window_length_days": 1.5, "polyorder": 2},
        }
    ).run(lc)
    phase_time = ((detrended.time - 1.2 + 1.5) % 3.0) - 1.5
    inside = np.abs(phase_time) <= 0.06
    outside = np.abs(phase_time) >= 0.2
    recovered = np.median(detrended.flux[outside]) - np.median(detrended.flux[inside])
    assert np.isclose(recovered, 0.01, rtol=0.20)


def test_bls_reports_snr_sde_and_sampled_events():
    lc = _box_lightcurve()
    result = TransitSearch(
        {
            "period_min": 1.0,
            "period_max": 6.0,
            "n_periods": 3000,
            "duration_min": 0.06,
            "duration_max": 0.18,
            "n_durations": 5,
            "min_snr": 6.0,
            "min_sde": 5.0,
            "minimum_transits": 2,
        }
    ).run(lc)
    assert np.isclose(result.best_period, 3.0, rtol=0.01)
    assert result.depth_snr > 6.0
    assert result.sde > 5.0
    assert result.n_transits >= 8
    assert result.detected


def test_mono_search_rejects_one_bad_cadence():
    time = np.arange(0.0, 10.0, 0.01)
    flux = np.ones_like(time)
    flux[len(flux) // 2] = 0.90
    lc = LightCurve(time, flux, np.full_like(time, 1e-3), {})
    result = MonoTransitSearch(
        {
            "min_duration": 0.05,
            "max_duration": 0.3,
            "n_durations": 12,
            "snr_threshold": 6.0,
            "morphology_threshold": 0.25,
        }
    ).run(lc)
    assert not result.detected


def test_mono_search_detects_coherent_box_event():
    rng = np.random.default_rng(9)
    time = np.arange(0.0, 10.0, 0.01)
    flux = 1.0 + rng.normal(0.0, 5e-4, len(time))
    flux[np.abs(time - 5.0) <= 0.08] -= 0.01
    lc = LightCurve(time, flux, np.full_like(time, 5e-4), {})
    result = MonoTransitSearch(
        {
            "min_duration": 0.05,
            "max_duration": 0.3,
            "n_durations": 16,
            "snr_threshold": 6.0,
            "morphology_threshold": 0.2,
        }
    ).run(lc)
    assert result.detected
    assert abs(result.best.t0 - 5.0) < 0.08
    assert result.best.depth > 0.007


def test_exact_circular_impact_geometry_round_trip():
    period = 10.0
    radius_ratio = 0.1
    impact = 0.4
    semi_major_axis, _ = ParameterEstimator._semi_major_axis(
        period, None, 1.0, None
    )
    scaled_a = semi_major_axis / 0.00465047
    argument = np.sqrt(((1.0 + radius_ratio) ** 2 - impact**2) / (scaled_a**2 - impact**2))
    duration = period / np.pi * np.arcsin(argument)
    recovered = ParameterEstimator._impact_from_geometry(
        period, duration, radius_ratio, 1.0, semi_major_axis
    )
    assert np.isclose(recovered, impact, atol=1e-8)


def test_mono_period_prior_is_reproducible_and_qualified():
    config = load_config("configs/config.yaml")
    estimator = ParameterEstimator(
        {**config["parameter_estimation"], "seed": 123, "planet_threshold": 0.5}
    )
    candidate = SimpleNamespace(
        depth=0.01,
        depth_err=0.001,
        duration=0.2,
        t0=5.0,
    )
    mono = SimpleNamespace(detected=True, best=candidate)
    first = estimator.estimate_mono(
        mono,
        stellar={"radius": 1.0, "radius_err": 0.03, "mass": 1.0, "mass_err": 0.03},
        planet_prob=0.9,
    )
    second = estimator.estimate_mono(
        mono,
        stellar={"radius": 1.0, "radius_err": 0.03, "mass": 1.0, "mass_err": 0.03},
        planet_prob=0.9,
    )
    assert first.period.available
    assert first.period.value == second.period.value
    assert first.credible_intervals["period_95"] == second.credible_intervals["period_95"]
    assert any("prior-dependent" in warning for warning in first.warnings)


def test_unmeasured_false_positive_features_remain_missing():
    config = load_config("configs/config.yaml")
    lc = _box_lightcurve()
    bls = TransitSearch(
        {
            "period_min": 1.0,
            "period_max": 6.0,
            "n_periods": 1000,
            "duration_min": 0.06,
            "duration_max": 0.18,
            "n_durations": 4,
        }
    ).run(lc)
    features = FeatureExtractor(
        {**config["features"], "phase_folding": config["phase_folding"]}
    ).extract(lc, bls)
    assert np.isnan(features["blend_score"])
    assert np.isnan(features["bootstrap_fap"])


def test_mono_radius_ratio_bounds_come_from_depth_uncertainty():
    low, high = _radius_ratio_interval(0.01, 0.001)
    assert low == np.sqrt(0.01 - 1.96e-3)
    assert high == np.sqrt(0.01 + 1.96e-3)
    assert np.isnan(_radius_ratio_interval(0.01, np.nan)[0])
