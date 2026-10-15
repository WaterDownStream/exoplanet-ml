"""Compute physically meaningful features from a searched light curve."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import entropy, kurtosis, skew

from .phase_folding import PhaseFolder
from .transit_search import BLSResult
from .utils import LightCurve, get_logger

logger = get_logger()


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy 1.x/2.x compatibility."""
    integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrate(y, x))

# Core features (backward compatible with v1 pipeline).
CORE_FEATURE_NAMES = [
    "transit_depth",
    "transit_duration",
    "orbital_period",
    "transit_snr",
    "transit_power",
    "flux_std",
    "flux_mad",
    "local_noise",
    "skewness",
    "kurtosis",
    "odd_even_depth_difference",
]

EXTENDED_FEATURE_NAMES = [
    "transit_epoch",
    "duty_cycle",
    "ingress_duration",
    "egress_duration",
    "estimated_impact_parameter",
    "depth_duration_ratio",
    "depth_symmetry",
    "transit_width",
    "transit_occupancy",
    "normalized_transit_width",
    "fractional_transit_duration",
    "integrated_bls_power",
    "secondary_peak_power",
    "primary_secondary_ratio",
    "detection_significance",
    "bootstrap_fap",
    "local_signal_contrast",
    "global_signal_contrast",
    "median_transit_depth",
    "depth_variance",
    "depth_consistency",
    "signal_persistence",
    "v_shape_score",
    "u_shape_score",
    "flat_bottom_score",
    "ingress_slope",
    "egress_slope",
    "ingress_egress_ratio",
    "transit_asymmetry",
    "transit_skewness",
    "transit_kurtosis",
    "bottom_curvature",
    "shoulder_curvature",
    "residual_rms",
    "transit_compactness",
    "transit_sharpness",
    "period_uncertainty",
    "period_stability",
    "n_observed_transits",
    "median_spacing",
    "spacing_variance",
    "transit_timing_scatter",
    "period_alias_score",
    "period_harmonic_score",
    "fractional_phase_coverage",
    "global_mean",
    "global_median",
    "flux_variance",
    "flux_rms",
    "iqr",
    "max_flux",
    "min_flux",
    "peak_to_peak",
    "percentile_spread",
    "flux_entropy",
    "flux_energy",
    "autocorr_lag1",
    "autocorr_lag10",
    "median_p2p_scatter",
    "rolling_rms",
    "rolling_mad",
    "outlier_fraction",
    "robust_scatter",
    "noise_stationarity",
    "low_freq_power",
    "high_freq_power",
    "variability_amplitude",
    "variability_rms",
    "long_term_slope",
    "dominant_frequency",
    "dominant_freq_amplitude",
    "spectral_entropy",
    "spectral_slope",
    "spectral_centroid",
    "spectral_spread",
    "wavelet_approx_energy",
    "wavelet_detail_energy",
    "wavelet_entropy",
    "observation_length",
    "cadence",
    "cadence_gaps",
    "missing_fraction",
    "coverage_score",
    "secondary_eclipse_depth",
    "transit_shape_ratio",
    "blend_score",
    "oot_variability",
    "template_planet_score",
    "template_residual_rms",
]

FEATURE_NAMES = CORE_FEATURE_NAMES + EXTENDED_FEATURE_NAMES
MASK_PREFIX = "mask__"

PERIODIC_FEATURES = {
    "orbital_period",
    "transit_epoch",
    "duty_cycle",
    "period_uncertainty",
    "period_stability",
    "n_observed_transits",
    "median_spacing",
    "spacing_variance",
    "transit_timing_scatter",
    "period_alias_score",
    "period_harmonic_score",
    "fractional_phase_coverage",
    "odd_even_depth_difference",
}


def feature_mask_name(feature_name: str) -> str:
    """Return the canonical availability-mask column name for a feature."""
    return f"{MASK_PREFIX}{feature_name}"


def model_input_names(use_masks: bool = True) -> list[str]:
    """Return the ordered model input names for feature-only or feature+mask ML."""
    names = list(FEATURE_NAMES)
    if use_masks:
        names.extend(feature_mask_name(name) for name in FEATURE_NAMES)
    return names


def build_model_input(
    feature_vector: np.ndarray,
    feature_masks: np.ndarray | None = None,
    use_masks: bool = True,
) -> np.ndarray:
    """Build one fixed-order model input vector from values and availability masks."""
    values = np.asarray(feature_vector, dtype=float)
    if not use_masks:
        return values
    if feature_masks is None:
        masks = np.isfinite(values).astype(float)
    else:
        masks = np.asarray(feature_masks, dtype=float)
    if values.shape != masks.shape:
        raise ValueError("feature_vector and feature_masks must share shape")
    return np.concatenate([values, masks])


def dataframe_to_model_matrix(df, use_masks: bool = True) -> tuple[np.ndarray, list[str]]:
    """Convert a feature DataFrame to the deterministic model matrix.

    Missing feature columns are represented as NaN and, when masks are enabled,
    receive a zero availability mask. Existing mask columns are honored.
    """
    n_rows = len(df)
    columns = []
    for name in FEATURE_NAMES:
        if name in df:
            columns.append(np.asarray(df[name], dtype=float))
        else:
            columns.append(np.full(n_rows, np.nan, dtype=float))
    features = np.column_stack(columns) if columns else np.empty((n_rows, 0))

    if not use_masks:
        return features, model_input_names(False)

    masks = []
    for idx, name in enumerate(FEATURE_NAMES):
        mask_col = feature_mask_name(name)
        if mask_col in df:
            masks.append(np.asarray(df[mask_col], dtype=float))
        else:
            masks.append(np.isfinite(features[:, idx]).astype(float))
    mask_matrix = np.column_stack(masks) if masks else np.empty((n_rows, 0))
    return np.column_stack([features, mask_matrix]), model_input_names(True)


class FeatureExtractor:
    """Build the fixed-dimension feature vector for ML classification."""

    def __init__(self, cfg: dict) -> None:
        self.weights = cfg.get("weights", {})
        self.weighting_enabled = bool(cfg.get("weighting_enabled", False))
        n_bins = cfg.get("n_bins") or cfg.get("phase_folding", {}).get("n_bins", 200)
        self.folder = PhaseFolder({"n_bins": int(n_bins)})

    def extract(
        self,
        lc: LightCurve,
        bls: BLSResult,
        quality_metrics: dict | None = None,
        is_mono: bool = False,
        mono_result=None,
    ) -> dict[str, float]:
        """Return fixed-schema features without inventing unavailable values."""
        flux = np.asarray(lc.flux, dtype=float)
        time = np.asarray(lc.time, dtype=float)
        if len(flux) < 3 or len(time) != len(flux):
            raise ValueError("Feature extraction requires at least three aligned samples")
        median = float(np.median(flux))
        robust_sigma = _robust_sigma(flux)
        mad = float(np.median(np.abs(flux - median)))

        mono_best = getattr(mono_result, "best", None) if is_mono else None
        if mono_best is not None:
            depth = float(mono_best.depth)
            duration = float(mono_best.duration)
            epoch = float(mono_best.t0)
            transit_snr = float(mono_best.snr)
            period = float("nan")
            transit_power = float("nan")
            profile = _time_event_profile(lc, epoch, duration)
            event_depths = np.array([depth])
        else:
            depth = max(float(bls.best_depth), 0.0)
            duration = float(bls.best_duration)
            epoch = float(bls.best_t0)
            period = float(bls.best_period)
            transit_snr = (
                float(bls.depth_snr)
                if np.isfinite(bls.depth_snr)
                else _safe_ratio(depth, robust_sigma)
            )
            transit_power = float(bls.best_power)
            profile = _periodic_event_profile(lc, bls)
            event_depths = _per_transit_depths(lc, bls)

        morphology = _morphology(profile, duration)
        ingress_duration = morphology.pop("ingress_duration")
        egress_duration = morphology.pop("egress_duration")
        finite_depths = event_depths[np.isfinite(event_depths) & (event_depths >= 0)]
        median_depth = float(np.median(finite_depths)) if len(finite_depths) else depth
        depth_variance = float(np.var(finite_depths, ddof=1)) if len(finite_depths) > 1 else float("nan")
        depth_consistency = (
            float(1.0 / (1.0 + np.std(finite_depths, ddof=1) / max(median_depth, 1e-12)))
            if len(finite_depths) > 1 and median_depth > 0
            else float("nan")
        )
        persistence = (
            float(np.mean(finite_depths > 3.0 * max(self._local_noise(flux), 1e-12)))
            if len(finite_depths)
            else float("nan")
        )

        periodic = not is_mono and np.isfinite(period) and period > 0
        duty_cycle = duration / period if periodic else float("nan")
        power = np.asarray(bls.power, dtype=float)
        finite_power = power[np.isfinite(power)]
        integrated_power = (
            float(np.mean(np.clip(finite_power - np.median(finite_power), 0.0, None)))
            if len(finite_power)
            else float("nan")
        )
        secondary_power = float(bls.secondary_peak_power)

        feats: dict[str, float] = {
            "transit_depth": depth,
            "transit_duration": duration,
            "orbital_period": period if periodic else float("nan"),
            "transit_snr": transit_snr,
            "transit_power": transit_power,
            "flux_std": robust_sigma,
            "flux_mad": mad,
            "local_noise": self._local_noise(flux),
            "skewness": _finite_stat(skew(flux, nan_policy="omit")),
            "kurtosis": _finite_stat(kurtosis(flux, nan_policy="omit")),
            "odd_even_depth_difference": self._odd_even(lc, bls) if periodic else float("nan"),
            "transit_epoch": epoch,
            "duty_cycle": duty_cycle,
            "ingress_duration": ingress_duration,
            "egress_duration": egress_duration,
            "estimated_impact_parameter": float("nan"),
            "depth_duration_ratio": _safe_ratio(depth, duration),
            "depth_symmetry": _depth_symmetry(profile),
            "transit_width": duration,
            "transit_occupancy": _transit_occupancy(lc, bls) if periodic else float("nan"),
            "normalized_transit_width": duty_cycle,
            "fractional_transit_duration": duty_cycle,
            "integrated_bls_power": integrated_power if periodic else float("nan"),
            "secondary_peak_power": secondary_power if periodic else float("nan"),
            "primary_secondary_ratio": _safe_ratio(transit_power, secondary_power) if periodic else float("nan"),
            "detection_significance": float(bls.sde) if periodic else transit_snr,
            "bootstrap_fap": float("nan"),
            "local_signal_contrast": _safe_ratio(depth, self._local_noise(flux)),
            "global_signal_contrast": _safe_ratio(depth, robust_sigma),
            "median_transit_depth": median_depth,
            "depth_variance": depth_variance,
            "depth_consistency": depth_consistency,
            "signal_persistence": persistence,
        }
        feats.update(morphology)
        if periodic:
            feats.update(_periodicity(lc, bls))
        else:
            feats.update({name: float("nan") for name in PERIODIC_FEATURES if name in FEATURE_NAMES})

        uniform_flux, uniform_time = _uniform_longest_segment(time, flux)
        feats.update(_flux_stats(flux))
        feats.update(_noise_stats(flux))
        feats.update(_variability(uniform_flux, uniform_time))
        feats.update(_spectral(uniform_flux, uniform_time))
        feats.update(_wavelet(uniform_flux))
        feats.update(_quality(time, flux, quality_metrics))
        feats.update(_fp_diagnostics(lc, bls, profile, is_mono))
        feats.update(_template_scores(profile))
        return {name: _finite_or_nan(feats.get(name, float("nan"))) for name in FEATURE_NAMES}

    def vectorize(
        self, feats: dict[str, float], apply_weights: bool = True
    ) -> np.ndarray:
        vec = np.array([feats.get(k, float("nan")) for k in FEATURE_NAMES], dtype=float)
        if apply_weights and self.weighting_enabled:
            w = np.array([self.weights.get(k, 1.0) for k in FEATURE_NAMES], dtype=float)
            vec = np.where(np.isfinite(vec), vec * w, vec)
        return vec

    def masks(self, feats: dict[str, float]) -> np.ndarray:
        """Binary mask: 1 = available, 0 = missing."""
        return np.array(
            [1.0 if np.isfinite(feats.get(k, float("nan"))) else 0.0 for k in FEATURE_NAMES],
            dtype=float,
        )

    def describe(self, feats: dict[str, float]) -> dict:
        return {
            "n_features": len(FEATURE_NAMES),
            "n_available": int(np.sum(self.masks(feats))),
            "names": FEATURE_NAMES,
        }

    def validate(self, feats: dict[str, float]) -> dict:
        """Return a structural validation report for an extracted feature dict."""
        missing_names = [name for name in FEATURE_NAMES if name not in feats]
        extra_names = [name for name in feats if name not in FEATURE_NAMES]
        masks = self.masks(feats)
        return {
            "valid": not missing_names,
            "n_features": len(FEATURE_NAMES),
            "n_available": int(np.sum(masks)),
            "missing_names": missing_names,
            "extra_names": extra_names,
        }

    @staticmethod
    def _local_noise(flux: np.ndarray) -> float:
        if len(flux) < 2:
            return float("nan")
        return float(_robust_sigma(np.diff(flux)) / np.sqrt(2.0))

    def _binned_transit(
        self, phase: np.ndarray, flux: np.ndarray, duration: float, period: float
    ) -> np.ndarray:
        if len(phase) == 0 or period <= 0:
            return np.array([])
        width = max(duration / period, 1e-6)
        delta = ((phase - 0.5 + 0.5) % 1.0) - 0.5
        selected = np.abs(delta) <= 2.5 * width
        if np.count_nonzero(selected) < 5:
            return np.array([])
        edges = np.linspace(-2.5 * width, 2.5 * width, 62)
        index = np.digitize(delta[selected], edges) - 1
        binned = np.full(61, np.nan)
        for bin_index in range(61):
            values = flux[selected][index == bin_index]
            if len(values):
                binned[bin_index] = np.median(values)
        return _interpolate_missing_profile(binned)

    def _odd_even(self, lc: LightCurve, bls: BLSResult) -> float:
        period = bls.best_period
        t0 = bls.best_t0
        if period <= 0:
            return 0.0
        cycle = np.floor((lc.time - t0) / period + 0.5)
        phase = ((lc.time - t0) / period + 0.5) % 1.0
        in_transit = np.abs(phase - 0.5) <= (0.5 * bls.best_duration / period)
        base = np.median(lc.flux[~in_transit]) if np.any(~in_transit) else 1.0
        odd = in_transit & (cycle.astype(int) % 2 == 1)
        even = in_transit & (cycle.astype(int) % 2 == 0)
        if np.count_nonzero(odd) < 2 or np.count_nonzero(even) < 2:
            return float("nan")
        d_odd = base - np.median(lc.flux[odd])
        d_even = base - np.median(lc.flux[even])
        return float(abs(d_odd - d_even))


def _depth_symmetry(binned: np.ndarray) -> float:
    if len(binned) < 4 or not np.all(np.isfinite(binned)):
        return float("nan")
    baseline = float(np.median(np.concatenate((binned[:5], binned[-5:]))))
    depth = baseline - float(np.min(binned))
    if depth <= 0:
        return float("nan")
    mid = len(binned) // 2
    mismatch = np.median(np.abs(binned[:mid] - binned[-mid:][::-1])) / depth
    return float(np.clip(1.0 - mismatch, 0.0, 1.0))


def _shape_durations(binned: np.ndarray, total_duration: float) -> tuple[float, float]:
    """Estimate 10%-to-90% ingress and egress durations."""
    if len(binned) < 11 or total_duration <= 0 or not np.all(np.isfinite(binned)):
        return float("nan"), float("nan")
    edge_count = max(2, len(binned) // 8)
    baseline = float(np.median(np.concatenate((binned[:edge_count], binned[-edge_count:]))))
    bottom = float(np.min(binned))
    depth = baseline - bottom
    if depth <= 0:
        return float("nan"), float("nan")
    deficit = baseline - binned
    center = int(np.argmax(deficit))
    left = deficit[: center + 1]
    right = deficit[center:]
    left_10 = np.flatnonzero(left >= 0.1 * depth)
    left_90 = np.flatnonzero(left >= 0.9 * depth)
    right_90 = np.flatnonzero(right >= 0.9 * depth)
    right_10 = np.flatnonzero(right >= 0.1 * depth)
    if not all(len(values) for values in (left_10, left_90, right_90, right_10)):
        return float("nan"), float("nan")
    bin_time = 5.0 * total_duration / max(len(binned) - 1, 1)
    ingress = max(float(left_90[0] - left_10[0]) * bin_time, 0.0)
    egress = max(float(right_10[-1] - right_90[-1]) * bin_time, 0.0)
    return ingress, egress


def _morphology(binned: np.ndarray, duration: float) -> dict[str, float]:
    if len(binned) < 11 or not np.all(np.isfinite(binned)):
        return {
            name: float("nan") for name in (
                "v_shape_score", "u_shape_score", "flat_bottom_score",
                "ingress_slope", "egress_slope", "ingress_egress_ratio",
                "transit_asymmetry", "transit_skewness", "transit_kurtosis",
                "bottom_curvature", "shoulder_curvature", "residual_rms",
                "transit_compactness", "transit_sharpness", "ingress_duration",
                "egress_duration",
            )
        }
    edge_count = max(2, len(binned) // 8)
    baseline = float(np.median(np.concatenate((binned[:edge_count], binned[-edge_count:]))))
    depth = baseline - float(np.min(binned))
    if depth <= 0:
        return _morphology(np.array([]), duration)
    profile = np.clip((baseline - binned) / depth, -1.0, 2.0)
    mid = int(np.argmax(profile))
    ingress_duration, egress_duration = _shape_durations(binned, duration)
    half = max(2, len(profile) // 12)
    bottom = profile[max(0, mid - half) : min(len(profile), mid + half + 1)]
    bottom_scatter = _robust_sigma(bottom)
    flat_bottom = float(np.clip(1.0 - bottom_scatter, 0.0, 1.0))
    symmetry = _depth_symmetry(binned)
    v_shape = float(np.clip(1.0 - flat_bottom, 0.0, 1.0))
    ingress_slope = _safe_ratio(depth, ingress_duration)
    egress_slope = _safe_ratio(depth, egress_duration)
    second = np.diff(profile, n=2)
    shoulder = np.concatenate((profile[:edge_count * 2], profile[-edge_count * 2 :]))
    template = _planet_template(len(profile))
    residual_rms = float(np.sqrt(np.mean((profile - template) ** 2)))
    return {
        "v_shape_score": v_shape,
        "u_shape_score": float(np.clip(flat_bottom * symmetry, 0.0, 1.0)),
        "flat_bottom_score": flat_bottom,
        "ingress_slope": ingress_slope,
        "egress_slope": egress_slope,
        "ingress_egress_ratio": _safe_ratio(ingress_duration, egress_duration),
        "transit_asymmetry": float(1.0 - symmetry),
        "transit_skewness": _finite_stat(skew(profile, nan_policy="omit")),
        "transit_kurtosis": _finite_stat(kurtosis(profile, nan_policy="omit")),
        "bottom_curvature": float(np.mean(np.abs(np.diff(bottom, n=2)))) if len(bottom) >= 3 else float("nan"),
        "shoulder_curvature": float(np.mean(np.abs(np.diff(shoulder, n=2)))) if len(shoulder) >= 3 else float("nan"),
        "residual_rms": residual_rms,
        "transit_compactness": float(np.mean(profile > 0.5)),
        "transit_sharpness": float(np.max(np.abs(np.diff(profile)))) + (float(np.mean(np.abs(second))) if len(second) else 0.0),
        "ingress_duration": ingress_duration,
        "egress_duration": egress_duration,
    }


def _periodicity(lc: LightCurve, bls: BLSResult) -> dict[str, float]:
    period = bls.best_period
    timings = _event_timings(lc, bls)
    spacing = np.diff(timings) if len(timings) > 1 else np.array([])
    timing_residual = (
        timings - (timings[0] + np.rint((timings - timings[0]) / period) * period)
        if len(timings) > 1 and period > 0
        else np.array([])
    )
    phase = ((lc.time - bls.best_t0) / period) % 1.0
    occupied, _ = np.histogram(phase, bins=100)
    return {
        "period_uncertainty": float(bls.period_uncertainty),
        "period_stability": float(bls.sde),
        "n_observed_transits": float(bls.n_transits),
        "median_spacing": float(np.median(spacing)) if len(spacing) else float("nan"),
        "spacing_variance": float(np.var(spacing, ddof=1)) if len(spacing) > 1 else float("nan"),
        "transit_timing_scatter": float(_robust_sigma(timing_residual)) if len(timing_residual) else float("nan"),
        "period_alias_score": _period_alias_score(bls),
        "period_harmonic_score": _harmonic_score(bls),
        "fractional_phase_coverage": float(np.mean(occupied > 0)),
    }


def _harmonic_score(bls: BLSResult) -> float:
    periods = np.asarray(bls.periods)
    power = np.asarray(bls.power)
    if len(periods) < 2 or bls.best_period <= 0:
        return 0.0
    harmonics = [0.5 * bls.best_period, 2.0 * bls.best_period]
    scores = []
    for target in harmonics:
        idx = int(np.argmin(np.abs(periods - target)))
        scores.append(power[idx] / (bls.best_power + 1e-9))
    return float(np.clip(max(scores), 0.0, 1.0))


def _period_alias_score(bls: BLSResult) -> float:
    """Return the strongest common 0.5x/2x period alias relative to the primary."""
    if bls.best_period <= 0 or not np.isfinite(bls.best_power) or bls.best_power <= 0:
        return float("nan")
    periods = np.asarray(bls.periods, dtype=float)
    power = np.asarray(bls.power, dtype=float)
    scores = []
    for ratio in (0.5, 2.0):
        target = ratio * bls.best_period
        if target < periods[0] or target > periods[-1]:
            continue
        index = int(np.nanargmin(np.abs(periods - target)))
        if np.isfinite(power[index]):
            scores.append(power[index] / bls.best_power)
    return float(np.clip(max(scores), 0.0, 1.0)) if scores else float("nan")


def _periodic_event_profile(lc: LightCurve, bls: BLSResult) -> np.ndarray:
    if bls.best_period <= 0 or bls.best_duration <= 0:
        return np.array([])
    relative_time = (
        (lc.time - bls.best_t0 + 0.5 * bls.best_period) % bls.best_period
    ) - 0.5 * bls.best_period
    return _bin_event_profile(relative_time, lc.flux, bls.best_duration)


def _time_event_profile(lc: LightCurve, epoch: float, duration: float) -> np.ndarray:
    return _bin_event_profile(lc.time - epoch, lc.flux, duration)


def _bin_event_profile(relative_time: np.ndarray, flux: np.ndarray, duration: float) -> np.ndarray:
    if duration <= 0 or not np.isfinite(duration):
        return np.array([])
    selected = np.abs(relative_time) <= 2.5 * duration
    if np.count_nonzero(selected) < 5:
        return np.array([])
    edges = np.linspace(-2.5 * duration, 2.5 * duration, 62)
    indices = np.digitize(relative_time[selected], edges) - 1
    values = np.asarray(flux)[selected]
    binned = np.full(61, np.nan)
    for index in range(61):
        sample = values[indices == index]
        if len(sample):
            binned[index] = np.median(sample)
    return _interpolate_missing_profile(binned)


def _interpolate_missing_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(profile)
    if np.count_nonzero(finite) < 5:
        return np.array([])
    if not np.all(finite):
        x = np.arange(len(profile))
        profile = profile.copy()
        profile[~finite] = np.interp(x[~finite], x[finite], profile[finite])
    return profile


def _transit_occupancy(lc: LightCurve, bls: BLSResult) -> float:
    mask, _ = _periodic_event_membership(lc.time, bls)
    return float(np.mean(mask)) if len(mask) else float("nan")


def _per_transit_depths(lc: LightCurve, bls: BLSResult) -> np.ndarray:
    in_transit, cycles = _periodic_event_membership(lc.time, bls)
    if not np.any(in_transit):
        return np.array([])
    baseline = float(np.median(lc.flux[~in_transit])) if np.any(~in_transit) else float(np.median(lc.flux))
    depths = []
    for cycle in np.unique(cycles[in_transit]):
        selected = in_transit & (cycles == cycle)
        if np.count_nonzero(selected) >= 2:
            depths.append(baseline - float(np.median(lc.flux[selected])))
    return np.asarray(depths, dtype=float)


def _event_timings(lc: LightCurve, bls: BLSResult) -> np.ndarray:
    in_transit, cycles = _periodic_event_membership(lc.time, bls)
    if not np.any(in_transit):
        return np.array([])
    baseline = float(np.median(lc.flux[~in_transit])) if np.any(~in_transit) else float(np.median(lc.flux))
    timings = []
    for cycle in np.unique(cycles[in_transit]):
        selected = in_transit & (cycles == cycle)
        if np.count_nonzero(selected) < 2:
            continue
        weights = np.clip(baseline - lc.flux[selected], 0.0, None)
        if np.sum(weights) > 0:
            timings.append(float(np.sum(lc.time[selected] * weights) / np.sum(weights)))
    return np.asarray(timings, dtype=float)


def _periodic_event_membership(
    time: np.ndarray, bls: BLSResult
) -> tuple[np.ndarray, np.ndarray]:
    if bls.best_period <= 0 or bls.best_duration <= 0:
        return np.zeros(len(time), dtype=bool), np.zeros(len(time), dtype=np.int64)
    cycles = np.rint((np.asarray(time) - bls.best_t0) / bls.best_period).astype(np.int64)
    centers = bls.best_t0 + cycles * bls.best_period
    return np.abs(np.asarray(time) - centers) <= 0.5 * bls.best_duration, cycles


def _uniform_longest_segment(
    time: np.ndarray,
    flux: np.ndarray,
    max_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only the longest contiguous segment onto a uniform grid."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if len(time) < 3:
        return flux, time
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if len(positive) == 0:
        return flux, np.arange(len(flux), dtype=float)
    cadence = float(np.median(positive))
    breaks = np.flatnonzero(dt > 5.0 * cadence) + 1
    edges = np.concatenate(([0], breaks, [len(time)]))
    start, stop = max(
        ((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])),
        key=lambda pair: pair[1] - pair[0],
    )
    segment_time = time[start:stop]
    segment_flux = flux[start:stop]
    if len(segment_time) < 3 or segment_time[-1] <= segment_time[0]:
        return segment_flux, segment_time
    count = min(len(segment_time), max_points)
    uniform_time = np.linspace(segment_time[0], segment_time[-1], count)
    uniform_flux = np.interp(uniform_time, segment_time, segment_flux)
    return uniform_flux, uniform_time


def _planet_template(length: int) -> np.ndarray:
    x = np.abs(np.linspace(-2.5, 2.5, length))
    template = np.zeros(length, dtype=float)
    template[x <= 0.35] = 1.0
    ingress = (x > 0.35) & (x < 0.5)
    template[ingress] = (0.5 - x[ingress]) / 0.15
    return template


def _spectral_integral(frequency: np.ndarray, power: np.ndarray) -> float:
    if len(frequency) < 2:
        return 0.0
    return _trapz(np.asarray(power, dtype=float), np.asarray(frequency, dtype=float))


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    center = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - center)))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = float(np.std(values))
    return sigma if np.isfinite(sigma) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= 1e-15:
        return float("nan")
    return float(numerator / denominator)


def _finite_stat(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _finite_or_nan(value) -> float:
    return _finite_stat(value)


def _flux_stats(flux: np.ndarray) -> dict[str, float]:
    q25, q75 = np.percentile(flux, [25, 75])
    hist, _ = np.histogram(flux, bins=20, density=False)
    hist = hist[hist > 0]
    centered = flux - np.median(flux)
    return {
        "global_mean": float(np.mean(flux)),
        "global_median": float(np.median(flux)),
        "flux_variance": float(np.var(flux)),
        "flux_rms": float(np.sqrt(np.mean(centered ** 2))),
        "iqr": float(q75 - q25),
        "max_flux": float(np.max(flux)),
        "min_flux": float(np.min(flux)),
        "peak_to_peak": float(np.ptp(flux)),
        "percentile_spread": float(q75 - q25),
        "flux_entropy": float(entropy(hist)) if len(hist) else 0.0,
        "flux_energy": float(np.mean((flux - np.mean(flux)) ** 2)),
        "autocorr_lag1": _autocorr(flux, 1),
        "autocorr_lag10": _autocorr(flux, 10),
    }


def _autocorr(flux: np.ndarray, lag: int) -> float:
    if len(flux) <= lag:
        return 0.0
    x = flux - np.mean(flux)
    denom = np.dot(x, x)
    if denom <= 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def _noise_stats(flux: np.ndarray) -> dict[str, float]:
    med = np.median(flux)
    robust_sigma = _robust_sigma(flux)
    win = min(51, len(flux) // 4 | 1)
    roll = (
        np.array([np.std(flux[max(0, i - win): i + win]) for i in range(0, len(flux), win)])
        if len(flux) > win else np.array([np.std(flux)])
    )
    return {
        "median_p2p_scatter": float(np.median(np.abs(np.diff(flux)))) if len(flux) > 1 else 0.0,
        "rolling_rms": float(np.mean(roll)),
        "rolling_mad": float(np.median(roll)),
        "outlier_fraction": float(np.mean(np.abs(flux - med) > 5 * max(robust_sigma, 1e-12))),
        "robust_scatter": float(robust_sigma),
        "noise_stationarity": float(1.0 / (1.0 + _safe_ratio(_robust_sigma(roll), np.median(roll)))),
    }


def _variability(flux: np.ndarray, time: np.ndarray) -> dict[str, float]:
    if len(flux) < 10:
        return {
            "low_freq_power": 0.0, "high_freq_power": 0.0,
            "variability_amplitude": 0.0, "variability_rms": 0.0, "long_term_slope": 0.0,
        }
    dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
    f, pxx = signal.periodogram(
        flux - np.mean(flux), fs=1.0 / dt, window="hann", scaling="density"
    )
    split = np.searchsorted(f, 1.0)
    slope = float(np.polyfit(time - time[0], flux, 1)[0]) if len(time) > 2 else 0.0
    return {
        "low_freq_power": _spectral_integral(f[:split], pxx[:split]),
        "high_freq_power": _spectral_integral(f[split:], pxx[split:]),
        "variability_amplitude": float(np.ptp(flux)),
        "variability_rms": float(np.std(flux)),
        "long_term_slope": slope,
    }


def _spectral(flux: np.ndarray, time: np.ndarray) -> dict[str, float]:
    if len(flux) < 8:
        return {
            "dominant_frequency": 0.0, "dominant_freq_amplitude": 0.0,
            "spectral_entropy": 0.0, "spectral_slope": 0.0,
            "spectral_centroid": 0.0, "spectral_spread": 0.0,
        }
    cadence = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    freqs, psd = signal.periodogram(
        flux - np.mean(flux),
        fs=1.0 / cadence,
        window="hann",
        scaling="density",
    )
    spec = np.sqrt(np.maximum(psd, 0.0))
    idx = int(np.argmax(psd[1:]) + 1) if len(psd) > 1 else 0
    psd_sum = np.sum(psd) + 1e-12
    psd_n = psd / (np.sum(psd) + 1e-12)
    psd_n = psd_n[psd_n > 0]
    centroid = float(np.sum(freqs * psd) / psd_sum)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum))
    return {
        "dominant_frequency": float(freqs[idx]),
        "dominant_freq_amplitude": float(spec[idx]),
        "spectral_entropy": float(entropy(psd_n)) if len(psd_n) else 0.0,
        "spectral_slope": float(np.polyfit(np.log(freqs[1:] + 1e-12), np.log(psd[1:] + 1e-24), 1)[0]) if len(freqs) > 2 else 0.0,
        "spectral_centroid": centroid,
        "spectral_spread": spread,
    }


def _wavelet(flux: np.ndarray) -> dict[str, float]:
    """Compute small deterministic Haar-wavelet summaries without extra deps."""
    x = np.asarray(flux, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return {
            "wavelet_approx_energy": 0.0,
            "wavelet_detail_energy": 0.0,
            "wavelet_entropy": 0.0,
        }
    x = x - np.mean(x)
    details = []
    approx = x.copy()
    for _ in range(3):
        if len(approx) < 2:
            break
        if len(approx) % 2:
            approx = approx[:-1]
        even = approx[0::2]
        odd = approx[1::2]
        details.append((even - odd) / np.sqrt(2))
        approx = (even + odd) / np.sqrt(2)
    approx_energy = float(np.sum(approx ** 2))
    detail_energy = float(sum(np.sum(d ** 2) for d in details))
    energies = np.array([np.sum(d ** 2) for d in details] + [approx_energy], dtype=float)
    probs = energies / (np.sum(energies) + 1e-12)
    probs = probs[probs > 0]
    return {
        "wavelet_approx_energy": approx_energy,
        "wavelet_detail_energy": detail_energy,
        "wavelet_entropy": float(entropy(probs)) if len(probs) else 0.0,
    }


def _quality(
    time: np.ndarray, flux: np.ndarray, metrics: dict | None
) -> dict[str, float]:
    if metrics:
        return {
            "observation_length": metrics.get("observation_length", float(time[-1] - time[0])),
            "cadence": metrics.get("cadence", 0.0),
            "cadence_gaps": metrics.get("cadence_gaps", 0.0),
            "missing_fraction": metrics.get("missing_fraction", 0.0),
            "coverage_score": metrics.get("coverage_score", 1.0),
        }
    span = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    dt = np.median(np.diff(time)) if len(time) > 1 else 1.0
    return {
        "observation_length": span,
        "cadence": float(dt),
        "cadence_gaps": 0.0,
        "missing_fraction": float(np.mean(~np.isfinite(flux))),
        "coverage_score": 1.0,
    }


def _fp_diagnostics(
    lc: LightCurve, bls: BLSResult, binned: np.ndarray, is_mono: bool
) -> dict[str, float]:
    oot = np.asarray(lc.flux, dtype=float)
    secondary_depth = float("nan")
    if bls.best_period > 0 and not is_mono:
        phase = ((lc.time - bls.best_t0) / bls.best_period) % 1.0
        primary_distance = np.minimum(phase, 1.0 - phase)
        half_width = 0.5 * bls.best_duration / bls.best_period
        primary = primary_distance <= half_width
        secondary = np.abs(phase - 0.5) <= half_width
        baseline_mask = ~(primary | secondary)
        oot = lc.flux[baseline_mask]
        if np.count_nonzero(secondary) >= 3 and np.count_nonzero(baseline_mask) >= 5:
            secondary_depth = float(
                np.median(lc.flux[baseline_mask]) - np.median(lc.flux[secondary])
            )
    crowding = lc.meta.get("crowding", np.nan)
    try:
        blend_score = 1.0 - float(crowding)
        blend_score = float(np.clip(blend_score, 0.0, 1.0))
    except (TypeError, ValueError):
        blend_score = float("nan")
    morph = _morphology(binned, max(float(bls.best_duration), 1e-12))
    return {
        "secondary_eclipse_depth": secondary_depth,
        "transit_shape_ratio": _safe_ratio(
            morph.get("ingress_duration", np.nan) + morph.get("egress_duration", np.nan),
            float(bls.best_duration),
        ),
        "blend_score": blend_score,
        "oot_variability": _robust_sigma(oot) if len(oot) else float("nan"),
    }


def _template_scores(binned: np.ndarray) -> dict[str, float]:
    if len(binned) < 5 or not np.all(np.isfinite(binned)):
        return {"template_planet_score": float("nan"), "template_residual_rms": float("nan")}
    edge_count = max(2, len(binned) // 8)
    baseline = float(np.median(np.concatenate((binned[:edge_count], binned[-edge_count:]))))
    depth = baseline - float(np.min(binned))
    if depth <= 0:
        return {"template_planet_score": float("nan"), "template_residual_rms": float("nan")}
    observed = np.clip((baseline - binned) / depth, -1.0, 2.0)
    planet = _planet_template(len(observed))
    x = np.linspace(-2.5, 2.5, len(observed))
    v_shape = np.clip(1.0 - 2.0 * np.abs(x), 0.0, 1.0)
    gaussian = np.exp(-0.5 * (x / 0.35) ** 2)
    rms = np.array([
        np.sqrt(np.mean((observed - template) ** 2))
        for template in (planet, v_shape, gaussian)
    ])
    likelihood = np.exp(-rms / 0.25)
    score = float(likelihood[0] / np.sum(likelihood)) if np.sum(likelihood) > 0 else float("nan")
    return {"template_planet_score": score, "template_residual_rms": float(rms[0])}
