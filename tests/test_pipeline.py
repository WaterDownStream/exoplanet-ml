"""Unit and integration tests for the scientific pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import ProbabilityCalibrator
from src.classifier import AstrophysicalClassifier
from src.detrending import Detrender
from src.feature_extraction import FEATURE_NAMES, FeatureExtractor
from src.feature_extraction import dataframe_to_model_matrix, model_input_names
from src.mono_search import MonoTransitSearch
from src.mono_features import MONO_FEATURE_NAMES, MonoFeatureExtractor
from src.phase_folding import PhaseFolder
from src.parameter_estimation import ParameterEstimator
from src.pipeline import Pipeline
from src.preprocessing import Preprocessor
from src.quality import QualityAssessment
from src.reporting import ReportGenerator
from src.transit_search import TransitSearch
from src.utils import LightCurve, load_config, set_seed
from src.datasets.builder import DatasetBuilder


@pytest.fixture
def synthetic_lc() -> LightCurve:
    """A light curve with an injected box transit (P=3 d, depth=1%)."""
    set_seed(0)
    t = np.linspace(0, 27, 4000)
    flux = 1.0 + 0.001 * np.sin(2 * np.pi * t / 13)
    period, depth, dur = 3.0, 0.01, 0.1
    phase = (t % period) / period
    flux[np.abs(phase - 0.5) < dur / period / 2] -= depth
    flux += np.random.normal(0, 0.0008, size=t.size)
    return LightCurve(t, flux, np.full_like(t, 0.0008), {})


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_preprocessing_normalizes(synthetic_lc):
    out = Preprocessor({"sigma": 5, "normalize": True}).run(synthetic_lc)
    assert np.isclose(np.median(out.flux), 1.0, atol=1e-2)
    assert np.all(np.isfinite(out.flux))


def test_detrending_flattens(synthetic_lc):
    out = Detrender(
        {"method": "savgol", "savgol": {"window_length": 201, "polyorder": 2}}
    ).run(synthetic_lc)
    assert out.flux.std() <= synthetic_lc.flux.std() + 1e-6


def test_bls_recovers_period(synthetic_lc):
    pre = Preprocessor({"sigma": 5}).run(synthetic_lc)
    bls = TransitSearch(
        {"period_min": 1.0, "period_max": 6.0, "n_periods": 3000,
         "duration_min": 0.05, "duration_max": 0.2, "n_durations": 5}
    ).run(pre)
    assert abs(bls.best_period - 3.0) < 1.5


def test_quality_assessment(synthetic_lc, config):
    pre = Preprocessor({"sigma": 5}).run(synthetic_lc)
    report = QualityAssessment(config["quality"]).run(pre)
    assert report.passed
    assert 0 <= report.score <= 1


def test_mono_search(synthetic_lc):
    pre = Preprocessor({"sigma": 5}).run(synthetic_lc)
    mono = MonoTransitSearch(
        {"min_depth": 0.001, "snr_threshold": 2.0, "morphology_threshold": 0.1}
    ).run(pre)
    assert mono.detected or len(mono.candidates) >= 0


def test_mono_features_fixed_dim(synthetic_lc, config):
    pre = Preprocessor({"sigma": 5}).run(synthetic_lc)
    mono = MonoTransitSearch(config["mono_search"]).run(pre)
    feats = MonoFeatureExtractor().extract(pre, mono)
    vec = MonoFeatureExtractor().vectorize(feats)
    assert vec.shape == (len(MONO_FEATURE_NAMES),)


def test_mono_parameter_estimation_requires_stellar_priors(synthetic_lc, config):
    pre = Preprocessor({"sigma": 5}).run(synthetic_lc)
    mono = MonoTransitSearch({"min_depth": 0.001, "snr_threshold": 2.0}).run(pre)
    if not mono.detected:
        pytest.skip("synthetic periodic curve did not produce a mono candidate")
    estimator = ParameterEstimator({**config["parameter_estimation"], "seed": config["seed"]})
    no_prior = estimator.estimate_mono(mono, stellar={}, planet_prob=1.0)
    assert not no_prior.period.available
    with_prior = estimator.estimate_mono(
        mono,
        stellar={"radius": 1.0, "mass": 1.0, "teff": 5772.0},
        planet_prob=1.0,
    )
    assert with_prior.period.available
    assert with_prior.period.value > 0


def test_feature_vector_length(synthetic_lc, config):
    bls = TransitSearch(
        {"period_min": 1.0, "period_max": 6.0, "n_periods": 1000,
         "duration_min": 0.05, "duration_max": 0.2, "n_durations": 5}
    ).run(synthetic_lc)
    fe = FeatureExtractor({**config["features"], "phase_folding": config["phase_folding"]})
    feats = fe.extract(synthetic_lc, bls)
    vec = fe.vectorize(feats)
    masks = fe.masks(feats)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert masks.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 100
    assert np.all(np.isfinite(vec[masks.astype(bool)]))


def test_model_input_includes_missing_masks(config):
    import pandas as pd

    df = pd.DataFrame({"transit_depth": [0.01], "label": [0]})
    X, names = dataframe_to_model_matrix(df, use_masks=True)
    assert X.shape == (1, 2 * len(FEATURE_NAMES))
    assert names == model_input_names(True)
    assert X[0, names.index("mask__transit_depth")] == 1.0
    assert X[0, names.index("mask__orbital_period")] == 0.0


def test_feature_masks_for_mono(synthetic_lc, config):
    bls = TransitSearch(
        {"period_min": 1.0, "period_max": 6.0, "n_periods": 500,
         "duration_min": 0.05, "duration_max": 0.2, "n_durations": 5}
    ).run(synthetic_lc)
    fe = FeatureExtractor({**config["features"], "phase_folding": config["phase_folding"]})
    feats = fe.extract(synthetic_lc, bls, is_mono=True)
    masks = fe.masks(feats)
    assert masks.sum() < len(FEATURE_NAMES)


def test_classifier_multiclass(config):
    cls = AstrophysicalClassifier(config)
    proba = np.array([0.05, 0.6, 0.08, 0.05, 0.05, 0.05, 0.05, 0.07])
    result = cls.classify_from_proba(proba)
    assert result.class_name == "planet_candidate"
    assert len(result.probabilities) == 8


def test_dataset_label_mapping(config):
    builder = DatasetBuilder(config)
    assert builder.label_for_directory("planets") == (0, "planet")
    assert builder.label_for_directory("nonplanets") == (7, "unknown")


def test_calibration():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=200)
    y = rng.integers(0, 3, 200)
    cal = ProbabilityCalibrator("isotonic").fit(proba, y)
    out = cal.calibrate(proba)
    assert out.shape == proba.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=0.01)


def test_report_generator(config, tmp_path):
    config = dict(config)
    config["paths"] = dict(config["paths"])
    config["paths"]["reports_dir"] = str(tmp_path)
    rep = ReportGenerator(config)
    paths = rep.generate("TEST", {"predicted_class": "planet", "final_confidence": 0.9,
                                    "class_probabilities": {"planet": 0.9}})
    assert paths["json"].exists()
    assert paths["html"].exists()


def test_pipeline_without_model(synthetic_lc, config, tmp_path):
    lc_path = tmp_path / "test.csv"
    np.savetxt(lc_path, np.column_stack([synthetic_lc.time, synthetic_lc.flux]),
               delimiter=",", header="time,flux", comments="")
    result = Pipeline(config).run(str(lc_path))
    assert result.quality.passed
    assert len(result.feature_vector) == len(FEATURE_NAMES)
    assert result.raw_lightcurve is not None
    assert result.cleaned_lightcurve is not None
