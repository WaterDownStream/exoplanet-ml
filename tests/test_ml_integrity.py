"""Regression tests for leakage prevention and class/probability contracts."""
from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration import ProbabilityCalibrator
from src.classifier import AstrophysicalClassifier
from src.datasets.splits import DatasetSplitter
from src.evaluator import Evaluator
from src.trainer import Trainer
from src.utils import load_config, validate_config
from train import _validate_class_support


class _SparseClassModel:
    classes_ = np.array([0, 7])

    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]]), (len(X), 1))


class _BinaryModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]]), (len(X), 1))


def _small_training_config(tmp_path):
    config = copy.deepcopy(load_config("configs/config.yaml"))
    config["paths"]["models_dir"] = str(tmp_path)
    config["model"]["type"] = "random_forest"
    config["model"]["random_forest"]["n_estimators"] = 20
    config["calibration"]["enabled"] = False
    return config


def test_preprocessing_is_fit_only_on_training_split(tmp_path):
    config = _small_training_config(tmp_path)
    trainer = Trainer(config)
    X_train = np.column_stack((np.arange(20, dtype=float), np.ones(20)))
    X_train[0, 0] = np.nan
    y_train = np.array([0, 7] * 10)
    X_val = np.full((6, 2), 10_000.0)
    y_val = np.array([0, 7, 0, 7, 0, 7])
    X_test = np.full((6, 2), -10_000.0)
    y_test = y_val.copy()
    trainer.train_pre_split(X_train, y_train, X_val, y_val, X_test, y_test)
    assert trainer.imputer.statistics_[0] == np.median(np.arange(1, 20, dtype=float))
    assert abs(trainer.scaler.center_[0]) < 20.0


def test_global_labels_are_saved_as_explicit_active_classes(tmp_path):
    config = _small_training_config(tmp_path)
    trainer = Trainer(config)
    X = np.arange(64, dtype=float).reshape(32, 2)
    y = np.array([0, 7] * 16)
    trainer.train_pre_split(X[:20], y[:20], X[20:26], y[20:26], X[26:], y[26:])
    metadata = joblib.load(tmp_path / "model_meta.joblib")
    assert metadata["active_global_class_ids"] == [0, 7]
    assert metadata["class_names"] == ["planet", "unknown"]
    assert metadata["artifact_version"] == 2


def test_grouped_split_never_crosses_host_boundaries():
    frame = pd.DataFrame(
        {
            "tic_id": np.repeat([f"TIC-{index}" for index in range(30)], 2),
            "label": np.repeat([0, 6] * 15, 2),
            "is_augmented": np.tile([False, True], 30),
        }
    )
    split = DatasetSplitter(
        {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}, 7
    ).split(frame)
    groups = {name: set(part["tic_id"]) for name, part in split.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert not split["val"]["is_augmented"].any()
    assert not split["test"]["is_augmented"].any()


def test_sparse_sklearn_class_columns_are_aligned():
    config = load_config("configs/config.yaml")
    config = copy.deepcopy(config)
    config["model"]["type"] = "random_forest"
    evaluator = Evaluator(config)
    probabilities = evaluator.predict_proba(_SparseClassModel(), np.zeros((2, 3)))
    assert probabilities.shape == (2, 8)
    assert np.allclose(probabilities[:, 0], 0.25)
    assert np.allclose(probabilities[:, 7], 0.75)
    assert np.allclose(probabilities[:, 1:7], 0.0)


def test_isotonic_refuses_under_supported_calibration():
    probabilities = np.tile(np.array([[0.8, 0.2]]), (12, 1))
    labels = np.array([0] * 10 + [1] * 2)
    with pytest.raises(ValueError, match="too flexible"):
        ProbabilityCalibrator("isotonic", min_samples_per_class=5).fit(
            probabilities, labels
        )


def test_unknown_prediction_does_not_open_transit_gate():
    config = copy.deepcopy(load_config("configs/config.yaml"))
    classifier = AstrophysicalClassifier(config)
    probabilities = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93])
    result = classifier.classify_from_proba(probabilities)
    assert result.class_name == "unknown"
    assert not result.transit_like
    assert result.transit_gate_prob == pytest.approx(0.02)


def test_config_validation_rejects_invalid_scientific_contract():
    config = copy.deepcopy(load_config("configs/config.yaml"))
    config["dataset"]["split_ratios"]["test"] = 0.5
    config["bls"]["period_min"] = -1.0
    with pytest.raises(ValueError, match="split_ratios"):
        validate_config(config)


def test_training_support_counts_independent_hosts_not_augmented_rows():
    config = copy.deepcopy(load_config("configs/config.yaml"))
    config["training"]["minimum_samples_per_class"] = 2
    frame = pd.DataFrame(
        {
            "label": [0, 0, 0, 6, 6],
            "tic_id": ["A", "A", "A", "B", "C"],
            "is_augmented": [False, True, True, False, False],
        }
    )
    with pytest.raises(RuntimeError, match="independent host"):
        _validate_class_support(frame, config, smoke_test=False)


def test_evaluator_does_not_invent_positive_class():
    config = copy.deepcopy(load_config("configs/config.yaml"))
    config["classes"]["astro_names"] = ["eclipsing_binary", "variable_star"]
    config["model"]["type"] = "random_forest"
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(
        _BinaryModel(), np.zeros((2, 3)), np.array([0, 1])
    )
    assert metrics["positive_class_indices"] == []
    assert np.isnan(metrics["pr_auc"])
