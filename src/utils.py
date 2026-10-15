"""Shared utilities: config loading, logging, seeding, IO helpers."""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # optional until load_config is called
    yaml = None

_LOGGER_NAME = "exoplanet_ml"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration file.

    Parameters
    ----------
    path : str | Path
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Nested configuration dictionary.

    References
    ----------
    PyYAML safe_load documentation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML config files. Install pyyaml.")
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return validate_config(config)


def validate_config(config: Any) -> dict[str, Any]:
    """Validate cross-module scientific and data-contract invariants."""
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    errors: list[str] = []
    required = ("paths", "classes", "preprocessing", "detrending", "bls", "features", "dataset", "model")
    for section in required:
        if not isinstance(config.get(section), dict):
            errors.append(f"{section} must be a mapping")

    classes = config.get("classes", {})
    names = classes.get("astro_names", [])
    if not isinstance(names, list) or len(names) < 2 or not all(isinstance(name, str) and name for name in names):
        errors.append("classes.astro_names must contain at least two non-empty names")
        names = []
    elif len(names) != len(set(names)):
        errors.append("classes.astro_names must be unique")
    legacy_names = classes.get("names")
    if legacy_names is not None and legacy_names != names:
        errors.append("classes.names and classes.astro_names must use the same ordered schema")

    aliases = config.get("dataset", {}).get("label_aliases", {})
    if not isinstance(aliases, dict):
        errors.append("dataset.label_aliases must be a mapping")
    elif names:
        invalid = sorted({str(value) for value in aliases.values() if value not in names})
        if invalid:
            errors.append(f"dataset.label_aliases references unknown classes: {invalid}")

    ratios = config.get("dataset", {}).get("split_ratios", {})
    try:
        split_values = [float(ratios[name]) for name in ("train", "val", "test")]
        if any(value <= 0.0 for value in split_values) or not np.isclose(sum(split_values), 1.0):
            errors.append("dataset.split_ratios must be positive and sum to 1")
    except (KeyError, TypeError, ValueError):
        errors.append("dataset.split_ratios must define numeric train, val, and test values")

    bls = config.get("bls", {})
    _ordered_positive(bls, "period_min", "period_max", "bls", errors)
    _ordered_positive(bls, "duration_min", "duration_max", "bls", errors)
    if int(bls.get("n_periods", 0)) < 100:
        errors.append("bls.n_periods must be at least 100")
    if int(bls.get("n_durations", 0)) < 1:
        errors.append("bls.n_durations must be positive")
    max_fraction = float(bls.get("max_duration_fraction", 0.2))
    if not 0.0 < max_fraction < 1.0:
        errors.append("bls.max_duration_fraction must be between 0 and 1")

    mono = config.get("mono_search", {})
    if mono:
        _ordered_positive(mono, "min_depth", "max_depth", "mono_search", errors)
        _ordered_positive(mono, "min_duration", "max_duration", "mono_search", errors)

    model_type = config.get("model", {}).get("type")
    if model_type not in {"random_forest", "xgboost", "neural_net", "hybrid_tcn"}:
        errors.append("model.type is not a supported model family")
    normalization = config.get("features", {}).get("normalization", "robust")
    if normalization not in {"robust", "standard", "minmax", "none"}:
        errors.append("features.normalization must be robust, standard, minmax, or none")

    excluded = config.get("training", {}).get("exclude_classes", [])
    if names and any(name not in names for name in excluded):
        errors.append("training.exclude_classes contains an unknown class")
    for section, key in (
        ("inference", "planet_threshold"),
        ("inference", "abstain_probability_threshold"),
        ("inference", "abstain_entropy_threshold"),
        ("parameter_estimation", "planet_threshold"),
    ):
        value = float(config.get(section, {}).get(key, 0.5))
        if not 0.0 <= value <= 1.0:
            errors.append(f"{section}.{key} must be between 0 and 1")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(dict.fromkeys(errors)))
    return config


def _ordered_positive(
    section: dict[str, Any], low_key: str, high_key: str, prefix: str, errors: list[str]
) -> None:
    try:
        low, high = float(section[low_key]), float(section[high_key])
        if low <= 0.0 or high <= low:
            errors.append(f"{prefix}.{low_key} and {prefix}.{high_key} must satisfy 0 < min < max")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{prefix}.{low_key} and {prefix}.{high_key} must be numeric")


def get_logger(level: str = "INFO") -> logging.Logger:
    """Return a singleton, formatted logger.

    Parameters
    ----------
    level : str
        Logging level name (e.g. ``"INFO"``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility (python, numpy, torch).

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:  # torch optional at util level
        pass


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class LightCurve:
    """Container for a single light curve.

    Attributes
    ----------
    time : np.ndarray
        Timestamps in days (BJD/BTJD).
    flux : np.ndarray
        Normalized or raw flux.
    flux_error : np.ndarray
        Per-point flux uncertainty.
    meta : dict
        Arbitrary metadata (target id, sector, etc.).
    """

    time: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray | None
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        if self.flux_error is None:
            self.flux_error = np.full_like(self.flux, np.nan)
        self.flux_error = np.asarray(self.flux_error, dtype=float)
        if self.time.ndim != 1 or self.flux.ndim != 1 or self.flux_error.ndim != 1:
            raise ValueError("time, flux, and flux_error must be one-dimensional")
        if not (len(self.time) == len(self.flux) == len(self.flux_error)):
            raise ValueError("time, flux, flux_error must share length")
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dictionary")

    def copy(self) -> "LightCurve":
        return LightCurve(
            self.time.copy(), self.flux.copy(),
            self.flux_error.copy(), dict(self.meta),
        )
