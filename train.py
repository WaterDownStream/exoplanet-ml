"""End-to-end training entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets import DatasetBuilder
from src.datasets.sampler import ClassSampler
from src.datasets.cache import FeatureCache
from src.evaluator import Evaluator
from src.feature_extraction import FEATURE_NAMES, dataframe_to_model_matrix
from src.trainer import Trainer
from src.utils import ensure_dirs, get_logger, load_config, set_seed
from src.visualizer import Visualizer


def _synthetic(n: int, seed: int, n_classes: int = 7) -> pd.DataFrame:
    """Generate a labeled synthetic feature table for smoke-testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        label = int(rng.integers(0, n_classes))
        planet_like = label in (0, 1)
        depth = abs(rng.normal(0.005 if planet_like else 0.0005, 0.002))
        rows.append({k: abs(rng.normal(0.01, 0.005)) for k in FEATURE_NAMES})
        rows[-1].update({
            "transit_depth": depth,
            "transit_snr": abs(rng.normal(12 if planet_like else 3, 2)),
            "transit_power": abs(rng.normal(0.8 if planet_like else 0.3, 0.1)),
            "label": label,
        })
    return pd.DataFrame(rows)


def _load_training_dataframe(
    config: dict,
    features_arg: str | None,
    logger,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """Load an existing feature table or build one through the dataset subsystem."""
    candidates: list[Path] = []
    if features_arg:
        candidates.append(Path(features_arg))
    else:
        features_dir = Path(config["paths"].get("features_dir", "data/processed/features"))
        candidates.append(features_dir / "master_dataset.csv")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            df = pd.read_csv(candidate)
            logger.info("Loaded features from %s (%d rows)", candidate, len(df))
            df = _remap_legacy_binary_labels(df, config, candidate, logger)
            return df

    logger.info("No feature table found; building dataset from raw light curves")
    builder = DatasetBuilder(config)
    df = builder.build_from_raw()
    if len(df):
        builder.export(df)
        return df

    if allow_synthetic:
        n_classes = len(config.get("classes", {}).get("astro_names", [])) or 2
        n_samples = int(config.get("training", {}).get("synthetic_samples", 2000))
        logger.warning("Using explicitly requested synthetic smoke-test dataset")
        return _synthetic(n_samples, config["seed"], n_classes)
    raise RuntimeError(
        "No real feature rows are available. Build the dataset first, or pass "
        "--smoke-test for a non-scientific software check."
    )


def _remap_legacy_binary_labels(
    df: pd.DataFrame,
    config: dict,
    source: Path,
    logger,
) -> pd.DataFrame:
    """Map old 0/1 binary tables into the configured astro class space."""
    if "label_name" in df.columns:
        return df
    labels = set(pd.Series(df.get("label", [])).dropna().astype(int).unique())
    astro_names = config.get("classes", {}).get("astro_names", [])
    if labels <= {0, 1} and len(astro_names) > 2 and "planet" in astro_names and "unknown" in astro_names:
        out = df.copy()
        planet = astro_names.index("planet")
        unknown = astro_names.index("unknown")
        out["label"] = out["label"].map({1: planet, 0: unknown}).astype(int)
        logger.warning("Remapped legacy binary labels from %s into astro class indices", source)
        return out
    return df


def _load_cached_flux(df: pd.DataFrame, config: dict, logger) -> np.ndarray | None:
    if "cache_id" not in df.columns:
        return None
    cache = FeatureCache(
        config["paths"].get("cache_dir", "data/cache"),
        config.get("dataset", {}).get("version", "1.0.0"),
    )
    arrays = []
    for cache_id in df["cache_id"].fillna(""):
        if not cache_id:
            return None
        try:
            arrays.append(cache.load_flux(str(cache_id)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load cached flux %s: %s", cache_id, exc)
            return None
    return np.asarray(arrays, dtype=float)


def _validate_feature_version(
    df: pd.DataFrame, config: dict, smoke_test: bool
) -> None:
    if smoke_test or "processing_version" not in df.columns:
        return
    expected = str(config.get("dataset", {}).get("version", ""))
    observed = set(df["processing_version"].dropna().astype(str))
    if observed and observed != {expected}:
        raise RuntimeError(
            f"Feature table versions {sorted(observed)} do not match configured "
            f"version {expected}; rebuild the dataset"
        )


def _validate_class_support(
    df: pd.DataFrame, config: dict, smoke_test: bool
) -> None:
    if "label" not in df or len(df) == 0:
        raise ValueError("Training data require non-empty integer labels")
    base = df
    if "is_augmented" in base.columns:
        base = base.loc[~base["is_augmented"].fillna(False).astype(bool)]
    if "tic_id" in base.columns or "filename" in base.columns:
        if "tic_id" in base.columns:
            hosts = base["tic_id"].fillna("").astype(str).str.strip()
            if "filename" in base.columns:
                hosts = hosts.mask(hosts == "", base["filename"].fillna("").astype(str))
        else:
            hosts = base["filename"].fillna("").astype(str)
        support = base.assign(_host=hosts).groupby("label")["_host"].nunique()
    else:
        support = base["label"].value_counts()
    counts = support.sort_index()
    if len(counts) < 2:
        raise ValueError("Training requires at least two observed classes")
    minimum = int(config.get("training", {}).get("minimum_samples_per_class", 20))
    under_supported = counts[counts < minimum]
    if len(under_supported) and not smoke_test:
        raise RuntimeError(
            "Scientifically reliable training was refused because classes are under-supported: "
            f"{under_supported.to_dict()}. Collect at least {minimum} independent host "
            "stars per active class."
        )


def _filter_supervised_rows(
    df: pd.DataFrame, config: dict, logger
) -> pd.DataFrame:
    class_names = list(config.get("classes", {}).get("astro_names", []))
    excluded_names = set(
        config.get("training", {}).get("exclude_classes", ["unknown"])
    )
    excluded_ids = {
        class_names.index(name) for name in excluded_names if name in class_names
    }
    if not excluded_ids:
        return df
    selected = ~df["label"].astype(int).isin(excluded_ids)
    removed = int(np.count_nonzero(~selected))
    if removed:
        logger.warning(
            "Excluded %d non-supervised rows with labels %s",
            removed,
            sorted(excluded_names),
        )
    return df.loc[selected].copy()


def _load_exported_splits(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    directory = Path(
        config["paths"].get("features_dir", "data/processed/features")
    )
    paths = [directory / f"{name}.csv" for name in ("train", "val", "test")]
    if not all(path.exists() for path in paths):
        return None
    frames = [pd.read_csv(path) for path in paths]
    if not all(len(frame) for frame in frames):
        raise RuntimeError("Exported train/validation/test splits must all be non-empty")
    for index in (1, 2):
        frame = frames[index]
        if "is_augmented" in frame.columns:
            augmented = frame["is_augmented"].fillna(False).astype(bool)
            frames[index] = frame.loc[~augmented].copy()
    _assert_disjoint_groups(*frames)
    return frames[0], frames[1], frames[2]


def _assert_disjoint_groups(*frames: pd.DataFrame) -> None:
    group_sets = []
    for frame in frames:
        if "tic_id" in frame.columns:
            groups = frame["tic_id"].fillna("").astype(str).str.strip()
            if "filename" in frame.columns:
                groups = groups.mask(
                    groups == "", frame["filename"].fillna("").astype(str)
                )
        elif "filename" in frame.columns:
            groups = frame["filename"].fillna("").astype(str)
        else:
            raise ValueError("Split tables require tic_id or filename for leakage checks")
        group_sets.append(set(groups))
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Host-star leakage detected across exported dataset splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exoplanet classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="allow synthetic data for a non-scientific pipeline smoke test",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger(config.get("log_level", "INFO"))
    set_seed(config["seed"])
    ensure_dirs(*config["paths"].values())

    df = _load_training_dataframe(
        config, args.features, logger, allow_synthetic=args.smoke_test
    )
    _validate_feature_version(df, config, args.smoke_test)
    if not args.smoke_test:
        df = _filter_supervised_rows(df, config, logger)
    _validate_class_support(df, config, args.smoke_test)
    use_masks = config.get("features", {}).get("use_masks", True)
    input_names = dataframe_to_model_matrix(df.iloc[:0], use_masks=use_masks)[1]
    trainer = Trainer(config)

    split_frames = None if args.features or args.smoke_test else _load_exported_splits(config)
    if split_frames is not None:
        train_df, val_df, test_df = split_frames
        train_df = _filter_supervised_rows(train_df, config, logger)
        val_df = _filter_supervised_rows(val_df, config, logger)
        test_df = _filter_supervised_rows(test_df, config, logger)
        train_df = ClassSampler(
            config.get("training", {}).get("class_balancing", "weighted_loss"),
            config["seed"],
        ).balance(train_df)
        X_train, _ = dataframe_to_model_matrix(train_df, use_masks=use_masks)
        X_val, _ = dataframe_to_model_matrix(val_df, use_masks=use_masks)
        X_test, _ = dataframe_to_model_matrix(test_df, use_masks=use_masks)
        y_train = train_df["label"].to_numpy(dtype=int)
        y_val = val_df["label"].to_numpy(dtype=int)
        y_test_raw = test_df["label"].to_numpy(dtype=int)
        flux_train = flux_val = flux_test_input = None
        if config["model"]["type"] == "hybrid_tcn":
            flux_train = _load_cached_flux(train_df, config, logger)
            flux_val = _load_cached_flux(val_df, config, logger)
            flux_test_input = _load_cached_flux(test_df, config, logger)
            if any(value is None for value in (flux_train, flux_val, flux_test_input)):
                raise RuntimeError(
                    "Hybrid TCN was requested but one or more split rows lack real cached flux"
                )
        result = trainer.train_pre_split(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test_raw,
            flux_train,
            flux_val,
            flux_test_input,
        )
    else:
        X, _ = dataframe_to_model_matrix(df, use_masks=use_masks)
        y = df["label"].to_numpy(dtype=int)
        flux = None
        if config["model"]["type"] == "hybrid_tcn":
            flux = _load_cached_flux(df, config, logger)
            if flux is None:
                raise RuntimeError("Hybrid TCN requires real cached flux for every row")
        groups = (
            df["tic_id"].fillna(df.get("filename", "")).astype(str).to_numpy()
            if "tic_id" in df
            else df.get("filename", pd.Series(df.index.astype(str))).astype(str).to_numpy()
        )
        result = trainer.train(X, y, flux, groups=groups)
    model = result[0]
    X_test, y_test = result[1], result[2]
    flux_test = result[3] if len(result) > 3 else None

    eval_config = dict(config)
    eval_config["classes"] = dict(config.get("classes", {}))
    eval_config["classes"]["astro_names"] = trainer.class_names
    eval_config["inference"] = dict(config.get("inference", {}))
    eval_config["inference"]["planet_threshold"] = trainer.planet_threshold
    evaluator = Evaluator(eval_config)
    metrics = evaluator.evaluate(
        model,
        X_test,
        y_test,
        flux_test,
        calibrator=trainer.calibrator,
    )
    metrics_path = Path(config["paths"]["models_dir"]) / "evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ("proba", "raw_proba")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    viz = Visualizer()
    positive_indices = metrics["positive_class_indices"]
    y_positive = np.isin(y_test, positive_indices).astype(int)
    if len(np.unique(y_positive)) == 2:
        positive_probability = np.sum(metrics["proba"][:, positive_indices], axis=1)
        viz.roc(y_positive, positive_probability)
        viz.pr(y_positive, positive_probability)
    labels = trainer.class_names
    viz.confusion(np.array(metrics["confusion_matrix"]), labels=labels)
    viz.reliability(
        y_test,
        metrics["proba"],
        n_bins=config.get("evaluation", {}).get("calibration_bins", 10),
    )
    viz.confidence_histogram(metrics["proba"])

    imp = evaluator.shap_importance(model, X_test, input_names)
    if imp is None:
        imp = evaluator.permutation_importance(
            model, X_test, y_test, input_names, flux_test
        )
    viz.importance(imp[0], imp[1])

    f1_key = "macro_f1" if "macro_f1" in metrics else "f1"
    logger.info("Training complete. Test %s=%.3f", f1_key, metrics.get(f1_key, metrics["accuracy"]))


if __name__ == "__main__":
    main()
