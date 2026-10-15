"""Strict validation gate for exported feature datasets and split integrity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_extraction import FEATURE_NAMES, feature_mask_name
from src.utils import get_logger, load_config


def validate_dataset(
    dataframe: pd.DataFrame,
    config: dict,
    source: Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(dataframe) == 0:
        return ["dataset has no rows"], warnings
    if "label" not in dataframe:
        errors.append("missing label column")
    else:
        numeric_labels = pd.to_numeric(dataframe["label"], errors="coerce")
        n_classes = len(config.get("classes", {}).get("astro_names", []))
        if (
            numeric_labels.isna().any()
            or (numeric_labels < 0).any()
            or (numeric_labels % 1 != 0).any()
        ):
            errors.append("labels must be non-negative integers")
        elif n_classes and (numeric_labels >= n_classes).any():
            errors.append(f"labels must be smaller than configured class count {n_classes}")
        class_count = numeric_labels.value_counts()
        if len(class_count) < 2:
            errors.append("fewer than two observed classes")
        minimum = int(config.get("training", {}).get("minimum_samples_per_class", 20))
        support = _independent_host_support(dataframe)
        class_names = list(config.get("classes", {}).get("astro_names", []))
        excluded_names = set(config.get("training", {}).get("exclude_classes", []))
        excluded_ids = {
            class_names.index(name) for name in excluded_names if name in class_names
        }
        supervised = support.loc[~support.index.astype(int).isin(excluded_ids)]
        sparse = supervised[supervised < minimum]
        if len(sparse):
            errors.append(
                f"supervised classes below {minimum} independent hosts: {sparse.to_dict()}"
            )

    missing_features = [name for name in FEATURE_NAMES if name not in dataframe]
    missing_masks = [
        feature_mask_name(name)
        for name in FEATURE_NAMES
        if feature_mask_name(name) not in dataframe
    ]
    if missing_features:
        errors.append(f"missing feature columns: {missing_features}")
    if config.get("features", {}).get("use_masks", True) and missing_masks:
        errors.append(f"missing feature-mask columns: {missing_masks}")

    for name in FEATURE_NAMES:
        if name not in dataframe:
            continue
        values = pd.to_numeric(dataframe[name], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isinf(values)):
            errors.append(f"feature {name} contains infinity")
        if np.all(~np.isfinite(values)):
            warnings.append(f"feature {name} is unavailable in every row")
        mask_name = feature_mask_name(name)
        if mask_name not in dataframe:
            continue
        masks = pd.to_numeric(dataframe[mask_name], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isin(masks[np.isfinite(masks)], [0.0, 1.0])):
            errors.append(f"mask {mask_name} contains values other than 0/1")
        mismatch = np.isfinite(values) != (masks == 1.0)
        if np.any(mismatch):
            errors.append(
                f"mask {mask_name} disagrees with feature availability in "
                f"{int(np.count_nonzero(mismatch))} rows"
            )

    expected_version = str(config.get("dataset", {}).get("version", ""))
    if "processing_version" in dataframe:
        versions = set(dataframe["processing_version"].dropna().astype(str))
        if versions and versions != {expected_version}:
            errors.append(
                f"processing versions {sorted(versions)} do not match {expected_version}"
            )
    else:
        warnings.append("processing_version column is missing")

    if "cache_id" in dataframe:
        cache_ids = dataframe["cache_id"].fillna("").astype(str)
        nonempty = cache_ids[cache_ids != ""]
        duplicated = nonempty.duplicated(keep=False)
        if duplicated.any():
            errors.append(
                f"duplicate cache_id values: {nonempty[duplicated].unique()[:10].tolist()}"
            )
    else:
        warnings.append("cache_id column is missing; hybrid training is unavailable")

    group_column = "tic_id" if "tic_id" in dataframe else "filename" if "filename" in dataframe else None
    if group_column and "label" in dataframe:
        conflicts = dataframe.groupby(group_column)["label"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            errors.append(
                f"host groups with conflicting labels: {conflicts.index.astype(str)[:10].tolist()}"
            )
    elif group_column is None:
        errors.append("neither tic_id nor filename is available for leakage control")

    if source.name in ("val.csv", "test.csv") and "is_augmented" in dataframe:
        if dataframe["is_augmented"].fillna(False).astype(bool).any():
            errors.append(f"{source.name} contains augmented samples")
    return errors, warnings


def _independent_host_support(dataframe: pd.DataFrame) -> pd.Series:
    base = dataframe
    if "is_augmented" in base:
        base = base.loc[~base["is_augmented"].fillna(False).astype(bool)]
    if "tic_id" in base:
        hosts = base["tic_id"].fillna("").astype(str).str.strip()
        if "filename" in base:
            hosts = hosts.mask(hosts == "", base["filename"].fillna("").astype(str))
    elif "filename" in base:
        hosts = base["filename"].fillna("").astype(str)
    else:
        return base["label"].value_counts().sort_index()
    return base.assign(_host=hosts).groupby("label")["_host"].nunique().sort_index()


def _split_leakage_errors(directory: Path) -> list[str]:
    paths = {name: directory / f"{name}.csv" for name in ("train", "val", "test")}
    if not all(path.exists() for path in paths.values()):
        return []
    groups = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, usecols=lambda column: column in ("tic_id", "filename"))
        column = "tic_id" if "tic_id" in frame else "filename"
        values = frame[column].fillna("").astype(str).str.strip()
        if column == "tic_id" and "filename" in frame:
            values = values.mask(values == "", frame["filename"].fillna("").astype(str))
        groups[name] = set(values)
    errors = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            errors.append(
                f"host leakage between {left} and {right}: {sorted(overlap)[:10]}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset schema and splits")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--csv", default="data/processed/features/master_dataset.csv")
    args = parser.parse_args()

    logger = get_logger()
    config = load_config(args.config)
    source = Path(args.csv)
    dataframe = pd.read_csv(source)
    errors, warnings = validate_dataset(dataframe, config, source)
    if source.name == "master_dataset.csv":
        errors.extend(_split_leakage_errors(source.parent))

    logger.info("Rows: %d", len(dataframe))
    logger.info("Columns: %d", len(dataframe.columns))
    if "label" in dataframe:
        logger.info("Class distribution: %s", dataframe["label"].value_counts().to_dict())
    for warning in warnings:
        logger.warning("Dataset warning: %s", warning)
    for error in errors:
        logger.error("Dataset error: %s", error)
    if errors:
        raise SystemExit(f"Dataset validation failed with {len(errors)} error(s)")
    logger.info("Dataset validation passed with %d warning(s)", len(warnings))


if __name__ == "__main__":
    main()
