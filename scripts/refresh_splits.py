"""Regenerate train/validation/test CSVs from an existing master dataset."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.splits import DatasetSplitter
from src.feature_extraction import FEATURE_NAMES
from src.utils import ensure_dirs, get_logger, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh dataset split CSVs")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--master", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))

    features_dir = Path(config["paths"].get("features_dir", "data/processed/features"))
    master_path = Path(args.master or features_dir / "master_dataset.csv")
    if not master_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_path}")

    df = pd.read_csv(master_path)
    splitter = DatasetSplitter(config.get("dataset", {}), config.get("seed", 42))
    splits = splitter.split(df)
    ensure_dirs(features_dir)
    for name, part in splits.items():
        path = features_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(part))

    metadata = {
        "version": config.get("dataset", {}).get("version", "1.0.0"),
        "created": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "feature_count": len(FEATURE_NAMES),
        "sample_count": len(df),
        "class_distribution": df["label"].value_counts().to_dict() if "label" in df else {},
        "split_distribution": {
            name: part["label"].value_counts().to_dict() if "label" in part else {}
            for name, part in splits.items()
        },
        "config_hash": DatasetSplitter.config_hash(config),
    }
    meta_path = features_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Updated %s", meta_path)


if __name__ == "__main__":
    main()
