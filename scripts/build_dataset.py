"""Build labeled feature dataset from raw FITS directories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exoplanet ML dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    df = build_dataset(args.config)
    logger.info("Dataset build complete: %d samples", len(df))


if __name__ == "__main__":
    main()
