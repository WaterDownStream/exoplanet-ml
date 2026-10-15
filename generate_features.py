"""Compatibility wrapper for feature extraction through the dataset builder."""
from __future__ import annotations

from pathlib import Path

from src.datasets import DatasetBuilder
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))

    builder = DatasetBuilder(config)
    df = builder.build_from_raw()
    if not len(df):
        logger.warning("No feature rows were generated")
        return

    builder.export(df)
    output = Path("data/processed/features_real.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved %d feature rows to %s", len(df), output)


if __name__ == "__main__":
    main()
