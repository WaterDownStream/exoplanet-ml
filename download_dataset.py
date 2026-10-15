"""Download confirmed-planet TESS light curves with shared dataset tooling."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import DownloadConfig, TESSDownloader, targets_from_catalog
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download confirmed TESS planet hosts")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--catalog", default="data/catalogs/confirmed_planets.csv")
    parser.add_argument("--column", default="hostname")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    ds_cfg = config.get("dataset", {})

    targets = targets_from_catalog(args.catalog, args.column, config["seed"])
    if args.limit is None:
        args.limit = int(ds_cfg.get("balance_targets", {}).get("confirmed_planet", 1000))

    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / "planets",
            source_catalog=str(args.catalog),
            target_class="planet",
            retry_count=int(ds_cfg.get("retry_count", 3)),
            overwrite=args.overwrite,
        )
    )
    downloader.download(targets, limit=args.limit)
    paths = downloader.write_logs(config["paths"].get("logs_dir", "data/logs"))
    logger.info("Download logs: %s", paths)


if __name__ == "__main__":
    main()
