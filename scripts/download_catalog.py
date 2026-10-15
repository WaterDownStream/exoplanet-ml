"""Download TESS light curves from any target catalog CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DownloadConfig, TESSDownloader, targets_from_catalog
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TESS targets from a CSV catalog")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--column", default="target")
    parser.add_argument("--out-subdir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    targets = targets_from_catalog(args.catalog, args.column, config["seed"])
    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / args.out_subdir,
            source_catalog=str(args.catalog),
            target_class=args.class_name,
            retry_count=int(config.get("dataset", {}).get("retry_count", 3)),
            overwrite=args.overwrite,
        )
    )
    downloader.download(targets, limit=args.limit)
    paths = downloader.write_logs(config["paths"].get("logs_dir", "data/logs"))
    logger.info("Downloaded into %s", Path(config["paths"]["raw_dir"]) / args.out_subdir)
    logger.info("Download logs: %s", paths)


if __name__ == "__main__":
    main()
