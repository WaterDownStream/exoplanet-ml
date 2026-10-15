"""Download one or more named TESS targets through the shared downloader."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import DownloadConfig, TESSDownloader
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download specific TESS light curves")
    parser.add_argument("targets", nargs="*", help="Target names, e.g. TOI-700")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--class-name", default="planet")
    parser.add_argument("--out-subdir", default="planets")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    targets = args.targets or ["TOI-700", "TOI-270", "Pi Mensae"]

    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / args.out_subdir,
            source_catalog="manual",
            target_class=args.class_name,
            retry_count=int(config.get("dataset", {}).get("retry_count", 3)),
            overwrite=args.overwrite,
        )
    )
    downloader.download(targets)
    paths = downloader.write_logs(config["paths"].get("logs_dir", "data/logs"))
    logger.info("Download logs: %s", paths)


if __name__ == "__main__":
    main()
