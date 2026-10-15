"""Download a small sample of well-known TESS planet hosts."""
from __future__ import annotations

from pathlib import Path

from src.datasets import DownloadConfig, TESSDownloader
from src.utils import get_logger, load_config, set_seed


TARGETS = [
    "TOI-700",
    "TOI-270",
    "Pi Mensae",
    "WASP-18",
    "HD 219134",
    "LHS 3844",
    "GJ 357",
    "TOI-1231",
    "AU Microscopii",
    "55 Cancri",
]


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / "planets",
            source_catalog="manual_planet_sample",
            target_class="planet",
            retry_count=int(config.get("dataset", {}).get("retry_count", 3)),
        )
    )
    downloader.download(TARGETS)
    logger.info("Download logs: %s", downloader.write_logs(config["paths"].get("logs_dir", "data/logs")))


if __name__ == "__main__":
    main()
