"""Smoke-test downloader for a single well-known TESS target."""
from __future__ import annotations

from pathlib import Path

from src.datasets import DownloadConfig, TESSDownloader
from src.utils import get_logger, load_config, set_seed


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / "planets",
            source_catalog="smoke_test",
            target_class="planet",
            retry_count=1,
        )
    )
    downloader.download(["TOI-700"], limit=1)
    logger.info("Download logs: %s", downloader.write_logs(config["paths"].get("logs_dir", "data/logs")))


if __name__ == "__main__":
    main()
