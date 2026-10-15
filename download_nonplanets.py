"""Download a small deterministic set of quiet/unknown TIC light curves."""
from __future__ import annotations

from pathlib import Path

from src.datasets import DownloadConfig, TESSDownloader
from src.utils import get_logger, load_config, set_seed


TARGETS = [
    "TIC 100100827",
    "TIC 198456033",
    "TIC 271971130",
    "TIC 310651245",
    "TIC 441462736",
]


def main() -> None:
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    downloader = TESSDownloader(
        DownloadConfig(
            out_dir=Path(config["paths"]["raw_dir"]) / "nonplanets",
            source_catalog="manual_quiet_tic",
            target_class="unknown",
            retry_count=int(config.get("dataset", {}).get("retry_count", 3)),
        )
    )
    downloader.download(TARGETS)
    logger.info("Download logs: %s", downloader.write_logs(config["paths"].get("logs_dir", "data/logs")))


if __name__ == "__main__":
    main()
