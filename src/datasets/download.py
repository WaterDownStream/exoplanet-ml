"""Shared resumable TESS download helpers."""
from __future__ import annotations

import hashlib
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils import ensure_dirs, get_logger

logger = get_logger()


@dataclass
class DownloadConfig:
    """Configuration for resumable light-curve downloads."""

    out_dir: str | Path
    source_catalog: str
    target_class: str
    retry_count: int = 3
    overwrite: bool = False
    mission: str = "TESS"
    cadence: str = "any"


@dataclass
class DownloadRecord:
    """One download attempt for durable CSV logs."""

    hostname: str
    tic_id: str
    sector: str
    cadence: str
    filename: str
    download_time: str
    source_catalog: str
    target_class: str
    file_size: int
    checksum: str
    processing_status: str
    reason: str = ""


class TESSDownloader:
    """Download TESS light curves through Lightkurve with uniform logging."""

    def __init__(self, cfg: DownloadConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        ensure_dirs(self.out_dir)
        self.records: list[DownloadRecord] = []

    def download(self, targets: list[str], limit: int | None = None) -> list[DownloadRecord]:
        """Download targets, continuing after per-target failures."""
        count = 0
        for target in targets:
            if limit is not None and count >= limit:
                break
            record = self.download_target(target)
            self.records.append(record)
            if record.processing_status == "downloaded":
                count += 1
        return self.records

    def download_target(self, target: str) -> DownloadRecord:
        """Download one target with retries and corrupt-file cleanup."""
        outfile = self.out_dir / f"{_safe_name(target)}.fits"
        if outfile.exists() and not self.cfg.overwrite:
            return self._record(target, outfile, "skipped_existing")

        last_reason = ""
        for attempt in range(1, self.cfg.retry_count + 1):
            try:
                path = self._download_once(target, outfile)
                self.validate(path)
                return self._record(target, path, "downloaded")
            except Exception as exc:  # noqa: BLE001
                last_reason = str(exc)
                logger.warning(
                    "Download failed for %s attempt %d/%d: %s",
                    target,
                    attempt,
                    self.cfg.retry_count,
                    exc,
                )
                if outfile.exists():
                    outfile.unlink()
                time.sleep(min(attempt, 5))
        return self._record(target, outfile, "failed", last_reason)

    def _download_once(self, target: str, outfile: Path) -> Path:
        import lightkurve as lk

        logger.info("Searching %s", target)
        search = lk.search_lightcurve(target, mission=self.cfg.mission)
        if len(search) == 0:
            raise RuntimeError("No TESS light-curve products found")
        selected = _select_preferred_product(search)
        with tempfile.TemporaryDirectory(
            prefix="mast_", dir=self.out_dir
        ) as temporary_directory:
            light_curve = selected.download(
                quality_bitmask="none",
                download_dir=temporary_directory,
                flux_column="pdcsap_flux",
            )
            if light_curve is None:
                raise RuntimeError("Lightkurve returned no light curve")
            row = selected.table[0]
            original = (
                Path(temporary_directory)
                / "mastDownload"
                / str(row["obs_collection"])
                / str(row["obs_id"])
                / str(row["productFilename"])
            )
            if not original.exists():
                raise RuntimeError("Downloaded original MAST FITS path was not found")
            shutil.copy2(original, outfile)
        return outfile

    @staticmethod
    def validate(path: Path) -> None:
        """Reject empty/corrupt FITS files and files missing time or flux columns."""
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("empty FITS file")
        from astropy.io import fits

        with fits.open(path) as hdul:
            if len(hdul) < 2:
                raise RuntimeError("missing FITS table extension")
            names = {name.upper() for name in hdul[1].data.columns.names}
            if "TIME" not in names:
                raise RuntimeError("missing TIME column")
            if not names.intersection({"PDCSAP_FLUX", "SAP_FLUX", "FLUX"}):
                raise RuntimeError("missing flux column")

    def write_logs(self, log_dir: str | Path = "data/logs") -> dict[str, Path]:
        """Write download and failure logs."""
        log_dir = Path(log_dir)
        ensure_dirs(log_dir)
        df = pd.DataFrame([record.__dict__ for record in self.records])
        paths: dict[str, Path] = {}
        download_log = log_dir / "download_log.csv"
        df.to_csv(download_log, index=False)
        paths["download_log"] = download_log
        failed = df[df["processing_status"] == "failed"] if len(df) else df
        failed_log = log_dir / "failed_downloads.csv"
        failed.to_csv(failed_log, index=False)
        paths["failed_downloads"] = failed_log
        return paths

    def _record(
        self,
        target: str,
        path: Path,
        status: str,
        reason: str = "",
    ) -> DownloadRecord:
        return DownloadRecord(
            hostname=socket.gethostname(),
            tic_id=target,
            sector="",
            cadence=self.cfg.cadence,
            filename=path.name,
            download_time=datetime.now(timezone.utc).isoformat(),
            source_catalog=self.cfg.source_catalog,
            target_class=self.cfg.target_class,
            file_size=path.stat().st_size if path.exists() else 0,
            checksum=_checksum(path) if path.exists() else "",
            processing_status=status,
            reason=reason,
        )


def targets_from_catalog(
    catalog_path: str | Path,
    column: str = "hostname",
    seed: int = 42,
) -> list[str]:
    """Load and deterministically shuffle target names from a CSV catalog."""
    df = pd.read_csv(catalog_path, comment="#")
    targets = df[column].dropna().astype(str).drop_duplicates().tolist()
    return pd.Series(targets).sample(frac=1, random_state=seed).tolist()


def _safe_name(target: str) -> str:
    return target.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _select_preferred_product(search):
    """Select a deterministic, highest-cadence pipeline product."""
    table = search.table
    author_priority = {
        "SPOC": 0,
        "TESS-SPOC": 1,
        "QLP": 2,
        "TASOC": 3,
        "PATHOS": 4,
        "CDIPS": 5,
    }
    candidates = []
    for index, row in enumerate(table):
        author = str(row["author"]) if "author" in table.colnames else ""
        try:
            exposure = float(row["exptime"])
        except (KeyError, TypeError, ValueError):
            exposure = float("inf")
        try:
            sequence = int(row["sequence_number"])
        except (KeyError, TypeError, ValueError):
            sequence = 10**9
        candidates.append(
            (author_priority.get(author.upper(), 99), exposure, sequence, index)
        )
    selected_index = min(candidates)[-1]
    selected = search[selected_index : selected_index + 1]
    row = selected.table[0]
    logger.info(
        "Selected TESS product author=%s sector=%s cadence=%ss",
        row["author"] if "author" in selected.table.colnames else "unknown",
        row["sequence_number"]
        if "sequence_number" in selected.table.colnames
        else "unknown",
        row["exptime"] if "exptime" in selected.table.colnames else "unknown",
    )
    return selected
