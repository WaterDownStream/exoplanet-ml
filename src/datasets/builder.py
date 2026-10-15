"""Dataset builder: download, process, label, split, export."""
from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..feature_extraction import FEATURE_NAMES, feature_mask_name
from ..pipeline import Pipeline
from ..utils import ensure_dirs, get_logger, load_config
from .augment import LightCurveAugmenter
from .cache import FeatureCache
from .sampler import ClassSampler
from .splits import DatasetSplitter

logger = get_logger()

DEFAULT_LABEL_ALIASES = {
    "planets": "planet",
    "confirmed_planets": "planet",
    "planet_candidates": "planet_candidate",
    "candidates": "planet_candidate",
    "mono": "planet_candidate",
    "mono_transits": "planet_candidate",
    "eb": "eclipsing_binary",
    "eclipsing_binaries": "eclipsing_binary",
    "background_blend": "background_blend",
    "background_blends": "background_blend",
    "blends": "background_blend",
    "instrument_artefact": "instrument_artefact",
    "instrumental": "instrument_artefact",
    "false_positive": "unknown",
    "false_positives": "unknown",
    "nonplanets": "unknown",
    "random": "unknown",
    "random_stars": "unknown",
    "quiet_star": "quiet_star",
    "quiet_stars": "quiet_star",
    "variable_star": "variable_star",
    "variable_stars": "variable_star",
}


class DatasetBuilder:
    """Build labeled feature datasets from raw FITS directories."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.pipeline = Pipeline(config)
        paths = config["paths"]
        ds_cfg = config.get("dataset", {})
        self.cache = FeatureCache(
            paths.get("cache_dir", "data/cache"),
            ds_cfg.get("version", "1.0.0"),
        )
        self.splitter = DatasetSplitter(ds_cfg, config.get("seed", 42))
        self.augmenter = LightCurveAugmenter(
            config.get("augmentation", {}), config.get("seed", 42)
        )
        self.sampler = ClassSampler(
            config.get("training", {}).get("class_balancing", "weighted_loss"),
            config.get("seed", 42),
        )
        self.class_names = config.get("classes", {}).get(
            "astro_names",
            [
                "planet",
                "planet_candidate",
                "eclipsing_binary",
                "background_blend",
                "instrument_artefact",
                "variable_star",
                "quiet_star",
                "unknown",
            ],
        )
        self.label_aliases = {
            **DEFAULT_LABEL_ALIASES,
            **config.get("dataset", {}).get("label_aliases", {}),
        }
        self.processing_log: list[dict] = []
        self.pipeline_config_hash = DatasetSplitter.config_hash(config)
        ensure_dirs(
            paths["processed_dir"],
            paths.get("features_dir", "data/processed/features"),
        )

    def label_for_directory(self, dirname: str) -> tuple[int, str] | None:
        """Resolve a raw-data directory name to the configured class index."""
        class_name = self.label_aliases.get(dirname.lower())
        if class_name is None:
            return None
        if class_name not in self.class_names:
            logger.warning(
                "Directory %s maps to unknown class %s; skipping",
                dirname,
                class_name,
            )
            return None
        return self.class_names.index(class_name), class_name

    def process_directory(
        self,
        raw_dir: Path,
        label: int,
        label_name: str | None = None,
        source_catalog: str = "local",
    ) -> list[dict]:
        """Process all FITS/CSV files in a directory."""
        rows: list[dict] = []
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.warning("Directory not found: %s", raw_dir)
            return rows

        patterns = ("*.fits", "*.fit", "*.csv", "*.hdf5", "*.h5")
        files: list[Path] = []
        for pat in patterns:
            files.extend(raw_dir.rglob(pat))

        for fpath in sorted(set(files)):
            start = time.perf_counter()
            try:
                cache_id = self._cache_identifier(fpath, label)
                cached_row = self._load_cached_row(fpath, cache_id, label)
                if cached_row is not None:
                    rows.append(cached_row)
                    cached_augmented = self._load_cached_row(
                        fpath, f"{cache_id}#augmented", label
                    )
                    if cached_augmented is not None:
                        rows.append(cached_augmented)
                    self._record_processing(
                        fpath, "cached", start, cached_row.get("quality_score"), None, []
                    )
                    continue
                lc_raw = self.pipeline.loader.load(str(fpath))
                result = self.pipeline.run_lightcurve(lc_raw, source_file=str(fpath))
                if not result.quality.passed:
                    logger.warning("Rejected %s: %s", fpath.name, result.quality.rejection_reason)
                    self._record_processing(
                        fpath,
                        "rejected",
                        start,
                        result.quality.score,
                        result.quality.rejection_reason,
                        result.warnings,
                    )
                    continue
                row = self._row_from_result(
                    fpath, result, label, label_name, source_catalog, cache_id=cache_id
                )
                rows.append(row)
                self.cache.save(
                    cache_id,
                    result.feature_vector,
                    result.feature_masks,
                    {
                        **result.metadata,
                        "source_size": fpath.stat().st_size,
                        "source_mtime_ns": fpath.stat().st_mtime_ns,
                        "pipeline_config_hash": self.pipeline_config_hash,
                        "label": int(label),
                        "dataset_row": row,
                    },
                    FEATURE_NAMES,
                    result.flux_resampled,
                )
                self._record_processing(
                    fpath,
                    "processed",
                    start,
                    result.quality.score,
                    None,
                    result.warnings,
                )
                aug_prob = float(self.config.get("dataset", {}).get("augmentation_probability", 0.0))
                aug_lc = self.augmenter.maybe_augment(
                    lc_raw, aug_prob, key=str(fpath.resolve())
                )
                if aug_lc is not lc_raw:
                    aug_start = time.perf_counter()
                    aug_result = self.pipeline.run_lightcurve(
                        aug_lc,
                        source_file=f"{fpath}#augmented",
                    )
                    if aug_result.quality.passed:
                        aug_cache_id = f"{cache_id}#augmented"
                        aug_row = self._row_from_result(
                            fpath,
                            aug_result,
                            label,
                            label_name,
                            f"{source_catalog}_augmented",
                            augmented=True,
                            cache_id=aug_cache_id,
                        )
                        rows.append(aug_row)
                        self.cache.save(
                            aug_cache_id,
                            aug_result.feature_vector,
                            aug_result.feature_masks,
                            {
                                **aug_result.metadata,
                                "source_size": fpath.stat().st_size,
                                "source_mtime_ns": fpath.stat().st_mtime_ns,
                                "pipeline_config_hash": self.pipeline_config_hash,
                                "label": int(label),
                                "dataset_row": aug_row,
                            },
                            FEATURE_NAMES,
                            aug_result.flux_resampled,
                        )
                    self._record_processing(
                        fpath,
                        "augmented" if aug_result.quality.passed else "augmentation_rejected",
                        aug_start,
                        aug_result.quality.score,
                        aug_result.quality.rejection_reason,
                        aug_result.warnings,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed %s: %s", fpath, exc)
                self._record_processing(
                    fpath,
                    "failed",
                    start,
                    None,
                    str(exc),
                    [],
                )
        return rows

    def _cache_identifier(self, fpath: Path, label: int) -> str:
        raw_root = Path(self.config["paths"]["raw_dir"]).resolve()
        try:
            relative = fpath.resolve().relative_to(raw_root).as_posix()
        except ValueError:
            relative = str(fpath.resolve())
        return f"{relative}|label={int(label)}"

    def _load_cached_row(
        self, fpath: Path, cache_id: str, label: int
    ) -> dict | None:
        if not self.cache.exists(cache_id):
            return None
        try:
            _, _, metadata = self.cache.load(cache_id)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return None
        if not metadata or not isinstance(metadata.get("dataset_row"), dict):
            return None
        matches = (
            metadata.get("pipeline_config_hash") == self.pipeline_config_hash
            and int(metadata.get("source_size", -1)) == fpath.stat().st_size
            and int(metadata.get("source_mtime_ns", -1)) == fpath.stat().st_mtime_ns
            and int(metadata.get("label", -1)) == int(label)
        )
        return dict(metadata["dataset_row"]) if matches else None

    def _row_from_result(
        self,
        fpath: Path,
        result,
        label: int,
        label_name: str | None,
        source_catalog: str,
        augmented: bool = False,
        cache_id: str | None = None,
    ) -> dict:
        feats = result.features
        row = {
            "filename": fpath.name,
            "hostname": socket.gethostname(),
            "tic_id": result.metadata.get("tic_id", fpath.stem),
            "sector": result.metadata.get("sector", ""),
            "cadence": result.quality.metrics.get("cadence", ""),
            "label": label,
            "label_name": label_name or str(label),
            "source_catalog": source_catalog,
            "download_date": result.metadata.get("download_date", ""),
            "quality_score": result.quality.score,
            "number_of_points": result.quality.metrics.get("n_points", ""),
            "median_flux": float(np.median(result.lightcurve.flux)) if len(result.lightcurve.flux) else "",
            "std_flux": float(result.lightcurve.flux.std()) if len(result.lightcurve.flux) else "",
            "processing_version": self.config.get("dataset", {}).get("version", "1.0.0"),
            "is_augmented": bool(augmented),
            "cache_id": cache_id or "",
            **feats,
        }
        for name, mask_value in zip(FEATURE_NAMES, result.feature_masks):
            row[feature_mask_name(name)] = float(mask_value)
        return row

    def _record_processing(
        self,
        path: Path,
        status: str,
        start_time: float,
        quality_score: float | None,
        rejection_reason: str | None,
        warnings: list[str],
    ) -> None:
        self.processing_log.append({
            "filename": path.name,
            "status": status,
            "runtime_s": time.perf_counter() - start_time,
            "warnings": "; ".join(warnings),
            "quality_score": quality_score,
            "rejection_reason": rejection_reason,
            "processing_version": self.config.get("dataset", {}).get("version", "1.0.0"),
        })

    def build_from_raw(self) -> pd.DataFrame:
        """Scan raw_dir subdirectories and build master dataset."""
        raw_root = Path(self.config["paths"]["raw_dir"])
        all_rows: list[dict] = []
        if not raw_root.exists():
            logger.warning("Raw root not found: %s", raw_root)
            return pd.DataFrame(columns=["filename", "label"] + FEATURE_NAMES)

        for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            resolved = self.label_for_directory(d.name)
            if resolved is None:
                logger.info("Skipping unmapped raw directory: %s", d)
                continue
            label, label_name = resolved
            all_rows.extend(self.process_directory(d, label, label_name, d.name))

        if not all_rows:
            logger.warning("No samples processed; returning empty DataFrame")
            return pd.DataFrame(columns=["filename", "label"] + FEATURE_NAMES)

        df = pd.DataFrame(all_rows)
        return df

    def export(self, df: pd.DataFrame, out_dir: Path | None = None) -> dict[str, Path]:
        """Export master CSV and train/val/test splits."""
        out_dir = Path(out_dir or self.config["paths"].get("features_dir", "data/processed/features"))
        ensure_dirs(out_dir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version = self.config.get("dataset", {}).get("version", "1.0.0")

        master_path = out_dir / "master_dataset.csv"
        df.to_csv(master_path, index=False)

        splits = self.splitter.split(df)
        paths = {"master": master_path}
        for name, part in splits.items():
            p = out_dir / f"{name}.csv"
            part.to_csv(p, index=False)
            paths[name] = p

        meta = {
            "version": version,
            "created": ts,
            "feature_count": len(FEATURE_NAMES),
            "sample_count": len(df),
            "class_distribution": df["label"].value_counts().to_dict() if "label" in df else {},
            "config_hash": DatasetSplitter.config_hash(self.config),
        }
        meta_path = out_dir / "dataset_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths["metadata"] = meta_path
        if self.processing_log:
            log_dir = Path(self.config["paths"].get("logs_dir", "data/logs"))
            ensure_dirs(log_dir)
            log_path = log_dir / "processing_log.csv"
            pd.DataFrame(self.processing_log).to_csv(log_path, index=False)
            paths["processing_log"] = log_path
        logger.info("Exported dataset: %d samples", len(df))
        return paths


def build_dataset(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """CLI-friendly entry point for dataset building."""
    config = load_config(config_path)
    builder = DatasetBuilder(config)
    df = builder.build_from_raw()
    if len(df):
        builder.export(df)
    return df
