"""Dataset cache for deterministic processing outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dirs, get_logger

logger = get_logger()


class FeatureCache:
    """Cache cleaned/detrended curves and extracted features (not BLS/TLS)."""

    def __init__(self, cache_dir: str | Path, version: str = "1.0.0") -> None:
        self.cache_dir = Path(cache_dir)
        self.version = version
        ensure_dirs(self.cache_dir)

    def _key(self, identifier: str) -> str:
        raw = f"{identifier}|{self.version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def path(self, identifier: str, ext: str = "npz") -> Path:
        return self.cache_dir / f"{self._key(identifier)}.{ext}"

    def exists(self, identifier: str) -> bool:
        return self.path(identifier, "npz").exists()

    def save(
        self,
        identifier: str,
        features: np.ndarray,
        masks: np.ndarray | None = None,
        metadata: dict | None = None,
        feature_names: list[str] | None = None,
        flux_resampled: np.ndarray | None = None,
    ) -> Path:
        """Save compressed feature cache."""
        p = self.path(identifier, "npz")
        payload = {"features": features}
        if masks is not None:
            payload["masks"] = masks
        if flux_resampled is not None:
            payload["flux_resampled"] = flux_resampled
        np.savez_compressed(p, **payload)
        processing_hash = self.processing_hash(features, masks)
        meta_payload = {
            "identifier": identifier,
            "version": self.version,
            "processing_hash": processing_hash,
            **(metadata or {}),
        }
        meta_path = self.path(identifier, "json")
        meta_path.write_text(json.dumps(meta_payload, indent=2, default=str), encoding="utf-8")
        if feature_names:
            row = {name: float(value) for name, value in zip(feature_names, features)}
            if masks is not None:
                row.update({f"mask__{name}": float(value) for name, value in zip(feature_names, masks)})
            row["processing_hash"] = processing_hash
            row["version"] = self.version
            pd.DataFrame([row]).to_csv(self.path(identifier, "csv"), index=False)
        logger.debug("Cached features: %s", p.name)
        return p

    def load(self, identifier: str) -> tuple[np.ndarray, np.ndarray | None, dict | None]:
        p = self.path(identifier, "npz")
        if not p.exists():
            raise FileNotFoundError(f"Cache miss: {identifier}")
        with np.load(p, allow_pickle=False) as data:
            features = np.asarray(data["features"]).copy()
            masks = np.asarray(data["masks"]).copy() if "masks" in data else None
        meta_path = self.path(identifier, "json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        return features, masks, metadata

    def load_flux(self, identifier: str) -> np.ndarray:
        with np.load(self.path(identifier, "npz"), allow_pickle=False) as data:
            if "flux_resampled" not in data:
                raise KeyError(f"Cached flux not found for {identifier}")
            return np.asarray(data["flux_resampled"], dtype=float).copy()

    def processing_hash(
        self, features: np.ndarray, masks: np.ndarray | None = None
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(np.asarray(features, dtype=float).tobytes())
        if masks is not None:
            hasher.update(np.asarray(masks, dtype=float).tobytes())
        hasher.update(self.version.encode())
        return hasher.hexdigest()[:16]
