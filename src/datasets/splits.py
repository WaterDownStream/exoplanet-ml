"""Train/validation/test splitting by host star."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import get_logger

logger = get_logger()


class DatasetSplitter:
    """Split datasets at the target-star level to prevent leakage."""

    def __init__(self, cfg: dict, seed: int = 42) -> None:
        self.ratios = cfg.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        self.seed = seed
        total = sum(float(self.ratios.get(name, 0.0)) for name in ("train", "val", "test"))
        if not np.isclose(total, 1.0) or any(float(self.ratios.get(name, 0.0)) <= 0 for name in ("train", "val", "test")):
            raise ValueError("dataset.split_ratios must be positive and sum to 1")

    def split(self, df: pd.DataFrame, group_col: str = "tic_id") -> dict[str, pd.DataFrame]:
        """Return train/val/test DataFrames grouped by star."""
        df = df.copy()
        if group_col not in df.columns:
            df[group_col] = df.get("filename", df.index.astype(str))
        else:
            groups = df[group_col].fillna("").astype(str).str.strip()
            if groups.nunique() < 3 and "filename" in df.columns:
                logger.warning(
                    "Group column %s has fewer than 3 usable groups; falling back to filename",
                    group_col,
                )
                df[group_col] = df["filename"].fillna("").astype(str)
            else:
                fallback = (
                    df["filename"].fillna("").astype(str)
                    if "filename" in df.columns
                    else pd.Series(df.index.astype(str), index=df.index)
                )
                df[group_col] = groups.mask(groups == "", fallback)

        if len(df) < 3 or df[group_col].nunique() < 3:
            logger.warning("Dataset too small for 70/15/15 split; assigning all rows to train")
            empty = df.iloc[0:0].copy()
            return {"train": df.copy(), "val": empty.copy(), "test": empty.copy()}

        label_counts = df.groupby(group_col)["label"].nunique() if "label" in df else None
        if label_counts is not None and np.any(label_counts > 1):
            conflicts = label_counts[label_counts > 1].index.astype(str).tolist()[:10]
            raise ValueError(
                "A host star cannot have conflicting class labels; conflicts include "
                f"{conflicts}"
            )

        groups = df.groupby(group_col, sort=True).first().reset_index()
        labels = groups.get("label", pd.Series(0, index=groups.index))
        stratify = labels if _can_stratify(labels) else None

        g_train, g_tmp = train_test_split(
            groups[group_col],
            test_size=1 - self.ratios["train"],
            stratify=stratify,
            random_state=self.seed,
        )
        val_ratio = self.ratios["val"] / (self.ratios["val"] + self.ratios["test"])
        tmp_labels = groups.set_index(group_col).loc[g_tmp]["label"] if "label" in groups else None
        g_val, g_test = train_test_split(
            g_tmp,
            test_size=1 - val_ratio,
            stratify=tmp_labels if tmp_labels is not None and _can_stratify(tmp_labels) else None,
            random_state=self.seed,
        )

        splits = {
            "train": df[df[group_col].isin(g_train)].copy(),
            "val": df[df[group_col].isin(g_val)].copy(),
            "test": df[df[group_col].isin(g_test)].copy(),
        }
        if "is_augmented" in df.columns:
            for name in ("val", "test"):
                augmented = splits[name]["is_augmented"].fillna(False).astype(bool)
                splits[name] = splits[name].loc[~augmented].copy()
        for name, part in splits.items():
            logger.info("Split %s: %d rows", name, len(part))
        return splits

    @staticmethod
    def config_hash(config: dict) -> str:
        relevant = {
            key: config.get(key)
            for key in (
                "seed",
                "preprocessing",
                "quality",
                "detrending",
                "bls",
                "tls",
                "mono_search",
                "phase_folding",
                "features",
                "dataset",
                "augmentation",
                "classes",
            )
        }
        payload = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and bool((counts >= 2).all())
