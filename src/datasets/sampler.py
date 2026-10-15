"""Class-balanced sampling for training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger()


class ClassSampler:
    """Balance classes via oversampling or weights."""

    def __init__(self, method: str = "weighted_loss", seed: int = 42) -> None:
        self.method = method
        self.rng = np.random.default_rng(seed)

    def sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights for imbalanced classes."""
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weight_map[yi] for yi in y], dtype=float)

    def oversample(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """Oversample minority classes to match majority count."""
        if self.method != "oversample":
            return df
        max_count = df[label_col].value_counts().max()
        parts = []
        for label, group in df.groupby(label_col):
            if len(group) < max_count:
                extra = group.sample(max_count - len(group), replace=True, random_state=int(self.rng.integers(1e9)))
                parts.append(pd.concat([group, extra]))
            else:
                parts.append(group)
        out = pd.concat(parts).sample(frac=1, random_state=int(self.rng.integers(1e9))).reset_index(drop=True)
        logger.info("Oversampled dataset: %d rows", len(out))
        return out

    def undersample(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """Downsample majority classes to match the smallest class."""
        if self.method != "undersample":
            return df
        min_count = df[label_col].value_counts().min()
        parts = [
            group.sample(min_count, random_state=int(self.rng.integers(1e9)))
            for _, group in df.groupby(label_col)
        ]
        out = pd.concat(parts).sample(frac=1, random_state=int(self.rng.integers(1e9))).reset_index(drop=True)
        logger.info("Undersampled dataset: %d rows", len(out))
        return out

    def balance(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """Apply the configured deterministic balancing strategy."""
        if self.method == "oversample":
            return self.oversample(df, label_col)
        if self.method == "undersample":
            return self.undersample(df, label_col)
        return df
