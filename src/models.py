"""Classifier factory: RF, XGBoost, MLP, TCN hybrid."""
from __future__ import annotations

import numpy as np

from .utils import get_logger

logger = get_logger()


def build_sklearn_model(cfg: dict):
    """Instantiate a Random Forest or XGBoost classifier from config."""
    mtype = cfg["type"]
    if mtype == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        c = cfg["random_forest"]
        return RandomForestClassifier(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            class_weight=c["class_weight"],
            min_samples_leaf=int(c.get("min_samples_leaf", 2)),
            max_features=c.get("max_features", "sqrt"),
            criterion=c.get("criterion", "log_loss"),
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
        )
    if mtype == "xgboost":
        from xgboost import XGBClassifier

        c = cfg["xgboost"]
        kwargs = {
            "tree_method": "hist",
            "device": c.get("device", "cpu"),
        }
        return XGBClassifier(
            n_estimators=c["n_estimators"],
            max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
            eval_metric="mlogloss",
            random_state=cfg.get("seed", 42),
            n_jobs=-1,
            **kwargs,
        )
    raise ValueError(f"{mtype} is not an sklearn-style model")


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TransitNet(nn.Module):
        """Feed-forward classifier (binary or multi-class)."""

        def __init__(
            self,
            input_dim: int,
            hidden_layers: list[int],
            dropout: float,
            n_classes: int = 2,
        ) -> None:
            super().__init__()
            self.n_classes = n_classes
            layers: list[nn.Module] = []
            prev = input_dim
            for i, h in enumerate(hidden_layers):
                layers.append(nn.Linear(prev, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if i < len(hidden_layers) - 1:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class TCNBlock(nn.Module):
        """Causal temporal convolution block."""

        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel: int,
            dropout: float,
            dilation: int = 1,
        ) -> None:
            super().__init__()
            pad = dilation * (kernel - 1) // 2
            groups = max(1, min(8, out_ch))
            while out_ch % groups:
                groups -= 1
            self.conv1 = nn.Conv1d(
                in_ch, out_ch, kernel, padding=pad, dilation=dilation
            )
            self.norm1 = nn.GroupNorm(groups, out_ch)
            self.conv2 = nn.Conv1d(
                out_ch, out_ch, kernel, padding=pad, dilation=dilation
            )
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.drop = nn.Dropout1d(dropout)
            self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = self.down(x)
            out = self.drop(F.gelu(self.norm1(self.conv1(x))))
            out = self.drop(self.norm2(self.conv2(out)))
            return F.gelu(out + residual)

    class TCNBranch(nn.Module):
        """Lightweight TCN for raw flux (RTX 4060 friendly)."""

        def __init__(self, channels: list[int], kernel_size: int, dropout: float) -> None:
            super().__init__()
            blocks = []
            in_ch = 1
            for index, ch in enumerate(channels):
                blocks.append(
                    TCNBlock(
                        in_ch,
                        ch,
                        kernel_size,
                        dropout,
                        dilation=2**index,
                    )
                )
                in_ch = ch
            self.blocks = nn.Sequential(*blocks)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.unsqueeze(1)
            encoded = self.blocks(x)
            return torch.cat(
                [self.avg_pool(encoded).squeeze(-1), self.max_pool(encoded).squeeze(-1)],
                dim=-1,
            )

    class HybridTCNClassifier(nn.Module):
        """Feature branch + TCN branch fusion for multi-class classification."""

        def __init__(
            self,
            n_features: int,
            n_classes: int,
            cfg: dict,
        ) -> None:
            super().__init__()
            self.n_classes = n_classes
            tcn_cfg = cfg.get("hybrid_tcn", cfg)
            self.tcn = TCNBranch(
                list(tcn_cfg.get("tcn_channels", [32, 64, 64])),
                int(tcn_cfg.get("tcn_kernel_size", 3)),
                float(tcn_cfg.get("tcn_dropout", 0.2)),
            )
            tcn_dim = 2 * tcn_cfg.get("tcn_channels", [32, 64, 64])[-1]
            feat_hidden = list(tcn_cfg.get("feature_hidden", [128, 64]))
            feat_layers: list[nn.Module] = []
            prev = n_features
            for h in feat_hidden:
                feat_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(tcn_cfg.get("dropout", 0.3))]
                prev = h
            self.feature_branch = nn.Sequential(*feat_layers)
            fusion_dim = prev + tcn_dim
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(tcn_cfg.get("dropout", 0.3)),
                nn.Linear(64, n_classes),
            )

        def forward(
            self, features: "torch.Tensor", flux: "torch.Tensor"
        ) -> "torch.Tensor":
            f_emb = self.feature_branch(features)
            t_emb = self.tcn(flux)
            fused = torch.cat([f_emb, t_emb], dim=-1)
            return self.head(fused)

except ImportError:  # pragma: no cover
    TransitNet = None  # type: ignore
    HybridTCNClassifier = None  # type: ignore
    logger.warning("PyTorch not installed; neural net unavailable")


def resample_flux(flux: np.ndarray, length: int) -> np.ndarray:
    """Robustly normalize and linearly resample flux for the temporal branch."""
    values = np.asarray(flux, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return np.zeros(length, dtype=np.float32)
    if not np.all(finite):
        index = np.arange(len(values))
        values = values.copy()
        values[~finite] = np.interp(index[~finite], index[finite], values[finite])
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    values = np.clip((values - center) / scale, -12.0, 12.0)
    if len(values) == length:
        return values.astype(np.float32)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, length)
    return np.interp(x_new, x_old, values).astype(np.float32)
