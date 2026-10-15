"""Publication-quality plots."""
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from .utils import LightCurve, ensure_dirs, get_logger

logger = get_logger()
sns.set_theme(context="paper", style="whitegrid")


class Visualizer:
    """Generate and save figures.

    Parameters
    ----------
    outdir : str | Path
        Directory where PNGs are written.
    """

    def __init__(
        self,
        outdir: str | Path = "data/processed/figures",
        output_dir: str | Path | None = None,
    ) -> None:
        if output_dir is not None:
            outdir = output_dir
        self.outdir = Path(outdir)
        ensure_dirs(self.outdir)

    def _save(self, fig, name: str) -> Path:
        path = self.outdir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure %s", path)
        return path

    def light_curve(self, lc: LightCurve, title: str, name: str) -> Path:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(lc.time, lc.flux, ".", ms=2, alpha=0.6, color="#1f77b4")
        ax.set(xlabel="Time [days]", ylabel="Normalized flux", title=title)
        return self._save(fig, name)

    def periodogram(self, periods, power, best_period, name="periodogram.png"):
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(periods, power, lw=0.8, color="#444")
        ax.axvline(best_period, color="crimson", ls="--",
                   label=f"P = {best_period:.3f} d")
        ax.set(xlabel="Period [days]", ylabel="BLS power",
               title="BLS Periodogram")
        ax.legend()
        return self._save(fig, name)

    def phase_fold(self, phase, flux, centers=None, binned=None,
                   name="phase_fold.png"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(phase, flux, ".", ms=2, alpha=0.3, color="grey",
                label="folded")
        if centers is not None and binned is not None:
            ax.plot(centers, binned, "-", color="crimson", lw=2,
                    label="binned")
        ax.set(xlabel="Phase", ylabel="Normalized flux",
               title="Phase-folded transit")
        ax.legend()
        return self._save(fig, name)

    def residuals(self, x, residual, name="residuals.png"):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.axhline(0, color="black", lw=1)
        ax.plot(x, residual, ".", ms=2, alpha=0.5, color="#555")
        ax.set(xlabel="Phase", ylabel="Residual flux", title="Transit Residuals")
        return self._save(fig, name)

    def importance(
        self,
        values,
        names,
        name="feature_importance.png",
        top_n: int = 25,
        aggregate_masks: bool = True,
    ):
        values_arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        names_arr = np.asarray([str(n) for n in names], dtype=object)
        if aggregate_masks:
            names_arr, values_arr = _aggregate_mask_importance(names_arr, values_arr)

        magnitude = np.abs(values_arr)
        if not len(magnitude):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
            ax.axis("off")
            return self._save(fig, name)

        keep = np.argsort(magnitude)[-min(top_n, len(magnitude)):]
        order = keep[np.argsort(magnitude[keep])]
        labels = [_wrap_label(names_arr[i]) for i in order]
        plot_values = magnitude[order]
        height = max(5.0, 0.34 * len(order) + 1.4)
        fig, ax = plt.subplots(figsize=(9.5, height))
        ax.barh(labels, plot_values, color="#2a6f97")
        ax.set(
            xlabel="Mean absolute contribution / score decrease",
            title=f"Top {len(order)} feature importances",
        )
        ax.tick_params(axis="y", labelsize=8)
        fig.subplots_adjust(left=0.34)
        return self._save(fig, name)

    def roc(self, y, proba, name="roc_curve.png"):
        fpr, tpr, _ = roc_curve(y, proba)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2, color="#1f77b4")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curve")
        return self._save(fig, name)

    def pr(self, y, proba, name="pr_curve.png"):
        prec, rec, _ = precision_recall_curve(y, proba)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rec, prec, lw=2, color="#d62728")
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Precision-Recall Curve")
        return self._save(fig, name)

    def reliability(self, y, proba, n_bins=10, name="reliability_curve.png"):
        if np.asarray(proba).ndim > 1:
            conf = np.max(proba, axis=1)
            pred = np.argmax(proba, axis=1)
        else:
            conf = np.asarray(proba)
            pred = (conf >= 0.5).astype(int)
        bins = np.linspace(0, 1, n_bins + 1)
        centers, accs, confs = [], [], []
        for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (conf >= lo) & (conf <= hi if index == len(bins) - 2 else conf < hi)
            if not np.any(mask):
                continue
            centers.append((lo + hi) / 2)
            accs.append(np.mean(np.asarray(y)[mask] == pred[mask]))
            confs.append(np.mean(conf[mask]))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(confs, accs, marker="o", lw=2, color="#1f77b4")
        ax.set(xlabel="Mean confidence", ylabel="Accuracy", title="Reliability")
        return self._save(fig, name)

    def confidence_histogram(self, proba, name="confidence_histogram.png"):
        conf = np.max(proba, axis=1) if np.asarray(proba).ndim > 1 else np.asarray(proba)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(conf, bins=np.linspace(0, 1, 21), color="#4c78a8", edgecolor="white")
        ax.set(xlabel="Confidence", ylabel="Count", title="Confidence Histogram")
        return self._save(fig, name)

    def confusion(self, cm, labels=None, name="confusion_matrix.png"):
        matrix = np.array(cm)
        if labels is None or len(labels) != matrix.shape[0]:
            labels = [str(i) for i in range(matrix.shape[0])]
        size = max(4.5, 0.6 * matrix.shape[0])
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        return self._save(fig, name)


def _aggregate_mask_importance(names: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine ``mask__feature`` importances with their base feature."""
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        base = str(name)
        if base.startswith("mask__"):
            base = base.removeprefix("mask__")
        totals[base] = totals.get(base, 0.0) + abs(float(value))
    ordered = list(totals.items())
    return (
        np.asarray([item[0] for item in ordered], dtype=object),
        np.asarray([item[1] for item in ordered], dtype=float),
    )


def _wrap_label(label: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(label.replace("_", " "), width=width)) or label
