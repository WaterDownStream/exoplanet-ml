"""Evaluate a trained model on a test feature CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import Evaluator
from src.feature_extraction import dataframe_to_model_matrix
from src.datasets.cache import FeatureCache
from src.utils import get_logger, load_config, set_seed


def _load_cached_flux(df: pd.DataFrame, config: dict) -> np.ndarray | None:
    if "cache_id" not in df.columns:
        return None
    cache = FeatureCache(
        config["paths"].get("cache_dir", "data/cache"),
        config.get("dataset", {}).get("version", "1.0.0"),
    )
    arrays = []
    for cache_id in df["cache_id"].fillna(""):
        if not cache_id:
            return None
        try:
            arrays.append(cache.load_flux(str(cache_id)))
        except Exception:  # noqa: BLE001
            return None
    return np.asarray(arrays, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exoplanet classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test", default="data/processed/features/test.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))

    df = pd.read_csv(args.test)
    y = df["label"].to_numpy(dtype=int)
    d = Path(config["paths"]["models_dir"])
    meta = joblib.load(d / "model_meta.joblib")
    if int(meta.get("artifact_version", 1)) < 2:
        raise RuntimeError("Model artifacts are obsolete; rebuild and retrain first")
    if meta.get("class_names"):
        config.setdefault("classes", {})["astro_names"] = list(meta["class_names"])
    use_masks = bool(meta.get("use_masks", config.get("features", {}).get("use_masks", True)))
    X, _ = dataframe_to_model_matrix(df, use_masks=use_masks)
    imputer = joblib.load(d / "imputer.joblib")
    selector = joblib.load(d / "selector.joblib")
    X = joblib.load(d / "scaler.joblib").transform(
        selector.transform(imputer.transform(X))
    )
    mtype = meta.get("type", config["model"]["type"])
    config["model"]["type"] = mtype
    input_dim = int(meta.get("input_dim", X.shape[1]))
    saved_model_config = meta.get("model_config", config["model"])

    if mtype == "hybrid_tcn":
        import torch
        from src.models import HybridTCNClassifier
        n_classes = meta.get("n_classes", 7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridTCNClassifier(input_dim, n_classes, saved_model_config).to(device)
        model.load_state_dict(
            torch.load(d / "hybrid_tcn.pt", map_location=device, weights_only=True)
        )
        model.eval()
    elif mtype == "neural_net":
        import torch
        from src.models import TransitNet
        c = saved_model_config["neural_net"]
        n_classes = meta.get("n_classes", 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransitNet(
            input_dim, list(c["hidden_layers"]), c["dropout"], n_classes
        ).to(device)
        model.load_state_dict(
            torch.load(d / "nn_model.pt", map_location=device, weights_only=True)
        )
        model.eval()
    else:
        model = joblib.load(d / f"{mtype}.joblib")

    flux = _load_cached_flux(df, config) if mtype == "hybrid_tcn" else None
    if mtype == "hybrid_tcn" and flux is None:
        raise RuntimeError("Hybrid evaluation requires cache_id rows with cached flux_resampled arrays")

    active_ids = list(meta.get("active_global_class_ids", range(len(config["classes"]["astro_names"]))))
    label_map = {int(global_id): index for index, global_id in enumerate(active_ids)}
    try:
        y = np.asarray([label_map[int(value)] for value in y], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Test set contains class absent from trained model: {exc.args[0]}") from exc
    calibrator = (
        joblib.load(d / "calibrator.joblib")
        if (d / "calibrator.joblib").exists()
        else None
    )
    metrics = Evaluator(config).evaluate(
        model, X, y, flux, calibrator=calibrator
    )
    logger.info("Evaluation metrics: %s", {k: v for k, v in metrics.items() if k != "proba"})


if __name__ == "__main__":
    main()
