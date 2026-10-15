"""Fair model-family benchmark on one fixed host-grouped test split."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import Evaluator
from src.feature_extraction import dataframe_to_model_matrix
from src.trainer import Trainer
from src.utils import ensure_dirs, get_logger, load_config, set_seed
from train import (
    _load_cached_flux,
    _load_exported_splits,
    _filter_supervised_rows,
    _synthetic,
    _validate_class_support,
    _validate_feature_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exoplanet model families")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "neural_net", "hybrid_tcn"],
        default=["random_forest", "xgboost"],
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    use_masks = config.get("features", {}).get("use_masks", True)

    if args.smoke_test:
        frame = _synthetic(
            args.samples,
            config["seed"],
            len(config.get("classes", {}).get("astro_names", [])) or 2,
        )
        split_frames = None
    else:
        split_frames = _load_exported_splits(config)
        if split_frames is None:
            raise RuntimeError("Build and export train/val/test datasets before benchmarking")
        master = pd.concat(split_frames, ignore_index=True)
        _validate_feature_version(master, config, False)
        split_frames = tuple(
            _filter_supervised_rows(frame, config, logger)
            for frame in split_frames
        )
        master = pd.concat(split_frames, ignore_index=True)
        _validate_class_support(master, config, False)

    output = Path(
        args.output
        or Path(config["paths"]["models_dir"]) / "benchmark_results.json"
    )
    ensure_dirs(output.parent)
    results = []

    for model_type in args.models:
        trial_config = copy.deepcopy(config)
        trial_config["model"]["type"] = model_type
        trial_config["paths"]["models_dir"] = str(
            output.parent / "benchmarks" / model_type
        )
        ensure_dirs(trial_config["paths"]["models_dir"])
        trainer = Trainer(trial_config)
        started = time.perf_counter()

        if split_frames is None:
            X, _ = dataframe_to_model_matrix(frame, use_masks=use_masks)
            y = frame["label"].to_numpy(dtype=int)
            flux = (
                np.tile(np.linspace(-1.0, 1.0, 1024), (len(frame), 1))
                if model_type == "hybrid_tcn"
                else None
            )
            trained = trainer.train(X, y, flux)
        else:
            train_frame, val_frame, test_frame = split_frames
            matrices = [
                dataframe_to_model_matrix(part, use_masks=use_masks)[0]
                for part in split_frames
            ]
            labels = [part["label"].to_numpy(dtype=int) for part in split_frames]
            fluxes = [None, None, None]
            if model_type == "hybrid_tcn":
                fluxes = [
                    _load_cached_flux(part, trial_config, logger)
                    for part in split_frames
                ]
                if any(value is None for value in fluxes):
                    raise RuntimeError("Hybrid benchmark requires cached flux for all splits")
            trained = trainer.train_pre_split(
                matrices[0],
                labels[0],
                matrices[1],
                labels[1],
                matrices[2],
                labels[2],
                fluxes[0],
                fluxes[1],
                fluxes[2],
            )

        model, X_test, y_test = trained[:3]
        flux_test = trained[3] if len(trained) > 3 else None
        training_seconds = time.perf_counter() - started
        evaluation_config = copy.deepcopy(trial_config)
        evaluation_config["classes"]["astro_names"] = trainer.class_names
        evaluation_config["inference"]["planet_threshold"] = trainer.planet_threshold
        evaluator = Evaluator(evaluation_config)
        metrics = evaluator.evaluate(
            model,
            X_test,
            y_test,
            flux_test,
            calibrator=trainer.calibrator,
        )
        inference_seconds = _latency(evaluator, model, X_test, flux_test)
        record = {
            "model": model_type,
            "training_seconds": training_seconds,
            "median_batch_inference_seconds": inference_seconds,
            "test_samples": len(y_test),
            "metrics": {
                key: value
                for key, value in metrics.items()
                if key not in ("proba", "raw_proba")
            },
            "class_names": trainer.class_names,
            "scientific": not args.smoke_test,
        }
        results.append(record)
        output.write_text(
            json.dumps({"results": results}, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(
            "%s: macro_f1=%.3f balanced_acc=%.3f train=%.1fs",
            model_type,
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            training_seconds,
        )

    logger.info("Benchmark written to %s", output)


def _latency(evaluator, model, X, flux) -> float:
    count = min(len(X), 256)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        evaluator.predict_proba(
            model,
            X[:count],
            flux[:count] if flux is not None else None,
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


if __name__ == "__main__":
    main()
