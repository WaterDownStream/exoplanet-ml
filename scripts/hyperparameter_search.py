"""Run reproducible hyperparameter search trials."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.sampler import ClassSampler
from src.evaluator import Evaluator
from src.feature_extraction import dataframe_to_model_matrix
from src.trainer import Trainer
from src.utils import ensure_dirs, get_logger, load_config, set_seed
from train import (
    _load_cached_flux,
    _load_exported_splits,
    _load_training_dataframe,
    _filter_supervised_rows,
    _validate_class_support,
    _validate_feature_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--strategy", choices=["random", "grid", "optuna"], default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)
    hpo_cfg = base_config.get("hpo", {})
    strategy = args.strategy or hpo_cfg.get("strategy", "random")
    max_trials = int(args.max_trials or hpo_cfg.get("max_trials", 12))
    metric_name = hpo_cfg.get("metric", "macro_f1")
    logger = get_logger(base_config.get("log_level", "INFO"))
    set_seed(base_config["seed"])

    df = _load_training_dataframe(base_config, args.features, logger)
    _validate_feature_version(df, base_config, False)
    df = _filter_supervised_rows(df, base_config, logger)
    if args.features is None:
        exported = _load_exported_splits(base_config)
        if exported is not None:
            import pandas as pd

            # The final test split is deliberately excluded from model selection.
            df = pd.concat((exported[0], exported[1]), ignore_index=True)
            df = _filter_supervised_rows(df, base_config, logger)
    _validate_class_support(df, base_config, False)
    df = ClassSampler(
        base_config.get("training", {}).get("class_balancing", "weighted_loss"),
        base_config["seed"],
    ).balance(df)
    y = df["label"].to_numpy(dtype=int)

    if strategy == "optuna":
        if _run_optuna_search(base_config, hpo_cfg, df, y, max_trials, metric_name, args.output, logger):
            return

    trials = _trial_parameters(
        hpo_cfg.get("search_space", {}),
        strategy,
        max_trials,
        base_config["seed"],
        logger,
    )

    output = Path(args.output or Path(base_config["paths"]["models_dir"]) / "hpo_results.json")
    ensure_dirs(output.parent)
    results = []
    best = None

    for idx, params in enumerate(trials, start=1):
        trial_config = copy.deepcopy(base_config)
        for key, value in params.items():
            _set_nested(trial_config, key, value)
        trial_config["paths"]["models_dir"] = str(output.parent / "hpo" / f"trial_{idx:03d}")
        ensure_dirs(trial_config["paths"]["models_dir"])

        logger.info("HPO trial %d/%d: %s", idx, len(trials), params)
        started = time.perf_counter()
        gpu_start = _gpu_memory_mb(reset=True)
        try:
            score, metrics = _run_trial(trial_config, df, y, logger)
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trial %d failed: %s", idx, exc)
            score = None
            metrics = {}
            status = "failed"
            error = str(exc)
        runtime = time.perf_counter() - started
        gpu_peak = _gpu_memory_mb(reset=False)
        record = {
            "trial": idx,
            "status": status,
            "parameters": params,
            "metric": metric_name,
            "score": score,
            "runtime_seconds": runtime,
            "gpu_memory_mb": max(gpu_start, gpu_peak),
            "metrics": metrics,
            "error": error,
        }
        results.append(record)
        if score is not None and (best is None or score > best["score"]):
            best = record
        output.write_text(
            json.dumps({"best": best, "trials": results}, indent=2, default=str),
            encoding="utf-8",
        )

    logger.info("HPO complete. Best %s=%.4f", metric_name, best["score"] if best else float("nan"))


def _run_optuna_search(
    base_config: dict,
    hpo_cfg: dict,
    df,
    y: np.ndarray,
    max_trials: int,
    metric_name: str,
    output_arg: str | None,
    logger,
) -> bool:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna is unavailable; falling back to seeded random search")
        return False

    search_space = hpo_cfg.get("search_space", {})
    if not search_space:
        logger.warning("Optuna requested with empty search space; falling back to base config")
        return False

    output = Path(output_arg or Path(base_config["paths"]["models_dir"]) / "hpo_results.json")
    ensure_dirs(output.parent)
    results = []

    def objective(trial) -> float:
        params = {
            key: trial.suggest_categorical(key, list(values))
            for key, values in search_space.items()
        }
        trial_config = copy.deepcopy(base_config)
        for key, value in params.items():
            _set_nested(trial_config, key, value)
        trial_config["paths"]["models_dir"] = str(
            output.parent / "hpo" / f"optuna_trial_{trial.number + 1:03d}"
        )
        ensure_dirs(trial_config["paths"]["models_dir"])

        started = time.perf_counter()
        _gpu_memory_mb(reset=True)
        try:
            score, metrics = _run_trial(trial_config, df, y, logger)
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001
            score = 0.0
            metrics = {}
            status = "failed"
            error = str(exc)
        record = {
            "trial": trial.number + 1,
            "status": status,
            "parameters": params,
            "metric": metric_name,
            "score": score,
            "runtime_seconds": time.perf_counter() - started,
            "gpu_memory_mb": _gpu_memory_mb(reset=False),
            "metrics": metrics,
            "error": error,
        }
        trial.set_user_attr("record", record)
        results.append(record)
        best = max(
            (item for item in results if item["status"] == "ok"),
            key=lambda item: item["score"],
            default=None,
        )
        output.write_text(
            json.dumps({"best": best, "trials": results}, indent=2, default=str),
            encoding="utf-8",
        )
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=base_config["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_trials)
    best = study.best_trial.user_attrs.get("record") if len(study.trials) else None
    output.write_text(
        json.dumps({"best": best, "trials": results}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Optuna HPO complete. Best %s=%.4f", metric_name, best["score"] if best else float("nan"))
    return True


def _run_trial(config: dict, df, y: np.ndarray, logger) -> tuple[float, dict[str, Any]]:
    X, _ = dataframe_to_model_matrix(
        df,
        use_masks=config.get("features", {}).get("use_masks", True),
    )
    flux = None
    if config["model"]["type"] == "hybrid_tcn":
        flux = _load_cached_flux(df, config, logger)
        if flux is None:
            logger.warning("Skipping hybrid trial without cached flux input")
            raise RuntimeError("hybrid_tcn requires cached flux_resampled arrays")

    trainer = Trainer(config)
    groups = (
        df["tic_id"].fillna(df.get("filename", "")).astype(str).to_numpy()
        if "tic_id" in df
        else df.get("filename", df.index.astype(str)).astype(str).to_numpy()
    )
    result = trainer.train(X, y, flux, groups=groups)
    model = result[0]
    X_test, y_test = result[1], result[2]
    flux_test = result[3] if len(result) > 3 else None
    evaluation_config = copy.deepcopy(config)
    evaluation_config.setdefault("classes", {})["astro_names"] = trainer.class_names
    evaluation_config.setdefault("inference", {})[
        "planet_threshold"
    ] = trainer.planet_threshold
    metrics = Evaluator(evaluation_config).evaluate(
        model,
        X_test,
        y_test,
        flux_test,
        calibrator=trainer.calibrator,
    )
    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in ("proba", "raw_proba")
    }
    metric_name = config.get("hpo", {}).get("metric", "macro_f1")
    score = float(clean_metrics.get(metric_name, clean_metrics.get("accuracy", 0.0)))
    return score, clean_metrics


def _trial_parameters(
    search_space: dict[str, list[Any]],
    strategy: str,
    max_trials: int,
    seed: int,
    logger,
) -> list[dict[str, Any]]:
    if not search_space:
        return [{}]
    if strategy == "grid":
        return _grid(search_space)[:max_trials]
    rng = random.Random(seed)
    keys = list(search_space)
    trials = []
    seen = set()
    attempts = 0
    while len(trials) < max_trials and attempts < max_trials * 20:
        attempts += 1
        params = {key: rng.choice(list(search_space[key])) for key in keys}
        signature = tuple((key, _jsonable(params[key])) for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(params)
    return trials or _grid(search_space)[:max_trials]


def _grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search_space)
    values = [list(search_space[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _gpu_memory_mb(reset: bool = False) -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        return 0.0


def _jsonable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
