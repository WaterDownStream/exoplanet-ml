"""Memory-bounded batch inference for large light-curve collections."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import load_model_bundle
from src.pipeline import Pipeline
from src.utils import ensure_dirs, get_logger, load_config, set_seed

_LOCAL = threading.local()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch exoplanet inference")
    parser.add_argument("lightcurves", nargs="*")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data/reports/batch_predictions.csv")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    logger = get_logger(config.get("log_level", "INFO"))
    files = _collect_files(args.lightcurves, args.input_dir)
    if not files:
        raise ValueError("No supported light-curve files were found")

    bundle = None
    model_directory = Path(config["paths"]["models_dir"])
    if (model_directory / "scaler.joblib").exists():
        bundle = load_model_bundle(config)
        config["model"]["type"] = bundle["type"]
        if bundle["meta"].get("class_names"):
            config["classes"]["astro_names"] = bundle["meta"]["class_names"]

    workers = max(int(args.workers), 1)
    if bundle is not None and bundle["type"] in ("neural_net", "hybrid_tcn"):
        try:
            import torch

            if torch.cuda.is_available() and workers > 1:
                logger.warning("CUDA batch inference uses one worker to avoid model contention")
                workers = 1
        except ImportError:
            pass

    output = Path(args.output)
    ensure_dirs(output.parent)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = [
        "source_file",
        "status",
        "error",
        "runtime_seconds",
        "target_id",
        "quality_score",
        "candidate_type",
        "predicted_class",
        "reported_probability",
        "is_calibrated",
        "class_probabilities_json",
        "bls_period_days",
        "bls_duration_days",
        "bls_depth",
        "bls_snr",
        "bls_sde",
        "bls_sampled_transits",
        "mono_epoch",
        "mono_duration_days",
        "mono_depth",
        "mono_snr",
        "warnings_json",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        worker = lambda path: _process_one(path, config, bundle)
        if workers == 1:
            rows = map(worker, files)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            rows = executor.map(worker, files)
        try:
            for index, row in enumerate(rows, start=1):
                writer.writerow(row)
                if index % 25 == 0:
                    handle.flush()
                    logger.info("Processed %d/%d light curves", index, len(files))
        finally:
            if workers > 1:
                executor.shutdown(wait=True)
    temporary.replace(output)
    logger.info("Batch inference complete: %d rows -> %s", len(files), output)


def _process_one(path: Path, config: dict, bundle: dict | None) -> dict:
    started = time.perf_counter()
    base = {
        "source_file": str(path),
        "status": "failed",
        "error": "",
    }
    try:
        pipeline = getattr(_LOCAL, "pipeline", None)
        if pipeline is None:
            pipeline = Pipeline(config)
            _LOCAL.pipeline = pipeline
        result = pipeline.run(
            path,
            model=bundle["model"] if bundle else None,
            scaler=bundle["scaler"] if bundle else None,
            imputer=bundle["imputer"] if bundle else None,
            selector=bundle["selector"] if bundle else None,
            is_hybrid=bundle["is_hybrid"] if bundle else False,
            calibrator=bundle["calibrator"] if bundle else None,
        )
        classification = result.classification
        bls = result.bls
        mono = result.mono.best if result.mono and result.mono.detected else None
        calibrated = bool(
            bundle
            and bundle.get("calibrator") is not None
            and getattr(bundle["calibrator"], "is_fitted", False)
        )
        base.update(
            {
                "status": "ok" if result.quality.passed else "rejected",
                "target_id": result.metadata.get("tic_id", path.stem),
                "quality_score": result.quality.score,
                "candidate_type": result.metadata.get("candidate_type", "none"),
                "predicted_class": classification.class_name if classification else "unavailable",
                "reported_probability": result.final_confidence if classification else "",
                "is_calibrated": calibrated,
                "class_probabilities_json": json.dumps(
                    classification.probabilities if classification else {}, sort_keys=True
                ),
                "bls_period_days": bls.best_period if bls else "",
                "bls_duration_days": bls.best_duration if bls else "",
                "bls_depth": bls.best_depth if bls else "",
                "bls_snr": bls.depth_snr if bls else "",
                "bls_sde": bls.sde if bls else "",
                "bls_sampled_transits": bls.n_transits if bls else "",
                "mono_epoch": mono.t0 if mono else "",
                "mono_duration_days": mono.duration if mono else "",
                "mono_depth": mono.depth if mono else "",
                "mono_snr": mono.snr if mono else "",
                "warnings_json": json.dumps(result.warnings),
            }
        )
    except Exception as exc:  # per-file failures are structured and do not stop the batch
        base["error"] = str(exc)
    base["runtime_seconds"] = time.perf_counter() - started
    return base


def _collect_files(paths: list[str], input_directory: str | None) -> list[Path]:
    supported = {".fits", ".fit", ".csv", ".hdf5", ".h5"}
    collected = [Path(path) for path in paths]
    if input_directory:
        collected.extend(
            path
            for path in Path(input_directory).rglob("*")
            if path.is_file() and path.suffix.lower() in supported
        )
    invalid = [path for path in collected if not path.is_file() or path.suffix.lower() not in supported]
    if invalid:
        raise ValueError(f"Unsupported or missing inputs: {[str(path) for path in invalid[:10]]}")
    return sorted(set(path.resolve() for path in collected))


if __name__ == "__main__":
    main()
