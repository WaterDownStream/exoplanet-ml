# Project Layout And Execution Plan

This is the practical path from the current codebase to a credible final
submission. Follow it in order; do not skip verification gates.

## 1. Repository Layout

```text
configs/
  config.yaml                  Main configuration for all tunable behavior

data/
  raw/                         Input light curves grouped by class alias
  catalogs/                    Target catalogs and confirmed-planet lists
  cache/                       Per-target feature/flux caches
  processed/features/          Master dataset and train/val/test CSVs
  processed/figures/           Evaluation and diagnostic figures
  models/                      Trained models, scaler, calibrator, metadata
  reports/                     Per-target prediction reports
  logs/                        Processing and download logs
  archive/                     Cold storage area

src/
  data_loader.py               FITS/CSV/HDF5 light-curve loading
  quality.py                   Observation quality assessment
  preprocessing.py             NaN removal, sorting, clipping, normalization
  detrending.py                Savitzky-Golay, median, GP/PCA detrending
  transit_search.py            BLS/TLS-compatible periodic search
  mono_search.py               Single-transit candidate search
  phase_folding.py             Fold and bin periodic candidates
  feature_extraction.py        100 engineered features + missing masks
  mono_features.py             Mono-transit morphology/uncertainty helpers
  classifier.py                Multi-class class/probability interpretation
  calibration.py               Isotonic/Platt/temperature calibration
  parameter_estimation.py      Physical parameter and uncertainty estimates
  pipeline.py                  End-to-end inference orchestration
  trainer.py                   Reproducible training/checkpointing
  evaluator.py                 Metrics and feature importance
  visualizer.py                Scientific diagnostic figures
  reporting.py                 JSON/CSV/TXT/HTML/PDF reports
  datasets/                    Download, build, split, cache, augment, balance

scripts/
  build_dataset.py             Build features from data/raw
  evaluate.py                  Evaluate saved models on feature tables
  benchmark.py                 Fixed-split model-family comparison
  batch_inference.py           Memory-bounded collection inference
  hyperparameter_search.py     Random/grid/Optuna hyperparameter search

docs/
  dataset.md                   Dataset workflow
  training.md                  Training workflow
  inference.md                 Inference workflow
  project_layout.md            This execution plan
```

## 2. Environment Setup

Create a clean Python 3.11+ environment and install the full stack.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verification gate:

```bash
python -m pytest
python -m compileall -q src scripts tests train.py inference.py
```

Do not start final training until imports and tests pass in this real
environment.

## 3. Data Layout

Put light curves under class-alias folders. Folder names are mapped through
`dataset.label_aliases` in `configs/config.yaml`.

```text
data/raw/
  planets/
  planet_candidates/
  eclipsing_binaries/
  background_blends/
  variable_stars/
  instrument_artefact/
  quiet_stars/
  nonplanets/                  Unknown provenance; excluded from fitting
```

Minimum target for a serious run:

- Hundreds to thousands of planet/candidate examples
- Comparable false-positive and nonplanet coverage
- At least several categories beyond binary planet/nonplanet
- FITS files from TESS SPOC/QLP where possible

Verification gate:

```bash
python check_dataset.py --config configs/config.yaml
```

## 4. Build The Dataset

Build deterministic features, masks, metadata, caches, and train/val/test CSVs.

```bash
python scripts/build_dataset.py --config configs/config.yaml
```

Expected outputs:

```text
data/processed/features/master_dataset.csv
data/processed/features/train.csv
data/processed/features/val.csv
data/processed/features/test.csv
data/cache/*.npz
data/cache/*.json
data/logs/*
```

Verification gate:

```bash
python check_dataset.py --config configs/config.yaml
```

Confirm:

- Every row has a valid `label`
- Feature count is stable
- `mask__*` columns exist
- `cache_id` exists for hybrid TCN training
- Rejected files have reasons in logs

## 5. First Reliable Training Run

Start with a robust feature-only model before tuning the hybrid model.

In `configs/config.yaml`, temporarily set:

```yaml
model:
  type: random_forest
```

Then run:

```bash
python train.py --config configs/config.yaml
```

Expected outputs:

```text
data/models/random_forest.joblib
data/models/scaler.joblib
data/models/calibrator.joblib
data/models/model_meta.joblib
data/models/training_run.json
data/models/evaluation_metrics.json
data/processed/figures/confusion_matrix.png
data/processed/figures/reliability_curve.png
data/processed/figures/confidence_histogram.png
data/processed/figures/feature_importance.png
```

Verification gate:

- Calibration file exists
- Confusion matrix covers all expected classes
- Reliability curve is not pathological
- Macro F1 and per-class recall are recorded
- False-positive classes are not collapsed into one bucket unless the data truly
  lacks labels

## 6. Hyperparameter Search

Run HPO after the first reliable baseline.

```bash
python scripts/hyperparameter_search.py --config configs/config.yaml --strategy random --max-trials 12
python scripts/hyperparameter_search.py --config configs/config.yaml --strategy optuna --max-trials 24
```

Expected output:

```text
data/models/hpo_results.json
data/models/hpo/trial_*/
```

Verification gate:

- Failed trials are explained, not silent
- Best trial improves the chosen metric
- Runtime and GPU memory are recorded
- Final selected config is copied back into `configs/config.yaml`

## 7. Hybrid TCN Training

Only use the hybrid model when cached real flux exists.

Set:

```yaml
model:
  type: hybrid_tcn
```

Then run:

```bash
python train.py --config configs/config.yaml
```

Verification gate:

- Training uses `data/cache/*` flux arrays
- Mixed precision is active on CUDA
- Checkpoints resume correctly
- Hybrid metrics beat or match the feature-only baseline
- VRAM stays below the configured budget

If cached flux is incomplete, stay with the feature-only model. Do not invent
raw flux for the hybrid branch.

## 8. Evaluation And Benchmarking

Evaluate the selected model:

```bash
python scripts/evaluate.py --config configs/config.yaml --test data/processed/features/test.csv
python scripts/benchmark.py --config configs/config.yaml
```

Required evidence:

- Accuracy, precision, recall, F1
- Macro and weighted metrics
- Per-class precision/recall
- Top-2 and top-3 accuracy
- ROC-AUC/PR-AUC where meaningful
- ECE, MCE, Brier score
- Reliability curve
- Confidence histogram
- Confusion matrix
- Feature importance or SHAP/permutation fallback

## 9. Inference And Reports

Run on representative examples from every class.

```bash
python inference.py data/raw/planets/example.fits --config configs/config.yaml
python inference.py data/raw/nonplanets/example.fits --config configs/config.yaml
```

Expected report folder:

```text
data/reports/<target>_<timestamp>/
  prediction.json
  prediction.csv
  summary.txt
  report.html
  report.pdf
  raw_light_curve.png
  cleaned_light_curve.png
  detrended_light_curve.png
  periodogram.png
  phase_fold.png
  residuals.png
```

Verification gate:

- Report includes all class probabilities
- Confidence is calibrated model probability; quality and uncertainty stay separate
- False-positive diagnosis includes evidence and follow-up
- Parameter estimates mark unavailable quantities honestly
- Explanation block exists in JSON
- Warnings are present when assumptions are weak

## 10. Scientific Sanity Checks

Before calling the project final, inspect at least 20 predictions manually:

- Confirmed planets
- Planet candidates
- Eclipsing binaries
- Quiet/nonplanet stars
- Variable stars
- Low-quality light curves
- Mono-transit-like examples

For each one, check:

- The light curve plot matches the predicted class
- BLS period is plausible
- Mono-transit flag is reasonable
- Depth, duration, and epoch are physically plausible
- Radius and semi-major axis are only present when stellar metadata exists
- Impact parameter is available only when geometry supports it
- Confidence drops for noisy or ambiguous cases

## 11. Final Submission Layout

The final project should contain:

```text
configs/config.yaml
requirements.txt
README.md
docs/
src/
scripts/
tests/
data/models/
data/processed/figures/
data/reports/sample_reports/
```

Do not include:

- `__pycache__/`
- `.pytest_cache/`
- Accidental installers or executables
- Temporary Office lock files
- Large raw datasets unless the submission rules allow them

## 12. Final Acceptance Checklist

The project is ready only when all are true:

- `python -m pytest` passes
- `python -m compileall -q src scripts tests train.py inference.py` passes
- Dataset builder produces `master_dataset.csv`
- Training produces model, scaler, metadata, calibrator, metrics, and figures
- HPO results are recorded
- Inference produces JSON, CSV, TXT, HTML, and PDF reports
- At least one report is generated for each major class
- Evaluation metrics are documented
- Known limitations are stated honestly
- No generated junk files remain in the repo

## 13. What To Do Next

1. Create the full Python environment and install `requirements.txt`.
2. Run `python -m pytest`.
3. Build the dataset from the raw FITS files.
4. Train a random forest baseline.
5. Run HPO.
6. Train the hybrid TCN only if cached flux coverage is complete.
7. Evaluate and benchmark.
8. Generate sample reports.
9. Inspect reports manually.
10. Package only clean code, docs, configs, trained artifacts, metrics, and sample
    reports.
