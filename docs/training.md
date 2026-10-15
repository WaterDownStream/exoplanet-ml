# Training Guide

1. Build the versioned dataset and require `check_dataset.py` to pass.
2. Train the default random-forest baseline with `python train.py --config configs/config.yaml`.
3. Inspect macro F1, per-class recall, positive-class PR metrics, NLL, Brier score,
   calibration error, reliability, confusion matrix, and feature importance.
4. Run group-safe HPO without touching the final test hosts.
5. Evaluate the chosen configuration once on the fixed test split.

```powershell
python scripts/hyperparameter_search.py --config configs/config.yaml --strategy optuna --max-trials 24
python scripts/evaluate.py --config configs/config.yaml --test data/processed/features/test.csv
python scripts/benchmark.py --config configs/config.yaml
```

Imputation, feature selection, and scaling are fit only on training data.
Calibration and the planet/candidate operating threshold use validation data.
The test set is evaluation-only. Class IDs are remapped explicitly, class
weights are capped, and insufficient class support stops a real training run.

Use `hybrid_tcn` only with complete cached real flux and enough independent
hosts to constrain it. CUDA mixed precision, early stopping, gradient clipping,
and checkpoint signatures are supported. Prefer the simpler model unless the
hybrid improves fixed-test class-wise reliability and calibration, not merely
accuracy.
