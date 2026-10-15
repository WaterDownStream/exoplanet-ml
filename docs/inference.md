# Inference Guide

```powershell
python inference.py path\to\lightcurve.fits --config configs/config.yaml
python scripts/batch_inference.py --input-dir path\to\folder --workers 4
```

Inference loads only artifact schema v2, applies the saved imputer, selected
feature schema, scaler, class mapping, calibrator, and validation-derived
operating threshold. Neural models use CUDA automatically when available;
CUDA batch inference is restricted to one worker to avoid model contention.

The pipeline emits quality diagnostics, periodic and mono candidates,
calibrated class probabilities, entropy, margin, epistemic spread where
available, abstention state, false-positive reasoning, and qualified physical
parameters. Confidence is model probability, not a quality-weighted score.

No model is required for candidate-generation mode, but that output is a signal
search result, not a classification. Poor quality, absent stellar metadata,
unsupported diagnostics, and weak parameter assumptions remain explicit in the
JSON report rather than being silently imputed as physical evidence.
