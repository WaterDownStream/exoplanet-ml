# Scientific And Engineering Audit

## Implemented Improvements

| Area | Change and scientific reason | Computational impact | Accuracy expectation and drawback |
|---|---|---|---|
| Periodic search | Astropy BLS with an S/N objective, frequency-uniform grid, period-valid durations, depth uncertainty, SDE, sampled-event count, and local period uncertainty. | Period-banded evaluation reduced one real-file run from about 7.9 s to 2.8 s. | Fewer cadence and long-duration aliases; a box template remains less sensitive than TLS to shallow limb-darkened transits. |
| Mono search | Vectorized multi-duration matched filter with empirical same-duration noise, local baselines, coverage checks, morphology, and event suppression. | Removes the previous quadratic point-window loops and scales approximately with durations times cadences. | Rejects isolated bad cadences and reports red-noise-aware S/N; correlated stellar dips can still imitate a transit. |
| Detrending | Gap-aware, day-scale, iterative transit-masked Savitzky-Golay or median trends. | Linear per segment for default methods; GP fitting is capped. | Regression tests preserve injected depth within 20%; any detrending can attenuate events near gaps or at trend timescales. |
| Features | Unsupported physics stays missing with masks; event-depth, timing, secondary-eclipse, morphology, and gap-safe spectral features use measured data. | Fixed 100-feature contract; train-only selector reduces redundant model input. | Better interpretability and less fake evidence; fixed engineered features cannot replace pixel diagnostics or learned views at large scale. |
| Labels and splits | Host-grouped stratification, conflicting-label rejection, deterministic augmentation, and no augmented validation/test rows. `unknown` is excluded from supervised fitting. | Small grouping overhead; caches avoid repeated search. | Removes severe target leakage and invalid negative labels; current usable sample count becomes smaller. |
| Training | Train-only imputation, feature selection, and scaling; contiguous active-class mapping; capped imbalance weights; focal/CE neural losses; deterministic checkpoints. | RF is the efficient default; hybrid TCN supports mixed precision on an 8 GB GPU. | Metrics become trustworthy rather than inflated. No numerical gain is claimed until a curated external benchmark exists. |
| Calibration | Validation-only temperature, Platt, or support-gated isotonic calibration; no fitting in evaluation. | Negligible for temperature scaling. | Improves probability meaning when validation support is adequate; calibration is disabled rather than fabricated on tiny classes. |
| Decisions | Validation-derived planet/candidate threshold plus low-probability and high-entropy abstention. | Negligible inference cost. | Allows recall-oriented triage without presenting ambiguity as certainty; increases the number of unclassified targets. |
| Physics | Exact circular transit-duration impact inversion, propagated radius/orbit/temperature uncertainties, and prior-qualified mono-period Monte Carlo. | Mono posterior sampling is bounded and deterministic. | More defensible intervals; mono periods remain prior-dependent and circular periodic estimates remain model assumptions. |
| Explainability | Global permutation/tree importance, local feature contributions, temporal saliency, probability margin, entropy, and ensemble/dropout spread. | Optional explanation work adds latency outside batch mode. | Exposes evidence and uncertainty; explanations are model diagnostics, not causal proof. |
| Operations | Versioned artifacts, strict configuration/data checks, atomic batch output, bounded workers, deterministic downloads, and structured failures. | Prevents repeated or corrupt work with minor metadata cost. | Reduces silent misuse; obsolete datasets and models must be rebuilt. |

## Current Data Verdict

The existing feature table is not suitable for a scientific training claim. It
contains 259 rows, approximately 253 under one legacy label and 6 under another,
and uses the obsolete `1.0.0` feature schema. The old 97.4% accuracy with about
0.494 macro F1 is a class-imbalance symptom, not evidence of a strong detector.
The current checker intentionally rejects this table and the old model artifacts.

## Remaining Scientific Boundaries

- No pixel-level centroid, aperture, difference-image, nearby-source, or sector
  consistency branch exists. Consequently `background_blend` cannot yet be
  validated at the standard of ExoMiner or the SPOC data-validation products.
- BLS and the optional TLS path are candidate generators. A detection is not a
  validated planet and no statistical bootstrap false-alarm probability is
  fabricated when it has not been computed.
- Mono-period inference uses explicit stellar, impact, and eccentricity priors;
  it is not a complete Bayesian selection-function model.
- Old locally generated generic FITS files lack stellar and contamination
  headers, so derived radii and orbital parameters should remain unavailable.
- Survey sensitivity, occurrence rates, and "best detector" claims require
  mission-scale injection-recovery, catalog provenance, cross-sector deduplication,
  and an external host-level holdout.
- Domain shift across SPOC, QLP, cadence, magnitude, sector, and stellar type
  must be measured before operational deployment.

## Required Validation Campaign

1. Assemble catalog-linked planets, candidates, eclipsing binaries, variables,
   instrumental failures, blends, and verified quiet stars with host IDs.
2. Rebuild schema `2.0.0`, pass the strict checker, and freeze host-level splits.
3. Establish RF and XGBoost baselines; tune only on grouped training/validation.
4. Measure fixed-test class-wise precision-recall and calibrated decision curves.
5. Inject transits across radius ratio, period, duration, magnitude, variability,
   gaps, sectors, and crowding; report recovery and false-alarm surfaces.
6. Test an external catalog and manually review high-confidence errors.
7. Add pixel-level vetting before promoting blend probabilities as validation evidence.

## Hardware Fit

The default RF fits comfortably within 32 GB RAM and runs inference on CPU. The
hybrid TCN uses a 1024-point view, compact dilated residual blocks, batch size 32,
mixed precision, and a 6 GB configured VRAM budget, fitting an RTX 4060 8 GB.
Feature caches belong on SSD during active experiments; raw/archive products can
reside on the 2 TB HDD. Streaming batch inference avoids loading tens of
thousands of light curves simultaneously.
