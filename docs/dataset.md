# Dataset Guide

Use original TESS mission FITS products where possible. The downloader selects
deterministically from SPOC, TESS-SPOC, then QLP products and preserves headers,
quality flags, cadence errors, contamination metadata, and stellar metadata.

Place files under `data/raw/<class_alias>/`. The canonical classes are
`planet`, `planet_candidate`, `eclipsing_binary`, `background_blend`,
`instrument_artefact`, `variable_star`, `quiet_star`, and `unknown`. Labels
must come from a catalog or documented review. The broad `unknown` class is
excluded from supervised training; do not treat an arbitrary target as quiet.

```powershell
python scripts/build_dataset.py --config configs/config.yaml
python check_dataset.py --config configs/config.yaml
```

The builder records source identity, configuration hash, dataset version,
features, masks, resampled flux cache, and processing status. Splits are grouped
by host identifier. Conflicting host labels are rejected and augmented rows are
limited to training. The checker must pass before training.

For a credible benchmark, collect at least 20 independent hosts in every class
you intend to fit; hundreds are preferable. Preserve catalog provenance and
deduplicate targets across sectors and aliases. Evaluate on untouched hosts,
then run transit injection-recovery and an external catalog holdout before
making sensitivity or completeness claims.
