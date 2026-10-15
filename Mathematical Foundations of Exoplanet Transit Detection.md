# FORMULAS.md — Mathematical Foundations of Exoplanet Transit Detection

This document derives and explains every formula used in the `exoplanet_ml` pipeline.
Symbols: `F` = stellar flux, `P` = orbital period, `R_p` = planet radius,
`R_s` = stellar radius, `a` = semi-major axis, `G` = gravitational constant.

---

## 1. Transit Depth

When a planet passes in front of its host star, it blocks a fraction of the
stellar disk, producing a dip in observed flux.

**Definition (observed):**

    δ = ΔF / F = (F_out − F_in) / F_out

where `F_out` is the median out-of-transit flux and `F_in` the in-transit flux.

**Geometric approximation** (uniform stellar disk, opaque planet):

    δ ≈ (R_p / R_s)²

**Physical meaning:** the depth equals the ratio of the planet's projected
area to the stellar disk area, `(π R_p²)/(π R_s²)`. A Jupiter transiting a
Sun-like star gives δ ≈ 0.01 (1%); an Earth gives δ ≈ 8.4×10⁻⁵.

---

## 2. Signal-to-Noise Ratio

    SNR = δ / σ

with the sample standard deviation of normalized flux:

    σ = sqrt( (1/(N−1)) Σ (F_i − F̄)² )

For a binned transit averaged over `n_in` in-transit points, the effective
per-transit SNR scales as `δ √(n_in) / σ_point`. High SNR (> ~7) is the
classic threshold for a credible TESS detection.

---

## 3. Phase Folding

To stack repeated transits we map every timestamp into a single orbital cycle:

    φ = ((t − t_0) / P) mod 1

- `P`  = orbital period
- `t_0` = reference epoch (a known transit center)

Phases near 0 (or equivalently shifted to be centered) correspond to transit.

---

## 4. Transit Duration

For a central transit (impact parameter b = 0) of a planet on a circular orbit,
the chord across the star has length `2 R_s`, traversed at orbital velocity
`v = 2πa / P`. The duration is:

    T = 2 R_s / v = (P R_s) / (π a)        (approximation)

Including R_p and impact parameter b:

    T = (P / π) · arcsin( (R_s / a) · sqrt((1+R_p/R_s)² − b²) / sinθ )

The simplified form `T ≈ P R_s / (π a)` is used for feature sanity checks.

---

## 5. Kepler's Third Law

    P² = (4π² a³) / (G (M_s + M_p))

Lets us relate the measured period `P` to the semi-major axis `a` given the
stellar mass. Since `M_p ≪ M_s`, we often use `a³ ≈ G M_s P² / (4π²)`.

---

## 6. Binary Cross Entropy (classification loss)

    L = −(1/N) Σ [ y log(p) + (1−y) log(1−p) ]

`y ∈ {0,1}` is the true label, `p ∈ (0,1)` the predicted planet probability.

---

## 7. Classification Metrics

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · (Precision · Recall) / (Precision + Recall)

- **Precision**: of all flagged candidates, how many are real planets.
- **Recall (completeness)**: of all real planets, how many we recovered.
- **F1**: harmonic mean balancing the two.

---

## 8. ROC-AUC

The ROC curve plots True Positive Rate (= Recall) vs False Positive Rate
(FPR = FP/(FP+TN)) across all decision thresholds. The Area Under the Curve:

    AUC = ∫₀¹ TPR(FPR) d(FPR)

equals the probability that a randomly chosen positive is ranked above a
randomly chosen negative. AUC = 0.5 is random; AUC = 1.0 is perfect.

---

## 9. Higher-Order Flux Statistics

**Skewness** (asymmetry): `γ₁ = E[(F−μ)³] / σ³`
**Kurtosis** (tailedness): `γ₂ = E[(F−μ)⁴] / σ⁴ − 3`
**MAD** (robust scatter): `median(|F − median(F)|)`

---

## 10. Odd–Even Depth Difference

Eclipsing binaries often show alternating deep/shallow eclipses. Comparing the
mean depth of odd-numbered vs even-numbered transits:

    Δ_oe = | δ_odd − δ_even |

Large `Δ_oe` flags a likely false positive (binary, not planet).

---

## References
- Kovács, Zucker & Mazeh (2002), *BLS*, A&A 391, 369.
- Seager & Mallén-Ornelas (2003), *Transit light curves*, ApJ 585, 1038.
- Winn (2010), *Transits and Occultations*, arXiv:1001.2010.
- Ricker et al. (2015), *TESS*, JATIS 1, 014003.
