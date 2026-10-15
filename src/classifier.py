"""Multi-class astrophysical classifier with transit gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger

logger = get_logger()


@dataclass
class ClassificationResult:
    """Classifier output with probabilities and metadata."""

    class_id: int
    class_name: str
    probabilities: dict[str, float]
    entropy: float
    transit_like: bool
    transit_gate_prob: float
    top_classes: list[tuple[str, float]]
    probability_margin: float = 0.0
    normalized_entropy: float = 0.0
    abstained: bool = False
    abstention_reason: str | None = None
    model_top_class: str | None = None


class AstrophysicalClassifier:
    """Wrap sklearn/torch models for multi-class astrophysical classification."""

    ASTRO_CLASSES = [
        "planet",
        "planet_candidate",
        "eclipsing_binary",
        "background_blend",
        "instrument_artefact",
        "variable_star",
        "quiet_star",
        "unknown",
    ]

    FP_DIAGNOSES = [
        "eclipsing_binary",
        "background_blend",
        "instrument_noise",
        "momentum_dump",
        "scattered_light",
        "starspot",
        "variable_star",
        "cosmic_ray",
        "unknown",
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        self.class_names = config.get("classes", {}).get(
            "astro_names", self.ASTRO_CLASSES
        )
        self.transit_threshold = config.get("inference", {}).get("planet_threshold", 0.5)
        self.abstain_probability = float(
            config.get("inference", {}).get("abstain_probability_threshold", 0.55)
        )
        self.abstain_entropy = float(
            config.get("inference", {}).get("abstain_entropy_threshold", 0.90)
        )

    def classify_from_proba(self, proba: np.ndarray) -> ClassificationResult:
        """Build result from probability vector."""
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 2:
            if len(proba) != 1:
                raise ValueError("classify_from_proba accepts one sample")
            proba = proba[0]
        if proba.ndim != 1 or len(proba) != len(self.class_names):
            raise ValueError(
                f"Probability vector has {len(proba)} classes; expected {len(self.class_names)}"
            )
        if np.any(~np.isfinite(proba)) or np.any(proba < 0):
            raise ValueError("Probabilities must be finite and non-negative")
        total = float(np.sum(proba))
        if total <= 0:
            raise ValueError("Probability vector sums to zero")
        proba = proba / total
        n = len(proba)
        names = self.class_names
        idx = int(np.argmax(proba))
        probs = {names[i]: float(proba[i]) for i in range(n)}
        entropy = float(-np.sum(proba * np.log(proba + 1e-12)))
        sorted_cls = sorted(probs.items(), key=lambda x: -x[1])
        planet_prob = probs.get("planet", 0.0) + probs.get("planet_candidate", 0.0)
        margin = sorted_cls[0][1] - sorted_cls[1][1] if len(sorted_cls) > 1 else sorted_cls[0][1]
        normalized_entropy = entropy / np.log(n) if n > 1 else 0.0
        abstention_reasons = []
        if proba[idx] < self.abstain_probability:
            abstention_reasons.append("top probability below abstention threshold")
        if normalized_entropy > self.abstain_entropy:
            abstention_reasons.append("predictive entropy above abstention threshold")
        abstained = bool(abstention_reasons)
        reported_name = "unknown" if abstained else names[idx]
        return ClassificationResult(
            class_id=idx,
            class_name=reported_name,
            probabilities=probs,
            entropy=entropy,
            transit_like=(not abstained and planet_prob >= self.transit_threshold),
            transit_gate_prob=float(planet_prob),
            top_classes=sorted_cls[:3],
            probability_margin=float(margin),
            normalized_entropy=float(normalized_entropy),
            abstained=abstained,
            abstention_reason="; ".join(abstention_reasons) or None,
            model_top_class=names[idx],
        )

    def diagnose_false_positive(self, result: ClassificationResult) -> dict:
        """Suggest false-positive mechanism when not a planet."""
        if result.class_name in ("planet", "planet_candidate") and not result.abstained:
            return {"primary": None, "confidence": 0.0, "evidence": []}
        mapping = {
            "eclipsing_binary": "eclipsing_binary",
            "background_blend": "background_blend",
            "instrument_artefact": "instrument_noise",
            "variable_star": "variable_star",
            "quiet_star": "unknown",
        }
        follow_up = {
            "eclipsing_binary": "inspect odd-even depths and secondary eclipse",
            "background_blend": "check centroid motion and high-resolution imaging",
            "instrument_noise": "inspect quality flags and spacecraft events",
            "variable_star": "compare against stellar rotation and activity indicators",
            "unknown": "collect more photometry or independent validation",
        }
        primary = mapping.get(result.class_name, "unknown")
        return {
            "primary": primary,
            "secondary": result.top_classes[1][0] if len(result.top_classes) > 1 else None,
            "confidence": result.probabilities.get(result.class_name, 0.0),
            "evidence": [
                f"top class: {result.class_name}",
                f"class entropy: {result.entropy:.3f}",
            ],
            "suggested_follow_up": follow_up.get(primary, follow_up["unknown"]),
        }
