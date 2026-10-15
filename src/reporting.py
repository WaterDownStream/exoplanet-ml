"""Automatic scientific report generation."""
from __future__ import annotations

import csv
import json
from html import escape
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from .utils import ensure_dirs, get_logger

logger = get_logger()


def _serialize(obj):
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


class ReportGenerator:
    """Generate JSON, CSV, text, and HTML scientific reports."""

    CONFIDENCE_LABELS = [
        (0.99, "Very High"),
        (0.95, "High"),
        (0.80, "Moderate"),
        (0.60, "Low"),
        (0.0, "Very Low"),
    ]

    def __init__(self, config: dict) -> None:
        self.config = config
        rep = config.get("reporting", {})
        self.write_json = rep.get("json", True)
        self.write_csv = rep.get("csv", True)
        self.write_html = rep.get("html", True)
        self.write_pdf = rep.get("pdf", False)
        self.reports_dir = Path(config["paths"].get("reports_dir", "data/reports"))
        ensure_dirs(self.reports_dir)

    def generate(
        self,
        target_id: str,
        payload: dict,
        figures: dict[str, Path] | None = None,
    ) -> dict[str, Path]:
        """Write prediction report files."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = self.reports_dir / f"{target_id}_{ts}"
        ensure_dirs(out_dir)

        conf = payload.get("final_confidence", payload.get("confidence", 0.0))
        payload["confidence_level"] = self._confidence_label(
            conf, bool(payload.get("is_calibrated", False))
        )
        payload["generated_at"] = ts
        payload["processing_version"] = self.config.get("dataset", {}).get("version", "1.0.0")
        staged_figures = self._stage_figures(figures or {}, out_dir)

        paths: dict[str, Path] = {}
        if self.write_json:
            jp = out_dir / "prediction.json"
            jp.write_text(json.dumps(_serialize(payload), indent=2), encoding="utf-8")
            paths["json"] = jp

        summary = self._summary_text(payload)
        sp = out_dir / "summary.txt"
        sp.write_text(summary, encoding="utf-8")
        paths["summary"] = sp

        if self.write_csv:
            cp = out_dir / "prediction.csv"
            self._write_csv(cp, payload)
            paths["csv"] = cp

        if self.write_html:
            hp = out_dir / "report.html"
            hp.write_text(self._html_report(payload, staged_figures), encoding="utf-8")
            paths["html"] = hp

        if self.write_pdf:
            pp = out_dir / "report.pdf"
            if self._write_pdf(pp, payload, staged_figures):
                paths["pdf"] = pp

        logger.info("Report written to %s", out_dir)
        return paths

    @staticmethod
    def _stage_figures(figures: dict[str, Path], out_dir: Path) -> dict[str, Path]:
        staged: dict[str, Path] = {}
        for name, path in figures.items():
            src = Path(path)
            if not src.exists():
                continue
            dest = out_dir / src.name
            if src.resolve() != dest.resolve():
                copy2(src, dest)
            staged[name] = dest
        return staged

    def _confidence_label(self, prob: float, calibrated: bool = True) -> str:
        if not calibrated:
            return "Uncalibrated"
        for threshold, label in self.CONFIDENCE_LABELS:
            if prob >= threshold:
                return label
        return "Very Low"

    @staticmethod
    def _summary_text(payload: dict) -> str:
        lines = [
            "Exoplanet ML Detection Report",
            "=" * 40,
            f"Target: {payload.get('target_id', 'unknown')}",
            f"Class: {payload.get('predicted_class', 'unknown')}",
            f"Confidence: {payload.get('final_confidence', 0):.1%} ({payload.get('confidence_level', '')})",
        ]
        if params := payload.get("parameters"):
            lines.append("\nParameters:")
            for key, value in params.items():
                if not isinstance(value, dict):
                    continue
                if value.get("available", True) and value.get("value") is not None:
                    uncertainty = value.get("uncertainty", "N/A")
                    unit = value.get("unit", "")
                    lines.append(f"  {key}: {value['value']} +/- {uncertainty} {unit}")
                elif not value.get("available", True):
                    lines.append(f"  {key}: Unavailable")
        if alternatives := payload.get("alternative_classes"):
            lines.append("\nAlternative classes:")
            for name, prob in alternatives:
                lines.append(f"  {name}: {prob:.1%}")
        if fp := payload.get("false_positive_diagnosis"):
            if fp.get("primary"):
                lines.append("\nFalse-positive diagnosis:")
                lines.append(f"  Primary: {fp.get('primary')}")
                if fp.get("secondary"):
                    lines.append(f"  Secondary: {fp.get('secondary')}")
                lines.append(f"  Confidence: {fp.get('confidence', 0):.1%}")
                if fp.get("suggested_follow_up"):
                    lines.append(f"  Suggested follow-up: {fp.get('suggested_follow_up')}")
        if top_features := payload.get("top_feature_values"):
            lines.append("\nTop feature values:")
            for name, value in top_features[:10]:
                lines.append(f"  {name}: {value:.4g}")
        if explanation := payload.get("explanation"):
            if confidence := explanation.get("confidence_explanation"):
                lines.append("\nConfidence explanation:")
                for key, value in confidence.items():
                    if ReportGenerator._is_scalar(value):
                        lines.append(f"  {key}: {value}")
        if warnings := payload.get("warnings"):
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    @staticmethod
    def _write_csv(path: Path, payload: dict) -> None:
        row = ReportGenerator._csv_row(payload)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _csv_row(payload: dict) -> dict[str, object]:
        row: dict[str, object] = {
            "target_id": payload.get("target_id", ""),
            "predicted_class": payload.get("predicted_class", ""),
            "final_confidence": payload.get("final_confidence", payload.get("confidence", "")),
            "confidence_level": payload.get("confidence_level", ""),
            "quality_score": payload.get("quality_score", ""),
            "is_mono": payload.get("is_mono", ""),
            "generated_at": payload.get("generated_at", ""),
            "processing_version": payload.get("processing_version", ""),
        }

        for name, value in (payload.get("class_probabilities") or {}).items():
            row[f"prob_{ReportGenerator._column_key(name)}"] = value

        for idx, item in enumerate(payload.get("alternative_classes") or [], start=1):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            row[f"alternative_{idx}_class"] = item[0]
            row[f"alternative_{idx}_probability"] = item[1]

        for name, value in (payload.get("parameters") or {}).items():
            if not isinstance(value, dict):
                continue
            prefix = f"param_{ReportGenerator._column_key(name)}"
            row[f"{prefix}_available"] = value.get("available", True)
            row[f"{prefix}_value"] = value.get("value", "")
            row[f"{prefix}_uncertainty"] = value.get("uncertainty", "")
            row[f"{prefix}_unit"] = value.get("unit", "")
            row[f"{prefix}_method"] = value.get("method", "")

        fp = payload.get("false_positive_diagnosis") or {}
        row.update({
            "fp_primary": fp.get("primary", ""),
            "fp_secondary": fp.get("secondary", ""),
            "fp_confidence": fp.get("confidence", ""),
            "fp_suggested_follow_up": fp.get("suggested_follow_up", ""),
        })

        for name, value in (payload.get("quality_metrics") or {}).items():
            if ReportGenerator._is_scalar(value):
                row[f"quality_{ReportGenerator._column_key(name)}"] = value

        for name, value in (payload.get("processing_metadata") or {}).items():
            if ReportGenerator._is_scalar(value):
                row[f"metadata_{ReportGenerator._column_key(name)}"] = value

        for idx, item in enumerate((payload.get("top_feature_values") or [])[:10], start=1):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            row[f"top_feature_{idx}_name"] = item[0]
            row[f"top_feature_{idx}_value"] = item[1]

        warnings = payload.get("warnings") or []
        row["warning_count"] = len(warnings)
        row["warnings"] = "; ".join(str(w) for w in warnings)
        return {key: ReportGenerator._csv_value(value) for key, value in row.items()}

    @staticmethod
    def _column_key(value: object) -> str:
        text = str(value).strip().lower()
        chars = [ch if ch.isalnum() else "_" for ch in text]
        key = "_".join("".join(chars).split("_"))
        return key or "value"

    @staticmethod
    def _is_scalar(value: object) -> bool:
        return isinstance(value, (float, int, str, bool)) or value is None

    @staticmethod
    def _csv_value(value: object) -> object:
        if ReportGenerator._is_scalar(value):
            return "" if value is None else value
        return json.dumps(_serialize(value), sort_keys=True)

    @staticmethod
    def _write_pdf(path: Path, payload: dict, figures: dict[str, Path]) -> bool:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError as exc:
            logger.warning("PDF report skipped; matplotlib is unavailable: %s", exc)
            return False

        lines = ReportGenerator._summary_text(payload).splitlines()
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                "\n".join(lines[:55]),
                ha="left",
                va="top",
                family="monospace",
                fontsize=9,
                transform=ax.transAxes,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            for name, fig_path in figures.items():
                src = Path(fig_path)
                if not src.exists():
                    continue
                img = plt.imread(src)
                fig = plt.figure(figsize=(8.27, 6.0))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.set_title(name.replace("_", " ").title())
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        return True

    @staticmethod
    def _html_report(payload: dict, figures: dict[str, Path]) -> str:
        fig_links = "".join(
            f'<li><a href="{escape(p.name, quote=True)}">{escape(str(name))}</a></li>'
            for name, p in figures.items()
        )
        probs = payload.get("class_probabilities", {})
        prob_rows = "".join(
            f"<tr><td>{escape(str(k))}</td><td>{v:.1%}</td></tr>" for k, v in probs.items()
        )
        alternatives = payload.get("alternative_classes", [])
        alt_items = "".join(
            f"<li>{escape(str(name))}: {prob:.1%}</li>" for name, prob in alternatives
        )
        top_features = payload.get("top_feature_values", [])
        feature_items = "".join(
            f"<li>{escape(str(name))}: {value:.4g}</li>" for name, value in top_features
        )
        warnings = payload.get("warnings", [])
        warning_items = "".join(f"<li>{escape(str(w))}</li>" for w in warnings)
        explanation = payload.get("explanation") or {}
        confidence = explanation.get("confidence_explanation") or {}
        confidence_items = "".join(
            f"<li>{escape(str(k))}: {escape(str(v))}</li>" for k, v in confidence.items()
        )
        posterior = explanation.get("posterior_summary") or {}
        posterior_items = "".join(
            f"<li>{escape(str(k))}: {escape(str(v))}</li>" for k, v in posterior.items()
        )
        fp = payload.get("false_positive_diagnosis") or {}
        fp_html = (
            f"<p><b>Primary:</b> {escape(str(fp.get('primary')))} | "
            f"<b>Secondary:</b> {escape(str(fp.get('secondary', 'N/A')))} | "
            f"<b>Confidence:</b> {fp.get('confidence', 0):.1%}</p>"
            f"<p><b>Suggested follow-up:</b> {escape(str(fp.get('suggested_follow_up', 'N/A')))}</p>"
            if fp.get("primary")
            else "<p>No false-positive diagnosis for planet-like prediction.</p>"
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Exoplanet Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 12px}}</style></head>
<body>
<h1>Exoplanet Detection Report</h1>
<p><b>Target:</b> {escape(str(payload.get('target_id','')))} |
<b>Class:</b> {escape(str(payload.get('predicted_class','')))} |
<b>Confidence:</b> {payload.get('final_confidence',0):.1%}</p>
<h2>Class Probabilities</h2>
<table><tr><th>Class</th><th>Probability</th></tr>{prob_rows}</table>
<h2>Alternative Hypotheses</h2><ul>{alt_items or '<li>None</li>'}</ul>
<h2>False-Positive Diagnosis</h2>{fp_html}
<h2>Top Feature Values</h2><ul>{feature_items or '<li>None</li>'}</ul>
<h2>Explanation</h2>
<h3>Confidence</h3><ul>{confidence_items or '<li>None</li>'}</ul>
<h3>Posterior Summary</h3><ul>{posterior_items or '<li>None</li>'}</ul>
<h2>Quality</h2><p>Score: {payload.get('quality_score', 'N/A')}</p>
<h2>Warnings</h2><ul>{warning_items or '<li>None</li>'}</ul>
<h2>Figures</h2><ul>{fig_links or '<li>No figures</li>'}</ul>
</body></html>"""
