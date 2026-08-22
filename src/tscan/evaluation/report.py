from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tscan.evaluation.runner import evaluate_manifest_details


def _score(detail: dict[str, Any]) -> float:
    metrics = detail["metrics"]
    return (
        metrics.get("field_value_f1", 0.0)
        + metrics.get("amount_tolerance_accuracy", 0.0)
        + metrics.get("date_normalized_accuracy", 0.0)
        + metrics.get("schema_valid_rate", 0.0)
        - metrics.get("hallucinated_field_rate", 0.0)
        - metrics.get("failure_rate", 0.0)
    )


def _table(mapping: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(f'{value:.4f}' if isinstance(value, float) else str(value))}</td></tr>"
        for key, value in mapping.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _samples(title: str, details: list[dict[str, Any]]) -> str:
    cards = []
    for detail in details:
        sample = detail["sample"]
        source = sample.get("source_path", "")
        try:
            source_uri = Path(source).expanduser().resolve().as_uri() if source else ""
        except ValueError:
            source_uri = str(source)
        image = f'<img src="{html.escape(source_uri)}" alt="source">' if source_uri else ""
        cards.append(
            "<article>"
            f"<h3>{html.escape(str(sample.get('document_id', sample.get('record_id', 'sample'))))}</h3>"
            f"<p>score={_score(detail):.4f} · errors={html.escape(', '.join(detail['errors']) or 'none')}</p>"
            f"{image}<div class=compare><pre>{html.escape(json.dumps(detail['reference'], ensure_ascii=False, indent=2))}</pre>"
            f"<pre>{html.escape(json.dumps(detail['prediction'], ensure_ascii=False, indent=2))}</pre></div></article>"
        )
    return f"<h2>{html.escape(title)}</h2>{''.join(cards)}"


def generate_html_report(manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary, details = evaluate_manifest_details(manifest_path)
    error_counts = Counter(error for detail in details for error in detail["errors"])
    ranked = sorted(details, key=_score, reverse=True)
    best, worst = ranked[:20], list(reversed(ranked[-20:]))
    breakdown_sections = "".join(
        f"<details><summary>{html.escape(name)}</summary>{_table(metrics)}</details>"
        for name, metrics in summary.breakdowns.items()
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>TScan baseline</title>
<style>body{{font:14px system-ui;margin:28px;color:#18212b}}table{{border-collapse:collapse;margin:8px 0 18px}}
th,td{{border:1px solid #ccd1d1;padding:6px 10px;text-align:left}}article{{border:1px solid #d5d8dc;padding:14px;margin:12px 0}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}pre{{background:#f6f8fa;padding:10px;overflow:auto}}
img{{max-width:360px;max-height:260px}}details{{margin:6px 0}}@media(max-width:800px){{.compare{{grid-template-columns:1fr}}}}</style>
</head><body><h1>TScan Phase 1 baseline</h1><p>Samples: {summary.sample_count}</p>
<h2>Overview</h2>{_table(summary.metrics)}<h2>Error taxonomy</h2>{_table(dict(error_counts))}
<h2>Benchmark matrix</h2>{breakdown_sections}{_samples('20 best samples', best)}{_samples('20 worst samples', worst)}
</body></html>"""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"summary": summary.model_dump(), "error_counts": dict(error_counts)}
