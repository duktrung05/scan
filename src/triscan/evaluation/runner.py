from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from triscan.evaluation.metrics import json_metrics, text_metrics
from triscan.exceptions import ConfigurationError
from triscan.types import EvaluationSummary


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Unable to read JSON: {path}") from exc


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: fmean(row[key] for row in rows if key in row) for key in keys}


def evaluate_manifest(manifest_path: str | Path) -> EvaluationSummary:
    manifest = Path(manifest_path).expanduser().resolve()
    base = manifest.parent
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid manifest JSON on line {line_number}") from exc
        samples.append(sample)
    if not samples:
        raise ConfigurationError("Evaluation manifest is empty")

    scored: list[tuple[dict[str, Any], dict[str, float]]] = []
    for sample in samples:
        mode = sample.get("mode", "json")
        reference_path = (base / sample["ground_truth_path"]).resolve()
        prediction_path = (base / sample["prediction_path"]).resolve()
        if mode == "markdown":
            metrics = text_metrics(
                reference_path.read_text(encoding="utf-8"),
                prediction_path.read_text(encoding="utf-8"),
            )
        elif mode == "json":
            metrics = json_metrics(_read_json(reference_path), _read_json(prediction_path))
        else:
            raise ConfigurationError(f"Unknown evaluation mode: {mode}")
        scored.append((sample, metrics))

    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample, metrics in scored:
        groups[f"language:{sample.get('language', 'unknown')}"].append(metrics)
        groups[f"document_type:{sample.get('document_type', 'unknown')}"].append(metrics)

    return EvaluationSummary(
        sample_count=len(scored),
        metrics=_average([metrics for _, metrics in scored]),
        breakdowns={name: _average(rows) for name, rows in sorted(groups.items())},
    )
