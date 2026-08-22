from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from tscan.evaluation.error_taxonomy import classify_errors
from tscan.evaluation.metrics import json_metrics, text_metrics
from tscan.exceptions import ConfigurationError
from tscan.types import EvaluationSummary


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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _load_samples(manifest: Path) -> list[dict[str, Any]]:
    samples = []
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
    return samples


def evaluate_manifest_details(
    manifest_path: str | Path,
) -> tuple[EvaluationSummary, list[dict[str, Any]]]:
    manifest = Path(manifest_path).expanduser().resolve()
    base = manifest.parent
    samples = _load_samples(manifest)
    scored: list[tuple[dict[str, Any], dict[str, float]]] = []
    details: list[dict[str, Any]] = []

    for sample in samples:
        mode = sample.get("mode", "json")
        reference_path = (base / sample["ground_truth_path"]).resolve()
        prediction_path = (base / sample["prediction_path"]).resolve() if sample.get("prediction_path") else None
        failure = sample.get("failure")
        schema = _read_json((base / sample["schema_path"]).resolve()) if sample.get("schema_path") else None
        if mode == "json":
            reference = _read_json(reference_path)
        else:
            reference = reference_path.read_text(encoding="utf-8")

        if failure:
            prediction = None
            is_timeout = "timeout" in str(failure).casefold()
            metrics = {"failure_rate": 1.0, "timeout_rate": float(is_timeout)}
            if mode == "json":
                metrics.update({"json_parse_rate": 0.0, "schema_valid_rate": 0.0})
        elif prediction_path is None:
            raise ConfigurationError("prediction_path is required for successful samples")
        elif mode == "markdown":
            prediction = prediction_path.read_text(encoding="utf-8")
            metrics = text_metrics(reference, prediction, language=sample.get("language"))
        elif mode == "json":
            try:
                prediction = _read_json(prediction_path)
            except ConfigurationError as exc:
                prediction = None
                failure = str(exc)
                metrics = {
                    "json_parse_rate": 0.0,
                    "failure_rate": 1.0,
                    "timeout_rate": 0.0,
                    "schema_valid_rate": 0.0,
                }
            else:
                metrics = json_metrics(reference, prediction, schema=schema)
        else:
            raise ConfigurationError(f"Unknown evaluation mode: {mode}")

        metrics.setdefault("failure_rate", 0.0)
        metrics.setdefault("timeout_rate", 0.0)
        if mode == "json":
            metrics.setdefault(
                "json_parse_rate",
                float(sample.get("syntax_valid", not bool(failure))),
            )
            if sample.get("schema_valid") is not None:
                metrics["schema_valid_rate"] = float(sample["schema_valid"])
        if sample.get("latency_ms") is not None:
            metrics["latency_ms"] = float(sample["latency_ms"])
        if sample.get("peak_gpu_memory_mb") is not None:
            metrics["peak_gpu_memory_mb"] = float(sample["peak_gpu_memory_mb"])
        scored.append((sample, metrics))
        errors = classify_errors(
            reference,
            prediction,
            schema_valid=bool(metrics.get("schema_valid_rate", 0.0)) if mode == "json" and not failure else None,
            warning_codes=sample.get("warning_codes", []),
            failure=failure,
        )
        details.append(
            {"sample": sample, "metrics": metrics, "errors": errors, "reference": reference, "prediction": prediction}
        )

    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    dimensions = (
        "language", "document_type", "template_family", "image_quality", "page_scope", "schema_seen", "dataset"
    )
    for sample, metrics in scored:
        for dimension in dimensions:
            groups[f"{dimension}:{sample.get(dimension, 'unknown')}"] .append(metrics)

    overall = _average([metrics for _, metrics in scored])
    latencies = [float(metrics["latency_ms"]) for _, metrics in scored if "latency_ms" in metrics]
    if latencies:
        overall.update(
            {
                "latency_mean_ms": fmean(latencies),
                "latency_median_ms": _percentile(latencies, 0.5),
                "latency_p90_ms": _percentile(latencies, 0.9),
                "latency_p95_ms": _percentile(latencies, 0.95),
            }
        )
    summary = EvaluationSummary(
        sample_count=len(scored),
        metrics=overall,
        breakdowns={name: _average(rows) for name, rows in sorted(groups.items())},
    )
    return summary, details


def evaluate_manifest(manifest_path: str | Path) -> EvaluationSummary:
    return evaluate_manifest_details(manifest_path)[0]
