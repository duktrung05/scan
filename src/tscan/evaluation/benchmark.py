from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tscan.pipeline import ExtractionPipeline
from tscan.prompts import PROMPT_VERSION
from tscan.schema_registry import SchemaRegistry


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return safe[:120] or hashlib.sha256(value.encode()).hexdigest()[:16]


def run_benchmark(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    pipeline: ExtractionPipeline | None = None,
) -> Path:
    manifest = Path(manifest_path).expanduser().resolve()
    base = manifest.parent
    output = Path(output_dir).expanduser().resolve()
    predictions = output / "predictions"
    schemas_output = output / "schemas"
    predictions.mkdir(parents=True, exist_ok=True)
    schemas_output.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline or ExtractionPipeline()
    evaluation_rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        sample = json.loads(line)
        record_id = _safe_id(
            str(sample.get("document_id") or sample.get("record_id") or f"sample-{line_number}")
        )
        source = (base / sample["source_path"]).resolve()
        schema_path = (base / sample["schema_path"]).resolve() if sample.get("schema_path") else None
        custom_schema = (
            SchemaRegistry.parse_custom(schema_path.read_text(encoding="utf-8"))
            if schema_path
            else None
        )
        schema = custom_schema or pipeline.registry.load(sample.get("document_type", "invoice"))
        benchmark_schema_path = schemas_output / f"{record_id}.schema.json"
        benchmark_schema_path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        row = {
            **sample,
            "mode": "json",
            "source_path": str(source),
            "ground_truth_path": str((base / sample["ground_truth_path"]).resolve()),
            "schema_path": str(benchmark_schema_path),
        }
        try:
            result = pipeline.extract(
                source,
                mode="json",
                document_type=sample.get("document_type", "invoice"),
                language=sample.get("language", "auto"),
                custom_schema=custom_schema,
            )
            prediction_path = predictions / f"{record_id}.json"
            prediction_path.write_text(json.dumps(result.normalized_value, ensure_ascii=False, indent=2), encoding="utf-8")
            row.update(
                {
                    "prediction_path": str(prediction_path),
                    "latency_ms": result.inference.latency_ms,
                    "warning_codes": [issue.code for issue in result.warnings],
                    "syntax_valid": result.syntax_valid,
                    "schema_valid": result.schema_valid,
                    "run_id": result.run_id,
                }
            )
            try:
                import torch

                row["peak_gpu_memory_mb"] = (
                    torch.cuda.max_memory_allocated() / 1024**2
                    if torch.cuda.is_available()
                    else 0
                )
            except ImportError:
                row["peak_gpu_memory_mb"] = 0
        except Exception as exc:
            row["failure"] = f"{type(exc).__name__}: {exc}"
        evaluation_rows.append(row)

    evaluation_manifest = output / "evaluation_manifest.jsonl"
    with evaluation_manifest.open("w", encoding="utf-8") as handle:
        for row in evaluation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    safe_settings = pipeline.settings.model_dump(mode="json")
    safe_settings.pop("api_key", None)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": _file_hash(manifest),
        "model": pipeline.settings.model_name,
        "provider": pipeline.settings.model_provider,
        "prompt_version": PROMPT_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "settings": safe_settings,
    }
    try:
        import torch

        metadata["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
    except ImportError:
        metadata["runtime"] = {"torch": None, "cuda_available": False}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return evaluation_manifest
