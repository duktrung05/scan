from __future__ import annotations

import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image, ImageStat, UnidentifiedImageError

from tscan.data.registry import DatasetEntry, DatasetRegistry
from tscan.schema_registry import SchemaRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(image: Image.Image, size: int = 8) -> str:
    grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= average else "0" for value in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"


def _hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _resolve_image_paths(record: dict[str, Any], entry: DatasetEntry) -> list[Path]:
    values = record.get("images")
    if not isinstance(values, list):
        value = record.get("source_path") or record.get("image_path")
        values = [value] if value else []
    paths = []
    for value in values:
        path = Path(str(value))
        paths.append(path if path.is_absolute() else (entry.image_root / path).resolve())
    return paths


def _read_ground_truth(record: dict[str, Any], entry: DatasetEntry) -> Any:
    if "ground_truth" in record:
        return record["ground_truth"]
    value = record.get("ground_truth_path")
    if not value:
        raise ValueError("missing_ground_truth")
    path = Path(str(value))
    if not path.is_absolute():
        path = (entry.jsonl.parent / path).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schema(entry: DatasetEntry, schemas: SchemaRegistry) -> dict[str, Any] | None:
    if entry.task != "json_extraction":
        return None
    if entry.schema_path:
        return SchemaRegistry.parse_custom(entry.schema_path.read_text(encoding="utf-8"))
    return schemas.load(entry.schema_id or entry.document_type)


def _render_html(report: dict[str, Any]) -> str:
    rows = []
    for dataset in report["datasets"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(dataset['name'])}</td>"
            f"<td>{dataset['accepted']}</td><td>{dataset['rejected']}</td>"
            f"<td>{html.escape(', '.join(dataset['reasons']))}</td></tr>"
        )
    return """<!doctype html><html><head><meta charset="utf-8"><title>TScan data audit</title>
<style>body{font:14px system-ui;margin:32px;color:#17202a}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ccd1d1;padding:8px;text-align:left}th{background:#f4f6f7}pre{white-space:pre-wrap}</style>
</head><body><h1>TScan data audit</h1>""" + (
        f"<p>Generated: {html.escape(report['generated_at'])}</p>"
        f"<p>Accepted: {report['summary']['accepted']} · Rejected: {report['summary']['rejected']}</p>"
        "<table><thead><tr><th>Dataset</th><th>Accepted</th><th>Rejected</th><th>Reasons</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<h2>Statistics</h2><pre>{html.escape(json.dumps(report['statistics'], ensure_ascii=False, indent=2))}</pre>"
        "</body></html>"
    )


def audit_registry(
    registry: DatasetRegistry,
    *,
    schemas_dir: Path,
    output_dir: Path,
    min_width: int = 64,
    min_height: int = 64,
    max_megapixels: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    artifacts_dir = output_dir / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    schemas = SchemaRegistry(schemas_dir)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    dataset_reports = []
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    exact_owner: dict[str, str] = {}
    perceptual_owners: list[tuple[str, str]] = []
    global_seen_ids: set[str] = set()

    for entry in registry.datasets:
        local_accepted = 0
        local_rejected = 0
        reasons_counter: Counter[str] = Counter()
        seen_ids: set[str] = set()
        try:
            schema = _load_schema(entry, schemas)
        except Exception as exc:
            schema = None
            dataset_schema_error = f"schema_error:{exc}"
        else:
            dataset_schema_error = ""

        try:
            lines = entry.jsonl.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            lines = []
            rejected.append({"dataset": entry.name, "record_id": None, "reasons": [f"jsonl_error:{exc}"]})
            local_rejected += 1

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            reasons: list[str] = []
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                rejected.append({"dataset": entry.name, "line": line_number, "record_id": None, "reasons": [f"invalid_jsonl:{exc}"]})
                local_rejected += 1
                reasons_counter["invalid_jsonl"] += 1
                continue

            record_id = str(record.get("document_id") or record.get("record_id") or "").strip()
            if not record_id:
                reasons.append("missing_id")
                record_id = f"{entry.name}:line-{line_number}"
            if record_id in seen_ids:
                reasons.append("duplicate_id")
            seen_ids.add(record_id)
            if record_id in global_seen_ids:
                reasons.append("duplicate_global_id")
            global_seen_ids.add(record_id)
            if dataset_schema_error:
                reasons.append(dataset_schema_error)
            if entry.usage == "train" and entry.license.strip().casefold() in {
                "verify",
                "unknown",
                "unverified",
            }:
                reasons.append("license_not_verified")

            try:
                ground_truth = _read_ground_truth(record, entry)
                if ground_truth in ({}, [], None, ""):
                    reasons.append("empty_ground_truth")
                if schema is not None:
                    errors = list(
                        Draft202012Validator(
                            schema, format_checker=FormatChecker()
                        ).iter_errors(ground_truth)
                    )
                    if errors:
                        reasons.append("ground_truth_schema_violation")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                ground_truth = None
                reasons.append(f"ground_truth_error:{exc}")

            image_details = []
            image_paths = _resolve_image_paths(record, entry)
            if not image_paths:
                reasons.append("missing_image_path")
            for image_path in image_paths:
                detail: dict[str, Any] = {"path": str(image_path)}
                try:
                    exact_hash = _sha256(image_path)
                    with Image.open(image_path) as image:
                        image.load()
                        width, height = image.size
                        perceptual_hash = _average_hash(image)
                        variance = sum(ImageStat.Stat(image.convert("L")).var) / 1
                    detail.update({"sha256": exact_hash, "perceptual_hash": perceptual_hash, "width": width, "height": height})
                    if width < min_width or height < min_height:
                        reasons.append("image_too_small")
                    if width * height > max_megapixels * 1_000_000:
                        reasons.append("image_too_large")
                    if variance < 1.0:
                        reasons.append("blank_image")
                    if exact_hash in exact_owner and exact_owner[exact_hash] != record_id:
                        detail["exact_duplicate_of"] = exact_owner[exact_hash]
                    else:
                        exact_owner[exact_hash] = record_id
                    nearest = min(
                        perceptual_owners,
                        key=lambda item: _hash_distance(perceptual_hash, item[0]),
                        default=None,
                    )
                    if nearest and nearest[1] != record_id and _hash_distance(perceptual_hash, nearest[0]) <= 5:
                        detail["perceptual_duplicate_of"] = nearest[1]
                        detail["perceptual_distance"] = _hash_distance(perceptual_hash, nearest[0])
                    else:
                        perceptual_owners.append((perceptual_hash, record_id))
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    reasons.append(f"image_load_error:{exc}")
                image_details.append(detail)

            manifest_record = {
                "record_id": record_id,
                "dataset": entry.name,
                "dataset_version": entry.version,
                "usage": entry.usage,
                "language": record.get("language", entry.language),
                "document_type": record.get("document_type", entry.document_type),
                "template_family": record.get("template_family", record.get("source_family", entry.template_family)),
                "image_quality": record.get("image_quality", "unknown"),
                "page_scope": "multi_page" if len(image_paths) > 1 else "single_page",
                "schema_id": entry.schema_id,
                "source_id": record.get("source_id", record.get("source_family", record_id)),
                "images": image_details,
                "ground_truth": ground_truth,
                "source_record": record,
            }
            if reasons or entry.usage == "reject":
                if entry.usage == "reject":
                    reasons.append("dataset_usage_reject")
                rejected.append({**manifest_record, "reasons": sorted(set(reasons))})
                local_rejected += 1
                for reason in set(reasons):
                    reasons_counter[reason.split(":", 1)[0]] += 1
            else:
                accepted.append(manifest_record)
                local_accepted += 1
                stats["language"][manifest_record["language"]] += 1
                stats["document_type"][manifest_record["document_type"]] += 1
                if isinstance(ground_truth, dict):
                    for field in ground_truth:
                        stats["field"][str(field)] += 1

        dataset_reports.append({"name": entry.name, "version": entry.version, "accepted": local_accepted, "rejected": local_rejected, "reasons": [f"{key}={value}" for key, value in reasons_counter.most_common()]})

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_version": registry.version,
        "summary": {"accepted": len(accepted), "rejected": len(rejected)},
        "datasets": dataset_reports,
        "statistics": {key: dict(value) for key, value in stats.items()},
    }
    (reports_dir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "data_audit.html").write_text(_render_html(report), encoding="utf-8")
    with (reports_dir / "rejected_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in rejected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (artifacts_dir / "dataset_manifest.json").write_text(json.dumps({"version": 1, "records": accepted}, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
