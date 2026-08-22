from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _bucket(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def build_split_manifest(
    dataset_manifest: str | Path,
    *,
    output_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    group_by: str = "source_id",
) -> dict[str, Any]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    source = Path(dataset_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", payload if isinstance(payload, list) else [])

    duplicate_parent: dict[str, str] = {}
    for record in records:
        record_id = record["record_id"]
        duplicate_parent.setdefault(record_id, record_id)
        for image in record.get("images", []):
            owner = image.get("exact_duplicate_of") or image.get("perceptual_duplicate_of")
            if owner:
                duplicate_parent[record_id] = duplicate_parent.get(owner, owner)

    split_records = []
    group_splits: dict[str, str] = {}
    for record in records:
        usage = record.get("usage", "train")
        if usage in {"eval_only", "test", "validation"}:
            split = usage
        else:
            lineage = duplicate_parent.get(record["record_id"], record["record_id"])
            configured_group = record.get(group_by)
            group = str(
                lineage
                if lineage != record["record_id"]
                else configured_group or record.get("source_id") or lineage
            )
            value = _bucket(group, seed)
            if value < train_ratio:
                split = "train"
            elif value < train_ratio + validation_ratio:
                split = "validation"
            else:
                split = "test"
            previous = group_splits.setdefault(group, split)
            split = previous
        split_records.append({**record, "split": split})

    leakage = []
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in split_records:
        for image in record.get("images", []):
            for key in ("sha256", "perceptual_hash"):
                if image.get(key):
                    owners[(key, image[key])].add(record["split"])
    for (kind, value), splits in owners.items():
        if len(splits) > 1:
            leakage.append({"kind": kind, "value": value, "splits": sorted(splits)})

    split_payload = {"version": 1, "seed": seed, "group_by": group_by, "ratios": {"train": train_ratio, "validation": validation_ratio, "test": 1 - train_ratio - validation_ratio}, "records": split_records}
    leakage_payload = {"leakage_count": len(leakage), "leaks": leakage, "template_distribution": _template_distribution(split_records)}
    (output / "split_manifest.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "leakage_report.json").write_text(json.dumps(leakage_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return split_payload


def _template_distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        result[str(record.get("template_family", "unknown"))][record["split"]] += 1
    return {template: dict(counts) for template, counts in result.items()}
