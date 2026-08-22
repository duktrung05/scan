from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tscan.dataset_schema import japanese_root_schema
from tscan.prompts import build_prompt
from tscan.types import DocumentLanguage, ExtractionMode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(data_dir: Path) -> list[dict[str, Any]]:
    registry_path = data_dir / "registry.json"
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise SystemExit(f"Invalid Japanese data registry: {registry_path}")
    return sources


def resolve_image_path(data_dir: Path, value: str) -> tuple[Path, Path]:
    relative = Path(value)
    images_root = (data_dir / "images").resolve()
    if relative.is_absolute():
        raise SystemExit(f"Absolute image paths are not allowed: {value}")
    candidate = (data_dir / relative).resolve()
    try:
        image_relative = candidate.relative_to(images_root)
    except ValueError as exc:
        raise SystemExit(f"Image path escapes data/japanese/images: {value}") from exc
    return candidate, image_relative


def prepare(data_dir: Path, output_dir: Path, *, validate_only: bool = False) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images = output_dir / "images"
    records: list[dict[str, Any]] = []
    missing_images: list[str] = []
    blocked_license = 0
    source_rows = 0

    for source in load_registry(data_dir):
        source_path = data_dir / str(source["file"])
        expected_hash = str(source["sha256"])
        actual_hash = sha256(source_path)
        if actual_hash != expected_hash:
            raise SystemExit(f"Hash mismatch for {source_path.name}: {actual_hash}")
        if source.get("license_status") != "OK":
            blocked_license += int(source.get("rows", 0))

        source_row_count = 0
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSONL in {source_path.name}:{line_number}"
                    ) from exc
                source_row_count += 1
                source_rows += 1
                if row.get("language") != "ja":
                    raise SystemExit(f"Expected language=ja in {source_path.name}:{line_number}")
                try:
                    answer_value = json.loads(row["answer"])
                except (KeyError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"Invalid answer in {source_path.name}:{line_number}") from exc

                image_path, image_relative = resolve_image_path(data_dir, str(row["image"]))
                if not image_path.is_file():
                    missing_images.append(str(Path("images") / image_relative))
                    continue

                schema = japanese_root_schema(
                    row["schema"],
                    title=str(row["id"]),
                    example=answer_value,
                )
                prompt = build_prompt(
                    mode=ExtractionMode.JSON,
                    language=DocumentLanguage.JA,
                    document_type=str(row.get("doc_type", "document")),
                    schema=schema,
                )
                destination = output_images / str(source["dataset_id"]) / image_relative
                if not validate_only:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, destination)
                records.append(
                    {
                        "document_id": str(row["id"]),
                        "source_id": str(row.get("source_id", row["id"])),
                        "dataset_id": str(source["dataset_id"]),
                        "images": [str(destination.relative_to(output_dir))],
                        "prompt": prompt,
                        "completion": json.dumps(answer_value, ensure_ascii=False, separators=(",", ":")),
                        "language": "ja",
                        "document_type": str(row.get("doc_type", "document")),
                        "mode": "json",
                        "split": str(source["split"]),
                        "data_role": str(source["role"]),
                        "license_status": str(source["license_status"]),
                    }
                )
        expected_rows = int(source.get("rows", -1))
        if source_row_count != expected_rows:
            raise SystemExit(
                f"Row-count mismatch for {source_path.name}: "
                f"expected {expected_rows}, found {source_row_count}"
            )

    summary = {
        "source_rows": source_rows,
        "prepared_rows": len(records),
        "missing_images": len(missing_images),
        "license_blocked_rows": blocked_license,
    }
    if missing_images:
        examples = "\n".join(f"  - {item}" for item in missing_images[:10])
        raise SystemExit(
            f"Missing {len(missing_images)} referenced Japanese images under {data_dir}.\n"
            f"First missing paths:\n{examples}\n"
            "Copy the matching images/ directory into data/japanese and run again."
        )
    if blocked_license:
        raise SystemExit(
            f"{blocked_license} rows have unverified licenses. Update data/japanese/registry.json "
            "only after documenting permission."
        )
    if not validate_only:
        dataset_path = output_dir / "dataset.jsonl"
        with dataset_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare Japanese TScan VLM data")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/japanese"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    summary = prepare(
        args.data_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        validate_only=args.validate_only,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
