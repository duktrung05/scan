from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tscan.config import get_settings
from tscan.ingestion import load_document
from tscan.prompts import build_prompt
from tscan.schema_registry import SchemaRegistry
from tscan.types import DocumentLanguage, ExtractionMode


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a TScan JSONL manifest for VLM SFT")
    parser.add_argument("manifest", type=Path, help="Source JSONL manifest")
    parser.add_argument("output_dir", type=Path, help="Prepared dataset directory")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    source_base = manifest.parent
    output = args.output_dir.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    registry = SchemaRegistry(get_settings().schemas_dir)
    records = []

    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        sample = json.loads(line)
        data_role = str(sample.get("data_role", sample.get("role", "TRAIN_CORE"))).upper()
        sample_id = safe_id(str(sample.get("document_id") or f"sample-{line_number}"))
        mode = ExtractionMode(sample.get("mode", "json"))
        language = DocumentLanguage(sample.get("language", "auto"))
        document_type = sample.get("document_type", "invoice")
        source_path = (source_base / sample["source_path"]).resolve()
        label_path = (source_base / sample["ground_truth_path"]).resolve()
        document = load_document(source_path, get_settings())
        image_paths = []
        for page in document.pages:
            image_path = images_dir / f"{sample_id}-p{page.page_number:03d}.png"
            page.image.save(image_path, format="PNG", optimize=True)
            page.image.close()
            image_paths.append(str(image_path.relative_to(output)))

        schema = registry.load(document_type) if mode == ExtractionMode.JSON else None
        prompt = build_prompt(
            mode=mode,
            language=language,
            document_type=document_type,
            schema=schema,
        )
        if mode == ExtractionMode.JSON:
            completion = json.dumps(
                json.loads(label_path.read_text(encoding="utf-8")),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            completion = label_path.read_text(encoding="utf-8")
        records.append(
            {
                "document_id": sample_id,
                "images": image_paths,
                "prompt": prompt,
                "completion": completion,
                "language": language.value,
                "document_type": document_type,
                "mode": mode.value,
                "split": sample.get("split", "train"),
                "data_role": data_role,
                "source_id": sample.get("source_id", sample_id),
            }
        )

    with (output / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Prepared {len(records)} samples in {output}")


if __name__ == "__main__":
    main()
