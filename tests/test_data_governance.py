from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from tscan.data import audit_registry, build_split_manifest, load_dataset_registry


def test_audit_and_duplicate_safe_split(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = Image.new("RGB", (160, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 80, 60), fill="black")
    image.save(image_root / "invoice.png")
    label = {
        "invoice_number": "INV-1",
        "invoice_date": "2026-08-20",
        "due_date": None,
        "seller_name": "A",
        "seller_tax_id": None,
        "buyer_name": "B",
        "buyer_tax_id": None,
        "currency": "VND",
        "subtotal": 100,
        "discount": None,
        "tax": 10,
        "total": 110,
        "line_items": [],
    }
    (tmp_path / "label.json").write_text(json.dumps(label), encoding="utf-8")
    records = [
        {"document_id": "one", "source_path": "invoice.png", "ground_truth_path": "label.json"},
        {"document_id": "two", "source_path": "invoice.png", "ground_truth_path": "label.json"},
    ]
    (tmp_path / "data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (tmp_path / "datasets.yaml").write_text(
        """version: '1'
datasets:
  - name: invoices
    version: '1'
    task: json_extraction
    language: vi
    document_type: invoice
    template_family: vendor-a
    jsonl: data.jsonl
    image_root: images
    schema_id: invoice
    source: test
    license: permitted
    usage: train
""",
        encoding="utf-8",
    )
    report = audit_registry(
        load_dataset_registry(tmp_path / "datasets.yaml"),
        schemas_dir=tmp_path / "schemas",
        output_dir=tmp_path / "out",
    )
    assert report["summary"] == {"accepted": 2, "rejected": 0}
    split = build_split_manifest(
        tmp_path / "out/artifacts/dataset_manifest.json",
        output_dir=tmp_path / "out/artifacts",
    )
    assert len({record["split"] for record in split["records"]}) == 1
    leakage = json.loads((tmp_path / "out/artifacts/leakage_report.json").read_text())
    assert leakage["leakage_count"] == 0
