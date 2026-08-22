from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tscan.evaluation.benchmark import run_benchmark
from tscan.evaluation.report import generate_html_report
from tscan.evaluation.runner import evaluate_manifest
from tscan.pipeline import ExtractionPipeline


def test_benchmark_writes_metrics_metadata_and_report(settings, tmp_path: Path) -> None:
    source = tmp_path / "invoice.png"
    Image.new("RGB", (160, 80), "white").save(source)
    reference = {
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "seller_name": None,
        "seller_tax_id": None,
        "buyer_name": None,
        "buyer_tax_id": None,
        "currency": None,
        "subtotal": None,
        "discount": None,
        "tax": None,
        "total": None,
        "line_items": [],
    }
    (tmp_path / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    sample = {
        "document_id": "invoice-one",
        "source_path": "invoice.png",
        "ground_truth_path": "reference.json",
        "language": "vi",
        "document_type": "invoice",
        "template_family": "test",
        "image_quality": "clean",
        "page_scope": "single_page",
        "schema_seen": "seen",
        "dataset": "unit",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    output = tmp_path / "benchmark"
    evaluation_manifest = run_benchmark(
        manifest,
        output_dir=output,
        pipeline=ExtractionPipeline(settings=settings),
    )
    summary = evaluate_manifest(evaluation_manifest)
    assert summary.metrics["exact_match"] == 1
    assert summary.metrics["json_parse_rate"] == 1
    assert (output / "run_metadata.json").is_file()
    report = tmp_path / "report.html"
    generate_html_report(evaluation_manifest, report)
    assert "20 best samples" in report.read_text(encoding="utf-8")
