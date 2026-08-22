from __future__ import annotations

import json
from pathlib import Path

from tscan.evaluation.metrics import json_metrics, text_metrics
from tscan.evaluation.runner import evaluate_manifest_details


def test_text_metrics_exact_match() -> None:
    metrics = text_metrics("Xin chào", "Xin chào")
    assert metrics["cer"] == 0
    assert metrics["wer"] == 0
    assert metrics["primary_error_rate"] == 0
    assert metrics["normalized_edit_similarity"] == 1


def test_json_metrics() -> None:
    reference = {"invoice_number": "001", "total": 100}
    prediction = {"invoice_number": "001", "total": 99}
    metrics = json_metrics(reference, prediction)
    assert metrics["key_f1"] == 1
    assert metrics["field_accuracy"] == 0.5
    assert metrics["critical_field_accuracy"] == 0.5
    assert metrics["exact_match"] == 0
    assert metrics["json_parse_rate"] == 1
    assert metrics["amount_exact_accuracy"] == 0


def test_invalid_prediction_is_counted_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "reference.json").write_text('{"total":100}', encoding="utf-8")
    (tmp_path / "prediction.json").write_text('{"total":', encoding="utf-8")
    sample = {
        "document_id": "broken",
        "ground_truth_path": "reference.json",
        "prediction_path": "prediction.json",
        "mode": "json",
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    summary, details = evaluate_manifest_details(tmp_path / "manifest.jsonl")
    assert summary.metrics["json_parse_rate"] == 0
    assert summary.metrics["failure_rate"] == 1
    assert details[0]["errors"] == ["json_parse_error"]
