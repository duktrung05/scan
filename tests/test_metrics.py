from __future__ import annotations

from triscan.evaluation.metrics import json_metrics, text_metrics


def test_text_metrics_exact_match() -> None:
    metrics = text_metrics("Xin chào", "Xin chào")
    assert metrics["cer"] == 0
    assert metrics["wer"] == 0
    assert metrics["normalized_edit_similarity"] == 1


def test_json_metrics() -> None:
    reference = {"invoice_number": "001", "total": 100}
    prediction = {"invoice_number": "001", "total": 99}
    metrics = json_metrics(reference, prediction)
    assert metrics["key_f1"] == 1
    assert metrics["field_accuracy"] == 0.5
    assert metrics["critical_field_accuracy"] == 0.5
    assert metrics["exact_match"] == 0
