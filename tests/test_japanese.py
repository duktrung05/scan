from __future__ import annotations

from jsonschema import Draft202012Validator

from tscan.dataset_schema import japanese_root_schema
from tscan.evaluation.metrics import text_metrics
from tscan.prompts import build_prompt
from tscan.training_safety import classify_training_row
from tscan.types import DocumentLanguage, ExtractionMode


def test_japanese_language_prompt() -> None:
    prompt = build_prompt(
        mode=ExtractionMode.JSON,
        language=DocumentLanguage.JA,
        document_type="receipt",
        schema={"type": "object"},
    )
    assert "Language: Japanese" in prompt
    assert "Never romanize Japanese text" in prompt
    assert "vertical Japanese text" in prompt


def test_japanese_shorthand_schema_is_valid_json_schema() -> None:
    schema = japanese_root_schema(
        {
            "店舗名": "string",
            "合計": "number",
            "商品": [{"品名": "verbatim-string", "数量": "integer"}],
        }
    )
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["合計"]["type"] == ["number", "null"]


def test_japanese_schema_repairs_source_shape_from_answer_without_copying_values() -> None:
    schema = japanese_root_schema(
        {"支払方法": ["現金"]},
        example={"支払方法": "現金"},
    )
    assert schema["properties"]["支払方法"]["type"] == ["string", "null"]
    assert "現金" not in str(schema["properties"]["支払方法"])


def test_japanese_uses_cer_as_primary_text_metric() -> None:
    metrics = text_metrics("領収書です", "領収証です", language="ja")
    assert metrics["primary_error_rate"] == metrics["cer"]


def test_benchmark_guard_rejects_eval_only_rows() -> None:
    assert classify_training_row({"data_role": "EVAL_ONLY", "split": "train"}) == "skip"
    assert classify_training_row({"split": "external_test"}) == "skip"
    assert classify_training_row({"split": "validation"}) == "validation"
