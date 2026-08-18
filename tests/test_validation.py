from __future__ import annotations

from triscan.schema_registry import SchemaRegistry
from triscan.validation import validate_and_normalize
from triscan.validation.json_parser import parse_json_output


def test_recovers_fenced_json_and_trailing_comma() -> None:
    result = parse_json_output('```json\n{"total": "1,234.50",}\n```')
    assert result.value == {"total": "1,234.50"}
    assert result.repaired is True
    assert "removed_code_fence" in result.actions
    assert "removed_trailing_commas" in result.actions


def test_normalizes_numbers_and_validates() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "subtotal": {"type": ["number", "null"]},
            "tax": {"type": ["number", "null"]},
            "discount": {"type": ["number", "null"]},
            "total": {"type": ["number", "null"]},
        },
        "required": ["subtotal", "tax", "discount", "total"],
    }
    result = validate_and_normalize(
        '{"subtotal":"1,000.00","tax":"100.00","discount":null,"total":"1,100.00"}',
        schema,
    )
    assert result.valid is True
    assert result.normalized["total"] == 1100
    assert result.repaired is True
    assert result.issues == []


def test_reports_financial_mismatch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "subtotal": {"type": "number"},
            "tax": {"type": "number"},
            "total": {"type": "number"},
        },
        "required": ["subtotal", "tax", "total"],
    }
    result = validate_and_normalize('{"subtotal":100,"tax":10,"total":900}', schema)
    assert result.valid is True
    assert any(issue.code == "total_mismatch" for issue in result.issues)


def test_builtin_registry(settings) -> None:
    registry = SchemaRegistry(settings.schemas_dir)
    assert {"invoice", "receipt", "form"}.issubset(registry.list())
    assert registry.load("invoice")["type"] == "object"
