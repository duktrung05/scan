from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from tscan.types import ValidationIssue


def json_path(parts: list[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def validate_schema(value: Any, schema: dict[str, Any], *, severity: str = "error", code_prefix: str = "schema") -> list[ValidationIssue]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        ValidationIssue(
            severity=severity,
            path=json_path(list(error.absolute_path)),
            code=f"{code_prefix}_{error.validator}",
            message=error.message,
        )
        for error in sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    ]
