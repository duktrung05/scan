from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from triscan.types import ValidationIssue
from triscan.validation.json_parser import parse_json_output
from triscan.validation.normalizer import normalize_to_schema
from triscan.validation.quality_rules import financial_consistency_issues


@dataclass(slots=True)
class ValidationResult:
    parsed: dict[str, Any] | list[Any] | None
    normalized: dict[str, Any] | list[Any] | None
    valid: bool
    repaired: bool
    repair_actions: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


def _json_path(parts: list[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def validate_and_normalize(raw: str, schema: dict[str, Any]) -> ValidationResult:
    parse = parse_json_output(raw)
    if parse.value is None:
        return ValidationResult(
            parsed=None,
            normalized=None,
            valid=False,
            repaired=parse.repaired,
            repair_actions=parse.actions,
            issues=[
                ValidationIssue(
                    severity="error",
                    path="$",
                    code="invalid_json",
                    message=parse.error or "Model output is not valid JSON",
                )
            ],
        )

    normalized = normalize_to_schema(parse.value, schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_issues = [
        ValidationIssue(
            severity="error",
            path=_json_path(list(error.absolute_path)),
            code=f"schema_{error.validator}",
            message=error.message,
        )
        for error in sorted(
            validator.iter_errors(normalized), key=lambda item: list(item.absolute_path)
        )
    ]
    issues = schema_issues + financial_consistency_issues(normalized)
    return ValidationResult(
        parsed=parse.value,
        normalized=normalized,
        valid=not any(issue.severity == "error" for issue in issues),
        repaired=parse.repaired or normalized != parse.value,
        repair_actions=parse.actions,
        issues=issues,
    )
