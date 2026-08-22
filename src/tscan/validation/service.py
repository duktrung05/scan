from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tscan.types import ValidationIssue
from tscan.validation.cross_field import cross_field_issues
from tscan.validation.fields import field_validation_issues
from tscan.validation.json_parser import parse_json_output
from tscan.validation.multipage import multipage_issues
from tscan.validation.normalizer import normalize_with_warnings
from tscan.validation.schema import validate_schema


@dataclass(slots=True)
class ValidationResult:
    parsed: dict[str, Any] | list[Any] | None
    normalized: dict[str, Any] | list[Any] | None
    valid: bool
    syntax_valid: bool = True
    raw_schema_valid: bool = False
    schema_valid: bool = False
    repaired: bool = False
    repair_actions: list[str] = field(default_factory=list)
    raw_schema_issues: list[ValidationIssue] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


def validate_and_normalize(raw: str, schema: dict[str, Any], *, language: str | None = None, page_count: int | None = None) -> ValidationResult:
    parse = parse_json_output(raw)
    if parse.value is None:
        return ValidationResult(
            parsed=None,
            normalized=None,
            valid=False,
            syntax_valid=False,
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

    raw_schema_issues = validate_schema(parse.value, schema, severity="warning", code_prefix="raw_schema")
    normalization = normalize_with_warnings(parse.value, schema, language=language)
    normalized = normalization.value
    schema_issues = validate_schema(normalized, schema)
    issues = (
        normalization.issues
        + schema_issues
        + field_validation_issues(normalized)
        + cross_field_issues(normalized)
        + multipage_issues(normalized, page_count=page_count)
    )
    schema_valid = not schema_issues
    return ValidationResult(
        parsed=parse.value,
        normalized=normalized,
        valid=schema_valid and not any(issue.severity == "error" for issue in issues),
        syntax_valid=True,
        raw_schema_valid=not raw_schema_issues,
        schema_valid=schema_valid,
        repaired=parse.repaired or normalized != parse.value,
        repair_actions=parse.actions + normalization.actions,
        raw_schema_issues=raw_schema_issues,
        issues=issues,
    )
