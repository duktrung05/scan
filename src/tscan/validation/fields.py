from __future__ import annotations

import re
from datetime import date
from typing import Any

from tscan.types import ValidationIssue


def field_validation_issues(value: Any) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return []
    issues: list[ValidationIssue] = []
    for key, child in value.items():
        lowered = key.casefold()
        if child is None:
            continue
        if lowered == "currency" and isinstance(child, str) and not re.fullmatch(r"[A-Z]{3}", child):
            issues.append(ValidationIssue(severity="warning", path=f"$.{key}", code="invalid_currency", message="Currency should be a visible three-letter ISO code"))
        if lowered.endswith("date") and isinstance(child, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", child):
            try:
                date.fromisoformat(child)
            except ValueError:
                issues.append(ValidationIssue(severity="error", path=f"$.{key}", code="invalid_date", message=f"Invalid calendar date: {child}"))
        if isinstance(child, dict):
            issues.extend(field_validation_issues(child))
    return issues
