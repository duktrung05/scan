from __future__ import annotations

from typing import Any

from tscan.types import ValidationIssue


def multipage_issues(value: Any, *, page_count: int | None = None) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return []
    pages = value.get("pages")
    if not isinstance(pages, list):
        return []
    issues: list[ValidationIssue] = []
    if page_count is not None and len(pages) != page_count:
        issues.append(ValidationIssue(severity="warning", path="$.pages", code="page_count_mismatch", message=f"Expected {page_count} pages, received {len(pages)} page records"))
    for field in ("invoice_number", "vendor", "seller_name", "buyer_name", "currency"):
        visible = {page.get(field) for page in pages if isinstance(page, dict) and page.get(field) is not None}
        if len(visible) > 1:
            issues.append(ValidationIssue(severity="warning", path=f"$.pages.*.{field}", code="multi_page_inconsistency", message=f"{field} changes across pages"))
    return issues
