from __future__ import annotations

from numbers import Real
from typing import Any

from tscan.types import ValidationIssue


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def financial_consistency_issues(value: Any, tolerance: float = 0.02) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return []
    issues: list[ValidationIssue] = []
    subtotal = _number(value.get("subtotal"))
    tax = _number(value.get("tax"))
    discount = _number(value.get("discount")) or 0.0
    total = _number(value.get("total"))
    if subtotal is not None and tax is not None and total is not None:
        expected = subtotal + tax - discount
        allowed_delta = max(tolerance, abs(total) * 0.005)
        if abs(expected - total) > allowed_delta:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path="$.total",
                    code="total_mismatch",
                    message=f"subtotal + tax - discount = {expected:g}, but total = {total:g}",
                )
            )

    item_key = "line_items" if "line_items" in value else "items"
    items = value.get(item_key)
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            quantity = _number(item.get("quantity"))
            unit_price = _number(item.get("unit_price"))
            amount = _number(item.get("amount"))
            if quantity is None or unit_price is None or amount is None:
                continue
            expected = quantity * unit_price
            allowed_delta = max(tolerance, abs(amount) * 0.005)
            if abs(expected - amount) > allowed_delta:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"$.{item_key}[{index}].amount",
                        code="line_amount_mismatch",
                        message=f"quantity × unit_price = {expected:g}, but amount = {amount:g}",
                    )
                )
    return issues
