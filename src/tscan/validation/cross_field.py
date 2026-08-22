from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from tscan.types import ValidationIssue


def _number(value: Any) -> Decimal | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    return None


def cross_field_issues(value: Any) -> list[ValidationIssue]:
    if not isinstance(value, dict):
        return []
    issues: list[ValidationIssue] = []
    subtotal, tax, discount, total = (_number(value.get(key)) for key in ("subtotal", "tax", "discount", "total"))
    if subtotal is not None and total is not None:
        expected = subtotal + (tax or Decimal(0)) - (discount or Decimal(0))
        tolerance = max(Decimal("0.01"), abs(total) * Decimal("0.001"))
        if abs(expected - total) > tolerance:
            issues.append(ValidationIssue(severity="warning", path="$.total", code="total_mismatch", message=f"Expected total near {expected}, got {total}"))
    invoice_date, due_date = value.get("invoice_date"), value.get("due_date")
    if isinstance(invoice_date, str) and isinstance(due_date, str):
        try:
            if date.fromisoformat(due_date) < date.fromisoformat(invoice_date):
                issues.append(ValidationIssue(severity="warning", path="$.due_date", code="due_date_before_invoice_date", message="Due date is earlier than invoice date"))
        except ValueError:
            pass
    items = value.get("items") or value.get("line_items")
    if isinstance(items, list) and total is not None:
        item_totals = [_number(item.get("total")) for item in items if isinstance(item, dict)]
        if item_totals and all(item is not None for item in item_totals):
            items_sum = sum((item for item in item_totals if item is not None), Decimal(0))
            if abs(items_sum - (subtotal if subtotal is not None else total)) > Decimal("0.01"):
                issues.append(ValidationIssue(severity="warning", path="$.items", code="item_total_mismatch", message="Line item totals do not match document subtotal/total"))
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            quantity = _number(item.get("quantity"))
            unit_price = _number(item.get("unit_price"))
            amount = _number(item.get("amount"))
            if (
                quantity is not None
                and unit_price is not None
                and amount is not None
                and abs(quantity * unit_price - amount)
                > max(
                    Decimal("0.01"), abs(amount) * Decimal("0.001")
                )
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"$.items[{index}].amount",
                        code="line_amount_mismatch",
                        message="quantity × unit_price does not match line amount",
                    )
                )
    return issues
