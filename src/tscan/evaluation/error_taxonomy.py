from __future__ import annotations

from typing import Any

from tscan.evaluation.metrics import flatten

ERROR_LABELS = {
    "image_load_error",
    "preprocessing_error",
    "json_parse_error",
    "schema_violation",
    "missing_field",
    "hallucinated_field",
    "wrong_field_value",
    "amount_error",
    "date_error",
    "locale_normalization_error",
    "cross_field_inconsistency",
    "multi_page_inconsistency",
    "truncation",
    "timeout",
}


def classify_errors(
    reference: Any,
    prediction: Any,
    *,
    schema_valid: bool | None = None,
    warning_codes: list[str] | None = None,
    failure: str | None = None,
) -> list[str]:
    labels: set[str] = set()
    failure_text = (failure or "").casefold()
    for label in ("image_load_error", "preprocessing_error", "timeout", "truncation"):
        if label.replace("_", " ") in failure_text or label in failure_text:
            labels.add(label)
    if failure and not labels:
        labels.add("json_parse_error")
    if schema_valid is False:
        labels.add("schema_violation")
    codes = warning_codes or []
    if any("normalization" in code for code in codes):
        labels.add("locale_normalization_error")
    if any(
        code in {"total_mismatch", "item_total_mismatch", "due_date_before_invoice_date"}
        for code in codes
    ):
        labels.add("cross_field_inconsistency")
    if any("multi_page" in code or "page_count" in code for code in codes):
        labels.add("multi_page_inconsistency")
    if reference is None or prediction is None:
        return sorted(labels)

    ref, pred = flatten(reference), flatten(prediction)
    if set(ref) - set(pred):
        labels.add("missing_field")
    if set(pred) - set(ref):
        labels.add("hallucinated_field")
    wrong = {key for key in set(ref) & set(pred) if ref[key] != pred[key]}
    if wrong:
        labels.add("wrong_field_value")
    if any(
        any(token in key.casefold() for token in ("amount", "total", "price", "tax", "subtotal"))
        for key in wrong
    ):
        labels.add("amount_error")
    if any("date" in key.casefold() for key in wrong):
        labels.add("date_error")
    return sorted(labels)
