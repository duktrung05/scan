from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(slots=True)
class NormalizedScalar:
    value: Any
    changed: bool = False
    warning: str | None = None
    action: str | None = None


def normalize_amount(value: Any, *, language: str | None = None) -> NormalizedScalar:
    if not isinstance(value, str):
        return NormalizedScalar(value)
    original = value
    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[^0-9,.+\-]", "", text)
    if not text or text in {"+", "-"}:
        return NormalizedScalar(original, warning="Amount contains no parseable digits")

    comma_count, dot_count = text.count(","), text.count(".")
    decimal_separator: str | None = None
    thousands_separator: str | None = None
    if comma_count and dot_count:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
    elif comma_count or dot_count:
        separator = "," if comma_count else "."
        parts = text.split(separator)
        locale = (language or "auto").lower()
        expected_thousands = "." if locale == "vi" else "," if locale in {"en", "ja", "ko"} else None
        expected_decimal = "," if locale == "vi" else "." if locale in {"en", "ja", "ko"} else None
        if len(parts) > 2:
            if all(len(part) == 3 for part in parts[1:]):
                thousands_separator = separator
            else:
                return NormalizedScalar(original, warning=f"Ambiguous repeated separator in amount: {original}")
        elif len(parts[-1]) in {1, 2}:
            decimal_separator = separator
            if expected_decimal and separator != expected_decimal:
                return NormalizedScalar(original, warning=f"Amount separator conflicts with {locale} locale: {original}")
        elif len(parts[-1]) == 3:
            if expected_thousands == separator:
                thousands_separator = separator
            else:
                return NormalizedScalar(original, warning=f"Ambiguous decimal/thousands separator: {original}")
        else:
            return NormalizedScalar(original, warning=f"Ambiguous amount separator: {original}")

    canonical = text
    if thousands_separator:
        canonical = canonical.replace(thousands_separator, "")
    if decimal_separator and decimal_separator != ".":
        canonical = canonical.replace(decimal_separator, ".")
    try:
        number = Decimal(canonical)
    except InvalidOperation:
        return NormalizedScalar(original, warning=f"Invalid amount: {original}")
    if negative:
        number = -abs(number)
    normalized: int | float = int(number) if number == number.to_integral_value() else float(number)
    return NormalizedScalar(normalized, changed=normalized != original, action="normalized_amount")
