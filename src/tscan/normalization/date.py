from __future__ import annotations

import re
from datetime import date
from typing import Any

from tscan.normalization.amount import NormalizedScalar


def _valid_iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def normalize_date(value: Any, *, language: str | None = None) -> NormalizedScalar:
    if not isinstance(value, str):
        return NormalizedScalar(value)
    text = value.strip()
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        valid = _valid_iso(*(int(part) for part in iso.groups()))
        return NormalizedScalar(valid if valid else value, warning=None if valid else f"Invalid ISO date: {value}")
    match = re.fullmatch(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})", text)
    if not match:
        return NormalizedScalar(value)
    first, middle, last = (int(part) for part in match.groups())
    locale = (language or "auto").lower()
    if first > 999:
        candidate = _valid_iso(first, middle, last)
    elif last > 999 and (first > 12 or locale == "vi"):
        candidate = _valid_iso(last, middle, first)
    elif last > 999 and locale == "en":
        candidate = _valid_iso(last, first, middle)
    elif last > 999 and first <= 12 and middle <= 12:
        return NormalizedScalar(value, warning=f"Ambiguous day/month order: {value}")
    else:
        return NormalizedScalar(value, warning=f"Ambiguous or incomplete date: {value}")
    if not candidate:
        return NormalizedScalar(value, warning=f"Invalid calendar date: {value}")
    return NormalizedScalar(candidate, changed=candidate != value, action="normalized_date")
