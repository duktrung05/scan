from __future__ import annotations

import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

NULL_STRINGS = {"", "null", "none", "n/a", "na", "not available", "unreadable"}
TRUE_STRINGS = {"true", "yes", "y", "1"}
FALSE_STRINGS = {"false", "no", "n", "0"}


def _types(schema: dict[str, Any]) -> set[str]:
    value = schema.get("type")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {item for item in value if isinstance(item, str)}
    return set()


def _parse_number(value: str) -> int | float | None:
    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if not text or text in {"-", "+"}:
        return None

    comma = text.rfind(",")
    dot = text.rfind(".")
    if comma >= 0 and dot >= 0:
        decimal_separator = "," if comma > dot else "."
        thousands_separator = "." if decimal_separator == "," else ","
        text = text.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif comma >= 0 or dot >= 0:
        separator = "," if comma >= 0 else "."
        parts = text.split(separator)
        if len(parts) > 2:
            if len(parts[-1]) in {1, 2}:
                text = "".join(parts[:-1]) + "." + parts[-1]
            else:
                text = "".join(parts)
        elif len(parts) == 2:
            text = "".join(parts) if len(parts[1]) == 3 else ".".join(parts)
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if negative:
        number = -abs(number)
    if number == number.to_integral_value():
        return int(number)
    return float(number)


def _normalize(value: Any, schema: dict[str, Any]) -> Any:
    allowed = _types(schema)
    if isinstance(value, str) and "null" in allowed and value.strip().lower() in NULL_STRINGS:
        return None
    if value is None:
        return None

    if isinstance(value, dict) and ("object" in allowed or "properties" in schema):
        properties = schema.get("properties", {})
        return {key: _normalize(child, properties.get(key, {})) for key, child in value.items()}
    if isinstance(value, list) and "array" in allowed:
        item_schema = schema.get("items", {})
        return [_normalize(item, item_schema) for item in value]
    if isinstance(value, str) and ("number" in allowed or "integer" in allowed):
        parsed = _parse_number(value)
        if parsed is not None:
            if "integer" in allowed and isinstance(parsed, float) and parsed.is_integer():
                return int(parsed)
            return parsed
    if isinstance(value, str) and "boolean" in allowed:
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return True
        if lowered in FALSE_STRINGS:
            return False
    return value


def normalize_to_schema(value: Any, schema: dict[str, Any]) -> Any:
    return _normalize(deepcopy(value), schema)
