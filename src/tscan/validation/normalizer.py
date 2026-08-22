from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from tscan.normalization import normalize_amount, normalize_date
from tscan.types import ValidationIssue

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


@dataclass(slots=True)
class SchemaNormalizationResult:
    value: Any
    issues: list[ValidationIssue] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def _path(parent: str, key: str | int) -> str:
    return f"{parent}[{key}]" if isinstance(key, int) else f"{parent}.{key}"


def _normalize(value: Any, schema: dict[str, Any], *, path: str, language: str | None) -> SchemaNormalizationResult:
    allowed = _types(schema)
    if isinstance(value, str) and "null" in allowed and value.strip().lower() in NULL_STRINGS:
        return SchemaNormalizationResult(None, actions=[f"{path}:normalized_null"])
    if value is None:
        return SchemaNormalizationResult(None)

    if isinstance(value, dict) and ("object" in allowed or "properties" in schema):
        properties = schema.get("properties", {})
        normalized: dict[str, Any] = {}
        result = SchemaNormalizationResult(normalized)
        for key, child in value.items():
            item = _normalize(child, properties.get(key, {}), path=_path(path, key), language=language)
            normalized[key] = item.value
            result.issues.extend(item.issues)
            result.actions.extend(item.actions)
        return result
    if isinstance(value, list) and "array" in allowed:
        item_schema = schema.get("items", {})
        normalized_items = []
        result = SchemaNormalizationResult(normalized_items)
        for index, item_value in enumerate(value):
            item = _normalize(item_value, item_schema, path=_path(path, index), language=language)
            normalized_items.append(item.value)
            result.issues.extend(item.issues)
            result.actions.extend(item.actions)
        return result
    if isinstance(value, str) and ("number" in allowed or "integer" in allowed):
        amount = normalize_amount(value, language=language)
        if amount.warning:
            return SchemaNormalizationResult(value, [ValidationIssue(severity="warning", path=path, code="locale_normalization_ambiguous", message=amount.warning)])
        if amount.changed:
            return SchemaNormalizationResult(amount.value, actions=[f"{path}:{amount.action}"])
    if isinstance(value, str) and schema.get("format") == "date":
        normalized_date = normalize_date(value, language=language)
        if normalized_date.warning:
            return SchemaNormalizationResult(value, [ValidationIssue(severity="warning", path=path, code="date_normalization_ambiguous", message=normalized_date.warning)])
        if normalized_date.changed:
            return SchemaNormalizationResult(normalized_date.value, actions=[f"{path}:{normalized_date.action}"])
    if isinstance(value, str) and "boolean" in allowed:
        lowered = value.strip().lower()
        if lowered in TRUE_STRINGS:
            return SchemaNormalizationResult(True, actions=[f"{path}:normalized_boolean"])
        if lowered in FALSE_STRINGS:
            return SchemaNormalizationResult(False, actions=[f"{path}:normalized_boolean"])
    return SchemaNormalizationResult(value)


def normalize_with_warnings(value: Any, schema: dict[str, Any], *, language: str | None = None) -> SchemaNormalizationResult:
    return _normalize(deepcopy(value), schema, path="$", language=language)


def normalize_to_schema(value: Any, schema: dict[str, Any]) -> Any:
    return normalize_with_warnings(value, schema).value
