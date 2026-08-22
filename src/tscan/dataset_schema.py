from __future__ import annotations

from typing import Any

STRING_HINTS = {
    "string",
    "verbatim-string",
    "phone-number",
    "email-address",
    "url",
    "date",
    "date-time",
    "time",
    "duration",
    "currency",
    "country",
}
TYPE_MARKERS = STRING_HINTS | {"number", "integer", "boolean"}


def _nullable(value_type: str) -> list[str]:
    return [value_type, "null"]


def shorthand_to_json_schema(value: Any) -> dict[str, Any]:
    """Convert the Japanese dataset's compact schema DSL to JSON Schema 2020-12.

    Unknown string markers are retained as hints instead of being interpreted as literal
    ground truth. This is deliberate: several source rows contain generated or domain-specific
    markers that are not JSON Schema types.
    """

    if isinstance(value, str):
        if value == "number":
            return {"type": _nullable("number")}
        if value == "integer":
            return {"type": _nullable("integer")}
        if value == "boolean":
            return {"type": _nullable("boolean")}
        if value in STRING_HINTS:
            return {"type": _nullable("string"), "x-tscan-hint": value}
        return {
            "type": _nullable("string"),
            "x-tscan-hint": value,
        }
    if isinstance(value, list):
        if value and all(isinstance(item, str) and item not in TYPE_MARKERS for item in value):
            return {
                "anyOf": [
                    {"type": _nullable("string"), "enum": [*value, None]},
                    {"type": ["array", "null"], "items": {"type": _nullable("string")}},
                ],
                "x-tscan-hint": "literal-list",
            }
        item = shorthand_to_json_schema(value[0]) if value else {}
        return {"type": ["array", "null"], "items": item}
    if isinstance(value, dict):
        properties = {str(key): shorthand_to_json_schema(child) for key, child in value.items()}
        return {
            "type": ["object", "null"],
            "additionalProperties": True,
            "properties": properties,
        }
    return {"type": ["string", "number", "integer", "boolean", "null"]}


def _schema_from_examples(examples: list[Any], hint: Any = None) -> dict[str, Any]:
    """Infer a permissive schema from every observed value, never from one list item.

    Japanese extraction rows occasionally contain heterogeneous arrays (for example a
    nullable first row followed by populated rows). Merging all examples keeps validation
    honest without copying answer literals into the generated schema.
    """

    non_null = [example for example in examples if example is not None]
    if not non_null:
        return shorthand_to_json_schema(hint) if hint is not None else {"type": "null"}

    if all(isinstance(example, bool) for example in non_null):
        return {"type": _nullable("boolean")}
    if all(isinstance(example, int) and not isinstance(example, bool) for example in non_null):
        return {"type": _nullable("integer")}
    if all(
        isinstance(example, (int, float)) and not isinstance(example, bool)
        for example in non_null
    ):
        return {"type": _nullable("number")}
    if all(isinstance(example, str) for example in non_null):
        result: dict[str, Any] = {"type": _nullable("string")}
        if isinstance(hint, str):
            result["x-tscan-hint"] = hint
        return result
    if all(isinstance(example, list) for example in non_null):
        hinted_item = hint[0] if isinstance(hint, list) and hint else hint
        items = [item for example in non_null for item in example]
        item_schema = (
            _schema_from_examples(items, hinted_item)
            if items
            else shorthand_to_json_schema(hinted_item)
        )
        return {"type": ["array", "null"], "items": item_schema}
    if all(isinstance(example, dict) for example in non_null):
        hinted_properties = hint if isinstance(hint, dict) else {}
        keys = {str(key) for example in non_null for key in example}
        properties = {
            key: _schema_from_examples(
                [example[key] for example in non_null if key in example],
                hinted_properties.get(key),
            )
            for key in sorted(keys)
        }
        return {
            "type": ["object", "null"],
            "additionalProperties": True,
            "properties": properties,
        }

    # Some source arrays genuinely mix scalars and structured values. Retain that
    # information as a JSON type union instead of rejecting otherwise valid labels.
    return {"type": ["string", "number", "boolean", "object", "array", "null"]}


def _schema_from_example(example: Any, hint: Any = None) -> dict[str, Any]:
    return _schema_from_examples([example], hint)


def japanese_root_schema(
    value: Any,
    *,
    title: str = "Japanese document",
    example: Any = None,
) -> dict[str, Any]:
    schema = _schema_from_example(example, value) if example is not None else shorthand_to_json_schema(value)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = title
    return schema
