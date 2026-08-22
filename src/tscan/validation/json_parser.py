from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class JsonParseResult:
    value: dict[str, Any] | list[Any] | None
    repaired_text: str
    repaired: bool
    actions: list[str] = field(default_factory=list)
    error: str | None = None


def _strip_code_fence(text: str) -> tuple[str, bool]:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.I | re.S)
    if match:
        return match.group(1), True
    return text.strip(), False


def _extract_balanced_json(text: str) -> str | None:
    starts = [position for token in ("{", "[") if (position := text.find(token)) >= 0]
    if not starts:
        return None
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def _loads_container(text: str) -> dict[str, Any] | list[Any]:
    value = json.loads(text)
    if not isinstance(value, (dict, list)):
        raise ValueError("Top-level JSON output must be an object or array")
    return value


def parse_json_output(raw: str) -> JsonParseResult:
    candidate, fenced = _strip_code_fence(raw)
    actions = ["removed_code_fence"] if fenced else []
    try:
        return JsonParseResult(_loads_container(candidate), candidate, fenced, actions)
    except (json.JSONDecodeError, ValueError) as first_error:
        last_error: Exception = first_error

    balanced = _extract_balanced_json(candidate)
    if balanced and balanced != candidate:
        candidate = balanced
        actions.append("extracted_balanced_json")
        try:
            return JsonParseResult(_loads_container(candidate), candidate, True, actions)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
    if without_trailing_commas != candidate:
        candidate = without_trailing_commas
        actions.append("removed_trailing_commas")
        try:
            return JsonParseResult(_loads_container(candidate), candidate, True, actions)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

    try:
        python_value = ast.literal_eval(candidate)
        if isinstance(python_value, (dict, list)):
            normalized_text = json.dumps(python_value, ensure_ascii=False)
            actions.append("converted_python_literal")
            return JsonParseResult(python_value, normalized_text, True, actions)
    except (ValueError, SyntaxError) as exc:
        last_error = exc

    return JsonParseResult(
        value=None,
        repaired_text=candidate,
        repaired=bool(actions),
        actions=actions,
        error=str(last_error),
    )
