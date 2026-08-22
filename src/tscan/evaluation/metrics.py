from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from tscan.validation.cross_field import cross_field_issues
from tscan.validation.multipage import multipage_issues
from tscan.validation.schema import validate_schema

CRITICAL_FIELDS = {
    "invoice_number", "invoice_date", "date", "currency", "subtotal", "tax", "total"
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def levenshtein_distance(reference: list[Any], hypothesis: list[Any]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def text_metrics(reference: str, prediction: str, *, language: str | None = None) -> dict[str, float]:
    reference, prediction = normalize_text(reference), normalize_text(prediction)
    char_distance = levenshtein_distance(list(reference), list(prediction))
    word_distance = levenshtein_distance(reference.split(), prediction.split())
    cer = char_distance / max(1, len(reference))
    wer = word_distance / max(1, len(reference.split()))
    return {
        "cer": cer,
        "wer": wer,
        "primary_error_rate": cer if language in {"ja", "ko"} else wer,
        "normalized_edit_similarity": 1 - char_distance / max(1, len(reference), len(prediction)),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}[{index}]"))
        if not value:
            result[prefix] = []
    else:
        result[prefix] = value
    return result


def _canonical_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return normalize_text(value).casefold()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / max(1e-12, precision + recall)


def _is_amount_path(path: str) -> bool:
    leaf = re.sub(r"\[\d+\]", "", path).rsplit(".", 1)[-1].casefold()
    return any(token in leaf for token in ("amount", "total", "subtotal", "tax", "price", "discount"))


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _normalized_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        match = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", value.strip())
        if not match:
            return None
        day, month, year = (int(item) for item in match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None


def _required_paths(schema: dict[str, Any], prefix: str = "") -> set[str]:
    result = {f"{prefix}.{key}" if prefix else key for key in schema.get("required", [])}
    for key, child in schema.get("properties", {}).items():
        if isinstance(child, dict):
            result.update(_required_paths(child, f"{prefix}.{key}" if prefix else key))
    return result


def _containers(value: Any, prefix: str = "$") -> dict[str, Any]:
    result = {prefix: value}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(_containers(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_containers(child, f"{prefix}[{index}]"))
    return result


def _list_item_accuracy(reference: Any, prediction: Any) -> float:
    ref_lists = {key: value for key, value in _containers(reference).items() if isinstance(value, list)}
    pred_lists = {key: value for key, value in _containers(prediction).items() if isinstance(value, list)}
    if not ref_lists and not pred_lists:
        return 1.0
    scores = []
    for key in set(ref_lists) | set(pred_lists):
        ref_items = Counter(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in ref_lists.get(key, []))
        pred_items = Counter(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in pred_lists.get(key, []))
        correct = sum((ref_items & pred_items).values())
        scores.append(correct / max(1, sum((ref_items | pred_items).values())))
    return sum(scores) / len(scores)


def json_metrics(reference: Any, prediction: Any, *, schema: dict[str, Any] | None = None) -> dict[str, float]:
    ref, pred = flatten(reference), flatten(prediction)
    ref_keys, pred_keys = set(ref), set(pred)
    common = ref_keys & pred_keys
    key_precision = len(common) / max(1, len(pred_keys))
    key_recall = len(common) / max(1, len(ref_keys))
    correct = sum(_canonical_scalar(ref[key]) == _canonical_scalar(pred[key]) for key in common)
    field_precision = correct / max(1, len(pred_keys))
    field_recall = correct / max(1, len(ref_keys))

    critical_keys = {key for key in ref_keys | pred_keys if key.rsplit(".", 1)[-1] in CRITICAL_FIELDS}
    critical_correct = sum(key in ref and key in pred and _canonical_scalar(ref[key]) == _canonical_scalar(pred[key]) for key in critical_keys)
    amount_keys = {key for key in ref_keys | pred_keys if _is_amount_path(key)}
    amount_exact = amount_tolerant = 0
    for key in amount_keys:
        if key not in ref or key not in pred:
            continue
        ref_number, pred_number = _decimal(ref[key]), _decimal(pred[key])
        if ref_number is None or pred_number is None:
            continue
        amount_exact += ref_number == pred_number
        amount_tolerant += abs(ref_number - pred_number) <= max(Decimal("0.01"), abs(ref_number) * Decimal("0.001"))
    date_keys = {key for key in ref_keys | pred_keys if "date" in key.casefold()}
    date_raw_correct = sum(key in ref and key in pred and str(ref[key]) == str(pred[key]) for key in date_keys)
    date_normalized_correct = sum(key in ref and key in pred and _normalized_date(ref[key]) is not None and _normalized_date(ref[key]) == _normalized_date(pred[key]) for key in date_keys)
    currency_keys = {key for key in ref_keys | pred_keys if key.rsplit(".", 1)[-1].casefold() == "currency"}
    currency_correct = sum(key in ref and key in pred and str(ref[key]).upper() == str(pred[key]).upper() for key in currency_keys)
    required = _required_paths(schema or {})
    required_missing = sum(key not in pred or pred.get(key) is None for key in required)

    return {
        "json_parse_rate": 1.0,
        "key_precision": key_precision,
        "key_recall": key_recall,
        "key_f1": _f1(key_precision, key_recall),
        "field_accuracy": correct / max(1, len(ref_keys | pred_keys)),
        "field_precision": field_precision,
        "field_recall": field_recall,
        "field_value_f1": _f1(field_precision, field_recall),
        "field_value_accuracy": correct / max(1, len(common)),
        "critical_field_accuracy": critical_correct / max(1, len(critical_keys)),
        "required_field_missing_rate": required_missing / max(1, len(required)),
        "hallucinated_field_rate": len(pred_keys - ref_keys) / max(1, len(pred_keys)),
        "amount_exact_accuracy": amount_exact / max(1, len(amount_keys)),
        "amount_tolerance_accuracy": amount_tolerant / max(1, len(amount_keys)),
        "currency_accuracy": currency_correct / max(1, len(currency_keys)),
        "date_raw_accuracy": date_raw_correct / max(1, len(date_keys)),
        "date_normalized_accuracy": date_normalized_correct / max(1, len(date_keys)),
        "list_item_accuracy": _list_item_accuracy(reference, prediction),
        "schema_valid_rate": float(not validate_schema(prediction, schema)) if schema else 1.0,
        "cross_field_consistency_rate": float(not cross_field_issues(prediction)),
        "multi_page_consistency_rate": float(not multipage_issues(prediction)),
        "exact_match": float(json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == json.dumps(prediction, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
    }
