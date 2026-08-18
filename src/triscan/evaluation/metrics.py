from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

CRITICAL_FIELDS = {
    "invoice_number",
    "invoice_date",
    "date",
    "currency",
    "subtotal",
    "tax",
    "total",
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
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def text_metrics(reference: str, prediction: str) -> dict[str, float]:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    ref_chars = list(reference)
    pred_chars = list(prediction)
    ref_words = reference.split()
    pred_words = prediction.split()
    char_distance = levenshtein_distance(ref_chars, pred_chars)
    word_distance = levenshtein_distance(ref_words, pred_words)
    return {
        "cer": char_distance / max(1, len(ref_chars)),
        "wer": word_distance / max(1, len(ref_words)),
        "normalized_edit_similarity": 1 - char_distance / max(1, len(ref_chars), len(pred_chars)),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, path))
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


def json_metrics(reference: Any, prediction: Any) -> dict[str, float]:
    ref = flatten(reference)
    pred = flatten(prediction)
    ref_keys = set(ref)
    pred_keys = set(pred)
    common = ref_keys & pred_keys
    precision = len(common) / max(1, len(pred_keys))
    recall = len(common) / max(1, len(ref_keys))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    correct = sum(_canonical_scalar(ref[key]) == _canonical_scalar(pred[key]) for key in common)
    field_accuracy = correct / max(1, len(ref_keys | pred_keys))

    critical_keys = {
        key for key in ref_keys | pred_keys if key.rsplit(".", 1)[-1] in CRITICAL_FIELDS
    }
    critical_correct = sum(
        key in ref and key in pred and _canonical_scalar(ref[key]) == _canonical_scalar(pred[key])
        for key in critical_keys
    )
    return {
        "key_precision": precision,
        "key_recall": recall,
        "key_f1": f1,
        "field_accuracy": field_accuracy,
        "critical_field_accuracy": critical_correct / max(1, len(critical_keys)),
        "exact_match": float(
            json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            == json.dumps(prediction, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
    }
