from __future__ import annotations

from typing import Any, Literal

TrainingPartition = Literal["train", "validation", "skip"]

EVALUATION_SPLITS = {"test", "eval", "evaluation", "external_test", "benchmark"}


def classify_training_row(row: dict[str, Any]) -> TrainingPartition:
    """Return the safe training partition for a prepared dataset row."""

    role = str(row.get("data_role", row.get("role", ""))).upper()
    split = str(row.get("split", "train")).lower()
    if role == "EVAL_ONLY" or split in EVALUATION_SPLITS:
        return "skip"
    if split in {"val", "validation", "dev"}:
        return "validation"
    return "train"
