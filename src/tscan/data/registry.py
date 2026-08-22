from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tscan.exceptions import ConfigurationError

DatasetUsage = Literal["train", "validation", "test", "eval_only", "reject"]


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    task: Literal["json_extraction", "markdown"] = "json_extraction"
    language: str
    document_type: str
    template_family: str
    jsonl: Path
    image_root: Path
    schema_id: str | None = None
    schema_path: Path | None = None
    source: str
    license: str
    usage: DatasetUsage
    weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def require_schema(self) -> DatasetEntry:
        if self.task == "json_extraction" and not (self.schema_id or self.schema_path):
            raise ValueError("json_extraction datasets require schema_id or schema_path")
        return self


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    datasets: list[DatasetEntry]
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def unique_names(self) -> DatasetRegistry:
        identities = [(item.name, item.version) for item in self.datasets]
        if len(identities) != len(set(identities)):
            raise ValueError("Dataset name/version pairs must be unique")
        return self


def load_dataset_registry(path: str | Path) -> DatasetRegistry:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        registry = DatasetRegistry.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to load dataset registry {source}: {exc}") from exc

    base = source.parent
    for entry in registry.datasets:
        if not entry.jsonl.is_absolute():
            entry.jsonl = (base / entry.jsonl).resolve()
        if not entry.image_root.is_absolute():
            entry.image_root = (base / entry.image_root).resolve()
        if entry.schema_path and not entry.schema_path.is_absolute():
            entry.schema_path = (base / entry.schema_path).resolve()
    registry.source_path = source
    return registry
