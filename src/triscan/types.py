from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class ExtractionMode(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"


class DocumentLanguage(StrEnum):
    AUTO = "auto"
    JA = "ja"
    EN = "en"
    VI = "vi"
    KO = "ko"


@dataclass(slots=True)
class DocumentPage:
    page_number: int
    image: Image.Image
    width: int
    height: int


@dataclass(slots=True)
class LoadedDocument:
    source_path: Path
    original_filename: str
    media_type: str
    sha256: str
    byte_size: int
    pages: list[DocumentPage]


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["error", "warning"]
    path: str = "$"
    code: str
    message: str


class InferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    prompt_version: str
    page_count: int = Field(ge=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    document_id: str
    original_filename: str
    document_sha256: str
    document_type: str
    language: DocumentLanguage
    mode: ExtractionMode
    raw_output: str
    parsed_output: dict[str, Any] | list[Any] | None = None
    normalized_output: dict[str, Any] | list[Any] | None = None
    valid: bool | None = None
    repaired: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)
    inference: InferenceMetadata
    artifacts: dict[str, str] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    sample_count: int
    metrics: dict[str, float]
    breakdowns: dict[str, dict[str, float]] = Field(default_factory=dict)
