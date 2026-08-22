from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tscan.types import DocumentLanguage, DocumentPage, ExtractionMode


@dataclass(slots=True)
class ModelRequest:
    pages: list[DocumentPage]
    prompt: str
    mode: ExtractionMode
    document_type: str
    language: DocumentLanguage
    schema: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionModel(ABC):
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate text from document pages and an extraction prompt."""
