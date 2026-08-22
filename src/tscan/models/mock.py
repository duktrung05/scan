from __future__ import annotations

import json
from typing import Any

from tscan.models.base import ModelRequest, ModelResponse, VisionModel
from tscan.types import ExtractionMode


def _mock_value(schema: dict[str, Any]) -> Any:
    value_type = schema.get("type")
    choices = value_type if isinstance(value_type, list) else [value_type]
    if "null" in choices:
        return None
    if "object" in choices or "properties" in schema:
        properties = schema.get("properties", {})
        return {name: _mock_value(child) for name, child in properties.items()}
    if "array" in choices:
        return []
    if "string" in choices:
        return ""
    if "number" in choices or "integer" in choices:
        return 0
    if "boolean" in choices:
        return False
    return None


class MockVisionModel(VisionModel):
    """Deterministic provider for local smoke tests; it does not perform OCR."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.mode == ExtractionMode.JSON:
            payload = _mock_value(request.schema or {"type": "object"})
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            blocks = [
                f"<!-- page: {page.page_number} -->\n\n"
                f"## Mock extraction for page {page.page_number}\n\n"
                "Configure `TSCAN_MODEL_PROVIDER` to run a real vision-language model."
                for page in request.pages
            ]
            text = "\n\n".join(blocks)
        return ModelResponse(
            text=text,
            provider="mock",
            model="tscan-mock",
            metadata={"warning": "Mock mode does not read document text."},
        )
