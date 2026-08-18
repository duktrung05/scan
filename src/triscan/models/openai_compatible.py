from __future__ import annotations

import base64
import io
from typing import Any

import httpx

from triscan.config import Settings
from triscan.exceptions import ModelError
from triscan.models.base import ModelRequest, ModelResponse, VisionModel


def _to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    raise ModelError("Model response content is not text")


class OpenAICompatibleVisionModel(VisionModel):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, request: ModelRequest) -> ModelResponse:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _to_data_url(page.image)}}
            for page in request.pages
        ]
        content.append({"type": "text", "text": request.prompt})
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.settings.max_new_tokens,
            "temperature": self.settings.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        endpoint = f"{self.settings.api_base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:1000]
            raise ModelError(
                f"Model server returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelError(f"Unable to call model server: {exc}") from exc
        try:
            choice = body["choices"][0]
            text = _extract_content(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("Model server returned an unexpected response shape") from exc
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return ModelResponse(
            text=text,
            provider="openai_compatible",
            model=self.settings.model_name,
            metadata={"usage": usage, "finish_reason": choice.get("finish_reason")},
        )
