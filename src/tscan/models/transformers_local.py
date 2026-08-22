from __future__ import annotations

from typing import Any

from tscan.config import Settings
from tscan.exceptions import ConfigurationError, ModelError
from tscan.models.base import ModelRequest, ModelResponse, VisionModel


def _assistant_text(generated: Any) -> str:
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
    raise ModelError("Local model returned an unexpected generated_text value")


class TransformersVisionModel(VisionModel):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ConfigurationError(
                "Local Transformers provider requires `uv sync --extra local`"
            ) from exc
        try:
            self.pipe = pipeline(
                "image-text-to-text",
                model=settings.model_name,
                revision=settings.model_revision,
                device_map="auto",
                dtype="auto",
            )
        except Exception as exc:
            raise ModelError(f"Unable to load local model {settings.model_name}: {exc}") from exc

    def generate(self, request: ModelRequest) -> ModelResponse:
        visual_content = [{"type": "image", "image": page.image} for page in request.pages]
        visual_content.append({"type": "text", "text": request.prompt})
        messages = [{"role": "user", "content": visual_content}]
        generation: dict[str, Any] = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            generation["temperature"] = self.settings.temperature
        try:
            output = self.pipe(text=messages, **generation)
            text = _assistant_text(output[0]["generated_text"])
        except Exception as exc:
            raise ModelError(f"Local inference failed: {exc}") from exc
        return ModelResponse(
            text=text,
            provider="transformers",
            model=self.settings.model_name,
            metadata={"model_revision": self.settings.model_revision},
        )
