from __future__ import annotations

from triscan.config import Settings
from triscan.exceptions import ConfigurationError
from triscan.models.base import VisionModel
from triscan.models.mock import MockVisionModel
from triscan.models.openai_compatible import OpenAICompatibleVisionModel
from triscan.models.transformers_local import TransformersVisionModel


def create_model(settings: Settings) -> VisionModel:
    if settings.model_provider == "mock":
        return MockVisionModel()
    if settings.model_provider == "openai_compatible":
        return OpenAICompatibleVisionModel(settings)
    if settings.model_provider == "transformers":
        return TransformersVisionModel(settings)
    raise ConfigurationError(f"Unknown model provider: {settings.model_provider}")
