from __future__ import annotations

from tscan.config import Settings
from tscan.exceptions import ConfigurationError
from tscan.models.base import VisionModel
from tscan.models.mock import MockVisionModel
from tscan.models.openai_compatible import OpenAICompatibleVisionModel
from tscan.models.transformers_local import TransformersVisionModel


def create_model(settings: Settings) -> VisionModel:
    if settings.model_provider == "mock":
        return MockVisionModel()
    if settings.model_provider == "openai_compatible":
        return OpenAICompatibleVisionModel(settings)
    if settings.model_provider == "transformers":
        return TransformersVisionModel(settings)
    raise ConfigurationError(f"Unknown model provider: {settings.model_provider}")
