class TScanError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(TScanError):
    """Raised when runtime configuration is incomplete or invalid."""


class DocumentError(TScanError):
    """Raised when an input document is unsupported or unsafe to process."""


class ModelError(TScanError):
    """Raised when the model provider fails or returns an unusable response."""


class ExtractionError(TScanError):
    """Raised when an extraction pipeline cannot complete."""
