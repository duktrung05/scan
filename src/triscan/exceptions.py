class TriScanError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(TriScanError):
    """Raised when runtime configuration is incomplete or invalid."""


class DocumentError(TriScanError):
    """Raised when an input document is unsupported or unsafe to process."""


class ModelError(TriScanError):
    """Raised when the model provider fails or returns an unusable response."""


class ExtractionError(TriScanError):
    """Raised when an extraction pipeline cannot complete."""
