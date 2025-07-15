"""
LLM provider exceptions

Author: Nik Jois <nikjois@llamasearch.ai>
"""


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class AuthenticationError(LLMError):
    """Raised when API authentication fails."""


class RateLimitError(LLMError):
    """Raised when rate limits are exceeded."""


class ModelNotFoundError(LLMError):
    """Raised when a requested model is not available."""


class TokenLimitError(LLMError):
    """Raised when token limits are exceeded."""


class ProviderError(LLMError):
    """Raised when a provider-specific error occurs."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.provider = provider


class ConfigurationError(LLMError):
    """Raised when provider configuration is invalid."""


class NetworkError(LLMError):
    """Raised when network-related errors occur."""
