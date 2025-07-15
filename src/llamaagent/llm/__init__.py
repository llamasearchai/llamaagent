"""
LLM Module for LlamaAgent

Provides unified access to all LLM providers and utilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Core message and response types
# Base classes
from .base import LLMProvider

# Exceptions
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
    TokenLimitError,
)

# Factory and utilities
from .factory import LLMFactory
from .messages import LLMMessage, LLMResponse

# Provider registry
from .providers import (
    MockProvider,
    ProviderFactory,
    create_provider,
    get_available_providers,
    get_provider_class,
    is_provider_available,
)
from .providers.base_provider import BaseLLMProvider

# Global factory instance
_global_factory = LLMFactory()


def get_factory() -> LLMFactory:
    """Get the global LLM factory instance."""
    return _global_factory


def list_providers() -> List[str]:
    """List all available providers."""
    return get_available_providers()


def list_models(provider: str) -> List[str]:
    """List available models for a provider."""
    return _global_factory.get_available_providers().get(provider, [])


def create_llm_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Create an LLM provider instance."""
    return _global_factory.create_provider(
        provider_type=provider_type, api_key=api_key, model_name=model_name, **kwargs
    )


async def quick_complete(
    provider_type: str,
    prompt: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Quick completion without needing to manage provider instances."""
    provider = create_llm_provider(
        provider_type, api_key=api_key, model_name=model_name, **kwargs
    )
    message = LLMMessage(role="user", content=prompt)
    response = await provider.complete([message])
    return response.content


def get_provider_info() -> Dict[str, Any]:
    """Get information about all available providers."""
    info = {
        "available_providers": get_available_providers(),
        "total_providers": len(get_available_providers()),
        "provider_models": _global_factory.get_available_providers(),
    }

    # Add detailed info for each provider
    provider_details: Dict[str, Any] = {}
    for provider_name in get_available_providers():
        provider_class = get_provider_class(provider_name)
        if provider_class:
            provider_details[provider_name] = {
                "class_name": provider_class.__name__,
                "module": provider_class.__module__,
                "available": is_provider_available(provider_name),
            }

    info["provider_details"] = provider_details
    return info


# Compatibility aliases
LLM = BaseLLMProvider
Provider = BaseLLMProvider
create_provider_direct = create_provider

__all__ = [
    # Core types
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "BaseLLMProvider",
    # Aliases
    "LLM",
    "Provider",
    # Factory
    "LLMFactory",
    "get_factory",
    # Providers
    "MockProvider",
    "ProviderFactory",
    "create_provider",
    "create_provider_direct",
    "get_available_providers",
    "get_provider_class",
    "is_provider_available",
    # Utilities
    "create_llm_provider",
    "quick_complete",
    "list_providers",
    "list_models",
    "get_provider_info",
    # Exceptions
    "LLMError",
    "AuthenticationError",
    "ConfigurationError",
    "ModelNotFoundError",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "TokenLimitError",
]
