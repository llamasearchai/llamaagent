"""
LLM Provider Registry and Factory

This module provides a centralized registry for LLM providers and a factory
for creating provider instances. It supports multiple provider types including
OpenAI, Anthropic, Cohere, and mock providers for testing.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseLLMProvider
from .providers.mock_provider import MockProvider
from .providers.mock_provider import MockProvider as _MockProvider
from .providers.openai_provider import OpenAIProvider

# Import optional providers with graceful fallback
try:
    from .providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicProvider = None

try:
    from .providers.cohere_provider import CohereProvider
except ImportError:
    CohereProvider = None

# Import message types
from ..types import LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

# Registry of available LLM providers
llm_provider_registry: Dict[str, Type[BaseLLMProvider]] = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
}

# Aliases for backward compatibility
MockLLMProvider = MockProvider
LLMProvider = BaseLLMProvider


def register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """Register a new LLM provider"""
    llm_provider_registry[name] = provider_class
    logger.info(f"Registered LLM provider: {name}")


def get_provider(name: str) -> Optional[Type[BaseLLMProvider]]:
    """Get a provider class by name"""
    return llm_provider_registry.get(name)


def create_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Create a provider instance by name"""
    provider_class = get_provider(name)
    if not provider_class:
        raise ValueError(f"Unknown LLM provider: {name}")

    try:
        return provider_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create provider {name}: {e}")
        raise


def list_providers() -> list[str]:
    """List all available provider names"""
    return list(llm_provider_registry.keys())


def is_provider_available(name: str) -> bool:
    """Check if a provider is available"""
    return name in llm_provider_registry


# Export main classes and functions
# Factory function alias
LLMFactory = create_provider

__all__ = [
    "BaseLLMProvider",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAIProvider",
    "MockProvider",
    "_MockProvider",
    "LLMMessage",
    "LLMResponse",
    "LLMFactory",
    "register_provider",
    "get_provider",
    "create_provider",
    "list_providers",
    "is_provider_available",
    "llm_provider_registry",
]

# Add optional providers if available
if AnthropicProvider is not None:
    __all__.append("AnthropicProvider")
    llm_provider_registry["anthropic"] = AnthropicProvider

if CohereProvider is not None:
    __all__.append("CohereProvider")
    llm_provider_registry["cohere"] = CohereProvider
