"""
LLM Providers Module

Comprehensive provider system with automatic fallbacks and dependency management.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    TOGETHER = "together"
    OLLAMA = "ollama"
    MLX = "mlx"
    CUDA = "cuda"
    LLAMA_LOCAL = "llama_local"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider_type: ProviderType
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Provider-specific configs
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    quantization: Optional[str] = None
    context_length: int = 4096
    batch_size: int = 1


@dataclass
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    metadata: Dict[str, Any]

# Base provider - always available
from .base import BaseProvider
from .base_provider import BaseLLMProvider

# Mock provider - always available for testing/fallback
from .mock_provider import MockProvider

# Provider registry
_AVAILABLE_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "mock": MockProvider,
}

# Try to import optional providers
try:
    from .openai_provider import OpenAIProvider

    _AVAILABLE_PROVIDERS["openai"] = OpenAIProvider
    logger.debug("OpenAI provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"OpenAI provider not available: {e}")

try:
    from .anthropic import AnthropicProvider

    _AVAILABLE_PROVIDERS["anthropic"] = AnthropicProvider
    logger.debug("Anthropic provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"Anthropic provider not available: {e}")

try:
    from .cohere import CohereProvider

    _AVAILABLE_PROVIDERS["cohere"] = CohereProvider
    logger.debug("Cohere provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"Cohere provider not available: {e}")

try:
    from .together import TogetherProvider

    _AVAILABLE_PROVIDERS["together"] = TogetherProvider
    logger.debug("Together provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"Together provider not available: {e}")

try:
    from .ollama_provider import OllamaProvider

    _AVAILABLE_PROVIDERS["ollama"] = OllamaProvider
    logger.debug("Ollama provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"Ollama provider not available: {e}")

try:
    from .mlx_provider import MLXProvider

    _AVAILABLE_PROVIDERS["mlx"] = MLXProvider
    logger.debug("MLX provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"MLX provider not available: {e}")

try:
    from .cuda_provider import CUDAProvider

    _AVAILABLE_PROVIDERS["cuda"] = CUDAProvider
    logger.debug("CUDA provider available")
except (ImportError, SyntaxError) as e:
    logger.debug(f"CUDA provider not available: {e}")


def get_available_providers() -> List[str]:
    """Get list of available provider names."""
    return list(_AVAILABLE_PROVIDERS.keys())


def get_provider_class(provider_name: str) -> Optional[Type[BaseLLMProvider]]:
    """Get provider class by name."""
    return _AVAILABLE_PROVIDERS.get(provider_name.lower())


def is_provider_available(provider_name: str) -> bool:
    """Check if a provider is available."""
    return provider_name.lower() in _AVAILABLE_PROVIDERS


def create_provider(
    provider_type: Optional[str] = None,
    provider_name: Optional[str] = None,  # Legacy compatibility
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Create provider instance with enhanced configuration support."""
    # Handle both new and legacy parameter names
    provider = provider_type or provider_name or "mock"
    
    provider_class = get_provider_class(provider)

    if not provider_class:
        raise ValueError(f"Provider '{provider}' not available. Available providers: {get_available_providers()}")

    # Enhanced provider creation with better defaults
    if provider == "mock":
        return provider_class(model_name=model_name or "mock-model")
    elif provider == "openai":
        # Use environment variable if no API key provided
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key provided, falling back to mock provider")
            return MockProvider(model_name="mock-openai")
        return provider_class(api_key=api_key, model_name=model_name or "gpt-3.5-turbo", **kwargs)
    else:
        return provider_class(api_key=api_key, model_name=model_name, **kwargs)


def list_available_providers() -> List[str]:
    """List available provider types as strings."""
    return get_available_providers()


# Export all available providers and functions
__all__ = [
    # New types and configs
    "ProviderType",
    "LLMConfig", 
    "LLMResponse",
    # Base classes
    "BaseLLMProvider",
    "BaseProvider",
    # Always available
    "MockProvider",
    # Functions
    "get_available_providers",
    "get_provider_class",
    "is_provider_available",
    "create_provider",
    "list_available_providers",
]

# Add conditionally available providers to exports
_exported_providers = []
for provider_name, provider_class in _AVAILABLE_PROVIDERS.items():
    if provider_name != "mock":
        globals()[provider_class.__name__] = provider_class
        _exported_providers.append(provider_class.__name__)

__all__.extend(_exported_providers)

# Provider factory alias for compatibility
ProviderFactory = create_provider
