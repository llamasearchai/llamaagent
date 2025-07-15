"""
LLM Factory for creating provider instances.

This module provides a factory pattern for creating LLM provider instances
with support for multiple providers, caching, and environment variables.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Import base provider - this is required for the system to work
from .providers.base_provider import BaseLLMProvider

# Import providers with optional dependencies
_PROVIDER_IMPORTS = {
    "openai": (".providers.openai_provider", "OpenAIProvider"),
    "anthropic": (".providers.anthropic", "AnthropicProvider"),
    "cohere": (".providers.cohere_provider", "CohereProvider"),
    "together": (".providers.together", "TogetherProvider"),
    "ollama": (".providers.ollama_provider", "OllamaProvider"),
    "mock": (".providers.mock_provider", "MockProvider"),
}

# Dictionary to store successfully imported provider classes
AVAILABLE_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {}

# Try to import each provider
for provider_name, (module_path, class_name) in _PROVIDER_IMPORTS.items():
    try:
        if provider_name == "openai":
            from .providers.openai_provider import OpenAIProvider

            AVAILABLE_PROVIDERS["openai"] = OpenAIProvider
        elif provider_name == "anthropic":
            from .providers.anthropic import AnthropicProvider

            AVAILABLE_PROVIDERS["anthropic"] = AnthropicProvider
        elif provider_name == "cohere":
            # Cohere provider temporarily disabled due to corrupted file
            pass
        elif provider_name == "together":
            from .providers.together import TogetherProvider

            AVAILABLE_PROVIDERS["together"] = TogetherProvider
        elif provider_name == "ollama":
            from .providers.ollama_provider import OllamaProvider

            AVAILABLE_PROVIDERS["ollama"] = OllamaProvider
        elif provider_name == "mock":
            from .providers.mock_provider import MockProvider

            AVAILABLE_PROVIDERS["mock"] = MockProvider
    except (ImportError, SyntaxError) as e:
        logger.debug(f"Could not import {provider_name} provider: {e}")
        # Provider-specific dependencies not installed or syntax errors, which is fine

# Import mock provider from the proper location
from .providers.mock_provider import MockProvider

# Ensure mock provider is always available
if "mock" not in AVAILABLE_PROVIDERS:
    AVAILABLE_PROVIDERS["mock"] = MockProvider


class LLMFactory:
    """Factory for creating LLM provider instances."""

    # Class-level mapping of provider names to their classes
    PROVIDER_CLASSES: Dict[str, Type[BaseLLMProvider]] = AVAILABLE_PROVIDERS

    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "cohere": "command",
        "together": "meta-llama/Llama-2-70b-chat-hf",
        "ollama": "llama3.2:3b",
        "mock": "mock-model",
    }

    # Available models for each provider
    PROVIDER_MODELS = {
        "openai": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
        ],
        "cohere": [
            "command",
            "command-light",
            "command-nightly",
        ],
        "together": [
            "meta-llama/Llama-2-70b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        "ollama": [
            "llama3.2:3b",
            "llama3.2:1b",
            "mistral:7b",
        ],
        "mock": [
            "mock-model",
            "test-model",
        ],
    }

    def __init__(self) -> None:
        """Initialize the factory with empty cache."""
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._api_keys: Dict[str, str] = self._load_api_keys()

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "cohere": os.getenv("COHERE_API_KEY", ""),
            "together": os.getenv("TOGETHER_API_KEY", ""),
            "ollama": "ollama-key",  # Ollama doesn't need real API key
            "mock": "mock-key",  # Mock doesn't need real API key
        }

    def get_provider(
        self,
        provider_type: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create or return a cached provider instance.

        Args:
            provider_type: Type of provider (openai, anthropic, etc.)
            model_name: Model to use (defaults to provider's default)
            api_key: API key (defaults to environment variable)
            **kwargs: Additional provider-specific parameters

        Returns:
            Provider instance

        Raises:
            ValueError: If provider requires API key and none provided
        """
        # Require explicit provider type
        if provider_type is None:
            raise ValueError("Provider type must be specified")

        provider_type = provider_type.lower()

        # Check if provider is supported
        if provider_type not in self.PROVIDER_CLASSES:
            raise ValueError(
                f"Unsupported provider: {provider_type}. Available providers: {list(self.PROVIDER_CLASSES.keys())}"
            )

        # Use provided API key or get from environment
        if api_key is None:
            api_key = self._api_keys.get(provider_type, "")

        # Validate API key for providers that require it
        def _is_valid_api_key(key: str, provider: str) -> bool:
            """Check if API key is valid (not placeholder or empty)."""
            if not key or key.strip() == "":
                return False
            
            # Reject common placeholder values
            placeholder_patterns = [
                "${OPENAI_API_KEY}",
                "${ANTHROPIC_API_KEY}",
                "${COHERE_API_KEY}",
                "${TOGETHER_API_KEY}",
                f"your_{provider}_api_key",
                f"your-{provider}-api-key",
                "replace-with-your-key",
                "your_key_here",
                "INSERT_YOUR_KEY_HERE",
                "ADD_YOUR_KEY_HERE",
                "sk-placeholder",
                "your_api_key_here",
                "your-api-key-here"
            ]
            
            if key.lower() in [p.lower() for p in placeholder_patterns]:
                return False
            
            # Allow test keys and real keys
            if key == "test-key" or key.startswith("sk-") or key.startswith("claude-"):
                return True
                
            # For other providers, accept any non-placeholder key
            return True
        
        if provider_type == "openai" and not _is_valid_api_key(api_key, "openai"):
            raise ValueError("OpenAI API key is required and cannot be a placeholder value")
        elif provider_type == "anthropic" and not _is_valid_api_key(api_key, "anthropic"):
            raise ValueError("Anthropic API key is required and cannot be a placeholder value")
        elif provider_type == "cohere" and not _is_valid_api_key(api_key, "cohere"):
            raise ValueError("Cohere API key is required and cannot be a placeholder value")
        elif provider_type == "together" and not _is_valid_api_key(api_key, "together"):
            raise ValueError("Together API key is required and cannot be a placeholder value")

        # Use default model if not provided
        if not model_name:
            model_name = self.DEFAULT_MODELS.get(provider_type, "default-model")

        # Create cache key
        cache_key = f"{provider_type}:{model_name}:{hash(api_key)}"

        # Return cached provider if exists
        if cache_key in self._providers:
            return self._providers[cache_key]

        # Create new provider instance
        provider_class = self.PROVIDER_CLASSES[provider_type]

        # Create provider with appropriate arguments
        if provider_type == "mock":
            provider = provider_class(
                api_key=api_key, model_name=model_name, **kwargs
            )
        else:
            provider = provider_class(api_key=api_key, model=model_name, **kwargs)

        # Cache the provider
        self._providers[cache_key] = provider

        logger.info(f"Created {provider_type} provider with model {model_name}")
        return provider

    def create_provider(self, provider_type: str, **kwargs: Any) -> BaseLLMProvider:
        """Alternative method name for compatibility."""
        return self.get_provider(provider_type, **kwargs)

    def create(self, **kwargs: Any) -> BaseLLMProvider:
        """Create provider with kwargs-only interface."""
        provider_type = kwargs.pop("provider_type", None)
        return self.get_provider(provider_type, **kwargs)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider types.

        Returns:
            List of provider names that can be instantiated
        """
        return list(self.PROVIDER_CLASSES.keys())

    def get_models_for_provider(self, provider_type: str) -> List[str]:
        """Get available models for a specific provider.

        Args:
            provider_type: Name of the provider

        Returns:
            List of available model names
        """
        provider_type = provider_type.lower()
        return self.PROVIDER_MODELS.get(provider_type, [])

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._providers.clear()


# Create global factory instance
_factory = LLMFactory()


# Convenience functions
def create_provider(provider_type: str, **kwargs: Any) -> BaseLLMProvider:
    """Create a provider using the global factory."""
    return _factory.get_provider(provider_type, **kwargs)


def get_available_providers() -> List[str]:
    """Get available providers from the global factory."""
    return _factory.get_available_providers()


def get_models_for_provider(provider_type: str) -> List[str]:
    """Get models for a provider from the global factory."""
    return _factory.get_models_for_provider(provider_type)


# Compatibility alias
ProviderFactory = LLMFactory

# Re-export for convenience
__all__ = [
    "LLMFactory",
    "ProviderFactory",
    "create_provider",
    "get_available_providers",
    "get_models_for_provider",
    "BaseLLMProvider",
]
