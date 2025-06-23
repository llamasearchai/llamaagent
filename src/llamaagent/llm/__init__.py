"""
LLM Module - Complete implementation for LlamaAgent
"""

from typing import Any, Union

from .factory import LLMFactory, ProviderFactory
from .providers import BaseProvider, OpenAIProvider, AnthropicProvider, TogetherProvider, CohereProvider
from .providers.mock_provider import MockProvider
from .messages import LLMMessage, LLMResponse
from .exceptions import LLMError, RateLimitError, AuthenticationError

# Compatibility alias
LLMProvider = BaseProvider

def create_provider(provider_type: str, **kwargs: Any) -> Union[MockProvider, OpenAIProvider, AnthropicProvider, TogetherProvider, CohereProvider]:
    """Factory function to create LLM providers."""
    provider_type = provider_type.lower()
    
    if provider_type == "mock":
        return MockProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        return AnthropicProvider(**kwargs)
    elif provider_type == "together":
        return TogetherProvider(**kwargs)
    elif provider_type == "cohere":
        return CohereProvider(**kwargs)
    else:
        # Default to mock for unknown providers
        return MockProvider(**kwargs)

__all__ = [
    'LLMFactory',
    'ProviderFactory',
    'BaseProvider',
    'LLMProvider',
    'OpenAIProvider', 
    'AnthropicProvider',
    'TogetherProvider',
    'CohereProvider',
    'MockProvider',
    'create_provider',
    'LLMMessage',
    'LLMResponse',
    'LLMError',
    'RateLimitError',
    'AuthenticationError'
]
