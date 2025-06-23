"""
LLM Factory for creating provider instances
"""

import os
from typing import Dict, Any, Optional
from .providers import BaseProvider, OpenAIProvider, AnthropicProvider, TogetherProvider, CohereProvider
from .exceptions import LLMError, AuthenticationError

class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    PROVIDER_CLASSES = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "together": TogetherProvider,
        "cohere": CohereProvider
    }
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        return {
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "together": os.getenv("TOGETHER_API_KEY", ""),
            "cohere": os.getenv("COHERE_API_KEY", "")
        }
    
    async def create_provider(
        self, 
        provider_type: str, 
        model_name: str,
        api_key: Optional[str] = None
    ) -> BaseProvider:
        """Create a provider instance."""
        
        provider_type = provider_type.lower()
        
        if provider_type not in self.PROVIDER_CLASSES:
            raise LLMError(f"Unsupported provider: {provider_type}")
        
        # Use provided API key or get from environment
        api_key = api_key or self._api_keys.get(provider_type)
        if not api_key:
            raise AuthenticationError(f"No API key found for {provider_type}")
        
        # Create cache key
        cache_key = f"{provider_type}:{model_name}:{hash(api_key)}"
        
        # Return cached provider if exists
        if cache_key in self._providers:
            return self._providers[cache_key]
        
        # Create new provider instance
        provider_class = self.PROVIDER_CLASSES[provider_type]
        provider = provider_class(api_key=api_key, model_name=model_name)
        
        # Validate the provider can access the model
        if hasattr(provider, 'validate_model'):
            is_valid = await provider.validate_model(model_name)
            if not is_valid:
                raise LLMError(f"Model {model_name} not available for provider {provider_type}")
        
        # Cache the provider
        self._providers[cache_key] = provider
        
        return provider
    
    def get_available_providers(self) -> Dict[str, list]:
        """Get list of available providers and their models."""
        return {
            "openai": [
                "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
                "gpt-4o", "gpt-4o-mini"
            ],
            "anthropic": [
                "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"
            ],
            "together": [
                "meta-llama/Llama-2-70b-chat-hf",
                "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ],
            "cohere": [
                "command", "command-light", "command-nightly"
            ]
        }
    
    def clear_cache(self):
        """Clear provider cache."""
        self._providers.clear()

# Compatibility alias
ProviderFactory = LLMFactory
