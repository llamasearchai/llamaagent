"""
Base LLM Provider class

This module provides the abstract base class for all LLM providers in the llamaagent system.
It defines the interface that all concrete provider implementations must follow.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...types import LLMMessage, LLMResponse


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None, **kwargs: Any
    ):
        """Initialize the provider with configuration.

        Args:
            api_key: API key for the provider (if required)
            model: Default model name to use
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model or "default"
        self.model_name = self.model  # Alias for compatibility
        self.provider_name = self.__class__.__name__.lower().replace("provider", "")
        self.kwargs = kwargs

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the generated text and metadata
        """

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation with messages.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the assistant's reply
        """

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing embeddings and metadata

        Raises:
            NotImplementedError: If the provider doesn't support embeddings
        """
        logger.warning(
            f"Embeddings not supported by {self.provider_name}, returning empty embeddings"
        )
        return {
            "embeddings": [[0.0] * 768],  # Default embedding dimension
            "model": "none",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @abstractmethod
    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from the LLM.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Yields:
            String chunks of the response as they are generated
        """

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion

        Returns:
            Estimated cost in USD
        """
        return 0.0  # Override in subclasses with actual pricing

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        return True  # Override in subclasses

    async def health_check(self) -> bool:
        """Check if the provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        return True  # Override in subclasses

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model='{self.model_name}', provider='{self.provider_name}')"
