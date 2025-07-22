"""
LiteLLM Provider implementation

This module implements the LiteLLM provider for the llamaagent system.
LiteLLM is a unified interface that supports 100+ LLM providers including
OpenAI, Anthropic, Cohere, Hugging Face, and more.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.llamaagent import LLMMessage, LLMResponse

from ..exceptions import LLMError
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

# Try to import LiteLLM
try:
    import litellm
    from litellm import acompletion, aembedding

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None
    acompletion = None
    aembedding = None


class LiteLLMProvider(BaseLLMProvider):
    """
    LiteLLM Provider with support for multiple LLM backends.

    Features:
    - Universal LLM interface through LiteLLM
    - Support for 100+ models (OpenAI, Anthropic, Cohere, etc.)
    - Automatic model selection based on task type
    - Cost tracking and budget management
    - Real-time streaming responses
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ):
        """Initialize LiteLLM provider.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3", "command-r")
            api_key: API key for the provider (optional, can use env vars)
            temperature: Default temperature for generation
            max_tokens: Default max tokens
            **kwargs: Additional configuration
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install with: pip install litellm"
            )

        super().__init__(api_key=api_key, model=model, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure LiteLLM
        if api_key:
            litellm.api_key = api_key

        # Set up model configurations
        self._setup_model_configs()

        logger.info(f"Initialized LiteLLMProvider with model: {model}")

    def _setup_model_configs(self) -> None:
        """Configure LiteLLM model settings"""
        if not litellm:
            return

        # Configure LiteLLM settings
        litellm.set_verbose = False  # Reduce logging noise

        # You can add custom model configurations here if needed
        # For example, custom endpoints, timeouts, etc.

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the assistant's reply
        """
        return await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **kwargs,
        )

    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a conversation with LiteLLM.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        try:
            # Format messages for LiteLLM
            formatted_messages = self._format_messages(messages)

            # Make the completion call
            response = await acompletion(
                model=model or self.model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract response data
            content = response.choices[0].message.content or ""
            usage = response.usage.dict() if response.usage else {}

            # Calculate cost if available
            cost = self._calculate_cost(response)

            # Add cost to usage dict
            if usage:
                usage["cost"] = cost
            else:
                usage = {"cost": cost}

            # Convert to our response format
            return LLMResponse(
                content=content,
                model=response.model,
                provider="litellm",
                tokens_used=usage.get("total_tokens", 0),
                usage=usage,
            )

        except Exception as e:
            logger.error(f"LiteLLM completion error: {e}")
            raise LLMError(f"Completion failed: {str(e)}")

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with LiteLLM.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Yields:
            String chunks of the response
        """
        try:
            formatted_messages = self._format_messages(messages)

            # Stream the response
            stream = await acompletion(
                model=model or self.model,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            raise LLMError(f"Streaming failed: {str(e)}")

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings using LiteLLM.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary containing embeddings and metadata
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            embedding_model = model or "text-embedding-ada-002"

            # Generate embeddings
            response = await aembedding(
                model=embedding_model,
                input=texts,
                **kwargs,
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            usage = response.usage.dict() if response.usage else {}
            cost = self._calculate_embedding_cost(len(texts))

            # Add cost to usage dict
            if usage:
                usage["cost"] = cost
            else:
                usage = {"cost": cost}

            return {
                "embeddings": embeddings,
                "model": embedding_model,
                "usage": usage,
            }

        except Exception as e:
            logger.error(f"LiteLLM embedding error: {e}")
            raise LLMError(f"Embedding failed: {str(e)}")

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for LiteLLM."""
        formatted: List[Dict[str, Any]] = []
        for msg in messages:
            formatted.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        return formatted

    def _calculate_cost(self, response: Any) -> float:
        """Calculate cost based on LiteLLM response."""
        try:
            # LiteLLM provides cost calculation
            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                return response._hidden_params["response_cost"]

            # Fallback to token-based calculation
            if hasattr(response, "usage") and response.usage:
                # Basic cost calculation (would be enhanced with real pricing)
                total_tokens = response.usage.total_tokens
                return total_tokens * 0.00002  # Example rate
            return 0.0
        except Exception:
            return 0.0

    def _calculate_embedding_cost(self, num_texts: int) -> float:
        """Calculate embedding cost."""
        # Example rate for embeddings
        return num_texts * 0.0001

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available through LiteLLM.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        try:
            # LiteLLM supports many models, we'll do a basic check
            # by trying to get model info

            # Check if it's a known model pattern
            known_prefixes = [
                "gpt-",
                "claude-",
                "command",
                "j2-",
                "text-",
                "mistral",
                "mixtral",
                "llama",
                "palm",
                "gemini",
            ]

            return any(model.startswith(prefix) for prefix in known_prefixes)
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check if the LiteLLM provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        try:
            # Test with a simple completion
            test_message = LLMMessage(role="user", content="Hello")
            response = await self.complete(
                messages=[test_message],
                max_tokens=10,
                temperature=0.1,
            )
            return bool(response.content)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # LiteLLM handles cost calculation internally, but we provide a fallback
        total_tokens = prompt_tokens + completion_tokens
        # Generic rate - actual cost depends on the model
        return total_tokens * 0.00002
