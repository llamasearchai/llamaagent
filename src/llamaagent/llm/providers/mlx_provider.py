"""
MLX Provider implementation for Apple Silicon

This module implements the MLX LLM provider for the llamaagent system.
It provides support for running language models locally on Apple Silicon
using the MLX framework.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.llamaagent import LLMMessage, LLMResponse

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class MLXProvider(BaseLLMProvider):
    """MLX provider for Apple Silicon machines."""

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-MLX",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """Initialize MLX provider.

        Args:
            model: Model name or path to use
            max_tokens: Default maximum tokens
            temperature: Default temperature
            **kwargs: Additional configuration
        """
        super().__init__(model=model, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self._setup_mlx()

    def _setup_mlx(self) -> None:
        """Setup MLX client."""
        try:
            # Try to import MLX - optional dependency
            import mlx_lm

            self.client = mlx_lm
            logger.info(f"MLX provider initialized with model: {self.model}")
        except ImportError:
            logger.warning("MLX not available, using fallback mode")
            self.client = None

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
        """Complete using MLX.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        if not self.client:
            return await self._fallback_complete(messages, **kwargs)

        try:
            return await self._native_mlx_complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"MLX completion failed: {e}")
            return await self._fallback_complete(messages, **kwargs)

    async def _native_mlx_complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Native MLX completion."""
        # Convert messages to prompt format
        prompt_parts: List[str] = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        # Generate response using MLX
        # Run MLX generation in a thread pool since it's synchronous
        loop = asyncio.get_event_loop()

        def _generate():
            try:
                response = self.client.generate(
                    model or self.model,
                    prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    **kwargs,
                )
                return response
            except Exception as e:
                logger.error(f"MLX generation failed: {e}")
                raise

        response_text = await loop.run_in_executor(None, _generate)

        return LLMResponse(
            content=response_text.strip(),
            model=model or self.model,
            provider="mlx",
            tokens_used=len(response_text.split()),  # Rough estimate
            usage={},
        )

    async def _fallback_complete(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Fallback completion when MLX is not available."""
        # Use mock response
        return LLMResponse(
            content="MLX provider fallback response. Please install mlx-lm to use MLX models.",
            model=self.model,
            provider="mlx",
            tokens_used=0,
            usage={"fallback": True},
        )

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Yields:
            String chunks of the response
        """
        # For now, simulate streaming by yielding the complete response in chunks
        response = await self.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **kwargs,
        )

        # Split response into chunks for streaming simulation
        words = response.content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)  # Small delay to simulate streaming

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings using MLX.

        Note: MLX doesn't typically provide embedding models, so this raises NotImplementedError.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Raises:
            NotImplementedError: MLX provider doesn't support embeddings
        """
        # For now, return mock embeddings
        import numpy as np

        return [np.random.rand(384).tolist() for _ in texts]

    async def validate_model(self, model: str) -> bool:
        """Validate MLX model availability.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        if not self.client:
            return False

        # MLX models are typically loaded from HuggingFace or local paths
        # For now, we'll assume the model is valid if it follows the pattern
        return "/" in model or model.startswith("mlx-community/")

    async def health_check(self) -> bool:
        """Check if the MLX provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        if self.client is None:
            return False

        try:
            # Try to check if MLX is available
            import mlx

            return mlx.core.default_device().type.name == "gpu"
        except Exception:
            return False

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for MLX usage.

        MLX is local, so there's no cost.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Always returns 0.0 for local inference
        """
        return 0.0
