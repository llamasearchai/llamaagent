"""
Together AI Provider implementation for LlamaAgent.

This module provides integration with Together AI's model serving platform.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.llamaagent import LLMMessage, LLMResponse

from .base_provider import BaseLLMProvider


class TogetherProvider(BaseLLMProvider):
    """Together AI LLM Provider."""

    BASE_URL = "https://api.together.xyz/v1"

    # Popular models and their approximate pricing
    PRICING = {
        # Meta Llama models
        "meta-llama/Llama-2-70b-chat-hf": {"input": 0.0009, "output": 0.0009},
        "meta-llama/Llama-2-13b-chat-hf": {"input": 0.00025, "output": 0.00025},
        "meta-llama/Llama-2-7b-chat-hf": {"input": 0.0002, "output": 0.0002},
        # Mistral models
        "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.0006, "output": 0.0006},
        "mistralai/Mistral-7B-Instruct-v0.1": {"input": 0.0002, "output": 0.0002},
        # Default pricing for unknown models
        "default": {"input": 0.0008, "output": 0.0008},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Together provider.

        Args:
            api_key: Together API key (defaults to TOGETHER_API_KEY env var)
            model: Model name (defaults to meta-llama/Llama-2-70b-chat-hf)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("TOGETHER_API_KEY")
        model = model or "meta-llama/Llama-2-70b-chat-hf"
        super().__init__(api_key=api_key, model=model, **kwargs)

        # Lazy import to avoid dependency issues
        self._client = None

    def _get_client(self) -> Any:
        """Get or create Together client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                # Together uses OpenAI-compatible API
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.BASE_URL,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client

    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Together's API.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            model: Model to use (overrides default)
            **kwargs: Additional Together-specific parameters

        Returns:
            LLMResponse with the generated content
        """
        client = self._get_client()
        model = model or self.model_name

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider="together",
                tokens_used=usage["total_tokens"],
                usage=usage,
            )

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise ValueError(f"Authentication error: {error_msg}")
            elif "rate" in error_msg.lower():
                raise ValueError(f"Rate limit error: {error_msg}")
            else:
                raise ValueError(f"Together API error: {error_msg}")

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from a simple prompt."""
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.complete(messages, max_tokens, temperature, **kwargs)

    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation - delegates to complete."""
        return await self.complete(messages, max_tokens, temperature, model, **kwargs)

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream response from Together."""
        client = self._get_client()
        model = model or self.model_name

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ValueError(f"Streaming error: {str(e)}")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Together usage."""
        # Get pricing for the model or use default
        pricing = self.PRICING.get(self.model_name, self.PRICING["default"])

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available."""
        # Together supports many models, so we'll be permissive
        return True

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Try a minimal completion
            await self.complete(
                [LLMMessage(role="user", content="test")], max_tokens=10
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "together",
            "model": self.model_name,
            "description": "Together AI - Open source model hosting",
            "context_length": 4096,  # Varies by model
            "supports_streaming": True,
            "supports_embeddings": False,
            "pricing": self.PRICING.get(self.model_name, self.PRICING["default"]),
        }
