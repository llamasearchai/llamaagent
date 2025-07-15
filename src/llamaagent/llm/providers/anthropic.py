"""
Anthropic Provider implementation for LlamaAgent.

This module provides integration with Anthropic's Claude models.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
from typing import Any, AsyncGenerator, List, Optional

from src.llamaagent import LLMMessage, LLMResponse

from .base_provider import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) LLM Provider."""

    BASE_URL = "https://api.anthropic.com/v1"
    ANTHROPIC_VERSION = "2023-06-01"

    # Model pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (defaults to claude-3-sonnet-20240229)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        model = model or "claude-3-sonnet-20240229"
        super().__init__(api_key=api_key, model=model, **kwargs)

        # Lazy import to avoid dependency issues
        self._client = None

    def _get_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
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
        """Generate a completion using Anthropic's API.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            model: Model to use (overrides default)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLMResponse with the generated content
        """
        client = self._get_client()
        model = model or self.model_name

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            request_params["system"] = system_message

        # Add any additional parameters
        request_params.update(kwargs)

        try:
            response = await client.messages.create(**request_params)

            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider="anthropic",
                tokens_used=usage["total_tokens"],
                usage=usage,
            )

        except Exception as e:
            # Handle specific Anthropic errors
            error_msg = str(e)
            if "authentication" in error_msg.lower():
                raise ValueError(f"Authentication error: {error_msg}")
            elif "rate" in error_msg.lower():
                raise ValueError(f"Rate limit error: {error_msg}")
            else:
                raise ValueError(f"Anthropic API error: {error_msg}")

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
        """Stream response from Anthropic."""
        client = self._get_client()
        model = model or self.model_name

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if system_message:
            request_params["system"] = system_message

        request_params.update(kwargs)

        try:
            async with client.messages.stream(**request_params) as stream:
                async for chunk in stream:
                    if chunk.type == "content_block_delta":
                        yield chunk.delta.text

        except Exception as e:
            raise ValueError(f"Streaming error: {str(e)}")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Anthropic usage."""
        if self.model_name not in self.PRICING:
            return 0.0

        pricing = self.PRICING[self.model_name]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available."""
        return model in self.PRICING

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            # Try a minimal completion
            await self.complete(
                [LLMMessage(role="user", content="test")],
                max_tokens=10
            )
            return True
        except Exception:
            return False