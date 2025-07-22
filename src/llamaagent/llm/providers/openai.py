"""
OpenAI Provider implementation

This module implements the OpenAI LLM provider for the llamaagent system.
It provides integration with OpenAI's API for text generation, chat completion,
and embeddings.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from src.llamaagent import LLMMessage, LLMResponse

from ..exceptions import AuthenticationError, LLMError, RateLimitError
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider implementation."""

    BASE_URL = "https://api.openai.com/v1"

    # Cost per 1K tokens (input, output)
    PRICING = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "text-embedding-ada-002": (0.0001, 0),
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Optional custom base URL
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.base_url = base_url or self.BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
        return await self.chat_completion(
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
        model = model or self.model

        payload = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid OpenAI API key")
                    elif response.status == 429:
                        raise RateLimitError("OpenAI rate limit exceeded")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenAI API error: {error_text}")

                    data = await response.json()

                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    cost = self.calculate_cost(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )

                    # Add cost to usage dict
                    if usage:
                        usage["cost"] = cost
                    else:
                        usage = {"cost": cost}

                    return LLMResponse(
                        content=content,
                        model=model,
                        provider="openai",
                        tokens_used=usage.get("total_tokens", 0),
                        usage=usage,
                    )

            except aiohttp.ClientError as e:
                raise LLMError(f"OpenAI request failed: {e}")

    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete method (alias for chat_completion).

        This is the primary method used throughout the llamaagent codebase.
        """
        return await self.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **kwargs,
        )

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Yields:
            String chunks of the response
        """
        model = model or self.model

        payload = {
            "model": model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenAI streaming error: {error_text}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

            except aiohttp.ClientError as e:
                raise LLMError(f"OpenAI streaming request failed: {e}")

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings using OpenAI.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary containing embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]

        model = model or "text-embedding-ada-002"

        payload = {"model": model, "input": texts}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenAI embeddings error: {error_text}")

                    data = await response.json()

                    embeddings = [item["embedding"] for item in data["data"]]
                    usage = data.get("usage", {})
                    cost = self.calculate_cost(usage.get("prompt_tokens", 0), 0)

                    # Add cost to usage dict
                    if usage:
                        usage["cost"] = cost
                    else:
                        usage = {"cost": cost}

                    return {
                        "embeddings": embeddings,
                        "model": model,
                        "usage": usage,
                    }

            except aiohttp.ClientError as e:
                raise LLMError(f"OpenAI embeddings request failed: {e}")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for OpenAI usage.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if self.model not in self.PRICING:
            return 0.0

        input_price, output_price = self.PRICING[self.model]
        cost = (prompt_tokens * input_price / 1000) + (
            completion_tokens * output_price / 1000
        )
        return round(cost, 6)

    async def validate_model(self, model: str) -> bool:
        """Validate OpenAI model availability.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/models", headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["id"] for m in data.get("data", [])]
                        return model in models
                    return False
            except Exception:
                return False

    async def health_check(self) -> bool:
        """Check if the OpenAI provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        try:
            # Try a minimal completion
            test_message = LLMMessage(role="user", content="Hello")
            response = await self.complete([test_message], max_tokens=10)
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
