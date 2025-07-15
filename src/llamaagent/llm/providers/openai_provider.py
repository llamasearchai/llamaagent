"""
OpenAI Provider implementation for LLM interactions.

This module provides integration with OpenAI's API for chat completions.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

import httpx

from src.llamaagent import LLMMessage, LLMResponse

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider with comprehensive error handling."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Base URL for API (defaults to OpenAI's)
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.max_retries = max_retries
        self.timeout = timeout

        self._client_config = {
            "timeout": httpx.Timeout(timeout),
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                )
            return self._client

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from a prompt.

        Converts the prompt to a message and calls chat_completion.
        """
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat_completion(messages, max_tokens, temperature, **kwargs)

    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation.

        This is the main implementation that does the actual API call.
        """
        return await self.complete(messages, max_tokens, temperature, model, **kwargs)

    async def complete(
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
            model: Model override
            **kwargs: Additional parameters

        Returns:
            LLMResponse with the completion
        """
        start_time = time.perf_counter()

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Convert messages to API format
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        payload = {
            "model": model or self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        # Retry logic
        last_error = None
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract response
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})

                    return LLMResponse(
                        content=content,
                        model=data.get("model", model or self.model),
                        provider="openai",
                        tokens_used=usage.get("total_tokens", 0),
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )

                elif response.status_code == 401:
                    raise ValueError("Invalid OpenAI API key")

                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue

                else:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", "Unknown error"
                    )
                    raise Exception(
                        f"OpenAI API error ({response.status_code}): {error_msg}"
                    )

            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

        # All retries failed
        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model override
            **kwargs: Additional parameters

        Yields:
            Chunks of the response as they arrive
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        payload = {
            "model": model or self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        client = await self._get_client()

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise Exception(f"OpenAI streaming error: {error_text}")

            async for line in response.aiter_lines():
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
                        logger.warning(f"Failed to decode streaming data: {data_str}")

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._client and not self._client.is_closed:
            try:
                asyncio.create_task(self._client.aclose())
            except Exception as e:
                logger.error(f"Error: {e}")
