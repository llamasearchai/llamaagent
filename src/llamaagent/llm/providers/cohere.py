"""
Cohere Provider implementation

This module implements the Cohere LLM provider for the llamaagent system.
It provides integration with Cohere's API for text generation and chat completion.

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


class CohereProvider(BaseLLMProvider):
    """Cohere LLM Provider implementation."""

    BASE_URL = "https://api.cohere.ai/v1"

    # Cost per 1K tokens (input, output)
    PRICING = {
        "command": (0.0006, 0.002),
        "command-light": (0.00015, 0.0006),
        "command-r": (0.0003, 0.0015),
        "command-r-plus": (0.003, 0.015),
    }

    def __init__(
        self,
        api_key: str,
        model: str = "command",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Cohere provider.

        Args:
            api_key: Cohere API key
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

        # Convert messages to Cohere format
        chat_history = []
        message = ""

        for msg in messages:
            if msg.role == "system":
                # Cohere doesn't have a system role, prepend to message
                message = f"{msg.content}\n\n" + message
            elif msg.role == "user":
                if message:  # If there was a previous assistant message
                    chat_history.append({"role": "CHATBOT", "message": message})
                    message = ""
                chat_history.append({"role": "USER", "message": msg.content})
            elif msg.role == "assistant":
                message = msg.content

        # Handle the last message
        if not chat_history or chat_history[-1]["role"] != "USER":
            # Ensure the last message is from the user
            if message:
                chat_history.append({"role": "USER", "message": "Continue."})
            else:
                chat_history.append({"role": "USER", "message": "Hello"})

        payload = {
            "model": model,
            "chat_history": chat_history[:-1] if len(chat_history) > 1 else [],
            "message": chat_history[-1]["message"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid Cohere API key")
                    elif response.status == 429:
                        raise RateLimitError("Cohere rate limit exceeded")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Cohere API error: {error_text}")

                    data = await response.json()

                    content = data["text"]
                    # Cohere's response structure for token usage
                    usage = {
                        "prompt_tokens": data.get("meta", {})
                        .get("billed_units", {})
                        .get("input_tokens", 0),
                        "completion_tokens": data.get("meta", {})
                        .get("billed_units", {})
                        .get("output_tokens", 0),
                    }
                    usage["total_tokens"] = (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                    )
                    cost = self.calculate_cost(
                        usage["prompt_tokens"], usage["completion_tokens"]
                    )

                    # Add cost to usage dict
                    usage["cost"] = cost

                    return LLMResponse(
                        content=content,
                        model=model,
                        provider="cohere",
                        tokens_used=usage["total_tokens"],
                        usage=usage,
                    )

            except aiohttp.ClientError as e:
                raise LLMError(f"Cohere request failed: {e}")

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
        """Generate streaming response from Cohere.

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

        # Convert messages to Cohere format (same as chat_completion)
        chat_history = []
        message = ""

        for msg in messages:
            if msg.role == "system":
                message = f"{msg.content}\n\n" + message
            elif msg.role == "user":
                if message:
                    chat_history.append({"role": "CHATBOT", "message": message})
                    message = ""
                chat_history.append({"role": "USER", "message": msg.content})
            elif msg.role == "assistant":
                message = msg.content

        if not chat_history or chat_history[-1]["role"] != "USER":
            if message:
                chat_history.append({"role": "USER", "message": "Continue."})
            else:
                chat_history.append({"role": "USER", "message": "Hello"})

        payload = {
            "model": model,
            "chat_history": chat_history[:-1] if len(chat_history) > 1 else [],
            "message": chat_history[-1]["message"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Cohere streaming error: {error_text}")

                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("event_type") == "text-generation":
                                    yield data.get("text", "")
                                elif data.get("event_type") == "stream-end":
                                    break
                            except json.JSONDecodeError:
                                continue

            except aiohttp.ClientError as e:
                raise LLMError(f"Cohere streaming request failed: {e}")

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings using Cohere.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary containing embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]

        model = model or "embed-english-v3.0"

        payload = {
            "model": model,
            "texts": texts,
            "input_type": kwargs.get("input_type", "search_document"),
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/embed",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Cohere embeddings error: {error_text}")

                    data = await response.json()

                    embeddings = data["embeddings"]
                    usage = data.get("meta", {}).get("billed_units", {})

                    return {
                        "embeddings": embeddings,
                        "model": model,
                        "usage": usage,
                    }

            except aiohttp.ClientError as e:
                raise LLMError(f"Cohere embeddings request failed: {e}")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Cohere usage.

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
        """Validate Cohere model availability.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        # Cohere doesn't provide a models endpoint, so we check against known models
        known_models = list(self.PRICING.keys()) + [
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
        ]
        return model in known_models

    async def health_check(self) -> bool:
        """Check if the Cohere provider is healthy.

        Returns:
            True if provider is operational, False otherwise
        """
        try:
            # Try a minimal completion
            test_message = LLMMessage(role="user", content="Hello")
            response = await self.complete([test_message], max_tokens=10)
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"Cohere health check failed: {e}")
            return False
