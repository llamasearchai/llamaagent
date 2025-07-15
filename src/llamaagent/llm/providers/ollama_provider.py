"""
Ollama Provider implementation

Ultra-optimized Ollama provider for M3 Max with advanced features.
Implements connection pooling, retry logic, and performance monitoring.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import importlib.util
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from src.llamaagent import LLMMessage, LLMResponse

from ..exceptions import LLMError
from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


def _is_http2_available() -> bool:
    """Return True when the optional h2 dependency is installed."""
    return importlib.util.find_spec("h2") is not None


class OllamaProvider(BaseLLMProvider):
    """Production-grade Ollama provider optimized for M3 Max architecture."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        max_connections: int = 20,
        retry_attempts: int = 3,
        **kwargs: Any,
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name to use
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_connections: Maximum concurrent connections
            retry_attempts: Number of retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts

        # Connection pool optimized for M3 Max
        self._client_config = {
            "base_url": self.base_url,
            "timeout": httpx.Timeout(timeout=timeout),
            "limits": httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections * 2,
                keepalive_expiry=30.0,
            ),
            "http2": _is_http2_available(),
        }
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _get_client(self) -> httpx.AsyncClient:
        """Thread-safe client management with connection pooling."""
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(**self._client_config)

        try:
            yield self._client
        except Exception as e:
            logger.error(f"Client error: {e}")
            # Reset client on error
            if self._client and not self._client.is_closed:
                await self._client.aclose()
            self._client = None
            raise

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
        """
        Complete chat with advanced error handling and performance monitoring.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with comprehensive metadata

        Raises:
            ConnectionError: When Ollama is unreachable
            ValueError: When request parameters are invalid
        """
        start_time = time.perf_counter()

        # Validate inputs
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Prepare request payload
        payload = {
            "model": model or self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
            "stream": False,
        }

        # Execute with retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                async with self._get_client() as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    response.raise_for_status()

                    data = response.json()
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000

                    # Extract response data
                    content = data.get("message", {}).get("content", "")
                    if not content:
                        raise ValueError("Empty response from Ollama")

                    # Calculate token usage (approximation)
                    tokens_used = self._estimate_tokens(messages, content)

                    return LLMResponse(
                        content=content,
                        model=model or self.model,
                        provider="ollama",
                        tokens_used=tokens_used,
                        usage={
                            "latency_ms": latency_ms,
                            "eval_count": data.get("eval_count", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "total_duration": data.get("total_duration", 0),
                            "load_duration": data.get("load_duration", 0),
                            "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                            "eval_duration": data.get("eval_duration", 0),
                        },
                    )

            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Ollama connection failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(f"Unexpected error in Ollama provider: {e}")
                last_exception = e
                break

        # All retries failed
        raise ConnectionError(
            f"Failed to connect to Ollama after {self.retry_attempts} attempts: {last_exception}"
        )

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional parameters

        Yields:
            String chunks of the response
        """
        # Prepare request payload
        payload = {
            "model": model or self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs.get("options", {}),
            },
            "stream": True,
        }

        try:
            async with self._get_client() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise LLMError(f"Ollama streaming failed: {e}")

    def _estimate_tokens(self, messages: List[LLMMessage], content: str) -> int:
        """Estimate token usage for monitoring."""
        # Rough estimation: ~4 characters per token
        input_chars = sum(len(msg.content) for msg in messages)
        output_chars = len(content)
        return (input_chars + output_chars) // 4

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings using Ollama.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            Dictionary containing embeddings and metadata
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            payload = {
                "model": model or self.model,
                "prompt": text,
            }

            try:
                async with self._get_client() as client:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data["embedding"])

            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise LLMError(f"Ollama embedding failed: {e}")

        return {
            "embeddings": embeddings,
            "model": model or self.model,
            "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts)},
        }

    async def validate_model(self, model: str) -> bool:
        """Validate if model is available in Ollama.

        Args:
            model: Model name to validate

        Returns:
            True if model is available, False otherwise
        """
        try:
            async with self._get_client() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return model in models
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Comprehensive health check for Ollama."""
        try:
            async with self._get_client() as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

                # Check if our model is available
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]

                if self.model not in models:
                    logger.warning(
                        f"Model {self.model} not found in Ollama. Available: {models}"
                    )
                    return False

                return True

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for Ollama usage.

        Ollama is local, so there's no cost.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Always returns 0.0 for local inference
        """
        return 0.0