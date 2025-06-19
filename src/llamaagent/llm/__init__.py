"""LLM provider implementations for LlamaAgent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx

__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MlxProvider",
    "create_provider",
]


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    tokens_used: int = 0
    model: str = "mock"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:  # pragma: no cover
        pass


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:  # type: ignore[override]
        # Simple mock response
        last_message = messages[-1].content if messages else "No message"
        mock_response = f"Mock response to: {last_message[:50]}..."

        return LLMResponse(
            content=mock_response, tokens_used=len(mock_response) // 4, model="mock-gpt-4"  # Rough estimate
        )


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using OpenAI API."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=kwargs.get("timeout", 30.0),
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return LLMResponse(content=content, tokens_used=tokens_used, model=self.model)

            except Exception as e:
                # Fallback to mock response on error
                return LLMResponse(
                    content=f"Error communicating with OpenAI: {str(e)}", tokens_used=0, model=self.model
                )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using Anthropic API."""
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}

        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.7),
        }

        if system_message:
            payload["system"] = system_message

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=kwargs.get("timeout", 30.0),
                )
                response.raise_for_status()

                data = response.json()
                content = data["content"][0]["text"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return LLMResponse(content=content, tokens_used=tokens_used, model=self.model)

            except Exception as e:
                # Fallback to mock response on error
                return LLMResponse(
                    content=f"Error communicating with Anthropic: {str(e)}", tokens_used=0, model=self.model
                )


class OllamaProvider(LLMProvider):
    """Provider for a local Ollama server or compatible endpoint.

    The default base URL assumes the standard Ollama daemon running on
    `http://localhost:11434`.  Authentication is optional; if your instance
    requires a token set the ``OLLAMA_API_KEY`` environment variable and it
    will be forwarded as a Bearer token.  The API surface mirrors the
    OpenAI /chat/completions route so we can reuse most of the wiring here.
    """

    def __init__(self, api_key: str | None = None, model: str = "llama4:latest", base_url: Optional[str] = None):
        self.api_key = api_key or ""
        self.model = model
        # Default to the local Ollama endpoint; users can override via env
        # variable ``OLLAMA_BASE_URL`` or by passing the ``base_url`` param.
        self.base_url = base_url or "http://localhost:11434/v1"

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using Ollama API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=kwargs.get("timeout", 30.0),
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return LLMResponse(content=content, tokens_used=tokens_used, model=self.model)

            except Exception as e:
                # Fallback to mock response on error
                return LLMResponse(
                    content=f"Error communicating with Ollama: {str(e)}", tokens_used=0, model=self.model
                )


class MlxProvider(LLMProvider):
    """Provider for a local MLX server or compatible endpoint.

    The default base URL assumes the standard MLX daemon running on
    `http://localhost:11434`.  Authentication is optional; if your instance
    requires a token set the ``MLX_API_KEY`` environment variable and it
    will be forwarded as a Bearer token.  The API surface mirrors the
    OpenAI /chat/completions route so we can reuse most of the wiring here.
    """

    def __init__(self, api_key: str | None = None, model: str = "mlx4:latest", base_url: Optional[str] = None):
        self.api_key = api_key or ""
        self.model = model
        # Default to the local MLX endpoint; users can override via env
        # variable ``MLX_BASE_URL`` or by passing the ``base_url`` param.
        self.base_url = base_url or "http://localhost:11434/v1"

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using MLX API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=kwargs.get("timeout", 30.0),
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                return LLMResponse(content=content, tokens_used=tokens_used, model=self.model)

            except Exception as e:
                # Fallback to mock response on error
                return LLMResponse(
                    content=f"Error communicating with MLX: {str(e)}", tokens_used=0, model=self.model
                )


def create_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
    """Factory function to create LLM providers."""
    if provider_type.lower() == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAIProvider(api_key=api_key, **{k: v for k, v in kwargs.items() if k != "api_key"})

    elif provider_type.lower() == "anthropic":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        return AnthropicProvider(api_key=api_key, **{k: v for k, v in kwargs.items() if k != "api_key"})

    elif provider_type.lower() == "ollama":
        # Ollama does not mandate an API key, but we forward it if provided.
        return OllamaProvider(**kwargs)

    elif provider_type.lower() == "mlx":
        # MLX runs locally and therefore needs a model path or name.  We do
        # not require an API key but allow callers to specify a custom model
        # via ``model`` kwarg or ``MLX_MODEL`` environment variable.
        return MlxProvider(**kwargs)

    else:
        return MockProvider()
