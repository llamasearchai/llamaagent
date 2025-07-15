"""
Extremely thin wrapper around Ollama's REST API – enough for unit-tests.
"""

from __future__ import annotations

import asyncio
from typing import List

import httpx

from ..models import LLMMessage, LLMResponse
from .base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 30.0,
        retry_attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retry_attempts = retry_attempts

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    async def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.post(f"{self.base_url}/api/chat", json=payload)
                    # `raise_for_status` is normally synchronous but may be an
                    # `AsyncMock` when patched in the unit-tests.  We therefore
                    # handle both cases gracefully.
                    rfs = res.raise_for_status  # alias
                    if asyncio.iscoroutinefunction(rfs):
                        await rfs()  # type: ignore[func-returns-value]
                    else:
                        rfs()
                    data = res.json()
                    if asyncio.iscoroutine(data):
                        data = await data  # type: ignore[assignment]

                content = data["message"]["content"]
                tokens_used = self._estimate_tokens(content, messages)
                latency_ms = data.get("total_duration", 0) / 1_000_000  # ns → ms

                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider="ollama",
                    tokens_used=tokens_used,
                    metadata={
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration": data.get("eval_duration", 0),
                        "total_duration": data.get("total_duration", 0),
                        "latency_ms": latency_ms,
                    },
                )
            except Exception as exc:
                if attempt >= self.retry_attempts - 1:
                    raise ConnectionError("Failed to connect to Ollama") from exc
                await asyncio.sleep(2**attempt)

        # This should never be reached due to the exception handling above
        raise ConnectionError("Failed to connect to Ollama after all retries")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                rfs = res.raise_for_status  # alias
                if asyncio.iscoroutinefunction(rfs):
                    await rfs()  # type: ignore[func-returns-value]
                else:
                    rfs()
                models_json = res.json()
                if asyncio.iscoroutine(models_json):
                    models_json = await models_json  # type: ignore[assignment]
                models = [m["name"] for m in models_json.get("models", [])]
                return self.model in models
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_tokens(content: str, messages: List[LLMMessage]) -> int:
        total_chars = len(content) + sum(len(m.content) for m in messages)
        return total_chars // 4  # very rough heuristic
