"""
Asynchronous OpenAI provider – minimal yet complete implementation used by tests.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Optional dependency handling                                                #
# --------------------------------------------------------------------------- #
# Make tests independent of the real `openai` package.  If it's not present,
# we register a skeletal stub that still works with `unittest.mock.patch`.
import sys
import types
from typing import List

try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    openai = types.ModuleType("openai")  # type: ignore

    class _StubAsyncOpenAI:  # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__) -> None:
            self.models = types.SimpleNamespace(list=lambda: types.SimpleNamespace(data=[]))

        class chat:  # noqa: D401 pylint: disable=invalid-name
            class completions:  # noqa: D401
                @staticmethod
                async def create(*_, **__) -> "types.SimpleNamespace":  # type: ignore
                    raise RuntimeError("openai stub – this should be patched in tests")

    openai.AsyncOpenAI = _StubAsyncOpenAI  # type: ignore
    sys.modules["openai"] = openai

# --------------------------------------------------------------------------- #

from ..models import LLMMessage, LLMResponse
from .base_provider import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """Wrapper around `openai` async client."""

    def __init__(self, *, api_key: str, model: str = "gpt-4", timeout: float = 30.0) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # Client is instantiated lazily to ensure that `patch('openai.AsyncOpenAI')`
        # in unit-tests intercepts construction correctly.
        self._client = None  # type: ignore

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            import openai  # Local import after potential `patch.dict` modifications

            self._client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)  # type: ignore[arg-type]
        return self._client

    # ------------------------------------------------------------------ #
    # BaseLLMProvider interface                                          #
    # ------------------------------------------------------------------ #
    async def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        payload = [{"role": m.role, "content": m.content} for m in messages]

        client = self._get_client()
        try:
            resp = await client.chat.completions.create(model=self.model, messages=payload)  # type: ignore
        except Exception as exc:  # pragma: no cover – network / auth errors
            raise Exception(f"API Error: {exc}") from exc

        choice = resp.choices[0]
        usage = resp.usage
        total_tokens = int(getattr(usage, "total_tokens", 0)) if usage else 0

        return LLMResponse(
            content=choice.message.content or "",
            tokens_used=total_tokens,
            model=resp.model,
            provider="openai",
            metadata={
                "finish_reason": choice.finish_reason,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)) if usage else 0,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0)) if usage else 0,
            },
        )

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            listing = await client.models.list()
            data = getattr(listing, "data", [])
            # `data` may be AsyncMock; ensure iterable
            model_ids = [getattr(m, "id", None) for m in data]
            return self.model in model_ids
        except Exception:  # pragma: no cover
            return False

    @property
    def client(self):  # type: ignore[override]
        """Return (and lazily instantiate) the underlying OpenAI client."""
        return self._get_client()
