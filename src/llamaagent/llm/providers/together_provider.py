"""
Together AI provider implementation.
"""

from typing import Any, Dict, List, Optional

from ..base import BaseLLMProvider
from ..messages import LLMMessage, LLMResponse


class TogetherProvider(BaseLLMProvider):
    """Together AI provider implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Together AI."""
        return LLMResponse(
            content="Together AI provider not implemented",
            provider="together",
            error="Not implemented",
        )

    async def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Complete using Together AI."""
        return LLMResponse(
            content="Together AI provider not implemented",
            provider="together",
            error="Not implemented",
        )
