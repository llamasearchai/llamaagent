"""
Base LLM Provider interface for the llamaagent system.

This module extends the BaseProvider class to provide the specific interface
used throughout the llamaagent codebase, particularly the 'complete' method.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import abc
from typing import Any, List, Optional

from ...types import LLMMessage, LLMResponse
from .base import BaseProvider


class BaseLLMProvider(BaseProvider):
    """Abstract interface for LLM providers within the llamaagent namespace.

    This class extends BaseProvider and adds the 'complete' method which is
    used throughout the codebase as the primary interface for chat completions.
    """

    @abc.abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation with messages.

        This is the primary method used throughout the llamaagent codebase.
        It's essentially an alias for chat_completion but maintains the
        naming convention used in the system.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (overrides default)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the assistant's reply
        """

    async def health_check(self) -> bool:
        """Check if the provider is healthy and available.

        Returns:
            True if provider is operational, False otherwise
        """
        try:
            # Basic health check - try a simple completion
            test_messages = [LLMMessage(role="user", content="test")]
            response = await self.complete(test_messages, max_tokens=10)
            return len(response.content) > 0
        except Exception:
            return False

    # The following methods are inherited from BaseProvider:
    # - generate_response
    # - chat_completion (implementations should make complete() call this)
    # - embed_text
    # - stream_chat_completion
    # - calculate_cost
    # - validate_model
