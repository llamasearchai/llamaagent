from __future__ import annotations

import abc
from typing import List

from ..models import LLMMessage, LLMResponse


class BaseLLMProvider(abc.ABC):
    """Tiny abstract interface that all concrete providers must satisfy."""

    @abc.abstractmethod
    async def complete(self, messages: List[LLMMessage]) -> LLMResponse: ...

    @abc.abstractmethod
    async def health_check(self) -> bool: ...
