"""
Message and response structures for LLM providers.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..types import LLMMessage, LLMResponse  # Re-export for compatibility


@dataclass
class StreamingResponse:
    """Represents a streaming response chunk."""

    chunk: str
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata: Dict[str, Any] = {}


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "StreamingResponse",
]
