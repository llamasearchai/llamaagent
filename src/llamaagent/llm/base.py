"""
Base LLM provider facade used by the public *llamaagent.llm* namespace.

The comprehensive integration test-suite expects to import
`llamaagent.llm.base.BaseLLMProvider`.  Internally the implementation lives
in `llamaagent.llm.providers.base_provider`.  This shim avoids import
errors by re-exporting the canonical definition.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

from .messages import LLMMessage, LLMResponse  # Shared immutable dataclasses
from .providers.base_provider import BaseLLMProvider as _Impl

__all__ = [
    "LLMProvider",
    "BaseLLMProvider",
    "LLMMessage",
    "LLMResponse",
]

# Public aliases — tests import these symbols directly
BaseLLMProvider = _Impl  # type: ignore
LLMProvider = _Impl  # type: ignore  # Alias for backwards compatibility

# Backwards-compatibility: expose message/response types here too so that
# `from llamaagent.llm.base import LLMMessage` works if used elsewhere.
LLMMessage = LLMMessage  # pylint: disable=invalid-name
LLMResponse = LLMResponse  # pylint: disable=invalid-name
