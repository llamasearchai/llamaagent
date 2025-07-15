"""Compatibility wrapper around *src.llamaagent.types*.

The historic test-suite imports ``LLMMessage`` and ``LLMResponse`` from the
top-level ``llm.models`` module.  The canonical definitions now live in
``src.llamaagent.types``.  We therefore pull them in and re-export so that
existing import paths keep working without code changes.
"""

from importlib import import_module as _import_module

_types = _import_module("src.llamaagent.types")

LLMMessage = _types.LLMMessage  # type: ignore[attr-defined]
LLMResponse = _types.LLMResponse  # type: ignore[attr-defined]

__all__ = ["LLMMessage", "LLMResponse"]
