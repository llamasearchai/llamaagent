"""Compatibility shim forwarding to the canonical implementation.

This module exists solely so that legacy imports like ``from llamaagent.types``
continue to work even after we migrated the real implementation to
``src.llamaagent.types``.  All names are re-exported unmodified.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from importlib import import_module as _import_module

_types = _import_module("src.llamaagent.types")

# Re-export everything publicly defined by the canonical module
__all__ = _types.__all__  # type: ignore[attr-defined]

globals().update({name: getattr(_types, name) for name in __all__}) 