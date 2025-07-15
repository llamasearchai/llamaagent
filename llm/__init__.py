"""
Light-weight LLM sub-package exposing models, providers and factory.
"""

# Dynamically import submodule so that it is discoverable as attribute.
import importlib as _importlib
import sys as _sys

from .factory import ProviderFactory  # noqa: F401
from .models import LLMMessage, LLMResponse  # noqa: F401

providers = _importlib.import_module(__name__ + ".providers")  # noqa: F401
_sys.modules[__name__ + ".providers"] = providers

__all__ = [
    "models",
    "providers",
    "LLMMessage",
    "LLMResponse",
    "ProviderFactory",
]
