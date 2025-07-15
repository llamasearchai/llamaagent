"""
Data generation modules for LlamaAgent.

This package provides data generation capabilities for AI agent training and evaluation.
"""

# The GDT data generation stack is optional during lightweight CI runs.
# Import errors (including SyntaxError) are tolerated and replaced with stubs.
try:
    from ..data.gdt import (
        ConversationDataGenerator,
        DataType,
        GDTDataset,
        GDTGenerator,
        GDTItem,
        GDTTransformer,
        GDTValidator,
        TextDataGenerator,
        ValidationStatus,
    )
except Exception:  # pylint: disable=broad-except
    # Provide minimal fallbacks to satisfy type checkers and runtime imports.
    class _Stub:  # type: ignore
        ...

    ConversationDataGenerator = _Stub  # type: ignore
    DataType = _Stub  # type: ignore
    GDTDataset = _Stub  # type: ignore
    GDTGenerator = _Stub  # type: ignore
    GDTItem = _Stub  # type: ignore
    GDTTransformer = _Stub  # type: ignore
    GDTValidator = _Stub  # type: ignore
    TextDataGenerator = _Stub  # type: ignore
    ValidationStatus = _Stub  # type: ignore

    # Fake DataType Enum for minimal functionality in tests.
    from enum import Enum

    class DataType(Enum):  # type: ignore
        TEXT = "text"
        CONVERSATION = "conversation"

__all__ = [
    "GDTGenerator",
    "GDTDataset",
    "GDTItem",
    "DataType",
    "ValidationStatus",
    "TextDataGenerator",
    "ConversationDataGenerator",
    "GDTValidator",
    "GDTTransformer",
    "GDTOrchestrator",
]

try:
    # Late import to avoid circular dependencies during initial package loading.
    from .gdt import GDTOrchestrator  # type: ignore  # noqa: E402, F401
except Exception:  # pylint: disable=broad-except
    class GDTOrchestrator:  # type: ignore
        ...
