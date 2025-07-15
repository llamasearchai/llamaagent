"""
LlamaAgent Tools Module

This module provides a comprehensive set of tools for the LlamaAgent framework,
including base tool interfaces, built-in tools, and dynamic tool loading capabilities.

Author: LlamaAgent Team
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

# Core imports - these are always available
from .base import BaseTool, ToolRegistry

# Backward compatibility alias
Tool = BaseTool

# Built-in tools
from .calculator import CalculatorTool
from .python_repl import PythonREPLTool

# Optional imports with graceful fallback
try:
    from .registry import ToolLoader, create_loader, get_registry
    from .registry import ToolMetadata as RegistryToolMetadata
except (ImportError, SyntaxError):
    ToolLoader = None
    RegistryToolMetadata = None
    create_loader = None
    get_registry = None

try:
    from .tool_registry import Tool as ToolRegistryTool
    from .tool_registry import (
        ToolCategory,
        ToolExecutionContext,
        ToolMetadata,
        ToolParameter,
        ToolResult,
        ToolSecurityLevel,
        ToolValidator,
    )
except (ImportError, SyntaxError):
    ToolRegistryTool = None
    ToolCategory = None
    ToolExecutionContext = None
    ToolMetadata = None
    ToolParameter = None
    ToolResult = None
    ToolSecurityLevel = None
    ToolValidator = None

try:
    from .dynamic_loader import DynamicToolLoader
    from .dynamic_loader import ToolMetadata as DynamicToolMetadata
except (ImportError, SyntaxError):
    DynamicToolLoader = None
    DynamicToolMetadata = None

try:
    from .plugin_framework import Plugin, PluginFramework, PluginManager, PluginState
except (ImportError, SyntaxError):
    Plugin = None
    PluginFramework = None
    PluginManager = None
    PluginState = None

# Logger
logger = logging.getLogger(__name__)


def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> BaseTool:
    """Create a Tool instance from a regular function.

    Args:
        func: The function to wrap as a tool
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to function docstring)

    Returns:
        A BaseTool instance that wraps the function
    """
    import inspect

    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or "No description available"

    class FunctionTool(BaseTool):
        """Dynamically created tool from a function."""

        @property
        def name(self) -> str:
            return tool_name

        @property
        def description(self) -> str:
            return tool_description

        def execute(self, **kwargs: Any) -> Any:
            """Execute the wrapped function."""
            # Filter kwargs to only include function parameters
            sig = inspect.signature(func)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return func(**filtered_kwargs)

    return FunctionTool()


def get_all_tools() -> List[BaseTool]:
    """Get all available default tools.

    Returns:
        List of instantiated default tools
    """
    return [
        CalculatorTool(),
        PythonREPLTool(),
    ]


# Export list
__all__ = [
    # Core
    "BaseTool",
    "Tool",  # Backward compatibility alias
    "ToolRegistry",
    # Built-in tools
    "CalculatorTool",
    "PythonREPLTool",
    # Utility functions
    "create_tool_from_function",
    "get_all_tools",
]

# Add optional exports if available
if ToolLoader is not None:
    __all__.extend(["ToolLoader", "create_loader", "get_registry"])
    if RegistryToolMetadata is not None:
        # Export as RegistryToolMetadata to avoid name conflicts
        __all__.append("RegistryToolMetadata")

if ToolCategory is not None:
    __all__.extend(
        [
            "ToolCategory",
            "ToolExecutionContext",
            "ToolMetadata",
            "ToolParameter",
            "ToolResult",
            "ToolSecurityLevel",
            "ToolValidator",
        ]
    )

if DynamicToolLoader is not None:
    __all__.append("DynamicToolLoader")
    if DynamicToolMetadata is not None and "ToolMetadata" not in __all__:
        # Only add if not already exported from tool_registry
        ToolMetadata = DynamicToolMetadata
        __all__.append("ToolMetadata")

if Plugin is not None:
    __all__.extend(
        [
            "Plugin",
            "PluginFramework",
            "PluginManager",
            "PluginState",
        ]
    )

# Log what's available
logger.debug(f"LlamaAgent tools module loaded with exports: {__all__}")
