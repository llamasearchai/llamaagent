"""
LlamaAgent Tools Module

This module provides a comprehensive set of tools for the LlamaAgent framework,
including base tool interfaces, built-in tools, and dynamic tool loading capabilities.

Author: LlamaAgent Team
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

# Logger
logger = logging.getLogger(__name__)

# Core imports - these are always available
from .base import BaseTool, ToolRegistry

# Backward compatibility alias
Tool = BaseTool

# Built-in tools
from .calculator import CalculatorTool
from .python_repl import PythonREPLTool


class ToolManager:
    """Basic tool manager for managing tool instances and execution."""
    
    def __init__(self) -> None:
        """Initialize the tool manager."""
        self.tools: dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__ + ".ToolManager")
        
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    async def cleanup(self) -> None:
        """Cleanup tool manager resources."""
        self.tools.clear()
        self.logger.info("Tool manager cleanup completed")


# Optional imports with graceful fallback
try:
    from .registry import ToolLoader
    from .registry import ToolMetadata as RegistryToolMetadata
    from .registry import create_loader, get_registry
except (ImportError, SyntaxError):
    ToolLoader = None
    RegistryToolMetadata = None
    create_loader = None
    get_registry = None

# Tool registry imports
ToolRegistryTool = None
ToolCategory = None
ToolExecutionContext = None
ToolParameter = None
ToolResult = None
ToolSecurityLevel = None

try:
    from .tool_registry import (  # type: ignore
        Tool as ToolRegistryTool,
        ToolCategory,
        ToolExecutionContext,
        ToolParameter,
        ToolResult,
        ToolSecurityLevel,
    )
except (ImportError, SyntaxError) as e:
    logger.debug(f"Failed to import tool_registry: {e}")
    # Ensure variables remain None if import fails
    ToolRegistryTool = None
    ToolCategory = None
    ToolExecutionContext = None
    ToolParameter = None
    ToolResult = None
    ToolSecurityLevel = None

try:
    from .dynamic_loader import DynamicToolLoader
    from .dynamic_loader import ToolMetadata as DynamicToolMetadata
except (ImportError, SyntaxError):
    DynamicToolLoader = None
    DynamicToolMetadata = None

# Plugin framework imports
PluginFramework = None

try:
    from .plugin_framework import (  # type: ignore
        PluginFramework,
    )
except (ImportError, SyntaxError) as e:
    logger.debug(f"Failed to import plugin_framework: {e}")
    # Ensure variables remain None if import fails
    PluginFramework = None


def create_tool_from_function(
    func: Callable[..., Any],
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
            return func(**filtered_kwargs)  # type: ignore

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
    # Core classes
    'BaseTool',
    'Tool', 
    'ToolRegistry',
    'ToolManager',
    
    # Built-in tools
    'CalculatorTool',
    'PythonREPLTool',
    
    # Registry system (when available)
    'ToolLoader',
    'RegistryToolMetadata',
    'create_loader',
    'get_registry',
    
    # Tool registry classes (when available)
    'ToolRegistryTool',
    'ToolParameter',
    'ToolResult', 
    'ToolCategory',
    'ToolSecurityLevel',
    'ToolExecutionContext',
    
    # Dynamic loading (when available)
    
    # Helper functions
    'get_all_tools',
    'create_tool_from_function',
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
            "ToolParameter",
            "ToolResult",
            "ToolSecurityLevel",
        ]
    )

if DynamicToolLoader is not None:
    __all__.append("DynamicToolLoader")
    if DynamicToolMetadata is not None:
        # Export as DynamicToolMetadata to avoid confusion
        __all__.append("DynamicToolMetadata")

if PluginFramework is not None:
    __all__.append("PluginFramework")

# Log what's available
logger.debug(f"LlamaAgent tools module loaded with exports: {__all__}")
