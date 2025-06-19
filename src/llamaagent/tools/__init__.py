"""Tools for LlamaAgent"""

from .base import BaseTool, Tool, ToolRegistry
from .calculator import CalculatorTool
from .python_repl import PythonREPLTool


def get_all_tools() -> list[BaseTool]:
    """Instantiate and return all built-in tools."""
    return [
        CalculatorTool(),
        PythonREPLTool(),
    ]


# Pre-instantiated registry with all default tools – this is imported by
# *ReactAgent* and other components.

default_registry = ToolRegistry()
for _tool in get_all_tools():
    default_registry.register(_tool)

__all__ = [
    "Tool",
    "BaseTool",
    "ToolRegistry",
    "CalculatorTool",
    "PythonREPLTool",
    "get_all_tools",
    "default_registry",
]
