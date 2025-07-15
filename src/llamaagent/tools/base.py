"""Base tool interface"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


class BaseTool(ABC):
    """Base class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        ...  # pragma: no cover

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool"""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Compatibility aliases & lightweight registry
# ---------------------------------------------------------------------------

# Older modules import `Tool` directly from the package. To keep backward
# compatibility we expose the same symbol as an alias to BaseTool so that any
# subclassing via `class MyTool(Tool):` keeps working unchanged.

# NOTE: This must come after the BaseTool definition so that the symbol
# is already available.

# Type alias for backward compatibility
if TYPE_CHECKING:
    Tool: TypeAlias = BaseTool
else:
    Tool = BaseTool


class ToolRegistry:
    """In-memory registry for managing tool instances during runtime.

    The registry is intentionally lightweight – it only stores instantiated
    tool objects and exposes a handful of helpers required by the test-suite
    (register, deregister, get, list_names).
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def register(self, tool: BaseTool) -> None:
        """Register tool under the value of its name attribute."""
        self._tools[tool.name] = tool

    def deregister(self, name: str) -> None:
        """Remove name from the registry if present (silently ignores missing)."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[BaseTool]:
        """Return the tool registered under name or None if absent."""
        return self._tools.get(name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Alias for get() method for API compatibility."""
        return self.get(name)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def list_names(self) -> List[str]:
        """Return a list of registered tool names (in insertion order)."""
        return list(self._tools.keys())

    def list_tools(self) -> List[BaseTool]:
        """Return the list of registered tool instances."""
        return list(self._tools.values())
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        """Property for backward compatibility - returns tools dictionary."""
        return self._tools.copy()
