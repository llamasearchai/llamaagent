"""
Type definitions for the llamaagent library.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

# Type variables
T = TypeVar("T")


@runtime_checkable
class Tool(Protocol):
    """Protocol for tools that can be used by the agent."""

    name: str
    description: str

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool with the given arguments."""
        ...


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_name: str
    observation: str
    success: bool = True
    error: Optional[str] = None
    requires_planning: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Plan:
    """A plan for executing a task."""

    steps: List["PlanStep"]
    reasoning: str
    estimated_steps: int


@dataclass
class PlanStep:
    """A single step in a plan."""

    id: str
    description: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None
    contingency: Optional[str] = None
    is_completed: bool = False


@dataclass
class AgentState:
    """The current state of the agent."""

    id: str
    task: Optional[str] = None
    plan: Optional[Plan] = None
    current_step: int = 0
    observations: List[str] = field(default_factory=list)
    results: List[ToolResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResponse:
    """The response from an agent after executing a task."""

    task: str
    summary: str
    steps: List[ToolResult]
    reflection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryItem(Protocol):
    """Protocol for items stored in memory."""

    id: str
    content: Any
    timestamp: datetime
    metadata: Dict[str, Any]


class MemoryRetrieval(Protocol):
    """Protocol for retrieving items from memory."""

    items: List[MemoryItem]
    relevance: float
