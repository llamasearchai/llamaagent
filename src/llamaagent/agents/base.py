"""Base agent classes and utilities for the llamaagent framework.

This module provides the foundation for all agent implementations, including
abstract base classes, configuration, and common data structures.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from ..types import TaskInput, TaskOutput, TaskResult, TaskStatus

# Import the classes we need to fix the type errors
if TYPE_CHECKING:
    from ..tools import ToolRegistry as ToolRegistryType
    from ..memory.base import SimpleMemory as SimpleMemoryType
else:
    try:
        from ..tools import ToolRegistry as ToolRegistryType
    except ImportError:
        ToolRegistryType = None
    
    try:
        from ..memory.base import SimpleMemory as SimpleMemoryType
    except ImportError:
        SimpleMemoryType = None

# Create fallback classes for runtime
class ToolRegistry:
    """Fallback ToolRegistry implementation."""
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
    
    def register(self, tool: Any) -> None:
        if hasattr(tool, 'name'):
            self._tools[tool.name] = tool
    
    def get(self, name: str) -> Any:
        return self._tools.get(name)
    
    def list_names(self) -> List[str]:
        return list(self._tools.keys())
    
    def list_tools(self) -> List[Any]:
        return list(self._tools.values())
    
    def get_tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)


class SimpleMemory:
    """Fallback SimpleMemory implementation."""
    
    def __init__(self):
        self._memories: List[Dict[str, Any]] = []
    
    async def add(self, content: str, **metadata: Any) -> str:
        memory_id = str(len(self._memories))
        memory_entry: Dict[str, Any] = {"id": memory_id, "content": content, **metadata}
        self._memories.append(memory_entry)
        return memory_id
    
    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for memory in self._memories:
            if query.lower() in memory["content"].lower():
                results.append(memory)
                if len(results) >= limit:
                    break
        return results
    
    def count(self) -> int:
        return len(self._memories)


TaskInputType = TaskInput
TaskOutputType = TaskOutput
TaskResultType = TaskResult
TaskStatusType = TaskStatus


class AgentRole(str, Enum):
    """Agent roles for multi-agent systems."""

    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    CRITIC = "critic"
    PLANNER = "planner"
    SPECIALIST = "specialist"
    TOOL_SPECIFIER = "tool_specifier"
    TOOL_SYNTHESIZER = "tool_synthesizer"
    ORCHESTRATOR = "orchestrator"
    GENERALIST = "generalist"


@dataclass
class AgentConfig:
    """Agent configuration."""

    # Core configuration
    name: str = "TestAgent"  # Tests expect default name to be 'TestAgent'
    role: AgentRole = AgentRole.GENERALIST
    description: str = ""

    # Execution parameters
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 300.0
    retry_attempts: int = 3

    # System configuration
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    streaming: bool = False
    spree_enabled: bool = True  # Tests assume SPRE enabled by default
    dynamic_tools: bool = False

    # Extended fields required by integration tests
    llm_provider: Any | None = None  # Inject explicit provider instance
    verbose: bool = False  # Verbose logging flag
    debug: bool = False  # Debug mode flag

    # Metadata storage
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        """Backward compatibility property for agent_name."""
        return self.name

    @agent_name.setter
    def agent_name(self, value: str) -> None:
        """Backward compatibility setter for agent_name."""
        self.name = value


@dataclass
class PlanStep:
    """Individual step in execution plan."""

    step_id: int
    description: str
    required_information: str
    expected_outcome: str
    is_completed: bool = False
    agent_assignment: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task."""

    original_task: str
    steps: List[PlanStep]
    current_step: int = 0
    dependencies: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message between agents or components."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    content: str = ""
    role: str = "user"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Agent execution response with full trace."""

    content: str
    success: bool = True
    messages: List[AgentMessage] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[ExecutionPlan] = None


@dataclass
class Step:
    """Individual reasoning step in agent execution."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: str = "reasoning"
    description: str = ""
    input_data: Any = None
    output_data: Any = None
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def complete(self, output: Any, error: Optional[str] = None) -> None:
        """Mark step as complete with output."""
        self.output_data = output
        self.error = error
        self.duration = time.time() - self.timestamp


@dataclass
class AgentTrace:
    """Execution trace for analysis and debugging."""

    agent_name: str
    task: str
    start_time: float
    end_time: float = 0.0
    steps: List[Step] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    tokens_used: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> float:
        """Calculate total execution time."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def add_step(self, step_type: str, description: str, **kwargs: Any) -> Step:
        """Add a step to the trace."""
        step = Step(step_type=step_type, description=description, metadata=kwargs)
        self.steps.append(step)
        return step


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[Any] = None,
        memory: Optional[Any] = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            config: Agent configuration
            tools: Optional tool registry
            memory: Optional memory implementation
        """
        self.config = config
        
        # Initialize tools with proper fallback
        if tools is not None:
            self.tools = tools
        else:
            # Try to use imported ToolRegistry, fall back to local implementation
            try:
                self.tools = ToolRegistryType() if ToolRegistryType else ToolRegistry()
            except Exception:
                self.tools = ToolRegistry()
        
        # Initialize memory with proper fallback
        if memory is not None:
            self.memory = memory
        elif config.memory_enabled:
            # Try to use imported SimpleMemory, fall back to local implementation
            try:
                self.memory = SimpleMemoryType() if SimpleMemoryType else SimpleMemory()
            except Exception:
                self.memory = SimpleMemory()
        else:
            self.memory = None
            
        self.trace: Optional[AgentTrace] = None
        self._current_step: Optional[Step] = None

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get the agent description."""
        return self.config.description

    @abstractmethod
    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute a task and return response.

        Args:
            task: The task to execute
            context: Optional context dictionary

        Returns:
            AgentResponse with execution results
        """

    async def execute_task(self, task_input: Any) -> Any:
        """Execute a task using TaskInput/TaskOutput interface.

        Args:
            task_input: The task input

        Returns:
            TaskOutput with execution results or dict fallback
        """
        try:
            # Execute using the abstract execute method
            response = await self.execute(task_input.task, task_input.context)

            # Convert AgentResponse to TaskOutput
            task_result = TaskResultType(
                success=response.success,
                data={"content": response.content, "metadata": response.metadata},
                error=response.error,
                metadata=response.metadata,
            )

            return TaskOutputType(
                task_id=task_input.id,
                status=TaskStatusType.COMPLETED if response.success else TaskStatusType.FAILED,
                result=task_result,
            )
        except Exception as e:
            task_result = TaskResultType(
                success=False,
                error=str(e),
                metadata={"agent_name": self.config.name},
            )

            return TaskOutputType(
                task_id=task_input.id, 
                status=TaskStatusType.FAILED, 
                result=task_result
            )

    async def stream_execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream execution results.

        Args:
            task: The task to execute
            context: Optional context dictionary

        Yields:
            String chunks of the response
        """
        response = await self.execute(task, context)
        yield response.content

    def get_trace(self) -> Optional[AgentTrace]:
        """Get the execution trace."""
        return self.trace

    def start_step(self, step_type: str, description: str, **kwargs: Any) -> Step:
        """Start a new execution step.

        Args:
            step_type: Type of step (e.g., "reasoning", "tool_call")
            description: Description of what the step does
            **kwargs: Additional metadata

        Returns:
            The created Step instance
        """
        if self.trace:
            self._current_step = self.trace.add_step(step_type, description, **kwargs)
            return self._current_step
        else:
            # Create standalone step if no trace
            step = Step(step_type=step_type, description=description, metadata=kwargs)
            self._current_step = step
            return step

    def complete_step(self, output: Any, error: Optional[str] = None) -> None:
        """Complete the current step.

        Args:
            output: The output from the step
            error: Optional error message
        """
        if self._current_step:
            self._current_step.complete(output, error)
            self._current_step = None

    async def cleanup(self) -> None:
        """Cleanup resources. Override in subclasses if needed."""

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.config.name})"

    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        # Safe tool count calculation
        tool_count = 0
        if self.tools:
            # Try different patterns for tool registries
            try:
                if hasattr(self.tools, 'list_names'):
                    tool_names = self.tools.list_names()
                    tool_count = len(tool_names) if tool_names else 0
                elif hasattr(self.tools, 'list_tools'):
                    tool_list = self.tools.list_tools()
                    tool_count = len(tool_list) if tool_list else 0
                else:
                    # Generic fallback for any tool registry
                    tool_count = 1  # At least we have a tools object
            except (TypeError, AttributeError):
                tool_count = 0

        memory_status = self.memory is not None

        return (
            f"{self.__class__.__name__}("
            f"config={self.config!r}, "
            f"tools={tool_count}, "
            f"memory={memory_status}"
            f")"
        )
