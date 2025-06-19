from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..memory import SimpleMemory
from ..tools import ToolRegistry


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    tokens_used: int = 0
    plan: Optional[ExecutionPlan] = None


@dataclass
class AgentTrace:
    """Execution trace for analysis and debugging."""

    agent_name: str
    task: str
    start_time: float
    end_time: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    tokens_used: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def execution_time(self) -> float:
        """Calculate total execution time."""
        return self.end_time - self.start_time

    def add_step(
        self, step_type: str, description: str, result: Any = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a step to the trace."""
        self.steps.append(
            {
                "timestamp": time.time(),
                "step_type": step_type,
                "description": description,
                "result": str(result) if result is not None else None,
                "metadata": metadata or {},
            }
        )


@dataclass
class AgentConfig:
    """Agent configuration."""

    name: str = "Agent"
    role: AgentRole = AgentRole.GENERALIST
    description: str = ""
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 300.0
    retry_attempts: int = 3
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    streaming: bool = False
    spree_enabled: bool = False
    dynamic_tools: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Base agent with core functionality."""

    def __init__(
        self, config: AgentConfig, tools: Optional[ToolRegistry] = None, memory: Optional[SimpleMemory] = None
    ):
        self.config = config
        self.tools = tools or ToolRegistry()
        self.memory = memory or (SimpleMemory() if config.memory_enabled else None)
        self.trace: Optional[AgentTrace] = None

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute a task and return response."""
        start_time = time.time()
        self.trace = AgentTrace(
            agent_name=self.config.name,
            task=task,
            start_time=start_time,
            end_time=start_time,  # Will be updated on completion
        )

        try:
            # Basic execution logic - to be overridden by subclasses
            self.trace.add_step("start", f"Starting task: {task}")

            # Placeholder response
            result = f"Task '{task}' processed by {self.config.name}"

            self.trace.final_result = result
            self.trace.success = True
            self.trace.end_time = time.time()

            return AgentResponse(
                content=result,
                success=True,
                execution_time=self.trace.execution_time,
                metadata={"agent_name": self.config.name},
            )

        except Exception as e:
            self.trace.error_message = str(e)
            self.trace.success = False
            self.trace.end_time = time.time()

            return AgentResponse(
                content=f"Error: {str(e)}",
                success=False,
                execution_time=self.trace.execution_time,
                metadata={"agent_name": self.config.name, "error": str(e)},
            )

    async def stream_execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Stream execution results."""
        response = await self.execute(task, context)
        yield response.content

    def get_trace(self) -> Optional[AgentTrace]:
        """Get the execution trace."""
        return self.trace
