"""
Base Agent Classes and Types

This module defines the base classes and types used throughout the LlamaAgent
system, including AgentConfig, AgentResponse, and base agent classes.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent role enumeration"""

    GENERALIST = "generalist"
    PLANNER = "planner"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    MONITOR = "monitor"


@dataclass
class PlanStep:
    """Represents a single step in an execution plan"""

    action: str
    description: str
    expected_outcome: Optional[str] = None
    completed: bool = False
    result: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan"""

    steps: List[PlanStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for agent instances"""

    name: str = "llamaagent"
    max_iterations: int = 10
    timeout: Optional[float] = 300.0
    enable_memory: bool = True
    enable_tools: bool = True
    enable_logging: bool = True
    debug_mode: bool = False
    role: AgentRole = AgentRole.GENERALIST
    spree_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None")


@dataclass
class AgentResponse:
    """Standard response format for agent operations"""

    content: str
    success: bool = True
    execution_time: Optional[float] = None
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Set default values after initialization"""
        if self.metadata is None:
            self.metadata = {}
        if self.execution_time is None:
            self.execution_time = 0.0


@dataclass
class AgentStats:
    """Statistics for agent performance tracking"""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: Optional[float] = 0.0
    average_execution_time: Optional[float] = 0.0
    total_tokens_used: int = 0
    last_execution_time: Optional[float] = None

    def update(
        self, execution_time: float, success: bool, tokens_used: int = 0
    ) -> None:
        """Update statistics with new execution data"""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        if self.total_execution_time is None:
            self.total_execution_time = 0.0

        self.total_execution_time += execution_time
        self.total_tokens_used += tokens_used
        self.last_execution_time = time.time()

        # Update average execution time
        if self.total_executions > 0:
            self.average_execution_time = (
                self.total_execution_time / self.total_executions
            )

    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def reset(self) -> None:
        """Reset all statistics"""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0
        self.total_tokens_used = 0
        self.last_execution_time = None


class BaseAgent:
    """Abstract base class for all agents"""

    def __init__(self, config: AgentConfig, **kwargs: Any) -> None:
        """Initialize base agent"""
        self.config = config
        self.stats = AgentStats()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize optional components
        self.memory = None
        self.tools = None

        if config.enable_memory:
            self._initialize_memory()

        if config.enable_tools:
            self._initialize_tools()

    def _initialize_memory(self) -> None:
        """Initialize memory component"""
        try:
            from ..memory import MemoryManager

            self.memory = MemoryManager()
            self.logger.info("Memory manager initialized")
        except ImportError:
            self.logger.warning("Memory manager not available")

    def _initialize_tools(self) -> None:
        """Initialize tools component"""
        try:
            from ..tools import ToolManager

            self.tools = ToolManager()
            self.logger.info("Tool manager initialized")
        except ImportError:
            self.logger.warning("Tool manager not available")

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute a task (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement execute method")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.memory:
            await self.memory.cleanup()
        if self.tools:
            await self.tools.cleanup()

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "name": self.config.name,
            "total_executions": self.stats.total_executions,
            "successful_executions": self.stats.successful_executions,
            "failed_executions": self.stats.failed_executions,
            "success_rate": self.stats.get_success_rate(),
            "average_execution_time": self.stats.average_execution_time,
            "total_tokens_used": self.stats.total_tokens_used,
            "last_execution_time": self.stats.last_execution_time,
        }

    def reset_stats(self) -> None:
        """Reset agent statistics"""
        self.stats.reset()
        self.logger.info("Agent statistics reset")


# Export main classes
__all__ = [
    "AgentConfig",
    "AgentResponse",
    "AgentStats",
    "BaseAgent",
    "AgentRole",
    "PlanStep",
    "ExecutionPlan",
]
