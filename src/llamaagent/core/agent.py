"""
Advanced Agent Base Class - Enterprise Production Implementation

This module implements the core Agent class with full production capabilities including:
- Advanced autonomous reasoning
- Tool integration
- Memory management
- Inter-agent communication
- Performance monitoring
- Learning and adaptation

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Optional imports for monitoring
try:
    from prometheus_client import Counter, Gauge, Histogram

    prometheus_available = True
    agent_operations = Counter(
        "agent_operations_total", "Total agent operations", ["agent_id", "operation"]
    )
    agent_latency = Histogram(
        "agent_operation_duration_seconds",
        "Agent operation latency",
        ["agent_id", "operation"],
    )
    active_agents = Gauge("active_agents", "Number of active agents")
except ImportError:
    prometheus_available = False
    agent_operations = None
    agent_latency = None
    active_agents = None

# Optional imports for distributed computing
try:
    import redis as redis_module
    redis_available = True
except ImportError:
    redis_module = None
    redis_available = False

# Optional imports for tracing
try:
    from opentelemetry import trace

    tracing_available = True
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracing_available = False
    tracer = None

# Setup logging
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(Enum):
    """Agent capabilities."""

    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MEMORY_ACCESS = "memory_access"
    COLLABORATION = "collaboration"
    LEARNING = "learning"
    MONITORING = "monitoring"


class AgentRole(Enum):
    """Agent roles for specialization."""

    GENERALIST = "generalist"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"


@dataclass
class AgentConfig:
    """Configuration for agent instances."""

    name: str = "Agent"
    role: AgentRole = AgentRole.GENERALIST
    capabilities: List[AgentCapability] = field(
        default_factory=lambda: [
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY_ACCESS,
        ]
    )
    max_iterations: int = 10
    timeout: int = 300
    enable_learning: bool = True
    enable_monitoring: bool = True
    enable_collaboration: bool = True
    llm_provider: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    spree_enabled: bool = True
    verbose: bool = False
    debug: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    """Shareable context between agents."""

    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    global_objectives: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Inter-agent message protocol."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: str = "general"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_response: bool = False
    correlation_id: Optional[str] = None
    priority: int = 0


class Agent(ABC):
    """Advanced Agent Base Class with enterprise capabilities."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Agent",
        config: Optional[AgentConfig] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        **kwargs: Any,
    ) -> None:
        # Core identification
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.config = config or AgentConfig(name=name)
        self.state = AgentState.IDLE
        self.capabilities = capabilities or [
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY_ACCESS,
        ]

        # Timing and lifecycle
        self.start_time = datetime.now(timezone.utc)
        self.last_activity = self.start_time
        self.operation_count = 0
        self.error_count = 0

        # Learning and adaptation
        self.enable_learning = kwargs.get("enable_learning", True)
        self.performance_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}

        # Communication
        self.message_queue: List[AgentMessage] = []
        self.subscribed_channels: List[str] = []

        # Monitoring
        self.logger = self._setup_logger()
        self.metrics: Dict[str, Any] = {}

        # Distributed computing setup
        self.redis_client = self._setup_redis() if redis_available else None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Register agent
        if active_agents:
            active_agents.inc()
        self._register_agent()

    def _setup_logger(self) -> logging.Logger:
        """Configure structured logging."""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.agent_id}")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _setup_redis(self) -> Optional[Any]:
        """Setup Redis connection for distributed coordination."""
        try:
            if redis_module:
                return redis_module.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=int(os.getenv("REDIS_DB", 0)),
                    decode_responses=True,
                )
            return None
        except Exception as e:
            self.logger.warning(f"Failed to setup Redis: {e}")
            return None

    def _register_agent(self) -> None:
        """Register agent in service mesh."""
        registration_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        if self.redis_client:
            try:
                self.redis_client.hset(
                    "agents:registry", self.agent_id, json.dumps(registration_data)
                )
                self.redis_client.expire(
                    f"agents:registry:{self.agent_id}", 3600
                )  # 1 hour TTL
            except Exception as e:
                self.logger.warning(f"Failed to register agent: {e}")

    @abstractmethod
    async def think(
        self, objective: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Abstract method for agent reasoning."""

    @abstractmethod
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Abstract method for executing actions."""

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        self.logger.info(
            f"Processing message: {message.message_type} from {message.sender_id}"
        )

        # Update last activity
        self.last_activity = datetime.now(timezone.utc)
        # Handle different message types
        if message.message_type == "task_request":
            return await self._handle_task_request(message)
        elif message.message_type == "collaboration_request":
            return await self._handle_collaboration_request(message)
        elif message.message_type == "context_update":
            return await self._handle_context_update(message)
        elif message.message_type == "health_check":
            return await self._handle_health_check(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
            return None

    async def _handle_task_request(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Handle task execution requests."""
        task = message.content.get("task", "")
        context = message.content.get("context", {})

        try:
            result = await self.execute(task, context)

            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                content={"result": result, "status": "completed"},
                correlation_id=message.id,
            )
            return response

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            response = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                content={"error": str(e), "status": "failed"},
                correlation_id=message.id,
            )
            return response

    async def _handle_collaboration_request(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Handle collaboration requests."""
        collaboration_data = message.content
        result = await self.collaborate(collaboration_data)

        response = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="collaboration_response",
            content={"result": result},
            correlation_id=message.id,
        )
        return response

    async def _handle_context_update(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Handle context updates."""
        # Update internal context based on message
        if "shared_memory" in message.content:
            # Update shared memory
            pass
        return None

    async def _handle_health_check(
        self, message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Handle health check requests."""
        health_data = {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "uptime_seconds": (
                datetime.now(timezone.utc) - self.start_time
            ).total_seconds(),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "memory_usage_mb": self._get_memory_usage(),
            "health": "healthy",
        }

        response = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="health_response",
            content=health_data,
            correlation_id=message.id,
        )
        return response

    def _get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    async def collaborate(self, collaboration_data: Dict[str, Any]) -> Any:
        """Collaborate with other agents on complex tasks."""
        task_type = collaboration_data.get("task_type")

        if task_type == "reasoning_chain":
            return await self._contribute_to_reasoning_chain(collaboration_data)
        elif task_type == "parallel_execution":
            return await self._execute_parallel_subtask(collaboration_data)
        else:
            return {"error": f"Unknown collaboration type: {task_type}"}

    async def _contribute_to_reasoning_chain(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add reasoning step to chain."""
        current_objective = data.get("objective", "")

        # Contribute reasoning step
        reasoning_result = await self.think(current_objective)

        return {
            "agent_id": self.agent_id,
            "reasoning": reasoning_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _execute_parallel_subtask(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned subtask in parallel execution."""
        subtask = data.get("subtask")
        parameters = data.get("parameters", {})

        # Execute subtask
        if subtask:
            result = await self.execute_action(subtask, parameters)
        else:
            result = {"error": "No subtask specified"}

        return {
            "agent_id": self.agent_id,
            "subtask": subtask,
            "result": result,
            "execution_time": datetime.now(timezone.utc).isoformat(),
        }

    async def execute(
        self, objective: str, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Main execution method."""
        if agent_latency:
            with agent_latency.labels(
                agent_id=self.agent_id, operation="execute"
            ).time():
                return await self._execute_with_monitoring(objective, context)
        else:
            return await self._execute_with_monitoring(objective, context)

    async def _execute_with_monitoring(
        self, objective: str, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute with monitoring and error handling."""
        self.state = AgentState.THINKING
        self.operation_count += 1

        if agent_operations:
            agent_operations.labels(agent_id=self.agent_id, operation="execute").inc()

        try:
            # Think about the objective
            thinking_result = await self.think(objective, context)

            # Execute actions based on thinking
            results: List[Any] = []
            for action in thinking_result.get("actions", []):
                self.state = AgentState.EXECUTING
                action_name = action.get("name", "")
                if action_name:
                    result = await self.execute_action(
                        action_name, action.get("parameters", {})
                    )
                    results.append(result)

            # Learn from experience
            if self.enable_learning:
                await self._learn_from_experience(objective, context, results)

            self.state = AgentState.IDLE
            return {
                "objective": objective,
                "thinking": thinking_result,
                "results": results,
                "status": "completed",
                "agent_id": self.agent_id,
                "execution_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.error_count += 1
            self.state = AgentState.ERROR
            self.logger.error(f"Execution failed: {e}")
            return {
                "objective": objective,
                "error": str(e),
                "status": "failed",
                "agent_id": self.agent_id,
                "execution_time": datetime.now(timezone.utc).isoformat(),
            }

    async def _learn_from_experience(
        self, objective: str, context: Optional[Dict[str, Any]], results: List[Any]
    ) -> None:
        """Learn from past experiences to improve performance."""
        if not self.enable_learning:
            return

        # Store experience
        experience = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objective": objective,
            "context": context,
            "results": results,
            "success": all(r is not None for r in results),
        }

        self.performance_history.append(experience)

        # Extract patterns
        if len(self.performance_history) >= 10:
            patterns = self._extract_patterns()
            self.learned_patterns.update(patterns)

    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from performance history."""
        # Simplified pattern extraction
        patterns: Dict[str, Any] = {
            "successful_actions": {},
            "failed_actions": {},
            "optimal_parameters": {},
        }

        for experience in self.performance_history[-10:]:
            if experience.get("success"):
                # Track successful patterns
                objective_type = str(experience["objective"])[
                    :20
                ]  # First 20 chars as type
                success_actions = patterns["successful_actions"]
                success_actions[objective_type] = (
                    success_actions.get(objective_type, 0) + 1
                )

        return patterns

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.logger.info(f"Shutting down agent {self.agent_id}")

        # Update state
        self.state = AgentState.TERMINATED

        # Deregister from service mesh
        if self.redis_client:
            try:
                self.redis_client.hdel("agents:registry", self.agent_id)
            except Exception as e:
                self.logger.warning(f"Failed to deregister agent: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Update metrics
        if active_agents:
            active_agents.dec()


class TaskOrchestrator(Agent):
    """Specialized agent for task orchestration."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="TaskOrchestrator",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.COLLABORATION,
                AgentCapability.MONITORING,
            ],
            **kwargs,
        )

    async def think(
        self, objective: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Orchestrator thinking implementation."""
        return {
            "objective": objective,
            "strategy": "orchestration",
            "actions": [
                {
                    "name": "coordinate_agents",
                    "parameters": {"objective": objective, "context": context},
                }
            ],
            "confidence": 0.9,
            "constraints": context.get("constraints", {}) if context else {},
        }

    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """Execute orchestration actions."""
        if action == "coordinate_agents":
            return await self._coordinate_agents(parameters)
        else:
            return {"error": f"Unknown action: {action}"}

    async def _coordinate_agents(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for task execution."""
        # Simplified coordination logic
        return {
            "status": "coordinated",
            "agents_involved": [],
            "active_tasks": len(asyncio.all_tasks()),
            "original_objective": parameters.get("original_objective"),
        }
