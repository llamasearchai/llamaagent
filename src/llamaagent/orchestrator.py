"""
Advanced Agent Orchestrator for LlamaAgent

This module provides intelligent agent coordination, task distribution,
and multi-agent workflow management.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .agents.react import ReactAgent
from .core.agent import AgentState
from .types import TaskInput, TaskOutput, TaskStatus

logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Orchestration strategies for task distribution."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"


@dataclass
class TaskAssignment:
    """Task assignment tracking."""

    task_id: str
    agent_id: str
    task_input: TaskInput
    status: TaskStatus
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[TaskOutput] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentPool:
    """Agent pool management."""

    agents: Dict[str, ReactAgent] = field(default_factory=dict)
    agent_states: Dict[str, AgentState] = field(default_factory=dict)
    agent_capabilities: Dict[str, Set[str]] = field(default_factory=dict)
    agent_load: Dict[str, int] = field(default_factory=dict)
    agent_health: Dict[str, bool] = field(default_factory=dict)


class AgentOrchestrator:
    """Advanced agent orchestrator with intelligent task distribution."""

    def __init__(
        self,
        strategy: OrchestrationStrategy = OrchestrationStrategy.CAPABILITY_BASED,
        max_concurrent_tasks: int = 10,
        health_check_interval: int = 30,
        enable_monitoring: bool = True,
    ):
        self.strategy = strategy
        self.max_concurrent_tasks = max_concurrent_tasks
        self.health_check_interval = health_check_interval
        self.enable_monitoring = enable_monitoring

        # Agent management
        self.agent_pool = AgentPool()
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.task_queue: asyncio.Queue[TaskAssignment] = asyncio.Queue()
        self.active_tasks: Set[str] = set()

        # Monitoring
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0,
            "agent_utilization": {},
        }

        # Background tasks
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._task_processor_task: Optional[asyncio.Task[None]] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        self.logger.info("Initializing Agent Orchestrator")

        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._task_processor_task = asyncio.create_task(self._task_processor_loop())

    async def shutdown(self) -> None:
        """Shutdown the orchestrator gracefully."""
        self.logger.info("Shutting down Agent Orchestrator")
        self._shutdown_event.set()

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._task_processor_task:
            self._task_processor_task.cancel()

        # Wait for tasks to complete
        tasks_to_wait = []
        if self._health_check_task:
            tasks_to_wait.append(self._health_check_task)
        if self._task_processor_task:
            tasks_to_wait.append(self._task_processor_task)

        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    def register_agent(self, agent: ReactAgent) -> None:
        """Register an agent with the orchestrator."""
        agent_id = agent.agent_id
        self.agent_pool.agents[agent_id] = agent
        self.agent_pool.agent_states[agent_id] = AgentState.IDLE
        self.agent_pool.agent_capabilities[agent_id] = set(agent.tools.list_names())
        self.agent_pool.agent_load[agent_id] = 0
        self.agent_pool.agent_health[agent_id] = True

        self.logger.info(f"Registered agent {agent_id} ({agent.name})")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator."""
        if agent_id in self.agent_pool.agents:
            del self.agent_pool.agents[agent_id]
            del self.agent_pool.agent_states[agent_id]
            del self.agent_pool.agent_capabilities[agent_id]
            del self.agent_pool.agent_load[agent_id]
            del self.agent_pool.agent_health[agent_id]

            self.logger.info(f"Unregistered agent {agent_id}")

    async def submit_task(self, task_input: TaskInput) -> str:
        """Submit a task for execution."""
        task_id = str(uuid.uuid4())
        # Create task assignment
        assignment = TaskAssignment(
            task_id=task_id,
            agent_id="",  # Will be assigned later
            task_input=task_input,
            status=TaskStatus.QUEUED,
        )

        self.task_assignments[task_id] = assignment
        await self.task_queue.put(assignment)

        self.metrics["total_tasks"] += 1
        self.logger.info(f"Submitted task {task_id}")

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[TaskAssignment]:
        """Get the status of a task."""
        return self.task_assignments.get(task_id)

    async def get_task_result(self, task_id: str) -> Optional[TaskOutput]:
        """Get the result of a completed task."""
        assignment = self.task_assignments.get(task_id)
        if assignment and assignment.status == TaskStatus.COMPLETED:
            return assignment.result
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        assignment = self.task_assignments.get(task_id)
        if not assignment:
            return False

        if assignment.status in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
            assignment.status = TaskStatus.CANCELLED
            assignment.completed_at = datetime.now(timezone.utc)
            # Remove from active tasks
            self.active_tasks.discard(task_id)

            self.logger.info(f"Cancelled task {task_id}")
            return True

        return False

    async def _task_processor_loop(self) -> None:
        """Main task processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                assignment = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Check if we have capacity
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    # Put task back in queue
                    await self.task_queue.put(assignment)
                    continue

                # Assign and execute task
                await self._assign_and_execute_task(assignment)

            except asyncio.TimeoutError:
                # Continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error in task processor loop: {e}")

    async def _assign_and_execute_task(self, assignment: TaskAssignment) -> None:
        """Assign a task to an agent and execute it."""
        # Select agent based on strategy
        agent_id = await self._select_agent(assignment.task_input)

        if not agent_id:
            assignment.status = TaskStatus.FAILED
            assignment.error = "No suitable agent available"
            assignment.completed_at = datetime.now(timezone.utc)
            self.metrics["failed_tasks"] += 1
            return

        # Assign task to agent
        assignment.agent_id = agent_id
        assignment.status = TaskStatus.RUNNING
        assignment.started_at = datetime.now(timezone.utc)
        # Update agent load
        self.agent_pool.agent_load[agent_id] += 1
        self.agent_pool.agent_states[agent_id] = AgentState.EXECUTING

        # Add to active tasks
        self.active_tasks.add(assignment.task_id)

        # Execute task
        asyncio.create_task(self._execute_task(assignment))

    async def _execute_task(self, assignment: TaskAssignment) -> None:
        """Execute a task using the assigned agent."""
        agent = self.agent_pool.agents[assignment.agent_id]

        try:
            # Execute task
            result = await agent.execute_task(assignment.task_input)

            # Update assignment
            assignment.result = result
            assignment.status = TaskStatus.COMPLETED
            assignment.completed_at = datetime.now(timezone.utc)
            # Update metrics
            self.metrics["completed_tasks"] += 1
            execution_time = (
                assignment.completed_at - assignment.started_at
            ).total_seconds()
            self._update_average_completion_time(execution_time)

            self.logger.info(f"Task {assignment.task_id} completed successfully")

        except Exception as e:
            # Handle failure
            assignment.error = str(e)
            assignment.status = TaskStatus.FAILED
            assignment.completed_at = datetime.now(timezone.utc)
            # Check if we should retry
            if assignment.retry_count < assignment.max_retries:
                assignment.retry_count += 1
                assignment.status = TaskStatus.QUEUED
                assignment.started_at = None
                assignment.completed_at = None

                # Put back in queue for retry
                await self.task_queue.put(assignment)
                self.logger.info(
                    f"Task {assignment.task_id} failed, retrying ({assignment.retry_count}/{assignment.max_retries})"
                )
            else:
                self.metrics["failed_tasks"] += 1
                self.logger.error(f"Task {assignment.task_id} failed permanently: {e}")

        finally:
            # Clean up
            self.active_tasks.discard(assignment.task_id)
            self.agent_pool.agent_load[assignment.agent_id] -= 1
            self.agent_pool.agent_states[assignment.agent_id] = AgentState.IDLE

    async def _select_agent(self, task_input: TaskInput) -> Optional[str]:
        """Select the best agent for a task based on the current strategy."""
        available_agents = [
            agent_id
            for agent_id, health in self.agent_pool.agent_health.items()
            if health and self.agent_pool.agent_states[agent_id] == AgentState.IDLE
        ]

        if not available_agents:
            return None

        if self.strategy == OrchestrationStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_agents)
        elif self.strategy == OrchestrationStrategy.LOAD_BALANCED:
            return self._select_load_balanced(available_agents)
        elif self.strategy == OrchestrationStrategy.CAPABILITY_BASED:
            return self._select_capability_based(available_agents, task_input)
        else:
            # Default to first available
            return available_agents[0]

    def _select_round_robin(self, available_agents: List[str]) -> str:
        """Select agent using round-robin strategy."""
        # Simple round-robin based on task count
        min_tasks = min(
            self.metrics.get("agent_utilization", {}).get(aid, 0)
            for aid in available_agents
        )
        for agent_id in available_agents:
            if self.metrics.get("agent_utilization", {}).get(agent_id, 0) == min_tasks:
                return agent_id
        return available_agents[0]

    def _select_load_balanced(self, available_agents: List[str]) -> str:
        """Select agent using load-balanced strategy."""
        # Select agent with lowest current load
        min_load = min(self.agent_pool.agent_load[aid] for aid in available_agents)
        for agent_id in available_agents:
            if self.agent_pool.agent_load[agent_id] == min_load:
                return agent_id
        return available_agents[0]

    def _select_capability_based(
        self, available_agents: List[str], task_input: TaskInput
    ) -> str:
        """Select agent based on capabilities required for the task."""
        # For now, just return the first available agent
        # In a real implementation, we would analyze the task requirements
        return available_agents[0]

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all agents."""
        for agent_id, agent in self.agent_pool.agents.items():
            try:
                # Simple health check - could be more sophisticated
                self.agent_pool.agent_health[agent_id] = True
            except Exception as e:
                self.logger.warning(f"Agent {agent_id} health check failed: {e}")
                self.agent_pool.agent_health[agent_id] = False

    def _update_average_completion_time(self, execution_time: float) -> None:
        """Update the average completion time metric."""
        completed_tasks = self.metrics["completed_tasks"]
        if completed_tasks == 1:
            self.metrics["average_completion_time"] = execution_time
        else:
            current_avg = self.metrics["average_completion_time"]
            self.metrics["average_completion_time"] = (
                current_avg * (completed_tasks - 1) + execution_time
            ) / completed_tasks

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            **self.metrics,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "registered_agents": len(self.agent_pool.agents),
            "healthy_agents": sum(
                1 for h in self.agent_pool.agent_health.values() if h
            ),
            "agent_states": dict(self.agent_pool.agent_states),
            "agent_loads": dict(self.agent_pool.agent_load),
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about registered agents."""
        return {
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "state": self.agent_pool.agent_states[agent_id].value,
                    "load": self.agent_pool.agent_load[agent_id],
                    "health": self.agent_pool.agent_health[agent_id],
                    "capabilities": list(self.agent_pool.agent_capabilities[agent_id]),
                }
                for agent_id, agent in self.agent_pool.agents.items()
            }
        }
