"""Agent pool management for efficient resource utilization."""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..agents.base import AgentConfig, AgentResponse, BaseAgent
from .agent_spawner import AgentSpawner, SpawnConfig

logger = logging.getLogger(__name__)


class PoolStrategy(str, Enum):
    """Load balancing strategies for agent pools."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CAPABILITY_BASED = "capability_based"
    ADAPTIVE = "adaptive"


class AgentStatus(str, Enum):
    """Status of agents in the pool."""

    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class PoolConfig:
    """Configuration for agent pools."""

    # Pool size settings
    min_agents: int = 1
    max_agents: int = 10
    initial_agents: int = 3

    # Scaling settings
    auto_scale: bool = True
    scale_up_threshold: float = 0.8  # 80% busy
    scale_down_threshold: float = 0.2  # 20% busy
    scale_check_interval: float = 30.0  # seconds

    # Load balancing
    strategy: PoolStrategy = PoolStrategy.LEAST_LOADED

    # Health checks
    health_check_interval: float = 60.0
    max_consecutive_errors: int = 3

    # Task queue settings
    max_queue_size: int = 1000
    task_timeout: float = 300.0

    # Agent configuration
    default_agent_config: Optional[AgentConfig] = None
    spawn_config: Optional[SpawnConfig] = None


@dataclass
class AgentStats:
    """Statistics for an agent in the pool."""

    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    last_task_time: float = 0.0
    consecutive_errors: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    current_task: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    performance_score: float = 1.0  # 0-1, higher is better


@dataclass
class PoolStats:
    """Overall statistics for the agent pool."""

    total_agents: int = 0
    idle_agents: int = 0
    busy_agents: int = 0
    error_agents: int = 0

    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_time: float = 0.0

    queue_size: int = 0
    pool_utilization: float = 0.0

    uptime: float = 0.0
    last_scale_event: Optional[float] = None
    scale_events: int = 0


@dataclass
class PoolTask:
    """Task to be executed by the pool."""

    task_id: str
    task: str
    context: Optional[Dict[str, Any]] = None
    priority: int = 1  # Higher is more important
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_agent: Optional[str] = None
    result: Optional[AgentResponse] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    requirements: Set[str] = field(default_factory=set)  # Required capabilities

    def __lt__(self, other: PoolTask) -> bool:
        """For priority queue comparison."""
        return self.priority > other.priority


class AgentPool:
    """Manages a pool of agents for efficient task distribution."""

    def __init__(
        self,
        config: PoolConfig,
        spawner: Optional[AgentSpawner] = None,
    ) -> None:
        self.config = config
        self.spawner = spawner or AgentSpawner()

        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_stats: Dict[str, AgentStats] = {}

        # Task queue (priority queue)
        self.task_queue: List[PoolTask] = []
        self.active_tasks: Dict[str, PoolTask] = {}
        self.completed_tasks: Dict[str, PoolTask] = {}

        # Pool state
        self.is_running = False
        self.start_time = time.time()
        self._task_counter = 0

        # Load balancing
        self._round_robin_index = 0

        # Locks
        self._agent_lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()

        # Background tasks
        self._worker_tasks: Set[asyncio.Task] = set()
        self._monitor_task: Optional[asyncio.Task] = None
        self._scaler_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the agent pool."""
        if self.is_running:
            return

        logger.info(f"Starting agent pool with {self.config.initial_agents} agents")

        # Spawn initial agents
        for i in range(self.config.initial_agents):
            await self._spawn_agent()

        self.is_running = True

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_health())

        if self.config.auto_scale:
            self._scaler_task = asyncio.create_task(self._auto_scaler())

        # Start worker tasks for each agent
        for agent_id in self.agents:
            task = asyncio.create_task(self._agent_worker(agent_id))
            self._worker_tasks.add(task)

    async def stop(self) -> None:
        """Stop the agent pool."""
        if not self.is_running:
            return

        logger.info("Stopping agent pool")
        self.is_running = False

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._scaler_task:
            self._scaler_task.cancel()

        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            *self._worker_tasks,
            self._monitor_task,
            self._scaler_task,
            return_exceptions=True,
        )

        # Terminate all agents
        for agent_id in list(self.agents.keys()):
            await self.spawner.terminate_agent(agent_id)

        self.agents.clear()
        self.agent_stats.clear()

    async def submit_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        requirements: Optional[Set[str]] = None,
    ) -> str:
        """Submit a task to the pool."""
        task_id = f"task_{self._task_counter}_{int(time.time())}"
        self._task_counter += 1

        pool_task = PoolTask(
            task_id=task_id,
            task=task,
            context=context,
            priority=priority,
            requirements=requirements or set(),
        )

        async with self._queue_lock:
            if len(self.task_queue) >= self.config.max_queue_size:
                raise ValueError("Task queue is full")

            heapq.heappush(self.task_queue, pool_task)

        logger.debug(f"Submitted task {task_id} with priority {priority}")
        return task_id

    async def get_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[AgentResponse]:
        """Get the result of a submitted task."""
        timeout = timeout or self.config.task_timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.error:
                    raise Exception(f"Task failed: {task.error}")
                return task.result

            await asyncio.sleep(0.1)

        # Check if task is still active
        if task_id in self.active_tasks:
            raise TimeoutError(f"Task {task_id} timed out")

        return None

    async def submit_and_wait(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        requirements: Optional[Set[str]] = None,
        timeout: Optional[float] = None,
    ) -> AgentResponse:
        """Submit a task and wait for the result."""
        task_id = await self.submit_task(task, context, priority, requirements)
        result = await self.get_result(task_id, timeout)

        if result is None:
            raise ValueError(f"No result for task {task_id}")

        return result

    def get_pool_stats(self) -> PoolStats:
        """Get current pool statistics."""
        stats = PoolStats(
            total_agents=len(self.agents),
            idle_agents=sum(
                1 for s in self.agent_stats.values() if s.status == AgentStatus.IDLE
            ),
            busy_agents=sum(
                1 for s in self.agent_stats.values() if s.status == AgentStatus.BUSY
            ),
            error_agents=sum(
                1 for s in self.agent_stats.values() if s.status == AgentStatus.ERROR
            ),
            total_tasks_completed=sum(
                s.tasks_completed for s in self.agent_stats.values()
            ),
            total_tasks_failed=sum(s.tasks_failed for s in self.agent_stats.values()),
            queue_size=len(self.task_queue),
            uptime=time.time() - self.start_time,
        )

        # Calculate average task time
        total_time = sum(s.total_execution_time for s in self.agent_stats.values())
        if stats.total_tasks_completed > 0:
            stats.average_task_time = total_time / stats.total_tasks_completed

        # Calculate utilization
        if stats.total_agents > 0:
            stats.pool_utilization = stats.busy_agents / stats.total_agents

        return stats

    async def _spawn_agent(self) -> Optional[str]:
        """Spawn a new agent for the pool."""
        config = self.config.spawn_config or SpawnConfig(
            agent_config=self.config.default_agent_config
        )

        result = await self.spawner.spawn_agent(
            task="Pool worker agent",
            config=config,
        )

        if result.success and result.agent:
            async with self._agent_lock:
                self.agents[result.agent_id] = result.agent
                self.agent_stats[result.agent_id] = AgentStats(agent_id=result.agent_id)

            # Start worker task for the new agent
            task = asyncio.create_task(self._agent_worker(result.agent_id))
            self._worker_tasks.add(task)

            logger.info(f"Spawned new pool agent: {result.agent_id}")
            return result.agent_id

        return None

    async def _agent_worker(self, agent_id: str) -> None:
        """Worker loop for an agent."""
        logger.debug(f"Starting worker for agent {agent_id}")

        while self.is_running and agent_id in self.agents:
            try:
                # Get next task
                task = await self._get_next_task(agent_id)
                if not task:
                    await asyncio.sleep(0.1)
                    continue

                # Update agent status
                self.agent_stats[agent_id].status = AgentStatus.BUSY
                self.agent_stats[agent_id].current_task = task.task_id

                # Execute task
                start_time = time.time()
                task.started_at = start_time
                self.active_tasks[task.task_id] = task

                try:
                    agent = self.agents[agent_id]
                    response = await agent.execute(task.task, task.context)

                    # Update task result
                    task.result = response
                    task.completed_at = time.time()

                    # Update agent stats
                    execution_time = task.completed_at - start_time
                    stats = self.agent_stats[agent_id]
                    stats.tasks_completed += 1
                    stats.total_execution_time += execution_time
                    stats.last_task_time = execution_time
                    stats.average_task_time = (
                        stats.total_execution_time / stats.tasks_completed
                    )
                    stats.consecutive_errors = 0
                    stats.performance_score = min(
                        1.0,
                        stats.performance_score * 1.01,  # Slight increase
                    )

                except Exception as e:
                    logger.error(f"Agent {agent_id} failed task: {e}")
                    task.error = str(e)
                    task.completed_at = time.time()

                    # Update agent stats
                    stats = self.agent_stats[agent_id]
                    stats.tasks_failed += 1
                    stats.consecutive_errors += 1
                    stats.performance_score = max(
                        0.1,
                        stats.performance_score * 0.9,  # Decrease
                    )

                    # Check for repeated errors
                    if stats.consecutive_errors >= self.config.max_consecutive_errors:
                        stats.status = AgentStatus.ERROR
                        logger.warning(
                            f"Agent {agent_id} marked as error after {stats.consecutive_errors} consecutive errors"
                        )

                finally:
                    # Move task to completed
                    del self.active_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task

                    # Update agent status
                    if self.agent_stats[agent_id].status != AgentStatus.ERROR:
                        self.agent_stats[agent_id].status = AgentStatus.IDLE
                    self.agent_stats[agent_id].current_task = None
                    self.agent_stats[agent_id].last_active = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error for agent {agent_id}: {e}")
                await asyncio.sleep(1)

    async def _get_next_task(self, agent_id: str) -> Optional[PoolTask]:
        """Get the next task for an agent based on the pool strategy."""
        async with self._queue_lock:
            if not self.task_queue:
                return None

            # Filter tasks based on agent capabilities
            agent_stats = self.agent_stats.get(agent_id)
            if not agent_stats:
                return None

            # Find suitable task
            suitable_tasks = []
            for i, task in enumerate(self.task_queue):
                if not task.requirements or task.requirements.issubset(
                    agent_stats.capabilities
                ):
                    suitable_tasks.append(i, task)

            if not suitable_tasks:
                return None

            # Select task based on strategy
            if self.config.strategy == PoolStrategy.ROUND_ROBIN:
                # Just take the first suitable task
                index, task = suitable_tasks[0]

            elif self.config.strategy == PoolStrategy.CAPABILITY_BASED:
                # Prefer tasks that match agent capabilities
                best_match = suitable_tasks[0]
                best_score = len(
                    best_match[1].requirements.intersection(agent_stats.capabilities)
                )

                for index, task in suitable_tasks[1:]:
                    score = len(
                        task.requirements.intersection(agent_stats.capabilities)
                    )
                    if score > best_score:
                        best_match = (index, task)
                        best_score = score

                index, task = best_match

            else:
                # Default: take highest priority
                index, task = suitable_tasks[0]

            # Remove task from queue
            self.task_queue.pop(index)
            heapq.heapify(self.task_queue)  # Restore heap property

            task.assigned_agent = agent_id
            return task

    async def _monitor_health(self) -> None:
        """Monitor agent health and handle failures."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check agent health
                for agent_id, stats in self.agent_stats.items():
                    if stats.status == AgentStatus.ERROR:
                        # Try to recover error agents
                        logger.info(f"Attempting to recover error agent {agent_id}")
                        stats.status = AgentStatus.IDLE
                        stats.consecutive_errors = 0

                    # Check for stale agents
                    if (
                        stats.status == AgentStatus.IDLE
                        and time.time() - stats.last_active > 3600  # 1 hour
                    ):
                        logger.info(f"Agent {agent_id} idle for too long")
                        # Could implement agent recycling here

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _auto_scaler(self) -> None:
        """Automatically scale the pool based on load."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.scale_check_interval)

                stats = self.get_pool_stats()

                # Scale up if utilization is high
                if (
                    stats.pool_utilization > self.config.scale_up_threshold
                    and stats.total_agents < self.config.max_agents
                ):
                    logger.info(f"Scaling up: utilization={stats.pool_utilization:.2%}")
                    await self._spawn_agent()
                    stats.scale_events += 1
                    stats.last_scale_event = time.time()

                # Scale down if utilization is low
                elif (
                    stats.pool_utilization < self.config.scale_down_threshold
                    and stats.total_agents > self.config.min_agents
                    and stats.idle_agents > 0
                ):
                    logger.info(
                        f"Scaling down: utilization={stats.pool_utilization:.2%}"
                    )

                    # Find an idle agent to remove
                    for agent_id, agent_stats in self.agent_stats.items():
                        if agent_stats.status == AgentStatus.IDLE:
                            await self.spawner.terminate_agent(agent_id)
                            async with self._agent_lock:
                                del self.agents[agent_id]
                                del self.agent_stats[agent_id]
                            stats.scale_events += 1
                            stats.last_scale_event = time.time()
                            break

            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")

    async def set_agent_capabilities(
        self,
        agent_id: str,
        capabilities: Set[str],
    ) -> None:
        """Set capabilities for an agent."""
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id].capabilities = capabilities

    async def pause_agent(self, agent_id: str) -> bool:
        """Pause an agent in the pool."""
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id].status = AgentStatus.PAUSED
            return await self.spawner.pause_agent(agent_id)
        return False

    async def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent."""
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id].status = AgentStatus.IDLE
            return await self.spawner.resume_agent(agent_id)
        return False
