"""
Distributed Orchestrator - Enterprise Production Implementation

This module implements advanced distributed orchestration capabilities including:
- Kafka-based message routing
- Redis-based state management
- Complex workflow execution
- Task dependency management
- Performance monitoring
- Fault tolerance and recovery

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import heapq
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from redis import Redis
except ImportError:
    Redis = None

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    # Create a local registry to avoid conflicts during testing
    local_registry = CollectorRegistry()

    orchestration_tasks = Counter(
        "orchestration_tasks_total",
        "Total orchestration tasks",
        ["status"],
        registry=local_registry,
    )
    active_workflows = Gauge(
        "active_workflows", "Number of active workflows", registry=local_registry
    )
    task_execution_time = Histogram(
        "task_execution_seconds", "Task execution time", registry=local_registry
    )
except ImportError:
    orchestration_tasks = None
    active_workflows = None
    task_execution_time = None

from .agent import AgentCapability


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Orchestration task definition"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    objective: str = ""
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    context: Dict[str, Any] = field(default_factory=dict)
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """Orchestration workflow definition"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class DistributedOrchestrator:
    """
    Advanced distributed orchestrator for managing agent tasks and workflows
    """

    def __init__(
        self,
        orchestrator_id: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        kafka_bootstrap_servers: Optional[List[str]] = None,
        max_concurrent_workflows: int = 10,
        agent_timeout_seconds: int = 300,
    ):
        self.id = orchestrator_id or str(uuid.uuid4())
        self.redis_url = redis_url
        self.kafka_servers = kafka_bootstrap_servers or ["localhost:9092"]
        self.max_concurrent_workflows = max_concurrent_workflows
        self.agent_timeout_seconds = agent_timeout_seconds

        # Storage - type annotated
        self.redis_client: Optional[Any] = None
        if Redis:
            try:
                self.redis_client = Redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self.redis_client = None

        # Message queue - type annotated
        self.kafka_producer: Optional[Any] = None
        if KafkaProducer:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode(
                        "utf-8"
                    ),  # type: ignore
                )
            except Exception:
                self.kafka_producer = None

        # State management
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: List[Tuple[int, Task]] = []
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Task] = {}

        # Monitoring
        self.logger = self._setup_logger()
        self._running: bool = False
        self._scheduler_task: Optional[asyncio.Task[None]] = None
        self._monitor_task: Optional[asyncio.Task[None]] = None

    def _setup_logger(self):
        """Setup logging for the orchestrator"""
        logger = logging.getLogger(f"Orchestrator_{self.id}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def start(self) -> None:
        """Start orchestrator services"""
        self.logger.info(f"Starting orchestrator {self.id}")
        self._running = True

        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
        self._monitor_task = asyncio.create_task(self._run_monitor())

        # Start agent discovery
        await self._discover_agents()

    async def stop(self) -> None:
        """Stop orchestrator services"""
        self.logger.info(f"Stopping orchestrator {self.id}")
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            self.redis_client.close()

    async def submit_workflow(self, workflow: Workflow) -> str:
        """Submit a workflow for execution"""
        self.logger.info(f"Submitting workflow {workflow.id}: {workflow.name}")

        # Store workflow
        self.workflows[workflow.id] = workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc)
        # Add tasks to queue
        for task in workflow.tasks.values():
            if not task.dependencies:
                heapq.heappush(self.task_queue, (task.priority.value, task))

        # Update metrics
        if active_workflows:
            active_workflows.inc()

        # Persist to Redis
        if self.redis_client:
            try:
                await self._persist_workflow(workflow)
            except Exception as e:
                self.logger.error(f"Failed to persist workflow {workflow.id}: {e}")

        return workflow.id

    async def _discover_agents(self) -> None:
        """Discover available agents in the network"""
        # In production, this would discover agents from a registry
        # For now, we'll use a simple approach
        self.logger.info("Discovering agents...")

        # Mock agent discovery
        self.agent_registry = {
            "agent_1": {
                "capabilities": [AgentCapability.ANALYSIS, AgentCapability.WEB_SEARCH],
                "status": "available",
                "last_seen": datetime.now(timezone.utc),
                "max_concurrent_tasks": 5,
            },
            "agent_2": {
                "capabilities": [
                    AgentCapability.CODE_EXECUTION,
                    AgentCapability.ANALYSIS,
                ],
                "status": "available",
                "last_seen": datetime.now(timezone.utc),
                "max_concurrent_tasks": 3,
            },
        }

    async def _run_scheduler(self) -> None:
        """Main scheduler loop"""
        while self._running:
            try:
                await self._process_task_queue()
                await asyncio.sleep(1)  # Scheduler interval
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")

    async def _run_monitor(self) -> None:
        """Monitor task execution and handle timeouts"""
        while self._running:
            try:
                await self._check_task_timeouts()
                await self._update_workflow_status()
                await asyncio.sleep(10)  # Monitor interval
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")

    async def _process_task_queue(self) -> None:
        """Process tasks in the queue"""
        while (
            self.task_queue and len(self.active_tasks) < self.max_concurrent_workflows
        ):
            _, task = heapq.heappop(self.task_queue)

            if task.status == TaskStatus.PENDING:
                success = await self._assign_task(task)
                if success:
                    self.active_tasks[task.id] = task
                    if orchestration_tasks:
                        orchestration_tasks.labels(status="assigned").inc()

    async def _assign_task(self, task: Task) -> bool:
        """Assign a task to suitable agents"""
        try:
            # Find suitable agents
            suitable_agents = self._find_suitable_agents(task)

            if not suitable_agents:
                self.logger.warning(f"No suitable agents found for task {task.id}")
                return False

            # Determine number of agents needed
            num_agents = min(
                len(suitable_agents),
                (
                    1
                    if AgentCapability.MULTIMODAL not in task.required_capabilities
                    else 3
                ),
            )

            # Select agents
            selected_agents = suitable_agents[:num_agents]
            task.assigned_agents = selected_agents
            task.status = TaskStatus.ASSIGNED
            task.started_at = datetime.now(timezone.utc)
            # Send task to agents
            for agent_id in selected_agents:
                await self._send_task_to_agent(agent_id, task)

            self.logger.info(
                f"Assigned task {task.id} to agents: {', '.join(selected_agents)}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to assign task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return False

    def _find_suitable_agents(self, task: Task) -> List[str]:
        """Find agents suitable for the task"""
        suitable = []

        for agent_id, agent_info in self.agent_registry.items():
            if agent_info["status"] != "available":
                continue

            # Check capabilities
            agent_capabilities = set(agent_info["capabilities"])
            required_capabilities = set(task.required_capabilities)

            if not required_capabilities.issubset(agent_capabilities):
                continue

            # Check capacity
            current_tasks = sum(
                1 for t in self.active_tasks.values() if agent_id in t.assigned_agents
            )
            if current_tasks >= agent_info["max_concurrent_tasks"]:
                continue

            suitable.append(agent_id)

        # Sort by availability and capabilities
        suitable.sort(
            key=lambda aid: (  # type: ignore
                len(self.agent_registry[aid]["capabilities"]),
                -sum(1 for t in self.active_tasks.values() if aid in t.assigned_agents),
            )
        )

        return suitable

    async def _send_task_to_agent(self, agent_id: str, task: Task) -> None:
        """Send task to a specific agent"""
        task_msg = {
            "type": "task_assignment",
            "task_id": task.id,
            "name": task.name,
            "objective": task.objective,
            "context": task.context,
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "priority": task.priority.value,
            "sender_id": self.id,
        }

        # Send via Kafka if available
        if self.kafka_producer:
            try:
                self.kafka_producer.send(f"agent_{agent_id}_inbox", task_msg)
                self.kafka_producer.flush()
            except Exception as e:
                self.logger.error(f"Failed to send task to agent {agent_id}: {e}")
        else:
            # Fallback to in-memory delivery
            self.logger.debug(f"Sending task {task.id} to agent {agent_id} (in-memory)")

    async def _check_task_timeouts(self) -> None:
        """Check for timed out tasks"""
        now = datetime.now(timezone.utc)
        for task_id, task in list(self.active_tasks.items()):
            if task.status not in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                continue

            # Check deadline
            if task.deadline and now > task.deadline:
                self.logger.warning(f"Task {task_id} deadline exceeded")
                await self._handle_task_timeout(task)

            # Check agent timeout
            elif task.started_at:
                elapsed = (now - task.started_at).total_seconds()
                if elapsed > self.agent_timeout_seconds:
                    self.logger.warning(f"Task {task_id} timed out after {elapsed}s")
                    await self._handle_task_timeout(task)

    async def _handle_task_timeout(self, task: Task) -> None:
        """Handle task timeout"""
        task.status = TaskStatus.FAILED
        task.error = "Task timeout"
        task.completed_at = datetime.now(timezone.utc)
        # Retry if possible
        if task.retry_count < 3:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_agents = []
            task.started_at = None
            task.error = None
            heapq.heappush(self.task_queue, (task.priority.value, task))
            self.logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
        else:
            self.logger.error(f"Task {task.id} failed after {task.retry_count} retries")
            if orchestration_tasks:
                orchestration_tasks.labels(status="failed").inc()

        # Remove from active tasks
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]

    async def _update_workflow_status(self) -> None:
        """Update workflow status based on task completion"""
        for workflow_id, workflow in self.workflows.items():
            if workflow.status != WorkflowStatus.RUNNING:
                continue

            # Check if all tasks are completed
            all_completed = all(
                task.status
                in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                for task in workflow.tasks.values()
            )

            if all_completed:
                # Determine workflow status
                failed_tasks = [
                    t for t in workflow.tasks.values() if t.status == TaskStatus.FAILED
                ]
                cancelled_tasks = [
                    t
                    for t in workflow.tasks.values()
                    if t.status == TaskStatus.CANCELLED
                ]

                if failed_tasks:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.error = f"{len(failed_tasks)} tasks failed"
                elif cancelled_tasks:
                    workflow.status = WorkflowStatus.CANCELLED
                else:
                    workflow.status = WorkflowStatus.COMPLETED

                workflow.completed_at = datetime.now(timezone.utc)
                workflow.updated_at = workflow.completed_at

                # Update metrics
                if active_workflows:
                    active_workflows.dec()

                self.logger.info(
                    f"Workflow {workflow_id} completed with status: {workflow.status}"
                )

            # Check for ready tasks (dependencies satisfied)
            for task in workflow.tasks.values():
                if (
                    task.status == TaskStatus.PENDING
                    and self._are_dependencies_satisfied(task, workflow)
                ):
                    heapq.heappush(self.task_queue, (task.priority.value, task))

    def _are_dependencies_satisfied(self, task: Task, workflow: Workflow) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in workflow.tasks:
                return False
            dep_task = workflow.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    async def _persist_workflow(self, workflow: Workflow) -> None:
        """Persist workflow to Redis"""
        if not self.redis_client:
            return

        try:
            key = f"workflow:{workflow.id}"
            data = {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "started_at": (
                    workflow.started_at.isoformat() if workflow.started_at else None
                ),
                "completed_at": (
                    workflow.completed_at.isoformat() if workflow.completed_at else None
                ),
                "task_count": len(workflow.tasks),
                "tasks": {
                    task_id: {
                        "id": task.id,
                        "name": task.name,
                        "objective": task.objective,
                        "status": task.status.value,
                        "assigned_agents": task.assigned_agents,
                        "created_at": task.created_at.isoformat(),
                        "updated_at": task.updated_at.isoformat(),
                        "started_at": (
                            task.started_at.isoformat() if task.started_at else None
                        ),
                        "completed_at": (
                            task.completed_at.isoformat() if task.completed_at else None
                        ),
                        "retry_count": task.retry_count,
                        "error": task.error,
                        "result": task.result,
                    }
                    for task_id, task in workflow.tasks.items()
                },
            }

            self.redis_client.setex(key, 604800, json.dumps(data))  # 7 days TTL

        except Exception as e:
            self.logger.error(f"Failed to persist workflow {workflow.id}: {e}")

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            return {
                "id": workflow.id,
                "name": workflow.name,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "started_at": (
                    workflow.started_at.isoformat() if workflow.started_at else None
                ),
                "completed_at": (
                    workflow.completed_at.isoformat() if workflow.completed_at else None
                ),
                "task_count": len(workflow.tasks),
                "progress_percentage": (
                    sum(
                        1
                        for task in workflow.tasks.values()
                        if task.status == TaskStatus.COMPLETED
                    )
                    / len(workflow.tasks)
                    * 100
                    if workflow.tasks
                    else 0
                ),
                "tasks": {
                    task_id: {
                        "id": task.id,
                        "name": task.name,
                        "objective": task.objective,
                        "status": task.status.value,
                        "assigned_agents": task.assigned_agents,
                        "created_at": task.created_at.isoformat(),
                        "updated_at": task.updated_at.isoformat(),
                        "deadline": (
                            task.deadline.isoformat() if task.deadline else None
                        ),
                        "retry_count": task.retry_count,
                        "error": task.error,
                        "result": task.result,
                    }
                    for task_id, task in workflow.tasks.items()
                },
            }
        return None

    async def handle_task_update(
        self, task_id: str, status: str, result: Any = None, error: Optional[str] = None
    ) -> None:
        """Handle task status update from agent"""
        if task_id not in self.active_tasks:
            self.logger.warning(f"Received update for unknown task {task_id}")
            return

        task = self.active_tasks[task_id]
        old_status = task.status

        try:
            task.status = TaskStatus(status)
            task.updated_at = datetime.now(timezone.utc)
            if result is not None:
                task.result = result
            if error:
                task.error = error

            if task.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                task.completed_at = datetime.now(timezone.utc)
                del self.active_tasks[task_id]

                if orchestration_tasks:
                    orchestration_tasks.labels(status=status).inc()

            self.logger.info(
                f"Task {task_id} status updated: {old_status.value} -> {status}"
            )

        except ValueError:
            self.logger.error(f"Invalid task status: {status}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "active_workflows": len(
                [
                    w
                    for w in self.workflows.values()
                    if w.status == WorkflowStatus.RUNNING
                ]
            ),
            "total_workflows": len(self.workflows),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "registered_agents": len(self.agent_registry),
            "available_agents": len(
                [a for a in self.agent_registry.values() if a["status"] == "available"]
            ),
        }
