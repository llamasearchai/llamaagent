"""
Task Planning Components for Dynamic Task Decomposition and Management

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    DEFERRED = 1


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""

    task_id: str
    dependency_type: str = "finish_to_start"  # finish_to_start, start_to_start, etc.
    lag_time: timedelta = field(default_factory=timedelta)
    is_critical: bool = True

    def __hash__(self):
        return hash(self.task_id)


@dataclass
class TaskResource:
    """Resource requirements for a task."""

    resource_type: str
    quantity: float
    is_exclusive: bool = False
    priority: int = 0


@dataclass
class Task:
    """Represents a single task in a plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "generic"
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: Set[TaskDependency] = field(default_factory=set)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    actual_duration: Optional[timedelta] = None
    resources: List[TaskResource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    subtasks: List["Task"] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now())
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[Any] = None

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute based on dependencies."""
        if self.status != TaskStatus.PENDING:
            return False

        for dep in self.dependencies:
            if dep.task_id not in completed_tasks:
                return False
        return True

    def add_dependency(self, task_id: str, **kwargs: Any):
        """Add a dependency to this task."""
        self.dependencies.add(TaskDependency(task_id=task_id, **kwargs))

    def add_subtask(self, subtask: "Task"):
        """Add a subtask."""
        subtask.parent_id = self.id
        self.subtasks.append(subtask)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": [d.task_id for d in self.dependencies],
            "estimated_duration": self.estimated_duration.total_seconds(),
            "subtasks": [t.to_dict() for t in self.subtasks],
            "metadata": self.metadata,
        }


@dataclass
class TaskPlan:
    """Represents a complete task execution plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    goal: str = ""
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now())
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def add_task(self, task: Task):
        """Add a task to the plan."""
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """Get all tasks ready for execution."""
        ready: List[Task] = []
        for task in self.tasks.values():
            if task.is_ready(completed_tasks):
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority.value, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "constraints": self.constraints,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class DependencyResolver:
    """Resolves task dependencies and detects cycles."""

    @staticmethod
    def topological_sort(tasks: Dict[str, Task]) -> List[str]:
        """Perform topological sort on tasks."""
        # Build adjacency list
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = defaultdict(int)

        for task_id, task in tasks.items():
            if task_id not in in_degree:
                in_degree[task_id] = 0
            for dep in task.dependencies:
                graph[dep.task_id].append(task_id)
                in_degree[task_id] += 1

        # Find all nodes with no incoming edges
        queue: deque[str] = deque([node for node in tasks if in_degree[node] == 0])
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree for neighbors
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(tasks):
            raise ValueError("Circular dependency detected in task plan")

        return result

    @staticmethod
    def find_critical_path(tasks: Dict[str, Task]) -> Tuple[List[str], timedelta]:
        """Find the critical path through the task network."""
        # Initialize earliest start times
        earliest_start: Dict[str, timedelta] = {}
        earliest_finish: Dict[str, timedelta] = {}
        sorted_tasks = DependencyResolver.topological_sort(tasks)

        # Forward pass
        for task_id in sorted_tasks:
            task = tasks[task_id]
            if not task.dependencies:
                earliest_start[task_id] = timedelta()
            else:
                max_finish = max(
                    earliest_finish.get(dep.task_id, timedelta()) + dep.lag_time
                    for dep in task.dependencies
                )
                earliest_start[task_id] = max_finish
            earliest_finish[task_id] = earliest_start[task_id] + task.estimated_duration

        # Backward pass
        latest_finish: Dict[str, timedelta] = {}
        latest_start: Dict[str, timedelta] = {}
        project_duration = (
            max(earliest_finish.values()) if earliest_finish else timedelta()
        )

        for task_id in reversed(sorted_tasks):
            task = tasks[task_id]

            # Find tasks that depend on this one
            dependents = [
                t
                for t in tasks.values()
                if any(dep.task_id == task_id for dep in t.dependencies)
            ]

            if not dependents:
                latest_finish[task_id] = project_duration
            else:
                min_start = min(
                    latest_start.get(t.id, project_duration)
                    - next(d.lag_time for d in t.dependencies if d.task_id == task_id)
                    for t in dependents
                )
                latest_finish[task_id] = min_start
            latest_start[task_id] = latest_finish[task_id] - task.estimated_duration

        # Determine critical path
        critical_path: List[str] = []
        for task_id in sorted_tasks:
            if latest_start[task_id] == earliest_start[task_id]:
                critical_path.append(task_id)

        return critical_path, project_duration


class PlanValidator:
    """Validates task plans for feasibility."""

    @staticmethod
    def validate(plan: TaskPlan) -> Tuple[bool, List[str]]:
        """Validate a task plan."""
        errors: List[str] = []

        # Check for empty plan
        if not plan.tasks:
            errors.append("Plan contains no tasks")
            return False, errors

        # Check for circular dependencies
        try:
            DependencyResolver.topological_sort(plan.tasks)
        except ValueError as e:
            errors.append(str(e))

        # Check task validity
        task_ids = set(plan.tasks.keys())
        for task_id, task in plan.tasks.items():
            # Check dependencies exist
            for dep in task.dependencies:
                if dep.task_id not in task_ids:
                    errors.append(
                        f"Task {task_id} depends on non-existent task {dep.task_id}"
                    )

            # Check resource requirements
            if task.estimated_duration <= timedelta():
                errors.append(f"Task {task_id} has invalid duration")

        # Validate constraints
        for constraint in plan.constraints:
            if not PlanValidator._validate_constraint(plan, constraint):
                errors.append(f"Constraint violation: {constraint}")

        plan.is_valid = len(errors) == 0
        plan.validation_errors = errors

        return plan.is_valid, errors

    @staticmethod
    def _validate_constraint(plan: TaskPlan, constraint: Dict[str, Any]) -> bool:
        """Validate a single constraint."""
        constraint_type = constraint.get("type")

        if constraint_type == "max_duration":
            _, duration = DependencyResolver.find_critical_path(plan.tasks)
            return duration <= timedelta(seconds=constraint.get("value", 0))

        elif constraint_type == "resource_limit":
            # Check resource usage doesn't exceed limits
            resource_type = constraint.get("resource_type")
            limit = constraint.get("limit", 0)

            # Simple check - more sophisticated scheduling needed for accurate validation
            max_usage = sum(
                r.quantity
                for task in plan.tasks.values()
                for r in task.resources
                if r.resource_type == resource_type
            )
            return max_usage <= limit

        return True


class TaskDecomposer:
    """Decomposes complex tasks into smaller, manageable subtasks."""

    def __init__(self):
        """Initialize the decomposer."""
        # This can be expanded to include models or configurations for decomposition
        pass

    def decompose(
        self, task: Task, context: Optional[Dict[str, Any]] = None
    ) -> List[Task]:
        """Decompose a task based on its type."""
        if context is None:
            context = {}

        decomposition_map = {
            "coding": self._decompose_coding_task,
            "research": self._decompose_research_task,
            "analysis": self._decompose_analysis_task,
            "testing": self._decompose_testing_task,
            "deployment": self._decompose_deployment_task,
            "generic": self._decompose_generic_task,
        }

        handler = decomposition_map.get(task.task_type, self._decompose_generic_task)
        return handler(task, context)

    def _decompose_coding_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a coding task."""
        logger.info(f"Decomposing coding task: {task.name}")
        steps: List[Task] = []

        # 1. Understand requirements
        steps.append(
            Task(
                name=f"Clarify requirements for '{task.name}'",
                description="Review and clarify all technical and functional requirements.",
                priority=TaskPriority.HIGH,
            )
        )

        # 2. Design solution
        steps.append(
            Task(
                name=f"Design solution for '{task.name}'",
                description="Create a technical design, including class diagrams and data models.",
                priority=TaskPriority.HIGH,
            )
        )

        # 3. Implement feature
        steps.append(
            Task(
                name=f"Implement '{task.name}'",
                description="Write the code for the feature, following the design.",
                priority=TaskPriority.MEDIUM,
            )
        )

        # 4. Write unit tests
        steps.append(
            Task(
                name=f"Write unit tests for '{task.name}'",
                description="Develop comprehensive unit tests to ensure code quality.",
                priority=TaskPriority.MEDIUM,
            )
        )
        return steps

    def _decompose_research_task(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Decompose a research task."""
        logger.info(f"Decomposing research task: {task.name}")
        steps: List[Task] = []

        steps.append(
            Task(
                name=f"Define research scope for '{task.name}'",
                description="Clearly define the research questions and scope.",
                priority=TaskPriority.HIGH,
            )
        )
        steps.append(
            Task(
                name=f"Gather information for '{task.name}'",
                description="Collect data and literature from reliable sources.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Analyze findings for '{task.name}'",
                description="Synthesize and analyze the collected information.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Summarize report for '{task.name}'",
                description="Write a comprehensive report summarizing the findings.",
                priority=TaskPriority.LOW,
            )
        )
        return steps

    def _decompose_analysis_task(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Decompose a data analysis task."""
        logger.info(f"Decomposing analysis task: {task.name}")
        steps: List[Task] = []

        steps.append(
            Task(
                name=f"Define analysis goals for '{task.name}'",
                description="Set clear objectives for the data analysis.",
                priority=TaskPriority.HIGH,
            )
        )
        steps.append(
            Task(
                name=f"Prepare data for '{task.name}'",
                description="Clean, preprocess, and prepare the dataset.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Perform analysis for '{task.name}'",
                description="Apply statistical methods and models to the data.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Visualize results for '{task.name}'",
                description="Create charts and visualizations to present findings.",
                priority=TaskPriority.LOW,
            )
        )
        return steps

    def _decompose_testing_task(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Decompose a testing task."""
        logger.info(f"Decomposing testing task: {task.name}")
        steps: List[Task] = []

        steps.append(
            Task(
                name=f"Create test plan for '{task.name}'",
                description="Develop a detailed test plan, including scope and strategy.",
                priority=TaskPriority.HIGH,
            )
        )
        steps.append(
            Task(
                name=f"Write test cases for '{task.name}'",
                description="Write specific test cases covering all requirements.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Execute tests for '{task.name}'",
                description="Run all test cases and document the results.",
                priority=TaskPriority.MEDIUM,
            )
        )
        steps.append(
            Task(
                name=f"Report defects for '{task.name}'",
                description="Log and report any defects found during testing.",
                priority=TaskPriority.MEDIUM,
            )
        )
        return steps

    def _decompose_deployment_task(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Decompose a deployment task."""
        logger.info(f"Decomposing deployment task: {task.name}")
        steps: List[Task] = []

        steps.append(
            Task(
                name=f"Prepare deployment environment for '{task.name}'",
                description="Configure servers and infrastructure for deployment.",
                priority=TaskPriority.CRITICAL,
            )
        )
        steps.append(
            Task(
                name=f"Deploy application for '{task.name}'",
                description="Deploy the application to the target environment.",
                priority=TaskPriority.CRITICAL,
            )
        )
        steps.append(
            Task(
                name=f"Verify deployment of '{task.name}'",
                description="Perform post-deployment checks to ensure success.",
                priority=TaskPriority.HIGH,
            )
        )
        steps.append(
            Task(
                name=f"Monitor application after '{task.name}'",
                description="Monitor application health and performance post-deployment.",
                priority=TaskPriority.HIGH,
            )
        )
        return steps

    def _decompose_generic_task(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Decompose a generic task into simple steps."""
        logger.info(f"Decomposing generic task: {task.name}")
        subtasks: List[Task] = []

        # Simple decomposition for generic tasks
        subtasks.append(
            Task(
                name=f"Subtask 1 for {task.name}",
                description=f"First step for completing {task.name}",
                priority=task.priority,
            )
        )
        subtasks.append(
            Task(
                name=f"Subtask 2 for {task.name}",
                description=f"Second step for completing {task.name}",
                priority=task.priority,
            )
        )
        return subtasks


class TaskPlanner:
    """Creates and manages task plans."""

    def __init__(self, decomposer: Optional[TaskDecomposer] = None):
        """Initialize the task planner."""
        self.decomposer = decomposer or TaskDecomposer()
        self.validator = PlanValidator()
        self.resolver = DependencyResolver()
        self.plans: Dict[str, TaskPlan] = {}

    def create_plan(
        self,
        goal: str,
        initial_tasks: Optional[List[Task]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        auto_decompose: bool = True,
    ) -> TaskPlan:
        """Create a new task plan."""
        plan = TaskPlan(
            name=f"Plan for '{goal}'",
            goal=goal,
            constraints=constraints or [],
        )

        tasks_to_process = initial_tasks or [Task(name=goal, description=goal)]

        for task in tasks_to_process:
            if auto_decompose and not task.subtasks:
                decomposed_subtasks = self.decomposer.decompose(task)
                for i, subtask in enumerate(decomposed_subtasks):
                    task.add_subtask(subtask)
                    if i > 0:
                        subtask.add_dependency(decomposed_subtasks[i - 1].id)
            plan.add_task(task)

        # Validate the plan
        is_valid, errors = PlanValidator.validate(plan)
        plan.is_valid = is_valid
        plan.validation_errors = errors

        return plan

    def optimize_plan(self, plan: TaskPlan) -> TaskPlan:
        """Optimize the task plan (placeholder)."""
        # This can be expanded with different optimization strategies
        logger.info(f"Optimizing plan: {plan.name}")
        return plan

    def get_execution_order(self, plan: TaskPlan) -> List[List[Task]]:
        """Get the execution order of tasks in parallelizable groups."""
        if not plan.is_valid:
            raise ValueError("Cannot get execution order for an invalid plan.")

        DependencyResolver.topological_sort(plan.tasks)

        completed: Set[str] = set()
        execution_levels: List[List[Task]] = []

        while len(completed) < len(plan.tasks):
            level_tasks: List[Task] = []

            # Find all tasks that are ready to run
            ready_tasks = plan.get_ready_tasks(completed)

            if not ready_tasks and len(completed) < len(plan.tasks):
                raise ValueError(
                    "Could not determine execution order; possible deadlock or missing dependency."
                )

            for task in ready_tasks:
                level_tasks.append(task)
                completed.add(task.id)

            if level_tasks:
                execution_levels.append(level_tasks)

        return execution_levels

    def replan(
        self, plan: TaskPlan, completed_tasks: Set[str], failed_tasks: Set[str]
    ) -> TaskPlan:
        """Create a new plan based on execution results."""
        # Create new plan with remaining tasks
        new_plan = TaskPlan(
            name=f"Replan: {plan.name}",
            goal=plan.goal,
            constraints=plan.constraints.copy(),
        )

        # Add uncompleted tasks
        for task_id, task in plan.tasks.items():
            if task_id not in completed_tasks:
                # Clone task
                new_task = Task(
                    name=task.name,
                    description=task.description,
                    task_type=task.task_type,
                    priority=task.priority,
                    estimated_duration=task.estimated_duration,
                    resources=task.resources.copy(),
                    metadata=task.metadata.copy(),
                )

                # Update dependencies - remove completed ones
                for dep in task.dependencies:
                    if dep.task_id not in completed_tasks:
                        new_task.add_dependency(dep.task_id)

                # Increase priority for previously failed tasks
                if task_id in failed_tasks:
                    new_task.priority = TaskPriority(
                        min(new_task.priority.value + 1, TaskPriority.CRITICAL.value)
                    )
                    new_task.metadata["retry_attempt"] = task.retry_count + 1

                new_plan.add_task(new_task)

        # Validate and optimize new plan
        self.validator.validate(new_plan)
        self.optimize_plan(new_plan)

        return new_plan
