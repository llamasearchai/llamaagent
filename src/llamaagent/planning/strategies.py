"""
Planning Strategies for Dynamic Task Planning

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import random
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List, Optional

from .task_planner import Task, TaskDecomposer, TaskPlan, TaskPriority

logger = logging.getLogger(__name__)


class PlanningStrategy(ABC):
    """Abstract base class for planning strategies."""

    @abstractmethod
    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create a task plan for the given goal."""

    @abstractmethod
    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Check if this strategy can handle the given goal."""


class GoalOrientedPlanning(PlanningStrategy):
    """Goal-oriented planning that works backwards from objectives."""

    def __init__(self, decomposer: Optional[TaskDecomposer] = None):
        self.decomposer = decomposer or TaskDecomposer()

    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create plan by working backwards from goal."""
        context = context or {}

        # Create main goal task
        main_task = Task(
            name=f"Achieve: {goal}",
            description=goal,
            task_type=self._infer_task_type(goal),
            priority=TaskPriority.HIGH,
            estimated_duration=self._estimate_duration(goal),
        )

        # Decompose goal into subgoals
        subgoals = self._identify_subgoals(goal, context)

        plan = TaskPlan(name=f"Goal-Oriented Plan: {goal[:50]}", goal=goal)

        plan.add_task(main_task)

        # Create tasks for each subgoal
        previous_task_id = None
        for i, subgoal in enumerate(subgoals):
            subgoal_task = Task(
                name=f"Subgoal {i + 1}: {subgoal['name']}",
                description=subgoal["description"],
                task_type=subgoal.get("type", "generic"),
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(minutes=subgoal.get("duration", 15)),
            )

            # Create dependencies
            if previous_task_id:
                subgoal_task.add_dependency(previous_task_id)

            plan.add_task(subgoal_task)
            main_task.add_dependency(subgoal_task.id)

            # Decompose complex subgoals
            if subgoal.get("complex", False):
                subtasks = self.decomposer.decompose(subgoal_task, context)
                for subtask in subtasks:
                    plan.add_task(subtask)

            previous_task_id = subgoal_task.id

        return plan

    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Can handle any goal with clear objectives."""
        return len(goal.strip() > 0)

    def _identify_subgoals(
        self, goal: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify subgoals for achieving the main goal."""
        goal_lower = goal.lower()

        if "build" in goal_lower or "create" in goal_lower or "develop" in goal_lower:
            return [
                {
                    "name": "Requirements Analysis",
                    "description": "Analyze and document requirements",
                    "type": "analysis",
                    "duration": 20,
                },
                {
                    "name": "Design Phase",
                    "description": "Create design and architecture",
                    "type": "design",
                    "duration": 30,
                },
                {
                    "name": "Implementation",
                    "description": "Build the solution",
                    "type": "coding",
                    "duration": 60,
                    "complex": True,
                },
                {
                    "name": "Testing",
                    "description": "Test and validate",
                    "type": "testing",
                    "duration": 25,
                },
                {
                    "name": "Documentation",
                    "description": "Create documentation",
                    "type": "documentation",
                    "duration": 15,
                },
            ]

        elif "analyze" in goal_lower or "research" in goal_lower:
            return [
                {
                    "name": "Data Collection",
                    "description": "Gather relevant data",
                    "type": "research",
                    "duration": 30,
                },
                {
                    "name": "Data Processing",
                    "description": "Clean and process data",
                    "type": "analysis",
                    "duration": 25,
                },
                {
                    "name": "Analysis",
                    "description": "Perform analysis",
                    "type": "analysis",
                    "duration": 35,
                    "complex": True,
                },
                {
                    "name": "Report Generation",
                    "description": "Generate findings report",
                    "type": "reporting",
                    "duration": 20,
                },
            ]

        elif "fix" in goal_lower or "debug" in goal_lower or "resolve" in goal_lower:
            return [
                {
                    "name": "Problem Investigation",
                    "description": "Investigate the issue",
                    "type": "analysis",
                    "duration": 20,
                },
                {
                    "name": "Root Cause Analysis",
                    "description": "Find root cause",
                    "type": "analysis",
                    "duration": 15,
                },
                {
                    "name": "Solution Design",
                    "description": "Design solution",
                    "type": "design",
                    "duration": 15,
                },
                {
                    "name": "Implementation",
                    "description": "Implement fix",
                    "type": "coding",
                    "duration": 30,
                },
                {
                    "name": "Verification",
                    "description": "Verify fix works",
                    "type": "testing",
                    "duration": 15,
                },
            ]

        else:
            # Generic subgoals
            return [
                {
                    "name": "Planning",
                    "description": "Plan approach",
                    "type": "planning",
                    "duration": 15,
                },
                {
                    "name": "Execution",
                    "description": "Execute main tasks",
                    "type": "execution",
                    "duration": 45,
                    "complex": True,
                },
                {
                    "name": "Validation",
                    "description": "Validate results",
                    "type": "validation",
                    "duration": 15,
                },
            ]

    def _infer_task_type(self, goal: str) -> str:
        """Infer task type from goal description."""
        goal_lower = goal.lower()

        if any(word in goal_lower for word in ["code", "program", "develop", "build"]):
            return "coding"
        elif any(word in goal_lower for word in ["test", "verify", "validate"]):
            return "testing"
        elif any(word in goal_lower for word in ["analyze", "research", "study"]):
            return "analysis"
        elif any(word in goal_lower for word in ["deploy", "release", "launch"]):
            return "deployment"
        else:
            return "generic"

    def _estimate_duration(self, goal: str) -> timedelta:
        """Estimate duration based on goal complexity."""
        goal_lower = goal.lower()
        complexity_indicators = [
            "complex",
            "comprehensive",
            "advanced",
            "sophisticated",
            "enterprise",
            "production",
            "scalable",
            "distributed",
        ]

        base_duration = 60  # 1 hour

        for indicator in complexity_indicators:
            if indicator in goal_lower:
                base_duration += 30

        return timedelta(minutes=base_duration)


class HierarchicalTaskNetwork(PlanningStrategy):
    """Hierarchical Task Network (HTN) planning approach."""

    def __init__(self):
        self.methods = self._initialize_methods()
        self.operators = self._initialize_operators()

    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create HTN-based plan."""
        context = context or {}

        plan = TaskPlan(name=f"HTN Plan: {goal[:50]}", goal=goal)

        # Start with high-level task
        root_task = Task(
            name=goal,
            description=f"Root task for: {goal}",
            task_type="compound",
            priority=TaskPriority.HIGH,
        )

        # Decompose using HTN methods
        task_network = await self._decompose_task(root_task, context)

        # Add all tasks to plan
        for task in task_network:
            plan.add_task(task)

        return plan

    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Can handle goals that match available methods."""
        return self._find_applicable_method(goal) is not None

    async def _decompose_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a compound task using HTN methods."""
        tasks = []

        # Find applicable method
        method = self._find_applicable_method(task.name)
        if not method:
            # Primitive task - no further decomposition
            return [task]

        # Apply method to create subtasks
        subtasks = method(task, context)
        tasks.extend(subtasks)

        # Recursively decompose compound subtasks
        for subtask in subtasks:
            if subtask.task_type == "compound":
                nested_tasks = await self._decompose_task(subtask, context)
                tasks.extend(nested_tasks)

        return tasks

    def _find_applicable_method(self, task_name: str) -> Optional[Any]:
        """Find method applicable to the task."""
        task_lower = task_name.lower()

        for pattern, method in self.methods.items():
            if pattern in task_lower:
                return method

        return None

    def _initialize_methods(self) -> Dict[str, Any]:
        """Initialize HTN methods."""
        return {
            "develop": self._method_develop_software,
            "test": self._method_test_software,
            "analyze": self._method_analyze_data,
            "deploy": self._method_deploy_system,
            "research": self._method_conduct_research,
        }

    def _initialize_operators(self) -> Dict[str, Any]:
        """Initialize primitive operators."""
        return {
            "write_code": self._operator_write_code,
            "run_tests": self._operator_run_tests,
            "collect_data": self._operator_collect_data,
            "generate_report": self._operator_generate_report,
        }

    def _method_develop_software(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Method for software development."""
        return [
            Task(
                name="Design architecture",
                task_type="primitive",
                priority=TaskPriority.HIGH,
            ),
            Task(name="Write code", task_type="primitive", priority=TaskPriority.HIGH),
            Task(
                name="Review code", task_type="primitive", priority=TaskPriority.MEDIUM
            ),
            Task(name="Test code", task_type="compound", priority=TaskPriority.HIGH),
        ]

    def _method_test_software(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Method for software testing."""
        return [
            Task(
                name="Write unit tests",
                task_type="primitive",
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Run unit tests", task_type="primitive", priority=TaskPriority.HIGH
            ),
            Task(
                name="Write integration tests",
                task_type="primitive",
                priority=TaskPriority.MEDIUM,
            ),
            Task(
                name="Run integration tests",
                task_type="primitive",
                priority=TaskPriority.MEDIUM,
            ),
        ]

    def _method_analyze_data(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Method for data analysis."""
        return [
            Task(
                name="Collect data", task_type="primitive", priority=TaskPriority.HIGH
            ),
            Task(name="Clean data", task_type="primitive", priority=TaskPriority.HIGH),
            Task(
                name="Analyze data", task_type="primitive", priority=TaskPriority.HIGH
            ),
            Task(
                name="Generate report",
                task_type="primitive",
                priority=TaskPriority.MEDIUM,
            ),
        ]

    def _method_deploy_system(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Method for system deployment."""
        return [
            Task(
                name="Prepare deployment",
                task_type="primitive",
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Deploy to staging",
                task_type="primitive",
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Test in staging", task_type="compound", priority=TaskPriority.HIGH
            ),
            Task(
                name="Deploy to production",
                task_type="primitive",
                priority=TaskPriority.CRITICAL,
            ),
        ]

    def _method_conduct_research(
        self, task: Task, context: Dict[str, Any]
    ) -> List[Task]:
        """Method for conducting research."""
        return [
            Task(
                name="Literature review",
                task_type="primitive",
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Collect data", task_type="primitive", priority=TaskPriority.HIGH
            ),
            Task(
                name="Analyze findings",
                task_type="compound",
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Write research paper",
                task_type="primitive",
                priority=TaskPriority.MEDIUM,
            ),
        ]

    def _operator_write_code(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Primitive operator for writing code."""
        return {"action": "write_code", "result": "Code written successfully"}

    def _operator_run_tests(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Primitive operator for running tests."""
        return {"action": "run_tests", "result": "Tests executed"}

    def _operator_collect_data(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Primitive operator for collecting data."""
        return {"action": "collect_data", "result": "Data collected"}

    def _operator_generate_report(
        self, task: Task, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Primitive operator for generating reports."""
        return {"action": "generate_report", "result": "Report generated"}


class ConstraintBasedPlanning(PlanningStrategy):
    """Constraint-based planning with complex constraints."""

    def __init__(self):
        self.constraints = []
        self.constraint_solvers = {
            "resource": self._solve_resource_constraints,
            "time": self._solve_time_constraints,
            "dependency": self._solve_dependency_constraints,
            "precedence": self._solve_precedence_constraints,
        }

    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create constraint-based plan."""
        context = context or {}
        constraints = context.get("constraints", [])

        # Create initial plan
        plan = TaskPlan(
            name=f"Constraint-Based Plan: {goal[:50]}",
            goal=goal,
            constraints=constraints,
        )

        # Generate initial tasks
        initial_tasks = self._generate_initial_tasks(goal, context)
        for task in initial_tasks:
            plan.add_task(task)

        # Apply constraint solving
        await self._solve_constraints(plan, constraints)

        return plan

    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Can handle goals with explicit constraints."""
        context = context or {}
        return "constraints" in context and len(context["constraints"]) > 0

    def _generate_initial_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate initial task set before constraint solving."""
        # This would typically be more sophisticated
        return [
            Task(
                name=f"Task for: {goal}",
                description=goal,
                task_type="generic",
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(minutes=30),
            )
        ]

    async def _solve_constraints(
        self, plan: TaskPlan, constraints: List[Dict[str, Any]]
    ):
        """Apply constraint solving to the plan."""
        for constraint in constraints:
            constraint_type = constraint.get("type")
            solver = self.constraint_solvers.get(constraint_type)

            if solver:
                await solver(plan, constraint)
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")

    async def _solve_resource_constraints(
        self, plan: TaskPlan, constraint: Dict[str, Any]
    ):
        """Solve resource allocation constraints."""
        resource_type = constraint.get("resource_type")
        limit = constraint.get("limit", float("inf"))

        # Adjust task scheduling to respect resource limits
        for task in plan.tasks.values():
            for resource in task.resources:
                if (
                    resource.resource_type == resource_type
                    and resource.quantity > limit
                ):
                    # Split task or reduce resource requirement
                    resource.quantity = min(resource.quantity, limit)

    async def _solve_time_constraints(self, plan: TaskPlan, constraint: Dict[str, Any]):
        """Solve time-based constraints."""
        deadline = constraint.get("deadline")
        if deadline:
            # Adjust task priorities and durations to meet deadline
            for task in plan.tasks.values():
                task.priority = TaskPriority.HIGH
                task.estimated_duration = (
                    task.estimated_duration * 0.8
                )  # Compress by 20%

    async def _solve_dependency_constraints(
        self, plan: TaskPlan, constraint: Dict[str, Any]
    ):
        """Solve dependency constraints."""
        task_a = constraint.get("task_a")
        task_b = constraint.get("task_b")
        dependency_type = constraint.get("dependency_type", "finish_to_start")

        if task_a in plan.tasks and task_b in plan.tasks:
            plan.tasks[task_b].add_dependency(task_a, dependency_type=dependency_type)

    async def _solve_precedence_constraints(
        self, plan: TaskPlan, constraint: Dict[str, Any]
    ):
        """Solve precedence constraints."""
        ordered_tasks = constraint.get("ordered_tasks", [])

        for i in range(1, len(ordered_tasks)):
            if ordered_tasks[i] in plan.tasks and ordered_tasks[i - 1] in plan.tasks:
                plan.tasks[ordered_tasks[i]].add_dependency(ordered_tasks[i - 1])


class ReactivePlanning(PlanningStrategy):
    """Reactive planning that adapts to changing conditions."""

    def __init__(self):
        self.triggers = {}
        self.adaptation_rules = {}

    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create reactive plan with adaptation capabilities."""
        context = context or {}

        plan = TaskPlan(name=f"Reactive Plan: {goal[:50]}", goal=goal)

        # Create basic tasks
        basic_tasks = self._create_basic_tasks(goal, context)
        for task in basic_tasks:
            plan.add_task(task)

        # Add reactive triggers
        self._setup_reactive_triggers(plan, context)

        return plan

    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Can handle goals in dynamic environments."""
        context = context or {}
        return context.get("dynamic_environment", False)

    def _create_basic_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Create basic task structure."""
        return [
            Task(
                name="Monitor environment",
                description="Monitor for changes",
                task_type="monitoring",
                priority=TaskPriority.HIGH,
                estimated_duration=timedelta(minutes=5),
            ),
            Task(
                name=f"Execute: {goal}",
                description=goal,
                task_type="execution",
                priority=TaskPriority.MEDIUM,
                estimated_duration=timedelta(minutes=30),
            ),
            Task(
                name="Adapt to changes",
                description="Adapt plan based on observations",
                task_type="adaptation",
                priority=TaskPriority.LOW,
                estimated_duration=timedelta(minutes=10),
            ),
        ]

    def _setup_reactive_triggers(self, plan: TaskPlan, context: Dict[str, Any]):
        """Setup triggers for reactive adaptation."""
        plan.metadata["reactive_triggers"] = {
            "failure_rate_threshold": 0.3,
            "time_overrun_threshold": 1.5,
            "resource_exhaustion_threshold": 0.9,
            "external_event_types": [
                "priority_change",
                "resource_change",
                "requirement_change",
            ],
        }


class ProbabilisticPlanning(PlanningStrategy):
    """Probabilistic planning that handles uncertainty."""

    def __init__(self):
        self.uncertainty_models = {}
        self.risk_factors = {}

    async def plan(self, goal: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Create probabilistic plan considering uncertainty."""
        context = context or {}

        plan = TaskPlan(name=f"Probabilistic Plan: {goal[:50]}", goal=goal)

        # Generate tasks with uncertainty estimates
        tasks_with_uncertainty = self._generate_uncertain_tasks(goal, context)

        for task_info in tasks_with_uncertainty:
            task = task_info["task"]
            uncertainty = task_info["uncertainty"]

            # Adjust task based on uncertainty
            self._apply_uncertainty(task, uncertainty)
            plan.add_task(task)

        # Add contingency tasks
        contingency_tasks = self._create_contingency_tasks(plan, context)
        for task in contingency_tasks:
            plan.add_task(task)

        return plan

    def can_handle(self, goal: str, context: Dict[str, Any] = None) -> bool:
        """Can handle goals with uncertainty information."""
        context = context or {}
        return "uncertainty" in context or "risk_factors" in context

    def _generate_uncertain_tasks(
        self, goal: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate tasks with uncertainty estimates."""
        base_task = Task(
            name=f"Primary: {goal}",
            description=goal,
            task_type="primary",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(minutes=45),
        )

        return [
            {
                "task": base_task,
                "uncertainty": {
                    "duration_variance": 0.3,  # ±30% duration uncertainty
                    "success_probability": 0.8,  # 80% chance of success
                    "resource_variance": 0.2,  # ±20% resource uncertainty
                },
            }
        ]

    def _apply_uncertainty(self, task: Task, uncertainty: Dict[str, Any]):
        """Apply uncertainty factors to task."""
        # Adjust duration based on variance
        duration_variance = uncertainty.get("duration_variance", 0)
        variance_factor = 1 + random.uniform(-duration_variance, duration_variance)
        task.estimated_duration = task.estimated_duration * variance_factor

        # Store uncertainty metadata
        task.metadata.update(
            {
                "success_probability": uncertainty.get("success_probability", 1.0),
                "duration_variance": duration_variance,
                "resource_variance": uncertainty.get("resource_variance", 0),
            }
        )

    def _create_contingency_tasks(
        self, plan: TaskPlan, context: Dict[str, Any]
    ) -> List[Task]:
        """Create contingency tasks for handling failures."""
        contingencies = []

        for task in plan.tasks.values():
            success_prob = task.metadata.get("success_probability", 1.0)

            if success_prob < 0.9:  # Create contingency for risky tasks
                contingency = Task(
                    name=f"Contingency for {task.name}",
                    description=f"Backup plan if {task.name} fails",
                    task_type="contingency",
                    priority=TaskPriority.LOW,
                    estimated_duration=task.estimated_duration * 0.5,
                )
                contingencies.append(contingency)

        return contingencies
