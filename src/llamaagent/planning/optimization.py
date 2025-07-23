"""
Optimization Components for Task Planning

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Set, Tuple

from .task_planner import DependencyResolver, Task, TaskPlan, TaskPriority

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives."""

    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_RISK = "minimize_risk"
    BALANCE_ALL = "balance_all"


@dataclass
class OptimizationConstraint:
    """Represents an optimization constraint."""

    type: str  # "time", "cost", "resource", "quality"
    operator: str  # "<=", ">=", "=="
    value: float
    weight: float = 1.0
    description: str = ""


@dataclass
class OptimizationResult:
    """Result of optimization process."""

    original_plan: TaskPlan
    optimized_plan: TaskPlan
    objective_value: float
    improvement_percentage: float
    optimization_time: timedelta
    constraints_satisfied: bool
    violated_constraints: List[str] = field(default_factory=list)
    optimization_details: Dict[str, Any] = field(default_factory=dict)


class PlanOptimizer:
    """Main plan optimizer that coordinates different optimization strategies."""

    def __init__(self):
        self.optimizers = {
            OptimizationObjective.MINIMIZE_TIME: TimeOptimizer(),
            OptimizationObjective.MINIMIZE_COST: CostOptimizer(),
            OptimizationObjective.MAXIMIZE_QUALITY: QualityOptimizer(),
            OptimizationObjective.MINIMIZE_RISK: RiskOptimizer(),
        }
        self.resource_allocator = ResourceAllocator()

    async def optimize(
        self,
        plan: TaskPlan,
        objective: OptimizationObjective = OptimizationObjective.MINIMIZE_TIME,
        constraints: List[OptimizationConstraint] = None,
        context: Dict[str, Any] = None,
    ) -> OptimizationResult:
        """Optimize a task plan according to the specified objective."""
        start_time = datetime.now()
        constraints = constraints or []
        context = context or {}

        logger.info(f"Starting optimization with objective: {objective.value}")

        # Calculate baseline metrics
        baseline_metrics = self._calculate_plan_metrics(plan)

        # Choose optimizer based on objective
        if objective == OptimizationObjective.BALANCE_ALL:
            optimized_plan = await self._multi_objective_optimization(
                plan, constraints, context
            )
        else:
            optimizer = self.optimizers.get(objective)
            if not optimizer:
                raise ValueError(f"Unknown optimization objective: {objective}")
            optimized_plan = await optimizer.optimize(plan, constraints, context)

        # Calculate optimized metrics
        optimized_metrics = self._calculate_plan_metrics(optimized_plan)

        # Validate constraints
        constraints_satisfied, violated = self._validate_constraints(
            optimized_plan, constraints
        )

        # Calculate improvement
        improvement = self._calculate_improvement(
            baseline_metrics, optimized_metrics, objective
        )

        optimization_time = datetime.now() - start_time

        result = OptimizationResult(
            original_plan=plan,
            optimized_plan=optimized_plan,
            objective_value=optimized_metrics.get(objective.value, 0),
            improvement_percentage=improvement,
            optimization_time=optimization_time,
            constraints_satisfied=constraints_satisfied,
            violated_constraints=violated,
            optimization_details={
                "baseline_metrics": baseline_metrics,
                "optimized_metrics": optimized_metrics,
                "optimization_steps": [],
            },
        )

        logger.info(
            f"Optimization completed in {optimization_time}. Improvement: {improvement:.2f}%"
        )
        return result

    async def _multi_objective_optimization(
        self,
        plan: TaskPlan,
        constraints: List[OptimizationConstraint],
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Perform multi-objective optimization using weighted sum approach."""
        # Define weights for each objective
        weights = context.get(
            "objective_weights",
            {
                OptimizationObjective.MINIMIZE_TIME: 0.3,
                OptimizationObjective.MINIMIZE_COST: 0.3,
                OptimizationObjective.MAXIMIZE_QUALITY: 0.2,
                OptimizationObjective.MINIMIZE_RISK: 0.2,
            },
        )

        best_plan = plan
        best_score = float("inf")

        # Try each single-objective optimization
        for objective, weight in weights.items():
            if weight > 0:
                optimizer = self.optimizers.get(objective)
                if optimizer:
                    candidate_plan = await optimizer.optimize(
                        plan, constraints, context
                    )
                    score = self._calculate_multi_objective_score(
                        candidate_plan, weights
                    )

                    if score < best_score:
                        best_score = score
                        best_plan = candidate_plan

        return best_plan

    def _calculate_multi_objective_score(
        self, plan: TaskPlan, weights: Dict[OptimizationObjective, float]
    ) -> float:
        """Calculate weighted score for multi-objective optimization."""
        metrics = self._calculate_plan_metrics(plan)

        score = 0.0
        for objective, weight in weights.items():
            if objective == OptimizationObjective.MINIMIZE_TIME:
                score += weight * metrics.get("total_duration", 0)
            elif objective == OptimizationObjective.MINIMIZE_COST:
                score += weight * metrics.get("total_cost", 0)
            elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
                score += weight * (1.0 - metrics.get("quality_score", 0.5))
            elif objective == OptimizationObjective.MINIMIZE_RISK:
                score += weight * metrics.get("risk_score", 0.5)

        return score

    def _calculate_plan_metrics(self, plan: TaskPlan) -> Dict[str, float]:
        """Calculate comprehensive metrics for a plan."""
        total_duration = 0
        total_cost = 0
        quality_scores = []
        risk_scores = []

        for task in plan.tasks.values():
            total_duration += task.estimated_duration.total_seconds()
            total_cost += task.metadata.get("cost", 0)
            quality_scores.append(task.metadata.get("quality_score", 0.7))
            risk_scores.append(task.metadata.get("risk_score", 0.3))

        # Calculate critical path duration
        try:
            _, critical_path_duration = DependencyResolver.find_critical_path(
                plan.tasks
            )
            critical_path_seconds = critical_path_duration.total_seconds()
        except:
            critical_path_seconds = total_duration

        return {
            "total_duration": total_duration,
            "critical_path_duration": critical_path_seconds,
            "total_cost": total_cost,
            "quality_score": (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            ),
            "risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.5,
            "task_count": len(plan.tasks),
        }

    def _calculate_improvement(
        self,
        baseline: Dict[str, float],
        optimized: Dict[str, float],
        objective: OptimizationObjective,
    ) -> float:
        """Calculate improvement percentage based on objective."""
        if objective == OptimizationObjective.MINIMIZE_TIME:
            metric = "critical_path_duration"
        elif objective == OptimizationObjective.MINIMIZE_COST:
            metric = "total_cost"
        elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
            metric = "quality_score"
        elif objective == OptimizationObjective.MINIMIZE_RISK:
            metric = "risk_score"
        else:
            return 0.0

        baseline_value = baseline.get(metric, 0)
        optimized_value = optimized.get(metric, 0)

        if baseline_value == 0:
            return 0.0

        if objective == OptimizationObjective.MAXIMIZE_QUALITY:
            return (optimized_value - baseline_value) / baseline_value * 100
        else:
            return (baseline_value - optimized_value) / baseline_value * 100

    def _validate_constraints(
        self, plan: TaskPlan, constraints: List[OptimizationConstraint]
    ) -> Tuple[bool, List[str]]:
        """Validate that plan satisfies constraints."""
        violations = []
        metrics = self._calculate_plan_metrics(plan)

        for constraint in constraints:
            value = metrics.get(constraint.type, 0)

            if constraint.operator == "<=" and value > constraint.value:
                violations.append(f"{constraint.type} {value} > {constraint.value}")
            elif constraint.operator == ">=" and value < constraint.value:
                violations.append(f"{constraint.type} {value} < {constraint.value}")
            elif constraint.operator == "==" and abs(value - constraint.value) > 0.01:
                violations.append(f"{constraint.type} {value} != {constraint.value}")

        return len(violations) == 0, violations


class TimeOptimizer:
    """Optimizes plans to minimize execution time."""

    async def optimize(
        self,
        plan: TaskPlan,
        constraints: List[OptimizationConstraint] = None,
        context: Dict[str, Any] = None,
    ) -> TaskPlan:
        """Optimize plan to minimize execution time."""
        optimized_plan = self._deep_copy_plan(plan)

        # Apply time optimization strategies
        await self._optimize_critical_path(optimized_plan)
        await self._optimize_parallelization(optimized_plan)
        await self._optimize_task_durations(optimized_plan)

        return optimized_plan

    async def _optimize_critical_path(self, plan: TaskPlan):
        """Optimize tasks on the critical path."""
        try:
            critical_path, _ = DependencyResolver.find_critical_path(plan.tasks)

            for task_id in critical_path:
                task = plan.tasks.get(task_id)
                if task:
                    # Reduce duration of critical path tasks
                    task.estimated_duration = task.estimated_duration * 0.8
                    task.priority = TaskPriority.CRITICAL
        except Exception as e:
            logger.warning(f"Critical path optimization failed: {e}")

    async def _optimize_parallelization(self, plan: TaskPlan):
        """Identify opportunities for task parallelization."""
        # Find tasks that can be parallelized
        independent_groups = self._find_independent_task_groups(plan)

        for group in independent_groups:
            if len(group) > 1:
                # Mark tasks as parallelizable
                for task_id in group:
                    task = plan.tasks.get(task_id)
                    if task:
                        task.metadata["parallelizable"] = True
                        task.metadata["parallel_group"] = str(
                            hash(tuple(sorted(group)))
                        )

    async def _optimize_task_durations(self, plan: TaskPlan):
        """Optimize individual task durations."""
        for task in plan.tasks.values():
            # Apply duration optimization heuristics
            task_type = task.task_type

            if task_type == "testing":
                # Tests can often be optimized
                task.estimated_duration = task.estimated_duration * 0.7
            elif task_type == "documentation":
                # Documentation can be streamlined
                task.estimated_duration = task.estimated_duration * 0.6
            elif task_type == "review":
                # Reviews can be made more efficient
                task.estimated_duration = task.estimated_duration * 0.8

    def _find_independent_task_groups(self, plan: TaskPlan) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel."""
        groups = []
        visited = set()

        for task_id in plan.tasks:
            if task_id not in visited:
                group = self._find_connected_component(plan, task_id, visited)
                if len(group) > 1:
                    groups.append(group)

        return groups

    def _find_connected_component(
        self, plan: TaskPlan, start_task: str, visited: Set[str]
    ) -> List[str]:
        """Find connected component of tasks."""
        component = []
        stack = [start_task]

        while stack:
            task_id = stack.pop()
            if task_id not in visited:
                visited.add(task_id)
                component.append(task_id)

                task = plan.tasks.get(task_id)
                if task:
                    # Add dependent tasks
                    for dep in task.dependencies:
                        if dep.task_id not in visited:
                            stack.append(dep.task_id)

        return component

    def _deep_copy_plan(self, plan: TaskPlan) -> TaskPlan:
        """Create a deep copy of the plan for optimization."""
        # Simple implementation - in practice you'd want proper deep copying
        new_plan = TaskPlan(
            name=f"Optimized: {plan.name}",
            goal=plan.goal,
            constraints=plan.constraints.copy(),
        )

        for task in plan.tasks.values():
            new_task = Task(
                id=task.id,
                name=task.name,
                description=task.description,
                task_type=task.task_type,
                priority=task.priority,
                estimated_duration=task.estimated_duration,
                resources=task.resources.copy(),
                metadata=task.metadata.copy(),
            )

            # Copy dependencies
            for dep in task.dependencies:
                new_task.dependencies.add(dep)

            new_plan.add_task(new_task)

        return new_plan


class CostOptimizer:
    """Optimizes plans to minimize cost."""

    async def optimize(
        self,
        plan: TaskPlan,
        constraints: List[OptimizationConstraint] = None,
        context: Dict[str, Any] = None,
    ) -> TaskPlan:
        """Optimize plan to minimize cost."""
        optimized_plan = TimeOptimizer()._deep_copy_plan(plan)

        await self._optimize_resource_costs(optimized_plan)
        await self._optimize_task_alternatives(optimized_plan)
        await self._eliminate_redundant_tasks(optimized_plan)

        return optimized_plan

    async def _optimize_resource_costs(self, plan: TaskPlan):
        """Optimize resource allocation to minimize costs."""
        for task in plan.tasks.values():
            for resource in task.resources:
                # Use cheaper alternatives where possible
                if resource.resource_type == "compute":
                    resource.quantity = min(
                        resource.quantity,
                        task.metadata.get("min_compute", resource.quantity),
                    )
                elif resource.resource_type == "storage":
                    resource.quantity = min(
                        resource.quantity,
                        task.metadata.get("min_storage", resource.quantity),
                    )

    async def _optimize_task_alternatives(self, plan: TaskPlan):
        """Find lower-cost alternatives for tasks."""
        for task in plan.tasks.values():
            # Check for cost-optimized alternatives
            if task.task_type == "testing":
                # Use automated testing instead of manual
                task.metadata["cost_factor"] = 0.3
            elif task.task_type == "analysis":
                # Use existing tools instead of custom development
                task.metadata["cost_factor"] = 0.5

    async def _eliminate_redundant_tasks(self, plan: TaskPlan):
        """Eliminate tasks that don't add value."""
        tasks_to_remove = []

        for task in plan.tasks.values():
            # Check if task is redundant
            if (
                task.task_type == "documentation"
                and task.priority == TaskPriority.LOW
                and len(task.dependencies) == 0
            ):
                tasks_to_remove.append(task.id)

        for task_id in tasks_to_remove:
            if task_id in plan.tasks:
                del plan.tasks[task_id]


class QualityOptimizer:
    """Optimizes plans to maximize quality."""

    async def optimize(
        self,
        plan: TaskPlan,
        constraints: List[OptimizationConstraint] = None,
        context: Dict[str, Any] = None,
    ) -> TaskPlan:
        """Optimize plan to maximize quality."""
        optimized_plan = TimeOptimizer()._deep_copy_plan(plan)

        await self._enhance_quality_tasks(optimized_plan)
        await self._add_quality_gates(optimized_plan)
        await self._optimize_review_processes(optimized_plan)

        return optimized_plan

    async def _enhance_quality_tasks(self, plan: TaskPlan):
        """Enhance tasks that contribute to quality."""
        for task in plan.tasks.values():
            if task.task_type in ["testing", "review", "validation"]:
                # Increase time allocation for quality tasks
                task.estimated_duration = task.estimated_duration * 1.3
                task.priority = TaskPriority.HIGH

    async def _add_quality_gates(self, plan: TaskPlan):
        """Add quality gates between major phases."""
        phases = self._identify_phases(plan)

        for i, phase in enumerate(phases[:-1]):
            quality_gate = Task(
                name=f"Quality Gate {i + 1}",
                description=f"Quality validation after {phase}",
                task_type="quality_gate",
                priority=TaskPriority.HIGH,
                estimated_duration=timedelta(minutes=15),
            )
            plan.add_task(quality_gate)

    async def _optimize_review_processes(self, plan: TaskPlan):
        """Optimize code review and validation processes."""
        for task in plan.tasks.values():
            if "review" in task.name.lower():
                # Add peer review component
                task.metadata["peer_review_required"] = True
                task.estimated_duration = task.estimated_duration * 1.2

    def _identify_phases(self, plan: TaskPlan) -> List[str]:
        """Identify major phases in the plan."""
        phases = ["planning", "development", "testing", "deployment"]
        return phases


class RiskOptimizer:
    """Optimizes plans to minimize risk."""

    async def optimize(
        self,
        plan: TaskPlan,
        constraints: List[OptimizationConstraint] = None,
        context: Dict[str, Any] = None,
    ) -> TaskPlan:
        """Optimize plan to minimize risk."""
        optimized_plan = TimeOptimizer()._deep_copy_plan(plan)

        await self._add_contingency_tasks(optimized_plan)
        await self._optimize_critical_dependencies(optimized_plan)
        await self._add_risk_mitigation(optimized_plan)

        return optimized_plan

    async def _add_contingency_tasks(self, plan: TaskPlan):
        """Add contingency tasks for high-risk activities."""
        high_risk_tasks = [
            task
            for task in plan.tasks.values()
            if task.metadata.get("risk_score", 0) > 0.7
        ]

        for risk_task in high_risk_tasks:
            contingency = Task(
                name=f"Contingency: {risk_task.name}",
                description=f"Backup plan for {risk_task.name}",
                task_type="contingency",
                priority=TaskPriority.MEDIUM,
                estimated_duration=risk_task.estimated_duration * 0.8,
            )
            plan.add_task(contingency)

    async def _optimize_critical_dependencies(self, plan: TaskPlan):
        """Optimize dependencies to reduce bottlenecks."""
        # Find tasks with many dependencies
        dependency_counts = defaultdict(int)
        for task in plan.tasks.values():
            for dep in task.dependencies:
                dependency_counts[dep.task_id] += 1

        # Add parallel paths for critical dependencies
        for task_id, count in dependency_counts.items():
            if count > 3:  # High dependency count
                task = plan.tasks.get(task_id)
                if task:
                    # Create alternative path
                    alt_task = Task(
                        name=f"Alternative: {task.name}",
                        description=f"Alternative approach for {task.name}",
                        task_type=task.task_type,
                        priority=TaskPriority.MEDIUM,
                        estimated_duration=task.estimated_duration * 1.2,
                    )
                    plan.add_task(alt_task)

    async def _add_risk_mitigation(self, plan: TaskPlan):
        """Add specific risk mitigation measures."""
        for task in plan.tasks.values():
            risk_score = task.metadata.get("risk_score", 0)

            if risk_score > 0.5:
                # Add extra validation for risky tasks
                task.metadata["validation_required"] = True
                task.metadata["approval_required"] = True

                # Increase buffer time
                task.estimated_duration = task.estimated_duration * 1.5


class ResourceAllocator:
    """Allocates resources optimally across tasks."""

    def __init__(self):
        self.resource_pools = {}
        self.allocation_strategies = {
            "greedy": self._greedy_allocation,
            "balanced": self._balanced_allocation,
            "priority_based": self._priority_based_allocation,
        }

    async def allocate_resources(
        self,
        plan: TaskPlan,
        available_resources: Dict[str, float],
        strategy: str = "priority_based",
    ) -> Dict[str, Dict[str, float]]:
        """Allocate available resources to tasks."""
        allocation_func = self.allocation_strategies.get(
            strategy, self._priority_based_allocation
        )
        return await allocation_func(plan, available_resources)

    async def _greedy_allocation(
        self, plan: TaskPlan, available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Greedy resource allocation - first come, first served."""
        allocation = {}
        remaining_resources = available_resources.copy()

        for task in plan.tasks.values():
            task_allocation = {}

            for resource in task.resources:
                available = remaining_resources.get(resource.resource_type, 0)
                allocated = min(resource.quantity, available)

                task_allocation[resource.resource_type] = allocated
                remaining_resources[resource.resource_type] = available - allocated

            allocation[task.id] = task_allocation

        return allocation

    async def _balanced_allocation(
        self, plan: TaskPlan, available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Balanced resource allocation."""
        allocation = {}

        # Calculate total demand for each resource type
        total_demand = defaultdict(float)
        for task in plan.tasks.values():
            for resource in task.resources:
                total_demand[resource.resource_type] += resource.quantity

        # Calculate allocation ratios
        allocation_ratios = {}
        for resource_type, demand in total_demand.items():
            available = available_resources.get(resource_type, 0)
            allocation_ratios[resource_type] = (
                min(1.0, available / demand) if demand > 0 else 1.0
            )

        # Allocate resources proportionally
        for task in plan.tasks.values():
            task_allocation = {}

            for resource in task.resources:
                ratio = allocation_ratios.get(resource.resource_type, 0)
                allocated = resource.quantity * ratio
                task_allocation[resource.resource_type] = allocated

            allocation[task.id] = task_allocation

        return allocation

    async def _priority_based_allocation(
        self, plan: TaskPlan, available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Priority-based resource allocation."""
        allocation = {}
        remaining_resources = available_resources.copy()

        # Sort tasks by priority
        sorted_tasks = sorted(
            plan.tasks.values(), key=lambda t: t.priority.value, reverse=True
        )

        for task in sorted_tasks:
            task_allocation = {}

            for resource in task.resources:
                available = remaining_resources.get(resource.resource_type, 0)
                allocated = min(resource.quantity, available)

                task_allocation[resource.resource_type] = allocated
                remaining_resources[resource.resource_type] = available - allocated

            allocation[task.id] = task_allocation

        return allocation
