"""
Execution Engine for Dynamic Task Planning and Execution

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Set

from .task_planner import Task, TaskPlan, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution engine status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of task execution."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class ExecutionContext:
    """Context for task execution."""

    plan: TaskPlan
    task: Task
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Schedules tasks based on priorities and dependencies."""

    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.ready_queue = PriorityQueue()
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self._lock = threading.Lock()

    def add_ready_tasks(self, tasks: List[Task]):
        """Add tasks that are ready for execution."""
        with self._lock:
            for task in tasks:
                if (
                    task.id not in self.running_tasks
                    and task.id not in self.completed_tasks
                ):
                    # Higher priority tasks have lower queue priority number
                    priority = -task.priority.value
                    self.ready_queue.put(priority, time.time(), task)

    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute."""
        with self._lock:
            if (
                len(self.running_tasks) >= self.max_concurrent_tasks
                or self.ready_queue.empty()
            ):
                return None

            try:
                _, _, task = self.ready_queue.get_nowait()
                self.running_tasks.add(task.id)
                return task
            except Exception as e:
                logger.error(f"Error: {e}")
                return None

    def mark_task_completed(self, task_id: str, success: bool):
        """Mark a task as completed."""
        with self._lock:
            if task_id in self.running_tasks:
                self.running_tasks.remove(task_id)

            if success:
                self.completed_tasks.add(task_id)
            else:
                self.failed_tasks.add(task_id)

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        with self._lock:
            return {
                "ready_queue_size": self.ready_queue.qsize(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "max_concurrent": self.max_concurrent_tasks,
            }


class ProgressTracker:
    """Tracks execution progress and provides updates."""

    def __init__(self):
        self.task_progress: Dict[str, float] = {}
        self.plan_progress: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
        self.start_times: Dict[str, datetime] = {}
        self.completion_times: Dict[str, datetime] = {}

    def start_task(self, task_id: str):
        """Mark task as started."""
        self.start_times[task_id] = datetime.now()
        self.task_progress[task_id] = 0.0
        self._notify_callbacks("task_started", task_id)

    def update_task_progress(self, task_id: str, progress: float):
        """Update task progress (0.0 to 1.0)."""
        self.task_progress[task_id] = max(0.0, min(1.0, progress))
        self._notify_callbacks("task_progress", task_id, progress)

    def complete_task(self, task_id: str, success: bool):
        """Mark task as completed."""
        self.completion_times[task_id] = datetime.now()
        self.task_progress[task_id] = 1.0 if success else -1.0
        self._notify_callbacks("task_completed", task_id, success)

    def update_plan_progress(self, plan_id: str, completed: int, total: int):
        """Update overall plan progress."""
        progress = completed / total if total > 0 else 0.0
        self.plan_progress[plan_id] = progress
        self._notify_callbacks("plan_progress", plan_id, progress)

    def add_callback(self, callback: Callable):
        """Add progress callback."""
        self.callbacks.append(callback)

    def _notify_callbacks(self, event_type: str, *args):
        """Notify all callbacks of progress events."""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_task_duration(self, task_id: str) -> Optional[timedelta]:
        """Get task execution duration."""
        if task_id in self.start_times:
            end_time = self.completion_times.get(task_id, datetime.now())
            return end_time - self.start_times[task_id]
        return None

    def get_estimated_completion(self, plan: TaskPlan) -> Optional[datetime]:
        """Estimate plan completion time."""
        completed_tasks = sum(1 for p in self.task_progress.values() if p == 1.0)
        total_tasks = len(plan.tasks)

        if completed_tasks == 0:
            return None

        # Calculate average task duration
        durations = [
            self.get_task_duration(tid) for tid in self.completion_times.keys()
        ]
        valid_durations = [d for d in durations if d is not None]

        if not valid_durations:
            return None

        avg_duration = sum(valid_durations, timedelta()) / len(valid_durations)
        remaining_tasks = total_tasks - completed_tasks
        estimated_remaining_time = avg_duration * remaining_tasks

        return datetime.now() + estimated_remaining_time


class AdaptiveExecutor:
    """Executor that adapts to execution results and replans as needed."""

    def __init__(self, replan_threshold: float = 0.3):
        self.replan_threshold = replan_threshold  # Replan if >30% of tasks fail
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}

    async def execute_with_adaptation(
        self, plan: TaskPlan, executor_func: Callable, context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Execute plan with adaptive replanning."""
        results = {}
        current_plan = plan
        attempt = 1
        max_attempts = 3

        while attempt <= max_attempts:
            logger.info(f"Executing plan attempt {attempt}/{max_attempts}")

            # Execute current plan
            attempt_results = await executor_func(current_plan, context)
            results.update(attempt_results)

            # Analyze results
            failed_tasks = {
                tid for tid, result in attempt_results.items() if not result.success
            }
            completed_tasks = {tid for tid, result in results.items() if result.success}

            failure_rate = (
                len(failed_tasks) / len(current_plan.tasks) if current_plan.tasks else 0
            )

            # Record execution history
            self.execution_history.append(
                {
                    "attempt": attempt,
                    "plan_id": current_plan.id,
                    "completed_tasks": len(completed_tasks),
                    "failed_tasks": len(failed_tasks),
                    "failure_rate": failure_rate,
                    "timestamp": datetime.now(),
                }
            )

            # Check if replanning is needed
            if failure_rate <= self.replan_threshold or attempt == max_attempts:
                break

            # Create new plan for failed tasks
            logger.info(
                f"Failure rate {failure_rate:.2%} exceeds threshold. Replanning..."
            )
            current_plan = await self._replan(
                current_plan, completed_tasks, failed_tasks, context
            )
            attempt += 1

        # Update performance metrics
        self._update_performance_metrics(results)

        return results

    async def _replan(
        self,
        original_plan: TaskPlan,
        completed_tasks: Set[str],
        failed_tasks: Set[str],
        context: ExecutionContext,
    ) -> TaskPlan:
        """Create a new plan for remaining tasks."""
        from .task_planner import TaskPlanner

        planner = TaskPlanner()

        # Get remaining tasks
        remaining_tasks = []
        for task_id, task in original_plan.tasks.items():
            if task_id not in completed_tasks:
                # Clone task with increased priority if it failed
                new_task = Task(
                    name=task.name,
                    description=task.description,
                    task_type=task.task_type,
                    priority=task.priority,
                    estimated_duration=task.estimated_duration * 1.2,  # Add buffer
                    resources=task.resources.copy(),
                    metadata=task.metadata.copy(),
                )

                if task_id in failed_tasks:
                    # Increase priority and add retry metadata
                    new_task.priority = TaskPriority(
                        min(task.priority.value + 1, TaskPriority.CRITICAL.value)
                    )
                    new_task.metadata["retry_attempt"] = task.retry_count + 1
                    new_task.metadata["previous_error"] = task.error

                # Update dependencies
                for dep in task.dependencies:
                    if dep.task_id not in completed_tasks:
                        new_task.add_dependency(dep.task_id)

                remaining_tasks.append(new_task)

        # Create new plan
        new_plan = planner.create_plan(
            goal=f"Replan: {original_plan.goal}",
            initial_tasks=remaining_tasks,
            constraints=original_plan.constraints.copy(),
            auto_decompose=False,
        )

        return new_plan

    def _update_performance_metrics(self, results: Dict[str, ExecutionResult]):
        """Update performance metrics based on execution results."""
        if not results:
            return

        success_rate = sum(1 for r in results.values() if r.success) / len(results)
        avg_duration = sum(
            (r.duration.total_seconds() if r.duration else 0) for r in results.values()
        ) / len(results)

        self.performance_metrics.update(
            {
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "total_executions": len(self.execution_history),
                "last_updated": datetime.now().isoformat(),
            }
        )


class ParallelExecutor:
    """Executes tasks in parallel using thread/async pools."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

    async def execute_parallel_tasks(
        self, tasks: List[Task], executor_func: Callable, context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks in parallel."""
        if not tasks:
            return {}

        # Separate async and sync tasks
        async_tasks = []
        sync_tasks = []

        for task in tasks:
            if task.metadata.get("execution_type") == "async":
                async_tasks.append(task)
            else:
                sync_tasks.append(task)

        results = {}

        # Execute async tasks
        if async_tasks:
            async_results = await self._execute_async_tasks(
                async_tasks, executor_func, context
            )
            results.update(async_results)

        # Execute sync tasks
        if sync_tasks:
            sync_results = await self._execute_sync_tasks(
                sync_tasks, executor_func, context
            )
            results.update(sync_results)

        return results

    async def _execute_async_tasks(
        self, tasks: List[Task], executor_func: Callable, context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Execute async tasks concurrently."""

        async def execute_task(task: Task) -> ExecutionResult:
            try:
                start_time = datetime.now()
                result = await executor_func(task, context)
                end_time = datetime.now()

                return ExecutionResult(
                    task_id=task.id,
                    success=True,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    start_time=start_time,
                    end_time=datetime.now(),
                )

        # Execute all async tasks concurrently
        task_coroutines = [execute_task(task) for task in tasks]
        execution_results = await asyncio.gather(
            *task_coroutines, return_exceptions=True
        )

        results = {}
        for result in execution_results:
            if isinstance(result, ExecutionResult):
                results[result.task_id] = result
            else:
                # Handle exceptions
                logger.error(f"Unexpected error in async execution: {result}")

        return results

    async def _execute_sync_tasks(
        self, tasks: List[Task], executor_func: Callable, context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Execute sync tasks using thread pool."""

        def execute_task_sync(task: Task) -> ExecutionResult:
            try:
                start_time = datetime.now()

                # Check if executor_func is async
                if asyncio.iscoroutinefunction(executor_func):
                    # Run async function in new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(executor_func(task, context))
                    finally:
                        loop.close()
                else:
                    result = executor_func(task, context)

                end_time = datetime.now()

                return ExecutionResult(
                    task_id=task.id,
                    success=True,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                )
            except Exception as e:
                logger.error(f"Task {task.id} failed: {e}")
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )

        # Submit tasks to thread pool
        futures = {
            self.thread_pool.submit(execute_task_sync, task): task.id for task in tasks
        }

        results = {}
        for future in as_completed(futures):
            try:
                result = future.result()
                results[result.task_id] = result
            except Exception as e:
                task_id = futures[future]
                logger.error(f"Thread pool execution failed for task {task_id}: {e}")
                results[task_id] = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )

        return results

    def shutdown(self):
        """Shutdown the thread pool."""
        self.thread_pool.shutdown(wait=True)


class ExecutionEngine:
    """Main execution engine that orchestrates task execution."""

    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        enable_adaptive_execution: bool = True,
        enable_parallel_execution: bool = True,
    ):
        self.scheduler = TaskScheduler(max_concurrent_tasks)
        self.progress_tracker = ProgressTracker()
        self.adaptive_executor = (
            AdaptiveExecutor() if enable_adaptive_execution else None
        )
        self.parallel_executor = (
            ParallelExecutor() if enable_parallel_execution else None
        )
        self.status = ExecutionStatus.IDLE
        self.current_plan: Optional[TaskPlan] = None
        self.execution_results: Dict[str, ExecutionResult] = {}
        self._stop_event = asyncio.Event()

    async def execute_plan(
        self,
        plan: TaskPlan,
        task_executor: Callable,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, ExecutionResult]:
        """Execute a complete task plan."""
        if self.status == ExecutionStatus.RUNNING:
            raise RuntimeError("Execution engine is already running")

        self.status = ExecutionStatus.RUNNING
        self.current_plan = plan
        self.execution_results = {}
        self._stop_event.clear()

        if context is None:
            context = ExecutionContext(plan=plan, task=None)

        try:
            logger.info(f"Starting execution of plan: {plan.name}")

            if self.adaptive_executor:
                # Use adaptive execution
                results = await self.adaptive_executor.execute_with_adaptation(
                    plan, self._execute_plan_internal, context
                )
            else:
                # Use regular execution
                results = await self._execute_plan_internal(plan, context)

            self.execution_results = results
            self.status = ExecutionStatus.IDLE

            logger.info(f"Plan execution completed. {len(results)} tasks executed.")
            return results

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            self.status = ExecutionStatus.ERROR
            raise

    async def _execute_plan_internal(
        self, plan: TaskPlan, context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Internal plan execution logic."""
        results = {}

        # Get execution levels (tasks that can run in parallel)
        from .task_planner import TaskPlanner

        planner = TaskPlanner()
        execution_levels = planner.get_execution_order(plan)

        total_tasks = len(plan.tasks)
        completed_tasks = 0

        # Execute each level
        for level_idx, level_tasks in enumerate(execution_levels):
            if self._stop_event.is_set():
                logger.info("Execution stopped by user request")
                break

            logger.info(
                f"Executing level {level_idx + 1}/{len(execution_levels)} with {len(level_tasks)} tasks"
            )

            # Update task statuses
            for task in level_tasks:
                task.status = TaskStatus.READY
                self.progress_tracker.start_task(task.id)

            # Execute tasks in this level
            if self.parallel_executor and len(level_tasks) > 1:
                # Parallel execution
                level_results = await self.parallel_executor.execute_parallel_tasks(
                    level_tasks, self._execute_single_task, context
                )
            else:
                # Sequential execution
                level_results = await self._execute_tasks_sequential(
                    level_tasks, context
                )

            # Process results
            for task_id, result in level_results.items():
                results[task_id] = result
                task = plan.get_task(task_id)

                if result.success:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.result
                    completed_tasks += 1
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.error

                task.actual_duration = result.duration
                self.progress_tracker.complete_task(task_id, result.success)

            # Update overall progress
            self.progress_tracker.update_plan_progress(
                plan.id, completed_tasks, total_tasks
            )

        return results

    async def _execute_tasks_sequential(
        self, tasks: List[Task], context: ExecutionContext
    ) -> Dict[str, ExecutionResult]:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            if self._stop_event.is_set():
                break

            result = await self._execute_single_task(task, context)
            results[task.id] = result

        return results

    async def _execute_single_task(
        self, task: Task, context: ExecutionContext
    ) -> ExecutionResult:
        """Execute a single task."""
        start_time = datetime.now()
        task.status = TaskStatus.RUNNING
        task.started_at = start_time

        try:
            logger.info(f"Executing task: {task.name}")

            # Update context with current task
            context.task = task

            # Execute task based on type
            result = await self._dispatch_task_execution(task, context)

            end_time = datetime.now()
            task.completed_at = end_time

            return ExecutionResult(
                task_id=task.id,
                success=True,
                result=result,
                start_time=start_time,
                end_time=end_time,
                metadata={"task_type": task.task_type},
            )

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            end_time = datetime.now()
            task.completed_at = end_time

            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                metadata={"task_type": task.task_type},
            )

    async def _dispatch_task_execution(
        self, task: Task, context: ExecutionContext
    ) -> Any:
        """Dispatch task execution based on task type."""
        # This is where you would integrate with actual execution systems
        # For now, we'll simulate task execution

        execution_time = task.estimated_duration.total_seconds()

        # Simulate different task types
        if task.task_type == "coding":
            await asyncio.sleep(min(execution_time, 5))  # Cap simulation time
            return {"code": f"# Generated code for {task.name}", "lines": 50}

        elif task.task_type == "testing":
            await asyncio.sleep(min(execution_time, 3))
            return {"tests_run": 10, "passed": 9, "failed": 1}

        elif task.task_type == "analysis":
            await asyncio.sleep(min(execution_time, 4))
            return {"insights": ["Finding 1", "Finding 2"], "confidence": 0.85}

        else:
            # Generic task execution
            await asyncio.sleep(min(execution_time, 2))
            return {"status": "completed", "output": f"Result for {task.name}"}

    def pause_execution(self):
        """Pause execution."""
        self.status = ExecutionStatus.PAUSED
        logger.info("Execution paused")

    def resume_execution(self):
        """Resume execution."""
        if self.status == ExecutionStatus.PAUSED:
            self.status = ExecutionStatus.RUNNING
            logger.info("Execution resumed")

    def stop_execution(self):
        """Stop execution."""
        self._stop_event.set()
        self.status = ExecutionStatus.STOPPED
        logger.info("Execution stop requested")

    def get_status(self) -> Dict[str, Any]:
        """Get execution engine status."""
        return {
            "status": self.status.value,
            "current_plan": self.current_plan.id if self.current_plan else None,
            "scheduler_status": self.scheduler.get_status(),
            "progress": self.progress_tracker.plan_progress.get(
                self.current_plan.id if self.current_plan else "", 0.0
            ),
            "completed_tasks": len(
                [r for r in self.execution_results.values() if r.success]
            ),
            "failed_tasks": len(
                [r for r in self.execution_results.values() if not r.success]
            ),
        }

    def shutdown(self):
        """Shutdown the execution engine."""
        self.stop_execution()
        if self.parallel_executor:
            self.parallel_executor.shutdown()
        logger.info("Execution engine shutdown complete")
