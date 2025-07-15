import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.react import ReactAgent
from ..llm import LLMProvider
from .baseline_agents import BaselineAgentFactory
from .gaia_benchmark import GAIABenchmark, GAIATask

"""
SPRE (Self-Planning and Resource-based Execution) evaluation component.
This module orchestrates the evaluation of different baseline agents against
the GAIA benchmark to quantify the performance improvements of SPRE.
"""

__all__ = ["BenchmarkResult", "SPREEvaluator"]


@dataclass
class TaskResult:
    """Individual task execution result."""

    task_id: str
    question: str
    expected_answer: str
    actual_answer: str
    success: bool
    execution_time: float
    tokens_used: int
    api_calls: int
    reasoning_tokens: int
    baseline_type: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark results for one agent configuration."""

    baseline_type: str
    agent_name: str
    task_results: List[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate percentage."""
        if not self.task_results:
            return 0.0
        successes = sum(1 for r in self.task_results if r.success)
        return (successes / len(self.task_results)) * 100

    @property
    def avg_api_calls(self) -> float:
        """Average API calls per task."""
        if not self.task_results:
            return 0.0
        return statistics.mean(r.api_calls for r in self.task_results)

    @property
    def avg_latency(self) -> float:
        """Average execution time per task in seconds."""
        if not self.task_results:
            return 0.0
        return statistics.mean(r.execution_time for r in self.task_results)

    @property
    def avg_tokens(self) -> float:
        """Average tokens used per task."""
        if not self.task_results:
            return 0.0
        return statistics.mean(r.tokens_used for r in self.task_results)

    @property
    def efficiency_ratio(self) -> float:
        """Success rate divided by average API calls (higher is better)."""
        if self.avg_api_calls == 0:
            return self.success_rate
        return self.success_rate / self.avg_api_calls

    def get_stats_by_difficulty(self) -> Dict[str, Dict[str, float]]:
        """Get statistics broken down by task difficulty."""
        from .gaia_benchmark import GAIABenchmark

        benchmark = GAIABenchmark()

        stats = {}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_tasks = []
            for result in self.task_results:
                task = benchmark.get_task_by_id(result.task_id)
                if task and task.difficulty == difficulty:
                    difficulty_tasks.append(result)

            if difficulty_tasks:
                successes = sum(1 for r in difficulty_tasks if r.success)
                stats[difficulty] = {
                    "success_rate": (successes / len(difficulty_tasks)) * 100,
                    "avg_api_calls": statistics.mean(
                        r.api_calls for r in difficulty_tasks
                    ),
                    "avg_latency": statistics.mean(
                        r.execution_time for r in difficulty_tasks
                    ),
                    "count": len(difficulty_tasks),
                }
            else:
                stats[difficulty] = {
                    "success_rate": 0.0,
                    "avg_api_calls": 0.0,
                    "avg_latency": 0.0,
                    "count": 0,
                }

        return stats


class SPREEvaluator:
    """Comprehensive SPRE evaluation framework."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        output_dir: Optional[Path] = None,
    ):
        self.llm_provider = llm_provider
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.benchmark = GAIABenchmark()

    async def run_full_evaluation(
        self,
        task_filter: Optional[Dict[str, Any]] = None,
        max_tasks_per_baseline: int = 20,
    ) -> Dict[str, BenchmarkResult]:
        """Run complete evaluation across all baseline types."""

        # Get filtered tasks
        tasks = self.benchmark.get_tasks(
            difficulty=task_filter.get("difficulty") if task_filter else None,
            domain=task_filter.get("domain") if task_filter else None,
            min_steps=task_filter.get("min_steps", 3) if task_filter else 3,
            limit=max_tasks_per_baseline,
        )

        if not tasks:
            raise ValueError("No tasks found matching filter criteria")

        print(f"Running evaluation on {len(tasks)} tasks...")

        results = {}

        # Evaluate each baseline type
        for baseline_type in BaselineAgentFactory.get_all_baseline_types():
            print(f"\nEvaluating {baseline_type}...")

            agent = BaselineAgentFactory.create_agent(
                baseline_type, self.llm_provider, name_suffix="-Eval"
            )

            result = await self._evaluate_agent_on_tasks(agent, tasks, baseline_type)
            results[baseline_type] = result

            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Avg API Calls: {result.avg_api_calls:.1f}")
            print(f"  Avg Latency: {result.avg_latency:.2f}s")

        # Save results
        await self._save_results(results)

        # Generate comparison report
        self._generate_comparison_report(results)

        return results

    async def _evaluate_agent_on_tasks(
        self, agent: ReactAgent, tasks: List[GAIATask], baseline_type: str
    ) -> BenchmarkResult:
        """Evaluate single agent on task list."""

        task_results = []

        for i, task in enumerate(tasks):
            print(f"  Task {i + 1}/{len(tasks)}: {task.task_id}")

            start_time = time.time()

            try:
                # Execute task
                response = await agent.execute(task.question)

                # Count API calls from trace
                api_calls = self._count_api_calls(response.trace)

                # Evaluate success
                success = self._evaluate_answer(task.expected_answer, response.content)

                task_result = TaskResult(
                    task_id=task.task_id,
                    question=task.question,
                    expected_answer=task.expected_answer,
                    actual_answer=response.content,
                    success=success,
                    execution_time=time.time() - start_time,
                    tokens_used=response.tokens_used,
                    api_calls=api_calls,
                    reasoning_tokens=len(response.content) // 4,  # Estimate
                    baseline_type=baseline_type,
                    trace=response.trace,
                )

            except Exception as e:
                task_result = TaskResult(
                    task_id=task.task_id,
                    question=task.question,
                    expected_answer=task.expected_answer,
                    actual_answer=f"ERROR: {str(e)}",
                    success=False,
                    execution_time=time.time() - start_time,
                    tokens_used=0,
                    api_calls=0,
                    reasoning_tokens=0,
                    baseline_type=baseline_type,
                    error_message=str(e),
                )

            task_results.append(task_result)

        return BenchmarkResult(
            baseline_type=baseline_type,
            agent_name=agent.config.name,
            task_results=task_results,
        )

    def _count_api_calls(self, trace: List[Dict[str, Any]]) -> int:
        """Count API calls from execution trace."""
        api_call_events = [
            "planner_response",
            "resource_assessment_detail",
            "internal_execution",
            "synthesis_complete",
        ]

        count = 0
        for event in trace:
            if event.get("type") in api_call_events:
                count += 1

        return count

    def _evaluate_answer(self, expected: str, actual: str) -> bool:
        """Evaluate if actual answer matches expected (with fuzzy matching)."""
        # Simple evaluation - can be enhanced with more sophisticated matching
        expected_clean = expected.lower().strip()
        actual_clean = actual.lower().strip()

        # Direct match
        if expected_clean == actual_clean:
            return True

        # Check if expected answer is contained in actual
        if expected_clean in actual_clean:
            return True

        # For numeric answers, extract numbers and compare
        import re

        expected_numbers = re.findall(r"\d+\.?\d*", expected)
        actual_numbers = re.findall(r"\d+\.?\d*", actual)

        if expected_numbers and actual_numbers:
            try:
                # Compare first number found
                exp_num = float(expected_numbers[0])
                act_num = float(actual_numbers[0])
                # Allow 5% tolerance for floating point comparisons
                return abs(exp_num - act_num) / max(exp_num, 1) < 0.05
            except ValueError:
                pass

        return False

    async def _save_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """Save detailed results to JSON files."""
        timestamp = int(time.time())

        for baseline_type, result in results.items():
            filename = f"benchmark_{baseline_type}_{timestamp}.json"
            filepath = self.output_dir / filename

            # Convert to serializable format
            data = {
                "baseline_type": result.baseline_type,
                "agent_name": result.agent_name,
                "timestamp": timestamp,
                "summary": {
                    "success_rate": result.success_rate,
                    "avg_api_calls": result.avg_api_calls,
                    "avg_latency": result.avg_latency,
                    "avg_tokens": result.avg_tokens,
                    "efficiency_ratio": result.efficiency_ratio,
                },
                "task_results": [
                    {
                        "task_id": tr.task_id,
                        "question": tr.question,
                        "expected_answer": tr.expected_answer,
                        "actual_answer": tr.actual_answer,
                        "success": tr.success,
                        "execution_time": tr.execution_time,
                        "tokens_used": tr.tokens_used,
                        "api_calls": tr.api_calls,
                        "error_message": tr.error_message,
                    }
                    for tr in result.task_results
                ],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def _generate_comparison_report(self, results: Dict[str, BenchmarkResult]) -> None:
        """Generate comparison report in markdown format."""
        timestamp = int(time.time())
        report_path = self.output_dir / f"comparison_report_{timestamp}.md"

        with open(report_path, "w") as f:
            f.write("# SPRE Benchmark Results\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary table
            f.write("## Summary Comparison\n\n")
            f.write(
                "| Agent Configuration | Task Success Rate (%) | Avg. API Calls | Avg. Latency (s) | Efficiency Ratio |\n"
            )
            f.write(
                "|---------------------|----------------------|----------------|------------------|------------------|\n"
            )

            for baseline_type in BaselineAgentFactory.get_all_baseline_types():
                if baseline_type in results:
                    result = results[baseline_type]
                    f.write(
                        f"| {result.agent_name} | {result.success_rate:.1f} | {result.avg_api_calls:.1f} | {result.avg_latency:.2f} | {result.efficiency_ratio:.2f} |\n"
                    )

            f.write("\n")

            # Detailed analysis
            f.write("## Detailed Analysis\n\n")

            for _baseline_type, result in results.items():
                f.write(f"### {result.agent_name}\n\n")
                f.write(f"- **Success Rate**: {result.success_rate:.1f}%\n")
                f.write(f"- **Average API Calls**: {result.avg_api_calls:.1f}\n")
                f.write(f"- **Average Latency**: {result.avg_latency:.2f}s\n")
                f.write(f"- **Average Tokens**: {result.avg_tokens:.0f}\n")
                f.write(f"- **Efficiency Ratio**: {result.efficiency_ratio:.2f}\n")

                # Difficulty breakdown
                difficulty_stats = result.get_stats_by_difficulty()
                f.write("\n**Performance by Difficulty:**\n")
                for difficulty, stats in difficulty_stats.items():
                    if stats["count"] > 0:
                        f.write(
                            f"- {difficulty.title()}: {stats['success_rate']:.1f}% success ({stats['count']} tasks)\n"
                        )

                f.write("\n")

            # Key findings
            f.write("## Key Findings\n\n")

            # Find best performing agent
            best_efficiency = max(results.values(), key=lambda r: r.efficiency_ratio)
            best_success = max(results.values(), key=lambda r: r.success_rate)
            lowest_api = min(results.values(), key=lambda r: r.avg_api_calls)

            f.write(
                f"- **Highest Efficiency**: {best_efficiency.agent_name} (ratio: {best_efficiency.efficiency_ratio:.2f})\n"
            )
            f.write(
                f"- **Highest Success Rate**: {best_success.agent_name} ({best_success.success_rate:.1f}%)\n"
            )
            f.write(
                f"- **Lowest API Usage**: {lowest_api.agent_name} ({lowest_api.avg_api_calls:.1f} calls/task)\n"
            )

        print(f"\nComparison report saved to: {report_path}")

    async def run_single_baseline_evaluation(
        self,
        baseline_type: str,
        task_ids: Optional[List[str]] = None,
        max_tasks: int = 10,
    ) -> BenchmarkResult:
        """Run evaluation for single baseline type."""

        if task_ids:
            tasks = [self.benchmark.get_task_by_id(tid) for tid in task_ids]
            tasks = [t for t in tasks if t is not None]
        else:
            tasks = self.benchmark.get_tasks(min_steps=3, limit=max_tasks)

        agent = BaselineAgentFactory.create_agent(baseline_type, self.llm_provider)
        return await self._evaluate_agent_on_tasks(agent, tasks, baseline_type)
