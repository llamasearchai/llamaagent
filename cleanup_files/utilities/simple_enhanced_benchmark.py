#!/usr/bin/env python3
"""
Simple Enhanced Benchmark System for LlamaAgent

A standalone benchmark system that provides comprehensive evaluation capabilities
without complex dependencies.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of benchmark tasks."""

    MATHEMATICAL = "mathematical"
    LOGICAL_REASONING = "logical_reasoning"
    PROGRAMMING = "programming"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    PROBLEM_SOLVING = "problem_solving"


class DifficultyLevel(Enum):
    """Difficulty levels for tasks."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkTask:
    """Individual benchmark task."""

    id: str
    task_type: TaskType
    difficulty: DifficultyLevel
    question: str
    expected_answer: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a benchmark task execution."""

    task_id: str
    agent_name: str
    actual_answer: str
    expected_answer: str
    score: float
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""

    agent_name: str
    task_results: List[TaskResult]
    overall_score: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class MockAgent:
    """Mock agent for testing purposes."""

    def __init__(self, name: str = "TestAgent"):
        self.name = name
        self.agent_id = f"agent_{uuid.uuid4().hex[:8]}"

    async def execute_task(self, task_input: str) -> str:
        """Execute a task and return response."""
        await asyncio.sleep(0.1)  # Simulate processing time

        # Intelligent mock responses based on task content
        if "%" in task_input and "of" in task_input:
            return self._handle_percentage_calculation(task_input)
        elif "perimeter" in task_input.lower():
            return self._handle_geometry_calculation(task_input)
        elif "compound interest" in task_input.lower():
            return self._handle_compound_interest(task_input)
        elif "derivative" in task_input.lower():
            return self._handle_derivative_calculation(task_input)
        elif "python function" in task_input.lower():
            return self._handle_programming_task(task_input)
        else:
            return "This is a mock response for testing purposes."

    def _handle_percentage_calculation(self, task: str) -> str:
        """Handle percentage calculations."""
        try:
            # Pattern: "Calculate X% of Y and then add Z"
            percentage_match = re.search(
                r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', task
            )
            add_match = re.search(r'add\s+(\d+(?:\.\d+)?)', task)

            if percentage_match and add_match:
                percentage = float(percentage_match.group(1))
                number = float(percentage_match.group(2))
                add_value = float(add_match.group(1))

                # Calculate: X% of Y + Z
                percent_result = (percentage / 100) * number
                final_result = percent_result + add_value

                return str(
                    int(final_result) if final_result.is_integer() else final_result
                )

            elif percentage_match:
                percentage = float(percentage_match.group(1))
                number = float(percentage_match.group(2))
                result = (percentage / 100) * number
                return str(int(result) if result.is_integer() else result)

        except Exception:
            pass

        return "Error calculating percentage"

    def _handle_geometry_calculation(self, task: str) -> str:
        """Handle geometry calculations."""
        try:
            # Pattern: "rectangle has length X cm and width Y cm, what is its perimeter?"
            length_match = re.search(r'length\s+(\d+(?:\.\d+)?)', task)
            width_match = re.search(r'width\s+(\d+(?:\.\d+)?)', task)

            if length_match and width_match:
                length = float(length_match.group(1))
                width = float(width_match.group(1))
                perimeter = 2 * (length + width)
                return f"{int(perimeter)} cm"

        except Exception:
            pass

        return "Error calculating geometry"

    def _handle_compound_interest(self, task: str) -> str:
        """Handle compound interest calculations."""
        try:
            # Pattern: "compound interest on $X at Y% annual rate for Z years"
            principal_match = re.search(r'\$(\d+(?:,\d+)?)', task)
            rate_match = re.search(r'(\d+(?:\.\d+)?)%', task)
            years_match = re.search(r'for\s+(\d+)\s+years?', task)

            if principal_match and rate_match and years_match:
                principal = float(principal_match.group(1).replace(',', ''))
                rate = float(rate_match.group(1))
                years = int(years_match.group(1))

                amount = principal * (1 + rate / 100) ** years
                return f"${amount:.2f}"

        except Exception:
            pass

        return "Error calculating compound interest"

    def _handle_derivative_calculation(self, task: str) -> str:
        """Handle derivative calculations."""
        try:
            # Pattern: "derivative of f(x) = 3x³ - 2x² + 5x - 1, then evaluate it at x = 2"
            if "3x³ - 2x² + 5x - 1" in task and "x = 2" in task:
                # f'(x) = 9x² - 4x + 5
                # f'(2) = 9(4) - 4(2) + 5 = 36 - 8 + 5 = 33
                return "33"
            elif "x = 2" in task:
                return "37"  # Alternative expected answer

        except Exception:
            pass

        return "Error calculating derivative"

    def _handle_programming_task(self, task: str) -> str:
        """Handle programming tasks."""
        if "maximum of two numbers" in task.lower():
            return "def max_two(a, b): return a if a > b else b"
        elif "factorial" in task.lower():
            return "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        elif "palindrome" in task.lower():
            return "def is_palindrome(s): return s == s[::-1]"
        else:
            return "def example_function(): pass"


class AdvancedScorer:
    """Advanced scoring system."""

    def score_task(self, task: BenchmarkTask, actual_answer: str) -> float:
        """Score a task result."""
        expected = task.expected_answer.strip().lower()
        actual = actual_answer.strip().lower()

        # Exact match
        if expected == actual:
            return 1.0

        # Numerical tolerance for math problems
        if task.task_type == TaskType.MATHEMATICAL:
            return self._score_numerical(expected, actual)

        # Programming task scoring
        if task.task_type == TaskType.PROGRAMMING:
            return self._score_programming(expected, actual)

        # Semantic similarity for other tasks
        return self._score_semantic_similarity(expected, actual)

    def _score_numerical(self, expected: str, actual: str) -> float:
        """Score numerical answers with tolerance."""
        try:
            # Extract numbers from strings
            expected_num = float(re.findall(r'-?\d+(?:\.\d+)?', expected)[0])
            actual_num = float(re.findall(r'-?\d+(?:\.\d+)?', actual)[0])

            # Check if within 1% tolerance
            if abs(expected_num - actual_num) <= abs(expected_num * 0.01):
                return 1.0
            else:
                # Partial credit based on relative error
                relative_error = abs(expected_num - actual_num) / abs(expected_num)
                return max(0.0, 1.0 - relative_error)

        except (ValueError, IndexError):
            return 0.0

    def _score_programming(self, expected: str, actual: str) -> float:
        """Score programming answers."""
        # Check for key elements
        if "def" in actual and any(word in actual for word in expected.split()):
            return 0.8  # Partial credit for function structure
        return 0.0

    def _score_semantic_similarity(self, expected: str, actual: str) -> float:
        """Score based on semantic similarity."""
        expected_words = set(expected.split())
        actual_words = set(actual.split())

        if not expected_words:
            return 0.0

        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)

        return len(intersection) / len(union) if union else 0.0


class BenchmarkSuite:
    """Collection of benchmark tasks."""

    def __init__(self, name: str):
        self.name = name
        self.tasks = []
        self._create_default_tasks()

    def _create_default_tasks(self):
        """Create default benchmark tasks."""

        # Mathematical tasks
        self.tasks.extend(
            [
                BenchmarkTask(
                    id="math_001",
                    task_type=TaskType.MATHEMATICAL,
                    difficulty=DifficultyLevel.MEDIUM,
                    question="Calculate 15% of 240 and then add 30 to the result.",
                    expected_answer="66",
                ),
                BenchmarkTask(
                    id="math_002",
                    task_type=TaskType.MATHEMATICAL,
                    difficulty=DifficultyLevel.EASY,
                    question="If a rectangle has length 8 cm and width 5 cm, what is its perimeter?",
                    expected_answer="26 cm",
                ),
                BenchmarkTask(
                    id="math_003",
                    task_type=TaskType.MATHEMATICAL,
                    difficulty=DifficultyLevel.HARD,
                    question="Calculate the compound interest on $5000 at 8% annual rate for 3 years.",
                    expected_answer="$6298.56",
                ),
                BenchmarkTask(
                    id="math_004",
                    task_type=TaskType.MATHEMATICAL,
                    difficulty=DifficultyLevel.HARD,
                    question="Find the derivative of f(x) = 3x³ - 2x² + 5x - 1, then evaluate it at x = 2.",
                    expected_answer="33",
                ),
            ]
        )

        # Programming tasks
        self.tasks.extend(
            [
                BenchmarkTask(
                    id="prog_001",
                    task_type=TaskType.PROGRAMMING,
                    difficulty=DifficultyLevel.EASY,
                    question="Write a Python function that returns the maximum of two numbers.",
                    expected_answer="def max_two(a, b): return a if a > b else b",
                ),
            ]
        )


class BenchmarkRunner:
    """Execute benchmark suites against agents."""

    def __init__(self):
        self.scorer = AdvancedScorer()

    async def run_benchmark(
        self, suite: BenchmarkSuite, agent: MockAgent
    ) -> BenchmarkResult:
        """Run a benchmark suite against an agent."""
        start_time = time.time()

        print(f"Running benchmark '{suite.name}' with {len(suite.tasks)} tasks")
        print(f"Agent Testing agent: {agent.name}")

        task_results = []

        for i, task in enumerate(suite.tasks, 1):
            print(f"  Task {i}/{len(suite.tasks)}: {task.id}")

            task_start = time.time()
            try:
                actual_answer = await agent.execute_task(task.question)
                task_time = time.time() - task_start

                score = self.scorer.score_task(task, actual_answer)
                success = score > 0.5

                result = TaskResult(
                    task_id=task.id,
                    agent_name=agent.name,
                    actual_answer=actual_answer,
                    expected_answer=task.expected_answer,
                    score=score,
                    execution_time=task_time,
                    success=success,
                )

                task_results.append(result)

                # Print result
                status = "PASS" if success else "FAIL"
                print(f"    {status} Expected: {task.expected_answer}")
                print(f"    {status} Got: {actual_answer}")
                print(f"    {status} Score: {score:.2f}")

            except Exception as e:
                task_time = time.time() - task_start
                print(f"    FAIL Error: {e}")

                result = TaskResult(
                    task_id=task.id,
                    agent_name=agent.name,
                    actual_answer="",
                    expected_answer=task.expected_answer,
                    score=0.0,
                    execution_time=task_time,
                    success=False,
                    error_message=str(e),
                )
                task_results.append(result)

        execution_time = time.time() - start_time
        overall_score = (
            statistics.mean([r.score for r in task_results]) if task_results else 0.0
        )

        result = BenchmarkResult(
            agent_name=agent.name,
            task_results=task_results,
            overall_score=overall_score,
            execution_time=execution_time,
        )

        return result


class BenchmarkAnalyzer:
    """Analyze benchmark results."""

    def analyze_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze a benchmark result."""
        successful_tasks = [r for r in result.task_results if r.success]
        failed_tasks = [r for r in result.task_results if not r.success]

        scores = [r.score for r in result.task_results]
        execution_times = [r.execution_time for r in result.task_results]

        analysis = {
            "overall_performance": {
                "success_rate": len(successful_tasks) / len(result.task_results)
                if result.task_results
                else 0.0,
                "average_score": statistics.mean(scores) if scores else 0.0,
                "total_tasks": len(result.task_results),
                "successful_tasks": len(successful_tasks),
                "failed_tasks": len(failed_tasks),
            },
            "time_analysis": {
                "average_time": statistics.mean(execution_times)
                if execution_times
                else 0.0,
                "total_time": sum(execution_times),
                "min_time": min(execution_times) if execution_times else 0.0,
                "max_time": max(execution_times) if execution_times else 0.0,
            },
            "task_breakdown": self._analyze_by_task_type(result),
        }

        return analysis

    def _analyze_by_task_type(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze performance by task type."""
        task_type_results = {}

        for task_result in result.task_results:
            # Infer task type from task_id
            if task_result.task_id.startswith("math_"):
                task_type = "mathematical"
            elif task_result.task_id.startswith("prog_"):
                task_type = "programming"
            else:
                task_type = "other"

            if task_type not in task_type_results:
                task_type_results[task_type] = []

            task_type_results[task_type].append(task_result)

        analysis = {}
        for task_type, results in task_type_results.items():
            scores = [r.score for r in results]
            analysis[task_type] = {
                "success_rate": len([r for r in results if r.success]) / len(results)
                if results
                else 0.0,
                "average_score": statistics.mean(scores) if scores else 0.0,
                "task_count": len(results),
            }

        return analysis


class SimpleEnhancedBenchmarkSystem:
    """Main benchmark system coordinator."""

    def __init__(self):
        self.runner = BenchmarkRunner()
        self.analyzer = BenchmarkAnalyzer()

    async def run_comprehensive_evaluation(self, agent: MockAgent) -> Dict[str, Any]:
        """Run a comprehensive evaluation."""

        # Create benchmark suite
        suite = BenchmarkSuite("Comprehensive Evaluation")

        # Run benchmark
        result = await self.runner.run_benchmark(suite, agent)

        # Analyze results
        analysis = self.analyzer.analyze_result(result)

        return {
            "result": result,
            "analysis": analysis,
            "summary": {
                "agent_name": agent.name,
                "overall_score": result.overall_score,
                "success_rate": analysis["overall_performance"]["success_rate"],
                "execution_time": result.execution_time,
                "total_tasks": len(result.task_results),
            },
        }

    def print_detailed_report(self, evaluation: Dict[str, Any]):
        """Print a detailed evaluation report."""

        summary = evaluation["summary"]
        analysis = evaluation["analysis"]

        print("\n" + "=" * 60)
        print("RESULTS ENHANCED BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\nAgent Agent: {summary['agent_name']}")
        print(f"Performance Overall Score: {summary['overall_score']:.2%}")
        print(f"PASS Success Rate: {summary['success_rate']:.2%}")
        print(f"TIME:  Total Execution Time: {summary['execution_time']:.2f}s")
        print(f"LIST: Total Tasks: {summary['total_tasks']}")

        # Performance breakdown
        print(f"\nRESULTS Performance Breakdown:")
        overall = analysis["overall_performance"]
        print(f"  Successful Tasks: {overall['successful_tasks']}")
        print(f"  Failed Tasks: {overall['failed_tasks']}")
        print(f"  Average Score: {overall['average_score']:.2%}")

        # Time analysis
        print(f"\nTIME:  Time Analysis:")
        time_analysis = analysis["time_analysis"]
        print(f"  Average Time per Task: {time_analysis['average_time']:.3f}s")
        print(f"  Fastest Task: {time_analysis['min_time']:.3f}s")
        print(f"  Slowest Task: {time_analysis['max_time']:.3f}s")

        # Task type breakdown
        print(f"\nLIST: Task Type Performance:")
        for task_type, stats in analysis["task_breakdown"].items():
            print(f"  {task_type.title()}:")
            print(f"    Success Rate: {stats['success_rate']:.2%}")
            print(f"    Average Score: {stats['average_score']:.2%}")
            print(f"    Task Count: {stats['task_count']}")

        # Key insights
        print(f"\nINSIGHT Key Insights:")
        if summary['success_rate'] > 0.8:
            print("  SUCCESS Excellent performance with >80% success rate")
        elif summary['success_rate'] > 0.6:
            print("   Good performance with >60% success rate")
        else:
            print("  WARNING:  Performance needs improvement")

        if time_analysis['average_time'] < 0.2:
            print("  Analyzing Very fast response times")
        elif time_analysis['average_time'] < 0.5:
            print("  Starting Good response times")
        else:
            print("   Consider optimizing response times")

        print("\n" + "=" * 60)


async def main():
    """Main demonstration function."""

    print("Starting Simple Enhanced Benchmark System Demo")
    print("=" * 50)

    # Initialize the benchmark system
    benchmark_system = SimpleEnhancedBenchmarkSystem()

    # Create test agents
    agents = [MockAgent("Intelligent-Mock-Agent"), MockAgent("Basic-Mock-Agent")]

    # Run evaluations for each agent
    for agent in agents:
        print(f"\nRunning Evaluating {agent.name}...")

        evaluation = await benchmark_system.run_comprehensive_evaluation(agent)
        benchmark_system.print_detailed_report(evaluation)

        # Save results to file
        result_file = f"benchmark_results_{agent.name.lower().replace('-', '_')}.json"
        with open(result_file, 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            serializable_result = {
                "agent_name": evaluation["result"].agent_name,
                "overall_score": evaluation["result"].overall_score,
                "execution_time": evaluation["result"].execution_time,
                "timestamp": evaluation["result"].timestamp.isoformat(),
                "task_results": [
                    {
                        "task_id": r.task_id,
                        "actual_answer": r.actual_answer,
                        "expected_answer": r.expected_answer,
                        "score": r.score,
                        "success": r.success,
                        "execution_time": r.execution_time,
                    }
                    for r in evaluation["result"].task_results
                ],
                "analysis": evaluation["analysis"],
            }
            json.dump(serializable_result, f, indent=2)

        print(f" Results saved to {result_file}")

    print("\nSUCCESS Enhanced benchmark demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
