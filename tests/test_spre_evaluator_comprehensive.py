#!/usr/bin/env python3
"""Comprehensive tests for llamaagent.benchmarks.spre_evaluator module."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from llamaagent.benchmarks.gaia_benchmark import GAIATask
from llamaagent.benchmarks.spre_evaluator import (
    BenchmarkResult,
    SPREEvaluator,
    TaskResult,
)


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test TaskResult creation with all fields."""
        result = TaskResult(
            task_id="test_001",
            question="What is 2+2?",
            expected_answer="4",
            actual_answer="4",
            success=True,
            execution_time=1.5,
            tokens_used=50,
            api_calls=2,
            reasoning_tokens=20,
            baseline_type="vanilla_react",
            trace=[{"step": 1}],
            error_message=None,
        )

        assert result.task_id == "test_001"
        assert result.question == "What is 2+2?"
        assert result.expected_answer == "4"
        assert result.actual_answer == "4"
        assert result.success
        assert result.execution_time == 1.5
        assert result.tokens_used == 50
        assert result.api_calls == 2
        assert result.reasoning_tokens == 20
        assert result.baseline_type == "vanilla_react"
        assert result.trace == [{"step": 1}]
        assert result.error_message is None

    def test_task_result_with_error(self):
        """Test TaskResult creation with error message."""
        result = TaskResult(
            task_id="test_002",
            question="Broken task",
            expected_answer="N/A",
            actual_answer="ERROR: Task failed",
            success=False,
            execution_time=0.1,
            tokens_used=0,
            api_calls=0,
            reasoning_tokens=0,
            baseline_type="test",
            error_message="Connection timeout",
        )

        assert not result.success
        assert result.error_message == "Connection timeout"


class TestBenchmarkResult:
    """Test BenchmarkResult class and its properties."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="Wrong",
                success=False,
                execution_time=2.0,
                tokens_used=20,
                api_calls=2,
                reasoning_tokens=10,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult(
            baseline_type="test_baseline",
            agent_name="TestAgent",
            task_results=task_results,
        )

        assert result.baseline_type == "test_baseline"
        assert result.agent_name == "TestAgent"
        assert len(result.task_results) == 2

    def test_success_rate_calculation(self):
        """Test success rate property calculation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_3",
                question="Q3",
                expected_answer="A3",
                actual_answer="Wrong",
                success=False,
                execution_time=1.5,
                tokens_used=15,
                api_calls=1,
                reasoning_tokens=7,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        assert result.success_rate == 50.0  # 1/2 * 100

    def test_success_rate_empty_results(self):
        """Test success rate with no task results."""
        result = BenchmarkResult("test", "TestAgent", [])
        assert result.success_rate == 0.0

    def test_avg_api_calls(self):
        """Test average API calls calculation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="A2",
                success=True,
                execution_time=2.0,
                tokens_used=20,
                api_calls=3,
                reasoning_tokens=10,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        assert result.avg_api_calls == 2.0  # (1 + 3) / 2

    def test_avg_api_calls_empty_results(self):
        """Test average API calls with no results."""
        result = BenchmarkResult("test", "TestAgent", [])
        assert result.avg_api_calls == 0.0

    def test_avg_latency(self):
        """Test average latency calculation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="A2",
                success=True,
                execution_time=3.0,
                tokens_used=20,
                api_calls=2,
                reasoning_tokens=10,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        assert result.avg_latency == 2.0  # (1.0 + 3.0) / 2

    def test_avg_latency_empty_results(self):
        """Test average latency with no results."""
        result = BenchmarkResult("test", "TestAgent", [])
        assert result.avg_latency == 0.0

    def test_avg_tokens(self):
        """Test average tokens calculation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="A2",
                success=True,
                execution_time=2.0,
                tokens_used=30,
                api_calls=2,
                reasoning_tokens=10,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        assert result.avg_tokens == 20.0  # (10 + 30) / 2

    def test_avg_tokens_empty_results(self):
        """Test average tokens with no results."""
        result = BenchmarkResult("test", "TestAgent", [])
        assert result.avg_tokens == 0.0

    def test_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=2,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="Wrong",
                success=False,
                execution_time=2.0,
                tokens_used=20,
                api_calls=4,
                reasoning_tokens=10,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        # Success rate: 50%, Avg API calls: 3, Efficiency ratio: 50/3 = 16.67
        assert abs(result.efficiency_ratio - 16.67) < 0.01

    def test_efficiency_ratio_zero_api_calls(self):
        """Test efficiency ratio when average API calls is zero."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=0,
                reasoning_tokens=5,
                baseline_type="test",
            )
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)
        assert (
            result.efficiency_ratio == 100.0
        )  # Should return success_rate when avg_api_calls is 0

    def test_get_stats_by_difficulty(self):
        """Test get_stats_by_difficulty method."""
        # Create mock benchmark data that matches what the method expects
        mock_benchmark_data = {
            "version": "1.0",
            "created": time.time(),
            "tasks": [
                {
                    "task_id": "task_1",
                    "question": "Q1",
                    "expected_answer": "A1",
                    "difficulty": "easy",
                    "steps_required": 1,
                    "domain": "math",
                    "metadata": {},
                },
                {
                    "task_id": "task_2",
                    "question": "Q2",
                    "expected_answer": "A2",
                    "difficulty": "medium",
                    "steps_required": 2,
                    "domain": "science",
                    "metadata": {},
                },
                {
                    "task_id": "task_3",
                    "question": "Q3",
                    "expected_answer": "A3",
                    "difficulty": "hard",
                    "steps_required": 3,
                    "domain": "logic",
                    "metadata": {},
                },
            ],
        }

        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_2",
                question="Q2",
                expected_answer="A2",
                actual_answer="Wrong",
                success=False,
                execution_time=2.0,
                tokens_used=20,
                api_calls=2,
                reasoning_tokens=10,
                baseline_type="test",
            ),
            TaskResult(
                task_id="task_3",
                question="Q3",
                expected_answer="A3",
                actual_answer="A3",
                success=True,
                execution_time=3.0,
                tokens_used=30,
                api_calls=3,
                reasoning_tokens=15,
                baseline_type="test",
            ),
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)

        # Mock the file system to return our test data
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data=json.dumps(mock_benchmark_data))
            ):
                stats = result.get_stats_by_difficulty()

                assert "easy" in stats
                assert "medium" in stats
                assert "hard" in stats

                # Easy task (task_1): 1 success out of 1 = 100%
                assert stats["easy"]["success_rate"] == 100.0
                assert stats["easy"]["count"] == 1

                # Medium task (task_2): 0 success out of 1 = 0%
                assert stats["medium"]["success_rate"] == 0.0
                assert stats["medium"]["count"] == 1

                # Hard task (task_3): 1 success out of 1 = 100%
                assert stats["hard"]["success_rate"] == 100.0
                assert stats["hard"]["count"] == 1

    def test_get_stats_by_difficulty_empty_category(self):
        """Test get_stats_by_difficulty with no tasks in a category."""
        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            )
        ]

        result = BenchmarkResult("test", "TestAgent", task_results)

        with patch(
            "llamaagent.benchmarks.spre_evaluator.GAIABenchmark"
        ) as mock_benchmark_class:
            mock_benchmark = Mock()
            mock_benchmark.get_task_by_id.return_value = None  # No task found
            mock_benchmark_class.return_value = mock_benchmark

            stats = result.get_stats_by_difficulty()

            # All categories should have zero stats since no tasks are found
            for difficulty in ["easy", "medium", "hard"]:
                assert stats[difficulty]["success_rate"] == 0.0
                assert stats[difficulty]["avg_api_calls"] == 0.0
                assert stats[difficulty]["avg_latency"] == 0.0
                assert stats[difficulty]["count"] == 0


class TestSPREEvaluator:
    """Test SPREEvaluator class."""

    def test_evaluator_initialization_default(self):
        """Test SPREEvaluator initialization with defaults."""
        evaluator = SPREEvaluator()

        assert evaluator.llm_provider is None
        assert evaluator.output_dir == Path("benchmark_results")
        assert evaluator.benchmark is not None

    def test_evaluator_initialization_custom(self):
        """Test SPREEvaluator initialization with custom parameters."""
        from llamaagent.llm import MockProvider

        provider = MockProvider()
        output_dir = Path("/tmp/test_results")

        evaluator = SPREEvaluator(llm_provider=provider, output_dir=output_dir)

        assert evaluator.llm_provider == provider
        assert evaluator.output_dir == output_dir

    @pytest.mark.asyncio
    async def test_run_full_evaluation_no_tasks(self):
        """Test run_full_evaluation when no tasks match filter."""
        evaluator = SPREEvaluator()

        with patch.object(evaluator.benchmark, "get_tasks", return_value=[]):
            with pytest.raises(
                ValueError, match="No tasks found matching filter criteria"
            ):
                await evaluator.run_full_evaluation()

    def test_count_api_calls(self):
        """Test _count_api_calls method."""
        evaluator = SPREEvaluator()

        trace = [
            {"type": "planner_response", "data": {}},
            {"type": "resource_assessment_detail", "data": {}},
            {"type": "other_event", "data": {}},
            {"type": "internal_execution", "data": {}},
            {"type": "synthesis_complete", "data": {}},
        ]

        count = evaluator._count_api_calls(trace)
        assert count == 4  # Should match 4 out of 5 events

    @pytest.mark.asyncio
    async def test_count_api_calls_empty_trace(self):
        """Test _count_api_calls with empty trace."""
        evaluator = SPREEvaluator()

        count = evaluator._count_api_calls([])
        assert count == 0

    def test_evaluate_answer_exact_match(self):
        """Test _evaluate_answer with exact match."""
        evaluator = SPREEvaluator()

        assert evaluator._evaluate_answer("42", "42")
        assert evaluator._evaluate_answer("hello world", "hello world")

    def test_evaluate_answer_case_insensitive(self):
        """Test _evaluate_answer with case differences."""
        evaluator = SPREEvaluator()

        assert evaluator._evaluate_answer("Hello", "hello")
        assert evaluator._evaluate_answer("WORLD", "world")

    def test_evaluate_answer_whitespace_normalization(self):
        """Test _evaluate_answer with whitespace differences."""
        evaluator = SPREEvaluator()

        # The actual implementation strips and lowercases but doesn't normalize whitespace
        assert evaluator._evaluate_answer("hello world", "hello world")
        assert evaluator._evaluate_answer("hello world", "  hello world  ")

    def test_evaluate_answer_numeric_extraction(self):
        """Test _evaluate_answer with numeric answers."""
        evaluator = SPREEvaluator()

        assert evaluator._evaluate_answer("42", "The answer is 42.")
        assert evaluator._evaluate_answer("3.14", "Pi is approximately 3.14159")
        assert evaluator._evaluate_answer("100", "The result equals 100%")

    def test_evaluate_answer_no_match(self):
        """Test _evaluate_answer with no match."""
        evaluator = SPREEvaluator()

        # 42 and 43 are close numbers but should be different enough to fail
        assert not evaluator._evaluate_answer("42", "100")
        assert not evaluator._evaluate_answer("hello", "goodbye")

    @pytest.mark.asyncio
    async def test_save_results(self):
        """Test _save_results method."""
        evaluator = SPREEvaluator(output_dir=Path("/tmp/test_benchmark"))

        results = {"test_baseline": BenchmarkResult("test_baseline", "TestAgent", [])}

        with patch("builtins.open", create=True) as mock_open:
            with patch("json.dump") as mock_json_dump:
                await evaluator._save_results(results)

                mock_open.assert_called()
                mock_json_dump.assert_called()

    def test_generate_comparison_report(self):
        """Test _generate_comparison_report method."""
        evaluator = SPREEvaluator(output_dir=Path("/tmp/test_benchmark"))

        task_results = [
            TaskResult(
                task_id="task_1",
                question="Q1",
                expected_answer="A1",
                actual_answer="A1",
                success=True,
                execution_time=1.0,
                tokens_used=10,
                api_calls=1,
                reasoning_tokens=5,
                baseline_type="test",
            )
        ]

        results = {
            "vanilla_react": BenchmarkResult("vanilla_react", "Agent1", task_results),
            "spre_full": BenchmarkResult("spre_full", "Agent2", task_results),
        }

        # Mock the BenchmarkResult.get_stats_by_difficulty method
        with patch.object(
            BenchmarkResult,
            "get_stats_by_difficulty",
            return_value={"easy": {"success_rate": 100.0, "count": 1}},
        ):
            with patch("builtins.open", create=True) as mock_open:
                evaluator._generate_comparison_report(results)
                mock_open.assert_called()

    @pytest.mark.asyncio
    async def test_run_single_baseline_evaluation(self):
        """Test run_single_baseline_evaluation method."""
        evaluator = SPREEvaluator()

        mock_tasks = [GAIATask("task_1", "What is 2+2?", "4", "easy", 1, "math")]

        with patch.object(evaluator.benchmark, "get_tasks", return_value=mock_tasks):
            with patch(
                "llamaagent.benchmarks.spre_evaluator.BaselineAgentFactory"
            ) as mock_factory:
                with patch.object(
                    evaluator, "_evaluate_agent_on_tasks"
                ) as mock_evaluate:
                    mock_agent = Mock()
                    mock_factory.create_agent.return_value = mock_agent

                    expected_result = BenchmarkResult("test_baseline", "TestAgent", [])
                    mock_evaluate.return_value = expected_result

                    result = await evaluator.run_single_baseline_evaluation(
                        "test_baseline"
                    )

                    assert result == expected_result
                    # The actual implementation doesn't pass name_suffix
                    mock_factory.create_agent.assert_called_once_with(
                        "test_baseline", evaluator.llm_provider
                    )

    @pytest.mark.asyncio
    async def test_evaluate_agent_on_tasks_success(self):
        """Test _evaluate_agent_on_tasks with successful execution."""
        evaluator = SPREEvaluator()

        mock_agent = Mock()
        from llamaagent.agents.base import AgentResponse

        mock_response = AgentResponse(
            content="4",
            success=True,
            tokens_used=10,
            trace=[{"type": "planner_response"}],
        )
        mock_agent.execute = AsyncMock(return_value=mock_response)
        mock_agent.config.name = "TestAgent"

        tasks = [GAIATask("task_1", "What is 2+2?", "4", "easy", 1, "math")]

        with patch.object(evaluator, "_evaluate_answer", return_value=True):
            result = await evaluator._evaluate_agent_on_tasks(
                mock_agent, tasks, "test_baseline"
            )

            assert result.baseline_type == "test_baseline"
            assert result.agent_name == "TestAgent"
            assert len(result.task_results) == 1
            assert result.task_results[0].success
            assert result.task_results[0].actual_answer == "4"

    @pytest.mark.asyncio
    async def test_evaluate_agent_on_tasks_exception(self):
        """Test _evaluate_agent_on_tasks when execution throws exception."""
        evaluator = SPREEvaluator()

        mock_agent = Mock()
        mock_agent.execute = AsyncMock(side_effect=Exception("Test error"))
        mock_agent.config.name = "TestAgent"

        tasks = [GAIATask("task_1", "What is 2+2?", "4", "easy", 1, "math")]

        result = await evaluator._evaluate_agent_on_tasks(
            mock_agent, tasks, "test_baseline"
        )

        assert len(result.task_results) == 1
        assert not result.task_results[0].success
        assert "ERROR: Test error" in result.task_results[0].actual_answer
        assert result.task_results[0].error_message == "Test error"

    @pytest.mark.asyncio
    async def test_run_full_evaluation_success(self):
        """Test run_full_evaluation method with successful execution."""
        evaluator = SPREEvaluator()

        mock_tasks = [GAIATask("task_1", "What is 2+2?", "4", "easy", 1, "math")]

        with patch.object(evaluator.benchmark, "get_tasks", return_value=mock_tasks):
            with patch(
                "llamaagent.benchmarks.spre_evaluator.BaselineAgentFactory"
            ) as mock_factory:
                with patch.object(
                    evaluator, "_evaluate_agent_on_tasks"
                ) as mock_evaluate:
                    with patch.object(evaluator, "_save_results") as mock_save:
                        with patch.object(
                            evaluator, "_generate_comparison_report"
                        ) as mock_report:
                            with patch("builtins.print"):  # Mock print statements
                                mock_agent = Mock()
                                mock_factory.create_agent.return_value = mock_agent
                                mock_factory.get_all_baseline_types.return_value = [
                                    "vanilla_react",
                                    "spre_full",
                                ]

                                # Create a result with task data that yields the expected metrics
                                task_results = [
                                    TaskResult(
                                        task_id="task_1",
                                        question="Q1",
                                        expected_answer="A1",
                                        actual_answer="A1",
                                        success=True,
                                        execution_time=1.2,
                                        tokens_used=10,
                                        api_calls=2,
                                        reasoning_tokens=5,
                                        baseline_type="test_baseline",
                                    )
                                ]
                                expected_result = BenchmarkResult(
                                    "test_baseline", "TestAgent", task_results
                                )
                                mock_evaluate.return_value = expected_result

                                results = await evaluator.run_full_evaluation()

                                assert len(results) == 2  # vanilla_react and spre_full
                                mock_save.assert_called_once()
                                mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_agent_on_tasks_with_print(self):
        """Test _evaluate_agent_on_tasks method that includes print statements."""
        evaluator = SPREEvaluator()

        mock_agent = Mock()
        from llamaagent.agents.base import AgentResponse

        mock_response = AgentResponse(
            content="4",
            success=True,
            tokens_used=10,
            trace=[{"type": "planner_response"}],
        )
        mock_agent.execute = AsyncMock(return_value=mock_response)
        mock_agent.config.name = "TestAgent"

        tasks = [GAIATask("task_1", "What is 2+2?", "4", "easy", 1, "math")]

        with patch.object(evaluator, "_evaluate_answer", return_value=True):
            with patch("builtins.print"):  # Mock the print statements
                result = await evaluator._evaluate_agent_on_tasks(
                    mock_agent, tasks, "test_baseline"
                )

                assert result.baseline_type == "test_baseline"
                assert result.agent_name == "TestAgent"
                assert len(result.task_results) == 1
                assert result.task_results[0].success
                assert result.task_results[0].actual_answer == "4"

    def test_evaluate_answer_value_error_handling(self):
        """Test _evaluate_answer with ValueError in numeric conversion."""
        evaluator = SPREEvaluator()

        # Test the ValueError exception handling path in numeric comparison
        with patch("re.findall") as mock_findall:
            # Mock to return non-numeric strings that will cause ValueError
            mock_findall.side_effect = [["invalid"], ["also_invalid"]]

            result = evaluator._evaluate_answer("42", "The answer is invalid")
            assert not result  # Should fall back to False when ValueError occurs
