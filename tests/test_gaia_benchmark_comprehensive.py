#!/usr/bin/env python3
"""Comprehensive tests for llamaagent.benchmarks.gaia_benchmark module."""

import json
import time
from unittest.mock import mock_open, patch

import pytest

from llamaagent.benchmarks.gaia_benchmark import GAIABenchmark, GAIATask


class TestGAIATask:
    """Test GAIATask dataclass validation."""

    def test_valid_task_creation(self):
        """Test creating a valid GAIA task."""
        task = GAIATask(
            task_id="test_001",
            question="What is 2+2?",
            expected_answer="4",
            difficulty="easy",
            steps_required=1,
            domain="math",
        )

        assert task.task_id == "test_001"
        assert task.question == "What is 2+2?"
        assert task.expected_answer == "4"
        assert task.difficulty == "easy"
        assert task.steps_required == 1
        assert task.domain == "math"

    def test_invalid_difficulty_raises_error(self):
        """Test that invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="Invalid difficulty: invalid"):
            GAIATask(
                task_id="test_001",
                question="What is 2+2?",
                expected_answer="4",
                difficulty="invalid",
                steps_required=1,
                domain="math",
            )

    def test_invalid_steps_required_raises_error(self):
        """Test that invalid steps_required raises ValueError."""
        with pytest.raises(ValueError, match="Invalid steps_required: 0"):
            GAIATask(
                task_id="test_001",
                question="What is 2+2?",
                expected_answer="4",
                difficulty="easy",
                steps_required=0,
                domain="math",
            )


class TestGAIABenchmark:
    """Test GAIABenchmark functionality."""

    def test_init_with_existing_file(self):
        """Test initialization when data file exists."""
        mock_data = {
            "version": "1.0",
            "created": time.time(),
            "tasks": [
                {
                    "task_id": "test_001",
                    "question": "What is 2+2?",
                    "expected_answer": "4",
                    "difficulty": "easy",
                    "steps_required": 1,
                    "domain": "math",
                    "metadata": {},
                }
            ],
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
                benchmark = GAIABenchmark()

                assert len(benchmark.tasks) == 1
                assert benchmark.tasks[0].task_id == "test_001"

    def test_init_creates_synthetic_tasks_when_file_missing(self):
        """Test initialization creates synthetic tasks when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(GAIABenchmark, "_save_tasks") as mock_save:
                benchmark = GAIABenchmark()

                # Should have created synthetic tasks
                assert len(benchmark.tasks) > 0
                mock_save.assert_called_once()

    def test_save_tasks(self):
        """Test _save_tasks method."""
        benchmark = GAIABenchmark()
        # Create a simple task
        benchmark.tasks = [GAIATask("test_001", "What is 2+2?", "4", "easy", 1, "math")]

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("json.dump") as mock_json_dump:
                    benchmark._save_tasks()

                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    mock_file.assert_called_once()
                    mock_json_dump.assert_called_once()

                    # Check the data structure passed to json.dump
                    call_args = mock_json_dump.call_args[0]
                    data = call_args[0]
                    assert "version" in data
                    assert "created" in data
                    assert "tasks" in data
                    assert len(data["tasks"]) == 1

    def test_get_tasks_with_filters(self):
        """Test get_tasks with various filters."""
        # Create benchmark with known tasks
        benchmark = GAIABenchmark()
        benchmark.tasks = [
            GAIATask("easy_001", "Q1", "A1", "easy", 1, "math"),
            GAIATask("medium_001", "Q2", "A2", "medium", 3, "science"),
            GAIATask("hard_001", "Q3", "A3", "hard", 5, "logic"),
            GAIATask("easy_002", "Q4", "A4", "easy", 2, "math"),
        ]

        # Test difficulty filter
        easy_tasks = benchmark.get_tasks(difficulty="easy")
        assert len(easy_tasks) == 2
        assert all(task.difficulty == "easy" for task in easy_tasks)

        # Test domain filter
        math_tasks = benchmark.get_tasks(domain="math")
        assert len(math_tasks) == 2
        assert all(task.domain == "math" for task in math_tasks)

        # Test min_steps filter
        complex_tasks = benchmark.get_tasks(min_steps=3)
        assert len(complex_tasks) == 2
        assert all(task.steps_required >= 3 for task in complex_tasks)

        # Test limit filter
        limited_tasks = benchmark.get_tasks(limit=2)
        assert len(limited_tasks) == 2

        # Test combined filters
        filtered_tasks = benchmark.get_tasks(difficulty="easy", domain="math", limit=1)
        assert len(filtered_tasks) == 1
        assert filtered_tasks[0].difficulty == "easy"
        assert filtered_tasks[0].domain == "math"

    def test_get_task_by_id(self):
        """Test get_task_by_id method."""
        benchmark = GAIABenchmark()
        benchmark.tasks = [
            GAIATask("test_001", "Q1", "A1", "easy", 1, "math"),
            GAIATask("test_002", "Q2", "A2", "medium", 2, "science"),
        ]

        # Test existing task
        task = benchmark.get_task_by_id("test_001")
        assert task is not None
        assert task.question == "Q1"

        # Test non-existing task
        task = benchmark.get_task_by_id("non_existent")
        assert task is None

    def test_get_stats(self):
        """Test get_stats method."""
        benchmark = GAIABenchmark()
        benchmark.tasks = [
            GAIATask("easy_001", "Q1", "A1", "easy", 1, "math"),
            GAIATask("easy_002", "Q2", "A2", "easy", 2, "math"),
            GAIATask("medium_001", "Q3", "A3", "medium", 3, "science"),
            GAIATask("hard_001", "Q4", "A4", "hard", 5, "logic"),
        ]

        stats = benchmark.get_stats()

        assert stats["total_tasks"] == 4
        assert stats["difficulties"]["easy"] == 2
        assert stats["difficulties"]["medium"] == 1
        assert stats["difficulties"]["hard"] == 1
        assert stats["domains"]["math"] == 2
        assert stats["domains"]["science"] == 1
        assert stats["domains"]["logic"] == 1
        assert stats["avg_steps"] == 2.75  # (1+2+3+5)/4
        assert stats["min_steps"] == 1
        assert stats["max_steps"] == 5

    def test_create_synthetic_tasks_coverage(self):
        """Test that _create_synthetic_tasks creates expected task types."""
        benchmark = GAIABenchmark()
        tasks = benchmark._create_synthetic_tasks()

        # Should have tasks of different difficulties
        difficulties = set(task.difficulty for task in tasks)
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

        # Should have tasks of different domains
        domains = set(task.domain for task in tasks)
        assert len(domains) > 1  # Multiple domains

        # Should have tasks with different step requirements
        step_counts = set(task.steps_required for task in tasks)
        assert len(step_counts) > 1  # Various step counts

        # All tasks should have valid IDs and non-empty questions/answers
        for task in tasks:
            assert task.task_id
            assert task.question
            assert task.expected_answer
            assert task.difficulty in ["easy", "medium", "hard"]
            assert task.steps_required >= 1
