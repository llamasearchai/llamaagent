from __future__ import annotations

"""GAIA benchmark implementation for multi-step reasoning evaluation.

Based on the GAIA benchmark from WebDancer paper (arXiv:2505.22648v1), this
module provides a comprehensive evaluation framework for testing agent
performance on complex, multi-step reasoning tasks.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["GAIATask", "GAIABenchmark"]


@dataclass
class GAIATask:
    """Individual GAIA benchmark task."""

    task_id: str
    question: str
    expected_answer: str
    difficulty: str  # "easy", "medium", "hard"
    steps_required: int
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task data."""
        if self.difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid difficulty: {self.difficulty}")
        if self.steps_required < 1:
            raise ValueError(f"Invalid steps_required: {self.steps_required}")


class GAIABenchmark:
    """GAIA benchmark dataset manager and evaluator."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path(__file__).parent / "data" / "gaia_tasks.json"
        self.tasks: List[GAIATask] = []
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load GAIA tasks from dataset."""
        if self.data_path.exists():
            with open(self.data_path, "r") as f:
                data = json.load(f)
                self.tasks = [GAIATask(**task) for task in data["tasks"]]
        else:
            # Create synthetic GAIA-style tasks for evaluation
            self.tasks = self._create_synthetic_tasks()
            self._save_tasks()

    def _create_synthetic_tasks(self) -> List[GAIATask]:
        """Create synthetic GAIA-style tasks for evaluation."""
        tasks = []

        # Mathematical reasoning tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="math_001",
                    question="Calculate the compound interest on $1000 invested at 5% annual rate for 3 years, then determine what percentage of the total amount the interest represents.",
                    expected_answer="The compound interest is $157.63, representing 13.6% of the total amount.",
                    difficulty="medium",
                    steps_required=3,
                    domain="mathematics",
                    metadata={"requires_calculation": True, "multi_step": True},
                ),
                GAIATask(
                    task_id="math_002",
                    question="Find the area of a triangle with vertices at (0,0), (4,0), and (2,3), then calculate how many such triangles would fit in a rectangle of area 50.",
                    expected_answer="The triangle area is 6 square units. 8 triangles would fit in the rectangle (with 2 square units remaining).",
                    difficulty="medium",
                    steps_required=4,
                    domain="mathematics",
                    metadata={"requires_calculation": True, "geometry": True},
                ),
            ]
        )

        # Programming tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="prog_001",
                    question="Write a Python function to find the longest palindromic substring in a given string, then test it with 'racecar' and explain the algorithm's time complexity.",
                    expected_answer="Function finds 'racecar' as the longest palindrome. Time complexity is O(n²) for the expand-around-centers approach.",
                    difficulty="hard",
                    steps_required=4,
                    domain="programming",
                    metadata={"requires_code": True, "algorithm_analysis": True},
                ),
                GAIATask(
                    task_id="prog_002",
                    question="Create a function that generates the first 10 Fibonacci numbers, then calculate their sum and determine what percentage the largest number represents of the total sum.",
                    expected_answer="Fibonacci sequence: [0,1,1,2,3,5,8,13,21,34]. Sum: 88. Largest (34) represents 38.6% of total.",
                    difficulty="medium",
                    steps_required=3,
                    domain="programming",
                    metadata={"requires_code": True, "mathematical_analysis": True},
                ),
            ]
        )

        # Multi-domain reasoning
        tasks.extend(
            [
                GAIATask(
                    task_id="multi_001",
                    question="If a company's revenue grows by 15% each quarter and starts at $100,000, what will be the revenue after 1 year? Then determine how much additional revenue they would need to reach $200,000 by year-end.",
                    expected_answer="After 1 year: $174,901. Additional revenue needed: $25,099.",
                    difficulty="medium",
                    steps_required=3,
                    domain="business_math",
                    metadata={"compound_growth": True, "business_reasoning": True},
                ),
                GAIATask(
                    task_id="multi_002",
                    question="Explain the concept of machine learning overfitting, provide a code example that demonstrates it, and suggest three practical solutions to prevent it.",
                    expected_answer="Overfitting occurs when models memorize training data. Code example shows high training accuracy but poor validation performance. Solutions: regularization, cross-validation, more data.",
                    difficulty="hard",
                    steps_required=5,
                    domain="machine_learning",
                    metadata={"conceptual_explanation": True, "requires_code": True, "solution_generation": True},
                ),
            ]
        )

        # Simple tasks for baseline testing
        tasks.extend(
            [
                GAIATask(
                    task_id="simple_001",
                    question="What is 25 * 16?",
                    expected_answer="400",
                    difficulty="easy",
                    steps_required=1,
                    domain="arithmetic",
                    metadata={"basic_math": True},
                ),
                GAIATask(
                    task_id="simple_002",
                    question="Calculate 144 / 12 and then add 7 to the result.",
                    expected_answer="19",
                    difficulty="easy",
                    steps_required=2,
                    domain="arithmetic",
                    metadata={"basic_math": True, "sequential_ops": True},
                ),
            ]
        )

        return tasks

    def _save_tasks(self) -> None:
        """Save tasks to file for reproducibility."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "created": time.time(),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "expected_answer": task.expected_answer,
                    "difficulty": task.difficulty,
                    "steps_required": task.steps_required,
                    "domain": task.domain,
                    "metadata": task.metadata,
                }
                for task in self.tasks
            ],
        }

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_tasks(
        self,
        difficulty: Optional[str] = None,
        domain: Optional[str] = None,
        min_steps: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[GAIATask]:
        """Get filtered tasks based on criteria."""
        filtered_tasks = self.tasks

        if difficulty:
            filtered_tasks = [t for t in filtered_tasks if t.difficulty == difficulty]

        if domain:
            filtered_tasks = [t for t in filtered_tasks if t.domain == domain]

        if min_steps:
            filtered_tasks = [t for t in filtered_tasks if t.steps_required >= min_steps]

        if limit:
            filtered_tasks = filtered_tasks[:limit]

        return filtered_tasks

    def get_task_by_id(self, task_id: str) -> Optional[GAIATask]:
        """Get specific task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        difficulties = {}
        domains = {}

        for task in self.tasks:
            difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1
            domains[task.domain] = domains.get(task.domain, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "difficulties": difficulties,
            "domains": domains,
            "avg_steps": sum(task.steps_required for task in self.tasks) / len(self.tasks),
            "max_steps": max(task.steps_required for task in self.tasks),
            "min_steps": min(task.steps_required for task in self.tasks),
        }
