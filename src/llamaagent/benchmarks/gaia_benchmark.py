"""GAIA benchmark integration for LlamaAgent evaluation."""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset
except (ImportError, ModuleNotFoundError):
    load_dataset = None

from ..agents import AgentConfig, AgentResponse, ReactAgent
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class GAIATask:
    """Individual GAIA task."""

    task_id: str
    question: str
    expected_answer: str
    difficulty: str  # "easy", "medium", "hard"
    steps_required: int
    domain: str
    file_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task parameters."""
        if self.difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Invalid difficulty: {self.difficulty}")
        if self.steps_required < 1:
            raise ValueError(f"Invalid steps_required: {self.steps_required}")


@dataclass
class GAIAResult:
    """Result from GAIA evaluation."""

    task_id: str
    question: str
    level: int  # 1=easy, 2=medium, 3=hard
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    agent_response: AgentResponse
    execution_time: float
    tokens_used: int


class GAIABenchmark:
    """GAIA benchmark evaluator with Hugging Face dataset integration."""

    def __init__(
        self,
        subset: str = "validation",
        max_tasks: Optional[int] = None,
        data_file: Optional[Path] = None,
    ):
        """Initialize GAIA benchmark.

        Args:
            subset: Dataset subset to use ("validation" or "test")
            max_tasks: Maximum number of tasks to evaluate (None for all)
            data_file: Optional path to existing data file
        """
        self.subset = subset
        self.max_tasks = max_tasks
        self.data_file = data_file or Path("benchmark_data") / "gaia_tasks.json"
        self.tasks: List[GAIATask] = []

        # Initialize tasks
        self._initialize_tasks()

    def _initialize_tasks(self) -> None:
        """Initialize tasks from file or create synthetic ones."""
        if self.data_file.exists():
            self._load_tasks_from_file()
        else:
            self.tasks = self._create_synthetic_tasks()
            self._save_tasks()

    def _load_tasks_from_file(self) -> None:
        """Load tasks from JSON file."""
        with open(self.data_file, "r") as f:
            data = json.load(f)
        self.tasks = []
        for task_data in data.get("tasks", []):
            task = GAIATask(
                task_id=task_data["task_id"],
                question=task_data["question"],
                expected_answer=task_data["expected_answer"],
                difficulty=task_data["difficulty"],
                steps_required=task_data["steps_required"],
                domain=task_data["domain"],
                file_name=task_data.get("file_name"),
                metadata=task_data.get("metadata", {}),
            )
            self.tasks.append(task)

    def _save_tasks(self) -> None:
        """Save tasks to JSON file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "subset": self.subset,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "expected_answer": task.expected_answer,
                    "difficulty": task.difficulty,
                    "steps_required": task.steps_required,
                    "domain": task.domain,
                    "file_name": task.file_name,
                    "metadata": task.metadata,
                }
                for task in self.tasks
            ],
        }

        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    def _create_synthetic_tasks(self) -> List[GAIATask]:
        """Create synthetic GAIA-style tasks."""
        tasks: List[GAIATask] = []

        # Mathematical reasoning tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="math_easy_001",
                    question="Calculate 15% of 240 and then add 30 to the result.",
                    expected_answer="66",
                    difficulty="easy",
                    steps_required=2,
                    domain="mathematics",
                ),
                GAIATask(
                    task_id="math_medium_001",
                    question="If a rectangle has length 8 cm and width 5 cm, what is its perimeter?",
                    expected_answer="26 cm",
                    difficulty="medium",
                    steps_required=3,
                    domain="mathematics",
                ),
                GAIATask(
                    task_id="math_hard_001",
                    question="Calculate the compound interest on $5000 at 8% annual rate for 3 years, compounded annually.",
                    expected_answer="$6298.56",
                    difficulty="hard",
                    steps_required=4,
                    domain="mathematics",
                ),
            ]
        )

        # Logic tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="logic_easy_001",
                    question="If all cats are animals and Fluffy is a cat, what can we conclude about Fluffy?",
                    expected_answer="Fluffy is an animal",
                    difficulty="easy",
                    steps_required=1,
                    domain="logic",
                ),
                GAIATask(
                    task_id="logic_medium_001",
                    question="In a group of 30 people, 15 like tea, 18 like coffee, and 8 like both. How many like neither?",
                    expected_answer="5",
                    difficulty="medium",
                    steps_required=3,
                    domain="logic",
                ),
                GAIATask(
                    task_id="logic_hard_001",
                    question="Solve: If A is true, then B is false. If B is false, then C is true. If C is true, then D is false. Given A is true, what is D?",
                    expected_answer="D is false",
                    difficulty="hard",
                    steps_required=4,
                    domain="logic",
                ),
            ]
        )

        # Programming tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="prog_easy_001",
                    question="Write a Python function that returns the maximum of two numbers.",
                    expected_answer="def max_two(a, b): return a if a > b else b",
                    difficulty="easy",
                    steps_required=1,
                    domain="programming",
                ),
                GAIATask(
                    task_id="prog_medium_001",
                    question="Write a Python function to find the factorial of a number using recursion.",
                    expected_answer="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                    difficulty="medium",
                    steps_required=2,
                    domain="programming",
                ),
                GAIATask(
                    task_id="prog_hard_001",
                    question="Implement a binary search algorithm and test it with array [1, 3, 5, 7, 9, 11] searching for 7.",
                    expected_answer="3",
                    difficulty="hard",
                    steps_required=5,
                    domain="programming",
                ),
            ]
        )

        # Science tasks
        tasks.extend(
            [
                GAIATask(
                    task_id="sci_easy_001",
                    question="What is the chemical formula for water?",
                    expected_answer="H2O",
                    difficulty="easy",
                    steps_required=1,
                    domain="science",
                ),
                GAIATask(
                    task_id="sci_medium_001",
                    question="Calculate the kinetic energy of a 2kg object moving at 10 m/s.",
                    expected_answer="100 J",
                    difficulty="medium",
                    steps_required=2,
                    domain="science",
                ),
                GAIATask(
                    task_id="sci_hard_001",
                    question="Explain the process of photosynthesis and write the balanced chemical equation.",
                    expected_answer="6CO2 + 6H2O + light energy → C6H12O6 + 6O2",
                    difficulty="hard",
                    steps_required=3,
                    domain="science",
                ),
            ]
        )

        # Mathematical analysis
        tasks.append(
            GAIATask(
                task_id="math_easy_002",
                question="Find the derivative of f(x) = 3x³ - 2x² + 5x - 1, then evaluate it at x = 2.",
                expected_answer="37",
                difficulty="easy",
                steps_required=2,
                domain="mathematics",
            )
        )

        # Apply max_tasks limit if specified
        if self.max_tasks and len(tasks) > self.max_tasks:
            tasks = tasks[: self.max_tasks]

        return tasks

    def get_tasks(
        self,
        difficulty: Optional[str] = None,
        domain: Optional[str] = None,
        min_steps: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[GAIATask]:
        """Get tasks with optional filtering."""
        filtered_tasks = self.tasks.copy()

        if difficulty:
            filtered_tasks = [t for t in filtered_tasks if t.difficulty == difficulty]

        if domain:
            filtered_tasks = [t for t in filtered_tasks if t.domain == domain]

        if min_steps is not None:
            filtered_tasks = [
                t for t in filtered_tasks if t.steps_required >= min_steps
            ]

        if limit:
            filtered_tasks = filtered_tasks[:limit]

        return filtered_tasks

    def get_task_by_id(self, task_id: str) -> Optional[GAIATask]:
        """Get a specific task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self.tasks:
            return {"total_tasks": 0}

        difficulties: Dict[str, int] = {}
        domains: Dict[str, int] = {}
        step_counts: List[int] = []

        for task in self.tasks:
            difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1
            domains[task.domain] = domains.get(task.domain, 0) + 1
            step_counts.append(task.steps_required)
        return {
            "total_tasks": len(self.tasks),
            "difficulties": difficulties,
            "domains": domains,
            "avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
            "min_steps": min(step_counts) if step_counts else 0,
            "max_steps": max(step_counts) if step_counts else 0,
        }

    async def load_dataset(self) -> None:
        """Load GAIA dataset from Hugging Face."""
        try:
            # Try to import datasets library
            if load_dataset is None:
                raise ImportError("datasets library not available")
            logger.info(f"Loading GAIA dataset subset: {self.subset}")
            # Load the official GAIA dataset
            dataset = load_dataset("gaia-benchmark/GAIA", self.subset)
            tasks: List[GAIATask] = []

            for item in dataset:
                # Map GAIA format to our format
                difficulty_map = {1: "easy", 2: "medium", 3: "hard"}

                # Type check item as dictionary
                if not isinstance(item, dict):
                    continue

                task = GAIATask(
                    task_id=item.get("task_id", f"gaia_{len(tasks)}"),
                    question=item.get("Question", ""),
                    expected_answer=item.get("Final answer", ""),
                    difficulty=difficulty_map.get(item.get("Level", 2), "medium"),
                    steps_required=item.get(
                        "Level", 2
                    ),  # Use level as steps approximation
                    domain="general",  # GAIA doesn't have explicit domains
                    file_name=item.get("file_name"),
                    metadata=item.get("Annotator Metadata", {}),
                )
                tasks.append(task)
                if self.max_tasks and len(tasks) >= self.max_tasks:
                    break

            self.tasks = tasks
            logger.info(f"Loaded {len(self.tasks)} GAIA tasks")

        except ImportError:
            logger.warning("datasets library not available, using fallback data")
            # Tasks already initialized in __init__
        except Exception as e:
            logger.error(f"Failed to load GAIA dataset: {e}")
            # Tasks already initialized in __init__

    async def evaluate_agent(
        self, agent: ReactAgent, shuffle: bool = True
    ) -> List[GAIAResult]:
        """Evaluate agent on GAIA tasks."""
        if not self.tasks:
            await self.load_dataset()

        tasks = self.tasks.copy()
        if shuffle:
            random.shuffle(tasks)
        results: List[GAIAResult] = []

        for i, task in enumerate(tasks):
            logger.info(f"Evaluating task {i + 1}/{len(tasks)}: {task.task_id}")

            try:
                # Execute task
                response = await agent.execute(task.question)
                # Extract predicted answer (last line or full content)
                predicted = response.content.strip().split("\n")[-1]

                # Simple answer matching (case-insensitive, stripped)
                is_correct = self._match_answers(predicted, task.expected_answer)
                # Map difficulty to level for backward compatibility
                level_map = {"easy": 1, "medium": 2, "hard": 3}

                result = GAIAResult(
                    task_id=task.task_id,
                    question=task.question,
                    level=level_map.get(task.difficulty, 2),
                    predicted_answer=predicted,
                    correct_answer=task.expected_answer,
                    is_correct=is_correct,
                    agent_response=response,
                    execution_time=response.execution_time,
                    tokens_used=response.tokens_used,
                )

                results.append(result)
                logger.info(
                    f"Task {task.task_id}: {'PASS' if is_correct else 'FAIL'} "
                    f"({response.execution_time:.2f}s, {response.tokens_used} tokens)"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate task {task.task_id}: {e}")
                # Add failed result
                results.append(
                    GAIAResult(
                        task_id=task.task_id,
                        question=task.question,
                        level=1,
                        predicted_answer=f"ERROR: {e}",
                        correct_answer=task.expected_answer,
                        is_correct=False,
                        agent_response=AgentResponse(
                            content=f"Error: {e}", success=False
                        ),
                        execution_time=0.0,
                        tokens_used=0,
                    )
                )

        return results

    def _match_answers(self, predicted: str, correct: str) -> bool:
        """Match predicted answer with correct answer."""
        # Normalize both answers
        pred_norm = predicted.lower().strip().replace(",", "").replace("$", "")
        correct_norm = correct.lower().strip().replace(",", "").replace("$", "")
        # Exact match
        if pred_norm == correct_norm:
            return True

        # Check if predicted contains correct answer
        if correct_norm in pred_norm:
            return True

        # For numeric answers, try parsing
        try:
            pred_num = float(pred_norm.replace("%", ""))
            correct_num = float(correct_norm.replace("%", ""))
            return abs(pred_num - correct_num) < 0.01
        except (ValueError, TypeError):
            pass

        return False

    def generate_report(self, results: List[GAIAResult]) -> Dict[str, Any]:
        """Generate evaluation report."""
        if not results:
            return {"error": "No results to report"}

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        # Level-wise breakdown
        level_stats: Dict[str, Any] = {}
        for level in [1, 2, 3]:
            level_results = [r for r in results if r.level == level]
            if level_results:
                level_correct = sum(1 for r in level_results if r.is_correct)
                level_stats[f"level_{level}"] = {
                    "total": len(level_results),
                    "correct": level_correct,
                    "accuracy": (
                        level_correct / len(level_results) if level_results else 0
                    ),
                    "avg_time": sum(r.execution_time for r in level_results)
                    / len(level_results),
                    "avg_tokens": sum(r.tokens_used for r in level_results)
                    / len(level_results),
                }

        return {
            "dataset": "GAIA",
            "subset": self.subset,
            "total_tasks": total,
            "correct_answers": correct,
            "overall_accuracy": correct / total if total > 0 else 0,
            "average_execution_time": (
                sum(r.execution_time for r in results) / total if total > 0 else 0
            ),
            "total_tokens_used": sum(r.tokens_used for r in results),
            "level_breakdown": level_stats,
            "failed_tasks": [
                {
                    "task_id": r.task_id,
                    "question": r.question[:100] + "...",
                    "error": r.predicted_answer,
                }
                for r in results
                if not r.is_correct and r.predicted_answer.startswith("ERROR:")
            ],
            "summary": self.generate_report(results),
            "created": time.time(),
        }

    async def save_results(self, results: List[GAIAResult], output_path: Path) -> None:
        """Save results to JSON file."""
        output_data = {
            "benchmark": "GAIA",
            "subset": self.subset,
            "results": [
                {
                    "task_id": r.task_id,
                    "question": r.question,
                    "level": r.level,
                    "predicted_answer": r.predicted_answer,
                    "correct_answer": r.correct_answer,
                    "is_correct": r.is_correct,
                    "execution_time": r.execution_time,
                    "tokens_used": r.tokens_used,
                    "agent_success": r.agent_response.success,
                }
                for r in results
            ],
            "summary": self.generate_report(results),
            "created": time.time(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")


#  convenience functions


async def run_gaia_evaluation(
    agent_config: AgentConfig,
    tools: ToolRegistry,
    subset: str = "validation",
    max_tasks: Optional[int] = 20,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run GAIA evaluation with given agent configuration."""

    # Create agent
    agent = ReactAgent(agent_config, tools=tools)
    # Create benchmark
    benchmark = GAIABenchmark(subset=subset, max_tasks=max_tasks)
    # Run evaluation
    results = await benchmark.evaluate_agent(agent)
    # Save results if output directory provided
    if output_dir:
        output_path = (
            output_dir / f"gaia_{subset}_{agent_config.name.lower()}_results.json"
        )
        await benchmark.save_results(results, output_path)
    # Return summary report
    return benchmark.generate_report(results)


# Legacy compatibility
async def generate_tasks(
    categories: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    count: int = 10,
) -> List[Dict[str, Any]]:
    """Generate GAIA-style tasks (legacy compatibility)."""
    benchmark = GAIABenchmark(max_tasks=count)
    await benchmark.load_dataset()

    return [
        {
            "id": task.task_id,
            "question": task.question,
            "answer": task.expected_answer,
            "difficulty": task.difficulty,
            "steps_required": task.steps_required,
            "domain": task.domain,
            "metadata": task.metadata,
        }
        for task in benchmark.tasks
    ]
