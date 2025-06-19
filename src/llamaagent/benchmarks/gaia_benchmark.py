from __future__ import annotations

"""GAIA benchmark integration for LlamaAgent evaluation."""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None

from ..agents import AgentConfig, AgentResponse, ReactAgent
from ..tools import ToolRegistry

# ─────────────────────────────── GAIA integration ───────────────────────────

logger = logging.getLogger(__name__)


@dataclass
class GAIATask:
    """Individual GAIA task."""
    
    task_id: str
    question: str
    level: int
    final_answer: str
    file_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GAIAResult:
    """Result from GAIA evaluation."""
    
    task_id: str
    question: str
    level: int
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    agent_response: AgentResponse
    execution_time: float
    tokens_used: int


class GAIABenchmark:
    """GAIA benchmark evaluator with Hugging Face dataset integration."""

    def __init__(self, subset: str = "validation", max_tasks: Optional[int] = None):
        """Initialize GAIA benchmark.
        
        Args:
            subset: Dataset subset to use ("validation" or "test")
            max_tasks: Maximum number of tasks to evaluate (None for all)
        """
        self.subset = subset
        self.max_tasks = max_tasks
        self.tasks: List[GAIATask] = []
        
    async def load_dataset(self) -> None:
        """Load GAIA dataset from Hugging Face."""
        try:
            # Try to import datasets library
            if load_dataset is None:
                raise ImportError("datasets library not available")
            
            logger.info(f"Loading GAIA dataset subset: {self.subset}")
            
            # Load the official GAIA dataset
            dataset = load_dataset("gaia-benchmark/GAIA", self.subset)
            
            tasks = []
            for item in dataset:
                task = GAIATask(
                    task_id=item.get("task_id", f"gaia_{len(tasks)}"),
                    question=item["Question"],
                    level=item["Level"],
                    final_answer=item.get("Final answer", ""),
                    file_name=item.get("file_name"),
                    metadata=item.get("Annotator Metadata", {})
                )
                tasks.append(task)
                
                if self.max_tasks and len(tasks) >= self.max_tasks:
                    break
                    
            self.tasks = tasks
            logger.info(f"Loaded {len(self.tasks)} GAIA tasks")
            
        except ImportError:
            logger.warning("datasets library not available, using fallback data")
            await self._load_fallback_data()
        except Exception as e:
            logger.error(f"Failed to load GAIA dataset: {e}")
            await self._load_fallback_data()
    
    async def _load_fallback_data(self) -> None:
        """Load fallback GAIA-style tasks when HF datasets unavailable."""
        fallback_tasks = [
            {
                "task_id": "gaia_math_001", 
                "question": "Calculate the compound interest on $5000 at 8% annual rate for 5 years, then write a Python function to calculate compound interest for any inputs.",
                "level": 2,
                "final_answer": "$7346.64"
            },
            {
                "task_id": "gaia_reasoning_001",
                "question": "If a train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours, what is the average speed for the entire journey?",
                "level": 1, 
                "final_answer": "68 mph"
            },
            {
                "task_id": "gaia_code_001",
                "question": "Write a Python function that finds the longest palindromic substring in a given string. Test it with 'babad' and return the result.",
                "level": 2,
                "final_answer": "bab"
            },
            {
                "task_id": "gaia_multi_001",
                "question": "Calculate the factorial of 8, then find what percentage 8! represents of 10!. Express as a percentage rounded to 2 decimal places.",
                "level": 2,
                "final_answer": "1.11%"
            }
        ]
        
        self.tasks = [
            GAIATask(
                task_id=task["task_id"],
                question=task["question"], 
                level=task["level"],
                final_answer=task["final_answer"]
            )
            for task in fallback_tasks[:self.max_tasks] if self.max_tasks else fallback_tasks
        ]
        
        logger.info(f"Loaded {len(self.tasks)} fallback GAIA tasks")

    async def evaluate_agent(self, agent: ReactAgent, shuffle: bool = True) -> List[GAIAResult]:
        """Evaluate agent on GAIA tasks."""
        if not self.tasks:
            await self.load_dataset()
            
        tasks = self.tasks.copy()
        if shuffle:
            random.shuffle(tasks)
            
        results = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task.task_id}")
            
            try:
                # Execute task
                response = await agent.execute(task.question)
                
                # Extract predicted answer (last line or full content)
                predicted = response.content.strip().split('\n')[-1]
                
                # Simple answer matching (case-insensitive, stripped)
                is_correct = self._match_answers(predicted, task.final_answer)
                
                result = GAIAResult(
                    task_id=task.task_id,
                    question=task.question,
                    level=task.level,
                    predicted_answer=predicted,
                    correct_answer=task.final_answer,
                    is_correct=is_correct,
                    agent_response=response,
                    execution_time=response.execution_time,
                    tokens_used=response.tokens_used
                )
                
                results.append(result)
                
                logger.info(f"Task {task.task_id}: {'✓' if is_correct else '✗'} "
                          f"({response.execution_time:.2f}s, {response.tokens_used} tokens)")
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {task.task_id}: {e}")
                # Add failed result
                results.append(GAIAResult(
                    task_id=task.task_id,
                    question=task.question,
                    level=task.level,
                    predicted_answer=f"ERROR: {e}",
                    correct_answer=task.final_answer,
                    is_correct=False,
                    agent_response=AgentResponse(content=f"Error: {e}", success=False),
                    execution_time=0.0,
                    tokens_used=0
                ))
                
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
        level_stats = {}
        for level in [1, 2, 3]:
            level_results = [r for r in results if r.level == level]
            if level_results:
                level_correct = sum(1 for r in level_results if r.is_correct)
                level_stats[f"level_{level}"] = {
                    "total": len(level_results),
                    "correct": level_correct,
                    "accuracy": level_correct / len(level_results) if level_results else 0,
                    "avg_time": sum(r.execution_time for r in level_results) / len(level_results),
                    "avg_tokens": sum(r.tokens_used for r in level_results) / len(level_results)
                }
        
        return {
            "dataset": "GAIA",
            "subset": self.subset,
            "total_tasks": total,
            "correct_answers": correct,
            "overall_accuracy": correct / total if total > 0 else 0,
            "average_execution_time": sum(r.execution_time for r in results) / total if total > 0 else 0,
            "total_tokens_used": sum(r.tokens_used for r in results),
            "level_breakdown": level_stats,
            "failed_tasks": [
                {"task_id": r.task_id, "question": r.question[:100] + "...", "error": r.predicted_answer}
                for r in results if not r.is_correct and r.predicted_answer.startswith("ERROR:")
            ]
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
                    "agent_success": r.agent_response.success
                }
                for r in results
            ],
            "summary": self.generate_report(results)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")


# ─────────────────────────────── convenience functions ──────────────────────

async def run_gaia_evaluation(
    agent_config: AgentConfig,
    tools: ToolRegistry,
    subset: str = "validation",
    max_tasks: Optional[int] = 20,
    output_dir: Optional[Path] = None
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
        output_path = output_dir / f"gaia_{subset}_{agent_config.name.lower()}_results.json"
        await benchmark.save_results(results, output_path)
    
    # Return summary report
    return benchmark.generate_report(results)


# Legacy compatibility
async def generate_tasks(
    categories: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None, 
    count: int = 10
) -> List[Dict[str, Any]]:
    """Generate GAIA-style tasks (legacy compatibility)."""
    benchmark = GAIABenchmark(max_tasks=count)
    await benchmark.load_dataset()
    
    return [
        {
            "id": task.task_id,
            "question": task.question,
            "level": task.level,
            "answer": task.final_answer,
            "metadata": task.metadata
        }
        for task in benchmark.tasks
    ]
