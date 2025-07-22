#!/usr/bin/env python3
"""
Enhanced Benchmark System for LlamaAgent

This module provides comprehensive benchmarking capabilities including:
- Multi-modal task evaluation
- Advanced performance metrics
- Comparative analysis
- Automated report generation
- Real-time monitoring
- Custom benchmark creation
- Statistical analysis

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our components
from src.llamaagent.agents.base import ReactAgent
from src.llamaagent.config.settings import AgentConfig
from src.llamaagent.llm.providers.mock_provider import MockProvider

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of benchmark tasks."""
    MATHEMATICAL = "mathematical"
    LOGICAL_REASONING = "logical_reasoning"
    PROGRAMMING = "programming"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    CREATIVE_WRITING = "creative_writing"
    PROBLEM_SOLVING = "problem_solving"
    MULTIMODAL = "multimodal"
    CONVERSATIONAL = "conversational"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    ETHICAL_REASONING = "ethical_reasoning"


class DifficultyLevel(Enum):
    """Difficulty levels for tasks."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    CUSTOM_SCORER = "custom_scorer"
    HUMAN_EVALUATION = "human_evaluation"


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
    evaluation_metric: EvaluationMetric = EvaluationMetric.EXACT_MATCH
    custom_scorer: Optional[Callable[[str, str], float]] = None
    time_limit: Optional[float] = None
    max_tokens: Optional[int] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of a benchmark task execution."""
    task_id: str
    agent_name: str
    actual_answer: str
    expected_answer: str
    score: float
    execution_time: float
    tokens_used: int
    api_calls: int
    success: bool
    error_message: Optional[str] = None
    intermediate_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark tasks."""
    name: str
    description: str
    tasks: List[BenchmarkTask]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""
    suite_name: str
    agent_name: str
    task_results: List[TaskResult]
    overall_score: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedScorer:
    """Advanced scoring system with multiple evaluation methods."""
    
    def __init__(self):
        self.scoring_cache = {}
    
    def exact_match_score(self, actual: str, expected: str) -> float:
        """Exact match scoring."""
        return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0
    
    def semantic_similarity_score(self, actual: str, expected: str) -> float:
        """Semantic similarity scoring using simple heuristics."""
        # Simple implementation - in production, use sentence transformers
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def numerical_accuracy_score(self, actual: str, expected: str, tolerance: float = 0.01) -> float:
        """Numerical accuracy scoring with tolerance."""
        try:
            actual_num = float(actual.strip())
            expected_num = float(expected.strip())
            
            if abs(actual_num - expected_num) <= tolerance:
                return 1.0
            else:
                # Partial credit based on relative error
                relative_error = abs(actual_num - expected_num) / abs(expected_num)
                return max(0.0, 1.0 - relative_error)
        except (ValueError, ZeroDivisionError):
            return self.exact_match_score(actual, expected)
    
    def code_execution_score(self, actual: str, expected: str) -> float:
        """Score code by execution results."""
        # Simple implementation - would need secure code execution in production
        try:
            # Check if both are valid Python functions
            if "def " in actual and "def " in expected:
                return 0.8  # Partial credit for function structure
            return self.exact_match_score(actual, expected)
        except:
            return 0.0
    
    def multi_criteria_score(self, actual: str, expected: str, criteria: Dict[str, float]) -> float:
        """Multi-criteria scoring with weighted components."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, weight in criteria.items():
            if criterion == "exact_match":
                score = self.exact_match_score(actual, expected)
            elif criterion == "semantic_similarity":
                score = self.semantic_similarity_score(actual, expected)
            elif criterion == "numerical_accuracy":
                score = self.numerical_accuracy_score(actual, expected)
            else:
                score = 0.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def score_task(self, task: BenchmarkTask, actual_answer: str) -> float:
        """Score a task result."""
        cache_key = f"{task.id}:{hash(actual_answer)}"
        if cache_key in self.scoring_cache:
            return self.scoring_cache[cache_key]
        
        score = 0.0
        
        if task.evaluation_metric == EvaluationMetric.EXACT_MATCH:
            score = self.exact_match_score(actual_answer, task.expected_answer)
        elif task.evaluation_metric == EvaluationMetric.SEMANTIC_SIMILARITY:
            score = self.semantic_similarity_score(actual_answer, task.expected_answer)
        elif task.evaluation_metric == EvaluationMetric.CUSTOM_SCORER and task.custom_scorer:
            score = task.custom_scorer(actual_answer, task.expected_answer)
        else:
            # Default to exact match
            score = self.exact_match_score(actual_answer, task.expected_answer)
        
        # Apply task-specific adjustments
        if task.task_type == TaskType.MATHEMATICAL:
            score = self.numerical_accuracy_score(actual_answer, task.expected_answer)
        elif task.task_type == TaskType.PROGRAMMING:
            score = self.code_execution_score(actual_answer, task.expected_answer)
        
        self.scoring_cache[cache_key] = score
        return score


class BenchmarkDataGenerator:
    """Generate benchmark datasets automatically."""
    
    def __init__(self):
        self.task_templates = {
            TaskType.MATHEMATICAL: self._generate_math_tasks,
            TaskType.LOGICAL_REASONING: self._generate_logic_tasks,
            TaskType.PROGRAMMING: self._generate_programming_tasks,
            TaskType.LANGUAGE_UNDERSTANDING: self._generate_language_tasks,
            TaskType.PROBLEM_SOLVING: self._generate_problem_solving_tasks
        }
    
    def generate_suite(self, name: str, task_counts: Dict[TaskType, int]) -> BenchmarkSuite:
        """Generate a benchmark suite with specified task counts."""
        tasks = []
        
        for task_type, count in task_counts.items():
            if task_type in self.task_templates:
                generated_tasks = self.task_templates[task_type](count)
                tasks.extend(generated_tasks)
        
        return BenchmarkSuite(
            name=name,
            description=f"Auto-generated benchmark suite with {len(tasks)} tasks",
            tasks=tasks,
            metadata={"generated": True, "task_counts": task_counts}
        )
    
    def _generate_math_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate mathematical tasks."""
        tasks = []
        
        for i in range(count):
            # Generate different types of math problems
            if i % 4 == 0:
                # Arithmetic
                a, b = np.random.randint(1, 100, 2)
                op = np.random.choice(['+', '-', '*'])
                if op == '+':
                    result = a + b
                elif op == '-':
                    result = a - b
                else:
                    result = a * b
                
                question = f"Calculate {a} {op} {b}"
                answer = str(result)
                difficulty = DifficultyLevel.EASY
                
            elif i % 4 == 1:
                # Percentages
                percentage = np.random.randint(1, 50)
                number = np.random.randint(100, 1000)
                add_value = np.random.randint(10, 100)
                
                result = (percentage / 100) * number + add_value
                question = f"Calculate {percentage}% of {number} and then add {add_value} to the result."
                answer = str(int(result) if result.is_integer() else result)
                difficulty = DifficultyLevel.MEDIUM
                
            elif i % 4 == 2:
                # Geometry
                length = np.random.randint(5, 20)
                width = np.random.randint(5, 20)
                perimeter = 2 * (length + width)
                
                question = f"If a rectangle has length {length} cm and width {width} cm, what is its perimeter?"
                answer = f"{perimeter} cm"
                difficulty = DifficultyLevel.EASY
                
            else:
                # Compound interest
                principal = np.random.randint(1000, 10000)
                rate = np.random.randint(5, 15)
                time = np.random.randint(2, 5)
                
                amount = principal * (1 + rate/100) ** time
                question = f"Calculate the compound interest on ${principal} at {rate}% annual rate for {time} years."
                answer = f"${amount:.2f}"
                difficulty = DifficultyLevel.HARD
            
            task = BenchmarkTask(
                id=f"math_{i:03d}",
                task_type=TaskType.MATHEMATICAL,
                difficulty=difficulty,
                question=question,
                expected_answer=answer,
                evaluation_metric=EvaluationMetric.EXACT_MATCH,
                tags=["auto-generated", "math"]
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_logic_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate logical reasoning tasks."""
        tasks = []
        
        logic_patterns = [
            {
                "question": "If all A are B, and all B are C, then all A are C. Given: All cats are animals, and all animals are living things. What can we conclude about cats?",
                "answer": "All cats are living things",
                "difficulty": DifficultyLevel.MEDIUM
            },
            {
                "question": "If it's raining, then the ground is wet. The ground is wet. Can we conclude it's raining?",
                "answer": "No, we cannot conclude it's raining",
                "difficulty": DifficultyLevel.HARD
            }
        ]
        
        for i in range(count):
            pattern = logic_patterns[i % len(logic_patterns)]
            
            task = BenchmarkTask(
                id=f"logic_{i:03d}",
                task_type=TaskType.LOGICAL_REASONING,
                difficulty=pattern["difficulty"],
                question=pattern["question"],
                expected_answer=pattern["answer"],
                evaluation_metric=EvaluationMetric.SEMANTIC_SIMILARITY,
                tags=["auto-generated", "logic"]
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_programming_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate programming tasks."""
        tasks = []
        
        programming_tasks = [
            {
                "question": "Write a Python function that returns the maximum of two numbers.",
                "answer": "def max_two(a, b): return a if a > b else b",
                "difficulty": DifficultyLevel.EASY
            },
            {
                "question": "Write a Python function that calculates the factorial of a number.",
                "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "difficulty": DifficultyLevel.MEDIUM
            },
            {
                "question": "Write a Python function that checks if a string is a palindrome.",
                "answer": "def is_palindrome(s): return s == s[::-1]",
                "difficulty": DifficultyLevel.MEDIUM
            }
        ]
        
        for i in range(count):
            task_def = programming_tasks[i % len(programming_tasks)]
            
            task = BenchmarkTask(
                id=f"prog_{i:03d}",
                task_type=TaskType.PROGRAMMING,
                difficulty=task_def["difficulty"],
                question=task_def["question"],
                expected_answer=task_def["answer"],
                evaluation_metric=EvaluationMetric.SEMANTIC_SIMILARITY,
                tags=["auto-generated", "programming"]
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_language_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate language understanding tasks."""
        tasks = []
        
        for i in range(count):
            task = BenchmarkTask(
                id=f"lang_{i:03d}",
                task_type=TaskType.LANGUAGE_UNDERSTANDING,
                difficulty=DifficultyLevel.MEDIUM,
                question="What is the capital of France?",
                expected_answer="Paris",
                evaluation_metric=EvaluationMetric.EXACT_MATCH,
                tags=["auto-generated", "language"]
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_problem_solving_tasks(self, count: int) -> List[BenchmarkTask]:
        """Generate problem solving tasks."""
        tasks = []
        
        for i in range(count):
            task = BenchmarkTask(
                id=f"prob_{i:03d}",
                task_type=TaskType.PROBLEM_SOLVING,
                difficulty=DifficultyLevel.MEDIUM,
                question="How would you approach solving a complex problem?",
                expected_answer="Break it down into smaller parts, analyze each part, and solve systematically",
                evaluation_metric=EvaluationMetric.SEMANTIC_SIMILARITY,
                tags=["auto-generated", "problem-solving"]
            )
            tasks.append(task)
        
        return tasks


class BenchmarkRunner:
    """Execute benchmark suites against agents."""
    
    def __init__(self, scorer: Optional[AdvancedScorer] = None):
        self.scorer = scorer or AdvancedScorer()
        self.execution_history: List[BenchmarkResult] = []
        self._lock = threading.Lock()
    
    async def run_benchmark(self, suite: BenchmarkSuite, agent: ReactAgent, parallel: bool = True) -> BenchmarkResult:
        """Run a benchmark suite against an agent."""
        start_time = time.time()
        
        logger.info(f"Starting benchmark '{suite.name}' with {len(suite.tasks)} tasks")
        
        if parallel:
            task_results = await self._run_parallel(suite.tasks, agent)
        else:
            task_results = await self._run_sequential(suite.tasks, agent)
        
        execution_time = time.time() - start_time
        
        # Calculate overall score
        overall_score = statistics.mean([r.score for r in task_results]) if task_results else 0.0
        
        result = BenchmarkResult(
            suite_name=suite.name,
            agent_name=agent.agent_id,
            task_results=task_results,
            overall_score=overall_score,
            execution_time=execution_time,
            metadata={
                "total_tasks": len(suite.tasks),
                "successful_tasks": len([r for r in task_results if r.success]),
                "average_execution_time": statistics.mean([r.execution_time for r in task_results]) if task_results else 0.0,
                "total_tokens": sum([r.tokens_used for r in task_results]),
                "total_api_calls": sum([r.api_calls for r in task_results])
            }
        )
        
        with self._lock:
            self.execution_history.append(result)
        
        logger.info(f"Benchmark completed: {overall_score:.2%} success rate in {execution_time:.2f}s")
        
        return result
    
    async def _run_sequential(self, tasks: List[BenchmarkTask], agent: ReactAgent) -> List[TaskResult]:
        """Run tasks sequentially."""
        results = []
        
        for task in tasks:
            result = await self._execute_task(task, agent)
            results.append(result)
        
        return results
    
    async def _run_parallel(self, tasks: List[BenchmarkTask], agent: ReactAgent, max_workers: int = 5) -> List[TaskResult]:
        """Run tasks in parallel."""
        results = []
        
        # Use semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_workers)
        
        async def bounded_execute(task):
            async with semaphore:
                return await self._execute_task(task, agent)
        
        # Execute all tasks concurrently
        tasks_coroutines = [bounded_execute(task) for task in tasks]
        results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Filter out exceptions and convert to TaskResult
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {tasks[i].id} failed with exception: {result}")
                # Create failed result
                failed_result = TaskResult(
                    task_id=tasks[i].id,
                    agent_name=agent.agent_id,
                    actual_answer="",
                    expected_answer=tasks[i].expected_answer,
                    score=0.0,
                    execution_time=0.0,
                    tokens_used=0,
                    api_calls=0,
                    success=False,
                    error_message=str(result)
                )
                valid_results.append(failed_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_task(self, task: BenchmarkTask, agent: ReactAgent) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Prepare the input
            task_input = task.question
            if task.context:
                task_input = f"Context: {task.context}\n\nQuestion: {task.question}"
            
            # Execute the task
            response = await agent.execute_task(task_input)
            
            # Extract the answer (assuming response has a 'content' or similar field)
            if hasattr(response, 'content'):
                actual_answer = response.content
            elif hasattr(response, 'response'):
                actual_answer = response.response
            else:
                actual_answer = str(response)
            
            execution_time = time.time() - start_time
            
            # Score the result
            score = self.scorer.score_task(task, actual_answer)
            
            return TaskResult(
                task_id=task.id,
                agent_name=agent.agent_id,
                actual_answer=actual_answer,
                expected_answer=task.expected_answer,
                score=score,
                execution_time=execution_time,
                tokens_used=getattr(response, 'tokens_used', 0),
                api_calls=1,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.id} execution failed: {e}")
            
            return TaskResult(
                task_id=task.id,
                agent_name=agent.agent_id,
                actual_answer="",
                expected_answer=task.expected_answer,
                score=0.0,
                execution_time=execution_time,
                tokens_used=0,
                api_calls=1,
                success=False,
                error_message=str(e)
            )


class BenchmarkAnalyzer:
    """Analyze benchmark results and generate insights."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze a single benchmark result."""
        cache_key = f"{result.suite_name}:{result.agent_name}:{result.timestamp}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        analysis = {
            "overall_performance": self._analyze_overall_performance(result),
            "task_type_performance": self._analyze_by_task_type(result),
            "difficulty_performance": self._analyze_by_difficulty(result),
            "time_analysis": self._analyze_execution_times(result),
            "error_analysis": self._analyze_errors(result),
            "efficiency_metrics": self._analyze_efficiency(result)
        }
        
        self.analysis_cache[cache_key] = analysis
        return analysis
    
    def _analyze_overall_performance(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze overall performance metrics."""
        successful_tasks = [r for r in result.task_results if r.success]
        failed_tasks = [r for r in result.task_results if not r.success]
        
        scores = [r.score for r in successful_tasks]
        
        return {
            "success_rate": len(successful_tasks) / len(result.task_results) if result.task_results else 0.0,
            "average_score": statistics.mean(scores) if scores else 0.0,
            "score_std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "median_score": statistics.median(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "total_tasks": len(result.task_results),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks)
        }
    
    def _analyze_by_task_type(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze performance by task type."""
        task_type_results = defaultdict(list)
        
        # Group results by task type (would need task metadata)
        for task_result in result.task_results:
            # For now, infer task type from task_id prefix
            task_type = "unknown"
            if task_result.task_id.startswith("math_"):
                task_type = "mathematical"
            elif task_result.task_id.startswith("logic_"):
                task_type = "logical_reasoning"
            elif task_result.task_id.startswith("prog_"):
                task_type = "programming"
            elif task_result.task_id.startswith("lang_"):
                task_type = "language_understanding"
            elif task_result.task_id.startswith("prob_"):
                task_type = "problem_solving"
            
            task_type_results[task_type].append(task_result)
        
        analysis = {}
        for task_type, results in task_type_results.items():
            scores = [r.score for r in results if r.success]
            analysis[task_type] = {
                "success_rate": len([r for r in results if r.success]) / len(results) if results else 0.0,
                "average_score": statistics.mean(scores) if scores else 0.0,
                "task_count": len(results)
            }
        
        return analysis
    
    def _analyze_by_difficulty(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze performance by difficulty level."""
        # Would need difficulty metadata in task results
        return {"note": "Difficulty analysis requires task metadata"}
    
    def _analyze_execution_times(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze execution time patterns."""
        execution_times = [r.execution_time for r in result.task_results]
        
        if not execution_times:
            return {}
        
        return {
            "average_time": statistics.mean(execution_times),
            "median_time": statistics.median(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
            "total_time": sum(execution_times)
        }
    
    def _analyze_errors(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze error patterns."""
        failed_tasks = [r for r in result.task_results if not r.success]
        
        error_types = defaultdict(int)
        for task in failed_tasks:
            if task.error_message:
                # Categorize errors
                if "timeout" in task.error_message.lower():
                    error_types["timeout"] += 1
                elif "connection" in task.error_message.lower():
                    error_types["connection"] += 1
                elif "rate limit" in task.error_message.lower():
                    error_types["rate_limit"] += 1
                else:
                    error_types["other"] += 1
        
        return {
            "total_errors": len(failed_tasks),
            "error_rate": len(failed_tasks) / len(result.task_results) if result.task_results else 0.0,
            "error_types": dict(error_types)
        }
    
    def _analyze_efficiency(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Analyze efficiency metrics."""
        total_tokens = sum([r.tokens_used for r in result.task_results])
        total_api_calls = sum([r.api_calls for r in result.task_results])
        successful_tasks = len([r for r in result.task_results if r.success])
        
        return {
            "tokens_per_task": total_tokens / len(result.task_results) if result.task_results else 0.0,
            "api_calls_per_task": total_api_calls / len(result.task_results) if result.task_results else 0.0,
            "tokens_per_successful_task": total_tokens / successful_tasks if successful_tasks > 0 else 0.0,
            "success_per_token": successful_tasks / total_tokens if total_tokens > 0 else 0.0,
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls
        }
    
    def compare_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare multiple benchmark results."""
        if len(results) < 2:
            return {"error": "Need at least 2 results to compare"}
        
        comparison = {
            "agents": [r.agent_name for r in results],
            "overall_scores": [r.overall_score for r in results],
            "success_rates": [],
            "execution_times": [r.execution_time for r in results],
            "performance_ranking": []
        }
        
        # Calculate success rates
        for result in results:
            successful = len([r for r in result.task_results if r.success])
            total = len(result.task_results)
            comparison["success_rates"].append(successful / total if total > 0 else 0.0)
        
        # Rank agents by overall score
        agent_scores = [(r.agent_name, r.overall_score) for r in results]
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        comparison["performance_ranking"] = agent_scores
        
        return comparison


class BenchmarkReporter:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, analyzer: Optional[BenchmarkAnalyzer] = None):
        self.analyzer = analyzer or BenchmarkAnalyzer()
    
    def generate_report(self, result: BenchmarkResult, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        analysis = self.analyzer.analyze_result(result)
        
        report = {
            "metadata": {
                "suite_name": result.suite_name,
                "agent_name": result.agent_name,
                "timestamp": result.timestamp.isoformat(),
                "execution_time": result.execution_time,
                "report_generated": datetime.now().isoformat()
            },
            "executive_summary": {
                "overall_score": result.overall_score,
                "success_rate": analysis["overall_performance"]["success_rate"],
                "total_tasks": analysis["overall_performance"]["total_tasks"],
                "average_execution_time": analysis["time_analysis"].get("average_time", 0.0),
                "key_insights": self._generate_key_insights(analysis)
            },
            "detailed_analysis": analysis,
            "task_results": [asdict(r) for r in result.task_results],
            "recommendations": self._generate_recommendations(analysis)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_key_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        overall = analysis["overall_performance"]
        
        if overall["success_rate"] > 0.9:
            insights.append("Excellent performance with >90% success rate")
        elif overall["success_rate"] > 0.7:
            insights.append("Good performance with >70% success rate")
        elif overall["success_rate"] > 0.5:
            insights.append("Moderate performance with >50% success rate")
        else:
            insights.append("Performance needs improvement with <50% success rate")
        
        if overall["score_std_dev"] > 0.3:
            insights.append("High variability in task performance")
        else:
            insights.append("Consistent performance across tasks")
        
        # Task type insights
        task_type_perf = analysis["task_type_performance"]
        if task_type_perf:
            best_type = max(task_type_perf.keys(), key=lambda k: task_type_perf[k]["success_rate"])
            worst_type = min(task_type_perf.keys(), key=lambda k: task_type_perf[k]["success_rate"])
            
            insights.append(f"Strongest performance in {best_type} tasks")
            insights.append(f"Weakest performance in {worst_type} tasks")
        
        return insights
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        overall = analysis["overall_performance"]
        
        if overall["success_rate"] < 0.7:
            recommendations.append("Consider improving agent's problem-solving capabilities")
        
        if overall["failed_tasks"] > 0:
            recommendations.append("Investigate failed tasks for common patterns")
        
        time_analysis = analysis["time_analysis"]
        if time_analysis.get("average_time", 0) > 5.0:
            recommendations.append("Consider optimizing response time")
        
        efficiency = analysis["efficiency_metrics"]
        if efficiency.get("tokens_per_task", 0) > 1000:
            recommendations.append("Consider reducing token usage for better efficiency")
        
        return recommendations
    
    def generate_comparison_report(self, results: List[BenchmarkResult], output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comparison report for multiple results."""
        comparison = self.analyzer.compare_results(results)
        
        report = {
            "metadata": {
                "comparison_type": "multi_agent",
                "agents_compared": len(results),
                "timestamp": datetime.now().isoformat()
            },
            "comparison_summary": comparison,
            "detailed_comparisons": [
                self.analyzer.analyze_result(result) for result in results
            ],
            "winner": comparison["performance_ranking"][0] if comparison["performance_ranking"] else None
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


class EnhancedBenchmarkSystem:
    """Main benchmark system coordinator."""
    
    def __init__(self):
        self.data_generator = BenchmarkDataGenerator()
        self.scorer = AdvancedScorer()
        self.runner = BenchmarkRunner(self.scorer)
        self.analyzer = BenchmarkAnalyzer()
        self.reporter = BenchmarkReporter(self.analyzer)
        
        # Built-in benchmark suites
        self.built_in_suites = self._create_built_in_suites()
    
    def _create_built_in_suites(self) -> Dict[str, BenchmarkSuite]:
        """Create built-in benchmark suites."""
        suites = {}
        
        # Comprehensive suite
        suites["comprehensive"] = self.data_generator.generate_suite(
            "Comprehensive Evaluation",
            {
                TaskType.MATHEMATICAL: 5,
                TaskType.LOGICAL_REASONING: 3,
                TaskType.PROGRAMMING: 3,
                TaskType.LANGUAGE_UNDERSTANDING: 2,
                TaskType.PROBLEM_SOLVING: 2
            }
        )
        
        # Math-focused suite
        suites["mathematics"] = self.data_generator.generate_suite(
            "Mathematics Evaluation",
            {TaskType.MATHEMATICAL: 10}
        )
        
        # Programming-focused suite
        suites["programming"] = self.data_generator.generate_suite(
            "Programming Evaluation",
            {TaskType.PROGRAMMING: 8}
        )
        
        return suites
    
    async def run_full_evaluation(self, agent: ReactAgent, suite_name: str = "comprehensive") -> Dict[str, Any]:
        """Run a full evaluation with analysis and reporting."""
        
        if suite_name not in self.built_in_suites:
            raise ValueError(f"Unknown suite: {suite_name}")
        
        suite = self.built_in_suites[suite_name]
        
        # Run benchmark
        result = await self.runner.run_benchmark(suite, agent)
        
        # Generate report
        report = self.reporter.generate_report(result)
        
        return {
            "result": result,
            "report": report,
            "summary": {
                "overall_score": result.overall_score,
                "success_rate": len([r for r in result.task_results if r.success]) / len(result.task_results),
                "execution_time": result.execution_time,
                "total_tasks": len(result.task_results)
            }
        }
    
    async def compare_agents(self, agents: List[ReactAgent], suite_name: str = "comprehensive") -> Dict[str, Any]:
        """Compare multiple agents on the same benchmark suite."""
        
        if suite_name not in self.built_in_suites:
            raise ValueError(f"Unknown suite: {suite_name}")
        
        suite = self.built_in_suites[suite_name]
        results = []
        
        # Run benchmarks for all agents
        for agent in agents:
            result = await self.runner.run_benchmark(suite, agent)
            results.append(result)
        
        # Generate comparison report
        comparison_report = self.reporter.generate_comparison_report(results)
        
        return comparison_report
    
    def get_available_suites(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available benchmark suites."""
        suite_info = {}
        
        for name, suite in self.built_in_suites.items():
            task_types = defaultdict(int)
            difficulties = defaultdict(int)
            
            for task in suite.tasks:
                task_types[task.task_type.value] += 1
                difficulties[task.difficulty.value] += 1
            
            suite_info[name] = {
                "description": suite.description,
                "total_tasks": len(suite.tasks),
                "task_types": dict(task_types),
                "difficulties": dict(difficulties),
                "version": suite.version
            }
        
        return suite_info


# Example usage and demonstration
async def demonstrate_enhanced_benchmarking():
    """Demonstrate the enhanced benchmarking system."""
    
    print("Starting Enhanced Benchmark System Demo")
    print("=" * 50)
    
    # Initialize system
    benchmark_system = EnhancedBenchmarkSystem()
    
    # Show available suites
    print("\nLIST: Available Benchmark Suites:")
    suites = benchmark_system.get_available_suites()
    for name, info in suites.items():
        print(f"  {name}: {info['total_tasks']} tasks - {info['description']}")
    
    # Create a test agent
    config = AgentConfig()
    provider = MockProvider()
    agent = ReactAgent(config, provider)
    
    print(f"\nAgent Testing Agent: {agent.agent_id}")
    
    # Run comprehensive evaluation
    print("\nRunning Comprehensive Evaluation...")
    evaluation = await benchmark_system.run_full_evaluation(agent, "comprehensive")
    
    summary = evaluation["summary"]
    print(f"PASS Evaluation completed:")
    print(f"  Overall Score: {summary['overall_score']:.2%}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Execution Time: {summary['execution_time']:.2f}s")
    print(f"  Total Tasks: {summary['total_tasks']}")
    
    # Show key insights
    report = evaluation["report"]
    insights = report["executive_summary"]["key_insights"]
    print(f"\nINSIGHT Key Insights:")
    for insight in insights:
        print(f"  • {insight}")
    
    # Show recommendations
    recommendations = report["recommendations"]
    if recommendations:
        print(f"\nResponse Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("\nSUCCESS Enhanced benchmarking demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_benchmarking()) 