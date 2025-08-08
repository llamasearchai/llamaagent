"""
Advanced Benchmark Engine for AI Agents

Provides comprehensive benchmarking capabilities with automated test execution,
performance metrics collection, and detailed reporting for model comparison.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

# Optional import for GoldenDatasetManager
try:
    from .golden_dataset import GoldenDatasetManager
except ImportError:
    GoldenDatasetManager = None

logger = logging.getLogger(__name__)


class BenchmarkType(str, Enum):
    """Types of benchmarks"""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    SAFETY = "safety"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    COMPREHENSIVE = "comprehensive"


class MetricType(str, Enum):
    """Types of metrics to collect"""

    ACCURACY_SCORE = "accuracy_score"
    F1_SCORE = "f1_score"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    LATENCY_MS = "latency_ms"
    THROUGHPUT_RPS = "throughput_rps"
    MEMORY_MB = "memory_mb"
    CPU_PERCENT = "cpu_percent"
    TOKEN_COUNT = "token_count"
    COST_USD = "cost_usd"


@dataclass
class BenchmarkTask:
    """Individual benchmark task"""

    id: str
    name: str
    description: str
    input_data: Union[str, Dict[str, Any]]
    expected_output: Union[str, Dict[str, Any]]
    metrics_to_collect: List[MetricType]
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a single benchmark task"""

    task_id: str
    success: bool
    actual_output: str = ""
    execution_time: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark execution result"""

    benchmark_id: str
    benchmark_name: str
    model_name: str
    started_at: datetime
    completed_at: datetime
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    task_results: List[TaskResult]
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_tasks / max(1, self.total_tasks)

    @property
    def execution_time(self) -> float:
        """Total execution time in seconds"""
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "execution_time": self.execution_time,
            "task_results": [asdict(tr) for tr in self.task_results],
            "overall_metrics": self.overall_metrics,
            "summary_stats": self.summary_stats,
        }


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks"""

    suite_id: str
    name: str
    description: str
    benchmarks: List[str]  # Benchmark IDs
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkEngine:
    """Advanced benchmark execution engine"""

    def __init__(
        self, dataset_manager: Optional[Any] = None, results_path: Optional[Path] = None
    ):
        """Initialize benchmark engine"""
        if GoldenDatasetManager and dataset_manager is None:
            self.dataset_manager = GoldenDatasetManager()
        else:
            self.dataset_manager = dataset_manager

        self.results_path = results_path or Path("./benchmark_results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        # Registry
        self.benchmark_registry: Dict[str, List[BenchmarkTask]] = {}
        self.suite_registry: Dict[str, BenchmarkSuite] = {}
        self.model_registry: Dict[str, Callable[..., Any]] = {}

        # Metrics calculators
        self.metric_calculators: Dict[MetricType, Callable[..., Any]] = {
            MetricType.ACCURACY_SCORE: self._calculate_accuracy,
            MetricType.F1_SCORE: self._calculate_f1_score,
            MetricType.BLEU_SCORE: self._calculate_bleu_score,
            MetricType.ROUGE_SCORE: self._calculate_rouge_score,
        }

        # Results cache
        self.results_cache: Dict[str, BenchmarkResult] = {}

    def register_model(
        self, model_name: str, model_function: Callable[..., Any]
    ) -> None:
        """Register a model for benchmarking"""
        self.model_registry[model_name] = model_function
        logger.info(f"Registered model: {model_name}")

    async def register_benchmark(
        self, benchmark_id: str, tasks: List[BenchmarkTask]
    ) -> None:
        """Register a new benchmark"""
        self.benchmark_registry[benchmark_id] = tasks
        logger.info(f"Registered benchmark: {benchmark_id} with {len(tasks)} tasks")

    async def create_benchmark_from_dataset(
        self,
        benchmark_id: str,
        dataset_name: str,
        sample_limit: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create benchmark from golden dataset"""
        if not self.dataset_manager:
            raise ValueError("Dataset manager not available")
        # Load dataset if not already loaded
        if (
            not hasattr(self.dataset_manager, 'datasets')
            or dataset_name not in self.dataset_manager.datasets
        ):
            if hasattr(self.dataset_manager, 'load_dataset'):
                await self.dataset_manager.load_dataset(dataset_name)
            else:
                raise ValueError(f"Cannot load dataset: {dataset_name}")
        samples = self.dataset_manager.datasets[dataset_name]
        if sample_limit:
            samples = samples[:sample_limit]

        tasks: List[BenchmarkTask] = []
        for sample in samples:
            task = BenchmarkTask(
                id=f"{benchmark_id}_{sample.id}",
                name=f"Task from {dataset_name}: {sample.id}",
                description="Benchmark task derived from dataset sample",
                input_data=sample.input,
                expected_output=sample.expected_output,
                metrics_to_collect=[MetricType.ACCURACY_SCORE, MetricType.LATENCY_MS],
                metadata=config or {},
            )
            tasks.append(task)
        await self.register_benchmark(benchmark_id, tasks)

    async def run_benchmark(
        self,
        benchmark_id: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run a complete benchmark"""
        if benchmark_id not in self.benchmark_registry:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not registered")
        tasks = self.benchmark_registry[benchmark_id]
        model_function = self.model_registry[model_name]
        config = config or {}

        logger.info(f"Starting benchmark {benchmark_id} for model {model_name}")
        started_at = datetime.now(timezone.utc)
        task_results: List[TaskResult] = []
        successful_tasks = 0
        failed_tasks = 0

        # Execute tasks
        for task in tasks:
            try:
                result = await self._execute_task(task, model_function, config)
                task_results.append(result)
                if result.success:
                    successful_tasks += 1
                else:
                    failed_tasks += 1

            except Exception as e:
                logger.error(f"Task {task.id} failed with error: {e}")
                failed_result = TaskResult(
                    task_id=task.id,
                    success=False,
                    error_message=str(e),
                    execution_time=0.0,
                )
                task_results.append(failed_result)
                failed_tasks += 1

        completed_at = datetime.now(timezone.utc)
        # Calculate overall metrics
        overall_metrics = await self._calculate_overall_metrics(task_results)
        # Generate summary statistics
        summary_stats = await self._generate_summary_stats(task_results)
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_id=f"{benchmark_id}_{model_name}_{int(started_at.timestamp())}",
            benchmark_name=benchmark_id,
            model_name=model_name,
            started_at=started_at,
            completed_at=completed_at,
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            task_results=task_results,
            overall_metrics=overall_metrics,
            summary_stats=summary_stats,
        )

        # Cache and save result
        self.results_cache[result.benchmark_id] = result
        await self._save_result(result)
        logger.info(
            f"Completed benchmark {benchmark_id}: {successful_tasks}/{len(tasks)} tasks successful"
        )
        return result

    async def run_benchmark_suite(
        self, suite_id: str, model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in a suite"""
        if suite_id not in self.suite_registry:
            raise ValueError(f"Benchmark suite {suite_id} not found")
        suite = self.suite_registry[suite_id]
        results: Dict[str, BenchmarkResult] = {}

        logger.info(
            f"Running benchmark suite {suite_id} with {len(suite.benchmarks)} benchmarks"
        )

        for benchmark_id in suite.benchmarks:
            try:
                result = await self.run_benchmark(benchmark_id, model_name, config)
                results[benchmark_id] = result
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_id}: {e}")
                continue

        return results

    async def compare_models(
        self,
        benchmark_id: str,
        model_names: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models on the same benchmark"""
        results: Dict[str, BenchmarkResult] = {}

        for model_name in model_names:
            try:
                result = await self.run_benchmark(benchmark_id, model_name, config)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")
                continue

        return results

    async def get_benchmark_history(
        self,
        benchmark_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get benchmark execution history"""
        # Load results from storage
        all_results: List[Dict[str, Any]] = []

        for result_file in self.results_path.glob("*.json"):
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                # Filter by criteria
                if benchmark_id and data.get("benchmark_name") != benchmark_id:
                    continue
                if model_name and data.get("model_name") != model_name:
                    continue

                all_results.append(data)
            except Exception as e:
                logger.warning(f"Failed to load result file {result_file}: {e}")
                continue

        # Sort by timestamp and limit
        all_results.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return all_results[:limit]

    async def _execute_task(
        self,
        task: BenchmarkTask,
        model_function: Callable[..., Any],
        config: Dict[str, Any],
    ) -> TaskResult:
        """Execute a single benchmark task"""
        start_time = time.time()

        try:
            # Execute model with timeout
            actual_output = await asyncio.wait_for(
                model_function(task.input_data, **config), timeout=task.timeout_seconds
            )

            execution_time = time.time() - start_time

            # Calculate metrics
            metrics: Dict[str, float] = {}
            for metric_type in task.metrics_to_collect:
                if metric_type in self.metric_calculators:
                    try:
                        metric_value = await self.metric_calculators[metric_type](
                            actual_output, task.expected_output
                        )
                        metrics[metric_type.value] = metric_value
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_type.value}: {e}")
            # Add execution time
            metrics["execution_time"] = execution_time

            return TaskResult(
                task_id=task.id,
                success=True,
                actual_output=str(actual_output),
                execution_time=execution_time,
                metrics=metrics,
                metadata=task.metadata,
            )

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=f"Task timed out after {task.timeout_seconds} seconds",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    async def _calculate_overall_metrics(
        self, task_results: List[TaskResult]
    ) -> Dict[str, Any]:
        """Calculate overall metrics from task results"""
        if not task_results:
            return {}

        successful_results = [r for r in task_results if r.success]

        overall_metrics: Dict[str, Any] = {}

        # Success rate
        overall_metrics["success_rate"] = len(successful_results) / len(task_results)
        # Average execution time
        if successful_results:
            overall_metrics["avg_execution_time"] = statistics.mean(
                [r.execution_time for r in successful_results]
            )
            # Aggregate other metrics
            all_metric_keys = set()
            for result in successful_results:
                all_metric_keys.update(result.metrics.keys())
            for metric_key in all_metric_keys:
                values = [
                    r.metrics[metric_key]
                    for r in successful_results
                    if metric_key in r.metrics
                ]
                if values:
                    overall_metrics[f"avg_{metric_key}"] = statistics.mean(values)
                    overall_metrics[f"std_{metric_key}"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )

        return overall_metrics

    async def _generate_summary_stats(
        self, task_results: List[TaskResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        if not task_results:
            return {}

        successful_results = [r for r in task_results if r.success]

        stats = {
            "total_tasks": len(task_results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(task_results) - len(successful_results),
        }

        if successful_results:
            execution_times = [r.execution_time for r in successful_results]
            stats["execution_time_stats"] = {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std": (
                    statistics.stdev(execution_times)
                    if len(execution_times) > 1
                    else 0.0
                ),
                "min": min(execution_times),
                "max": max(execution_times),
                "p95": np.percentile(execution_times, 95),
                "p99": np.percentile(execution_times, 99),
            }

            # Error analysis
            error_messages = [
                r.error_message
                for r in task_results
                if not r.success and r.error_message
            ]
            error_counts = defaultdict(int)
            for error in error_messages:
                error_counts[error] += 1

            stats["error_analysis"] = dict(error_counts)
        return stats

    async def _calculate_accuracy(self, actual: Any, expected: Any) -> float:
        """Calculate accuracy score"""
        if isinstance(actual, str) and isinstance(expected, str):
            return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0
        return 1.0 if actual == expected else 0.0

    async def _calculate_f1_score(self, actual: Any, expected: Any) -> float:
        """Calculate F1 score for text comparison"""
        if isinstance(actual, str) and isinstance(expected, str):
            actual_tokens = set(actual.lower().split())
            expected_tokens = set(expected.lower().split())

            if not expected_tokens:
                return 1.0 if not actual_tokens else 0.0

            intersection = actual_tokens & expected_tokens
            precision = len(intersection) / len(actual_tokens) if actual_tokens else 0
            recall = len(intersection) / len(expected_tokens)
            if precision + recall == 0:
                return 0.0

            return 2 * (precision * recall) / (precision + recall)

        return 0.0

    async def _calculate_bleu_score(self, actual: Any, expected: Any) -> float:
        """Calculate BLEU score (simplified version)"""
        if not isinstance(actual, str) or not isinstance(expected, str):
            return 0.0

        actual_words = actual.lower().split()
        expected_words = expected.lower().split()

        if not expected_words:
            return 1.0 if not actual_words else 0.0

        # Simplified BLEU calculation (1-gram precision)
        matches = sum(1 for word in actual_words if word in expected_words)
        precision = matches / len(actual_words) if actual_words else 0

        # Apply brevity penalty
        brevity_penalty = min(1.0, len(actual_words) / len(expected_words))
        return precision * brevity_penalty

    async def _calculate_rouge_score(self, actual: Any, expected: Any) -> float:
        """Calculate ROUGE score (simplified version)"""
        if not isinstance(actual, str) or not isinstance(expected, str):
            return 0.0

        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 1.0 if not actual_words else 0.0

        intersection = actual_words & expected_words
        return len(intersection) / len(expected_words)

    async def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to storage"""
        result_file = self.results_path / f"{result.benchmark_id}.json"

        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def create_suite(
        self,
        suite_id: str,
        name: str,
        description: str,
        benchmark_ids: List[str],
        tags: Optional[List[str]] = None,
    ) -> None:
        """Create a benchmark suite"""
        suite = BenchmarkSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            benchmarks=benchmark_ids,
            tags=tags or [],
        )
        self.suite_registry[suite_id] = suite
        logger.info(
            f"Created benchmark suite: {suite_id} with {len(benchmark_ids)} benchmarks"
        )
