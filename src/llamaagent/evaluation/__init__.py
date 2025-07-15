"""
Premium Evaluation Infrastructure for LlamaAgent

Provides comprehensive evaluation systems including golden dataset creation,
automated benchmarking, and model comparison tools.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .a_b_testing import ABTestingFramework, ABTestResult, StatisticalSignificance
from .automated_grading import AutomatedGrader, GradingCriteria, GradingResult
from .benchmark_engine import BenchmarkEngine, BenchmarkResult, BenchmarkSuite
from .evaluation_framework import (
    EvaluationFramework,
    EvaluationMetrics,
    EvaluationReport,
)
from .golden_dataset import DataQualityReport, DatasetMetrics, GoldenDatasetManager
from .model_comparison import ComparisonReport, ModelComparator, ModelPerformance

__all__ = [
    "GoldenDatasetManager",
    "DatasetMetrics",
    "DataQualityReport",
    "BenchmarkEngine",
    "BenchmarkResult",
    "BenchmarkSuite",
    "ModelComparator",
    "ComparisonReport",
    "ModelPerformance",
    "EvaluationFramework",
    "EvaluationMetrics",
    "EvaluationReport",
    "AutomatedGrader",
    "GradingCriteria",
    "GradingResult",
    "ABTestingFramework",
    "ABTestResult",
    "StatisticalSignificance",
]
