"""
Premium Evaluation Infrastructure for LlamaAgent

Provides comprehensive evaluation systems including golden dataset creation,
automated benchmarking, and model comparison tools.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .benchmark_engine import BenchmarkEngine, BenchmarkResult, BenchmarkSuite
from .golden_dataset import (
    DataQualityReport,
    DatasetMetrics,
    GoldenDatasetManager,
)
from .model_comparison import ComparisonReport, ModelComparator, ModelPerformance


class GoldenDataset:  # minimal placeholder for test imports
    pass

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
    # Backward-compat aliases used by some tests
    "GoldenDataset",
]
