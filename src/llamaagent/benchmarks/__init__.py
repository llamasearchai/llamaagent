from __future__ import annotations

"""Comprehensive benchmarking infrastructure for SPRE evaluation.

This module implements the scientific testing protocol outlined in the research
document, including GAIA benchmark integration, baseline comparisons, and
statistical analysis of agent performance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .baseline_agents import BaselineAgentFactory  # noqa: F401
from .gaia_benchmark import GAIABenchmark, GAIATask  # noqa: F401
from .spre_evaluator import BenchmarkResult, SPREEvaluator  # noqa: F401
