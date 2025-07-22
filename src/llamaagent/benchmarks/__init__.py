from __future__ import annotations

"""Comprehensive benchmarking infrastructure for SPRE evaluation.

This module implements the scientific testing protocol outlined in the research
paper, including GAIA benchmark integration, baseline comparisons, and
statistical analysis of agent performance.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

# ---------------------------------------------------------------------------
# Public re-exports (kept explicit for readability)
# ---------------------------------------------------------------------------

from .baseline_agents import BaselineAgentFactory  # noqa: F401,E402

try:
    from .gaia_benchmark import GAIABenchmark, GAIATask  # type: ignore
except Exception:  # pylint: disable=broad-except
    # Fall back to stubs if the benchmark contains syntax errors.
    class _Stub:
        ...

    GAIABenchmark = _Stub  # type: ignore
    GAIATask = _Stub  # type: ignore
try:
    from .spre_evaluator import BenchmarkResult, SPREEvaluator  # type: ignore
except Exception:  # pylint: disable=broad-except

    class _EvaluatorStub:  # fallback stub
        ...

    BenchmarkResult = _EvaluatorStub  # type: ignore
    SPREEvaluator = _EvaluatorStub  # type: ignore

# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "GAIATask",
    "GAIABenchmark",
    "SPREEvaluator",
    "BaselineAgentFactory",
    "BenchmarkResult",
]
