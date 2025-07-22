"""
Statistical Analysis Module for LlamaAgent

This module provides statistical analysis tools for experiment results.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Any, Dict, List

# Optional imports for advanced statistical analysis
try:
    import numpy as np
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class StatisticalAnalyzer:
    """
    Utility class providing basic statistical analysis helpers for experiment results.
    """

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        """Initialize the statistical analyzer with results."""
        self.results = results

    def compare_techniques(
        self, metric: str, technique_a: str, technique_b: str
    ) -> Dict[str, Any]:
        """Perform t-test comparison between two techniques."""
        if not SCIPY_AVAILABLE:
            return {
                "comparison": f"{technique_a} vs {technique_b}",
                "metric": metric,
                "error": "SciPy not available for statistical tests",
            }

        a_vals: List[float] = [
            float(res[metric])
            for res in self.results
            if res.get("technique") == technique_a and metric in res
        ]
        b_vals: List[float] = [
            float(res[metric])
            for res in self.results
            if res.get("technique") == technique_b and metric in res
        ]

        if len(a_vals) < 2 or len(b_vals) < 2:
            return {
                "comparison": f"{technique_a} vs {technique_b}",
                "metric": metric,
                "error": "Insufficient data for statistical comparison",
            }

        t_stat_raw, p_val = stats.ttest_ind(a_vals, b_vals)
        p_value: float = float(p_val)
        effect_size = self._cohens_d(a_vals, b_vals)

        return {
            "comparison": f"{technique_a} vs {technique_b}",
            "metric": metric,
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": p_value < 0.05,
            "mean_a": np.mean(a_vals),
            "mean_b": np.mean(b_vals),
            "n": len(a_vals) + len(b_vals),
        }

    def correlation_analysis(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """Calculate correlation between two metrics."""
        if not SCIPY_AVAILABLE:
            return {
                "metric1": metric1,
                "metric2": metric2,
                "error": "SciPy not available for correlation analysis",
            }

        vals1: List[float] = [
            float(res[metric1])
            for res in self.results
            if metric1 in res and res[metric1] is not None
        ]
        vals2: List[float] = [
            float(res[metric2])
            for res in self.results
            if metric2 in res and res[metric2] is not None
        ]

        if len(vals1) < 2 or len(vals2) < 2 or len(vals1) != len(vals2):
            return {
                "metric1": metric1,
                "metric2": metric2,
                "error": "Insufficient or mismatched data for correlation",
            }

        r_raw, p_val = stats.pearsonr(vals1, vals2)
        r: float = float(r_raw)
        p_value: float = float(p_val)

        return {
            "metric1": metric1,
            "metric2": metric2,
            "correlation": r,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n": len(vals1),
        }

    def _cohens_d(self, x: List[float], y: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not SCIPY_AVAILABLE:
            return 0.0

        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return 0.0

        dof = nx + ny - 2
        pooled_std = np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2)
            / dof
        )

        if pooled_std == 0:
            return 0.0

        return (np.mean(x) - np.mean(y)) / pooled_std

    def analyze_benchmark_results(self) -> Dict[str, Any]:
        """Return a concise statistical summary for benchmark results."""
        summary: Dict[str, Any] = {}

        techniques = {r.get("technique", "Unknown") for r in self.results}
        for tech in techniques:
            tech_results = [r for r in self.results if r.get("technique") == tech]
            if not tech_results:
                continue

            # Calculate mean execution time
            durations = [r.get("duration", 0) for r in tech_results if "duration" in r]
            mean_time = sum(durations) / len(durations) if durations else 0.0

            # Calculate success rate
            success_rate = sum(
                1
                for r in tech_results
                if r.get("success", False) or r.get("consensus_reached", False)
            ) / len(tech_results)

            summary[tech] = {
                "mean_time": mean_time,
                "success_rate": success_rate,
            }

        return summary
