# pyright: reportGeneralTypeIssues=false, reportOptionalCall=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false

from typing import Any, Dict, List

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """Utility class providing basic statistical analysis helpers for experiment results."""

    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results

    def compare_techniques(self, metric: str, technique_a: str, technique_b: str) -> Dict[str, Any]:
        """Perform t-test comparison between two techniques"""
        a_vals: List[float] = [float(res[metric]) for res in self.results if res["technique"] == technique_a]  # type: ignore[arg-type]
        b_vals: List[float] = [float(res[metric]) for res in self.results if res["technique"] == technique_b]  # type: ignore[arg-type]

        t_stat_raw, p_val = stats.ttest_ind(a_vals, b_vals)  # type: ignore[arg-type]
        p_value: float = float(p_val)
        effect_size = self._cohens_d(a_vals, b_vals)

        return {
            "comparison": f"{technique_a} vs {technique_b}",
            "metric": metric,
            "mean_a": np.mean(a_vals),
            "mean_b": np.mean(b_vals),
            "p_value": p_value,
            "effect_size": effect_size,
            "significant": p_value < 0.05,
        }

    def _cohens_d(self, x, y) -> float:
        """Calculate Cohen's d effect size"""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std

    def correlation_analysis(self, metric1: str, metric2: str) -> Dict[str, Any]:
        """Calculate correlation between two metrics"""
        vals1: List[float] = [float(res[metric1]) for res in self.results]  # type: ignore[arg-type]
        vals2: List[float] = [float(res[metric2]) for res in self.results]  # type: ignore[arg-type]

        r_raw, p_val = stats.pearsonr(vals1, vals2)  # type: ignore[arg-type]
        r: float = float(r_raw)
        p_value: float = float(p_val)
        return {
            "metric1": metric1,
            "metric2": metric2,
            "correlation": r,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    # ------------------------------------------------------------------
    # Convenience helper expected by interactive CLI
    # ------------------------------------------------------------------

    def analyze_benchmark_results(self) -> Dict[str, Any]:  # type: ignore[return-value]
        """Return a concise statistical summary for benchmark *results*.

        The current implementation reports mean execution time and success
        rate per technique.  It is intentionally lightweight so as not to
        introduce heavyweight dependencies.  Extend as required for more
        sophisticated analysis (ANOVA, regression, etc.).
        """

        summary: Dict[str, Any] = {}

        techniques = {r["technique"] for r in self.results}
        for tech in techniques:
            tech_results = [r for r in self.results if r["technique"] == tech]
            if not tech_results:
                continue

            mean_time = float(np.mean([float(r.get("duration", 0)) for r in tech_results]))  # type: ignore[arg-type]
            success_rate = sum(1 for r in tech_results if r.get("success", False) or r.get("consensus_reached", False)) / len(tech_results)  # type: ignore[arg-type]

            summary[tech] = {
                "mean_time": mean_time,
                "success_rate": success_rate,
                "n": len(tech_results),
            }

        return summary
