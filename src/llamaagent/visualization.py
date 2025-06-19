from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


class ResearchVisualizer:
    def __init__(self, results: List[Dict[str, Any]], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_performance_comparison(self):
        """Create performance comparison boxplot"""
        plt.figure(figsize=(12, 8))
        data = [self._get_metric_values("duration", tech) for tech in ["SPRE", "GDT", "DTSR", "ATES"]]

        plt.boxplot(data, labels=["SPRE", "GDT", "DTSR", "ATES"])  # type: ignore[arg-name]
        plt.title("Execution Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.savefig(self.output_dir / "performance_comparison.png")
        plt.close()

    def plot_success_rates(self):
        """Create success rate bar chart"""
        plt.figure(figsize=(10, 6))
        techniques = ["SPRE", "GDT", "DTSR", "ATES"]
        success_rates = [self._calculate_success_rate(tech) for tech in techniques]

        plt.bar(techniques, success_rates, color="skyblue")
        plt.title("Success Rate by Technique")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1)
        plt.savefig(self.output_dir / "success_rates.png")
        plt.close()

    def _get_metric_values(self, metric: str, technique: str) -> List[float]:
        return [r[metric] for r in self.results if r["technique"] == technique]

    def _calculate_success_rate(self, technique: str) -> float:
        tech_results = [r for r in self.results if r["technique"] == technique]
        successes = sum(1 for r in tech_results if r.get("success", False) or r.get("consensus_reached", False))
        return successes / len(tech_results) if tech_results else 0


# ---------------------------------------------------------------------------
# Convenience wrapper so that higher-level modules don't need to fiddle with
# the *ResearchVisualizer* class directly.  This keeps the public API small
# and matches older import paths used in the interactive CLI.
# ---------------------------------------------------------------------------


def create_performance_plots(results: List[Dict[str, Any]], output_dir: Path | str) -> None:  # type: ignore[export]
    """Generate all standard performance plots for *results*.

    Parameters
    ----------
    results
        A list of dictionaries produced by *ExperimentRunner* or compatible
        benchmarking routines.  Each entry must contain at least the keys
        ``technique`` and relevant metric names (e.g. ``duration``).
    output_dir
        Directory path where the PNG files should be written.  The directory
        is created if it does not already exist.
    """

    path = Path(output_dir)
    vis = ResearchVisualizer(results, path)
    vis.plot_performance_comparison()
    vis.plot_success_rates()


# Public re-exports
__all__ = [
    "ResearchVisualizer",
    "create_performance_plots",
]
