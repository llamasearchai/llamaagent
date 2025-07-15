"""
Visualization utilities for LlamaAgent research and performance metrics.

This module provides comprehensive visualization capabilities for analyzing
agent performance, research results, and system metrics.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

logger = logging.getLogger(__name__)


class ResearchVisualizer:
    """Visualizer for research experiments and performance metrics."""

    def __init__(
        self,
        results: List[Dict[str, Any]],
        output_dir: Union[Path, str],
        style: str = "whitegrid",
    ) -> None:
        """Initialize the visualizer with results and output directory."""
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Set visualization style
        sns.set_theme(style=style)  # type: ignore[misc, call-arg]
        plt.rcParams["figure.dpi"] = 100  # type: ignore[attr-defined]
        plt.rcParams["savefig.dpi"] = 300  # type: ignore[attr-defined]
        plt.rcParams["font.size"] = 10  # type: ignore[attr-defined]

    def plot_performance_comparison(self) -> None:
        """Create performance comparison boxplot."""
        plt.figure(figsize=(12, 8))  # type: ignore[misc]

        techniques = self._get_unique_techniques()
        data: List[List[float]] = []

        for tech in techniques:
            durations = self._get_metric_values("duration", tech)
            if durations:
                data.append(durations)

        if data:
            box_plot = plt.boxplot(data, labels=techniques, patch_artist=True)  # type: ignore[misc]

            # Customize box colors
            colors = sns.color_palette("husl", len(techniques))  # type: ignore[misc]
            for patch, color in zip(box_plot["boxes"], colors, strict=False):
                patch.set_facecolor(color)  # type: ignore[misc]

            plt.title("Execution Time Comparison by Technique", fontsize=14, fontweight="bold")  # type: ignore[misc]
            plt.ylabel("Time (seconds)", fontsize=12)  # type: ignore[misc]
            plt.xlabel("Technique", fontsize=12)  # type: ignore[misc]
            plt.grid(True, alpha=0.3)  # type: ignore[misc]

            # Add median values as text
            medians = [np.median(d) for d in data]
            for i, median in enumerate(medians):
                plt.text(i + 1, median, f"{median:.2f}", horizontalalignment="center", fontsize=9)  # type: ignore[misc]

        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / "performance_comparison.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def plot_success_rates(self) -> None:
        """Create success rate bar chart."""
        plt.figure(figsize=(10, 6))  # type: ignore[misc]

        techniques = self._get_unique_techniques()
        success_rates = [self._calculate_success_rate(tech) for tech in techniques]

        bars = plt.bar(
            techniques,
            success_rates,
            color=sns.color_palette("viridis", len(techniques)),
        )  # type: ignore[misc]

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates, strict=False):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.title("Success Rate by Technique", fontsize=14, fontweight="bold")  # type: ignore[misc]
        plt.ylabel("Success Rate", fontsize=12)  # type: ignore[misc]
        plt.xlabel("Technique", fontsize=12)  # type: ignore[misc]
        plt.ylim(0, 1.1)  # type: ignore[misc]
        plt.grid(True, axis="y", alpha=0.3)  # type: ignore[misc]

        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))  # type: ignore[misc]

        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / "success_rates.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def plot_metric_distribution(
        self, metric: str, title: Optional[str] = None
    ) -> None:
        """Create distribution plot for a specific metric."""
        plt.figure(figsize=(10, 6))  # type: ignore[misc]

        techniques = self._get_unique_techniques()

        for tech in techniques:
            values = self._get_metric_values(metric, tech)
            if values:
                sns.kdeplot(data=values, label=tech, fill=True, alpha=0.3)

        plt.title(
            title or f"Distribution of {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )  # type: ignore[misc]
        plt.xlabel(metric.replace("_", " ").title(), fontsize=12)  # type: ignore[misc]
        plt.ylabel("Density", fontsize=12)  # type: ignore[misc]
        plt.legend()  # type: ignore[misc]
        plt.grid(True, alpha=0.3)  # type: ignore[misc]

        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / f"{metric}_distribution.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def plot_correlation_heatmap(self, metrics: List[str]) -> None:
        """Create correlation heatmap for specified metrics."""
        # Extract metric values
        data_dict = {}
        for metric in metrics:
            values = []
            for result in self.results:
                if metric in result:
                    try:
                        values.append(float(result[metric]))
                    except (ValueError, TypeError):
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            data_dict[metric] = values

        # Create DataFrame-like structure for correlation
        df = pd.DataFrame(data_dict)

        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))  # type: ignore[misc]
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Metric Correlation Heatmap", fontsize=14, fontweight="bold")  # type: ignore[misc]
        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / "correlation_heatmap.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def plot_timeline(self) -> None:
        """Create timeline visualization of experiments."""
        plt.figure(figsize=(12, 6))  # type: ignore[misc]

        # Sort results by timestamp if available
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

        techniques = self._get_unique_techniques()
        colors = dict(
            zip(techniques, sns.color_palette("husl", len(techniques)), strict=False)
        )

        for i, result in enumerate(sorted_results):
            tech = result.get("technique", "Unknown")
            success = result.get("success", False)
            duration = result.get("duration", 0)

            # Plot as scatter with different markers for success/failure
            marker = "o" if success else "x"
            plt.scatter(
                i,
                duration,
                color=colors.get(tech, "gray"),
                marker=marker,
                s=100,
                alpha=0.7,
                label=tech if i == 0 else "",
            )

        plt.title("Experiment Timeline", fontsize=14, fontweight="bold")  # type: ignore[misc]
        plt.xlabel("Experiment Index", fontsize=12)  # type: ignore[misc]
        plt.ylabel("Duration (seconds)", fontsize=12)  # type: ignore[misc]
        plt.grid(True, alpha=0.3)  # type: ignore[misc]

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        plt.legend(by_label.values(), by_label.keys())  # type: ignore[misc]

        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / "experiment_timeline.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def plot_agent_performance_radar(self, agent_name: str, metrics: List[str]) -> None:
        """Create radar chart for agent performance across multiple metrics."""
        # Filter results for specific agent
        agent_results = [r for r in self.results if r.get("agent_name") == agent_name]

        if not agent_results:
            logger.warning(f"No results found for agent: {agent_name}")
            return

        # Calculate average for each metric
        avg_metrics = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in agent_results]
            avg_metrics[metric] = np.mean(values) if values else 0

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values = list(avg_metrics.values())

        # Close the plot
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))  # type: ignore[misc]

        ax.plot(angles, values, "o-", linewidth=2, color="blue")  # type: ignore[misc]
        ax.fill(angles, values, alpha=0.25, color="blue")  # type: ignore[misc]
        ax.set_xticks(angles[:-1])  # type: ignore[misc]
        ax.set_xticklabels(metrics)  # type: ignore[misc]
        ax.set_ylim(0, max(values) * 1.1 if values else 1)  # type: ignore[misc]

        plt.title(
            f"Agent Performance: {agent_name}", fontsize=14, fontweight="bold", pad=20
        )  # type: ignore[misc]
        plt.tight_layout()  # type: ignore[misc]
        plt.savefig(self.output_dir / f"agent_radar_{agent_name}.png")  # type: ignore[misc]
        plt.close()  # type: ignore[misc]

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if not self.results:
            logger.warning("No results for summary report")
            return {}

        summary: Dict[str, Any] = {
            "total_experiments": len(self.results),
            "techniques": self._get_unique_techniques(),
            "overall_success_rate": self._calculate_overall_success_rate(),
            "metrics_summary": {},
            "technique_summary": {},
        }

        # Summarize by technique
        for tech in summary["techniques"]:
            tech_results = [r for r in self.results if r.get("technique") == tech]
            summary["technique_summary"][tech] = {
                "count": len(tech_results),
                "success_rate": self._calculate_success_rate(tech),
                "avg_duration": np.mean([r.get("duration", 0) for r in tech_results]),
                "median_duration": np.median(
                    [r.get("duration", 0) for r in tech_results]
                ),
            }

        # Save summary as JSON
        with (self.output_dir / "summary_report.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _get_unique_techniques(self) -> List[str]:
        """Get unique technique names from results."""
        techniques = set(str(r.get("technique", "Unknown")) for r in self.results)
        return sorted(list(techniques))

    def _get_metric_values(self, metric: str, technique: str) -> List[float]:
        """Extract metric values for a specific technique."""
        values = []
        for result in self.results:
            if result.get("technique") == technique and metric in result:
                try:
                    values.append(float(result[metric]))
                except (ValueError, TypeError):
                    continue
        return values

    def _calculate_success_rate(self, technique: str) -> float:
        """Calculate success rate for a specific technique."""
        tech_results = [r for r in self.results if r.get("technique") == technique]
        if not tech_results:
            return 0.0

        successes = sum(
            1
            for r in tech_results
            if r.get("success", False) or r.get("consensus_reached", False)
        )
        return successes / len(tech_results)

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all experiments."""
        if not self.results:
            return 0.0

        successes = sum(
            1
            for r in self.results
            if r.get("success", False) or r.get("consensus_reached", False)
        )
        return successes / len(self.results)


def create_performance_plots(
    results: List[Dict[str, Any]], output_dir: Union[Path, str] = "./visualizations"
) -> None:
    """Generate all standard performance plots for results.

    This is a convenience wrapper that creates all standard visualizations
    for experiment results.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        A list of dictionaries produced by ExperimentRunner or compatible
        benchmarking routines. Each entry must contain at least the keys
        'technique' and relevant metric names (e.g., 'duration').
    output_dir : Union[Path, str]
        Directory path where the PNG files should be written. The directory
        is created if it does not already exist.
    """
    vis = ResearchVisualizer(results, output_dir)

    # Generate standard plots
    vis.plot_performance_comparison()
    vis.plot_success_rates()
    vis.plot_timeline()

    # Generate distribution plots for common metrics
    for metric in ["duration", "memory_usage", "token_count"]:
        if any(metric in r for r in results):
            vis.plot_metric_distribution(metric)

    # Generate summary report
    vis.generate_summary_report()

    logger.info(f"Visualizations saved to: {output_dir}")


# Public exports
__all__ = ["ResearchVisualizer", "create_performance_plots"]
