"""
Report generation for research.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Report:
    """Research report data structure."""

    title: str
    content: str
    sections: List[str] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)


class ReportGenerator:
    """Generates research reports."""

    def __init__(self):
        self.reports = []

    def create_report(self, title: str, content: str) -> Report:
        """Create a new report."""
        report = Report(title=title, content=content)
        self.reports.append(report)
        return report

    def add_figure(self, report: Report, figure: Dict[str, Any]) -> None:
        """Add a figure to a report."""
        report.figures.append(figure)
