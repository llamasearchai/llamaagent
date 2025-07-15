"""Monitoring module for LlamaAgent."""

from typing import Any, Dict


class Monitor:
    """Basic monitoring."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, metric: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[metric] = value

__all__ = ['Monitor']
