"""
Base classes and utilities for data generation.

This module provides base classes and utilities that can be used across different
data generation modules.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BaseDataProcessor(ABC):
    """Base class for data processors."""

    @abstractmethod
    async def process_data(self, data: Any) -> Any:
        """Process input data."""
        pass

    @abstractmethod
    async def generate_dataset(self, inputs: List[Any], output_file: str) -> None:
        """Generate a complete dataset and save to file."""
        pass


@dataclass
class DebateNode:
    """Node in the debate tree."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    proposal: str = ""  # The reasoning step or argument
    proposing_agent_role: str = "generalist"
    critique: str = ""
    score: float = 0.0
    is_terminal: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class DebateTrace:
    """Final output format for training data."""

    original_problem: str
    final_answer: str
    full_debate_transcript: List[Dict[str, Any]] = field(default_factory=list)
    winning_path: List[DebateNode] = field(default_factory=list)
    total_nodes: int = 0
    tree_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_problem": self.original_problem,
            "final_answer": self.final_answer,
            "full_debate_transcript": self.full_debate_transcript,
            "winning_path": [
                {
                    "node_id": node.node_id,
                    "proposal": node.proposal,
                    "critique": node.critique,
                    "score": node.score,
                }
                for node in self.winning_path
            ],
            "total_nodes": self.total_nodes,
            "tree_depth": self.tree_depth,
        }


class DataGenerationConfig:
    """Configuration for data generation."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value


class DataMetrics:
    """Metrics collection for data generation."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[Any]] = {}

    def record(self, metric_name: str, value: Any) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average value for a metric."""
        if metric_name not in self.metrics:
            return None
        values = self.metrics[metric_name]
        if not values:
            return None
        return sum(values) / len(values)

    def get_all(self) -> Dict[str, List[Any]]:
        """Get all recorded metrics."""
        return self.metrics.copy()


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    async def generate_data(self, input_data: Any) -> Any:
        """Generate training data from input."""
        pass


# Define our own base classes instead of importing from data.gdt to avoid conflicts
__all__ = [
    "BaseDataProcessor",
    "DebateNode",
    "DebateTrace",
    "DataGenerationConfig",
    "DataMetrics",
    "DataGenerator",
]