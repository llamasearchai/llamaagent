from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.base import AgentRole


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    async def generate_data(self, input_data: Any, **kwargs) -> List[Dict[str, Any]]:
        """Generate training data from input."""
        pass

    @abstractmethod
    async def generate_dataset(self, inputs: List[Any], output_file: str, **kwargs) -> None:
        """Generate a complete dataset and save to file."""
        pass


@dataclass
class DebateNode:
    """Node in the debate tree."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    proposal: str = ""  # The reasoning step or argument
    proposing_agent_role: AgentRole = AgentRole.GENERALIST
    critique: str = ""
    score: float = 0.0
    is_terminal: bool = False
    children: List[str] = field(default_factory=list)


@dataclass
class DebateTrace:
    """Final output format for training data."""

    original_problem: str
    final_answer: str
    full_debate_transcript: List[Dict[str, str]]  # Formatted for ShareGPT
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
                    "parent_id": node.parent_id,
                    "proposal": node.proposal,
                    "proposing_agent_role": node.proposing_agent_role.value,
                    "critique": node.critique,
                    "score": node.score,
                    "is_terminal": node.is_terminal,
                    "children": node.children,
                }
                for node in self.winning_path
            ],
            "total_nodes": self.total_nodes,
            "tree_depth": self.tree_depth,
        }
