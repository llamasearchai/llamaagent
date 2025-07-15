"""
Knowledge base module for cooperative insights and learning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CooperationKnowledge:
    """Vector-based knowledge base for cooperative insights."""

    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the knowledge base."""
        self.persist_path = persist_path
        self.insights: List[str] = []
        self.embeddings: List[List[float]] = []
        self._load_if_exists()

    def add_insight(self, insight_text: str) -> None:
        """Add a new cooperative insight."""
        self.insights.append(insight_text)

        # Simple embedding (just use text length and word count as features)
        embedding = self._simple_embed(insight_text)
        self.embeddings.append(embedding)

        if self.persist_path:
            self._save()

    def retrieve_relevant_insights(
        self, task_description: str, top_k: int = 3
    ) -> List[str]:
        """Retrieve most relevant insights for a task."""
        if not self.insights:
            return []

        task_embedding = self._simple_embed(task_description)

        # Calculate similarities
        similarities: List[Tuple[float, int]] = []
        for i, insight_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(task_embedding, insight_embedding)
            similarities.append(similarity, i)

        # Sort by similarity and return top-k
        similarities.sort(reverse=True)
        return [self.insights[i] for _, i in similarities[:top_k]]

    def _simple_embed(self, text: str) -> List[float]:
        """Simple embedding based on text features."""
        words = text.split()
        return [
            len(text),  # character count
            len(words),  # word count
            len(set(words)),  # unique words
            text.count("if"),  # conditional statements
            text.count("when"),  # temporal conditions
            text.count("should"),  # recommendations
        ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _save(self) -> None:
        """Save knowledge base to disk."""
        if not self.persist_path:
            return

        data = {
            "insights": self.insights,
            "embeddings": self.embeddings,
        }

        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_if_exists(self) -> None:
        """Load knowledge base if it exists."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.insights = data.get("insights", [])
            self.embeddings = data.get("embeddings", [])
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            # Reset if corrupted
            self.insights = []
            self.embeddings = []

    def export_insights(self, output_file: str) -> None:
        """Export insights to a text file."""
        with open(output_file, "w") as f:
            for i, insight in enumerate(self.insights, 1):
                f.write(f"{i}. {insight}\n\n")

    def get_stats(self) -> Dict[str, float]:
        """Get knowledge base statistics."""
        return {
            "total_insights": len(self.insights),
            "avg_insight_length": (
                sum(len(insight) for insight in self.insights) / len(self.insights)
                if self.insights
                else 0
            ),
        }
