"""
Citation management for research.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Citation:
    """Citation data structure."""

    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    url: Optional[str] = None


class CitationManager:
    """Manages citations for research."""

    def __init__(self):
        self.citations = []

    def add_citation(self, citation: Citation) -> None:
        """Add a citation."""
        self.citations.append(citation)

    def get_citations(self) -> List[Citation]:
        """Get all citations."""
        return self.citations
