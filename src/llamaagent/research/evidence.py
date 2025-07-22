"""
Evidence management for research.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Evidence:
    """Evidence data structure."""

    content: str
    source: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceManager:
    """Manages evidence for research."""

    def __init__(self):
        self.evidence_list = []

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence."""
        self.evidence_list.append(evidence)

    def get_evidence(self) -> List[Evidence]:
        """Get all evidence."""
        return self.evidence_list
