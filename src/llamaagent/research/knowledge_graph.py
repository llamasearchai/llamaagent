"""
Knowledge graph implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KnowledgeNode:
    """Knowledge graph node."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Knowledge graph implementation."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
