"""Knowledge module for LlamaAgent."""

from typing import Any, Dict, List


class KnowledgeBase:
    """Basic knowledge base."""
    
    def __init__(self):
        self.knowledge = {}
    
    def add(self, key: str, value: Any) -> None:
        """Add knowledge."""
        self.knowledge[key] = value
    
    def get(self, key: str) -> Any:
        """Get knowledge."""
        return self.knowledge.get(key)

__all__ = ['KnowledgeBase']
