"""Memory module for LlamaAgent."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import SimpleMemory


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


class MemoryManager:
    """Basic memory manager."""
    
    def __init__(self):
        self.memory = {}
    
    def store(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        self.memory[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        return self.memory.get(key)


__all__ = ['MemoryManager', 'MemoryEntry', 'SimpleMemory']
