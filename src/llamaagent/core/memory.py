"""
Memory implementation for LlamaAgent
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MemoryItem(BaseModel):
    """
    A single memory item
    """

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def __str__(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp)
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {self.content}"


class ShortTermMemory(BaseModel):
    """
    Short-term memory store (recent history)
    """

    capacity: int = 50
    items: List[MemoryItem] = Field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        """
        Add an item to short-term memory
        """
        self.items.append(item)

        # Ensure we don't exceed capacity
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def get_recent(self, n: int = 10) -> List[MemoryItem]:
        """
        Get the n most recent items
        """
        return self.items[-n:]

    def clear(self) -> None:
        """
        Clear the short-term memory
        """
        self.items = []

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """
        Simple search for items containing the query
        """
        # Basic string matching - in a real implementation, this would use embeddings
        results = [item for item in self.items if query.lower() in item.content.lower()]
        return results[:limit]


class LongTermMemory(BaseModel):
    """
    Long-term memory store (persistent)
    """

    db_path: Optional[str] = None
    enabled: bool = False
    items: Dict[str, MemoryItem] = Field(default_factory=dict)

    def add(self, item: MemoryItem) -> None:
        """
        Add an item to long-term memory
        """
        if not self.enabled:
            return

        self.items[item.id] = item
        self._save_to_disk()

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """
        Get an item by ID
        """
        return self.items.get(item_id)

    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """
        Simple search for items containing the query
        """
        # Basic string matching - in a real implementation, this would use embeddings
        results = [
            item
            for item in self.items.values()
            if query.lower() in item.content.lower()
        ]
        return results[:limit]

    def _save_to_disk(self) -> None:
        """
        Save items to disk if db_path is set
        """
        if not self.db_path:
            return

        try:
            with open(self.db_path, "w") as f:
                # Convert items to dicts for serialization
                items_dict = {k: v.dict() for k, v in self.items.items()}
                json.dump(items_dict, f)
        except Exception as e:
            logger.error(f"Error saving long-term memory to disk: {e}")

    def _load_from_disk(self) -> None:
        """
        Load items from disk if db_path is set
        """
        if not self.db_path:
            return

        try:
            with open(self.db_path, "r") as f:
                items_dict = json.load(f)
                self.items = {k: MemoryItem(**v) for k, v in items_dict.items()}
        except FileNotFoundError:
            logger.warning(f"Memory file not found: {self.db_path}")
        except Exception as e:
            logger.error(f"Error loading long-term memory from disk: {e}")


class Memory(BaseModel):
    """
    Complete memory system with short-term and long-term storage
    """

    short_term: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = Field(default_factory=LongTermMemory)
    short_term_capacity: int = 50
    long_term_enabled: bool = False
    long_term_db_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        # Configure short-term memory
        self.short_term.capacity = self.short_term_capacity

        # Configure long-term memory
        self.long_term.enabled = self.long_term_enabled
        if self.long_term_enabled and self.long_term_db_path:
            self.long_term.db_path = self.long_term_db_path
            self.long_term._load_from_disk()

    def add(self, content: str, metadata: Dict[str, Any] = None, **kwargs) -> None:
        """
        Add a memory item to both short-term and long-term memory
        """
        import uuid

        if metadata is None:
            metadata = {}

        # Add any additional keyword arguments to metadata
        metadata.update(kwargs)

        # Create the memory item
        item = MemoryItem(id=str(uuid.uuid4()), content=content, metadata=metadata)

        # Add to short-term memory
        self.short_term.add(item)

        # Add to long-term memory if enabled
        if self.long_term.enabled:
            self.long_term.add(item)

    def get_context(self, query: str = None, limit: int = 5) -> str:
        """
        Get a context string for a query, combining recent and relevant memories
        """
        # Get recent items
        recent_items = self.short_term.get_recent(limit)

        # If query is provided, search for relevant items
        relevant_items = []
        if query and self.long_term.enabled:
            relevant_items = self.long_term.search(query, limit)

        # Combine without duplicates
        all_items = []
        seen_ids = set()

        for item in recent_items + relevant_items:
            if item.id not in seen_ids:
                all_items.append(item)
                seen_ids.add(item.id)

        # Sort by timestamp
        all_items.sort(key=lambda x: x.timestamp)

        # Format into a context string
        if not all_items:
            return "No relevant memory items found."

        context = "Memory items:\n\n"
        for item in all_items:
            context += f"{str(item)}\n"

        return context

    def clear_short_term(self) -> None:
        """
        Clear short-term memory
        """
        self.short_term.clear()
