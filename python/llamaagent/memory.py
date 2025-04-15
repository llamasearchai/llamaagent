"""
Memory component for the LlamaAgent framework.

The Memory module provides mechanisms for storing, retrieving, and managing
agent memory, including:
- Short-term memory for the current session
- Long-term memory via vector storage
- Memory organization and retrieval by relevance
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .config import MemoryConfig
from .types import MemoryItem, MemoryRetrieval

logger = logging.getLogger(__name__)


class MemoryItemImpl:
    """Implementation of a memory item stored in memory."""

    def __init__(
        self,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize a memory item.

        Args:
            content: The content to store
            metadata: Additional metadata about the item
            id: Unique identifier (generated if not provided)
            timestamp: When the item was created (now if not provided)
        """
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()


class MemoryRetrievalImpl:
    """Implementation of the result of a memory retrieval operation."""

    def __init__(self, items: List[MemoryItem], relevance: float = 0.0):
        """
        Initialize a memory retrieval result.

        Args:
            items: List of retrieved memory items
            relevance: Score indicating the overall relevance (0-1)
        """
        self.items = items
        self.relevance = relevance


class Memory:
    """
    Memory system for storing and retrieving information.

    The Memory class provides a flexible system for managing both short-term
    context and long-term knowledge, with intelligent retrieval based on
    relevance to the current context.

    Examples:
        >>> from llamaagent import Memory
        >>> memory = Memory()
        >>> memory.add({"type": "fact", "content": "Paris is the capital of France"})
        >>> results = memory.retrieve("What is the capital of France?")
        >>> for item in results.items:
        ...     print(item.content)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        vector_store: Optional[Any] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize the memory system.

        Args:
            config: Configuration for the memory system
            vector_store: Optional external vector store for long-term memory
            embedding_function: Function to convert text to vector embeddings
        """
        self.config = config or MemoryConfig()
        self.vector_store = vector_store
        self.embedding_function = embedding_function

        # In-memory storage (for short-term memory or when no vector store is provided)
        self._items: List[MemoryItemImpl] = []

        logger.debug("Memory system initialized")

    def add(self, item: Union[Dict[str, Any], MemoryItem]) -> str:
        """
        Add an item to memory.

        Args:
            item: The item to add, either as a dictionary or MemoryItem object

        Returns:
            The ID of the added item
        """
        # Convert to internal representation if needed
        if not isinstance(item, MemoryItemImpl):
            # Extract fields if it's a MemoryItem
            if (
                hasattr(item, "id")
                and hasattr(item, "content")
                and hasattr(item, "metadata")
                and hasattr(item, "timestamp")
            ):
                memory_item = MemoryItemImpl(
                    id=getattr(item, "id"),
                    content=getattr(item, "content"),
                    metadata=getattr(item, "metadata"),
                    timestamp=getattr(item, "timestamp"),
                )
            # Otherwise assume it's a dictionary
            else:
                memory_item = MemoryItemImpl(
                    content=item.get("content", item),
                    metadata=item.get("metadata", {}),
                    id=item.get("id"),
                    timestamp=item.get("timestamp"),
                )
        else:
            memory_item = item

        # Store in internal list (short-term memory)
        self._items.append(memory_item)

        # Limit the size of short-term memory if configured
        if (
            self.config.short_term_limit
            and len(self._items) > self.config.short_term_limit
        ):
            self._items = self._items[-self.config.short_term_limit :]

        # Store in vector database if available (long-term memory)
        if self.vector_store and self.embedding_function:
            try:
                # Convert content to string if needed
                content_str = (
                    str(memory_item.content)
                    if not isinstance(memory_item.content, str)
                    else memory_item.content
                )

                # Generate embedding
                embedding = self.embedding_function(content_str)

                # Store in vector database
                self.vector_store.add(
                    id=memory_item.id,
                    vector=embedding,
                    metadata={
                        "content": content_str,
                        "timestamp": memory_item.timestamp.isoformat(),
                        **memory_item.metadata,
                    },
                )

                logger.debug(f"Added item {memory_item.id} to vector store")

            except Exception as e:
                logger.warning(f"Failed to add item to vector store: {e}")

        logger.debug(f"Added item {memory_item.id} to memory")
        return memory_item.id

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        filter_func: Optional[Callable[[MemoryItem], bool]] = None,
    ) -> MemoryRetrieval:
        """
        Retrieve items from memory relevant to the query.

        Args:
            query: The query to match against
            limit: Maximum number of items to return
            threshold: Minimum relevance score (0-1)
            filter_func: Optional function to filter results

        Returns:
            A MemoryRetrieval object with the matching items
        """
        logger.debug(f"Retrieving items for query: {query}")

        items: List[MemoryItem] = []
        relevance_score = 0.0

        # If vector store is available, use semantic search
        if self.vector_store and self.embedding_function:
            try:
                # Generate embedding for the query
                query_embedding = self.embedding_function(query)

                # Search in vector store
                results = self.vector_store.search(
                    query_vector=query_embedding, limit=limit, threshold=threshold
                )

                # Convert results to memory items
                for result in results:
                    metadata = result.get("metadata", {})
                    content = metadata.pop("content", "")

                    # Parse timestamp if available
                    timestamp = datetime.now()
                    if "timestamp" in metadata:
                        try:
                            timestamp = datetime.fromisoformat(metadata["timestamp"])
                        except (ValueError, TypeError):
                            pass

                    item = MemoryItemImpl(
                        id=result.get("id", str(uuid.uuid4())),
                        content=content,
                        metadata=metadata,
                        timestamp=timestamp,
                    )

                    # Apply filter if provided
                    if filter_func is None or filter_func(item):
                        items.append(item)

                # Set overall relevance score
                if results:
                    relevance_score = sum(r.get("score", 0) for r in results) / len(
                        results
                    )

                logger.debug(f"Retrieved {len(items)} items from vector store")

            except Exception as e:
                logger.warning(f"Failed to retrieve from vector store: {e}")

        # Fall back to simple keyword matching for in-memory items
        if not items or not self.vector_store:
            query_lower = query.lower()
            matched_items = []

            for item in self._items:
                content_str = (
                    str(item.content)
                    if not isinstance(item.content, str)
                    else item.content
                )

                # Simple relevance based on substring matching
                relevance = 0.0
                if query_lower in content_str.lower():
                    # Calculate a crude relevance score based on length ratio
                    relevance = min(1.0, len(query) / max(1, len(content_str)))

                if relevance > threshold:
                    item.metadata["relevance"] = relevance
                    matched_items.append((item, relevance))

            # Sort by relevance (descending)
            matched_items.sort(key=lambda x: x[1], reverse=True)

            # Apply limit
            matched_items = matched_items[:limit]

            # Apply filter if provided
            if filter_func:
                matched_items = [
                    (item, score) for item, score in matched_items if filter_func(item)
                ]

            # Extract items and calculate average relevance
            items = [item for item, _ in matched_items]
            if matched_items:
                relevance_score = sum(score for _, score in matched_items) / len(
                    matched_items
                )

            logger.debug(f"Retrieved {len(items)} items from in-memory storage")

        return MemoryRetrievalImpl(items, relevance_score)

    def get(self, id: str) -> Optional[MemoryItem]:
        """
        Get a specific item by ID.

        Args:
            id: The ID of the item to retrieve

        Returns:
            The memory item if found, None otherwise
        """
        # Check in-memory storage first
        for item in self._items:
            if item.id == id:
                return item

        # Check vector store if available
        if self.vector_store:
            try:
                result = self.vector_store.get(id)
                if result:
                    metadata = result.get("metadata", {})
                    content = metadata.pop("content", "")

                    # Parse timestamp if available
                    timestamp = datetime.now()
                    if "timestamp" in metadata:
                        try:
                            timestamp = datetime.fromisoformat(metadata["timestamp"])
                        except (ValueError, TypeError):
                            pass

                    return MemoryItemImpl(
                        id=id, content=content, metadata=metadata, timestamp=timestamp
                    )
            except Exception as e:
                logger.warning(f"Failed to get item from vector store: {e}")

        return None

    def clear(self) -> None:
        """Clear all items from short-term memory."""
        self._items = []
        logger.debug("Cleared short-term memory")

    def __len__(self) -> int:
        """Return the number of items in short-term memory."""
        return len(self._items)
