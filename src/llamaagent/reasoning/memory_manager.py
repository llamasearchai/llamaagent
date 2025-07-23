"""
Memory Manager - Enterprise Implementation

This module implements advanced memory management for reasoning systems, including:
- Working memory for temporary data
- Long-term memory for persistent knowledge
- Episodic memory for experiences
- Semantic memory for concepts
- Memory consolidation and retrieval

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sklearn_available = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    sklearn_available = False

try:
    import redis

    redis_available = True
except ImportError:
    redis = None
    redis_available = False

SKLEARN_AVAILABLE = sklearn_available
REDIS_AVAILABLE = redis_available

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory supported by the system."""

    WORKING = "working"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryItem:
    """Individual memory item with metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.WORKING
    tags: Set[str] = field(default_factory=set)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default expiration for working memory."""
        if self.memory_type == MemoryType.WORKING and self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=1)

    def access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "tags": list(self.tags),
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            tags=set(data["tags"]),
            importance=data["importance"],
            access_count=data["access_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data["expires_at"]
            else None,
            metadata=data["metadata"],
        )


class MemoryManager:
    """Advanced memory management system with multiple memory types."""

    def __init__(
        self,
        redis_client: Any = None,
        max_working_memory: int = 1000,
        max_long_term_memory: int = 10000,
        consolidation_threshold: int = 5,
    ):
        # Redis setup (optional)
        if REDIS_AVAILABLE and redis_client is not None:
            self.redis_client = redis_client
        elif REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(decode_responses=True)
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None

        self.max_working_memory = max_working_memory
        self.max_long_term_memory = max_long_term_memory
        self.consolidation_threshold = consolidation_threshold

        # In-memory storage
        self.working_memory: Dict[str, MemoryItem] = {}
        self.long_term_memory: Dict[str, MemoryItem] = {}
        self.episodic_memory: Dict[str, MemoryItem] = {}
        self.semantic_memory: Dict[str, MemoryItem] = {}
        self.procedural_memory: Dict[str, MemoryItem] = {}

        # Semantic network for concept relationships
        self.semantic_network: Dict[str, Set[str]] = {}

        # Text vectorization for similarity (if sklearn available)
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.vectorizer = None

        self.logger = logger

    async def store_memory(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        tags: Optional[Set[str]] = None,
        importance: float = 0.5,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store item in memory."""
        memory_item = MemoryItem(
            content=content,
            memory_type=memory_type,
            tags=tags or set(),
            importance=importance,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        # Store based on memory type
        if memory_type == MemoryType.WORKING:
            await self._store_working_memory(memory_item)
        elif memory_type == MemoryType.LONG_TERM:
            await self._store_long_term_memory(memory_item)
        elif memory_type == MemoryType.EPISODIC:
            await self._store_episodic_memory(memory_item)
        elif memory_type == MemoryType.SEMANTIC:
            await self._store_semantic_memory(memory_item)
        elif memory_type == MemoryType.PROCEDURAL:
            await self._store_procedural_memory(memory_item)
        self.logger.info(f"Stored {memory_type.value} memory: {memory_item.id}")
        return memory_item.id

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        # Search all memory stores
        stores = [
            self.working_memory,
            self.long_term_memory,
            self.episodic_memory,
            self.semantic_memory,
            self.procedural_memory,
        ]

        for store in stores:
            if memory_id in store:
                item = store[memory_id]
                item.access()
                await self._check_consolidation(item)
                return item

        # Try Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(f"memory:{memory_id}")
                if data:
                    item = MemoryItem.from_dict(json.loads(data))
                    item.access()
                    return item
            except Exception as e:
                self.logger.error(f"Redis get error: {e}")
        return None

    async def search_memory(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[Set[str]] = None,
        similarity_threshold: float = 0.3,
        limit: int = 10,
    ) -> List[MemoryItem]:
        """Search memory using various criteria."""
        results = []

        # Search all memory stores
        stores = [
            self.working_memory,
            self.long_term_memory,
            self.episodic_memory,
            self.semantic_memory,
            self.procedural_memory,
        ]

        for store in stores:
            for item in store.values():
                if self._matches_criteria(item, memory_types, tags):
                    similarity = self._calculate_similarity(query, item.content)
                    if similarity >= similarity_threshold:
                        results.append((item, similarity))
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:limit]]

    async def consolidate_memory(self) -> None:
        """Consolidate frequently accessed working memory to long-term."""
        consolidation_candidates = []

        for item in self.working_memory.values():
            if item.access_count >= self.consolidation_threshold:
                consolidation_candidates.append(item)
        for item in consolidation_candidates:
            await self._consolidate_item(item)

    async def cleanup_expired_memory(self) -> None:
        """Remove expired memory items."""
        current_time = datetime.now(timezone.utc)
        expired_ids = []

        for item_id, item in self.working_memory.items():
            if item.expires_at and current_time > item.expires_at:
                expired_ids.append(item_id)
        for item_id in expired_ids:
            del self.working_memory[item_id]
            self.logger.debug(f"Expired working memory: {item_id}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "working_memory_count": len(self.working_memory),
            "long_term_memory_count": len(self.long_term_memory),
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "procedural_memory_count": len(self.procedural_memory),
            "semantic_concepts": len(self.semantic_network),
            "redis_available": self.redis_client is not None,
        }

    async def get_related_concepts(self, concept: str) -> List[str]:
        """Get concepts related to the given concept."""
        return list(self.semantic_network.get(concept, set()))

    # Private methods

    async def _store_working_memory(self, item: MemoryItem) -> None:
        """Store in working memory with capacity management."""
        if len(self.working_memory) >= self.max_working_memory:
            await self._evict_working_memory()

        self.working_memory[item.id] = item
        await self._persist_to_redis(item, "working")

    async def _store_long_term_memory(self, item: MemoryItem) -> None:
        """Store in long-term memory."""
        self.long_term_memory[item.id] = item
        await self._persist_to_redis(item, "long_term")

    async def _store_episodic_memory(self, item: MemoryItem) -> None:
        """Store episodic memory (experiences)."""
        item.metadata["temporal_context"] = {
            "timestamp": item.created_at.isoformat(),
            "context_type": "episodic",
        }
        self.episodic_memory[item.id] = item
        await self._persist_to_redis(item, "episodic")

    async def _store_semantic_memory(self, item: MemoryItem) -> None:
        """Store semantic memory (concepts and relationships)."""
        concepts = self._extract_concepts(item.content)
        for concept in concepts:
            if concept not in self.semantic_network:
                self.semantic_network[concept] = set()

            # Link related concepts
            for other_concept in concepts:
                if concept != other_concept:
                    self.semantic_network[concept].add(other_concept)
        self.semantic_memory[item.id] = item
        await self._persist_to_redis(item, "semantic")

    async def _store_procedural_memory(self, item: MemoryItem) -> None:
        """Store procedural memory (skills and procedures)."""
        item.metadata["procedure_type"] = "skill"
        item.metadata["execution_count"] = 0

        self.procedural_memory[item.id] = item
        await self._persist_to_redis(item, "procedural")

    async def _persist_to_redis(self, item: MemoryItem, category: str) -> None:
        """Persist memory item to Redis."""
        if not self.redis_client:
            return

        try:
            key = f"memory:{category}:{item.id}"
            data = json.dumps(item.to_dict())
            # Set TTL based on memory type
            ttl_map = {
                "working": 3600,  # 1 hour
                "long_term": 86400 * 30,  # 30 days
                "episodic": 86400 * 7,  # 7 days
                "semantic": 86400 * 30,  # 30 days
                "procedural": 86400 * 30,  # 30 days
            }

            ttl = ttl_map.get(category, 86400)
            self.redis_client.setex(key, ttl, data)
        except Exception as e:
            self.logger.error(f"Redis persist error: {e}")

    async def _evict_working_memory(self) -> None:
        """Evict least important items from working memory."""
        items = list(self.working_memory.values())
        items.sort(key=lambda x: (x.importance, x.last_accessed))
        # Remove bottom 10% of items
        evict_count = max(1, len(items) // 10)
        for item in items[:evict_count]:
            del self.working_memory[item.id]

    async def _consolidate_item(self, item: MemoryItem) -> None:
        """Consolidate frequently accessed working memory to long-term."""
        if item.memory_type == MemoryType.WORKING:
            item.memory_type = MemoryType.LONG_TERM
            item.importance = min(1.0, item.importance + 0.1)
            await self._store_long_term_memory(item)
            if item.id in self.working_memory:
                del self.working_memory[item.id]

            self.logger.info(f"Consolidated memory to long-term: {item.id}")

    async def _check_consolidation(self, item: MemoryItem) -> None:
        """Check if item should be consolidated."""
        if (
            item.memory_type == MemoryType.WORKING
            and item.access_count >= self.consolidation_threshold
        ):
            await self._consolidate_item(item)

    def _matches_criteria(
        self,
        item: MemoryItem,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[Set[str]] = None,
    ) -> bool:
        """Check if item matches search criteria."""
        if memory_types and item.memory_type not in memory_types:
            return False

        if tags and not tags.intersection(item.tags):
            return False

        return True

    def _calculate_similarity(self, query: str, content: Any) -> float:
        """Calculate similarity between query and content."""
        if not isinstance(content, str):
            content = str(content)
        if SKLEARN_AVAILABLE and self.vectorizer:
            try:
                texts = [query, content]
                vectors = self.vectorizer.fit_transform(texts).toarray()
                similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
                return similarity
            except Exception as e:
                logger.error(f"Error: {e}")
        # Fallback to simple string matching
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        return len(intersection) / len(union) if union else 0.0

    def _extract_concepts(self, content: Any) -> List[str]:
        """Extract concepts from content."""
        if not isinstance(content, str):
            content = str(content)
        # Simple concept extraction - split on common delimiters
        words = content.lower().replace(',', ' ').replace('.', ' ').split()

        # Filter out common words and short words
        stop_words = {
            'the',
            'a',
            'an',
            'and',
            'or',
            'but',
            'in',
            'on',
            'at',
            'to',
            'for',
            'of',
            'with',
            'by',
            'is',
            'are',
            'was',
            'were',
        }
        concepts = [word for word in words if len(word) > 2 and word not in stop_words]

        return list(set(concepts))  # Remove duplicates
