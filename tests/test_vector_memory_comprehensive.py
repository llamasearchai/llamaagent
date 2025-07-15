#!/usr/bin/env python3
"""Comprehensive tests for vector memory with mock implementations.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from numpy.typing import NDArray


class MockVectorMemory:
    """Mock vector memory implementation for testing."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension: int = dimension
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.index_count: int = 0

    async def add_vector(
        self,
        vector: List[float],
        metadata: Dict[str, Any],
        vector_id: Optional[str] = None,
    ) -> str:
        """Add a vector to memory."""
        if vector_id is None:
            vector_id = f"vec_{self.index_count}"
            self.index_count += 1

        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match expected {self.dimension}"
            )

        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata
        return vector_id

    async def search(
        self, query_vector: List[float], top_k: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match expected {self.dimension}"
            )

        results: List[Dict[str, Any]] = []
        query_np: NDArray[np.float32] = np.array(query_vector, dtype=np.float32)

        for vec_id, vector in self.vectors.items():
            # Calculate cosine similarity
            vec_np: NDArray[np.float32] = np.array(vector, dtype=np.float32)
            similarity: float = float(
                np.dot(query_np, vec_np)
                / (np.linalg.norm(query_np) * np.linalg.norm(vec_np))
            )

            if similarity >= threshold:
                results.append(
                    {
                        "id": vec_id,
                        "similarity": similarity,
                        "metadata": self.metadata[vec_id],
                        "vector": vector,
                    }
                )

        # Sort by similarity and return top_k
        results.sort(key=lambda x: float(x["similarity"]), reverse=True)
        return results[:top_k]


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension: int = dimension
        self.call_count: int = 0

    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding for text."""
        self.call_count += 1
        # Generate deterministic but varied embeddings based on text hash
        seed: int = hash(text) % 1000
        np.random.seed(seed)
        embedding: NDArray[np.float64] = np.random.normal(0, 1, self.dimension)
        return embedding.tolist()


@pytest.fixture
def mock_vector_memory() -> MockVectorMemory:
    """Provide a mock vector memory instance."""
    return MockVectorMemory()


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Provide a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.mark.asyncio
async def test_vector_memory_add_and_search(
    mock_vector_memory: MockVectorMemory, mock_embedding_provider: MockEmbeddingProvider
) -> None:
    """Test adding vectors and searching."""
    # Add some test vectors
    text1: str = "This is a test document about AI"
    text2: str = "Machine learning is a subset of AI"
    text3: str = "Python is a programming language"

    embedding1: List[float] = await mock_embedding_provider.embed_text(text1)
    embedding2: List[float] = await mock_embedding_provider.embed_text(text2)
    embedding3: List[float] = await mock_embedding_provider.embed_text(text3)

    # Add vectors to memory
    id1: str = await mock_vector_memory.add_vector(
        embedding1, {"text": text1, "category": "AI"}
    )
    id2: str = await mock_vector_memory.add_vector(
        embedding2, {"text": text2, "category": "AI"}
    )
    id3: str = await mock_vector_memory.add_vector(
        embedding3, {"text": text3, "category": "Programming"}
    )

    assert id1 == "vec_0"
    assert id2 == "vec_1"
    assert id3 == "vec_2"

    # Search for similar vectors
    query_embedding: List[float] = await mock_embedding_provider.embed_text(
        "Artificial intelligence topics"
    )
    results: List[Dict[str, Any]] = await mock_vector_memory.search(
        query_embedding, top_k=2
    )

    assert len(results) <= 2
    assert all("similarity" in result for result in results)
    assert all("metadata" in result for result in results)


@pytest.mark.asyncio
async def test_vector_memory_dimension_validation(
    mock_vector_memory: MockVectorMemory,
) -> None:
    """Test vector dimension validation."""
    # Try to add vector with wrong dimension
    wrong_dimension_vector: List[float] = [
        1.0,
        2.0,
        3.0,
    ]  # Only 3 dimensions instead of 768

    with pytest.raises(
        ValueError, match="Vector dimension 3 does not match expected 768"
    ):
        await mock_vector_memory.add_vector(wrong_dimension_vector, {"test": "data"})

    # Try to search with wrong dimension
    with pytest.raises(
        ValueError, match="Query vector dimension 3 does not match expected 768"
    ):
        await mock_vector_memory.search(wrong_dimension_vector)


@pytest.mark.asyncio
async def test_embedding_provider_functionality(
    mock_embedding_provider: MockEmbeddingProvider,
) -> None:
    """Test embedding provider functionality."""
    text: str = "Test document for embedding"
    embedding: List[float] = await mock_embedding_provider.embed_text(text)

    assert len(embedding) == mock_embedding_provider.dimension
    assert mock_embedding_provider.call_count == 1

    # Test consistency
    embedding2: List[float] = await mock_embedding_provider.embed_text(text)
    assert embedding == embedding2
    assert mock_embedding_provider.call_count == 2


@pytest.mark.asyncio
async def test_vector_memory_performance() -> None:
    """Test vector memory performance with multiple operations."""
    memory: MockVectorMemory = MockVectorMemory(dimension=128)
    provider: MockEmbeddingProvider = MockEmbeddingProvider(dimension=128)

    # Add multiple vectors
    texts: List[str] = [f"Document {i}" for i in range(10)]

    for i, text in enumerate(texts):
        embedding: List[float] = await provider.embed_text(text)
        vector_id: str = await memory.add_vector(embedding, {"text": text, "index": i})
        assert vector_id == f"vec_{i}"

    # Search
    query_embedding: List[float] = await provider.embed_text("Search query")
    results: List[Dict[str, Any]] = await memory.search(query_embedding, top_k=5)

    assert len(results) <= 5
    assert len(memory.vectors) == len(texts)
