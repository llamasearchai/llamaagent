#!/usr/bin/env python3
"""Comprehensive tests for llamaagent.storage.vector_memory module."""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from llamaagent.storage.vector_memory import PostgresVectorMemory
from llamaagent.llm import MockProvider


class TestPostgresVectorMemory:
    """Test PostgresVectorMemory implementation."""
    
    @pytest.mark.asyncio
    async def test_init_with_default_provider(self):
        """Test initialization with default LLM provider."""
        with patch.dict(os.environ, {'LLAMAAGENT_LLM_PROVIDER': 'mock'}):
            memory = PostgresVectorMemory("test-agent-id")
            assert memory.agent_id == "test-agent-id"
            assert isinstance(memory.llm, MockProvider)
            assert not memory._schema_ready
    
    @pytest.mark.asyncio
    async def test_init_with_custom_provider(self):
        """Test initialization with custom LLM provider."""
        custom_provider = MockProvider()
        memory = PostgresVectorMemory("test-agent-id", provider=custom_provider)
        assert memory.agent_id == "test-agent-id"
        assert memory.llm == custom_provider
        assert not memory._schema_ready
    
    @pytest.mark.asyncio 
    async def test_add_text_success(self):
        """Test successful text addition."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        
        with patch.object(memory, '_ensure_schema', new_callable=AsyncMock) as mock_schema:
            with patch.object(memory, '_embed', new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]) as mock_embed:
                with patch('llamaagent.storage.vector_memory.Database') as mock_db:
                    mock_db.execute = AsyncMock()
                    
                    await memory.add("test text")
                    
                    mock_schema.assert_called_once()
                    mock_embed.assert_called_once_with("test text")
                    mock_db.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful text search."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        
        mock_results = [
            {"text": "result 1"},
            {"text": "result 2"}
        ]
        
        with patch.object(memory, '_ensure_schema', new_callable=AsyncMock) as mock_schema:
            with patch.object(memory, '_embed', new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]) as mock_embed:
                with patch('llamaagent.storage.vector_memory.Database') as mock_db:
                    mock_db.fetch = AsyncMock(return_value=mock_results)
                    
                    results = await memory.search("test query", limit=10)
                    
                    mock_schema.assert_called_once()
                    mock_embed.assert_called_once_with("test query")
                    mock_db.fetch.assert_called_once()
                    assert results == ["result 1", "result 2"]
    
    @pytest.mark.asyncio
    async def test_search_default_limit(self):
        """Test search with default limit."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        
        with patch.object(memory, '_ensure_schema', new_callable=AsyncMock):
            with patch.object(memory, '_embed', new_callable=AsyncMock, return_value=[0.1, 0.2, 0.3]):
                with patch('llamaagent.storage.vector_memory.Database') as mock_db:
                    mock_db.fetch = AsyncMock(return_value=[])
                    
                    await memory.search("test query")  # Default limit=5
                    
                    # Check that the SQL was called with limit=5
                    args = mock_db.fetch.call_args[0]
                    assert args[3] == 5  # limit parameter
    
    @pytest.mark.asyncio
    async def test_ensure_schema_first_time(self):
        """Test schema creation on first call."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        
        with patch('llamaagent.storage.vector_memory.Database') as mock_db:
            mock_db.execute = AsyncMock()
            
            await memory._ensure_schema()
            
            assert memory._schema_ready
            mock_db.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_schema_already_ready(self):
        """Test that schema creation is skipped when already ready."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        memory._schema_ready = True
        
        with patch('llamaagent.storage.vector_memory.Database') as mock_db:
            mock_db.execute = AsyncMock()
            
            await memory._ensure_schema()
            
            mock_db.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_ensure_schema_concurrent_calls(self):
        """Test schema creation with concurrent calls."""
        memory = PostgresVectorMemory("test-agent-id", provider=MockProvider())
        
        with patch('llamaagent.storage.vector_memory.Database') as mock_db:
            mock_db.execute = AsyncMock()
            
            # Simulate concurrent calls
            import asyncio
            tasks = [memory._ensure_schema() for _ in range(3)]
            await asyncio.gather(*tasks)
            
            assert memory._schema_ready
            # Should only be called once due to lock
            mock_db.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embed_with_embedding_attribute(self):
        """Test embedding when response has embedding attribute."""
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.embedding = [0.1, 0.2, 0.3]
        mock_provider.embed = AsyncMock(return_value=mock_response)
        
        memory = PostgresVectorMemory("test-agent-id", provider=mock_provider)
        
        result = await memory._embed("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_provider.embed.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_embed_without_embedding_attribute(self):
        """Test embedding when response is direct vector."""
        mock_provider = Mock()
        mock_provider.embed = AsyncMock(return_value=[0.4, 0.5, 0.6])
        
        memory = PostgresVectorMemory("test-agent-id", provider=mock_provider)
        
        result = await memory._embed("test text")
        
        assert result == [0.4, 0.5, 0.6]
        mock_provider.embed.assert_called_once_with("test text")
    
    @pytest.mark.asyncio
    async def test_embed_converts_to_float(self):
        """Test that embedding values are converted to floats."""
        mock_provider = Mock()
        # Return integers that should be converted to floats
        mock_provider.embed = AsyncMock(return_value=[1, 2, 3])
        
        memory = PostgresVectorMemory("test-agent-id", provider=mock_provider)
        
        result = await memory._embed("test text")
        
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result) 