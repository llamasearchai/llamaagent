#!/usr/bin/env python3
"""Tests for llamaagent.storage.database module."""

import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from llamaagent.storage.database import Database, _NullPool


class TestNullPool:
    """Test the fallback _NullPool implementation."""
    
    def test_null_pool_creation(self):
        """Test that _NullPool can be instantiated."""
        pool = _NullPool()
        assert pool is not None
    
    @pytest.mark.asyncio
    async def test_null_pool_execute_raises(self):
        """Test that _NullPool.execute raises RuntimeError."""
        pool = _NullPool()
        with pytest.raises(RuntimeError, match="Database support is not enabled"):
            await pool.execute("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_null_pool_fetch_raises(self):
        """Test that _NullPool.fetch raises RuntimeError."""
        pool = _NullPool()
        with pytest.raises(RuntimeError, match="Database support is not enabled"):
            await pool.fetch("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_null_pool_fetchrow_raises(self):
        """Test that _NullPool.fetchrow raises RuntimeError."""
        pool = _NullPool()
        with pytest.raises(RuntimeError, match="Database support is not enabled"):
            await pool.fetchrow("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_null_pool_fetchval_raises(self):
        """Test that _NullPool.fetchval raises RuntimeError."""
        pool = _NullPool()
        with pytest.raises(RuntimeError, match="Database support is not enabled"):
            await pool.fetchval("SELECT 1")


class TestDatabase:
    """Test Database connection management."""
    
    def setup_method(self):
        """Reset Database state before each test."""
        Database._pool = None
    
    def teardown_method(self):
        """Clean up after each test."""
        Database._pool = None
    
    @pytest.mark.asyncio
    async def test_initialize_no_asyncpg(self):
        """Test initialization when asyncpg is not available."""
        with patch('llamaagent.storage.database.asyncpg', None):
            await Database.initialise()
            assert Database._pool is None
    
    @pytest.mark.asyncio
    async def test_initialize_no_database_url(self):
        """Test initialization when DATABASE_URL is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('llamaagent.storage.database.asyncpg') as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock()
                await Database.initialise()
                assert Database._pool is None
                mock_asyncpg.create_pool.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_with_database_url(self):
        """Test successful initialization with DATABASE_URL."""
        mock_pool = Mock()
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://localhost/test'}):
            with patch('llamaagent.storage.database.asyncpg') as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
                await Database.initialise()
                assert Database._pool == mock_pool
                mock_asyncpg.create_pool.assert_called_once_with(
                    'postgresql://localhost/test', min_size=1, max_size=10
                )
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that multiple initialization calls are idempotent."""
        mock_pool = Mock()
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://localhost/test'}):
            with patch('llamaagent.storage.database.asyncpg') as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
                await Database.initialise()
                await Database.initialise()  # Second call should be no-op
                assert Database._pool == mock_pool
                mock_asyncpg.create_pool.assert_called_once()
    
    def test_get_pool_with_null_pool(self):
        """Test _get_pool returns _NullPool when no pool is set."""
        Database._pool = None
        pool = Database._get_pool()
        assert isinstance(pool, _NullPool)
    
    def test_get_pool_with_real_pool(self):
        """Test _get_pool returns actual pool when set."""
        mock_pool = Mock()
        Database._pool = mock_pool
        pool = Database._get_pool()
        assert pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_execute_calls_initialize(self):
        """Test that execute calls initialize."""
        with patch.object(Database, 'initialise', new_callable=AsyncMock) as mock_init:
            with patch.object(Database, '_get_pool') as mock_get_pool:
                mock_pool = Mock()
                mock_pool.execute = AsyncMock(return_value="result")
                mock_get_pool.return_value = mock_pool
                
                result = await Database.execute("SELECT 1", "arg1")
                
                mock_init.assert_called_once()
                mock_pool.execute.assert_called_once_with("SELECT 1", "arg1")
                assert result == "result"
    
    @pytest.mark.asyncio
    async def test_fetch_calls_initialize(self):
        """Test that fetch calls initialize."""
        with patch.object(Database, 'initialise', new_callable=AsyncMock) as mock_init:
            with patch.object(Database, '_get_pool') as mock_get_pool:
                mock_pool = Mock()
                mock_pool.fetch = AsyncMock(return_value=[{"id": 1}])
                mock_get_pool.return_value = mock_pool
                
                result = await Database.fetch("SELECT * FROM test", "arg1")
                
                mock_init.assert_called_once()
                mock_pool.fetch.assert_called_once_with("SELECT * FROM test", "arg1")
                assert result == [{"id": 1}]
    
    @pytest.mark.asyncio
    async def test_fetchrow_calls_initialize(self):
        """Test that fetchrow calls initialize."""
        with patch.object(Database, 'initialise', new_callable=AsyncMock) as mock_init:
            with patch.object(Database, '_get_pool') as mock_get_pool:
                mock_pool = Mock()
                mock_pool.fetchrow = AsyncMock(return_value={"id": 1})
                mock_get_pool.return_value = mock_pool
                
                result = await Database.fetchrow("SELECT * FROM test WHERE id = $1", 1)
                
                mock_init.assert_called_once()
                mock_pool.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)
                assert result == {"id": 1}
    
    @pytest.mark.asyncio
    async def test_fetchval_calls_initialize(self):
        """Test that fetchval calls initialize."""
        with patch.object(Database, 'initialise', new_callable=AsyncMock) as mock_init:
            with patch.object(Database, '_get_pool') as mock_get_pool:
                mock_pool = Mock()
                mock_pool.fetchval = AsyncMock(return_value=42)
                mock_get_pool.return_value = mock_pool
                
                result = await Database.fetchval("SELECT COUNT(*) FROM test")
                
                mock_init.assert_called_once()
                mock_pool.fetchval.assert_called_once_with("SELECT COUNT(*) FROM test")
                assert result == 42
    
    def test_pool_property_with_null_pool(self):
        """Test pool property returns _NullPool when no pool is set."""
        Database._pool = None
        db = Database()
        pool = db.pool
        assert isinstance(pool, _NullPool)
    
    def test_pool_property_with_real_pool(self):
        """Test pool property returns actual pool when set."""
        mock_pool = Mock()
        Database._pool = mock_pool
        db = Database()
        pool = db.pool
        assert pool == mock_pool 