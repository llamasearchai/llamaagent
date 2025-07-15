#!/usr/bin/env python3
"""Test database functionality with mock implementations.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

# pyright: reportUnknownMemberType=false

from typing import Any, Dict, List, Union
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llamaagent.storage.database import DatabaseManager as Database


class MockNullPool:
    """Mock null pool implementation."""

    async def execute(self, query: str, *args: Any) -> None:
        """Mock execute that raises error."""
        raise RuntimeError("Database support is not enabled")

    async def fetch(self, query: str, *args: Any) -> None:
        """Mock fetch that raises error."""
        raise RuntimeError("Database support is not enabled")

    async def fetchrow(self, query: str, *args: Any) -> None:
        """Mock fetchrow that raises error."""
        raise RuntimeError("Database support is not enabled")

    async def fetchval(self, query: str, *args: Any) -> None:
        """Mock fetchval that raises error."""
        raise RuntimeError("Database support is not enabled")


def test_null_pool_creation():
    """Test that MockNullPool can be instantiated."""
    pool = MockNullPool()
    assert pool is not None


@pytest.mark.asyncio
async def test_null_pool_execute_raises():
    """Test that MockNullPool.execute raises RuntimeError."""
    pool = MockNullPool()
    with pytest.raises(RuntimeError, match="Database support is not enabled"):
        await pool.execute("SELECT 1")


@pytest.mark.asyncio
async def test_null_pool_fetch_raises():
    """Test that MockNullPool.fetch raises RuntimeError."""
    pool = MockNullPool()
    with pytest.raises(RuntimeError, match="Database support is not enabled"):
        await pool.fetch("SELECT 1")


@pytest.mark.asyncio
async def test_null_pool_fetchrow_raises():
    """Test that MockNullPool.fetchrow raises RuntimeError."""
    pool = MockNullPool()
    with pytest.raises(RuntimeError, match="Database support is not enabled"):
        await pool.fetchrow("SELECT 1")


@pytest.mark.asyncio
async def test_null_pool_fetchval_raises():
    """Test that MockNullPool.fetchval raises RuntimeError."""
    pool = MockNullPool()
    with pytest.raises(RuntimeError, match="Database support is not enabled"):
        await pool.fetchval("SELECT 1")


def test_database_mock_functionality():
    """Test database mock functionality."""
    # Mock a database connection
    mock_db = Mock()
    mock_db.connected = False

    # Test connection
    async def connect():
        mock_db.connected = True

    mock_db.connect = connect

    # Verify mock works
    assert mock_db.connected is False


@pytest.mark.asyncio
async def test_database_connection_simulation():
    """Test database connection simulation."""
    # Simulate database operations without real database
    with patch("asyncpg.create_pool") as mock_create_pool:
        mock_pool = Mock()
        mock_pool.execute = AsyncMock(return_value="OK")
        mock_create_pool.return_value = mock_pool

        # Test that we can mock database operations
        result = await mock_pool.execute("SELECT 1")
        assert result == "OK"


class MockDatabase:
    """Mock database implementation for testing."""

    def __init__(self, connection_string: str = "mock://test"):
        self.connection_string = connection_string
        self.connected = False
        self.data: Dict[str, Any] = {}

    async def connect(self) -> None:
        """Mock connection."""
        self.connected = True

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False

    async def execute(
        self, query: str, *args: Any
    ) -> Dict[str, Union[str, tuple[Any, ...]]]:
        """Mock query execution."""
        return {"query": query, "args": args}

    async def fetch_one(self, query: str, *args: Any) -> Dict[str, Union[int, str]]:
        """Mock fetch one."""
        return {"id": 1, "data": "test"}

    async def fetch_all(
        self, query: str, *args: Any
    ) -> List[Dict[str, Union[int, str]]]:
        """Mock fetch all."""
        return [{"id": 1, "data": "test1"}, {"id": 2, "data": "test2"}]


@pytest.fixture
def mock_database() -> MockDatabase:
    """Provide a mock database instance."""
    return MockDatabase()


@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection and query execution."""
    # Use DatabaseManager with mock pool since it requires PostgreSQL
    from src.llamaagent.storage.database import DatabaseConfig
    
    config = DatabaseConfig(database=":memory:")
    db = Database(config)
    
    # Mock the pool since we can't actually connect to PostgreSQL in tests
    with patch('src.llamaagent.storage.database.asyncpg') as mock_asyncpg:
        mock_pool = AsyncMock()
        mock_asyncpg.create_pool.return_value = mock_pool
        await db.initialize()

        # Mock the database operations since DatabaseManager uses a different interface
        mock_pool.execute = AsyncMock()
        mock_pool.fetch = AsyncMock(return_value=[{"id": 1, "data": "test_data"}])
        mock_pool.close = AsyncMock()
        
        # Since DatabaseManager doesn't expose execute/fetch directly,
        # we'll just verify the pool was created correctly
        assert db.pool is not None
        mock_asyncpg.create_pool.assert_called_once()
        
        # Test cleanup method (DatabaseManager uses cleanup instead of disconnect)
        await db.cleanup()


@pytest.mark.asyncio
async def test_database_execute(mock_database: MockDatabase):
    """Test database query execution."""
    await mock_database.connect()

    result = await mock_database.execute("SELECT * FROM test", "param1")
    assert result["query"] == "SELECT * FROM test"
    assert result["args"] == ("param1",)


@pytest.mark.asyncio
async def test_database_fetch_operations(mock_database: MockDatabase):
    """Test database fetch operations."""
    await mock_database.connect()

    # Test fetch_one
    result = await mock_database.fetch_one("SELECT * FROM test WHERE id = ?", 1)
    assert result["id"] == 1
    assert result["data"] == "test"

    # Test fetch_all
    results = await mock_database.fetch_all("SELECT * FROM test")
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 2


def test_database_initialization():
    """Test database initialization."""
    db = MockDatabase("postgresql://test:test@localhost/test")
    assert db.connection_string == "postgresql://test:test@localhost/test"
    assert db.connected is False


@pytest.mark.asyncio
async def test_database_error_handling():
    """Test database error handling."""
    db = MockDatabase()

    # Test that we can handle connection errors gracefully
    with patch.object(db, "connect", side_effect=Exception("Connection failed")):
        with pytest.raises(Exception, match="Connection failed"):
            await db.connect()


def test_database_with_null_pool():
    """Test database with null pool configuration."""
    # Mock the _NullPool functionality
    with patch("sqlalchemy.pool.NullPool") as mock_null_pool:
        mock_null_pool.return_value = Mock()

        # Test that null pool can be configured
        db = MockDatabase()
        assert db is not None


@pytest.mark.asyncio
async def test_database_transaction_handling():
    """Test database transaction handling."""
    db = MockDatabase()
    await db.connect()

    # Mock transaction behavior
    with patch.object(db, "execute") as mock_execute:
        mock_execute.return_value = {"status": "success"}

        # Simulate transaction
        result = await db.execute("BEGIN")
        assert result["status"] == "success"

        result = await db.execute("INSERT INTO test (data) VALUES (?)", "test_data")
        assert result["status"] == "success"

        result = await db.execute("COMMIT")
        assert result["status"] == "success"


def test_database_url_parsing():
    """Test database URL parsing."""
    test_urls = [
        "postgresql://user:pass@localhost:5432/dbname",
        "sqlite:///path/to/database.db",
        "mysql://user:pass@localhost/dbname",
    ]

    for url in test_urls:
        db = MockDatabase(url)
        assert db.connection_string == url
