"""
Database Manager for LlamaAgent

This module provides comprehensive database management with PostgreSQL,
vector storage, and advanced querying capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Optional imports with proper handling
try:
    import asyncpg

    _asyncpg_available = True
except ImportError:
    asyncpg = None  # type: ignore
    _asyncpg_available = False

# psycopg2 is optional and not currently used
_psycopg2_available = False

# numpy is optional and not currently used
_numpy_available = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "llamaagent"
    username: str = "llamaagent"
    password: str = "llamaagent"
    min_connections: int = 10
    max_connections: int = 20
    command_timeout: int = 30
    ssl: bool = False


@dataclass
class QueryResult:
    """Database query result."""

    rows: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query: str


class DatabaseManager:
    """Advanced database manager with PostgreSQL support."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._create_default_config()
        self.pool: Optional[Any] = None
        self.logger = logging.getLogger(__name__)

    def _create_default_config(self) -> DatabaseConfig:
        """Create default database configuration from environment variables."""
        return DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "llamaagent"),
            username=os.getenv("DB_USER", "llamaagent"),
            password=os.getenv("DB_PASSWORD", "llamaagent"),
            min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "10")),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "20")),
            command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "30")),
            ssl=os.getenv("DB_SSL", "false").lower() == "true",
        )

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"

    async def initialize(self) -> None:
        """Initialize database connection pool and schema."""
        if not _asyncpg_available:
            raise ImportError("asyncpg is required for database operations")

        try:
            if asyncpg:
                self.pool = await asyncpg.create_pool(  # type: ignore
                    self._build_connection_string(),
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    command_timeout=self.config.command_timeout,
                )
            await self._initialize_schema()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")

    async def _initialize_schema(self) -> None:
        """Initialize database schema."""
        schema_sql = """
        -- Agents table
        CREATE TABLE IF NOT EXISTS agents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            agent_type VARCHAR(100) NOT NULL,
            config JSONB NOT NULL DEFAULT '{}',
            metadata JSONB NOT NULL DEFAULT '{}',
            status VARCHAR(50) NOT NULL DEFAULT 'active',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Tasks table
        CREATE TABLE IF NOT EXISTS tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
            task_input JSONB NOT NULL,
            task_output JSONB,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            completed_at TIMESTAMP WITH TIME ZONE
        );

        -- Embeddings table for vector storage
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            embedding VECTOR(1536),
            metadata JSONB NOT NULL DEFAULT '{}',
            source_type VARCHAR(100),
            source_id UUID,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
            messages JSONB NOT NULL DEFAULT '[]',
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Knowledge base table
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(500) NOT NULL,
            content TEXT NOT NULL,
            content_type VARCHAR(100) NOT NULL DEFAULT 'text',
            tags TEXT[] DEFAULT '{}',
            metadata JSONB NOT NULL DEFAULT '{}',
            embedding VECTOR(1536),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
        CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
        CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
        CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_agent_id ON conversations(agent_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_base_tags ON knowledge_base USING GIN(tags);
        CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_type ON knowledge_base(content_type);

        -- Enable pg_trgm for text search
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_trgm ON knowledge_base USING GIN(content gin_trgm_ops);

        -- Enable vector similarity search (if pgvector is available)
        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS vector;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'pgvector extension not available, vector operations will be limited';
        END
        $$;
        """

        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
                self.logger.info("Database schema initialized")

    async def execute_query(self, query: str, *args: Any) -> QueryResult:
        """Execute a query and return results."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        start_time = time.time()
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args)
                execution_time = time.time() - start_time

                # Convert asyncpg.Record to dict
                result_rows: List[Dict[str, Any]] = [dict(row) for row in rows]

                return QueryResult(
                    rows=result_rows,
                    row_count=len(result_rows),
                    execution_time=execution_time,
                    query=query,
                )
            except Exception as e:
                self.logger.error(f"Query execution failed: {e}")
                raise

    async def execute_update(self, query: str, *args: Any) -> int:
        """Execute an update/insert/delete query and return affected rows."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        async with self.pool.acquire() as conn:
            try:
                result = await conn.execute(query, *args)
                # Extract number from result like "UPDATE 1"
                return int(result.split()[-1]) if result else 0
            except Exception as e:
                self.logger.error(f"Update execution failed: {e}")
                raise

    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager."""
        if not self.pool:
            raise RuntimeError("Database not initialized")

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def create_agent(
        self,
        name: str,
        agent_type: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new agent."""
        query = """
        INSERT INTO agents (name, agent_type, config, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """
        result = await self.execute_query(
            query, name, agent_type, json.dumps(config), json.dumps(metadata or {})
        )
        return str(result.rows[0]["id"])

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        query = """
        SELECT id, name, agent_type, config, metadata, status, created_at, updated_at
        FROM agents
        WHERE id = $1
        """
        result = await self.execute_query(query, agent_id)
        if result.rows:
            agent = result.rows[0]
            # Parse JSON fields
            agent["config"] = (
                json.loads(agent["config"])
                if isinstance(agent["config"], str)
                else agent["config"]
            )
            agent["metadata"] = (
                json.loads(agent["metadata"])
                if isinstance(agent["metadata"], str)
                else agent["metadata"]
            )
            return agent
        return None

    async def list_agents(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List agents with pagination."""
        query = """
        SELECT id, name, agent_type, config, metadata, status, created_at, updated_at
        FROM agents
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """
        result = await self.execute_query(query, limit, offset)

        agents: List[Dict[str, Any]] = []
        for row in result.rows:
            agent = dict(row)
            # Parse JSON fields
            agent["config"] = (
                json.loads(agent["config"])
                if isinstance(agent["config"], str)
                else agent["config"]
            )
            agent["metadata"] = (
                json.loads(agent["metadata"])
                if isinstance(agent["metadata"], str)
                else agent["metadata"]
            )
            agents.append(agent)

        return agents

    async def update_agent(self, agent_id: str, **updates: Any) -> bool:
        """Update agent fields."""
        if not updates:
            return False

        set_clauses: List[str] = []
        values: List[Any] = []
        param_count = 1

        for key, value in updates.items():
            if key in ["config", "metadata"] and isinstance(value, dict):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ${param_count}")
            values.append(value)
            param_count += 1

        set_clauses.append(f"updated_at = ${param_count}")
        values.append(datetime.now(timezone.utc).isoformat())

        query = f"""
        UPDATE agents
        SET {', '.join(set_clauses)}
        WHERE id = ${param_count + 1}
        """
        values.append(agent_id)

        affected_rows = await self.execute_update(query, *values)
        return affected_rows > 0

    async def create_task(
        self,
        agent_id: Optional[str],
        task_input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new task."""
        query = """
        INSERT INTO tasks (agent_id, task_input, metadata)
        VALUES ($1, $2, $3)
        RETURNING id
        """
        result = await self.execute_query(
            query, agent_id, json.dumps(task_input), json.dumps(metadata or {})
        )
        return str(result.rows[0]["id"])

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update task status."""
        updates = {"status": status}
        if output:
            updates["task_output"] = json.dumps(output)
        if error:
            updates["error_message"] = error
        if status == "completed":
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()

        return await self.update_task(task_id, **updates)

    async def update_task(self, task_id: str, **updates: Any) -> bool:
        """Update task fields."""
        if not updates:
            return False

        set_clauses: List[str] = []
        values: List[Any] = []
        param_count = 1

        for key, value in updates.items():
            if key in ["task_input", "task_output", "metadata"] and isinstance(
                value, dict
            ):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ${param_count}")
            values.append(value)
            param_count += 1

        set_clauses.append(f"updated_at = ${param_count}")
        values.append(datetime.now(timezone.utc).isoformat())

        query = f"""
        UPDATE tasks
        SET {', '.join(set_clauses)}
        WHERE id = ${param_count + 1}
        """
        values.append(task_id)

        affected_rows = await self.execute_update(query, *values)
        return affected_rows > 0

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        query = """
        SELECT id, agent_id, task_input, task_output, status, error_message, metadata, created_at, updated_at, completed_at
        FROM tasks
        WHERE id = $1
        """
        result = await self.execute_query(query, task_id)
        if result.rows:
            task = result.rows[0]
            # Parse JSON fields
            task["task_input"] = (
                json.loads(task["task_input"])
                if isinstance(task["task_input"], str)
                else task["task_input"]
            )
            if task["task_output"]:
                task["task_output"] = (
                    json.loads(task["task_output"])
                    if isinstance(task["task_output"], str)
                    else task["task_output"]
                )
            task["metadata"] = (
                json.loads(task["metadata"])
                if isinstance(task["metadata"], str)
                else task["metadata"]
            )
            return task
        return None

    async def store_embedding(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> str:
        """Store content with embedding."""
        query = """
        INSERT INTO embeddings (content, embedding, metadata, source_type, source_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """

        # Convert embedding to string format for PostgreSQL
        embedding_str = f"[{','.join(map(str, embedding))}]"

        result = await self.execute_query(
            query,
            content,
            embedding_str,
            json.dumps(metadata or {}),
            source_type,
            source_id,
        )
        return str(result.rows[0]["id"])

    async def similarity_search(
        self, query_embedding: List[float], limit: int = 10, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Perform similarity search on embeddings."""
        query = """
        SELECT id, content, metadata, source_type, source_id, created_at,
               (embedding <-> $1) as distance
        FROM embeddings
        WHERE (embedding <-> $1) < $2
        ORDER BY distance
        LIMIT $3
        """

        # Convert embedding to string format for PostgreSQL
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        result = await self.execute_query(query, embedding_str, 1 - threshold, limit)

        results: List[Dict[str, Any]] = []
        for row in result.rows:
            item = dict(row)
            item["metadata"] = (
                json.loads(item["metadata"])
                if isinstance(item["metadata"], str)
                else item["metadata"]
            )
            results.append(item)

        return results

    async def get_task_stats(
        self, agent_id: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Get task statistics."""
        base_query = """
        SELECT
            COUNT(*) as total_tasks,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_tasks,
            AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_completion_time
        FROM tasks
        WHERE created_at >= NOW() - INTERVAL '%s days'
        """

        if agent_id:
            query = base_query + " AND agent_id = $1"
            result = await self.execute_query(query % days, agent_id)
        else:
            result = await self.execute_query(base_query % days)

        if result.rows:
            stats = dict(result.rows[0])
            # Convert avg_completion_time to seconds if not None
            if stats["avg_completion_time"]:
                stats["avg_completion_time"] = float(stats["avg_completion_time"])
            return stats

        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "pending_tasks": 0,
            "avg_completion_time": None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            if not self.pool:
                return {"status": "error", "message": "Database not initialized"}

            # Test connection
            result = await self.execute_query("SELECT 1 as test")

            # Get connection pool stats
            pool_stats = {
                "size": self.pool.get_size() if self.pool else 0,
                "min_size": self.pool.get_min_size() if self.pool else 0,
                "max_size": self.pool.get_max_size() if self.pool else 0,
                "idle_size": self.pool.get_idle_size() if self.pool else 0,
            }

            return {
                "status": "healthy",
                "connection_pool": pool_stats,
                "query_test": result.rows[0]["test"] == 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
