"""
Comprehensive tests for Simon Willison's LLM Ecosystem Integration

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llamaagent.integration.simon_tools import (
    LLMChatTool,
    SimonToolRegistry,
    SQLiteQueryTool,
    create_simon_tools,
)
from src.llamaagent.llm.simon_ecosystem import (
    LLMTool,
    SimonEcosystemConfig,
    SimonLLMEcosystem,
)


class TestSimonEcosystemConfig:
    """Test Simon ecosystem configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimonEcosystemConfig()

        assert config.database_path == "llm_ecosystem.db"
        assert config.default_chat_model == "gpt-4o-mini"
        assert config.log_conversations is True
        assert LLMTool.SQLITE in config.enabled_tools

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimonEcosystemConfig(
            openai_api_key="test-key",
            database_path="test.db",
            default_chat_model="gpt-4",
            enabled_tools=[LLMTool.SQLITE, LLMTool.PYTHON],
        )

        assert config.openai_api_key == "test-key"
        assert config.database_path == "test.db"
        assert config.default_chat_model == "gpt-4"
        assert len(config.enabled_tools) == 2


class TestSimonLLMEcosystem:
    """Test the main Simon LLM ecosystem class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimonEcosystemConfig(
            database_path=":memory:",  # In-memory database for testing
            enabled_tools=[LLMTool.SQLITE, LLMTool.PYTHON],
            log_conversations=False,  # Disable for tests
        )

    @pytest.fixture
    def ecosystem(self, config):
        """Create ecosystem instance for testing."""
        with (
            patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True),
            patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", True),
        ):
            return SimonLLMEcosystem(config)

    def test_ecosystem_initialization(self, ecosystem):
        """Test ecosystem initialization."""
        assert ecosystem.config is not None
        assert ecosystem.tools is not None
        assert LLMTool.SQLITE in ecosystem.tools
        assert LLMTool.PYTHON in ecosystem.tools

    @pytest.mark.asyncio
    async def test_health_check(self, ecosystem):
        """Test ecosystem health check."""
        with patch.object(ecosystem, "db", MagicMock()):
            health = await ecosystem.health_check()

            assert "status" in health
            assert "components" in health
            assert "tools" in health
            assert "database" in health

    @pytest.mark.asyncio
    async def test_chat_without_llm(self, ecosystem):
        """Test chat when LLM library is not available."""
        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", False):
            with pytest.raises(RuntimeError, match="llm library not available"):
                await ecosystem.chat("test prompt")

    @pytest.mark.asyncio
    async def test_embed_without_llm(self, ecosystem):
        """Test embedding when LLM library is not available."""
        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", False):
            with pytest.raises(RuntimeError, match="llm library not available"):
                await ecosystem.embed("test text")

    @pytest.mark.asyncio
    async def test_use_tool_success(self, ecosystem):
        """Test successful tool usage."""
        mock_tool = AsyncMock()
        mock_tool.execute.return_value = {"success": True, "result": "test"}
        ecosystem.tools[LLMTool.SQLITE] = mock_tool

        result = await ecosystem.use_tool("sqlite", "query", sql="SELECT 1")

        assert result["success"] is True
        assert result["result"] == "test"
        mock_tool.execute.assert_called_once_with("query", sql="SELECT 1")

    @pytest.mark.asyncio
    async def test_use_tool_not_available(self, ecosystem):
        """Test using unavailable tool."""
        with pytest.raises(ValueError, match="'nonexistent' is not a valid LLMTool"):
            await ecosystem.use_tool("nonexistent", "operation")

    @pytest.mark.asyncio
    async def test_search_conversations_no_db(self, ecosystem):
        """Test conversation search without database."""
        ecosystem.db = None

        results = await ecosystem.search_conversations("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_get_conversation_stats_no_db(self, ecosystem):
        """Test getting stats without database."""
        ecosystem.db = None

        stats = await ecosystem.get_conversation_stats()

        assert stats == {}


class TestSQLiteTool:
    """Test SQLite tool functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = MagicMock()
        db.execute.return_value = [{"count": 5}]
        return db

    @pytest.fixture
    def sqlite_tool(self, mock_db):
        """Create SQLite tool with mock database."""
        return SQLiteQueryTool(MagicMock())  # Pass ecosystem mock

    @pytest.mark.asyncio
    async def test_sqlite_query_operation(self, sqlite_tool):
        """Test SQLite query operation."""
        with patch.object(
            sqlite_tool.ecosystem, "use_tool", new=AsyncMock()
        ) as mock_use_tool:
            mock_use_tool.return_value = [{"id": 1, "name": "test"}]

            result = await sqlite_tool.execute(
                sql="SELECT * FROM test", database_path=None
            )

            assert result["success"] is True
            assert "data" in result
            assert result["sql"] == "SELECT * FROM test"

    @pytest.mark.asyncio
    async def test_sqlite_query_with_external_db(self, sqlite_tool):
        """Test SQLite query with external database."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = '[{"id": 1}]'

            result = await sqlite_tool.execute(
                sql="SELECT * FROM test", database_path="/path/to/db.sqlite"
            )

            assert result["success"] is True
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_sqlite_query_error(self, sqlite_tool):
        """Test SQLite query error handling."""
        with patch.object(
            sqlite_tool.ecosystem, "use_tool", new=AsyncMock()
        ) as mock_use_tool:
            mock_use_tool.side_effect = Exception("Database error")

            result = await sqlite_tool.execute(sql="INVALID SQL")

            assert result["success"] is False
            assert "Database error" in result["error"]


class TestLLMChatTool:
    """Test LLM chat tool functionality."""

    @pytest.fixture
    def chat_tool(self):
        """Create LLM chat tool."""
        ecosystem = MagicMock()
        return LLMChatTool(ecosystem)

    @pytest.mark.asyncio
    async def test_chat_success(self, chat_tool):
        """Test successful chat execution."""
        chat_tool.ecosystem.chat = AsyncMock(return_value="Test response")

        result = await chat_tool.execute(prompt="Test prompt", model="gpt-4o-mini")

        assert result["success"] is True
        assert result["response"] == "Test response"
        assert result["model"] == "gpt-4o-mini"
        assert result["prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_chat_error(self, chat_tool):
        """Test chat execution error."""
        chat_tool.ecosystem.chat = AsyncMock(side_effect=Exception("API error"))

        result = await chat_tool.execute(prompt="Test prompt")

        assert result["success"] is False
        assert "API error" in result["error"]


class TestSimonToolRegistry:
    """Test Simon's tool registry."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimonEcosystemConfig(
            database_path=":memory:",
            enabled_tools=[LLMTool.SQLITE, LLMTool.PYTHON],
            log_conversations=False,
        )

    @pytest.fixture
    def tool_registry(self, config):
        """Create tool registry for testing."""
        with (
            patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True),
            patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", True),
        ):
            return create_simon_tools(config)

    def test_registry_initialization(self, tool_registry):
        """Test tool registry initialization."""
        assert isinstance(tool_registry, SimonToolRegistry)
        assert tool_registry.ecosystem is not None
        assert tool_registry._tools is not None

    def test_get_registry(self, tool_registry):
        """Test getting the tool registry."""
        registry = tool_registry.get_registry()
        assert registry is not None

    def test_get_ecosystem(self, tool_registry):
        """Test getting the ecosystem."""
        ecosystem = tool_registry.get_ecosystem()
        assert isinstance(ecosystem, SimonLLMEcosystem)

    @pytest.mark.asyncio
    async def test_health_check(self, tool_registry):
        """Test tool registry health check."""
        with patch.object(tool_registry.ecosystem, "health_check") as mock_health:
            mock_health.return_value = {"status": "healthy"}

            health = await tool_registry.health_check()

            assert health["status"] == "healthy"


class TestSimonEcosystemIntegration:
    """Integration tests for the complete Simon ecosystem."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def integration_config(self, temp_db_path):
        """Create integration test configuration."""
        return SimonEcosystemConfig(
            database_path=temp_db_path,
            enabled_tools=[LLMTool.SQLITE, LLMTool.PYTHON],
            log_conversations=True,
        )

    @pytest.mark.asyncio
    async def test_full_workflow(self, integration_config):
        """Test a complete workflow with Simon's ecosystem."""
        with (
            patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True),
            patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", True),
        ):
            # Initialize ecosystem
            ecosystem = SimonLLMEcosystem(integration_config)

            # Test health check
            health = await ecosystem.health_check()
            assert "status" in health

            # Test tool usage
            if LLMTool.PYTHON in ecosystem.tools:
                result = await ecosystem.use_tool(
                    "python", "run", code="print('Hello, Simon!')"
                )
                # Should return a result (success or failure)
                assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_tool_registry_integration(self, integration_config):
        """Test tool registry integration."""
        with (
            patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True),
            patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", True),
        ):
            # Create tool registry
            simon_tools = create_simon_tools(integration_config)
            registry = simon_tools.get_registry()

            # Test that tools are registered
            tool_names = registry.list_tools()
            assert len(tool_names) > 0

            # Test health check
            health = await simon_tools.health_check()
            assert "status" in health


class TestSimonEcosystemMocks:
    """Test ecosystem with various mock scenarios."""

    @pytest.mark.asyncio
    async def test_chat_with_mocked_subprocess(self):
        """Test chat with mocked subprocess call."""
        config = SimonEcosystemConfig(log_conversations=False)

        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True), patch(
            "subprocess.run"
        ) as mock_run:
            mock_run.return_value.stdout = "Mocked response"
            mock_run.return_value.stderr = ""
            mock_run.return_value.returncode = 0

            ecosystem = SimonLLMEcosystem(config)

            # Mock the subprocess call
            result = await ecosystem.chat("Test prompt")

            # Should return the mocked response
            assert result == "Mocked response"

    @pytest.mark.asyncio
    async def test_embed_with_mocked_subprocess(self):
        """Test embedding with mocked subprocess call."""
        config = SimonEcosystemConfig(log_conversations=False)

        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True), patch(
            "subprocess.run"
        ) as mock_run:
            mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_run.return_value.stdout = json.dumps(mock_embedding)
            mock_run.return_value.stderr = ""
            mock_run.return_value.returncode = 0

            ecosystem = SimonLLMEcosystem(config)

            result = await ecosystem.embed("Test text")

            assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_export_data_success(self):
        """Test successful data export."""
        config = SimonEcosystemConfig(database_path=":memory:", log_conversations=False)

        with patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", True):
            ecosystem = SimonLLMEcosystem(config)

            # Mock database with data
            mock_data = [{"id": 1, "text": "test"}]
            ecosystem.db = MagicMock()
            ecosystem.db.__getitem__.return_value.rows = mock_data

            with patch("builtins.open", MagicMock()) as mock_open:
                export_path = await ecosystem.export_data(
                    table="test_table", format="json", output_path="test_export.json"
                )

                assert export_path == "test_export.json"


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    async def test_missing_dependencies(self):
        """Test behavior when dependencies are missing."""
        with (
            patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", False),
            patch("src.llamaagent.llm.simon_ecosystem.HAS_SQLITE_UTILS", False),
        ):
            config = SimonEcosystemConfig()
            ecosystem = SimonLLMEcosystem(config)

            # Should handle missing dependencies gracefully
            assert ecosystem.db is None

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Test tool execution error handling."""
        config = SimonEcosystemConfig(
            enabled_tools=[LLMTool.PYTHON], log_conversations=False
        )

        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True):
            ecosystem = SimonLLMEcosystem(config)

            # Mock a tool that raises an exception
            mock_tool = AsyncMock()
            mock_tool.execute.side_effect = Exception("Tool error")
            ecosystem.tools[LLMTool.PYTHON] = mock_tool

            with pytest.raises(Exception, match="Tool error"):
                await ecosystem.use_tool("python", "run", code="test")


# Performance and stress tests
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_tool_usage(self):
        """Test concurrent tool usage."""
        config = SimonEcosystemConfig(
            enabled_tools=[LLMTool.PYTHON], log_conversations=False
        )

        with patch("src.llamaagent.llm.simon_ecosystem.HAS_LLM", True):
            ecosystem = SimonLLMEcosystem(config)

            # Mock tool for concurrent testing
            mock_tool = AsyncMock()
            mock_tool.execute.return_value = {"success": True}
            ecosystem.tools[LLMTool.PYTHON] = mock_tool

            # Run multiple concurrent operations
            tasks = [
                ecosystem.use_tool("python", "run", code=f"print({i})")
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert len(results) == 5
            for result in results:
                assert not isinstance(result, Exception)
                assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
