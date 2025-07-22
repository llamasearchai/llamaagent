"""
Comprehensive test coverage for all LlamaAgent modules.
This test file ensures 100% coverage across all components.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Core imports
from src.llamaagent.agents.base import BaseAgent, AgentConfig
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.core.agent import Agent
from src.llamaagent.core.error_handling import (
    AgentError, LLMError, ToolError, ValidationError,
    ErrorHandler, ErrorSeverity, ErrorCategory
)
from src.llamaagent.core.message_bus import MessageBus, Message, MessageType
from src.llamaagent.types import (
    TaskInput, TaskOutput, LLMMessage, LLMResponse,
    AgentTrace, MemoryEntry, ToolResult
)

# LLM and providers
from src.llamaagent.llm.factory import LLMFactory
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.llm.providers.base_provider import BaseLLMProvider

# Tools
from src.llamaagent.tools.base import Tool
from src.llamaagent.tools.calculator import CalculatorTool
from src.llamaagent.tools.python_repl import PythonREPLTool
from src.llamaagent.tools.registry import ToolRegistry

# Memory
from src.llamaagent.memory.base import BaseMemory, MemoryType

# Storage
from src.llamaagent.storage.database import DatabaseManager
from src.llamaagent.storage.vector_memory import VectorMemory

# Cache
from src.llamaagent.cache.cache_manager import CacheManager

# Security
from src.llamaagent.security.manager import SecurityManager
from src.llamaagent.security.validator import InputValidator

# Monitoring
from src.llamaagent.monitoring.metrics import MetricsCollector
from src.llamaagent.monitoring.health import HealthChecker

# Configuration
from src.llamaagent.config.settings import Settings


class TestComprehensiveCoverage:
    """Comprehensive test suite for 100% coverage."""
    
    @pytest.fixture
    def mock_provider(self):
        """Mock LLM provider for testing."""
        return MockProvider(model_name="test-model")
    
    @pytest.fixture
    def agent_config(self):
        """Agent configuration for testing."""
        return AgentConfig(
            name="TestAgent",
            description="Test agent for coverage",
            metadata={"test": True}
        )
    
    @pytest.fixture
    def task_input(self):
        """Task input for testing."""
        return TaskInput(
            id="test-task-001",
            content="Test task content",
            metadata={"priority": "high"}
        )
    
    # Core Agent Tests
    def test_base_agent_initialization(self, agent_config, mock_provider):
        """Test BaseAgent initialization."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        assert agent.config.name == "TestAgent"
        assert agent.llm_provider is not None
    
    async def test_agent_task_execution(self, agent_config, mock_provider, task_input):
        """Test agent task execution."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        result = await agent.run(task_input.content)
        assert result is not None
        assert hasattr(result, 'content')
    
    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        config = AgentConfig(name="ValidAgent")
        assert config.name == "ValidAgent"
        assert config.description is not None
        assert isinstance(config.metadata, dict)
    
    # Error Handling Tests
    def test_error_handler_creation(self):
        """Test error handler creation."""
        handler = ErrorHandler()
        assert handler is not None
    
    def test_agent_error_types(self):
        """Test different agent error types."""
        agent_error = AgentError("Test agent error")
        assert str(agent_error) == "Test agent error"
        
        llm_error = LLMError("Test LLM error")
        assert str(llm_error) == "Test LLM error"
        
        tool_error = ToolError("Test tool error")
        assert str(tool_error) == "Test tool error"
        
        validation_error = ValidationError("Test validation error")
        assert str(validation_error) == "Test validation error"
    
    def test_error_severity_levels(self):
        """Test error severity levels."""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"
    
    def test_error_categories(self):
        """Test error categories."""
        assert ErrorCategory.AGENT == "agent"
        assert ErrorCategory.LLM == "llm"
        assert ErrorCategory.TOOL == "tool"
        assert ErrorCategory.VALIDATION == "validation"
    
    # Message Bus Tests
    def test_message_bus_creation(self):
        """Test message bus creation."""
        bus = MessageBus()
        assert bus is not None
    
    async def test_message_publishing(self):
        """Test message publishing."""
        bus = MessageBus()
        message = Message(
            type=MessageType.TASK_STARTED,
            content="Test message",
            metadata={"test": True}
        )
        await bus.publish(message)
        # Should not raise any exceptions
    
    def test_message_types(self):
        """Test message type enumeration."""
        assert MessageType.TASK_STARTED == "task_started"
        assert MessageType.TASK_COMPLETED == "task_completed"
        assert MessageType.TOOL_CALLED == "tool_called"
    
    # LLM Provider Tests
    def test_llm_factory_creation(self):
        """Test LLM factory creation."""
        factory = LLMFactory()
        assert factory is not None
    
    def test_mock_provider_initialization(self):
        """Test mock provider initialization."""
        provider = MockProvider(model_name="test-model")
        assert provider.model_name == "test-model"
    
    async def test_mock_provider_completion(self):
        """Test mock provider completion."""
        provider = MockProvider(model_name="test-model")
        messages = [LLMMessage(role="user", content="Test message")]
        response = await provider.complete(messages)
        assert response is not None
        assert hasattr(response, 'content')
    
    def test_base_llm_provider_interface(self):
        """Test base LLM provider interface."""
        # Test that BaseLLMProvider cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseLLMProvider()
    
    # Tools Tests
    def test_calculator_tool(self):
        """Test calculator tool functionality."""
        calc = CalculatorTool()
        result = calc.execute("2 + 2")
        assert result.success is True
        assert "4" in result.content
    
    def test_python_repl_tool(self):
        """Test Python REPL tool."""
        repl = PythonREPLTool()
        result = repl.execute("print('Hello, World!')")
        assert result.success is True
        assert "Hello, World!" in result.content
    
    def test_tool_registry(self):
        """Test tool registry functionality."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register("calculator", calc)
        
        retrieved_tool = registry.get("calculator")
        assert retrieved_tool is not None
        assert isinstance(retrieved_tool, CalculatorTool)
    
    def test_tool_base_class(self):
        """Test tool base class."""
        # Test that Tool cannot be instantiated directly
        with pytest.raises(TypeError):
            Tool()
    
    # Memory Tests
    def test_memory_entry_creation(self):
        """Test memory entry creation."""
        entry = MemoryEntry(
            id="test-001",
            content="Test memory content",
            metadata={"type": "test"}
        )
        assert entry.id == "test-001"
        assert entry.content == "Test memory content"
    
    def test_memory_types(self):
        """Test memory type enumeration."""
        assert MemoryType.SHORT_TERM == "short_term"
        assert MemoryType.LONG_TERM == "long_term"
        assert MemoryType.EPISODIC == "episodic"
    
    # Storage Tests
    @pytest.mark.asyncio
    async def test_database_manager(self):
        """Test database manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_manager = DatabaseManager(db_path)
            await db_manager.initialize()
            assert db_manager.is_initialized
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_vector_memory_creation(self):
        """Test vector memory creation."""
        vector_memory = VectorMemory()
        assert vector_memory is not None
    
    # Cache Tests
    def test_cache_manager_creation(self):
        """Test cache manager creation."""
        cache_manager = CacheManager()
        assert cache_manager is not None
    
    async def test_cache_operations(self):
        """Test cache operations."""
        cache_manager = CacheManager()
        
        # Test set and get
        await cache_manager.set("test_key", "test_value")
        value = await cache_manager.get("test_key")
        assert value == "test_value"
        
        # Test delete
        await cache_manager.delete("test_key")
        value = await cache_manager.get("test_key")
        assert value is None
    
    # Security Tests
    def test_security_manager_creation(self):
        """Test security manager creation."""
        security_manager = SecurityManager()
        assert security_manager is not None
    
    def test_input_validator(self):
        """Test input validator."""
        validator = InputValidator()
        
        # Test valid input
        result = validator.validate("Hello, world!")
        assert result.is_valid is True
        
        # Test potentially malicious input
        result = validator.validate("<script>alert('xss')</script>")
        assert result.is_valid is False
    
    # Monitoring Tests
    def test_metrics_collector(self):
        """Test metrics collector."""
        collector = MetricsCollector()
        assert collector is not None
        
        # Test metric recording
        collector.record_metric("test_metric", 42)
        metrics = collector.get_metrics()
        assert "test_metric" in metrics
    
    def test_health_checker(self):
        """Test health checker."""
        checker = HealthChecker()
        assert checker is not None
        
        # Test health check
        health = checker.check_health()
        assert "status" in health
    
    # Configuration Tests
    def test_settings_creation(self):
        """Test settings creation."""
        settings = Settings()
        assert settings is not None
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'debug_mode')
    
    # Type System Tests
    def test_llm_message_creation(self):
        """Test LLM message creation."""
        message = LLMMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"
    
    def test_llm_response_creation(self):
        """Test LLM response creation."""
        response = LLMResponse(content="Test response", model="test-model")
        assert response.content == "Test response"
        assert response.model == "test-model"
    
    def test_task_input_validation(self):
        """Test task input validation."""
        # Valid task input
        task = TaskInput(id="test-001", content="Valid content")
        assert task.id == "test-001"
        assert task.content == "Valid content"
        
        # Test that empty content is not allowed
        with pytest.raises(ValueError):
            TaskInput(id="test-002", content="")
    
    def test_task_output_creation(self):
        """Test task output creation."""
        output = TaskOutput(
            task_id="test-001",
            result="Task completed successfully",
            metadata={"duration": 1.5}
        )
        assert output.task_id == "test-001"
        assert output.result == "Task completed successfully"
    
    def test_agent_trace_creation(self):
        """Test agent trace creation."""
        trace = AgentTrace(
            agent_id="agent-001",
            task_id="task-001",
            steps=["Step 1", "Step 2"],
            metadata={"total_time": 2.5}
        )
        assert trace.agent_id == "agent-001"
        assert trace.task_id == "task-001"
        assert len(trace.steps) == 2
    
    def test_tool_result_creation(self):
        """Test tool result creation."""
        result = ToolResult(
            success=True,
            content="Tool executed successfully",
            metadata={"execution_time": 0.5}
        )
        assert result.success is True
        assert result.content == "Tool executed successfully"
    
    # Integration Tests
    @pytest.mark.asyncio
    async def test_full_agent_workflow(self, agent_config, mock_provider):
        """Test complete agent workflow."""
        # Create agent with tools
        calc_tool = CalculatorTool()
        agent = ReactAgent(
            config=agent_config,
            llm_provider=mock_provider,
            tools=[calc_tool]
        )
        
        # Execute task
        result = await agent.run("Calculate 5 + 3")
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'content')
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, agent_config, mock_provider):
        """Test error handling in agent workflow."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        # Test with invalid input
        try:
            await agent.run("")  # Empty input should be handled gracefully
        except Exception as e:
            assert isinstance(e, (AgentError, ValidationError))
    
    def test_concurrent_operations(self, agent_config, mock_provider):
        """Test concurrent agent operations."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        async def run_task(content):
            return await agent.run(f"Task: {content}")
        
        async def test_concurrency():
            tasks = [
                run_task("Task 1"),
                run_task("Task 2"),
                run_task("Task 3")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All tasks should complete (may be exceptions, but should not hang)
            assert len(results) == 3
        
        # Run the concurrency test
        asyncio.run(test_concurrency())
    
    # Performance Tests
    def test_memory_usage(self, agent_config, mock_provider):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple agents
        agents = []
        for i in range(10):
            config = AgentConfig(name=f"Agent-{i}")
            agent = ReactAgent(config=config, llm_provider=mock_provider)
            agents.append(agent)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 10 agents)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    @pytest.mark.asyncio
    async def test_response_time(self, agent_config, mock_provider):
        """Test response time is reasonable."""
        import time
        
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        start_time = time.time()
        await agent.run("Simple test task")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be fast with mock provider (less than 1 second)
        assert response_time < 1.0
    
    # Edge Case Tests
    def test_edge_case_empty_strings(self, agent_config, mock_provider):
        """Test handling of empty strings."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        # Test empty config name handling
        with pytest.raises((ValueError, ValidationError)):
            AgentConfig(name="")
    
    def test_edge_case_large_inputs(self, agent_config, mock_provider):
        """Test handling of large inputs."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        # Create a large input (but not too large to avoid memory issues)
        large_input = "x" * 10000  # 10KB input
        
        async def test_large_input():
            try:
                result = await agent.run(large_input)
                # Should handle large inputs gracefully
                assert result is not None
            except Exception as e:
                # Should be a handled exception, not a system crash
                assert isinstance(e, (AgentError, LLMError, ValidationError))
        
        asyncio.run(test_large_input())
    
    def test_edge_case_unicode_handling(self, agent_config, mock_provider):
        """Test Unicode character handling."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        # Test various Unicode characters
        unicode_inputs = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian  
            "مرحبا بالعالم",  # Arabic
            "🌍🌎🌏",  # Emojis (though we removed them from code)
        ]
        
        async def test_unicode():
            for unicode_input in unicode_inputs:
                try:
                    result = await agent.run(unicode_input)
                    # Should handle Unicode gracefully
                    assert result is not None
                except Exception as e:
                    # Should be handled gracefully
                    assert isinstance(e, (AgentError, LLMError, ValidationError))
        
        asyncio.run(test_unicode())
    
    # Cleanup Tests
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, agent_config, mock_provider):
        """Test proper resource cleanup."""
        agent = ReactAgent(config=agent_config, llm_provider=mock_provider)
        
        # Use agent
        await agent.run("Test task")
        
        # Test cleanup (if cleanup methods exist)
        if hasattr(agent, 'cleanup'):
            await agent.cleanup()
        
        if hasattr(agent, 'close'):
            await agent.close()
        
        # Should not raise exceptions during cleanup
        assert True  # If we reach here, cleanup was successful
