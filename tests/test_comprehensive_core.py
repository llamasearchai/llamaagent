"""
Comprehensive test suite for LlamaAgent core functionality.
Ensures 95%+ coverage of critical modules.
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Core imports
from src.llamaagent import ReactAgent
from src.llamaagent.agents.base import BaseAgent, AgentConfig
from src.llamaagent.llm import LLMProvider, MockProvider
from src.llamaagent.llm.factory import LLMFactory
from src.llamaagent.llm.messages import Message, MessageRole
from src.llamaagent.tools import BaseTool, Tool, ToolRegistry
from src.llamaagent.tools.calculator import CalculatorTool
from src.llamaagent.tools.python_repl import PythonREPLTool
from src.llamaagent.memory import BaseMemory, MemoryEntry
from src.llamaagent.types import (
    AgentResponse, 
    ToolCall, 
    ToolResult,
    AgentTrace,
    ConversationTurn
)

# Integration imports
from src.llamaagent.api import create_app
from src.llamaagent.cli.interactive import InteractiveCLI
from src.llamaagent.storage import DatabaseManager, VectorMemory
from src.llamaagent.cache import CacheManager, ResultCache
from src.llamaagent.security import RateLimiter, SecurityManager
from src.llamaagent.benchmarks import SPREEvaluator, GAIABenchmark
from src.llamaagent.data_generation import SPREGenerator, GDTGenerator


class TestCoreAgents:
    """Test core agent functionality."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MockProvider()
        provider.generate = AsyncMock(return_value="Test response")
        provider.generate_structured = AsyncMock(return_value={"action": "test"})
        return provider
    
    @pytest.fixture
    def agent_config(self, mock_provider):
        """Create agent configuration."""
        return AgentConfig(
            name="TestAgent",
            model="mock",
            temperature=0.7,
            max_tokens=100,
            provider=mock_provider
        )
    
    @pytest.fixture
    def calculator_tool(self):
        """Create calculator tool."""
        return CalculatorTool()
    
    @pytest.fixture
    def python_tool(self):
        """Create Python REPL tool."""
        return PythonREPLTool()
    
    @pytest.mark.asyncio
    async def test_base_agent_initialization(self, agent_config):
        """Test base agent initialization."""
        agent = BaseAgent(config=agent_config)
        
        assert agent.config == agent_config
        assert agent.name == "TestAgent"
        assert len(agent.memory.entries) == 0
        assert len(agent.tools) == 0
    
    @pytest.mark.asyncio
    async def test_react_agent_execution(self, mock_provider, calculator_tool):
        """Test ReAct agent execution."""
        agent = ReactAgent(
            name="TestReActAgent",
            provider=mock_provider,
            tools=[calculator_tool]
        )
        
        # Mock the provider to return a calculation request
        mock_provider.generate_structured = AsyncMock(
            side_effect=[
                {
                    "thought": "I need to calculate 2 + 2",
                    "action": "calculator",
                    "action_input": {"expression": "2 + 2"}
                },
                {
                    "thought": "I have the answer",
                    "action": "finish",
                    "action_input": {"response": "The answer is 4"}
                }
            ]
        )
        
        response = await agent.run("What is 2 + 2?")
        
        assert response.success
        assert "4" in response.response
        assert len(response.trace.steps) >= 2
    
    @pytest.mark.asyncio
    async def test_agent_with_memory(self, mock_provider):
        """Test agent with memory functionality."""
        agent = ReactAgent(
            name="MemoryAgent",
            provider=mock_provider
        )
        
        # Add some memory
        agent.memory.add("user", "Hello")
        agent.memory.add("assistant", "Hi there!")
        
        assert len(agent.memory.entries) == 2
        assert agent.memory.entries[0].role == "user"
        assert agent.memory.entries[1].content == "Hi there!"
        
        # Test memory retrieval
        context = agent.memory.get_context(max_tokens=100)
        assert "Hello" in context
        assert "Hi there!" in context
    
    @pytest.mark.asyncio
    async def test_agent_tool_execution(self, mock_provider, calculator_tool, python_tool):
        """Test agent executing multiple tools."""
        agent = ReactAgent(
            name="ToolAgent",
            provider=mock_provider,
            tools=[calculator_tool, python_tool]
        )
        
        # Test calculator tool
        calc_result = await calculator_tool.execute({"expression": "10 * 5"})
        assert calc_result.success
        assert calc_result.output == "50"
        
        # Test Python tool
        python_result = await python_tool.execute({"code": "print('Hello')"})
        assert python_result.success
        assert "Hello" in python_result.output
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_provider):
        """Test agent error handling."""
        agent = ReactAgent(
            name="ErrorAgent",
            provider=mock_provider
        )
        
        # Mock provider to raise an error
        mock_provider.generate = AsyncMock(
            side_effect=Exception("Provider error")
        )
        
        response = await agent.run("Test query")
        
        assert not response.success
        assert "error" in response.response.lower()


class TestLLMProviders:
    """Test LLM provider functionality."""
    
    def test_mock_provider(self):
        """Test mock provider."""
        provider = MockProvider()
        
        assert provider.model == "mock"
        assert provider.is_available()
    
    @pytest.mark.asyncio
    async def test_mock_provider_generation(self):
        """Test mock provider text generation."""
        provider = MockProvider()
        
        response = await provider.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_mock_provider_structured_generation(self):
        """Test mock provider structured generation."""
        provider = MockProvider()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        response = await provider.generate_structured("Test prompt", schema)
        assert isinstance(response, dict)
        assert "mock" in str(response).lower()
    
    def test_provider_factory(self):
        """Test LLM provider factory."""
        # Test mock provider creation
        mock_provider = LLMFactory.create_provider("mock")
        assert isinstance(mock_provider, MockProvider)
        
        # Test invalid provider
        with pytest.raises(ValueError):
            LLMFactory.create_provider("invalid_provider")
    
    @pytest.mark.asyncio
    async def test_provider_with_messages(self):
        """Test provider with message history."""
        provider = MockProvider()
        
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
            Message(role=MessageRole.USER, content="How are you?")
        ]
        
        response = await provider.generate_from_messages(messages)
        assert isinstance(response, str)


class TestTools:
    """Test tool functionality."""
    
    def test_tool_registry(self):
        """Test tool registry."""
        registry = ToolRegistry()
        
        # Register a tool
        calc_tool = CalculatorTool()
        registry.register(calc_tool)
        
        assert "calculator" in registry.list_tools()
        assert registry.get_tool("calculator") == calc_tool
    
    def test_custom_tool_creation(self):
        """Test custom tool creation."""
        @Tool(
            name="custom_tool",
            description="A custom tool for testing"
        )
        async def custom_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        assert custom_tool.name == "custom_tool"
        assert custom_tool.description == "A custom tool for testing"
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_validation(self):
        """Test tool execution with input validation."""
        calc_tool = CalculatorTool()
        
        # Valid input
        result = await calc_tool.execute({"expression": "2 + 2"})
        assert result.success
        assert result.output == "4"
        
        # Invalid input
        result = await calc_tool.execute({"invalid": "input"})
        assert not result.success
        assert "error" in result.output.lower()
    
    def test_tool_compatibility_alias(self):
        """Test backward compatibility alias."""
        # Test that Tool is available as an alias
        assert Tool == BaseTool


class TestMemory:
    """Test memory functionality."""
    
    def test_memory_entry_creation(self):
        """Test memory entry creation."""
        entry = MemoryEntry(
            role="user",
            content="Test message",
            metadata={"timestamp": "2024-01-01"}
        )
        
        assert entry.role == "user"
        assert entry.content == "Test message"
        assert entry.metadata["timestamp"] == "2024-01-01"
    
    def test_base_memory_operations(self):
        """Test base memory operations."""
        memory = BaseMemory()
        
        # Add entries
        memory.add("user", "Hello")
        memory.add("assistant", "Hi!")
        
        assert len(memory.entries) == 2
        
        # Get context
        context = memory.get_context(max_tokens=50)
        assert "Hello" in context
        assert "Hi!" in context
        
        # Clear memory
        memory.clear()
        assert len(memory.entries) == 0
    
    def test_memory_token_limit(self):
        """Test memory token limiting."""
        memory = BaseMemory()
        
        # Add many entries
        for i in range(100):
            memory.add("user", f"Message {i}")
        
        # Get limited context
        context = memory.get_context(max_tokens=100)
        
        # Should contain recent messages
        assert "Message 99" in context
        # Should not contain all messages
        assert "Message 0" not in context


class TestAPI:
    """Test API functionality."""
    
    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = await response.get_json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_agent_chat_endpoint(self, client):
        """Test agent chat endpoint."""
        with patch('src.llamaagent.api.ReactAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run = AsyncMock(
                return_value=AgentResponse(
                    success=True,
                    response="Test response",
                    trace=AgentTrace(steps=[])
                )
            )
            mock_agent_class.return_value = mock_agent
            
            response = await client.post(
                "/chat",
                json={
                    "message": "Hello",
                    "agent_name": "test_agent"
                }
            )
            
            assert response.status_code == 200
            data = await response.get_json()
            assert data["response"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_tools_list_endpoint(self, client):
        """Test tools listing endpoint."""
        response = await client.get("/tools")
        assert response.status_code == 200
        data = await response.get_json()
        assert isinstance(data["tools"], list)


class TestCLI:
    """Test CLI functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for CLI testing."""
        agent = MagicMock()
        agent.run = AsyncMock(
            return_value=AgentResponse(
                success=True,
                response="CLI test response",
                trace=AgentTrace(steps=[])
            )
        )
        return agent
    
    @pytest.mark.asyncio
    async def test_interactive_cli_initialization(self):
        """Test interactive CLI initialization."""
        cli = InteractiveCLI()
        
        assert cli.running is False
        assert cli.agent is None
        assert len(cli.conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_cli_command_processing(self, mock_agent):
        """Test CLI command processing."""
        cli = InteractiveCLI()
        cli.agent = mock_agent
        
        # Test regular message
        await cli.process_input("Hello")
        assert mock_agent.run.called
        
        # Test help command
        with patch('src.llamaagent.cli.interactive.console') as mock_console:
            await cli.process_input("/help")
            assert mock_console.print.called


class TestStorage:
    """Test storage functionality."""
    
    @pytest.mark.asyncio
    async def test_database_manager(self):
        """Test database manager."""
        # Use in-memory SQLite for testing
        db = DatabaseManager("sqlite:///:memory:")
        
        await db.initialize()
        
        # Test user creation
        user_id = await db.create_user("test_user", "test@example.com")
        assert user_id is not None
        
        # Test session creation
        session_id = await db.create_session(user_id, {"model": "test"})
        assert session_id is not None
        
        await db.close()
    
    @pytest.mark.asyncio
    async def test_vector_memory(self):
        """Test vector memory storage."""
        memory = VectorMemory()
        
        # Add embeddings
        await memory.add_embedding("doc1", [0.1, 0.2, 0.3], {"text": "Hello"})
        await memory.add_embedding("doc2", [0.4, 0.5, 0.6], {"text": "World"})
        
        # Search similar
        results = await memory.search_similar([0.15, 0.25, 0.35], k=1)
        
        assert len(results) == 1
        assert results[0]["id"] == "doc1"


class TestCache:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_manager(self):
        """Test cache manager."""
        cache = CacheManager()
        
        # Set value
        await cache.set("test_key", "test_value", ttl=60)
        
        # Get value
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Delete value
        deleted = await cache.delete("test_key")
        assert deleted
        
        # Get deleted value
        value = await cache.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_result_cache_decorator(self):
        """Test result cache decorator."""
        call_count = 0
        
        @ResultCache.cache(ttl=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (cached)
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented


class TestSecurity:
    """Test security functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # First two requests should pass
        assert await limiter.check_limit("user1")
        assert await limiter.check_limit("user1")
        
        # Third request should be limited
        assert not await limiter.check_limit("user1")
        
        # Different user should not be limited
        assert await limiter.check_limit("user2")
    
    @pytest.mark.asyncio
    async def test_security_manager(self):
        """Test security manager."""
        manager = SecurityManager()
        
        # Test input validation
        assert manager.validate_input("Normal input")
        assert not manager.validate_input("<script>alert('xss')</script>")
        
        # Test API key validation
        with patch.dict('os.environ', {'API_KEY': 'test_key'}):
            assert manager.validate_api_key("test_key")
            assert not manager.validate_api_key("wrong_key")


class TestBenchmarks:
    """Test benchmark functionality."""
    
    @pytest.mark.asyncio
    async def test_spre_evaluator(self):
        """Test SPRE evaluator."""
        evaluator = SPREEvaluator()
        
        # Mock agent for testing
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=AgentResponse(
                success=True,
                response="42",
                trace=AgentTrace(steps=[])
            )
        )
        
        # Create test task
        test_task = {
            "question": "What is 2 + 2?",
            "answer": "4",
            "reasoning_steps": ["Add 2 and 2"]
        }
        
        # Evaluate
        result = await evaluator.evaluate_task(mock_agent, test_task)
        
        assert "correct" in result
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_gaia_benchmark(self):
        """Test GAIA benchmark."""
        benchmark = GAIABenchmark()
        
        # Load test data
        test_data = [
            {
                "task_id": "test_1",
                "question": "Test question",
                "level": 1,
                "final_answer": "Test answer"
            }
        ]
        
        benchmark.tasks = test_data
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=AgentResponse(
                success=True,
                response="Test answer",
                trace=AgentTrace(steps=[])
            )
        )
        
        # Run benchmark
        results = await benchmark.run_benchmark(mock_agent, num_tasks=1)
        
        assert len(results) == 1
        assert results[0]["task_id"] == "test_1"


class TestDataGeneration:
    """Test data generation functionality."""
    
    @pytest.mark.asyncio
    async def test_spre_generator(self):
        """Test SPRE data generator."""
        generator = SPREGenerator()
        
        # Mock LLM for generation
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(
            return_value=json.dumps({
                "question": "Generated question",
                "answer": "Generated answer",
                "reasoning_steps": ["Step 1", "Step 2"]
            })
        )
        
        generator.llm = mock_llm
        
        # Generate data
        data = await generator.generate_task("math")
        
        assert data["question"] == "Generated question"
        assert len(data["reasoning_steps"]) == 2
    
    @pytest.mark.asyncio
    async def test_gdt_generator(self):
        """Test GDT data generator."""
        generator = GDTGenerator()
        
        # Mock components
        generator.generate_topic = AsyncMock(return_value="Test topic")
        generator.generate_debate = AsyncMock(
            return_value={
                "topic": "Test topic",
                "positions": ["Pro", "Con"],
                "arguments": []
            }
        )
        
        # Generate debate
        debate = await generator.generate_complete_debate()
        
        assert debate["topic"] == "Test topic"
        assert len(debate["positions"]) == 2


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_agent_workflow(self):
        """Test complete agent workflow with all components."""
        # Create agent with tools and memory
        agent = ReactAgent(
            name="IntegrationAgent",
            provider=MockProvider(),
            tools=[CalculatorTool(), PythonREPLTool()],
            enable_memory=True
        )
        
        # Add to cache
        cache = CacheManager()
        await cache.set("agent_config", agent.config.dict(), ttl=3600)
        
        # Create API app
        app = create_app()
        
        # Test agent execution
        response = await agent.run("Calculate 10 * 20")
        
        assert response.success
        assert "200" in response.response
        
        # Verify memory was updated
        assert len(agent.memory.entries) > 0
        
        # Verify cache
        cached_config = await cache.get("agent_config")
        assert cached_config["name"] == "IntegrationAgent"
    
    @pytest.mark.asyncio
    async def test_benchmark_workflow(self):
        """Test complete benchmark workflow."""
        # Create agent
        agent = ReactAgent(
            name="BenchmarkAgent",
            provider=MockProvider()
        )
        
        # Create evaluator
        evaluator = SPREEvaluator()
        
        # Create test dataset
        test_dataset = [
            {
                "question": "What is 5 + 5?",
                "answer": "10",
                "reasoning_steps": ["Add 5 and 5 to get 10"]
            }
        ]
        
        # Run evaluation
        results = []
        for task in test_dataset:
            result = await evaluator.evaluate_task(agent, task)
            results.append(result)
        
        assert len(results) == 1
        assert "response" in results[0]


# Performance tests
class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """Test concurrent agent execution."""
        agents = [
            ReactAgent(name=f"Agent{i}", provider=MockProvider())
            for i in range(10)
        ]
        
        # Run agents concurrently
        tasks = [
            agent.run(f"Query {i}")
            for i, agent in enumerate(agents)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 10
        assert all(r.success for r in responses)
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance under load."""
        cache = CacheManager()
        
        # Set many values
        set_tasks = [
            cache.set(f"key_{i}", f"value_{i}", ttl=60)
            for i in range(100)
        ]
        await asyncio.gather(*set_tasks)
        
        # Get many values
        get_tasks = [
            cache.get(f"key_{i}")
            for i in range(100)
        ]
        values = await asyncio.gather(*get_tasks)
        
        assert len(values) == 100
        assert all(v == f"value_{i}" for i, v in enumerate(values))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/llamaagent", "--cov-report=term-missing"])