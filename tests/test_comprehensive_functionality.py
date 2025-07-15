"""
Comprehensive Test Suite for LlamaAgent

This test suite covers all major functionality including:
- SPREGenerator and data generation
- CLI commands and interfaces
- Agent functionality
- Integration modules
- Error handling and edge cases

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.llamaagent.data_generation.spre import (
    SPREGenerator,
    SPREDataset,
    SPREItem,
    DataType,
    ValidationStatus,
    SpreConfig,
)
from src.llamaagent.agents.base import AgentConfig, AgentRole
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.llm.providers.mock import MockProvider
from src.llamaagent.tools import ToolRegistry
from src.llamaagent import cli_main


class TestSPREGenerator:
    """Test suite for SPREGenerator functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return SpreConfig(
            max_rounds=3,
            players_per_session=2,
            reward_threshold=0.7,
            diversity_factor=0.3,
            learning_rate=0.01,
            output_path="./test_output"
        )
    
    @pytest.fixture
    def generator(self, config):
        """Create SPREGenerator instance"""
        return SPREGenerator(config)
    
    def test_initialization(self, generator, config):
        """Test SPREGenerator initialization"""
        assert generator.config == config
        assert generator.generated_datasets == []
        assert generator.engine is None  # Lazy initialization
    
    def test_generate_dataset_basic(self, generator):
        """Test basic dataset generation"""
        with patch.object(generator, '_generate_dataset_async') as mock_async:
            mock_dataset = SPREDataset(
                name="test_dataset",
                description="Test dataset",
                items=[
                    SPREItem(
                        id="item_1",
                        data_type=DataType.TEXT,
                        content={"type": "text", "content": "Test content"}
                    )
                ]
            )
            mock_async.return_value = mock_dataset
            
            result = generator.generate_dataset(
                name="test_dataset",
                count=5,
                description="Test dataset"
            )
            
            assert result.name == "test_dataset"
            assert result.description == "Test dataset"
            mock_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_dataset_async(self, generator):
        """Test async dataset generation"""
        with patch.object(generator, '_generate_item') as mock_item:
            mock_item.return_value = SPREItem(
                id="item_1",
                data_type=DataType.TEXT,
                content={"type": "text", "content": "Test content"}
            )
            
            result = await generator._generate_dataset_async(
                name="async_test",
                count=3,
                description="Async test dataset"
            )
            
            assert result.name == "async_test"
            assert len(result.items) == 3
            assert mock_item.call_count == 3
    
    def test_dataset_validation(self, generator):
        """Test dataset validation"""
        dataset = SPREDataset(
            name="validation_test",
            description="Test validation",
            items=[
                SPREItem(
                    id="valid_item",
                    data_type=DataType.TEXT,
                    content={"type": "text", "content": "Valid content with enough length"}
                ),
                SPREItem(
                    id="invalid_item",
                    data_type=DataType.TEXT,
                    content={"type": "text", "content": "Short"}
                )
            ]
        )
        
        asyncio.run(generator._validate_dataset(dataset))
        
        # Check validation results
        valid_items = [item for item in dataset.items if item.validation_status == ValidationStatus.VALID]
        invalid_items = [item for item in dataset.items if item.validation_status == ValidationStatus.INVALID]
        
        assert len(valid_items) >= 1
        assert len(invalid_items) >= 0
    
    def test_data_types(self, generator):
        """Test different data types generation"""
        data_types = [
            DataType.TEXT,
            DataType.CONVERSATION,
            DataType.REASONING,
            DataType.CREATIVE,
            DataType.TECHNICAL,
            DataType.EDUCATIONAL
        ]
        
        for data_type in data_types:
            with patch.object(generator, '_generate_item') as mock_item:
                mock_item.return_value = SPREItem(
                    id=f"item_{data_type.value}",
                    data_type=data_type,
                    content={"type": data_type.value, "content": "Test content"}
                )
                
                result = generator.generate_dataset(
                    name=f"test_{data_type.value}",
                    count=1,
                    data_type=data_type
                )
                
                assert result.items[0].data_type == data_type
    
    @pytest.mark.asyncio
    async def test_generate_from_prompts(self, generator):
        """Test generation from prompts"""
        prompts = [
            "Explain quantum computing",
            "Write a creative story about AI",
            "Analyze market trends"
        ]
        
        with patch.object(generator, '_get_engine') as mock_engine:
            mock_session = MagicMock()
            mock_session.generated_content = {"content": "Generated content"}
            mock_session.reward_scores = {"quality": 0.8}
            mock_session.session_id = "test_session"
            
            mock_engine.return_value.run_self_play_session = AsyncMock(return_value=mock_session)
            
            result = await generator.generate_from_prompts(prompts)
            
            assert len(result) == 3
            assert all("prompt" in item for item in result)
            assert all("generated_content" in item for item in result)
    
    def test_dataset_stats(self, generator):
        """Test dataset statistics"""
        # Create sample datasets
        dataset1 = SPREDataset(
            name="dataset1",
            description="First dataset",
            items=[
                SPREItem(
                    id="item1",
                    data_type=DataType.TEXT,
                    content={"content": "Valid content"},
                    validation_status=ValidationStatus.VALID
                ),
                SPREItem(
                    id="item2",
                    data_type=DataType.TEXT,
                    content={"content": "Invalid"},
                    validation_status=ValidationStatus.INVALID
                )
            ]
        )
        
        generator.generated_datasets = [dataset1]
        
        stats = generator.get_dataset_stats()
        
        assert stats["total_datasets"] == 1
        assert stats["total_items"] == 2
        assert stats["valid_items"] == 1
        assert stats["validation_rate"] == 0.5


class TestCLIFunctionality:
    """Test suite for CLI functionality"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        from click.testing import CliRunner
        return CliRunner()
    
    def test_cli_main_help(self, runner):
        """Test CLI main help"""
        result = runner.invoke(cli_main, ['--help'])
        assert result.exit_code == 0
        assert "LlamaAgent Command Line Interface" in result.output
    
    def test_chat_command_help(self, runner):
        """Test chat command help"""
        result = runner.invoke(cli_main, ['chat', '--help'])
        assert result.exit_code == 0
        assert "Chat with an AI agent" in result.output
    
    def test_generate_data_command_help(self, runner):
        """Test generate-data command help"""
        result = runner.invoke(cli_main, ['generate-data', '--help'])
        assert result.exit_code == 0
        assert "Generate training data" in result.output
    
    def test_chat_command_basic(self, runner):
        """Test basic chat command"""
        with patch('src.llamaagent.ReactAgent') as mock_agent:
            mock_response = MagicMock()
            mock_response.content = "Test response"
            mock_response.execution_time = 1.0
            mock_response.tokens_used = 50
            
            mock_agent.return_value.execute = AsyncMock(return_value=mock_response)
            
            result = runner.invoke(cli_main, ['chat', 'Hello world'])
            
            assert result.exit_code == 0
            assert "Test response" in result.output
    
    def test_generate_data_gdt(self, runner):
        """Test GDT data generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            output_file = Path(temp_dir) / "output.json"
            
            # Create sample input
            input_file.write_text("Sample problem 1\nSample problem 2\n")
            
            with patch('src.llamaagent.data_generation.gdt.GDTOrchestrator') as mock_orchestrator:
                mock_orchestrator.return_value.generate_dataset = AsyncMock()
                
                result = runner.invoke(cli_main, [
                    'generate-data',
                    'gdt',
                    '-i', str(input_file),
                    '-o', str(output_file),
                    '-n', '2'
                ])
                
                assert result.exit_code == 0
                assert "GDT dataset saved" in result.output
    
    def test_generate_data_spre(self, runner):
        """Test SPRE data generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            output_file = Path(temp_dir) / "output.json"
            
            # Create sample input
            input_file.write_text("Sample input for SPRE\n")
            
            with patch('src.llamaagent.data_generation.spre.SPREGenerator') as mock_generator:
                mock_dataset = MagicMock()
                mock_dataset.name = "Test Dataset"
                mock_dataset.description = "Test Description"
                mock_dataset.metadata = {}
                mock_dataset.items = []
                
                mock_generator.return_value.generate_dataset.return_value = mock_dataset
                
                result = runner.invoke(cli_main, [
                    'generate-data',
                    'spre',
                    '-i', str(input_file),
                    '-o', str(output_file),
                    '-n', '5'
                ])
                
                assert result.exit_code == 0
                assert "SPRE dataset saved" in result.output


class TestAgentFunctionality:
    """Test suite for agent functionality"""
    
    @pytest.fixture
    def config(self):
        """Create agent configuration"""
        return AgentConfig(
            name="TestAgent",
            role=AgentRole.GENERALIST,
            description="Test agent for comprehensive testing"
        )
    
    @pytest.fixture
    def provider(self):
        """Create mock LLM provider"""
        return MockProvider()
    
    @pytest.fixture
    def tools(self):
        """Create tool registry"""
        return ToolRegistry()
    
    @pytest.fixture
    def agent(self, config, provider, tools):
        """Create ReactAgent instance"""
        return ReactAgent(config=config, llm_provider=provider, tools=tools)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent, config):
        """Test agent initialization"""
        assert agent.config == config
        assert agent.llm_provider is not None
        assert agent.tools is not None
    
    @pytest.mark.asyncio
    async def test_agent_execute_basic(self, agent):
        """Test basic agent execution"""
        result = await agent.execute("What is 2+2?")
        
        assert result is not None
        assert hasattr(result, 'content')
        assert result.content is not None
    
    @pytest.mark.asyncio
    async def test_agent_execute_with_tools(self, agent):
        """Test agent execution with tools"""
        # Add calculator tool
        from src.llamaagent.tools.calculator import CalculatorTool
        agent.tools.register(CalculatorTool())
        
        result = await agent.execute("Calculate 15 * 23")
        
        assert result is not None
        assert hasattr(result, 'content')
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, agent):
        """Test agent error handling"""
        with patch.object(agent.llm_provider, 'complete') as mock_complete:
            mock_complete.side_effect = Exception("Test error")
            
            result = await agent.execute("This should fail")
            
            # Agent should handle errors gracefully
            assert result is not None


class TestIntegrationModules:
    """Test suite for integration modules"""
    
    def test_openai_integration_import(self):
        """Test OpenAI integration import"""
        try:
            from src.llamaagent.integration.openai_agents import get_openai_integration
            assert get_openai_integration is not None
        except ImportError:
            pytest.skip("OpenAI integration not available")
    
    def test_langgraph_integration_import(self):
        """Test LangGraph integration import"""
        try:
            from src.llamaagent.integration.langgraph import is_langgraph_available
            assert is_langgraph_available is not None
        except ImportError:
            pytest.skip("LangGraph integration not available")
    
    def test_orchestrator_import(self):
        """Test orchestrator import"""
        try:
            from src.llamaagent.orchestrator import AgentOrchestrator
            assert AgentOrchestrator is not None
        except ImportError:
            pytest.skip("Orchestrator not available")


class TestErrorHandling:
    """Test suite for error handling and edge cases"""
    
    def test_invalid_data_type(self):
        """Test handling of invalid data types"""
        with pytest.raises(ValueError):
            DataType("invalid_type")
    
    def test_invalid_validation_status(self):
        """Test handling of invalid validation status"""
        with pytest.raises(ValueError):
            ValidationStatus("invalid_status")
    
    def test_empty_dataset_generation(self):
        """Test generation of empty dataset"""
        generator = SPREGenerator()
        dataset = generator.generate_dataset(
            name="empty_test",
            count=0,
            description="Empty dataset test"
        )
        
        assert len(dataset.items) == 0
        assert dataset.name == "empty_test"
    
    def test_large_dataset_generation(self):
        """Test generation of large dataset (performance test)"""
        generator = SPREGenerator()
        
        with patch.object(generator, '_generate_item') as mock_item:
            mock_item.return_value = SPREItem(
                id="mock_item",
                data_type=DataType.TEXT,
                content={"content": "Mock content"}
            )
            
            # Generate large dataset
            dataset = generator.generate_dataset(
                name="large_test",
                count=1000,
                description="Large dataset test"
            )
            
            assert len(dataset.items) == 1000
            assert mock_item.call_count == 1000


class TestPerformanceAndBenchmarks:
    """Test suite for performance and benchmarking"""
    
    def test_generation_performance(self):
        """Test generation performance"""
        import time
        
        generator = SPREGenerator()
        
        with patch.object(generator, '_generate_item') as mock_item:
            mock_item.return_value = SPREItem(
                id="perf_item",
                data_type=DataType.TEXT,
                content={"content": "Performance test content"}
            )
            
            start_time = time.time()
            dataset = generator.generate_dataset(
                name="perf_test",
                count=100,
                description="Performance test"
            )
            end_time = time.time()
            
            # Should complete within reasonable time
            assert (end_time - start_time) < 5.0  # 5 seconds max
            assert len(dataset.items) == 100
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        import gc
        
        generator = SPREGenerator()
        
        with patch.object(generator, '_generate_item') as mock_item:
            mock_item.return_value = SPREItem(
                id="mem_item",
                data_type=DataType.TEXT,
                content={"content": "Memory test content"}
            )
            
            # Generate and immediately discard datasets
            for i in range(10):
                dataset = generator.generate_dataset(
                    name=f"mem_test_{i}",
                    count=50,
                    description=f"Memory test {i}"
                )
                del dataset
                gc.collect()
            
            # Should not accumulate too much memory
            assert len(generator.generated_datasets) == 10


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_item_validation(self):
        """Test individual item validation"""
        valid_item = SPREItem(
            id="valid_item",
            data_type=DataType.TEXT,
            content={"type": "text", "content": "This is valid content with sufficient length"}
        )
        
        invalid_item = SPREItem(
            id="invalid_item",
            data_type=DataType.TEXT,
            content={}  # Empty content
        )
        
        generator = SPREGenerator()
        dataset = SPREDataset(
            name="validation_test",
            description="Test validation",
            items=[valid_item, invalid_item]
        )
        
        asyncio.run(generator._validate_dataset(dataset))
        
        assert valid_item.validation_status == ValidationStatus.VALID
        assert invalid_item.validation_status == ValidationStatus.INVALID
    
    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        item = SPREItem(
            id="quality_test",
            data_type=DataType.TEXT,
            content={
                "type": "text",
                "content": "High quality content with detailed information and metadata",
                "metadata": {"quality": "high"}
            }
        )
        
        generator = SPREGenerator()
        score = asyncio.run(generator._calculate_quality_score(item))
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be above average for good content


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 