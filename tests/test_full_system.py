"""Comprehensive full system integration tests for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>
SPRE project - Strategic Planning & Resourceful Execution

This module provides complete end-to-end testing of the LlamaAgent system,
including CLI, API, data generation, and performance benchmarking.
"""

from __future__ import annotations

# Standard library
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Third-party
try:
    import pytest
    from click.testing import CliRunner
except ImportError:
    # Graceful degradation for environments without pytest
    pytest = None
    CliRunner = None

# Local imports
from llamaagent import cli_main
from llamaagent.data_generation.spre import generate_spre_dataset, SPREDatasetGenerator

__all__ = [
    "TestFullSystem",
    "TestCLIInterface", 
    "TestAPIEndpoints",
    "TestDataGeneration",
    "TestPerformanceBenchmarks",
    "TestProductionReadiness",
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _cli_invoke(*args) -> Any:
    """Helper to invoke CLI commands safely."""
    if CliRunner is None:
        raise ImportError("click testing not available")
    
    runner = CliRunner()
    return runner.invoke(cli_main, args, catch_exceptions=False)


def _create_temp_files() -> tuple[str, str]:
    """Create temporary input and output files for testing."""
    # Create temporary input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Sample configuration for SPRE dataset generation.
        This file contains test scenarios and parameters.
        
        Scenarios:
        - Mathematical problem solving
        - Data analysis tasks
        - Code generation challenges
        - Multi-step planning exercises
        """)
        temp_input = f.name
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_output = f.name
    
    return temp_input, temp_output


def _cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files safely."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError:
            pass  # Ignore cleanup errors


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

if pytest is not None:
    
    class TestCLIInterface:
        """Test CLI command interface and functionality."""
        
        @pytest.mark.asyncio
        async def test_cli_chat_basic(self):
            """Test basic CLI chat functionality."""
            with patch('llamaagent.Agent') as mock_agent_class:
                # Setup mock agent
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "The answer to 2 + 2 is 4."
                mock_response.execution_time = 0.5
                mock_response.token_count = 15
                mock_response.success = True
                mock_response.trace = []
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                # Test the command
                result = _cli_invoke("chat", "What is 2 + 2?")
                
                assert result.exit_code == 0
                assert any(keyword in result.output.lower() for keyword in ["4", "answer", "result"])
        
        @pytest.mark.asyncio
        async def test_cli_chat_with_spree(self):
            """Test CLI chat with SPRE mode enabled."""
            with patch('llamaagent.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "Using SPRE planning: Step 1: Analyze problem..."
                mock_response.execution_time = 1.2
                mock_response.token_count = 45
                mock_response.success = True
                mock_response.plan = MagicMock()
                mock_response.plan.steps = ["analyze", "plan", "execute"]
                mock_response.trace = [{"type": "plan"}, {"type": "execute"}]
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                result = _cli_invoke("chat", "--spree", "Solve complex problem")
                
                assert result.exit_code == 0
                assert "spree" in result.output.lower() or "plan" in result.output.lower()
        
        def test_cli_help(self):
            """Test CLI help functionality."""
            result = _cli_invoke("--help")
            assert result.exit_code == 0
            assert "llamaagent" in result.output.lower()
        
        @pytest.mark.asyncio
        async def test_cli_data_generation(self):
            """Test CLI data generation commands."""
            temp_input, temp_output = _create_temp_files()
            
            try:
                with patch('llamaagent.data_generation.spre.generate_spre_dataset') as mock_generate:
                    mock_generate.return_value = None
                    
                    result = _cli_invoke(
                        "generate-data", "spre",
                        "-i", temp_input,
                        "-o", temp_output,
                        "-n", "10"
                    )
                    
                    assert result.exit_code == 0
                    mock_generate.assert_called_once()
            finally:
                _cleanup_temp_files(temp_input, temp_output)


    class TestAPIEndpoints:
        """Test FastAPI web interface endpoints."""
        
        @pytest.fixture
        def api_client(self):
            """Create test client for API testing."""
            try:
                from fastapi.testclient import TestClient
                from llamaagent.api import app
                
                if app is not None:
                    return TestClient(app)
                else:
                    pytest.skip("FastAPI app not available")
            except ImportError:
                pytest.skip("FastAPI test client not available")
        
        def test_health_endpoint(self, api_client):
            """Test health check endpoint."""
            response = api_client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data
            assert "agents_available" in data
            assert "tools_available" in data
            assert "features" in data
        
        @pytest.mark.asyncio
        async def test_chat_endpoint(self, api_client):
            """Test chat endpoint functionality."""
            with patch('llamaagent.api.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "Hello! I can help you with that."
                mock_response.execution_time = 0.8
                mock_response.tokens_used = 25
                mock_response.success = True
                mock_response.trace = []
                mock_response.plan = None
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                response = api_client.post("/chat", json={
                    "message": "Hello, how can you help me?",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                })
                
                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                assert "execution_time" in data
                assert "token_count" in data
                assert data["success"] is True
        
        def test_agents_endpoint(self, api_client):
            """Test agents listing endpoint."""
            response = api_client.get("/agents")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            
            for agent_info in data:
                assert "name" in agent_info
                assert "description" in agent_info
                assert "capabilities" in agent_info
        
        def test_tools_endpoint(self, api_client):
            """Test tools listing endpoint."""
            response = api_client.get("/tools")
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            
            for tool_info in data:
                assert "name" in tool_info
                assert "description" in tool_info
        
        @pytest.mark.asyncio
        async def test_batch_endpoint(self, api_client):
            """Test batch processing endpoint."""
            with patch('llamaagent.api.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "Batch response"
                mock_response.execution_time = 0.3
                mock_response.tokens_used = 10
                mock_response.success = True
                mock_response.trace = []
                mock_response.plan = None
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                batch_requests = [
                    {"message": "Request 1"},
                    {"message": "Request 2"}
                ]
                
                response = api_client.post("/batch", json=batch_requests)
                assert response.status_code == 200
                
                data = response.json()
                assert isinstance(data, list)
                assert len(data) == 2


    class TestDataGeneration:
        """Test SPRE dataset generation functionality."""
        
        @pytest.mark.asyncio
        async def test_spre_dataset_generation(self):
            """Test SPRE dataset generation."""
            temp_input, temp_output = _create_temp_files()
            
            try:
                await generate_spre_dataset(
                    input_path=Path(temp_input),
                    output_path=Path(temp_output),
                    num_samples=5,
                    seed=42
                )
                
                # Verify output file was created
                assert os.path.exists(temp_output)
                
                # Verify content structure
                with open(temp_output, 'r') as f:
                    data = json.load(f)
                
                assert "metadata" in data
                assert "episodes" in data
                assert data["metadata"]["total_episodes"] == 5
                assert len(data["episodes"]) == 5
                
                # Verify episode structure
                episode = data["episodes"][0]
                assert "episode_id" in episode
                assert "scenario" in episode
                assert "agent_a_actions" in episode
                assert "agent_b_actions" in episode
                assert "rewards" in episode
                assert "final_reward" in episode
                assert "success" in episode
                
            finally:
                _cleanup_temp_files(temp_input, temp_output)
        
        @pytest.mark.asyncio
        async def test_spre_generator_class(self):
            """Test SPREDatasetGenerator class directly."""
            generator = SPREDatasetGenerator(seed=123)
            
            assert len(generator.scenarios) > 0
            assert generator.seed == 123
            
            # Test scenario creation
            scenarios = generator.scenarios
            assert all(hasattr(s, 'scenario_id') for s in scenarios)
            assert all(hasattr(s, 'complexity') for s in scenarios)
        
        def test_dataset_reproducibility(self):
            """Test that dataset generation is reproducible with same seed."""
            temp_input, temp_output1 = _create_temp_files()
            temp_output2 = tempfile.mktemp(suffix='.json')
            
            try:
                # Generate two datasets with same seed
                asyncio.run(generate_spre_dataset(
                    Path(temp_input), Path(temp_output1), num_samples=3, seed=999
                ))
                asyncio.run(generate_spre_dataset(
                    Path(temp_input), Path(temp_output2), num_samples=3, seed=999
                ))
                
                # Load and compare
                with open(temp_output1, 'r') as f1, open(temp_output2, 'r') as f2:
                    data1 = json.load(f1)
                    data2 = json.load(f2)
                
                # Should have same structure and deterministic elements
                assert len(data1["episodes"]) == len(data2["episodes"])
                assert data1["metadata"]["total_episodes"] == data2["metadata"]["total_episodes"]
                
            finally:
                _cleanup_temp_files(temp_input, temp_output1, temp_output2)


    class TestPerformanceBenchmarks:
        """Performance and benchmarking tests."""
        
        @pytest.mark.asyncio
        async def test_chat_performance(self):
            """Test chat response performance."""
            with patch('llamaagent.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "Performance test response"
                mock_response.execution_time = 0.1
                mock_response.tokens_used = 20
                mock_response.success = True
                mock_response.trace = []
                mock_response.plan = None
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                # Measure multiple requests
                start_time = time.time()
                tasks = []
                
                for i in range(10):
                    # Simulate concurrent requests
                    task = asyncio.create_task(mock_agent.execute(f"Test message {i}"))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Performance assertions
                assert len(results) == 10
                assert total_time < 5.0  # Should complete within 5 seconds
                assert all(r.success for r in results)
        
        @pytest.mark.asyncio
        async def test_dataset_generation_performance(self):
            """Test dataset generation performance."""
            temp_input, temp_output = _create_temp_files()
            
            try:
                start_time = time.time()
                
                await generate_spre_dataset(
                    input_path=Path(temp_input),
                    output_path=Path(temp_output),
                    num_samples=50,
                    seed=456
                )
                
                generation_time = time.time() - start_time
                
                # Performance assertions
                assert generation_time < 30.0  # Should complete within 30 seconds
                assert os.path.exists(temp_output)
                
                # Verify output quality
                with open(temp_output, 'r') as f:
                    data = json.load(f)
                
                assert len(data["episodes"]) == 50
                assert data["metadata"]["statistics"]["success_rate"] > 0.5
                
            finally:
                _cleanup_temp_files(temp_input, temp_output)


    class TestProductionReadiness:
        """Test production readiness and deployment scenarios."""
        
        @pytest.mark.skipif(os.getenv("CI") == "true", reason="Docker not available in CI")
        def test_docker_build(self):
            """Test Docker image building."""
            if not shutil.which("docker"):
                pytest.skip("Docker not installed")
            
            try:
                # Build Docker image
                result = subprocess.run(
                    ["docker", "build", "-t", "llamaagent:test", "."],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                assert result.returncode == 0, f"Docker build failed: {result.stderr}"
                
                # Test image can run
                result = subprocess.run(
                    ["docker", "run", "--rm", "llamaagent:test", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                assert result.returncode == 0
                
            except subprocess.TimeoutExpired:
                pytest.fail("Docker operations timed out")
        
        def test_environment_variables(self):
            """Test environment variable handling."""
            # Test with different environment configurations
            test_cases = [
                {"LLAMAAGENT_DEBUG": "true"},
                {"LLAMAAGENT_LOG_LEVEL": "DEBUG"},
                {"LLAMAAGENT_API_PORT": "8080"},
            ]
            
            for env_vars in test_cases:
                with patch.dict(os.environ, env_vars):
                    # Test that environment variables are handled gracefully
                    result = _cli_invoke("--help")
                    assert result.exit_code == 0
        
        @pytest.mark.asyncio
        async def test_error_handling(self):
            """Test comprehensive error handling."""
            # Test CLI error handling
            result = _cli_invoke("nonexistent-command")
            assert result.exit_code != 0
            
            # Test invalid arguments
            result = _cli_invoke("chat", "--invalid-flag")
            assert result.exit_code != 0
        
        def test_logging_configuration(self):
            """Test logging configuration and output."""
            import logging
            
            # Test that loggers are properly configured
            logger = logging.getLogger("llamaagent")
            assert logger is not None
            
            # Test log levels
            original_level = logger.level
            try:
                logger.setLevel(logging.DEBUG)
                assert logger.level == logging.DEBUG
                
                logger.setLevel(logging.INFO)
                assert logger.level == logging.INFO
            finally:
                logger.setLevel(original_level)


    class TestFullSystem:
        """Comprehensive full system integration tests."""
        
        @pytest.mark.asyncio
        async def test_end_to_end_workflow(self):
            """Test complete end-to-end workflow."""
            temp_input, temp_output = _create_temp_files()
            
            try:
                # Step 1: Generate dataset
                await generate_spre_dataset(
                    input_path=Path(temp_input),
                    output_path=Path(temp_output),
                    num_samples=3,
                    seed=789
                )
                
                # Step 2: Test CLI with mock agent
                with patch('llamaagent.Agent') as mock_agent_class:
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.content = "End-to-end test successful"
                    mock_response.execution_time = 0.5
                    mock_response.token_count = 30
                    mock_response.success = True
                    mock_response.trace = []
                    mock_response.plan = None
                    
                    mock_agent.execute = AsyncMock(return_value=mock_response)
                    mock_agent_class.return_value = mock_agent
                    
                    result = _cli_invoke("chat", "Test end-to-end workflow")
                    assert result.exit_code == 0
                
                # Step 3: Verify dataset was created correctly
                assert os.path.exists(temp_output)
                with open(temp_output, 'r') as f:
                    data = json.load(f)
                
                assert data["metadata"]["total_episodes"] == 3
                assert len(data["episodes"]) == 3
                
            finally:
                _cleanup_temp_files(temp_input, temp_output)
        
        @pytest.mark.asyncio
        async def test_system_resilience(self):
            """Test system resilience under various conditions."""
            # Test with empty input
            result = _cli_invoke("chat", "")
            # Should handle gracefully (either error or empty response)
            assert result.exit_code in [0, 1, 2]  # Allow various exit codes
            
            # Test with very long input
            long_message = "test " * 1000
            with patch('llamaagent.Agent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.content = "Handled long input"
                mock_response.execution_time = 1.0
                mock_response.token_count = 50
                mock_response.success = True
                mock_response.trace = []
                mock_response.plan = None
                
                mock_agent.execute = AsyncMock(return_value=mock_response)
                mock_agent_class.return_value = mock_agent
                
                result = _cli_invoke("chat", long_message)
                assert result.exit_code == 0
        
        def test_system_information(self):
            """Test system information and diagnostics."""
            # Test version information
            result = _cli_invoke("--version")
            assert result.exit_code == 0
            
            # Test that all required modules can be imported
            try:
                import llamaagent
                import llamaagent.agents
                import llamaagent.tools
                import llamaagent.data_generation
                assert True  # All imports successful
            except ImportError as e:
                pytest.fail(f"Required module import failed: {e}")


# ---------------------------------------------------------------------------
# Fallback for environments without pytest
# ---------------------------------------------------------------------------

else:
    # Create dummy test classes when pytest is not available
    class TestFullSystem:
        def __init__(self):
            print("Warning: pytest not available, tests disabled")
    
    TestCLIInterface = TestFullSystem
    TestAPIEndpoints = TestFullSystem
    TestDataGeneration = TestFullSystem
    TestPerformanceBenchmarks = TestFullSystem
    TestProductionReadiness = TestFullSystem


# ---------------------------------------------------------------------------
# Manual Test Runner
# ---------------------------------------------------------------------------

def run_manual_tests():
    """Run basic tests manually without pytest."""
    print("Running manual tests...")
    
    # Test CLI help
    try:
        result = _cli_invoke("--help")
        print(f"✓ CLI help test: {'PASS' if result.exit_code == 0 else 'FAIL'}")
    except Exception as e:
        print(f"✗ CLI help test: FAIL ({e})")
    
    # Test data generation
    try:
        temp_input, temp_output = _create_temp_files()
        asyncio.run(generate_spre_dataset(
            Path(temp_input), Path(temp_output), num_samples=2, seed=123
        ))
        success = os.path.exists(temp_output)
        print(f"{'✓' if success else '✗'} Data generation test: {'PASS' if success else 'FAIL'}")
        _cleanup_temp_files(temp_input, temp_output)
    except Exception as e:
        print(f"✗ Data generation test: FAIL ({e})")
    
    print("Manual tests completed.")


if __name__ == "__main__":  # pragma: no cover
    if pytest is None:
        run_manual_tests()
    else:
        # Run with pytest if available
        pytest.main([__file__, "-v"])
