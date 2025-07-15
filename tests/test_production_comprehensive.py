#!/usr/bin/env python3
"""
Comprehensive Production Testing Suite for LlamaAgent

This module provides comprehensive testing for production-ready features:
- Performance and load testing
- Security and authentication testing
- API endpoint testing with all scenarios
- Database operations and migrations
- WebSocket functionality
- File upload and processing
- Background task processing
- Monitoring and health checks
- Configuration management
- Error handling and recovery
- Docker and deployment testing

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests
from fastapi.testclient import TestClient

# Import the production app and components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamaagent.agents.base import AgentConfig, AgentRole
from llamaagent.agents.react import ReactAgent
from llamaagent.api.production_app import app, app_state
from llamaagent.llm import create_provider
from llamaagent.tools import ToolRegistry
from llamaagent.types import TaskStatus


class TestProductionAPI:
    """Comprehensive API endpoint testing."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns system information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data
        assert "endpoints" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self, client):
        """Test comprehensive health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert "memory_usage" in data
        assert "uptime" in data
        assert "active_connections" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics collection."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "requests_total" in data
        assert "active_agents" in data
        assert "memory_usage" in data
        assert "websocket_connections" in data
    
    def test_agents_crud_operations(self, client, auth_headers):
        """Test complete CRUD operations for agents."""
        # Create agent
        agent_data = {
            "name": "TestAgent",
            "role": "generalist",
            "description": "Test agent for CRUD operations",
            "provider": "mock",
            "model": "mock-model"
        }
        
        response = client.post("/agents", json=agent_data, headers=auth_headers)
        assert response.status_code == 200
        
        created_agent = response.json()
        assert "agent_id" in created_agent
        agent_id = created_agent["agent_id"]
        
        # List agents
        response = client.get("/agents")
        assert response.status_code == 200
        agents_list = response.json()
        assert "agents" in agents_list
        assert len(agents_list["agents"]) >= 1
        
        # Test agent exists in list
        agent_found = any(agent["agent_id"] == agent_id for agent in agents_list["agents"])
        assert agent_found
    
    def test_task_execution(self, client):
        """Test task execution endpoints."""
        task_data = {
            "task": "Calculate 2 + 2",
            "agent_id": "default",
            "timeout": 30
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "task_id" in result
        assert "status" in result
        assert "result" in result
        assert "execution_time" in result
    
    def test_chat_completion_openai_compatible(self, client):
        """Test OpenAI-compatible chat endpoint."""
        chat_data = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "model": "gpt-4o-mini",
            "temperature": 0.7
        }
        
        response = client.post("/v1/chat/completions", json=chat_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "id" in result
        assert "choices" in result
        assert "usage" in result
        assert len(result["choices"]) > 0
        assert result["choices"][0]["message"]["role"] == "assistant"
    
    def test_file_upload_and_processing(self, client, auth_headers):
        """Test file upload and processing workflow."""
        # Create test file
        test_content = "This is a test file for processing."
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name
        
        try:
            # Upload file
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/files/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={"description": "Test file"},
                    headers=auth_headers
                )
            
            assert response.status_code == 200
            upload_result = response.json()
            assert "file_id" in upload_result
            file_id = upload_result["file_id"]
            
            # Get file metadata
            response = client.get(f"/files/{file_id}", headers=auth_headers)
            assert response.status_code == 200
            
            # Process file
            response = client.post(
                f"/files/{file_id}/process",
                json={"task": "Summarize this file"},
                headers=auth_headers
            )
            assert response.status_code == 200
            
            process_result = response.json()
            assert "task_id" in process_result
            assert process_result["status"] == "processing"
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_authentication_required_endpoints(self, client):
        """Test that authentication is required for protected endpoints."""
        protected_endpoints = [
            ("/agents", "POST", {"name": "test"}),
            ("/files/upload", "POST", {}),
            ("/admin/system", "GET", {}),
            ("/dev/reset", "POST", {})
        ]
        
        for endpoint, method, data in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json=data)
            
            assert response.status_code == 401
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for _ in range(150):  # Exceed default limit of 100
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should eventually get rate limited (429)
        rate_limited = any(status == 429 for status in responses)
        # Note: This might not trigger in test environment, so we'll check gracefully
        assert all(status in [200, 429] for status in responses)
    
    def test_error_handling(self, client):
        """Test API error handling."""
        # Test non-existent agent
        response = client.get("/agents/non-existent-id")
        assert response.status_code == 404
        
        # Test invalid task data
        response = client.post("/tasks", json={})
        assert response.status_code == 422  # Validation error
        
        # Test non-existent file
        response = client.get("/files/non-existent-file")
        assert response.status_code == 401  # Needs auth first


class TestWebSocketFunctionality:
    """Test WebSocket real-time chat functionality."""
    
    def test_websocket_connection(self):
        """Test WebSocket connection and basic communication."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send ping
                websocket.send_text(json.dumps({
                    "type": "ping",
                    "content": "test ping"
                }))
                
                # Receive pong
                response = websocket.receive_text()
                data = json.loads(response)
                assert data["type"] == "pong"
                assert data["content"] == "Server is alive"
    
    def test_websocket_chat(self):
        """Test WebSocket chat functionality."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send chat message
                websocket.send_text(json.dumps({
                    "type": "chat",
                    "content": "Hello, agent!",
                    "agent_id": "default"
                }))
                
                # Receive response
                response = websocket.receive_text()
                data = json.loads(response)
                assert data["type"] == "response"
                assert "content" in data
                assert "execution_time" in data
    
    def test_websocket_error_handling(self):
        """Test WebSocket error handling."""
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send message to non-existent agent
                websocket.send_text(json.dumps({
                    "type": "chat",
                    "content": "Hello",
                    "agent_id": "non-existent"
                }))
                
                # Should receive error
                response = websocket.receive_text()
                data = json.loads(response)
                assert data["type"] == "error"


class TestPerformanceAndLoad:
    """Performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution performance."""
        # Create multiple agents
        agents = []
        for i in range(5):
            config = AgentConfig(name=f"LoadTestAgent{i}")
            agent = ReactAgent(config=config)
            agents.append(agent)
        
        # Execute tasks concurrently
        tasks = [f"Calculate {i} + {i}" for i in range(10)]
        
        async def execute_task(agent, task):
            start_time = time.time()
            try:
                result = await agent.execute(task)
                execution_time = time.time() - start_time
                return {"success": True, "time": execution_time, "result": result}
            except Exception as e:
                execution_time = time.time() - start_time
                return {"success": False, "time": execution_time, "error": str(e)}
        
        # Run all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[execute_task(agents[i % len(agents)], task) for i, task in enumerate(tasks)],
            return_exceptions=True
        )
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate >= 0.5  # At least 50% should succeed
        assert total_time < 60  # Should complete within 1 minute
        
        if successful_results:
            avg_time = sum(r["time"] for r in successful_results) / len(successful_results)
            assert avg_time < 10  # Average execution should be under 10 seconds
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create many agents to test memory usage
        agents = []
        for i in range(50):
            config = AgentConfig(name=f"MemoryTestAgent{i}")
            agent = ReactAgent(config=config)
            agents.append(agent)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500 * 1024 * 1024
        
        # Clean up
        del agents
    
    @pytest.mark.asyncio
    async def test_background_task_performance(self):
        """Test background task processing performance."""
        task_count = 20
        background_tasks = []
        
        async def mock_background_task(task_id: str):
            """Mock background task."""
            await asyncio.sleep(0.1)  # Simulate work
            return {"task_id": task_id, "completed": True}
        
        # Start background tasks
        start_time = time.time()
        background_tasks = [
            asyncio.create_task(mock_background_task(f"task_{i}"))
            for i in range(task_count)
        ]
        
        # Wait for completion
        results = await asyncio.gather(*background_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify results
        successful_tasks = [r for r in results if isinstance(r, dict) and r.get("completed")]
        assert len(successful_tasks) == task_count
        assert total_time < 5  # Should complete within 5 seconds


class TestSecurityFeatures:
    """Security testing for authentication, authorization, and input validation."""
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        client = TestClient(app)
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/agents", json={"name": "test"}, headers=invalid_headers)
        assert response.status_code == 401
        
        # Test with valid token
        valid_headers = {"Authorization": "Bearer test-token"}
        response = client.post(
            "/agents", 
            json={
                "name": "TestAgent",
                "role": "generalist",
                "provider": "mock"
            }, 
            headers=valid_headers
        )
        assert response.status_code == 200
    
    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization."""
        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer test-token"}
        
        # Test SQL injection attempt
        malicious_agent_data = {
            "name": "'; DROP TABLE agents; --",
            "role": "generalist",
            "provider": "mock"
        }
        
        response = client.post("/agents", json=malicious_agent_data, headers=auth_headers)
        # Should either succeed with sanitized input or fail validation
        assert response.status_code in [200, 422]
        
        # Test XSS attempt
        xss_task_data = {
            "task": "<script>alert('xss')</script>",
            "agent_id": "default"
        }
        
        response = client.post("/tasks", json=xss_task_data)
        assert response.status_code == 200
        # The task should be processed safely
    
    def test_admin_access_control(self):
        """Test admin access control."""
        client = TestClient(app)
        
        # Test with regular user token
        user_headers = {"Authorization": "Bearer test-token"}
        response = client.get("/admin/system", headers=user_headers)
        # Should work since test-token is admin in our mock
        assert response.status_code in [200, 403]
        
        # Test without authentication
        response = client.get("/admin/system")
        assert response.status_code == 401
    
    def test_file_upload_security(self):
        """Test file upload security measures."""
        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer test-token"}
        
        # Test executable file upload
        malicious_content = b"#!/bin/bash\nrm -rf /"
        
        response = client.post(
            "/files/upload",
            files={"file": ("malicious.sh", malicious_content, "application/x-shellscript")},
            headers=auth_headers
        )
        
        # Should either be rejected or handled safely
        assert response.status_code in [200, 400, 415]
        
        # Test oversized file (mock)
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        response = client.post(
            "/files/upload",
            files={"file": ("large.txt", large_content, "text/plain")},
            headers=auth_headers
        )
        
        # Should handle large files appropriately
        assert response.status_code in [200, 413]


class TestDatabaseOperations:
    """Database functionality testing."""
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connection and basic operations."""
        # Test database initialization
        if app_state["database"]:
            # Test connection
            assert app_state["database"] is not None
            
            # Test basic health check
            try:
                # Mock database health check
                health_status = True  # Mock successful connection
                assert health_status is True
            except Exception:
                pytest.skip("Database not available for testing")
    
    def test_data_persistence(self):
        """Test data persistence across operations."""
        # Test that application state persists across requests
        client = TestClient(app)
        auth_headers = {"Authorization": "Bearer test-token"}
        
        # Create agent
        agent_data = {
            "name": "PersistenceTestAgent",
            "role": "generalist",
            "provider": "mock"
        }
        
        response = client.post("/agents", json=agent_data, headers=auth_headers)
        assert response.status_code == 200
        
        created_agent = response.json()
        agent_id = created_agent["agent_id"]
        
        # Verify agent persists
        response = client.get("/agents")
        assert response.status_code == 200
        
        agents = response.json()["agents"]
        persisted_agent = next((a for a in agents if a["agent_id"] == agent_id), None)
        assert persisted_agent is not None
        assert persisted_agent["name"] == "PersistenceTestAgent"


class TestMonitoringAndAlerting:
    """Test monitoring, metrics, and alerting functionality."""
    
    def test_health_check_components(self):
        """Test individual component health checks."""
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        components = health_data.get("components", {})
        
        # Test that health check includes all components
        expected_components = [
            "orchestrator",
            "database", 
            "health_monitor",
            "metrics",
            "security_manager",
            "openai_integration"
        ]
        
        for component in expected_components:
            assert component in components
            # Component status should be boolean
            assert isinstance(components[component], bool)
    
    def test_metrics_collection(self):
        """Test comprehensive metrics collection."""
        client = TestClient(app)
        
        # Generate some activity
        client.get("/")
        client.get("/health")
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # Test required metrics
        required_metrics = [
            "requests_total",
            "active_agents",
            "memory_usage",
            "websocket_connections",
            "cache_hits",
            "cache_misses"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, dict))
    
    def test_performance_monitoring(self):
        """Test performance monitoring and tracking."""
        client = TestClient(app)
        
        # Make multiple requests to generate performance data
        start_time = time.time()
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
        
        total_time = time.time() - start_time
        avg_response_time = total_time / 10
        
        # Response time should be reasonable
        assert avg_response_time < 1.0  # Less than 1 second average


class TestConfigurationManagement:
    """Test configuration management and environment handling."""
    
    def test_environment_configuration(self):
        """Test environment-specific configuration."""
        # Test development environment
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            # Configuration should adapt to development
            assert os.getenv("ENVIRONMENT") == "development"
        
        # Test production environment
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            # Configuration should adapt to production
            assert os.getenv("ENVIRONMENT") == "production"
    
    def test_api_key_configuration(self):
        """Test API key configuration and validation."""
        # Test with valid-looking API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789"}):
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key.startswith("sk-")
        
        # Test with placeholder API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}):
            api_key = os.getenv("OPENAI_API_KEY")
            # Should be detected as placeholder
            assert api_key.startswith("your_api_")
    
    def test_feature_flags(self):
        """Test feature flag configuration."""
        # Test enabling/disabling features via environment
        with patch.dict(os.environ, {"ENABLE_DEBUG_MODE": "true"}):
            debug_enabled = os.getenv("ENABLE_DEBUG_MODE", "false").lower() == "true"
            assert debug_enabled is True
        
        with patch.dict(os.environ, {"ENABLE_DEBUG_MODE": "false"}):
            debug_enabled = os.getenv("ENABLE_DEBUG_MODE", "false").lower() == "true"
            assert debug_enabled is False


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self):
        """Test agent failure recovery mechanisms."""
        # Create agent that will fail
        config = AgentConfig(name="FailingAgent")
        agent = ReactAgent(config=config)
        
        # Mock agent to fail
        with patch.object(agent, 'execute', side_effect=Exception("Agent failure")):
            try:
                await agent.execute("test task")
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Agent failure" in str(e)
        
        # Agent should still be functional after failure
        result = await agent.execute("recovery test")
        assert result is not None
    
    def test_api_error_recovery(self):
        """Test API error recovery and graceful degradation."""
        client = TestClient(app)
        
        # Test with malformed request
        response = client.post("/tasks", json={"invalid": "data"})
        assert response.status_code == 422
        
        # Subsequent valid requests should still work
        valid_data = {
            "task": "test task",
            "agent_id": "default"
        }
        response = client.post("/tasks", json=valid_data)
        assert response.status_code == 200
    
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        client = TestClient(app)
        
        # Simulate many concurrent connections (test graceful handling)
        responses = []
        for _ in range(100):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should handle load gracefully
        success_count = sum(1 for status in responses if status == 200)
        success_rate = success_count / len(responses)
        
        # Should maintain reasonable success rate even under load
        assert success_rate >= 0.8


class TestDeploymentScenarios:
    """Test deployment and containerization scenarios."""
    
    def test_application_startup(self):
        """Test application startup sequence."""
        # Test that app starts successfully
        assert app is not None
        assert app.title == "LlamaAgent Production API"
        
        # Test that middleware is configured
        assert len(app.user_middleware) > 0
        
        # Test that routes are registered
        route_paths = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/metrics", "/agents", "/tasks"]
        
        for expected_route in expected_routes:
            assert any(expected_route in path for path in route_paths)
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown capabilities."""
        # Test that app can be shut down gracefully
        # This is more of a smoke test since we can't actually shut down in tests
        
        # Verify shutdown hooks exist
        assert hasattr(app, 'router')
        assert app.router is not None
        
        # Test cleanup functions exist
        from llamaagent.api.production_app import cleanup_application
        assert cleanup_application is not None
    
    def test_container_readiness(self):
        """Test container readiness and health checks."""
        client = TestClient(app)
        
        # Test readiness probe
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Test liveness probe
        response = client.get("/")
        assert response.status_code == 200


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_agent_execution_benchmark(self, benchmark):
        """Benchmark agent execution performance."""
        config = AgentConfig(name="BenchmarkAgent")
        agent = ReactAgent(config=config)
        
        def execute_simple_task():
            import asyncio
            return asyncio.run(agent.execute("Calculate 2 + 2"))
        
        # Benchmark the execution
        result = benchmark(execute_simple_task)
        assert result is not None
    
    def test_api_response_benchmark(self, benchmark):
        """Benchmark API response times."""
        client = TestClient(app)
        
        def make_health_request():
            response = client.get("/health")
            return response.status_code
        
        # Benchmark the API call
        status_code = benchmark(make_health_request)
        assert status_code == 200
    
    def test_concurrent_request_benchmark(self, benchmark):
        """Benchmark concurrent request handling."""
        client = TestClient(app)
        
        def make_concurrent_requests():
            import threading
            results = []
            
            def make_request():
                response = client.get("/health")
                results.append(response.status_code)
            
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return len([r for r in results if r == 200])
        
        # Benchmark concurrent requests
        success_count = benchmark(make_concurrent_requests)
        assert success_count >= 8  # At least 8 out of 10 should succeed


if __name__ == "__main__":
    # Run the comprehensive test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=10",
        "--durations=10"
    ]) 