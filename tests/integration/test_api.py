"""
Integration tests for API endpoints.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from fastapi.testclient import TestClient

from src.llamaagent.api.main import app


class TestAPI:
    """Test suite for API endpoints."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_agent_creation(self):
        """Test agent creation endpoint."""
        agent_data = {
            "agent_name": "TestAgent",
            "llm_provider": "mock",
            "metadata": {"spree_enabled": False},
        }
        response = self.client.post("/agents", json=agent_data)
        assert response.status_code == 200
        assert "agent_id" in response.json()

    def test_task_processing(self):
        """Test task processing endpoint."""
        # First create an agent
        agent_data = {
            "agent_name": "TestAgent",
            "llm_provider": "mock",
            "metadata": {"spree_enabled": False},
        }
        agent_response = self.client.post("/agents", json=agent_data)
        agent_id = agent_response.json()["agent_id"]

        # Then process a task
        task_data = {"task": "Test task", "agent_id": agent_id}
        response = self.client.post("/tasks", json=task_data)
        assert response.status_code == 200
        assert "result" in response.json()

    def test_error_handling(self):
        """Test API error handling."""
        # Test with invalid agent data
        invalid_data = {"invalid": "data"}
        response = self.client.post("/agents", json=invalid_data)
        assert response.status_code == 422  # Validation error
