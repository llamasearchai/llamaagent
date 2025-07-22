"""
End-to-end tests for complete system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from fastapi.testclient import TestClient

from src.llamaagent.api.main import app


class TestE2E:
    """End-to-end test suite."""

    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)

    def test_complete_user_journey(self):
        """Test complete user journey from agent creation to task completion."""
        # Step 1: Create agent
        agent_data = {
            "agent_name": "E2EAgent",
            "llm_provider": "mock",
            "metadata": {"spree_enabled": False},
        }
        agent_response = self.client.post("/agents", json=agent_data)
        assert agent_response.status_code == 200
        agent_id = agent_response.json()["agent_id"]

        # Step 2: Process multiple tasks
        tasks = [
            "Hello, introduce yourself",
            "What can you help me with?",
            "Solve this math problem: 15 + 27",
        ]

        for task in tasks:
            task_data = {"task": task, "agent_id": agent_id}
            response = self.client.post("/tasks", json=task_data)
            assert response.status_code == 200
            assert "result" in response.json()

        # Step 3: Get agent status
        status_response = self.client.get(f"/agents/{agent_id}")
        assert status_response.status_code == 200
        assert "agent_name" in status_response.json()

    def test_multi_agent_scenario(self):
        """Test scenario with multiple agents."""
        # Create multiple agents
        agents = []
        for i in range(3):
            agent_data = {
                "agent_name": f"Agent{i}",
                "llm_provider": "mock",
                "metadata": {"spree_enabled": False},
            }
            response = self.client.post("/agents", json=agent_data)
            assert response.status_code == 200
            agents.append(response.json()["agent_id"])

        # Process tasks with different agents
        for i, agent_id in enumerate(agents):
            task_data = {"task": f"Task for agent {i}", "agent_id": agent_id}
            response = self.client.post("/tasks", json=task_data)
            assert response.status_code == 200

        # Verify all agents are still active
        for agent_id in agents:
            response = self.client.get(f"/agents/{agent_id}")
            assert response.status_code == 200
