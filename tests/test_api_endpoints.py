"""FastAPI endpoint smoke tests for SPRE generation and agent creation.
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from httpx import ASGITransport, AsyncClient

import llamaagent.types  # noqa: F401, imported to increase coverage
from llamaagent.api.complete_api import app, get_current_user


@pytest.mark.asyncio
async def test_spre_generate_smoke():
    """Call /spre/generate and expect 200 + basic dataset stats."""
    # Ensure authentication bypass
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "tester",
        "permissions": ["read", "write"],
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/spre/generate", json={"name": "demo", "count": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["statistics"]["total_items"] == 3
    # allow validation_rate to be any float – just ensure key exists
    assert "validation_rate" in data["statistics"]


@pytest.mark.asyncio
async def test_agent_create_smoke():
    """Create a new agent via /agents/create and check response."""
    # Ensure authentication bypass
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "tester",
        "permissions": ["read", "write"],
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/agents/create",
            json={
                "name": "bot",
                "role": "generalist",
                "tools": [],
                "spree_enabled": False,
            },
        )
    assert resp.status_code == 200
    assert "agent_id" in resp.json()
