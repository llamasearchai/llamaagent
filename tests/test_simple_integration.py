#!/usr/bin/env python3
"""
Simple Integration Test

Basic integration test to verify core functionality works.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio

import pytest

from llamaagent import AgentConfig, AgentRole, ReactAgent, create_provider
from llamaagent.tools import CalculatorTool, ToolRegistry


@pytest.mark.asyncio
async def test_basic_agent():
    """Test basic agent creation and execution."""
    agent = ReactAgent(
        config=AgentConfig(name="TestAgent", description="A test agent"),
        llm_provider=create_provider("mock"),
    )

    response = await agent.execute("Hello, world!")
    assert response.success is True
    assert response.content is not None
    assert agent.config.name == "TestAgent"


@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent with tool usage."""
    tools = ToolRegistry()
    tools.register(CalculatorTool())

    agent = ReactAgent(
        config=AgentConfig(name="CalculatorAgent", tools=["calculator"]),
        tools=tools,
        llm_provider=create_provider("mock"),
    )

    response = await agent.execute("Calculate 25 + 75")
    assert response.success is True


@pytest.mark.asyncio
async def test_agent_roles():
    """Test different agent roles."""
    for role in [AgentRole.RESEARCHER, AgentRole.ANALYZER, AgentRole.SPECIALIST]:
        agent = ReactAgent(
            config=AgentConfig(name=f"{role.value}Agent", role=role),
            llm_provider=create_provider("mock"),
        )

        response = await agent.execute(f"Test as {role.value}")
        assert response.success is True
        assert agent.config.role == role


@pytest.mark.asyncio
async def test_spre_enabled_agent():
    """Test SPRE-enabled agent."""
    agent = ReactAgent(
        config=AgentConfig(
            name="SPREAgent", role=AgentRole.PLANNER, spree_enabled=True
        ),
        llm_provider=create_provider("mock"),
    )

    response = await agent.execute("Plan a complex task")
    assert response.success is True


@pytest.mark.asyncio
async def test_streaming():
    """Test streaming execution."""
    agent = ReactAgent(
        config=AgentConfig(name="StreamAgent", streaming=True),
        llm_provider=create_provider("mock"),
    )

    chunks = []
    async for chunk in agent.stream_execute("Generate text"):
        chunks.append(chunk)

    assert len(chunks) > 0


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_basic_agent())
    asyncio.run(test_agent_with_tools())
    asyncio.run(test_agent_roles())
    asyncio.run(test_spre_enabled_agent())
    asyncio.run(test_streaming())
    print("All simple integration tests passed!")
