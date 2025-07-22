#!/usr/bin/env python3
"""
OpenAI Agents Complete Integration Module

Provides comprehensive OpenAI Agents SDK integration:
- Agent lifecycle management
- Conversation handling
- Tool execution
- State management
- Response processing

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# OpenAI imports with fallback
try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionMessage

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None
    ChatCompletion = None
    ChatCompletionMessage = None

from ..llm.factory import LLMFactory
from ..types import LLMMessage

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_FOR_INPUT = "waiting_for_input"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentContext:
    """Agent execution context"""

    agent_id: str
    session_id: str
    state: AgentState
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolCall:
    """Represents a tool call with execution context"""

    id: str
    function: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    executed: bool = False
    execution_time: Optional[float] = None


@dataclass
class AgentResponse:
    """Agent response with tool calls and reasoning"""

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    requires_action: bool = False
    next_steps: List[str] = field(default_factory=list)
    context: Optional[AgentContext] = None


class OpenAIAgentsManager:
    """Complete OpenAI Agents SDK integration manager"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        enable_tools: bool = True,
        enable_reasoning: bool = True,
        max_tool_calls: int = 10,
        tool_timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.enable_tools = enable_tools
        self.enable_reasoning = enable_reasoning
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout

        # Initialize components
        self.client = None
        self.llm_factory = LLMFactory()

        # Agent state management
        self.active_agents: Dict[str, AgentContext] = {}
        self.conversation_history: Dict[str, List[LLMMessage]] = {}

        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and api_key:
            self.client = AsyncOpenAI(api_key=api_key)

    async def create_agent(
        self,
        agent_id: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentContext:
        """Create a new agent with specified configuration"""
        context = AgentContext(
            agent_id=agent_id,
            session_id=session_id,
            state=AgentState.IDLE,
            metadata=metadata or {},
        )

        # Initialize conversation history
        self.conversation_history[agent_id] = []

        # Add system prompt if provided
        if system_prompt:
            system_message = LLMMessage(role="system", content=system_prompt)
            self.conversation_history[agent_id].append(system_message)

        # Configure tools
        if tools:
            context.metadata["available_tools"] = tools

        self.active_agents[agent_id] = context

        logger.info(f"Created agent {agent_id} in session {session_id}")
        return context

    async def send_message(
        self,
        agent_id: str,
        message: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Send message to agent and get response"""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")

        context = self.active_agents[agent_id]
        context.state = AgentState.THINKING
        context.updated_at = datetime.now(timezone.utc)
        # Add user message to history
        user_message = LLMMessage(role=role, content=message)
        if metadata:
            user_message.metadata = metadata

        self.conversation_history[agent_id].append(user_message)

        try:
            # Get response
            llm_provider = self.llm_factory.create_provider(
                "openai", model_name=self.model
            )

            # Convert conversation history to provider format
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history[agent_id]
            ]

            # Get LLM response
            response = await llm_provider.generate(messages=messages, max_tokens=1000)

            # Create assistant message
            assistant_message = LLMMessage(role="assistant", content=response.content)
            self.conversation_history[agent_id].append(assistant_message)

            # Update agent state
            context.state = AgentState.COMPLETED
            context.updated_at = datetime.now(timezone.utc)
            return AgentResponse(
                content=response.content, confidence=0.8, context=context
            )

        except Exception as e:
            logger.error(f"Error processing message for agent {agent_id}: {e}")
            context.state = AgentState.ERROR
            context.updated_at = datetime.now(timezone.utc)
            return AgentResponse(
                content=f"Error processing message: {str(e)}",
                confidence=0.0,
                context=context,
            )

    async def get_agent_status(self, agent_id: str) -> Optional[AgentContext]:
        """Get current agent status"""
        return self.active_agents.get(agent_id)

    async def list_agents(self) -> List[AgentContext]:
        """List all active agents"""
        return list(self.active_agents.values())

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete agent and cleanup resources"""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]

        if agent_id in self.conversation_history:
            del self.conversation_history[agent_id]

        logger.info(f"Deleted agent {agent_id}")
        return True

    async def export_conversation(self, agent_id: str) -> List[Dict[str, Any]]:
        """Export conversation history for an agent"""
        if agent_id not in self.conversation_history:
            return []

        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": getattr(msg, "timestamp", None),
                "metadata": getattr(msg, "metadata", {}),
            }
            for msg in self.conversation_history[agent_id]
        ]

    async def clear_conversation(self, agent_id: str) -> bool:
        """Clear conversation history for an agent"""
        if agent_id in self.conversation_history:
            self.conversation_history[agent_id] = []
            logger.info(f"Cleared conversation history for agent {agent_id}")
            return True
        return False

    async def execute_tool(self, agent_id: str, tool_call: ToolCall) -> ToolCall:
        """Execute a tool call for an agent"""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")

        context = self.active_agents[agent_id]
        context.state = AgentState.EXECUTING
        context.updated_at = datetime.now(timezone.utc)
        start_time = datetime.now()

        try:
            # Mock tool execution - replace with actual tool execution logic
            if tool_call.function == "calculate":
                # Simple calculator tool
                expression = tool_call.arguments.get("expression", "")
                result = eval(expression)  # Note: In production, use safer evaluation
                tool_call.result = result
            elif tool_call.function == "read_file":
                # File reading tool
                filename = tool_call.arguments.get("filename", "")
                # Mock file content
                tool_call.result = f"Contents of {filename}: [file content here]"
            else:
                tool_call.error = f"Unknown tool function: {tool_call.function}"

            tool_call.executed = True
            tool_call.execution_time = (datetime.now() - start_time).total_seconds()

            context.state = AgentState.COMPLETED
            context.updated_at = datetime.now(timezone.utc)
        except Exception as e:
            tool_call.error = str(e)
            tool_call.executed = False
            context.state = AgentState.ERROR
            context.updated_at = datetime.now(timezone.utc)
        return tool_call

    async def process_reasoning_step(
        self, agent_id: str, step: str, confidence: float = 0.0
    ) -> bool:
        """Process a reasoning step for an agent"""
        if agent_id not in self.active_agents:
            return False

        context = self.active_agents[agent_id]

        # Add reasoning step to metadata
        if "reasoning_history" not in context.metadata:
            context.metadata["reasoning_history"] = []

        context.metadata["reasoning_history"].append(
            {
                "step": step,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        context.updated_at = datetime.now(timezone.utc)
        logger.info(f"Added reasoning step for agent {agent_id}: {step}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        total_agents = len(self.active_agents)
        states = {}

        for agent in self.active_agents.values():
            state = agent.state.value
            states[state] = states.get(state, 0) + 1

        return {
            "total_agents": total_agents,
            "agent_states": states,
            "conversations": len(self.conversation_history),
            "openai_available": OPENAI_AVAILABLE,
            "model": self.model,
            "tools_enabled": self.enable_tools,
            "reasoning_enabled": self.enable_reasoning,
        }


def create_openai_agent_manager(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    enable_tools: bool = True,
    enable_reasoning: bool = True,
    **kwargs,
) -> OpenAIAgentsManager:
    """Create OpenAI Agents Manager with default configuration"""
    return OpenAIAgentsManager(
        api_key=api_key,
        model=model,
        enable_tools=enable_tools,
        enable_reasoning=enable_reasoning,
        **kwargs,
    )


# Example usage and testing
async def main() -> None:
    """Example usage of OpenAI Agents integration"""
    # Create manager
    manager = create_openai_agent_manager(
        model="gpt-4o-mini", enable_tools=True, enable_reasoning=True
    )

    # Create agent
    agent_context = await manager.create_agent(
        agent_id="test_agent",
        system_prompt="You are a helpful assistant with access to calculation and file reading tools.",
    )

    print(f"Created agent: {agent_context.agent_id}")

    # Send message
    response = await manager.send_message(
        agent_id="test_agent", message="What time is it and what is 25 * 37?"
    )

    print(f"Response: {response.content}")
    print(f"Tool calls: {len(response.tool_calls)}")
    print(f"Confidence: {response.confidence}")
    print(f"Next steps: {response.next_steps}")

    # Export conversation
    conversation = await manager.export_conversation("test_agent")
    print(f"Conversation has {len(conversation)} messages")

    # Get stats
    stats = manager.get_stats()
    print(f"Manager stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
