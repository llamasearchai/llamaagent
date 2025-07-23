"""
OpenAI Agents Integration for LlamaAgent

This module provides seamless integration with OpenAI's Agents API,
including budget tracking and agent adapter functionality.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# Optional OpenAI imports
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional OpenAI Agents imports
try:
    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    OPENAI_AGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIAgentMode(Enum):
    """Modes for OpenAI agent integration."""

    ASSISTANT = "assistant"
    AGENT = "agent"
    CHAT = "chat"


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""

    def __init__(self, message: str, current_cost: float, budget_limit: float):
        super().__init__(message)
        self.current_cost = current_cost
        self.budget_limit = budget_limit


@dataclass
class OpenAIIntegrationConfig:
    """Configuration for OpenAI integration."""

    api_key: str
    organization: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 30.0
    budget_limit: Optional[float] = None
    enable_tracking: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetTracker:
    """Tracks budget usage for OpenAI API calls."""

    total_cost: float = 0.0
    session_cost: float = 0.0
    request_count: int = 0
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0}
    )
    daily_usage: Dict[str, float] = field(default_factory=dict)
    monthly_usage: Dict[str, float] = field(default_factory=dict)
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OpenAIAgentAdapter:
    """
    Adapter for OpenAI Agents SDK integration.

    Features:
    - Budget tracking and limits
    - Usage monitoring
    - Model switching
    - Error handling and retries
    - Session management
    """

    def __init__(self, config: OpenAIIntegrationConfig):
        self.config = config
        self.budget_tracker = BudgetTracker()
        self.session_active = False

        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE:
            self.client = openai.OpenAI(
                api_key=config.api_key,
                organization=config.organization,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        else:
            self.client = None
            logger.warning("OpenAI client not available")

        # Initialize agents if available
        self.agents: Dict[str, Any] = {}
        self.assistants: Dict[str, Any] = {}

        self.logger = logging.getLogger(__name__)

    async def create_agent(
        self,
        name: str,
        instructions: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mode: OpenAIAgentMode = OpenAIAgentMode.ASSISTANT,
    ) -> Optional[str]:
        """Create a new OpenAI agent."""
        if not OPENAI_AGENTS_AVAILABLE:
            self.logger.warning("OpenAI Agents SDK not available")
            return None

        try:
            model = model or self.config.model

            if mode == OpenAIAgentMode.ASSISTANT:
                assistant = await self._create_assistant(
                    name, instructions, model, tools
                )
                if assistant:
                    self.assistants[name] = assistant
                    return assistant.id
            elif mode == OpenAIAgentMode.AGENT:
                agent = await self._create_openai_agent(
                    name, instructions, model, tools
                )
                if agent:
                    self.agents[name] = agent
                    return agent.id

            return None
        except Exception as e:
            self.logger.error(f"Failed to create agent {name}: {e}")
            return None

    async def _create_assistant(
        self,
        name: str,
        instructions: str,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[Any]:
        """Create an OpenAI Assistant."""
        if not self.client:
            return None

        try:
            assistant = await self.client.beta.assistants.acreate(
                name=name, instructions=instructions, model=model, tools=tools or []
            )

            self.logger.info(f"Created OpenAI Assistant: {name} ({assistant.id})")
            return assistant
        except Exception as e:
            self.logger.error(f"Failed to create assistant {name}: {e}")
            return None

    async def _create_openai_agent(
        self,
        name: str,
        instructions: str,
        model: str,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[Any]:
        """Create an OpenAI Agent (if SDK available)."""
        # This would use the OpenAI Agents SDK when available
        # For now, return a placeholder
        self.logger.info(f"Would create OpenAI Agent: {name}")
        return {"id": f"agent_{name}", "name": name, "instructions": instructions}

    async def execute_task(
        self, agent_id: str, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a task using an OpenAI agent."""
        self._check_budget()

        try:
            # Find the agent
            agent = self._get_agent(agent_id)
            if not agent:
                self.logger.error(f"Agent not found: {agent_id}")
                return None

            # Execute the task
            result = await self._execute_with_agent(agent, task, context)

            # Track usage
            if result and self.config.enable_tracking:
                await self._track_usage(result)

            return result
        except BudgetExceededError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to execute task with agent {agent_id}: {e}")
            return None

    def _get_agent(self, agent_id: str) -> Optional[Any]:
        """Get agent by ID."""
        # Check assistants
        for assistant in self.assistants.values():
            if hasattr(assistant, "id") and assistant.id == agent_id:
                return assistant

        # Check agents
        for agent in self.agents.values():
            if isinstance(agent, dict) and agent.get("id") == agent_id:
                return agent
            elif hasattr(agent, "id") and agent.id == agent_id:
                return agent

        return None

    async def _execute_with_agent(
        self, agent: Any, task: str, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Execute task with a specific agent."""
        if not self.client:
            return {"error": "OpenAI client not available"}

        try:
            # For assistants
            if hasattr(agent, "id") and hasattr(self.client, "beta"):
                thread = await self.client.beta.threads.acreate()

                await self.client.beta.threads.messages.acreate(
                    thread_id=thread.id, role="user", content=task
                )

                run = await self.client.beta.threads.runs.acreate(
                    thread_id=thread.id, assistant_id=agent.id
                )

                # Wait for completion
                while run.status in ["queued", "in_progress"]:
                    await asyncio.sleep(1)
                    run = await self.client.beta.threads.runs.aretrieve(
                        thread_id=thread.id, run_id=run.id
                    )

                if run.status == "completed":
                    messages = await self.client.beta.threads.messages.alist(
                        thread_id=thread.id
                    )

                    if messages.data:
                        response = messages.data[0].content[0].text.value
                        return {
                            "response": response,
                            "status": "completed",
                            "thread_id": thread.id,
                            "run_id": run.id,
                        }

                return {"error": f"Run failed with status: {run.status}"}

            # For custom agents, use chat completion
            else:
                response = await self.client.chat.completions.acreate(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": task},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                return {
                    "response": response.choices[0].message.content,
                    "status": "completed",
                    "usage": response.usage.dict() if response.usage else {},
                }

        except Exception as e:
            self.logger.error(f"Failed to execute with agent: {e}")
            return {"error": str(e)}

    async def _track_usage(self, result: Dict[str, Any]) -> None:
        """Track API usage and costs."""
        if not result or "usage" not in result:
            return

        usage = result["usage"]

        # Update token usage
        self.budget_tracker.token_usage["prompt"] += usage.get("prompt_tokens", 0)
        self.budget_tracker.token_usage["completion"] += usage.get(
            "completion_tokens", 0
        )
        self.budget_tracker.token_usage["total"] += usage.get("total_tokens", 0)

        # Estimate cost (simplified pricing)
        prompt_cost = usage.get("prompt_tokens", 0) * 0.00003  # $0.03 per 1K tokens
        completion_cost = (
            usage.get("completion_tokens", 0) * 0.00006
        )  # $0.06 per 1K tokens
        total_cost = prompt_cost + completion_cost

        self.budget_tracker.total_cost += total_cost
        self.budget_tracker.session_cost += total_cost
        self.budget_tracker.request_count += 1

        # Track daily usage
        today = datetime.now(timezone.utc).date().isoformat()
        self.budget_tracker.daily_usage[today] = (
            self.budget_tracker.daily_usage.get(today, 0) + total_cost
        )

        # Track monthly usage
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        self.budget_tracker.monthly_usage[month] = (
            self.budget_tracker.monthly_usage.get(month, 0) + total_cost
        )

        self.logger.info(
            f"Usage tracked: {total_cost:.6f} USD, Total: {self.budget_tracker.total_cost:.6f} USD"
        )

    def _check_budget(self) -> None:
        """Check if budget limits are exceeded."""
        if not self.config.budget_limit:
            return

        if self.budget_tracker.total_cost >= self.config.budget_limit:
            raise BudgetExceededError(
                f"Budget limit exceeded: {self.budget_tracker.total_cost:.6f} >= {self.config.budget_limit:.6f}",
                self.budget_tracker.total_cost,
                self.config.budget_limit,
            )

    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all created agents."""
        agents = []

        # Add assistants
        for name, assistant in self.assistants.items():
            agents.append(
                {
                    "id": (
                        assistant.id
                        if hasattr(assistant, "id")
                        else f"assistant_{name}"
                    ),
                    "name": name,
                    "type": "assistant",
                    "model": getattr(assistant, "model", self.config.model),
                }
            )

        # Add agents
        for name, agent in self.agents.items():
            agent_id = (
                agent.get("id")
                if isinstance(agent, dict)
                else getattr(agent, "id", f"agent_{name}")
            )
            agents.append(
                {
                    "id": agent_id,
                    "name": name,
                    "type": "agent",
                    "model": self.config.model,
                }
            )

        return agents

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        try:
            # Try to delete from assistants
            for name, assistant in list(self.assistants.items()):
                if hasattr(assistant, "id") and assistant.id == agent_id:
                    if self.client and hasattr(self.client, "beta"):
                        await self.client.beta.assistants.adelete(assistant.id)
                    del self.assistants[name]
                    self.logger.info(f"Deleted assistant: {name} ({agent_id})")
                    return True

            # Try to delete from agents
            for name, agent in list(self.agents.items()):
                agent_id_check = (
                    agent.get("id")
                    if isinstance(agent, dict)
                    else getattr(agent, "id", None)
                )
                if agent_id_check == agent_id:
                    del self.agents[name]
                    self.logger.info(f"Deleted agent: {name} ({agent_id})")
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Failed to delete agent {agent_id}: {e}")
            return False

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "total_cost": self.budget_tracker.total_cost,
            "session_cost": self.budget_tracker.session_cost,
            "budget_limit": self.config.budget_limit,
            "remaining_budget": (
                (self.config.budget_limit - self.budget_tracker.total_cost)
                if self.config.budget_limit
                else None
            ),
            "request_count": self.budget_tracker.request_count,
            "token_usage": self.budget_tracker.token_usage.copy(),
            "daily_usage": self.budget_tracker.daily_usage.copy(),
            "monthly_usage": self.budget_tracker.monthly_usage.copy(),
            "last_reset": self.budget_tracker.last_reset.isoformat(),
        }

    def reset_session_budget(self) -> None:
        """Reset session budget tracking."""
        self.budget_tracker.session_cost = 0.0
        self.budget_tracker.last_reset = datetime.now(timezone.utc)
        self.logger.info("Session budget reset")

    async def start_session(self) -> None:
        """Start a new session."""
        self.session_active = True
        self.reset_session_budget()
        self.logger.info("OpenAI agent session started")

    async def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        self.session_active = False

        summary = {
            "session_cost": self.budget_tracker.session_cost,
            "session_requests": self.budget_tracker.request_count,
            "session_tokens": self.budget_tracker.token_usage.copy(),
            "total_cost": self.budget_tracker.total_cost,
        }

        self.logger.info(
            f"OpenAI agent session ended. Cost: {self.budget_tracker.session_cost:.6f} USD"
        )
        return summary


class OpenAIAgentsIntegration:
    """
    Main integration class for OpenAI Agents.

    Provides high-level interface for managing OpenAI agents and assistants
    with built-in budget tracking and error handling.
    """

    def __init__(self, config: OpenAIIntegrationConfig):
        self.config = config
        self.adapter = OpenAIAgentAdapter(config)
        self.active_sessions: Set[str] = set()

        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> bool:
        """Initialize the OpenAI integration."""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI library not available")
            return False

        try:
            await self.adapter.start_session()
            self.logger.info("OpenAI Agents integration initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI integration: {e}")
            return False

    async def create_agent(
        self,
        name: str,
        instructions: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        mode: OpenAIAgentMode = OpenAIAgentMode.ASSISTANT,
    ) -> Optional[str]:
        """Create a new OpenAI agent."""
        return await self.adapter.create_agent(name, instructions, model, tools, mode)

    async def execute_task(
        self, agent_id: str, task: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a task using an OpenAI agent."""
        return await self.adapter.execute_task(agent_id, task, context)

    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents."""
        return await self.adapter.list_agents()

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        return await self.adapter.delete_agent(agent_id)

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return self.adapter.get_budget_status()

    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the integration and return session summary."""
        summary = await self.adapter.end_session()
        self.logger.info("OpenAI Agents integration shutdown")
        return summary


# Global integration instance
_integration: Optional[OpenAIAgentsIntegration] = None


def get_openai_integration(
    config: Optional[OpenAIIntegrationConfig] = None,
) -> Optional[OpenAIAgentsIntegration]:
    """Get or create OpenAI integration instance."""
    global _integration

    if not OPENAI_AVAILABLE:
        return None

    if _integration is None and config:
        _integration = OpenAIAgentsIntegration(config)

    return _integration


# Create OpenAI integration function
def create_openai_integration(
    openai_api_key: str,
    model_name: str = "gpt-4",
    budget_limit: Optional[float] = None,
) -> OpenAIAgentsIntegration:
    """Create an OpenAI integration instance.

    Args:
        openai_api_key: OpenAI API key
        model_name: Model to use
        budget_limit: Optional budget limit

    Returns:
        OpenAI integration instance
    """
    config = OpenAIIntegrationConfig(
        api_key=openai_api_key,
        model=model_name,
        budget_limit=budget_limit,
    )
    return OpenAIAgentsIntegration(config)


# Convenience functions
async def create_openai_agent(
    name: str,
    instructions: str,
    api_key: str,
    model: str = "gpt-4",
    mode: OpenAIAgentMode = OpenAIAgentMode.ASSISTANT,
) -> Optional[str]:
    """Quick function to create an OpenAI agent."""
    config = OpenAIIntegrationConfig(api_key=api_key, model=model)
    integration = get_openai_integration(config)

    if integration:
        await integration.initialize()
        return await integration.create_agent(
            name, instructions, model=model, mode=mode
        )

    return None


async def execute_openai_task(
    agent_id: str, task: str, api_key: str, context: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Quick function to execute a task with an OpenAI agent."""
    config = OpenAIIntegrationConfig(api_key=api_key)
    integration = get_openai_integration(config)

    if integration:
        await integration.initialize()
        return await integration.execute_task(agent_id, task, context)

    return None


def create_openai_integration(
    config: OpenAIIntegrationConfig,
) -> Optional[OpenAIAgentsIntegration]:
    """Create a new OpenAI integration instance."""
    if not OPENAI_AVAILABLE:
        return None

    return OpenAIAgentsIntegration(config)
