"""LangGraph Integration Module.

This module provides integration with LangGraph for building stateful,
multi-agent workflows. It includes graceful fallback behavior when
LangGraph is not installed.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from ..agents.base import AgentConfig, BaseAgent
from ..llm.providers.base_provider import BaseLLMProvider
from ..tools import ToolRegistry
from ..tools.base import BaseTool
from ..types import TaskInput, TaskOutput, TaskResult, TaskStatus

logger = logging.getLogger(__name__)

# Check for LangGraph availability
LANGGRAPH_AVAILABLE = False

try:
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode

    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    ToolNode = None
    logger.debug("LangGraph not available. Install with: pip install langgraph")


class LangGraphState(TypedDict):
    """State schema for LangGraph workflows."""

    messages: List[Dict[str, Any]]
    next_step: str
    output: str
    metadata: Dict[str, Any]


class LangGraphAgent(BaseAgent):
    """Agent implementation that uses LangGraph for workflow execution.

    This agent wraps LangGraph functionality and provides a consistent
    interface with the rest of the llamaagent framework.
    """

    def __init__(
        self,
        name: str,
        llm_provider: BaseLLMProvider,
        tools: Optional[List[BaseTool]] = None,
        description: str = "A LangGraph-based agent",
        **kwargs: Any,
    ):
        """Initialize the LangGraph agent.

        Args:
            name: Agent name
            llm_provider: LLM provider for generating responses
            tools: List of tools available to the agent
            description: Agent description
            **kwargs: Additional configuration options
        """
        # Create config for base agent
        config = AgentConfig(
            name=name, description=description, llm_provider=llm_provider
        )

        super().__init__(config=config)

        self.llm_provider = llm_provider
        self.tools = tools or []
        self.tool_registry = ToolRegistry()

        # Register tools
        for tool in self.tools:
            self.tool_registry.register(tool)

        # Build the graph if LangGraph is available
        self._graph = None
        if LANGGRAPH_AVAILABLE:
            self._graph = self._build_graph()

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> "AgentResponse":
        """Execute a task and return response.

        Args:
            task: The task to execute
            context: Optional context for the task

        Returns:
            AgentResponse with the result
        """
        # Import here to avoid circular imports
        from ..agents.base import AgentResponse

        # Create TaskInput from task string
        task_input = TaskInput(
            id=f"{self.name}-{id(task)}", task=task, context=context or {}
        )

        # Execute using execute_task
        output = await self.execute_task(task_input)

        # Convert to AgentResponse
        return AgentResponse(
            content=output.result.data.get("response", "")
            if output.result.data
            else "",
            success=output.result.success,
            metadata=output.result.metadata or {},
        )

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute a task using the LangGraph workflow.

        Args:
            task_input: The task to execute

        Returns:
            TaskOutput with the result
        """
        try:
            if not LANGGRAPH_AVAILABLE or self._graph is None:
                # Fallback to simple execution
                return await self._fallback_execute(task_input)

            # Prepare initial state
            initial_state = {
                "messages": [{"role": "user", "content": task_input.task}],
                "next_step": "agent",
                "output": "",
                "metadata": {},
            }

            # Execute the graph
            result = await self._graph.ainvoke(initial_state)

            # Extract output
            output = result.get("output", "")
            metadata = result.get("metadata", {})

            return TaskOutput(
                task_id=task_input.id,
                result=TaskResult(
                    success=True, data={"response": output}, metadata=metadata
                ),
                status=TaskStatus.COMPLETED,
            )

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return TaskOutput(
                task_id=task_input.id,
                result=TaskResult(
                    success=False, error=str(e), metadata={"error": str(e)}
                ),
                status=TaskStatus.FAILED,
            )

    async def _fallback_execute(self, task_input: TaskInput) -> TaskOutput:
        """Fallback execution when LangGraph is not available.

        This provides a simple implementation that mimics basic
        LangGraph behavior without the full workflow capabilities.
        """
        try:
            # Simple execution: just call the LLM
            from ..llm.messages import LLMMessage

            messages = [LLMMessage(role="user", content=task_input.task)]

            response = await self.llm_provider.complete(messages=messages)

            return TaskOutput(
                task_id=task_input.id,
                result=TaskResult(
                    success=True, data={"response": response.content}, metadata={}
                ),
                status=TaskStatus.COMPLETED,
            )

        except Exception as e:
            logger.error(f"Fallback execution error: {e}")
            return TaskOutput(
                task_id=task_input.id,
                result=TaskResult(
                    success=False, error=str(e), metadata={"error": str(e)}
                ),
                status=TaskStatus.FAILED,
            )

    def _build_graph(self) -> Any:
        """Build the LangGraph workflow graph."""
        if not LANGGRAPH_AVAILABLE or StateGraph is None:
            return None

        # Create state graph
        graph = StateGraph(LangGraphState)

        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("final", self._final_node)

        # Set entry point
        graph.set_entry_point("agent")

        # Add conditional edges
        graph.add_conditional_edges(
            "agent", self._should_use_tools, {"tools": "tools", "final": "final"}
        )

        # Add edge from tools back to agent
        graph.add_edge("tools", "agent")
        graph.add_edge("final", "__end__")

        # Compile the graph
        return graph.compile()

    async def _agent_node(self, state: LangGraphState) -> LangGraphState:
        """Agent node that generates responses."""
        from ..llm.messages import LLMMessage

        # Convert messages to LLMMessage format
        llm_messages = []
        for msg in state["messages"]:
            llm_messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        # Generate response
        response = await self.llm_provider.complete(messages=llm_messages)

        # Add response to messages
        state["messages"].append({"role": "assistant", "content": response.content})

        # Store output
        state["output"] = response.content

        # Check if we need tools (simplified check)
        if self._has_tool_calls(response):
            state["next_step"] = "tools"
        else:
            state["next_step"] = "final"

        return state

    async def _tool_node(self, state: LangGraphState) -> LangGraphState:
        """Tool execution node."""
        # Get the last assistant message
        last_message = state["messages"][-1]
        content = last_message.get("content", "")

        # Extract and execute tool calls (simplified)
        tool_results = []
        tool_calls = self._extract_tool_calls(content)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            if tool_name and self.tool_registry.has_tool(tool_name):
                tool = self.tool_registry.get_tool(tool_name)
                try:
                    result = tool.execute(**tool_args)
                    tool_results.append({"tool": tool_name, "result": str(result)})
                except Exception as e:
                    tool_results.append({"tool": tool_name, "error": str(e)})

        # Add tool results to messages
        if tool_results:
            state["messages"].append({"role": "tool", "content": str(tool_results)})

        return state

    def _final_node(self, state: LangGraphState) -> LangGraphState:
        """Final node to prepare output."""
        # Extract final output from messages
        for msg in reversed(state["messages"]):
            if msg.get("role") == "assistant":
                state["output"] = msg.get("content", "")
                break

        return state

    def _should_use_tools(self, state: LangGraphState) -> str:
        """Decide whether to use tools or go to final node."""
        return state.get("next_step", "final")

    def _has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls."""
        # Simplified check - in real implementation would parse response
        if hasattr(response, "tool_calls"):
            return bool(response.tool_calls)
        return False

    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from content."""
        # Simplified extraction - in real implementation would parse content
        return []


def create_langgraph_agent(
    name: str,
    llm_provider: BaseLLMProvider,
    tools: Optional[List[BaseTool]] = None,
    **kwargs: Any,
) -> LangGraphAgent:
    """Create a LangGraph agent.

    This is a helper function that creates a LangGraphAgent instance.
    If LangGraph is not available, the agent will use fallback behavior.

    Args:
        name: Agent name
        llm_provider: LLM provider for generating responses
        tools: List of tools available to the agent
        **kwargs: Additional configuration options

    Returns:
        LangGraphAgent instance
    """
    if not LANGGRAPH_AVAILABLE:
        logger.warning(
            "LangGraph is not available. Agent will use fallback behavior. Install with: pip install langgraph"
        )

    return LangGraphAgent(name=name, llm_provider=llm_provider, tools=tools, **kwargs)


def is_langgraph_available() -> bool:
    """Check if LangGraph is available.

    Returns:
        True if LangGraph is installed and can be imported, False otherwise
    """
    return LANGGRAPH_AVAILABLE


# Additional compatibility utilities for tests
class LangGraphIntegration:
    """Facade that exposes a minimal interface for LangGraph workflows.

    This class provides compatibility methods for existing tests.
    """

    def __init__(self) -> None:
        self._graphs: List[Any] = []

    def create_graph(self, *args: Any, **kwargs: Any) -> Any:
        """Return a new state-graph object (mock when LangGraph absent)."""
        if LANGGRAPH_AVAILABLE and StateGraph is not None:
            return StateGraph(LangGraphState)
        # Fallback: simple dict acting as a stub graph
        return {}

    def compile_graph(self, graph: Any) -> Any:
        """Compile the supplied graph when possible (no-op for stub)."""
        if hasattr(graph, "compile") and callable(graph.compile):
            return graph.compile()
        return graph  # Stub graph – nothing to do


# Singleton instance for compatibility
_integration_singleton: Optional[LangGraphIntegration] = None


def get_integration() -> LangGraphIntegration:
    """Return a shared LangGraphIntegration instance (lazy-initialized)."""
    global _integration_singleton
    if _integration_singleton is None:
        _integration_singleton = LangGraphIntegration()
    return _integration_singleton


# Re-export ReactAgent for compatibility
try:
    from ..agents.react import ReactAgent
except ImportError:
    ReactAgent = None  # type: ignore


# Export public API
__all__ = [
    "LANGGRAPH_AVAILABLE",
    "LangGraphAgent",
    "create_langgraph_agent",
    "is_langgraph_available",
    "LangGraphIntegration",
    "get_integration",
    "ReactAgent",
]
