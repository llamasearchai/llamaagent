from __future__ import annotations

"""Experimental LangGraph integration.

The wrapper converts an :class:`llamaagent.agents.base.Agent` into a node that
can be composed in a LangGraph DAG.  The dependency is optional – the module
imports at runtime and raises a helpful error if the library is not available.
"""

from typing import Any, Callable

try:
    from langgraph.graph import END, Graph  # type: ignore
except ModuleNotFoundError as _exc:  # pragma: no cover – optional dependency
    raise ImportError(
        "The 'langgraph' package is required for this integration. Install via 'pip install langgraph'"
    ) from _exc

from ..agents.base import AgentConfig, AgentResponse
from ..agents.react import ReactAgent
from ..tools import ToolRegistry, get_all_tools

__all__ = ["build_react_chain"]


def _build_agent(name: str = "LG-Agent", spree: bool = False) -> ReactAgent:
    """Internal helper to construct a ReactAgent with default tools."""

    tools = ToolRegistry()
    for t in get_all_tools():
        tools.register(t)

    cfg = AgentConfig(name=name, spree_enabled=spree)
    return ReactAgent(cfg, tools=tools)


def build_react_chain(spree: bool = False) -> Callable[[str], Any]:  # noqa: D401
    """Return a callable LangGraph chain executing a single ReactAgent node.

    Example::

        chain = build_react_chain()
        result = chain.invoke("What is 42 * 17?")
    """

    agent = _build_agent(spree=spree)

    async def _node(state: dict, inputs: dict):  # noqa: D401, ANN001
        prompt = inputs["prompt"]
        response: AgentResponse = await agent.execute(prompt)
        state["response"] = response.content
        return state, END  # type: ignore[return-value]

    g = Graph()
    g.add_node("agent", _node)
    g.set_entry_point("agent")
    g.add_edge("agent", END)

    return g.compile()
