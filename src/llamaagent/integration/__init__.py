"""Integration module for llamaagent"""

from typing import List

__all__: List[str] = []

# Try to import LangGraph integration
try:
    from .langgraph import (
        LANGGRAPH_AVAILABLE,
        LangGraphAgent,
        LangGraphIntegration,
        create_langgraph_agent,
        get_integration,
        is_langgraph_available,
    )

    __all__.extend(
        [
            "LANGGRAPH_AVAILABLE",
            "LangGraphAgent",
            "create_langgraph_agent",
            "is_langgraph_available",
            "LangGraphIntegration",
            "get_integration",
        ]
    )
except (ImportError, SyntaxError) as e:
    # Log but don't fail if langgraph import fails
    import logging

    logging.debug(f"Could not import langgraph integration: {e}")

# Try to import OpenAI integration
try:
    from .openai_agents import (
        OPENAI_AGENTS_AVAILABLE,
        OpenAIAgentMode,
        OpenAIIntegrationConfig,
        create_openai_integration,
    )

    __all__.extend(
        [
            "OpenAIAgentMode",
            "OpenAIIntegrationConfig",
            "create_openai_integration",
            "OPENAI_AGENTS_AVAILABLE",
        ]
    )
except (ImportError, SyntaxError) as e:
    # Log but don't fail if openai_agents import fails
    import logging

    logging.debug(f"Could not import openai_agents integration: {e}")
