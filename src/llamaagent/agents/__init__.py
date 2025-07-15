"""Public re-exports for the *llamaagent.agents* package.

Only a subset of concrete classes are re-exported here to keep the import
surface small while satisfying the test-suite expectations.
"""

# Core building blocks
# Advanced agent implementations
from .advanced_reasoning import (
    AdvancedReasoningAgent,
    ReasoningStrategy,
    ReasoningTrace,
    ThoughtNode,
)
from .base import AgentConfig, AgentResponse, AgentRole, BaseAgent
from .multimodal_advanced import (
    CrossModalContext,
    ModalityData,
    ModalityType,
    MultiModalAdvancedAgent,
)

# Concrete agent implementations
from .react import ReactAgent

# (A more fully-featured *Agent* class can be added later; for now we expose
# *BaseAgent* under that name so external imports don't break.)

Agent = BaseAgent  # type: ignore[assignment]

# Public API
__all__ = [
    "BaseAgent",
    "AgentConfig",
    "AgentRole",
    "AgentResponse",
    "ReactAgent",
    "Agent",
    # Advanced agents
    "AdvancedReasoningAgent",
    "ReasoningStrategy",
    "ReasoningTrace",
    "ThoughtNode",
    "MultiModalAdvancedAgent",
    "ModalityType",
    "ModalityData",
    "CrossModalContext",
]
