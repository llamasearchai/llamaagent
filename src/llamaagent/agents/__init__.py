"""
LlamaAgent Agents Module

This module contains all agent implementations for the LlamaAgent system,
including the base ReactAgent and advanced cognitive agents.

Author: LlamaAgent Development Team
"""

from .advanced_reasoning import AdvancedReasoningAgent, ReasoningStrategy
from .base import AgentConfig, AgentResponse, AgentRole, BaseAgent
from .multimodal_advanced import MultiModalAdvancedAgent
from .multimodal_reasoning import ReasoningMode as ModalityType  # test alias

# Import ReactAgent from react module
from .react import ReactAgent

# Make the ReactAgent easily accessible for backward compatibility
ReactAgentAlias = ReactAgent

# Import task types from types module
try:
    from ..types import TaskInput, TaskOutput, TaskResult, TaskStatus
except ImportError:
    # Fallback if types not available
    TaskInput = None
    TaskOutput = None
    TaskResult = None
    TaskStatus = None

__all__ = [
    "BaseAgent",
    "ReactAgent",
    "AdvancedReasoningAgent",
    "ReasoningStrategy",
    "ReactAgentAlias",
    "AgentConfig",
    "AgentResponse",
    "AgentRole",
    "MultiModalAdvancedAgent",
    "TaskInput",
    "TaskOutput",
    "TaskResult",
    "TaskStatus",
    "ModalityType",
]
