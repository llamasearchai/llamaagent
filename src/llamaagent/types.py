"""
Type definitions for LlamaAgent

This module contains all type definitions, enums, and data structures
used throughout the LlamaAgent system.

Author: LlamaAgent Development Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

# Import base types
from .agents.base import AgentConfig, AgentResponse


@dataclass
class LLMMessage:
    """Message structure for LLM providers"""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class LLMResponse:
    """Response structure from LLM providers"""
    content: str
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: int = 0
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}


class TaskStatus(Enum):
    """Status of a task execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInput:
    """Input data for a task"""
    task: str
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskOutput:
    """Output data from a task"""
    result: str
    success: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskResult:
    """Complete result of a task execution"""
    input: TaskInput
    output: TaskOutput
    status: TaskStatus
    execution_time: Optional[float] = None
    error: Optional[str] = None


__all__ = [
    "LLMMessage",
    "LLMResponse", 
    "TaskStatus",
    "TaskInput", 
    "TaskOutput",
    "TaskResult",
    "AgentConfig",
    "AgentResponse"
]
