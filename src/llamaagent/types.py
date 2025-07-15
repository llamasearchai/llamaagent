"""llamaagent.types

Centralized type definitions that are shared throughout the llamaagent
code-base. Having all commonly-used primitives in a single module makes
static type-checking and import management easier, and prevents circular
import problems.

Only lightweight, data-centric definitions live here. Behavioural logic
belongs in dedicated modules.

All dataclasses are deliberately kept minimal – they only contain the
fields that are required by the current test-suite. Additional fields
can be added later without breaking existing imports.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Type alias for embedding vectors
Embedding = List[float]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Lifecycle status for an asynchronous task."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Role of a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentCapability(str, Enum):
    """Capabilities that an agent can have."""

    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    DATABASE_ACCESS = "database_access"
    API_CALLS = "api_calls"
    MULTIMODAL = "multimodal"


# ---------------------------------------------------------------------------
# Core Agent and LLM Types
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Configuration for AI agents"""

    model_config = ConfigDict(protected_namespaces=())

    agent_name: str
    storage: Optional[Dict[str, Any]] = None
    llm_provider: Optional[Any] = None
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    tools: Optional[List[Any]] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LLMMessage:
    """Message structure for LLM conversations"""

    role: str
    content: str
    images: Optional[List[str]] = None
    timestamp: Optional[datetime] = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {self.role!r}")
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")


@dataclass(frozen=True)
class LLMResponse:
    """Response structure from LLM providers"""

    content: str
    model: str
    provider: str
    message_id: Optional[str] = None
    role: str = "assistant"
    tokens_used: int = 0
    usage: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Task-related Types
# ---------------------------------------------------------------------------


@dataclass
class TaskInput:
    """Input payload that kicks-off a unit of work inside the system."""

    id: str
    task: str
    prompt: Optional[str] = None
    data: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for JSON output."""
        return {
            "task_id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "data": self.data,
            "context": self.context,
            "agent_name": self.agent_name,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """Structured result returned by an agent or pipeline."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutput:
    """Final state of a task once processing has finished."""

    task_id: str
    status: TaskStatus
    result: Optional[TaskResult] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Convenience helpers – these make the test-suite a bit nicer to read
    # by allowing direct access via ``output.success`` instead of the more
    # verbose ``output.result.success`` chain. They intentionally return
    # sensible fallbacks so that missing results never raise an AttributeError.
    # ------------------------------------------------------------------
    @property
    def success(self) -> bool:
        """Whether the underlying result was successful (``False`` if absent)."""
        return bool(self.result and self.result.success)

    @property
    def error(self) -> Optional[str]:
        """Return the error message (``None`` if the task did not fail)."""
        return self.result.error if self.result else None

    def model_dump(self) -> Dict[str, Any]:
        """Return a serialisable representation suitable for JSON output."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result.__dict__ if self.result else None,
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Chain and Message Types
# ---------------------------------------------------------------------------


@dataclass
class ChainMessage:
    """Message for reasoning chains"""

    content: str
    role: str = "user"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: Optional[datetime] = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "content",
            _strip_or_fail(self.content, err="Content cannot be empty"),
        )


@dataclass
class ChainOutput:
    """Output from reasoning chains"""

    task: str
    data: Optional[Any] = None
    prompt: Optional[str] = None
    tools: Optional[List[Any]] = None
    tool_outputs: Optional[List[Any]] = None
    previous_thoughts: Optional[List[Any]] = None
    agent_scratchpad: Optional[str] = None
    agent_name: Optional[str] = None


@dataclass
class Message:
    """Lightweight chat-style message used by some integration tests."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reasoning and Embedding Types
# ---------------------------------------------------------------------------


@dataclass
class SubTask:
    """Subtask for agent execution"""

    id: str
    description: str
    tool_call: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None


@dataclass
class Thought:
    """Thought in reasoning process"""

    content: str
    reasoning_type: str = "general"
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""

    embedding: List[float]
    model: str
    usage: Optional[Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _strip_or_fail(value: str, err: str) -> str:
    """Return *value.strip()* or raise *ValueError* if the result is empty."""
    if not value or not value.strip():
        raise ValueError(err)
    return value.strip()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "TaskStatus",
    "MessageRole",
    "AgentCapability",
    "AgentConfig",
    "LLMMessage",
    "LLMResponse",
    "TaskInput",
    "TaskOutput",
    "TaskResult",
    "ChainMessage",
    "ChainOutput",
    "Message",
    "SubTask",
    "Thought",
    "Embedding",
    "EmbeddingResult",
]
