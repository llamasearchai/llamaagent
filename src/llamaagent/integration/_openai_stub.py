"""
OpenAI API stub for testing and development.

This module provides mock implementations of OpenAI API components for testing
without making real API calls.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockChatCompletionMessage:
    """Mock chat completion message."""

    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    function_call: Optional[Dict[str, Any]] = None


@dataclass
class MockChatCompletionChoice:
    """Mock chat completion choice."""

    index: int
    message: MockChatCompletionMessage
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


@dataclass
class MockChatCompletionUsage:
    """Mock usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MockChatCompletion:
    """Mock chat completion response."""

    id: str = "chatcmpl-test"
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "gpt-4"
    choices: List[MockChatCompletionChoice] = field(default_factory=list)
    usage: MockChatCompletionUsage = field(
        default_factory=lambda: MockChatCompletionUsage(
            prompt_tokens=10, completion_tokens=15, total_tokens=25
        )
    )
    system_fingerprint: Optional[str] = "fp_test"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": choice.message.tool_calls,
                        "function_call": choice.message.function_call,
                    },
                    "finish_reason": choice.finish_reason,
                    "logprobs": choice.logprobs,
                }
                for choice in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "system_fingerprint": self.system_fingerprint,
        }


@dataclass
class MockEmbeddingData:
    """Mock embedding data."""

    object: str = "embedding"
    embedding: List[float] = field(default_factory=lambda: [0.1] * 1536)
    index: int = 0


@dataclass
class MockEmbeddingUsage:
    """Mock embedding usage statistics."""

    prompt_tokens: int
    total_tokens: int


@dataclass
class MockEmbedding:
    """Mock embedding response."""

    object: str = "list"
    data: List[MockEmbeddingData] = field(default_factory=list)
    model: str = "text-embedding-ada-002"
    usage: MockEmbeddingUsage = field(
        default_factory=lambda: MockEmbeddingUsage(prompt_tokens=8, total_tokens=8)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "object": self.object,
            "data": [
                {
                    "object": item.object,
                    "embedding": item.embedding,
                    "index": item.index,
                }
                for item in self.data
            ],
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


@dataclass
class MockModerationCategories:
    """Mock moderation categories."""

    hate: bool = False
    hate_threatening: bool = False
    harassment: bool = False
    harassment_threatening: bool = False
    self_harm: bool = False
    self_harm_intent: bool = False
    self_harm_instructions: bool = False
    sexual: bool = False
    sexual_minors: bool = False
    violence: bool = False
    violence_graphic: bool = False


@dataclass
class MockModerationCategoryScores:
    """Mock moderation category scores."""

    hate: float = 0.001
    hate_threatening: float = 0.001
    harassment: float = 0.001
    harassment_threatening: float = 0.001
    self_harm: float = 0.001
    self_harm_intent: float = 0.001
    self_harm_instructions: float = 0.001
    sexual: float = 0.001
    sexual_minors: float = 0.001
    violence: float = 0.001
    violence_graphic: float = 0.001


@dataclass
class MockModerationResult:
    """Mock moderation result."""

    flagged: bool = False
    categories: MockModerationCategories = field(
        default_factory=MockModerationCategories
    )
    category_scores: MockModerationCategoryScores = field(
        default_factory=MockModerationCategoryScores
    )


@dataclass
class MockModeration:
    """Mock moderation response."""

    id: str = "modr-test"
    model: str = "text-moderation-latest"
    results: List[MockModerationResult] = field(
        default_factory=lambda: [MockModerationResult()]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "model": self.model,
            "results": [
                {
                    "flagged": result.flagged,
                    "categories": {
                        "hate": result.categories.hate,
                        "hate/threatening": result.categories.hate_threatening,
                        "harassment": result.categories.harassment,
                        "harassment/threatening": result.categories.harassment_threatening,
                        "self-harm": result.categories.self_harm,
                        "self-harm/intent": result.categories.self_harm_intent,
                        "self-harm/instructions": result.categories.self_harm_instructions,
                        "sexual": result.categories.sexual,
                        "sexual/minors": result.categories.sexual_minors,
                        "violence": result.categories.violence,
                        "violence/graphic": result.categories.violence_graphic,
                    },
                    "category_scores": {
                        "hate": result.category_scores.hate,
                        "hate/threatening": result.category_scores.hate_threatening,
                        "harassment": result.category_scores.harassment,
                        "harassment/threatening": result.category_scores.harassment_threatening,
                        "self-harm": result.category_scores.self_harm,
                        "self-harm/intent": result.category_scores.self_harm_intent,
                        "self-harm/instructions": result.category_scores.self_harm_instructions,
                        "sexual": result.category_scores.sexual,
                        "sexual/minors": result.category_scores.sexual_minors,
                        "violence": result.category_scores.violence,
                        "violence/graphic": result.category_scores.violence_graphic,
                    },
                }
                for result in self.results
            ],
        }


class MockChatCompletions:
    """Mock chat completions endpoint."""

    def create(self, **kwargs: Any) -> MockChatCompletion:
        """Mock synchronous chat completion creation."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gpt-4")

        # Generate content based on the last message
        content = "This is a mock response from OpenAI stub."
        if messages:
            last_msg = messages[-1].get("content", "").lower()
            if "calculate" in last_msg:
                content = "The calculation result is 42."
            elif "code" in last_msg:
                content = "```python\nprint('Hello, World!')\n```"
            elif "test successful" in last_msg:
                content = "test successful"
            elif "error" in last_msg:
                content = "I encountered an error processing your request."

        # Calculate token usage
        prompt_tokens = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
        completion_tokens = len(content) // 4

        choice = MockChatCompletionChoice(
            index=0,
            message=MockChatCompletionMessage(role="assistant", content=content),
        )

        return MockChatCompletion(
            model=model,
            choices=[choice],
            usage=MockChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class MockEmbeddings:
    """Mock embeddings endpoint."""

    def create(self, **kwargs: Any) -> MockEmbedding:
        """Mock synchronous embedding creation."""
        input_text = kwargs.get("input", "")
        model = kwargs.get("model", "text-embedding-ada-002")

        # Handle list of inputs
        if isinstance(input_text, list):
            data = []
            for i, text in enumerate(input_text):
                # Generate deterministic embeddings based on input
                embedding = [
                    hash(str(text) + str(j)) % 100 / 100.0 for j in range(1536)
                ]
                data.append(MockEmbeddingData(embedding=embedding, index=i))
            prompt_tokens = sum(len(str(text)) // 4 for text in input_text)
        else:
            # Single input
            embedding = [
                hash(str(input_text) + str(j) % 100 / 100.0 for j in range(1536))
            ]
            data = [MockEmbeddingData(embedding=embedding)]
            prompt_tokens = len(str(input_text)) // 4

        return MockEmbedding(
            model=model,
            data=data,
            usage=MockEmbeddingUsage(
                prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
            ),
        )


class MockModerations:
    """Mock moderations endpoint."""

    def create(self, **kwargs: Any) -> MockModeration:
        """Mock synchronous moderation creation."""
        input_text = kwargs.get("input", "")
        model = kwargs.get("model", "text-moderation-latest")

        # Check for obviously problematic content
        if isinstance(input_text, list):
            joined_text = " ".join(str(t) for t in input_text)
        else:
            joined_text = str(input_text)

        # Simple keyword-based flagging
        problematic_keywords = ["violence", "hate", "harassment", "self-harm", "sexual"]
        flagged = any(word in joined_text.lower() for word in problematic_keywords)

        categories = MockModerationCategories()
        category_scores = MockModerationCategoryScores()

        if flagged:
            # Set flags and scores based on content
            for keyword in problematic_keywords:
                if keyword in joined_text.lower():
                    setattr(categories, keyword.replace("-", "_"), True)
                    setattr(category_scores, keyword.replace("-", "_"), 0.8)

        return MockModeration(
            model=model,
            results=[
                MockModerationResult(
                    flagged=flagged,
                    categories=categories,
                    category_scores=category_scores,
                )
            ],
        )


class MockOpenAIClient:
    """Mock OpenAI client that prevents real API calls."""

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self._api_key = api_key  # Some code might check _api_key

        # Check for test API key patterns that should fail
        if api_key and ("test-key" in str(api_key) or "test_api" in str(api_key)):
            # We'll check this in the create methods
            self._should_fail_auth = True
        else:
            self._should_fail_auth = False

        # Initialize endpoints
        self.chat = type('Chat', (), {})()  # Simple namespace object
        self.chat.completions = MockChatCompletions()
        self.embeddings = MockEmbeddings()
        self.moderations = MockModerations()

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAsyncOpenAIClient(MockOpenAIClient):
    """Mock async OpenAI client."""

    def __init__(self, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)

        # Create async-specific endpoints
        self.chat = type('Chat', (), {})()  # Simple namespace object
        self.chat.completions = MockAsyncChatCompletions()
        self.embeddings = MockAsyncEmbeddings()
        self.moderations = MockAsyncModerations()


class MockAsyncChatCompletions:
    """Mock async chat completions endpoint."""

    def __init__(self):
        self._sync = MockChatCompletions()

    async def create(self, **kwargs: Any) -> MockChatCompletion:
        """Mock async chat completion creation."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self._sync.create(**kwargs)


class MockAsyncEmbeddings:
    """Mock async embeddings endpoint."""

    def __init__(self):
        self._sync = MockEmbeddings()

    async def create(self, **kwargs: Any) -> MockEmbedding:
        """Mock async embedding creation."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self._sync.create(**kwargs)


class MockAsyncModerations:
    """Mock async moderations endpoint."""

    def __init__(self):
        self._sync = MockModerations()

    async def create(self, **kwargs: Any) -> MockModeration:
        """Mock async moderation creation."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self._sync.create(**kwargs)


# Exception classes
class MockOpenAIError(Exception):
    """Base exception for mock OpenAI errors."""

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message)
        self.message = message
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAuthenticationError(MockOpenAIError):
    """Mock authentication error."""


class MockRateLimitError(MockOpenAIError):
    """Mock rate limit error."""


class MockAPIConnectionError(MockOpenAIError):
    """Mock API connection error."""


class MockAPIError(MockOpenAIError):
    """Mock generic API error."""


def install_openai_stub():
    """Install the OpenAI stub to prevent real API calls during testing."""
    # Create mock openai module
    mock_openai = type('MockOpenAI', (), {})()

    # Set up the main classes and functions
    mock_openai.OpenAI = MockOpenAIClient
    mock_openai.AsyncOpenAI = MockAsyncOpenAIClient

    # Set up exception classes
    mock_openai.OpenAIError = MockOpenAIError
    mock_openai.AuthenticationError = MockAuthenticationError
    mock_openai.RateLimitError = MockRateLimitError
    mock_openai.APIConnectionError = MockAPIConnectionError
    mock_openai.APIError = MockAPIError

    # Common imports
    mock_openai.ChatCompletion = MockChatCompletion
    mock_openai.Embedding = MockEmbedding
    mock_openai.Moderation = MockModeration

    # Install in sys.modules
    sys.modules["openai"] = mock_openai

    return mock_openai


def uninstall_openai_stub():
    """Remove the OpenAI stub and restore original imports."""
    if "openai" in sys.modules:
        del sys.modules["openai"]

    # Also remove any sub-modules
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("openai.")]
    for module in modules_to_remove:
        del sys.modules[module]
