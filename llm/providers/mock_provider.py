"""Mock LLM provider for testing and development."""

import asyncio
import random
from typing import Dict, Optional

from ..models import LLMMessage, LLMResponse
from .base_provider import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Mock provider for testing and development."""

    def __init__(
        self,
        model: str = "mock-gpt-4",
        simulate_latency: bool = True,
        failure_rate: float = 0.0,
        responses: Optional[Dict[str, str]] = None,
        health_check_result: bool = True,
    ):
        self.model = model
        self.simulate_latency = simulate_latency
        self.failure_rate = failure_rate
        self.responses = responses or {}
        self.call_count = 0
        self.health_check_result = health_check_result

    async def complete(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        self.call_count += 1

        # Simulate latency
        await asyncio.sleep(random.uniform(0.1, 0.5))

        # Check for predefined responses
        for pattern, response in self.responses.items():
            if any(pattern in msg.content for msg in messages):
                return LLMResponse(
                    content=response,
                    model=self.model,
                    provider="mock",
                    tokens_used=random.randint(10, 100),
                    metadata={"simulated": True, "latency_ms": random.uniform(100, 500)},
                )

        # Simulate failure if set
        if random.random() < self.failure_rate:
            raise Exception("Mock failure simulation")

        # Generate a mock response based on the last message
        last_message = messages[-1].content if messages else "Hello?"
        response_content = f"I understand your request. How can I help you further with: {last_message}?"

        # Specialized responses for certain contexts
        if "math" in last_message.lower() or "calculate" in last_message.lower():
            response_content = "I'll help you with that calculation. The result is 42."
        elif "programming" in last_message.lower() or "python" in last_message.lower():
            # Add code block formatting for programming responses
            response_content = 'Here\'s a Python function for you:\n\n```python\ndef example_function():\n    return "Hello, World!"\n```'
        elif "plan" in last_message.lower() or "strategy" in last_message.lower():
            response_content = """Phase 1: Analysis\n- Understand requirements\n\nPhase 2: Planning\n- Create roadmap\n\nPhase 3: Execution\n- Implement steps\n\nPhase 4: Review\n- Evaluate results"""

        return LLMResponse(
            content=response_content,
            model=self.model,
            provider="mock",
            tokens_used=random.randint(10, 100),
            metadata={"simulated": True, "latency_ms": random.uniform(100, 500)},
        )

    def reset_call_count(self) -> None:
        """Reset the call counter to zero."""
        self.call_count = 0

    async def health_check(self) -> bool:
        """Mock health check - always returns True unless configured to fail."""
        return self.health_check_result
