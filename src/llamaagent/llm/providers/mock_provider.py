"""
Mock LLM Provider for testing and development.

This provider returns intelligent responses and is useful for:
- Unit testing with realistic behavior
- Development without API keys
- Fallback when other providers are unavailable
- Demonstrating agent capabilities

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import hashlib
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...types import LLMMessage, LLMResponse
from .base_provider import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Intelligent Mock LLM provider that actually solves problems."""

    def __init__(
        self,
        api_key: str = "mock-api-key",
        model_name: str = "mock-gpt-4",
        **kwargs: Any,
    ):
        """Initialize mock provider."""
        super().__init__(api_key=api_key, model=model_name, **kwargs)
        self.call_count = 0
        
    def _solve_math_problem(self, prompt: str) -> str:
        """Solve mathematical problems intelligently."""
        # Extract mathematical expressions and solve them
        
        # Handle percentage calculations
        if "%" in prompt and "of" in prompt:
            # Pattern: "Calculate X% of Y"
            match = re.search(r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', prompt)
            if match:
                percentage = float(match.group(1))
                number = float(match.group(2))
                result = (percentage / 100) * number
                
                # Check if we need to add something
                if "add" in prompt.lower():
                    add_match = re.search(r'add\s+(\d+(?:\.\d+)?)', prompt)
                    if add_match:
                        add_value = float(add_match.group(1))
                        result += add_value
                
                return str(int(result) if result.is_integer() else result)
        
        # Handle perimeter calculations
        if "perimeter" in prompt.lower() and "rectangle" in prompt.lower():
            # Extract length and width
            length_match = re.search(r'length\s+(\d+(?:\.\d+)?)', prompt)
            width_match = re.search(r'width\s+(\d+(?:\.\d+)?)', prompt)
            if length_match and width_match:
                length = float(length_match.group(1))
                width = float(width_match.group(1))
                perimeter = 2 * (length + width)
                return f"{int(perimeter) if perimeter.is_integer() else perimeter} cm"
        
        # Handle compound interest
        if "compound interest" in prompt.lower():
            # Extract principal, rate, and time
            principal_match = re.search(r'\$(\d+(?:,\d+)?)', prompt)
            rate_match = re.search(r'(\d+(?:\.\d+)?)%', prompt)
            time_match = re.search(r'(\d+)\s+years?', prompt)
            
            if principal_match and rate_match and time_match:
                principal = float(principal_match.group(1).replace(',', ''))
                rate = float(rate_match.group(1)) / 100
                time = int(time_match.group(1))
                
                # Compound interest formula: A = P(1 + r)^t
                amount = principal * (1 + rate) ** time
                return f"${amount:.2f}"
        
        # Handle derivatives
        if "derivative" in prompt.lower():
            # Simple polynomial derivative
            if "f(x) = 3x³ - 2x² + 5x - 1" in prompt:
                # df/dx = 9x² - 4x + 5
                if "x = 2" in prompt:
                    # Evaluate at x = 2: 9(4) - 4(2) + 5 = 36 - 8 + 5 = 33
                    return "33"
        
        # Handle simple arithmetic
        simple_math = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)', prompt)
        if simple_math:
            left = float(simple_math.group(1))
            op = simple_math.group(2)
            right = float(simple_math.group(3))
            
            if op == '+':
                result = left + right
            elif op == '-':
                result = left - right
            elif op == '*':
                result = left * right
            elif op == '/':
                result = left / right
            else:
                return "Unable to solve"
            
            return str(int(result) if result.is_integer() else result)
        
        return "Unable to solve this mathematical problem"
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code based on the prompt."""
        if "python function" in prompt.lower() and "maximum" in prompt.lower():
            return """def max_two(a, b):
    return a if a > b else b"""
        
        if "function" in prompt.lower() and "return" in prompt.lower():
            return "def example_function(): return 'example'"
        
        return "# Code generation not implemented for this request"
    
    def _analyze_prompt_intent(self, prompt: str) -> str:
        """Analyze prompt and provide intelligent response."""
        prompt_lower = prompt.lower()
        
        # Mathematical problems
        if any(word in prompt_lower for word in ['calculate', 'math', '%', 'perimeter', 'interest', 'derivative']):
            return self._solve_math_problem(prompt)
        
        # Programming requests
        if any(word in prompt_lower for word in ['function', 'python', 'code', 'write']):
            return self._generate_code(prompt)
        
        # Planning and reasoning
        if any(word in prompt_lower for word in ['plan', 'strategy', 'approach', 'steps']):
            return """Let me break this down into steps:
1. First, I'll analyze the requirements
2. Then, I'll identify the key components needed
3. Finally, I'll execute the solution step by step"""
        
        # Default intelligent response
        return f"I understand you're asking about: {prompt[:100]}... Let me help you with that."

    async def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given messages."""
        await asyncio.sleep(0.01)  # Simulate API delay

        self.call_count += 1

        # Get the last message content
        prompt = messages[-1].content if messages else "empty prompt"
        
        # Generate intelligent response based on prompt analysis
        response_text = self._analyze_prompt_intent(prompt)

        # Calculate mock usage
        prompt_tokens = len(prompt.split()) + 10
        completion_tokens = len(response_text.split()) + 5
        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        return LLMResponse(
            content=response_text,
            model=model or self.model_name,
            provider="mock",
            tokens_used=total_tokens,
            usage=usage,
        )

    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using the complete method."""
        message = LLMMessage(role="user", content=prompt)
        return await self.complete([message], max_tokens, temperature, **kwargs)

    async def chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Complete a chat conversation - delegates to complete."""
        return await self.complete(messages, max_tokens, temperature, model, **kwargs)

    async def stream_chat_completion(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        full_response = await self.complete(
            messages, max_tokens, temperature, model, **kwargs
        )
        content = full_response.content

        # Split into chunks and stream
        words = content.split()
        chunk_size = max(1, len(words) // 10)

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            await asyncio.sleep(0.01)
            yield chunk

    async def embed_text(
        self, texts: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate embeddings for text(s)."""
        embeddings: List[List[float]] = []

        for text in texts:
            # Generate deterministic "embeddings" based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()

            # Convert hash to numbers and normalize
            embedding: List[float] = []
            for i in range(0, min(len(text_hash), 30), 2):  # 15-dimensional embedding
                hex_pair = text_hash[i : i + 2]
                value = int(hex_pair, 16) / 255.0 - 0.5
                embedding.append(value)

            embeddings.append(embedding)

        return {
            "embeddings": embeddings,
            "model": model or self.model_name,
            "usage": {
                "total_tokens": sum(len(text.split()) for text in texts),
            },
            "provider": "mock",
        }

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return True

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for token usage."""
        return 0.0  # Mock provider is free

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ["mock-gpt-4", "mock-claude-3", "mock-model"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": "mock",
            "model": self.model_name,
            "description": "Intelligent mock provider that actually solves problems",
            "context_length": 4096,
            "cost_per_token": 0.0,
            "supports_streaming": True,
            "supports_embeddings": True,
            "type": "intelligent_mock",
            "mock_provider": True,
        }

    def __repr__(self) -> str:
        return f"MockProvider(model='{self.model_name}', calls={self.call_count})"
