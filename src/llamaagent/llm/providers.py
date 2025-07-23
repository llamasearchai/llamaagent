"""
Enhanced LLM Provider Framework
==============================

Swappable LLM provider system supporting:
- OpenAI GPT models via OpenAI SDK
- Local Llama models via transformers/llama-cpp
- Mock providers for testing
- Configurable provider selection

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available LLM provider types."""

    OPENAI = "openai"
    LLAMA_LOCAL = "llama_local"
    LLAMA_CPP = "llama_cpp"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider_type: ProviderType
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Provider-specific configs
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    quantization: Optional[str] = None
    context_length: int = 4096
    batch_size: int = 1


@dataclass
class LLMResponse:
    """Standardized LLM response format."""

    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    metadata: Dict[str, Any]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model_name = config.model_name
        self.provider_type = config.provider_type

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate response from prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from prompt."""
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate response from chat messages."""
        pass

    @abstractmethod
    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and properly configured."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider using official OpenAI SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            # Try importing OpenAI SDK
            try:
                from openai import AsyncOpenAI
            except ImportError:
                logger.error(
                    "OpenAI SDK not installed. Install with: pip install openai"
                )
                return

            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "No OpenAI API key provided. Provider will not be available."
                )
                return

            # Initialize client
            client_kwargs = {"api_key": api_key}
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base

            self.client = AsyncOpenAI(api_key=api_key, base_url=self.config.api_base)
            logger.info(f"Initialized OpenAI provider with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self.client = None

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        return self.client is not None

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI provider not properly initialized")

        try:
            # Use chat completions for newer models
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                frequency_penalty=kwargs.get(
                    "frequency_penalty", self.config.frequency_penalty
                ),
                presence_penalty=kwargs.get(
                    "presence_penalty", self.config.presence_penalty
                ),
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                usage={
                    "prompt_tokens": (
                        response.usage.prompt_tokens if response.usage else 0
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "total_tokens": (
                        response.usage.total_tokens if response.usage else 0
                    ),
                },
                model=response.model,
                provider="openai",
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI provider not properly initialized")

        try:
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].text:
                    yield chunk.choices[0].text

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI provider not properly initialized")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                frequency_penalty=kwargs.get(
                    "frequency_penalty", self.config.frequency_penalty
                ),
                presence_penalty=kwargs.get(
                    "presence_penalty", self.config.presence_penalty
                ),
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                model=response.model,
                provider="openai",
                metadata={"finish_reason": response.choices[0].finish_reason},
            )

        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI provider not properly initialized")

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI chat streaming error: {e}")
            raise

    async def cleanup(self):
        """Cleanup OpenAI provider."""
        if self.client:
            await self.client.close()


class LlamaLocalProvider(BaseLLMProvider):
    """Local Llama provider using Transformers."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize local Llama model."""
        try:
            # Try importing transformers
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                logger.error(
                    "Transformers not installed. Install with: pip install transformers torch"
                )
                return

            # Determine device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device

            logger.info(f"Loading Llama model: {self.model_name} on {device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
            }

            if self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            elif self.config.quantization == "4bit":
                model_kwargs["load_in_4bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            if device == "cpu":
                self.model = self.model.to(device)

            logger.info("Llama model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}")
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Check if Llama provider is available."""
        return self.model is not None and self.tokenizer is not None

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using local Llama model."""
        if not self.is_available():
            raise RuntimeError("Llama provider not properly initialized")

        try:
            import torch

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available() and self.model.device.type == "cuda":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return LLMResponse(
                content=response_text.strip(),
                usage={
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": outputs.shape[1]
                    - inputs["input_ids"].shape[1],
                    "total_tokens": outputs.shape[1],
                },
                model=self.model_name,
                provider="llama_local",
                metadata={"device": str(self.model.device)},
            )

        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response (simulated for local models)."""
        # For now, generate full response and yield in chunks
        response = await self.generate(prompt, **kwargs)
        words = response.content.split()

        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = " ".join(words[i : i + 3])
            if i + 3 < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat response using local Llama model."""
        # Convert chat messages to a single prompt
        prompt = self._format_chat_prompt(messages)
        return await self.generate(prompt, **kwargs)

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat response."""
        prompt = self._format_chat_prompt(messages)
        async for chunk in self.generate_stream(prompt, **kwargs):
            yield chunk

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a single prompt."""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "user":
                formatted_messages.append(f"Human: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")

        formatted_messages.append("Assistant:")
        return "\n".join(formatted_messages)

    async def cleanup(self):
        """Cleanup Llama provider."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class MockProvider(BaseLLMProvider):
    """Mock provider for testing and development."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.responses = [
            "This is a mock response from the LLM provider.",
            "I'm a simulated AI assistant for testing purposes.",
            "Mock response: I can help you with various tasks.",
            "This is a test response to demonstrate the system.",
            "Simulated AI response for development and testing.",
        ]
        self.response_index = 0

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response."""
        await asyncio.sleep(0.5)  # Simulate processing time

        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1

        return LLMResponse(
            content=response,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            },
            model=self.model_name,
            provider="mock",
            metadata={"mock": True},
        )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming mock response."""
        response = await self.generate(prompt, **kwargs)
        words = response.content.split()

        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate mock chat response."""
        last_message = messages[-1].get("content", "") if messages else ""
        return await self.generate(f"Chat: {last_message}", **kwargs)

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming mock chat response."""
        last_message = messages[-1].get("content", "") if messages else ""
        async for chunk in self.generate_stream(f"Chat: {last_message}", **kwargs):
            yield chunk

    async def cleanup(self):
        """No cleanup needed for mock provider."""
        pass


class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(config: LLMConfig) -> BaseLLMProvider:
        """Create provider based on configuration."""
        if config.provider_type == ProviderType.OPENAI:
            return OpenAIProvider(config)
        elif config.provider_type == ProviderType.LLAMA_LOCAL:
            return LlamaLocalProvider(config)
        elif config.provider_type == ProviderType.MOCK:
            return MockProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")

    @staticmethod
    def get_available_providers() -> List[ProviderType]:
        """Get list of available providers."""
        available = [ProviderType.MOCK]  # Mock is always available

        # Check OpenAI availability
        try:
            import openai

            if os.getenv("OPENAI_API_KEY"):
                available.append(ProviderType.OPENAI)
        except ImportError:
            pass

        # Check Transformers availability
        try:
            import torch
            import transformers

            available.append(ProviderType.LLAMA_LOCAL)
        except ImportError:
            pass

        return available


def create_provider(
    provider_type: str = None, model_name: str = None, **kwargs
) -> BaseLLMProvider:
    """Convenience function to create a provider."""
    # Default to mock if no provider specified
    if not provider_type:
        provider_type = os.getenv("LLAMAAGENT_LLM_PROVIDER", "mock")

    # Default model names
    if not model_name:
        if provider_type == "openai":
            model_name = "gpt-3.5-turbo"
        elif provider_type == "llama_local":
            model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
        else:
            model_name = "mock-model"

    # Create config
    config = LLMConfig(
        provider_type=ProviderType(provider_type), model_name=model_name, **kwargs
    )

    return ProviderFactory.create_provider(config)


# Convenience functions for backward compatibility
def get_provider():
    """Get default provider based on environment."""
    return create_provider()


def list_available_providers() -> List[str]:
    """List available provider types as strings."""
    return [p.value for p in ProviderFactory.get_available_providers()]


# Export main classes and functions
__all__ = [
    "ProviderType",
    "LLMConfig",
    "LLMResponse",
    "BaseLLMProvider",
    "OpenAIProvider",
    "LlamaLocalProvider",
    "MockProvider",
    "ProviderFactory",
    "create_provider",
    "get_provider",
    "list_available_providers",
]
