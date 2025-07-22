"""
Comprehensive OpenAI API Integration for llamaagent.

This module provides complete integration with all OpenAI APIs and model types:
- Reasoning models (o-series)
- Flagship chat models
- Cost-optimized models
- Deep research models
- Realtime models
- Image generation models
- Text-to-speech models
- Transcription models
- Embeddings models
- Moderation models
- Older GPT models
- GPT base models

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, BinaryIO, Dict, List, Optional, Union

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIModelType(Enum):
    """OpenAI model categories."""

    REASONING = "reasoning"  # o-series models
    FLAGSHIP_CHAT = "flagship_chat"  # GPT-4o, etc.
    COST_OPTIMIZED = "cost_optimized"  # GPT-4o-mini, etc.
    DEEP_RESEARCH = "deep_research"  # Research models
    REALTIME = "realtime"  # Realtime models
    IMAGE_GENERATION = "image_generation"  # DALL-E models
    TEXT_TO_SPEECH = "text_to_speech"  # TTS models
    TRANSCRIPTION = "transcription"  # Whisper models
    EMBEDDINGS = "embeddings"  # Text embedding models
    MODERATION = "moderation"  # Moderation models
    LEGACY_GPT = "legacy_gpt"  # Older GPT models
    GPT_BASE = "gpt_base"  # Base GPT models


@dataclass
class OpenAIModelConfig:
    """Configuration for OpenAI models."""

    model_name: str
    model_type: OpenAIModelType
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    cost_per_1k_input: float = 0.001
    cost_per_1k_output: float = 0.002
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = True
    context_window: int = 4096
    description: str = ""


@dataclass
class OpenAIComprehensiveConfig:
    """Comprehensive configuration for OpenAI integration."""

    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    budget_limit: float = 100.0
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 60000
    enable_usage_tracking: bool = True
    enable_cost_warnings: bool = True
    default_models: Dict[OpenAIModelType, str] = field(
        default_factory=lambda: {
            OpenAIModelType.REASONING: "o3-mini",
            OpenAIModelType.FLAGSHIP_CHAT: "gpt-4o",
            OpenAIModelType.COST_OPTIMIZED: "gpt-4o-mini",
            OpenAIModelType.DEEP_RESEARCH: "gpt-4o",
            OpenAIModelType.REALTIME: "gpt-4o-realtime-preview",
            OpenAIModelType.IMAGE_GENERATION: "dall-e-3",
            OpenAIModelType.TEXT_TO_SPEECH: "tts-1",
            OpenAIModelType.TRANSCRIPTION: "whisper-1",
            OpenAIModelType.EMBEDDINGS: "text-embedding-3-large",
            OpenAIModelType.MODERATION: "text-moderation-latest",
            OpenAIModelType.LEGACY_GPT: "gpt-3.5-turbo",
            OpenAIModelType.GPT_BASE: "babbage-002",
        }
    )
    extra_headers: Dict[str, str] = field(default_factory=dict)


class OpenAIUsageTracker:
    """Track OpenAI API usage and costs."""

    def __init__(self, budget_limit: float = 100.0) -> None:
        self.budget_limit = budget_limit
        self.total_cost = 0.0
        self.usage_by_model: Dict[str, Dict[str, Any]] = {}
        self.usage_log: List[Dict[str, Any]] = []
        self.request_count = 0
        self.start_time = datetime.now(timezone.utc)

    def add_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: Optional[float] = None,
        operation: str = "completion",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add usage record."""
        if cost is None:
            cost = self._estimate_cost(model, input_tokens, output_tokens)

        self.total_cost += cost
        self.request_count += 1

        # Update model usage
        if model not in self.usage_by_model:
            self.usage_by_model[model] = {
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "request_count": 0,
            }

        model_usage = self.usage_by_model[model]
        model_usage["total_cost"] += cost
        model_usage["total_input_tokens"] += input_tokens
        model_usage["total_output_tokens"] += output_tokens
        model_usage["request_count"] += 1

        # Add to usage log
        usage_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata or {},
        }
        self.usage_log.append(usage_record)

        # Check budget
        if self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit of ${self.budget_limit:.2f} exceeded. Current cost: ${self.total_cost:.4f}"
            )

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost based on model and token usage."""
        # Updated pricing as of 2025
        pricing = {
            # Reasoning models
            "o3-mini": (0.00015, 0.0006),
            "o1-mini": (0.003, 0.012),
            "o1": (0.015, 0.06),
            # Flagship models
            "gpt-4o": (0.0025, 0.01),
            "gpt-4o-2024-11-20": (0.0025, 0.01),
            "gpt-4o-2024-08-06": (0.0025, 0.01),
            # Cost-optimized models
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
            # Legacy models
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
            # Embeddings models
            "text-embedding-3-large": (0.00013, 0.00013),
            "text-embedding-3-small": (0.00002, 0.00002),
            "text-embedding-ada-002": (0.0001, 0.0001),
            # Special models
            "dall-e-3": (0.04, 0.04),  # Per image
            "dall-e-2": (0.016, 0.016),  # Per image
            "tts-1": (0.015, 0.015),  # Per 1K characters
            "tts-1-hd": (0.03, 0.03),  # Per 1K characters
            "whisper-1": (0.006, 0.006),  # Per minute
            "text-moderation-latest": (0.0, 0.0),  # Free
        }

        input_rate, output_rate = pricing.get(model, (0.001, 0.002))
        input_cost = (input_tokens / 1000) * input_rate
        output_cost = (output_tokens / 1000) * output_rate

        return input_cost + output_cost

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        runtime = datetime.now(timezone.utc) - self.start_time

        return {
            "budget_limit": self.budget_limit,
            "total_cost": self.total_cost,
            "remaining_budget": max(0.0, self.budget_limit - self.total_cost),
            "budget_utilization_percent": (self.total_cost / self.budget_limit) * 100,
            "is_near_limit": self.total_cost > (self.budget_limit * 0.9),
            "total_requests": self.request_count,
            "runtime_seconds": runtime.total_seconds(),
            "cost_per_request": self.total_cost / max(1, self.request_count),
            "usage_by_model": self.usage_by_model,
            "recent_usage": self.usage_log[-10:] if self.usage_log else [],
        }


class BudgetExceededError(Exception):
    """Exception raised when budget limit is exceeded."""


class OpenAIComprehensiveIntegration:
    """Comprehensive OpenAI API integration supporting all model types."""

    # Model configurations
    MODEL_CONFIGS = {
        # Reasoning models (o-series)
        "o3-mini": OpenAIModelConfig(
            model_name="o3-mini",
            model_type=OpenAIModelType.REASONING,
            max_tokens=65536,
            temperature=0.1,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            supports_function_calling=True,
            context_window=200000,
            description="Efficient and cost-effective reasoning model",
        ),
        "o1-mini": OpenAIModelConfig(
            model_name="o1-mini",
            model_type=OpenAIModelType.REASONING,
            max_tokens=65536,
            temperature=0.1,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.012,
            supports_function_calling=True,
            context_window=128000,
            description="Faster reasoning model for coding and STEM tasks",
        ),
        "o1": OpenAIModelConfig(
            model_name="o1",
            model_type=OpenAIModelType.REASONING,
            max_tokens=100000,
            temperature=0.1,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.06,
            supports_function_calling=True,
            context_window=200000,
            description="Most capable reasoning model for complex multi-step tasks",
        ),
        # Flagship chat models
        "gpt-4o": OpenAIModelConfig(
            model_name="gpt-4o",
            model_type=OpenAIModelType.FLAGSHIP_CHAT,
            max_tokens=16384,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            supports_vision=True,
            supports_function_calling=True,
            context_window=128000,
            description="High-intelligence flagship model for complex tasks",
        ),
        "gpt-4o-2024-11-20": OpenAIModelConfig(
            model_name="gpt-4o-2024-11-20",
            model_type=OpenAIModelType.FLAGSHIP_CHAT,
            max_tokens=16384,
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            supports_vision=True,
            supports_function_calling=True,
            context_window=128000,
            description="Latest flagship model with enhanced capabilities",
        ),
        # Cost-optimized models
        "gpt-4o-mini": OpenAIModelConfig(
            model_name="gpt-4o-mini",
            model_type=OpenAIModelType.COST_OPTIMIZED,
            max_tokens=16384,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            supports_vision=True,
            supports_function_calling=True,
            context_window=128000,
            description="Affordable and intelligent small model",
        ),
        # Legacy GPT models
        "gpt-4": OpenAIModelConfig(
            model_name="gpt-4",
            model_type=OpenAIModelType.LEGACY_GPT,
            max_tokens=8192,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            supports_function_calling=True,
            context_window=8192,
            description="Original GPT-4 model",
        ),
        "gpt-4-turbo": OpenAIModelConfig(
            model_name="gpt-4-turbo",
            model_type=OpenAIModelType.LEGACY_GPT,
            max_tokens=4096,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_vision=True,
            supports_function_calling=True,
            context_window=128000,
            description="Turbo version of GPT-4",
        ),
        "gpt-3.5-turbo": OpenAIModelConfig(
            model_name="gpt-3.5-turbo",
            model_type=OpenAIModelType.LEGACY_GPT,
            max_tokens=4096,
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015,
            supports_function_calling=True,
            context_window=16385,
            description="Fast and cost-effective legacy model",
        ),
        # Embeddings models
        "text-embedding-3-large": OpenAIModelConfig(
            model_name="text-embedding-3-large",
            model_type=OpenAIModelType.EMBEDDINGS,
            cost_per_1k_input=0.00013,
            cost_per_1k_output=0.00013,
            supports_streaming=False,
            context_window=8192,
            description="Most capable embedding model",
        ),
        "text-embedding-3-small": OpenAIModelConfig(
            model_name="text-embedding-3-small",
            model_type=OpenAIModelType.EMBEDDINGS,
            cost_per_1k_input=0.00002,
            cost_per_1k_output=0.00002,
            supports_streaming=False,
            context_window=8192,
            description="Efficient and cost-effective embedding model",
        ),
        "text-embedding-ada-002": OpenAIModelConfig(
            model_name="text-embedding-ada-002",
            model_type=OpenAIModelType.EMBEDDINGS,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0001,
            supports_streaming=False,
            context_window=8192,
            description="Legacy embedding model",
        ),
        # Image generation models
        "dall-e-3": OpenAIModelConfig(
            model_name="dall-e-3",
            model_type=OpenAIModelType.IMAGE_GENERATION,
            cost_per_1k_input=0.04,
            cost_per_1k_output=0.04,
            supports_streaming=False,
            description="Most capable image generation model",
        ),
        "dall-e-2": OpenAIModelConfig(
            model_name="dall-e-2",
            model_type=OpenAIModelType.IMAGE_GENERATION,
            cost_per_1k_input=0.016,
            cost_per_1k_output=0.016,
            supports_streaming=False,
            description="Previous generation image model",
        ),
        # Text-to-speech models
        "tts-1": OpenAIModelConfig(
            model_name="tts-1",
            model_type=OpenAIModelType.TEXT_TO_SPEECH,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.015,
            supports_streaming=False,
            description="Standard text-to-speech model",
        ),
        "tts-1-hd": OpenAIModelConfig(
            model_name="tts-1-hd",
            model_type=OpenAIModelType.TEXT_TO_SPEECH,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.03,
            supports_streaming=False,
            description="High-definition text-to-speech model",
        ),
        # Transcription models
        "whisper-1": OpenAIModelConfig(
            model_name="whisper-1",
            model_type=OpenAIModelType.TRANSCRIPTION,
            cost_per_1k_input=0.006,
            cost_per_1k_output=0.006,
            supports_streaming=False,
            description="Speech-to-text transcription model",
        ),
        # Moderation models
        "text-moderation-latest": OpenAIModelConfig(
            model_name="text-moderation-latest",
            model_type=OpenAIModelType.MODERATION,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            supports_streaming=False,
            description="Latest content moderation model",
        ),
        "text-moderation-stable": OpenAIModelConfig(
            model_name="text-moderation-stable",
            model_type=OpenAIModelType.MODERATION,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            supports_streaming=False,
            description="Stable content moderation model",
        ),
        # Realtime models
        "gpt-4o-realtime-preview": OpenAIModelConfig(
            model_name="gpt-4o-realtime-preview",
            model_type=OpenAIModelType.REALTIME,
            max_tokens=4096,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.02,
            supports_function_calling=True,
            context_window=128000,
            description="Realtime text and audio input/output model",
        ),
        # Base models
        "babbage-002": OpenAIModelConfig(
            model_name="babbage-002",
            model_type=OpenAIModelType.GPT_BASE,
            max_tokens=16384,
            cost_per_1k_input=0.0004,
            cost_per_1k_output=0.0004,
            context_window=16384,
            description="Base model for fine-tuning",
        ),
        "davinci-002": OpenAIModelConfig(
            model_name="davinci-002",
            model_type=OpenAIModelType.GPT_BASE,
            max_tokens=16384,
            cost_per_1k_input=0.002,
            cost_per_1k_output=0.002,
            context_window=16384,
            description="More capable base model for fine-tuning",
        ),
    }

    def __init__(self, config: Optional[OpenAIComprehensiveConfig] = None) -> None:
        self.config = config or OpenAIComprehensiveConfig()
        self.usage_tracker = OpenAIUsageTracker(self.config.budget_limit)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize OpenAI client - allow for testing without API key
        self.client = self._create_client()

        self.logger.info("OpenAI Comprehensive Integration initialized")

    def _create_client(self) -> Optional[AsyncOpenAI]:
        """Create OpenAI client with configuration."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")

        # Allow initialization without API key for testing
        if not api_key:
            # Check if we're in a test environment
            import sys

            if "pytest" in sys.modules or os.getenv("TESTING") == "true":
                self.logger.warning("No OpenAI API key provided - running in test mode")
                return None
            else:
                raise ValueError("OpenAI API key is required")

        client_kwargs = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        if self.config.organization:
            client_kwargs["organization"] = self.config.organization

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        if self.config.extra_headers:
            client_kwargs["default_headers"] = self.config.extra_headers

        return AsyncOpenAI(**client_kwargs)

    def get_model_config(self, model_name: str) -> OpenAIModelConfig:
        """Get configuration for a model."""
        if model_name not in self.MODEL_CONFIGS:
            # Return default config for unknown models
            return OpenAIModelConfig(
                model_name=model_name,
                model_type=OpenAIModelType.FLAGSHIP_CHAT,
                description="Unknown model",
            )
        return self.MODEL_CONFIGS[model_name]

    def get_models_by_type(self, model_type: OpenAIModelType) -> List[str]:
        """Get all models of a specific type."""
        return [
            config.model_name
            for config in self.MODEL_CONFIGS.values()
            if config.model_type == model_type
        ]

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Chat completion with comprehensive model support."""
        model_config = self.get_model_config(model)

        # Prepare request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature or model_config.temperature,
            "max_tokens": max_tokens or model_config.max_tokens,
            **kwargs,
        }

        if tools and model_config.supports_function_calling:
            params["tools"] = tools

        if stream:
            return self._stream_chat_completion(params, model_config)
        else:
            return await self._complete_chat_completion(params, model_config)

    async def _complete_chat_completion(
        self, params: Dict[str, Any], model_config: OpenAIModelConfig
    ) -> Dict[str, Any]:
        """Complete chat completion request."""
        if not self.client:
            raise ValueError("OpenAI client not initialized - API key required")

        try:
            response = await self.client.chat.completions.create(**params)

            # Track usage
            if response.usage:
                # Flexible extraction for both attribute-style and dict-style mocks
                usage_obj = response.usage
                if isinstance(usage_obj, dict):
                    prompt_tokens = usage_obj.get("prompt_tokens", 0)
                    completion_tokens = usage_obj.get("completion_tokens", 0)
                else:
                    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0)
                    completion_tokens = getattr(usage_obj, "completion_tokens", 0)

                self.usage_tracker.add_usage(
                    model=params["model"],
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    operation="chat_completion",
                )

            # Flexible mapping for choices
            def _map_choice(ch: Any) -> Dict[str, Any]:
                if isinstance(ch, dict):
                    msg = ch.get("message", {})
                    return {
                        "index": ch.get("index"),
                        "message": {
                            "role": (
                                msg.get("role")
                                if isinstance(msg, dict)
                                else getattr(msg, "role", None)
                            ),
                            "content": (
                                msg.get("content")
                                if isinstance(msg, dict)
                                else getattr(msg, "content", None)
                            ),
                        },
                        "finish_reason": ch.get("finish_reason"),
                    }
                # attribute-style
                return {
                    "index": ch.index,
                    "message": {
                        "role": ch.message.role,
                        "content": ch.message.content,
                    },
                    "finish_reason": ch.finish_reason,
                }

            choices_payload = [_map_choice(c) for c in response.choices]

            return {
                "id": getattr(response, "id", "resp_mock"),
                "object": getattr(response, "object", "chat.completion"),
                "created": getattr(response, "created", 0),
                "model": getattr(response, "model", params["model"]),
                "choices": choices_payload,
                "usage": (
                    {
                        "prompt_tokens": (
                            response.usage.prompt_tokens
                            if not isinstance(response.usage, dict)
                            else response.usage.get("prompt_tokens")
                        ),
                        "completion_tokens": (
                            response.usage.completion_tokens
                            if not isinstance(response.usage, dict)
                            else response.usage.get("completion_tokens")
                        ),
                        "total_tokens": (
                            response.usage.total_tokens
                            if not isinstance(response.usage, dict)
                            else response.usage.get("total_tokens")
                        ),
                    }
                    if response.usage
                    else None
                ),
            }

        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise

    async def _stream_chat_completion(
        self, params: Dict[str, Any], model_config: OpenAIModelConfig
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion response."""
        params["stream"] = True

        try:
            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                yield {
                    "id": chunk.id,
                    "object": chunk.object,
                    "created": chunk.created,
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {
                                "role": choice.delta.role,
                                "content": choice.delta.content,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in chunk.choices
                    ],
                }

        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise

    async def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "url",
        style: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate images using DALL-E models."""
        model_config = self.get_model_config(model)

        params = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "response_format": response_format,
        }

        if style and model == "dall-e-3":
            params["style"] = style

        try:
            if self.client is None:
                raise ValueError(
                    "OpenAI client is not initialized. API key is required."
                )
            response = await self.client.images.generate(**params)

            # Track usage (cost per image)
            cost_per_image = model_config.cost_per_1k_input
            total_cost = cost_per_image * n

            self.usage_tracker.add_usage(
                model=model,
                cost=total_cost,
                operation="image_generation",
                metadata={"images_generated": n, "size": size, "quality": quality},
            )

            return {
                "created": response.created,
                "data": [
                    {
                        "url": image.url,
                        "b64_json": image.b64_json,
                        "revised_prompt": image.revised_prompt,
                    }
                    for image in response.data
                ],
            }

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise

    async def text_to_speech(
        self,
        input_text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> Union[bytes, Dict[str, Any]]:
        """Convert text to speech."""
        model_config = self.get_model_config(model)

        params = {
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        try:
            response = await self.client.audio.speech.create(**params)

            # Track usage (cost per 1K characters)
            char_count = len(input_text)
            cost = (char_count / 1000) * model_config.cost_per_1k_input

            self.usage_tracker.add_usage(
                model=model,
                cost=cost,
                operation="text_to_speech",
                metadata={
                    "characters": char_count,
                    "voice": voice,
                    "format": response_format,
                },
            )

            if response_format == "json":
                return {
                    "text": response.text,
                    "duration": getattr(response, "duration", None),
                }

            return response.content

        except Exception as e:
            self.logger.error(f"Text-to-speech failed: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_file: Union[BinaryIO, str, Path],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Transcribe audio to text."""
        model_config = self.get_model_config(model)

        # Handle different input types
        if isinstance(audio_file, (str, Path)):
            with open(audio_file, "rb") as f:
                file_content = f.read()
        else:
            file_content = audio_file.read()

        params = {
            "file": io.BytesIO(file_content),
            "model": model,
            "response_format": response_format,
            "temperature": temperature,
        }

        if language:
            params["language"] = language
        if prompt:
            params["prompt"] = prompt

        try:
            response = await self.client.audio.transcriptions.create(**params)

            # Estimate duration and cost (approximate)
            duration_minutes = len(file_content) / (16000 * 2 * 60)  # Rough estimate
            cost = duration_minutes * model_config.cost_per_1k_input

            self.usage_tracker.add_usage(
                model=model,
                cost=cost,
                operation="transcription",
                metadata={
                    "estimated_duration_minutes": duration_minutes,
                    "language": language,
                },
            )

            if response_format == "json":
                return {
                    "text": response.text,
                    "language": getattr(response, "language", None),
                    "duration": getattr(response, "duration", None),
                    "segments": getattr(response, "segments", None),
                }
            else:
                return {"text": response}

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            raise

    async def create_embeddings(
        self,
        input_texts: Union[str, List[str]],
        model: str = "text-embedding-3-large",
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create text embeddings."""
        self.get_model_config(model)

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        params = {
            "model": model,
            "input": input_texts,
            "encoding_format": encoding_format,
        }

        if dimensions and model in ["text-embedding-3-large", "text-embedding-3-small"]:
            params["dimensions"] = dimensions

        try:
            if self.client is None:
                raise ValueError(
                    "OpenAI client is not initialized. API key is required."
                )
            response = await self.client.embeddings.create(**params)

            # Track usage
            total_tokens = response.usage.total_tokens if response.usage else 0

            self.usage_tracker.add_usage(
                model=model,
                input_tokens=total_tokens,
                operation="embeddings",
                metadata={"text_count": len(input_texts), "dimensions": dimensions},
            )

            def _map_emb(ed: Any) -> Dict[str, Any]:
                if isinstance(ed, dict):
                    return {
                        "object": ed.get("object"),
                        "index": ed.get("index"),
                        "embedding": ed.get("embedding"),
                    }
                return {
                    "object": ed.object,
                    "index": ed.index,
                    "embedding": ed.embedding,
                }

            return {
                "object": getattr(response, "object", "list"),
                "data": [_map_emb(e) for e in response.data],
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except Exception as e:
            self.logger.error(f"Embeddings creation failed: {e}")
            raise

    async def moderate_content(
        self, input_text: Union[str, List[str]], model: str = "text-moderation-latest"
    ) -> Dict[str, Any]:
        """Moderate content for safety."""
        if isinstance(input_text, str):
            input_text = [input_text]

        try:
            if self.client is None:
                raise ValueError(
                    "OpenAI client is not initialized. API key is required."
                )
            response = await self.client.moderations.create(
                input=input_text, model=model
            )

            # Track usage (moderation is typically free)
            self.usage_tracker.add_usage(
                model=model,
                cost=0.0,
                operation="moderation",
                metadata={"text_count": len(input_text)},
            )

            return {
                "id": response.id,
                "model": response.model,
                "results": [
                    {
                        "flagged": (
                            result.get("flagged")
                            if isinstance(result, dict)
                            else result.flagged
                        ),
                        "categories": (
                            result.get("categories")
                            if isinstance(result, dict)
                            else (
                                dict(result.categories)
                                if hasattr(result, "categories")
                                and isinstance(result.categories, dict)
                                else {}
                            )
                        ),
                        "category_scores": (
                            result.get("category_scores")
                            if isinstance(result, dict)
                            else (
                                dict(result.category_scores)
                                if hasattr(result, "category_scores")
                                and isinstance(result.category_scores, dict)
                                else {}
                            )
                        ),
                    }
                    for result in response.results
                ],
            }

        except Exception as e:
            self.logger.error(f"Content moderation failed: {e}")
            raise

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = await self.client.models.list()

            models: List[Any] = []
            for model in response.data:
                model_info = {
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by,
                }

                # Add our configuration info if available
                if model.id in self.MODEL_CONFIGS:
                    config = self.MODEL_CONFIGS[model.id]
                    model_info.update(
                        {
                            "type": config.model_type.value,
                            "description": config.description,
                            "supports_vision": config.supports_vision,
                            "supports_function_calling": config.supports_function_calling,
                            "context_window": config.context_window,
                            "cost_per_1k_input": config.cost_per_1k_input,
                            "cost_per_1k_output": config.cost_per_1k_output,
                        }
                    )

                models.append(model_info)

            return models

        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            raise

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        return self.usage_tracker.get_usage_summary()

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        summary = self.usage_tracker.get_usage_summary()
        return {
            "budget_limit": summary["budget_limit"],
            "total_cost": summary["total_cost"],
            "remaining_budget": summary["remaining_budget"],
            "budget_utilization_percent": summary["budget_utilization_percent"],
            "is_near_limit": summary["is_near_limit"],
        }

    @asynccontextmanager
    async def budget_guard(self, operation: str = "api_call") -> None:
        """Context manager to guard against budget overruns."""
        if self.usage_tracker.total_cost >= self.config.budget_limit:
            raise BudgetExceededError(
                f"Budget limit reached before {operation}. "
                f"Current: ${self.usage_tracker.total_cost:.4f}, "
                f"Limit: ${self.config.budget_limit:.2f}"
            )

        try:
            yield
        except BudgetExceededError:
            self.logger.warning(f"Budget exceeded during {operation}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Test basic API connectivity
            models = await self.client.models.list()
            available_models = [m.id for m in models.data]

            # Test key model types
            health_status = {
                "api_accessible": True,
                "available_models_count": len(available_models),
                "budget_status": self.get_budget_status(),
                "usage_summary": self.get_usage_summary(),
                "model_types_available": {},
            }

            # Check each model type
            for model_type in OpenAIModelType:
                type_models = self.get_models_by_type(model_type)
                available_type_models = [
                    m for m in type_models if m in available_models
                ]

                health_status["model_types_available"][model_type.value] = {
                    "configured_models": type_models,
                    "available_models": available_type_models,
                    "availability_ratio": len(available_type_models)
                    / max(1, len(type_models)),
                }

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "api_accessible": False,
                "error": str(e),
                "budget_status": self.get_budget_status(),
                "usage_summary": self.get_usage_summary(),
            }


# Utility functions for easy integration
def create_comprehensive_openai_integration(
    api_key: Optional[str] = None, budget_limit: float = 100.0, **kwargs
) -> OpenAIComprehensiveIntegration:
    """Create comprehensive OpenAI integration with default settings."""
    config = OpenAIComprehensiveConfig(
        api_key=api_key, budget_limit=budget_limit, **kwargs
    )
    return OpenAIComprehensiveIntegration(config)


# Export all public classes and functions
__all__ = [
    "OpenAIComprehensiveIntegration",
    "OpenAIComprehensiveConfig",
    "OpenAIModelType",
    "OpenAIModelConfig",
    "OpenAIUsageTracker",
    "BudgetExceededError",
    "create_comprehensive_openai_integration",
]
