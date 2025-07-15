"""
OpenAI Tools for llamaagent.

This module provides comprehensive tools that integrate with all OpenAI APIs:
- Text generation and reasoning
- Image generation and editing
- Text-to-speech synthesis
- Audio transcription
- Text embeddings
- Content moderation
- And more

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

from ..integration.openai_comprehensive import (
    OpenAIComprehensiveIntegration,
    OpenAIModelType,
    create_comprehensive_openai_integration,
)
from .base import BaseTool

logger = logging.getLogger(__name__)


class OpenAIReasoningTool(BaseTool):
    """Tool for OpenAI reasoning models (o-series)."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_reasoning"

    @property
    def description(self) -> str:
        return "Advanced reasoning and problem-solving using OpenAI o-series models for complex multi-step tasks, mathematical problems, and logical reasoning."

    def execute(
        self,
        problem: str,
        model: str = "o3-mini",
        reasoning_type: str = "step_by_step",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute reasoning task synchronously."""
        return asyncio.run(self.aexecute(problem, model, reasoning_type, **kwargs))

    async def aexecute(
        self,
        problem: str,
        model: str = "o3-mini",
        reasoning_type: str = "step_by_step",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute reasoning task asynchronously."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an advanced reasoning AI. Think step by step and provide clear, logical explanations for your reasoning process.",
                },
                {"role": "user", "content": problem},
            ]

            response = await self.integration.chat_completion(
                messages=messages,
                model=model,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4096),
                **kwargs,
            )

            return {
                "success": True,
                "model": model,
                "response": response["choices"][0]["message"]["content"],
                "usage": response.get("usage", {}),
            }

        except Exception as e:
            self.logger.error(f"Reasoning task failed: {e}")
            return {"success": False, "error": str(e)}


class OpenAIImageGenerationTool(BaseTool):
    """Tool for OpenAI image generation using DALL-E models."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_image_generation"

    @property
    def description(self) -> str:
        return "Generate high-quality images from text descriptions using OpenAI DALL-E models."

    def execute(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute image generation synchronously."""
        return asyncio.run(self.aexecute(prompt, model, n, size, quality, **kwargs))

    async def aexecute(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute image generation asynchronously."""
        try:
            response = await self.integration.generate_image(
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                quality=quality,
                response_format="url",
                style=kwargs.get("style"),
            )

            return {
                "success": True,
                "model": model,
                "prompt": prompt,
                "images": response["data"],
                "count": n,
            }

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return {"success": False, "error": str(e)}


class OpenAITextToSpeechTool(BaseTool):
    """Tool for OpenAI text-to-speech conversion."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_text_to_speech"

    @property
    def description(self) -> str:
        return "Convert text to natural-sounding speech using OpenAI TTS models with various voice options."

    def execute(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        output_format: str = "mp3",
        speed: float = 1.0,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute text-to-speech conversion synchronously."""
        return asyncio.run(
            self.aexecute(text, model, voice, output_format, speed, save_path, **kwargs)
        )

    async def aexecute(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        output_format: str = "mp3",
        speed: float = 1.0,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute text-to-speech conversion asynchronously."""
        try:
            audio_data = await self.integration.text_to_speech(
                input_text=text,
                model=model,
                voice=voice,
                response_format=output_format,
                speed=speed,
            )

            # Save to file if path provided
            if save_path:
                with open(save_path, "wb") as f:
                    f.write(audio_data)
            else:
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    suffix=f".{output_format}", delete=False
                ) as f:
                    f.write(audio_data)
                    save_path = f.name

            return {
                "success": True,
                "model": model,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "voice": voice,
                "format": output_format,
                "audio_path": save_path,
                "audio_size_bytes": len(audio_data),
            }

        except Exception as e:
            self.logger.error(f"Text-to-speech conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class OpenAITranscriptionTool(BaseTool):
    """Tool for OpenAI audio transcription using Whisper models."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_transcription"

    @property
    def description(self) -> str:
        return "Transcribe audio files to text using OpenAI Whisper models with support for multiple languages."

    def execute(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        response_format: str = "json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute audio transcription synchronously."""
        return asyncio.run(
            self.aexecute(audio_file_path, model, language, response_format, **kwargs)
        )

    async def aexecute(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        response_format: str = "json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute audio transcription asynchronously."""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            response = await self.integration.transcribe_audio(
                audio_file=audio_file_path,
                model=model,
                language=language,
                prompt=kwargs.get("prompt"),
                response_format=response_format,
                temperature=kwargs.get("temperature", 0.0),
            )

            return {
                "success": True,
                "model": model,
                "transcription": response,
                "audio_file": audio_file_path,
            }

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class OpenAIEmbeddingsTool(BaseTool):
    """Tool for OpenAI text embeddings."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_embeddings"

    @property
    def description(self) -> str:
        return "Generate text embeddings for semantic search, clustering, and similarity analysis using OpenAI embedding models."

    def execute(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute embedding generation synchronously."""
        return asyncio.run(self.aexecute(texts, model, dimensions, **kwargs))

    async def aexecute(
        self,
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute embedding generation asynchronously."""
        try:
            response = await self.integration.create_embeddings(
                input_texts=texts,
                model=model,
                encoding_format="float",
                dimensions=dimensions,
            )

            return {
                "success": True,
                "model": model,
                "embeddings": response["data"],
                "text_count": len(texts) if isinstance(texts, list) else 1,
                "dimensions": dimensions,
            }

        except Exception as e:
            self.logger.error(f"Embeddings generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class OpenAIModerationTool(BaseTool):
    """Tool for OpenAI content moderation."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def name(self) -> str:
        return "openai_moderation"

    @property
    def description(self) -> str:
        return "Moderate content for safety and policy compliance using OpenAI moderation models."

    def execute(
        self,
        content: Union[str, List[str]],
        model: str = "text-moderation-latest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute content moderation synchronously."""
        return asyncio.run(self.aexecute(content, model, **kwargs))

    async def aexecute(
        self,
        content: Union[str, List[str]],
        model: str = "text-moderation-latest",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute content moderation asynchronously."""
        try:
            response = await self.integration.moderate_content(
                input_text=content, model=model
            )

            return {
                "success": True,
                "model": model,
                "moderation_results": response["results"],
                "content_count": len(content) if isinstance(content, list) else 1,
            }

        except Exception as e:
            self.logger.error(f"Content moderation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class OpenAIComprehensiveTool(BaseTool):
    """Comprehensive tool that provides access to all OpenAI APIs."""

    def __init__(
        self, integration: Optional[OpenAIComprehensiveIntegration] = None
    ) -> None:
        self.integration = integration or create_comprehensive_openai_integration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize sub-tools
        self.reasoning_tool = OpenAIReasoningTool(self.integration)
        self.image_tool = OpenAIImageGenerationTool(self.integration)
        self.tts_tool = OpenAITextToSpeechTool(self.integration)
        self.transcription_tool = OpenAITranscriptionTool(self.integration)
        self.embeddings_tool = OpenAIEmbeddingsTool(self.integration)
        self.moderation_tool = OpenAIModerationTool(self.integration)

    @property
    def name(self) -> str:
        return "openai_comprehensive"

    @property
    def description(self) -> str:
        return "Comprehensive access to all OpenAI APIs including reasoning, image generation, text-to-speech, transcription, embeddings, and moderation."

    def execute(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute comprehensive OpenAI operation synchronously."""
        return asyncio.run(self.aexecute(operation, **kwargs))

    async def aexecute(self, operation: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute comprehensive OpenAI operation asynchronously."""
        try:
            if operation == "reasoning":
                return await self.reasoning_tool.aexecute(**kwargs)
            elif operation == "image_generation":
                return await self.image_tool.aexecute(**kwargs)
            elif operation == "text_to_speech":
                return await self.tts_tool.aexecute(**kwargs)
            elif operation == "transcription":
                return await self.transcription_tool.aexecute(**kwargs)
            elif operation == "embeddings":
                return await self.embeddings_tool.aexecute(**kwargs)
            elif operation == "moderation":
                return await self.moderation_tool.aexecute(**kwargs)
            elif operation == "chat":
                return await self._chat_completion(**kwargs)
            elif operation == "health_check":
                return await self._health_check()
            elif operation == "budget_status":
                return self._get_budget_status()
            elif operation == "models":
                return await self._get_models(**kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}",
                    "available_operations": [
                        "reasoning",
                        "image_generation",
                        "text_to_speech",
                        "transcription",
                        "embeddings",
                        "moderation",
                        "chat",
                        "health_check",
                        "budget_status",
                        "models",
                    ],
                }

        except Exception as e:
            self.logger.error(f"Comprehensive operation '{operation}' failed: {e}")
            return {"success": False, "error": str(e), "operation": operation}

    async def _chat_completion(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute chat completion."""
        messages = kwargs.get("messages", [])
        if not messages:
            return {"success": False, "error": "No messages provided"}

        # Ensure non-streaming response
        kwargs.setdefault("stream", False)
        response = await self.integration.chat_completion(**kwargs)

        # Ensure response is a dict (not streaming)
        if not isinstance(response, dict):
            raise ValueError("Expected dict response but got streaming response")

        return {
            "success": True,
            "response": response["choices"][0]["message"]["content"],
            "full_response": response,
        }

    async def _health_check(self) -> Dict[str, Any]:
        """Execute health check."""
        health = await self.integration.health_check()
        return {"success": True, "health_status": health}

    def _get_budget_status(self) -> Dict[str, Any]:
        """Get budget status."""
        budget = self.integration.get_budget_status()
        return {"success": True, "budget_status": budget}

    async def _get_models(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get available models."""
        if model_type:
            try:
                model_type_enum = OpenAIModelType(model_type)
                models = self.integration.get_models_by_type(model_type_enum)
                return {"success": True, "models": models, "model_type": model_type}
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid model type: {model_type}",
                    "available_types": [t.value for t in OpenAIModelType],
                }
        else:
            models = await self.integration.get_available_models()
            return {"success": True, "models": models}


# Registry for easy access to tools
OPENAI_TOOLS = {
    "reasoning": OpenAIReasoningTool,
    "image_generation": OpenAIImageGenerationTool,
    "text_to_speech": OpenAITextToSpeechTool,
    "transcription": OpenAITranscriptionTool,
    "embeddings": OpenAIEmbeddingsTool,
    "moderation": OpenAIModerationTool,
    "comprehensive": OpenAIComprehensiveTool,
}


def create_openai_tool(
    tool_type: str, integration: Optional[OpenAIComprehensiveIntegration] = None
) -> BaseTool:
    """Create OpenAI tool by type."""
    if tool_type not in OPENAI_TOOLS:
        raise ValueError(
            f"Unknown tool type: {tool_type}. Available: {list(OPENAI_TOOLS.keys())}"
        )

    tool_class = OPENAI_TOOLS[tool_type]
    return tool_class(integration)


def create_all_openai_tools(
    integration: Optional[OpenAIComprehensiveIntegration] = None,
) -> List[BaseTool]:
    """Create all OpenAI tools."""
    return [tool_class(integration) for tool_class in OPENAI_TOOLS.values()]


# Export all tools and utilities
__all__ = [
    "OpenAIReasoningTool",
    "OpenAIImageGenerationTool",
    "OpenAITextToSpeechTool",
    "OpenAITranscriptionTool",
    "OpenAIEmbeddingsTool",
    "OpenAIModerationTool",
    "OpenAIComprehensiveTool",
    "OPENAI_TOOLS",
    "create_openai_tool",
    "create_all_openai_tools",
]
