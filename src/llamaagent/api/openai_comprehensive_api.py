"""
Comprehensive FastAPI endpoints for all OpenAI APIs and model types.

This module provides REST API endpoints for:
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

import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..integration.openai_comprehensive import (
    BudgetExceededError,
    OpenAIComprehensiveIntegration,
    OpenAIModelType,
    create_comprehensive_openai_integration,
)
from ..tools.openai_tools import OPENAI_TOOLS, create_openai_tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    temperature: Optional[float] = Field(
        default=0.7, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream response")
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Function tools"
    )


class ReasoningRequest(BaseModel):
    problem: str = Field(..., description="Problem or question to solve")
    model: str = Field(default="o3-mini", description="Reasoning model to use")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Image description prompt")
    model: str = Field(default="dall-e-3", description="Image generation model")
    size: str = Field(default="1024x1024", description="Image size")
    quality: str = Field(default="standard", description="Image quality")
    n: int = Field(default=1, description="Number of images")
    style: Optional[str] = Field(default=None, description="Image style (for DALL-E 3)")


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    model: str = Field(default="tts-1", description="TTS model")
    voice: str = Field(default="alloy", description="Voice to use")
    response_format: str = Field(default="mp3", description="Audio format")
    speed: float = Field(default=1.0, description="Speech speed")


class EmbeddingsRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default="text-embedding-3-large", description="Embedding model")
    dimensions: Optional[int] = Field(default=None, description="Embedding dimensions")


class ModerationRequest(BaseModel):
    content: Union[str, List[str]] = Field(..., description="Content to moderate")
    model: str = Field(default="text-moderation-latest", description="Moderation model")


class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default=None, description="Audio language")
    model: str = Field(default="whisper-1", description="Transcription model")
    response_format: str = Field(default="json", description="Response format")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    prompt: Optional[str] = Field(default=None, description="Transcription prompt")


class APIResponse(BaseModel):
    success: bool = Field(..., description="Whether operation succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Usage statistics"
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Disable protected namespace warning

    status: str = Field(..., description="Health status")
    api_accessible: bool = Field(..., description="API accessibility")
    budget_status: Dict[str, Any] = Field(..., description="Budget information")
    model_availability: Dict[str, Any] = Field(..., description="Model availability")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# Global integration instance
integration: Optional[OpenAIComprehensiveIntegration] = None
tools: Dict[str, Any] = {}


def get_integration() -> OpenAIComprehensiveIntegration:
    """Get or create OpenAI integration."""
    global integration
    if integration is None:
        integration = create_comprehensive_openai_integration()
    return integration


def get_tool(tool_type: str) -> Any:
    """Get or create OpenAI tool."""
    global tools
    if tool_type not in tools:
        tools[tool_type] = create_openai_tool(tool_type, get_integration())
    return tools[tool_type]


# FastAPI app
app = FastAPI(
    title="LlamaAgent Comprehensive OpenAI API",
    description="Complete REST API for all OpenAI models and services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(BudgetExceededError)
async def budget_exceeded_handler(request: Request, exc: BudgetExceededError) -> Any:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Budget exceeded: {str(exc)}",
    )


# Health and status endpoints
@app.get("/", tags=["Status"])
async def root() -> Dict[str, Any]:
    """Root endpoint with system information."""
    try:
        integration_instance = get_integration()
        health = await integration_instance.health_check()

        return {
            "message": "LlamaAgent Comprehensive OpenAI API",
            "version": "1.0.0",
            "status": "healthy" if health["api_accessible"] else "degraded",
            "available_model_types": [t.value for t in OpenAIModelType],
            "available_tools": list(OPENAI_TOOLS.keys()),
            "budget_status": integration_instance.get_budget_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {
            "message": "LlamaAgent Comprehensive OpenAI API",
            "version": "1.0.0",
            "status": "error",
            "error": str(e),
        }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check() -> HealthResponse:
    """Comprehensive health check."""
    try:
        integration_instance = get_integration()
        health = await integration_instance.health_check()

        return HealthResponse(
            status="healthy" if health["api_accessible"] else "degraded",
            api_accessible=health["api_accessible"],
            budget_status=integration_instance.get_budget_status(),
            model_availability=health.get("model_types_available", {}),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            api_accessible=False,
            budget_status={"error": str(e)},
            model_availability={},
        )


@app.get("/budget", response_model=APIResponse, tags=["Status"])
async def get_budget_status() -> APIResponse:
    """Get current budget status."""
    try:
        integration_instance = get_integration()
        budget = integration_instance.get_budget_status()

        return APIResponse(success=True, data=budget)
    except Exception as e:
        logger.error(f"Budget status error: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/models", response_model=APIResponse, tags=["Models"])
async def list_models(model_type: Optional[str] = None) -> APIResponse:
    """List available models."""
    try:
        integration_instance = get_integration()

        if model_type:
            try:
                model_type_enum = OpenAIModelType(model_type)
                models = integration_instance.get_models_by_type(model_type_enum)
                return APIResponse(
                    success=True,
                    data={
                        "model_type": model_type,
                        "models": models,
                        "count": len(models),
                    },
                )
            except ValueError:
                return APIResponse(
                    success=False,
                    error=f"Invalid model type: {model_type}",
                    data={"available_types": [t.value for t in OpenAIModelType]},
                )
        else:
            models = await integration_instance.get_available_models()
            return APIResponse(
                success=True,
                data={"models": models, "count": len(models)},
            )
    except Exception as e:
        logger.error(f"List models error: {e}")
        return APIResponse(success=False, error=str(e))


# Chat completion endpoints
@app.post("/chat/completions", response_model=APIResponse, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest) -> APIResponse:
    """Chat completions with comprehensive model support."""
    try:
        integration_instance = get_integration()

        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        if request.stream:
            # For streaming, we'll return the first chunk for now
            # In production, you'd want to use StreamingResponse
            response = await integration_instance.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                stream=False,  # Simplified for this example
            )
        else:
            response = await integration_instance.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
            )

        # Handle both Dict and AsyncGenerator return types
        if isinstance(response, dict):
            return APIResponse(success=True, data=response, usage=response.get("usage"))
        else:
            # Handle AsyncGenerator case - collect first response
            async for chunk in response:
                return APIResponse(
                    success=True,
                    data=chunk,
                    usage=chunk.get("usage") if isinstance(chunk, dict) else None,
                )
            # If no chunks, return empty response
            return APIResponse(success=True, data={}, usage=None)

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        return APIResponse(success=False, error=str(e))


# Reasoning endpoints
@app.post("/reasoning/solve", response_model=APIResponse, tags=["Reasoning"])
async def reasoning_solve(request: ReasoningRequest) -> APIResponse:
    """Solve problems using reasoning models."""
    try:
        tool = get_tool("reasoning")
        result = await tool.aexecute(
            problem=request.problem,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return APIResponse(
            success=result["success"],
            data=result,
            usage=result.get("usage"),
        )

    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        return APIResponse(success=False, error=str(e))


# Image generation endpoints
@app.post("/images/generate", response_model=APIResponse, tags=["Images"])
async def generate_images(request: ImageGenerationRequest) -> APIResponse:
    """Generate images using DALL-E models."""
    try:
        tool = get_tool("image_generation")
        result = await tool.aexecute(
            prompt=request.prompt,
            model=request.model,
            size=request.size,
            quality=request.quality,
            n=request.n,
            style=request.style,
        )
        return APIResponse(
            success=result["success"],
            data=result,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return APIResponse(success=False, error=str(e))


# Embeddings endpoints
@app.post("/embeddings", response_model=APIResponse, tags=["Embeddings"])
async def create_embeddings(request: EmbeddingsRequest) -> APIResponse:
    """Create text embeddings."""
    try:
        tool = get_tool("embeddings")
        result = await tool.aexecute(
            texts=request.texts,
            model=request.model,
            dimensions=request.dimensions,
        )
        return APIResponse(
            success=result["success"],
            data=result,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Embeddings error: {e}")
        return APIResponse(success=False, error=str(e))


# Moderation endpoints
@app.post("/moderations", response_model=APIResponse, tags=["Moderation"])
async def moderate_content(request: ModerationRequest) -> APIResponse:
    """Moderate content for safety."""
    try:
        tool = get_tool("moderation")
        result = await tool.aexecute(content=request.content, model=request.model)
        return APIResponse(
            success=result["success"],
            data=result,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Moderation error: {e}")
        return APIResponse(success=False, error=str(e))


# Text-to-speech endpoints
@app.post("/audio/speech", response_model=APIResponse, tags=["Audio"])
async def text_to_speech(request: TextToSpeechRequest) -> APIResponse:
    """Convert text to speech."""
    try:
        tool = get_tool("text_to_speech")
        result = await tool.aexecute(
            text=request.text,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            output_format=request.response_format,
        )
        if result["success"]:
            # Return file path for now - in production you might return the file directly
            return APIResponse(success=True, data=result)
        return APIResponse(
            success=result["success"],
            data=result,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/audio/speech/{file_id}", tags=["Audio"])
async def download_speech_file(file_id: str) -> FileResponse:
    """Download generated speech file."""
    # This is a simplified implementation
    # In production, you'd want proper file management
    try:
        file_path = f"/tmp/{file_id}"
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/mpeg")
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Transcription endpoints
@app.post("/audio/transcriptions", response_model=APIResponse, tags=["Audio"])
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> APIResponse:
    """Transcribe audio to text."""
    try:
        # Save uploaded file temporarily
        filename = file.filename or "audio.wav"
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{filename.split('.')[-1]}",
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            tool = get_tool("transcription")
            result = await tool.aexecute(
                audio_file_path=tmp_file_path,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
            return APIResponse(
                success=result["success"],
                data=result,
                error=result.get("error"),
            )
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return APIResponse(success=False, error=str(e))


# Batch operations endpoint
@app.post("/batch/process", response_model=APIResponse, tags=["Batch"])
async def batch_process(requests: List[Dict[str, Any]]) -> APIResponse:
    """Process multiple requests in batch."""
    try:
        results: List[Any] = []

        integration_instance = get_integration()

        for i, request_data in enumerate(requests):
            try:
                operation_type = request_data.get("type", "chat")
                if operation_type == "chat":
                    result = await integration_instance.chat_completion(
                        **request_data.get("params", {})
                    )
                elif operation_type == "reasoning":
                    tool = get_tool("reasoning")
                    result = await tool.aexecute(**request_data.get("params", {}))
                elif operation_type == "embeddings":
                    tool = get_tool("embeddings")
                    result = await tool.aexecute(**request_data.get("params", {}))
                elif operation_type == "image_generation":
                    tool = get_tool("image_generation")
                    result = await tool.aexecute(**request_data.get("params", {}))
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown operation type: {operation_type}",
                    }

                results.append(
                    {
                        "index": i,
                        "request_id": request_data.get("id", f"batch_{i}"),
                        "success": result.get("success", True),
                        "result": result,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "request_id": request_data.get("id", f"batch_{i}"),
                        "success": False,
                        "error": str(e),
                    }
                )

        # Calculate batch statistics
        successful_requests = sum(1 for r in results if r["success"])
        return APIResponse(
            success=True,
            data={
                "results": results,
                "batch_stats": {
                    "total_requests": len(requests),
                    "successful_requests": successful_requests,
                    "failed_requests": len(requests) - successful_requests,
                    "success_rate": (
                        successful_requests / len(requests) if requests else 0
                    ),
                },
            },
        )

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return APIResponse(success=False, error=str(e))


# Usage and analytics endpoints
@app.get("/usage/summary", response_model=APIResponse, tags=["Analytics"])
async def get_usage_summary() -> APIResponse:
    """Get comprehensive usage summary."""
    try:
        integration_instance = get_integration()
        usage = integration_instance.get_usage_summary()

        return APIResponse(success=True, data=usage)
    except Exception as e:
        logger.error(f"Usage summary error: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/usage/by-model", response_model=APIResponse, tags=["Analytics"])
async def get_usage_by_model() -> APIResponse:
    """Get usage breakdown by model."""
    try:
        integration_instance = get_integration()
        usage = integration_instance.get_usage_summary()

        return APIResponse(
            success=True,
            data={
                "usage_by_model": usage.get("usage_by_model", {}),
                "total_cost": usage.get("total_cost", 0),
                "total_requests": usage.get("total_requests", 0),
            },
        )

    except Exception as e:
        logger.error(f"Model usage error: {e}")
        return APIResponse(success=False, error=str(e))


# Comprehensive tool endpoint
@app.post("/tools/{tool_type}", response_model=APIResponse, tags=["Tools"])
async def use_tool(tool_type: str, operation_data: Dict[str, Any]) -> APIResponse:
    """Use any OpenAI tool with arbitrary parameters."""
    try:
        if tool_type not in OPENAI_TOOLS:
            return APIResponse(
                success=False,
                error=f"Unknown tool type: {tool_type}",
                data={"available_tools": list(OPENAI_TOOLS.keys())},
            )

        tool = get_tool(tool_type)
        if tool_type == "comprehensive":
            operation = operation_data.pop("operation", "chat")
            result = await tool.aexecute(operation=operation, **operation_data)
        else:
            result = await tool.aexecute(**operation_data)
        return APIResponse(
            success=result["success"],
            data=result,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return APIResponse(success=False, error=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "llamaagent.api.openai_comprehensive_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
