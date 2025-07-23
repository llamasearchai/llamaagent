#!/usr/bin/env python3
"""
Simple FastAPI Application for LlamaAgent

A clean, fully working API that demonstrates all core functionality:
- LLM provider integration
- Agent management
- Chat completions
- Health monitoring
- OpenAI compatibility

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import LlamaAgent components
from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent
from ..llm.factory import create_provider
from ..llm.providers.mock_provider import MockProvider
from ..tools import CalculatorTool, ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Properly typed global state
class AppState:
    def __init__(self):
        self.agents: Dict[str, ReactAgent] = {}
        self.providers: Dict[str, Any] = {}
        self.tools: Optional[ToolRegistry] = None
        self.start_time: float = time.time()


# Global state instance
app_state = AppState()


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field("gpt-4o-mini", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    agent_name: Optional[str] = Field(None, description="Specific agent to use")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage")


class AgentCreateRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(AgentRole.GENERALIST, description="Agent role")
    provider: str = Field("mock", description="LLM provider")
    model: str = Field("mock-model", description="Model name")
    description: str = Field("", description="Agent description")
    tools: List[str] = Field(default_factory=list, description="Tool names")
    api_key: Optional[str] = Field(None, description="API key for provider")


class AgentResponse(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    tools: List[str] = Field(..., description="Available tools")
    created_at: str = Field(..., description="Creation timestamp")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    agents_count: int = Field(..., description="Number of active agents")
    providers_available: List[str] = Field(..., description="Available providers")
    uptime: float = Field(..., description="Application uptime in seconds")


class SystemInfoResponse(BaseModel):
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment")
    agents_count: int = Field(..., description="Number of agents")
    tools_count: int = Field(..., description="Number of tools")
    providers: List[str] = Field(..., description="Available providers")
    uptime: float = Field(..., description="Uptime in seconds")


class AgentListResponse(BaseModel):
    agents: List[Dict[str, Any]] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")


class ToolListResponse(BaseModel):
    tools: List[Dict[str, Any]] = Field(..., description="List of tools")
    total: int = Field(..., description="Total number of tools")


class MetricsResponse(BaseModel):
    agents_count: int = Field(..., description="Number of agents")
    tools_count: int = Field(..., description="Number of tools")
    uptime: float = Field(..., description="Uptime in seconds")
    memory_usage: str = Field(..., description="Memory usage")
    active_tasks: int = Field(..., description="Active tasks")
    request_count: str = Field(..., description="Request count")


# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting LlamaAgent Simple API...")

    # Initialize tools
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    app_state.tools = tool_registry
    logger.info(
        "Tool registry initialized with %d tools", len(tool_registry.list_names())
    )

    # Create a default agent
    try:
        provider = MockProvider(model_name="mock-model")
        config = AgentConfig(
            name="default-agent",
            role=AgentRole.GENERALIST,
            description="Default AI assistant",
        )
        agent = ReactAgent(
            config=config,
            llm_provider=provider,
            tools=tool_registry,
        )
        app_state.agents["default"] = agent
        logger.info("Default agent created successfully")
    except Exception as e:
        logger.error("Failed to create default agent: %s", e)

    logger.info("LlamaAgent Simple API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down LlamaAgent Simple API...")

    # Cleanup agents
    for agent in app_state.agents.values():
        if hasattr(agent, "cleanup"):
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error("Agent cleanup error: %s", e)

    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="LlamaAgent Simple API",
    description="Clean, fully working AI agent platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with system information."""
    return {
        "message": "LlamaAgent Simple API",
        "version": "1.0.0",
        "author": "Nik Jois <nikjois@llamasearch.ai>",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "agents": "/agents",
            "system": "/system/info",
        },
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check."""
    providers_available = ["mock"]

    # Test other providers if available
    for provider_name in ["openai", "anthropic", "ollama"]:
        try:
            create_provider(provider_name, api_key="test")
            providers_available.append(provider_name)
        except Exception as e:
            # Provider not available, but this is expected in many environments
            logger.debug(f"Provider {provider_name} not available: {e}")
            continue

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        agents_count=len(app_state.agents),
        providers_available=providers_available,
        uptime=time.time() - app_state.start_time,
    )


# System information endpoint
@app.get("/system/info", response_model=SystemInfoResponse, tags=["System"])
async def system_info():
    """Get comprehensive system information."""
    return SystemInfoResponse(
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        agents_count=len(app_state.agents),
        tools_count=len(app_state.tools.list_names()) if app_state.tools else 0,
        providers=["mock", "openai", "anthropic", "ollama"],
        uptime=time.time() - app_state.start_time,
    )


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Get or create agent
        agent_name = request.agent_name or "default"
        agent = app_state.agents.get(agent_name)

        if not agent:
            # Create agent on demand
            try:
                if "gpt" in request.model:
                    provider = create_provider("openai", model_name=request.model)
                else:
                    provider = MockProvider(model_name=request.model)

                config = AgentConfig(
                    name=agent_name,
                    role=AgentRole.GENERALIST,
                    description=f"Agent for {request.model}",
                )
                agent = ReactAgent(
                    config=config,
                    llm_provider=provider,
                    tools=app_state.tools,
                )
                app_state.agents[agent_name] = agent
            except Exception as e:
                logger.error("Failed to create agent: %s", e)
                # Fallback to default agent
                agent = app_state.agents.get("default")
                if not agent:
                    raise HTTPException(status_code=500, detail="No agents available")

        # Extract task from messages
        if request.messages:
            task = request.messages[-1].content
        else:
            task = "Hello"

        # Execute task
        response = await agent.execute(task)

        # Format response
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": "stop",
        }

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage={
                "prompt_tokens": len(task) // 4,
                "completion_tokens": response.tokens_used,
                "total_tokens": (len(task) // 4) + response.tokens_used,
            },
        )

    except Exception as e:
        logger.error("Chat completion error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Agent management endpoints
@app.post("/agents", response_model=AgentResponse, tags=["Agents"])
async def create_agent(request: AgentCreateRequest):
    """Create a new agent."""
    try:
        # Create provider
        if request.provider == "mock":
            provider = MockProvider(model_name=request.model)
        else:
            provider = create_provider(
                request.provider, model_name=request.model, api_key=request.api_key
            )

        # Create agent config
        config = AgentConfig(
            name=request.name,
            role=request.role,
            description=request.description,
        )

        # Create agent
        agent = ReactAgent(
            config=config,
            llm_provider=provider,
            tools=app_state.tools,
        )

        # Generate unique ID and store agent
        agent_id = str(uuid.uuid4())
        app_state.agents[agent_id] = agent

        # Return response
        return AgentResponse(
            agent_id=agent_id,
            name=request.name,
            role=request.role.value,
            provider=request.provider,
            model=request.model,
            tools=request.tools,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error("Agent creation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents", response_model=AgentListResponse, tags=["Agents"])
async def list_agents():
    """List all agents."""
    agents: List[Dict[str, Any]] = []
    for agent_id, agent in app_state.agents.items():
        agents.append(
            {
                "agent_id": agent_id,
                "name": agent.config.name,
                "role": (
                    agent.config.role.value
                    if hasattr(agent.config.role, 'value')
                    else str(agent.config.role)
                ),
                "description": agent.config.description,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return AgentListResponse(agents=agents, total=len(agents))


@app.get("/agents/{agent_id}", tags=["Agents"])
async def get_agent(agent_id: str):
    """Get agent by ID."""
    agent = app_state.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return {
        "agent_id": agent_id,
        "name": agent.config.name,
        "role": (
            agent.config.role.value
            if hasattr(agent.config.role, 'value')
            else str(agent.config.role)
        ),
        "description": agent.config.description,
        "status": "active",
    }


@app.delete("/agents/{agent_id}", tags=["Agents"])
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if agent_id not in app_state.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Cleanup agent
    agent = app_state.agents[agent_id]
    if hasattr(agent, "cleanup"):
        try:
            await agent.cleanup()
        except Exception as e:
            logger.error("Agent cleanup error: %s", e)

    # Remove from state
    del app_state.agents[agent_id]

    return {"message": f"Agent {agent_id} deleted successfully"}


# Tools endpoint
@app.get("/tools", response_model=ToolListResponse, tags=["Tools"])
async def list_tools():
    """List available tools."""
    if not app_state.tools:
        return ToolListResponse(tools=[], total=0)

    tools: List[Dict[str, Any]] = []
    for tool_name in app_state.tools.list_names():
        tool = app_state.tools.get(tool_name)
        tools.append(
            {
                "name": tool_name,
                "description": getattr(tool, "description", "No description"),
                "enabled": True,
            }
        )

    return ToolListResponse(tools=tools, total=len(tools))


# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get system metrics."""
    return MetricsResponse(
        agents_count=len(app_state.agents),
        tools_count=len(app_state.tools.list_names()) if app_state.tools else 0,
        uptime=time.time() - app_state.start_time,
        memory_usage="N/A",
        active_tasks=0,
        request_count="N/A",
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "src.llamaagent.api.simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
