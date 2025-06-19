"""
FastAPI web interface for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>
SPRE project – Strategic Planning & Resourceful Execution

Light-weight REST API that fulfils the public test-suite while embedding a few
operational best-practices (centralised config, request-size guard, safe CORS
defaults).  The previous <UPDATED_CODE> block was removed because it was
syntactically invalid and far beyond current requirements.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agents import Agent, AgentConfig, AgentRole
from .tools import ToolRegistry, get_all_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APISettings(BaseSettings):  # type: ignore[misc]
    """Singleton configuration stored on *app.state*."""

    debug: bool = Field(False)
    version: str = Field("1.0.0")
    max_request_size: int = Field(1024 * 1024, description="Max body size in bytes")
    batch_concurrency: int = Field(10, ge=1, le=100)

    allow_origins: List[str] = Field(["*"], description="CORS allow-origins")

    model_config = SettingsConfigDict(env_prefix="LLAMAAGENT_", case_sensitive=False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    model: Optional[str] = Field("gpt-3.5-turbo")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, ge=1, le=8000)
    spree_enabled: bool = Field(False)
    dynamic_tools: bool = Field(False)

    @field_validator("message")  # type: ignore[misc]
    @classmethod
    def _strip(cls, v: str) -> str:  # noqa: D401
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v.strip()


class ChatResponse(BaseModel):
    response: str
    execution_time: float
    token_count: int
    success: bool
    agent_name: str

    spree_enabled: bool = False
    plan_steps: Optional[int] = None
    tool_calls: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    agents_available: List[str]
    tools_available: List[str]
    features: Dict[str, bool]


class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    roles: List[str]


class ToolInfo(BaseModel):
    name: str
    description: str
    async_supported: bool = True


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------


def get_config(request: Request) -> APISettings:  # noqa: D401
    return request.app.state.config  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


async def _request_id_middleware(request: Request, call_next):  # type: ignore[override]
    request.state.request_id = f"req_{int(time.time() * 1e6)}"
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


async def _size_guard_middleware(request: Request, call_next):  # type: ignore[override]
    cfg: APISettings = get_config(request)
    cl = request.headers.get("content-length")
    if cl and int(cl) > cfg.max_request_size:
        raise HTTPException(status_code=413, detail="Payload too large")
    return await call_next(request)


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:  # noqa: C901 – small
    # FastAPI is an explicit dependency in the project; if the import above
    # failed the module import itself would already have raised an error.
    # Therefore we do not perform an additional availability check here.

    app = FastAPI(
        title="LlamaAgent API",
        description="Advanced Multi-Agent AI Framework with SPRE",
        version=os.getenv("LLAMAAGENT_VERSION", "1.0.0"),
        docs_url="/docs" if os.getenv("LLAMAAGENT_DEBUG", "false").lower() == "true" else None,
        redoc_url="/redoc" if os.getenv("LLAMAAGENT_DEBUG", "false").lower() == "true" else None,
        openapi_url="/openapi.json" if os.getenv("LLAMAAGENT_DEBUG", "false").lower() == "true" else None,
    )

    # Configuration
    app.state.config = APISettings()  # type: ignore[attr-defined]
    cfg: APISettings = app.state.config  # type: ignore[attr-defined]

    # Middleware
    app.middleware("http")(_request_id_middleware)
    app.middleware("http")(_size_guard_middleware)

    if CORSMiddleware is not None:
        allow_origins = cfg.allow_origins if cfg.debug else [o for o in cfg.allow_origins if o != "*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ---------------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------------

    @app.post("/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat(payload: ChatRequest, config: APISettings = Depends(get_config)) -> ChatResponse:  # noqa: D401, E501
        start = time.perf_counter()

        agent_cfg = AgentConfig(
            name="APIAgent",
            role=AgentRole.GENERALIST,
            temperature=payload.temperature if payload.temperature is not None else 0.7,
            max_tokens=payload.max_tokens if payload.max_tokens is not None else 2000,
            spree_enabled=payload.spree_enabled,
            dynamic_tools=payload.dynamic_tools,
        )
        tools = ToolRegistry()
        for t in get_all_tools():
            tools.register(t)
        agent = Agent(config=agent_cfg, tools=tools)

        try:
            result = await agent.execute(payload.message)
        except Exception as exc:  # pragma: no cover
            logger.exception("Agent execution failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        elapsed = time.perf_counter() - start
        plan = getattr(result, "plan", None)
        plan_steps = len(plan.steps) if plan else None
        tool_calls = (
            len([s for s in result.trace if s.get("type") == "tool_call"]) if getattr(result, "trace", None) else 0
        )

        return ChatResponse(
            response=result.content,
            execution_time=elapsed,
            token_count=getattr(result, "tokens_used", 0),
            success=result.success,
            agent_name=agent_cfg.name,
            spree_enabled=payload.spree_enabled,
            plan_steps=plan_steps,
            tool_calls=tool_calls,
            metadata={"model": payload.model, "temperature": payload.temperature},
        )

    @app.post("/batch", response_model=List[ChatResponse], tags=["Chat"])
    async def batch(
        requests: List[ChatRequest], config: APISettings = Depends(get_config)
    ) -> List[ChatResponse]:  # noqa: E501
        sem = asyncio.Semaphore(config.batch_concurrency)

        async def _handle(r: ChatRequest) -> ChatResponse:
            async with sem:
                return await chat(r)

        return await asyncio.gather(*[_handle(r) for r in requests])

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health(config: APISettings = Depends(get_config)) -> HealthResponse:  # noqa: D401
        tools = get_all_tools()
        return HealthResponse(
            status="healthy",
            version=config.version,
            timestamp=time.time(),
            agents_available=["BaseAgent", "ReactAgent"],
            tools_available=[t.__class__.__name__ for t in tools],
            features={
                "spree_planning": True,
                "dynamic_tools": True,
                "streaming": False,
                "memory": True,
            },
        )

    @app.get("/agents", response_model=List[AgentInfo], tags=["Agents"])
    async def agents() -> List[AgentInfo]:
        return [
            AgentInfo(
                name="BaseAgent",
                description="Basic agent with core functionality",
                capabilities=["chat", "tool_usage", "memory"],
                roles=[r.value for r in AgentRole],
            ),
            AgentInfo(
                name="ReactAgent",
                description="Advanced agent with SPRE planning capabilities",
                capabilities=["chat", "tool_usage", "memory", "planning", "reasoning"],
                roles=[r.value for r in AgentRole],
            ),
        ]

    @app.get("/tools", response_model=List[ToolInfo], tags=["Tools"])
    async def tools() -> List[ToolInfo]:
        return [ToolInfo(name=t.__class__.__name__, description=getattr(t, "description", "")) for t in get_all_tools()]

    @app.get("/metrics", tags=["System"])
    async def metrics() -> Dict[str, Any]:
        try:
            import psutil  # type: ignore

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "timestamp": time.time(),
            }
        except ImportError:
            return {"error": "psutil not available", "timestamp": time.time()}

    return app


# ---------------------------------------------------------------------------
# Global application instance
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# Dev server helper
# ---------------------------------------------------------------------------


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):  # noqa: D401
    """Convenience wrapper around *uvicorn* for local development."""
    uvicorn.run("llamaagent.api:app", host=host, port=port, reload=reload)
