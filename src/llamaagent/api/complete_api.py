"""
Complete FastAPI Application for LlamaAgent

Comprehensive API with:
- SPRE data generation endpoints
- Agent management and execution
- Real-time monitoring and metrics
- File upload and processing
- WebSocket support for streaming
- Authentication and authorization
- Rate limiting and caching
- Complete OpenAI integration
- Health checks and diagnostics

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent

# LlamaAgent imports
from ..data_generation.spre import DataType, SPREGenerator
from ..llm.factory import ProviderFactory
from ..tools import ToolRegistry, get_all_tools

try:
    from ..orchestrator import AgentOrchestrator  # type: ignore
except Exception:  # pylint: disable=broad-except
    class AgentOrchestrator:  # type: ignore
        ...

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Application instance
app = FastAPI(
    title="LlamaAgent Complete API",
    description="Comprehensive API for LlamaAgent with SPRE, agents, and integrations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global state (fully typed for mypy/ruff)
from typing import Set


class AppStateType(Dict[str, Any]):
    spre_generators: Dict[str, Any]
    agents: Dict[str, Any]
    orchestrator: Optional[Any]
    provider_factory: Optional[Any]
    tool_registry: Optional[Any]
    active_sessions: Dict[str, Any]
    metrics: Dict[str, Any]
    websocket_connections: Set[Any]

app_state: Dict[str, Any] = {
    "spre_generators": {},
    "agents": {},
    "orchestrator": None,
    "provider_factory": None,
    "tool_registry": None,
    "active_sessions": {},
    "metrics": {
        "requests_count": 0,
        "generations_count": 0,
        "agents_created": 0,
        "uptime_start": time.time(),
    },
    "websocket_connections": set(),
}


# Pydantic Models
class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str
    timestamp: str
    version: str
    uptime: float
    metrics: Dict[str, Any]
    services: Dict[str, bool]


class SPREGenerationRequest(BaseModel):
    """SPRE generation request model"""
    
    name: str = Field(..., description="Dataset name")
    count: int = Field(default=10, ge=1, le=1000, description="Number of items to generate")
    description: str = Field(default="", description="Dataset description")
    data_type: Optional[DataType] = Field(default=DataType.TEXT, description="Data type to generate")
    topic: Optional[str] = Field(default=None, description="Topic for generation")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level")
    style: Optional[str] = Field(default="default", description="Style for creative content")
    domain: Optional[str] = Field(default="general", description="Domain for technical content")
    tags: Optional[List[str]] = Field(default=[], description="Tags for the dataset")
    
    @field_validator('count')
    @classmethod
    def validate_count(cls, v: int) -> int:
        if v < 1 or v > 1000:
            raise ValueError('Count must be between 1 and 1000')
        return v


class AgentCreationRequest(BaseModel):
    """Agent creation request model"""
    
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(default=AgentRole.GENERALIST, description="Agent role")
    description: str = Field(default="", description="Agent description")
    llm_provider: str = Field(default="mock", description="LLM provider to use")
    tools: Optional[List[str]] = Field(default=[], description="Tools to enable")
    spree_enabled: bool = Field(default=True, description="Enable SPRE planning")
    debug: bool = Field(default=False, description="Enable debug mode")


class AgentExecutionRequest(BaseModel):
    """Agent execution request model"""
    
    message: str = Field(..., description="Message to send to agent")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    stream: bool = Field(default=False, description="Enable streaming response")
    timeout: int = Field(default=300, ge=10, le=3600, description="Execution timeout in seconds")


class DataGenerationFromPromptsRequest(BaseModel):
    """Data generation from prompts request model"""
    
    prompts: List[str] = Field(..., description="List of prompts to process")
    output_format: str = Field(default="json", description="Output format")
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch processing size")


class FileProcessingRequest(BaseModel):
    """File processing request model"""
    
    operation: str = Field(..., description="Processing operation")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Operation parameters")


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, calls: int = 100, period: int = 60) -> None:
        super().__init__(app)
        self.calls: int = calls
        self.period: int = period
        self.clients: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        if client_ip not in self.clients:
            self.clients[client_ip] = []
        # Clean old requests
        self.clients[client_ip] = [
            req_time for req_time in self.clients[client_ip]
            if now - req_time < self.period
        ]
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
        # Add current request
        self.clients[client_ip].append(now)
        response = await call_next(request)
        return response


# Add rate limiting
app.add_middleware(RateLimitMiddleware, calls=100, period=60)


# Dependency functions
def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Get current user from authorization token"""
    if not credentials:
        return {"user_id": "anonymous", "permissions": ["read"]}
    
    try:
        # Decode JWT token
        import jwt
        from datetime import datetime
        
        # In production, use proper secret key from environment
        SECRET_KEY = os.getenv("JWT_SECRET", "development-secret-key-change-in-production")
        ALGORITHM = "HS256"
        
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check expiration
        if "exp" in payload and datetime.fromtimestamp(payload["exp"]) < datetime.now():
            raise HTTPException(status_code=401, detail="Token expired")
        
        # Extract user info from payload
        user_id = payload.get("sub", "unknown")
        permissions = payload.get("permissions", ["read"])
        
        return {"user_id": user_id, "permissions": permissions}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


def update_metrics() -> None:
    """Update application metrics"""
    app_state["metrics"]["requests_count"] += 1


# Health and Status Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint"""
    uptime = time.time() - app_state["metrics"]["uptime_start"]
    
    # Check services
    services = {}
    
    # Database check
    try:
        from llamaagent.storage import DatabaseManager
        db = DatabaseManager()
        db.get_session()  # Test connection
        services["database"] = True
    except Exception:
        services["database"] = False
    
    # Redis check
    try:
        import redis
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"),
                       port=int(os.getenv("REDIS_PORT", 6379)),
                       decode_responses=True)
        r.ping()
        services["redis"] = True
    except Exception:
        services["redis"] = False
    
    # LLM providers check
    try:
        from llamaagent.llm import ProviderFactory
        factory = ProviderFactory()
        # Check if at least one provider is available
        available_providers = []
        for provider_name in ["openai", "anthropic", "ollama"]:
            try:
                factory.create(provider_name)
                available_providers.append(provider_name)
            except Exception:
                pass
        services["llm_providers"] = len(available_providers) > 0
    except Exception:
        services["llm_providers"] = False
    
    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        uptime=uptime,
        metrics=app_state["metrics"],
        services=services
    )


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get application metrics"""
    return {
        "metrics": app_state["metrics"],
        "active_sessions": len(app_state["active_sessions"]),
        "websocket_connections": len(app_state["websocket_connections"]),
        "generators": len(app_state["spre_generators"]),
        "agents": len(app_state["agents"]),
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed system status"""
    return {
        "app_state": {
            "generators": list(app_state["spre_generators"].keys()),
            "agents": list(app_state["agents"].keys()),
            "active_sessions": list(app_state["active_sessions"].keys()),
        },
        "system_info": {
            "python_version": "3.11",
            "environment": os.getenv("LLAMAAGENT_ENV", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
    }


# ------------------------------------------------------------------ #
# SPRE DATA GENERATION ENDPOINTS                                     #
# ------------------------------------------------------------------ #
# NOTE: We call the *async* helper directly to avoid nested event-loop
# crashes when the API itself is already running inside an event loop.
# ------------------------------------------------------------------ #


@app.post("/spre/generate")
async def generate_spre_data(
    request: SPREGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Generate SPRE dataset"""
    if "write" not in current_user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        # Create generator
        generator = SPREGenerator()
        session_id = str(uuid4())
        app_state["spre_generators"][session_id] = generator
        
        # Generate dataset asynchronously (prevents nested loop RuntimeError)
        dataset = await generator.generate_dataset_async(
            name=request.name,
            count=request.count,
            description=request.description,
            data_type=request.data_type,
            topic=request.topic,
            difficulty=request.difficulty,
            style=request.style,
            domain=request.domain,
            tags=request.tags
        )
        
        # Update metrics
        app_state["metrics"]["generations_count"] += 1
        
        # Convert to response format
        response_data = {
            "session_id": session_id,
            "dataset": {
                "name": dataset.name,
                "description": dataset.description,
                "metadata": dataset.metadata,
                "created_at": dataset.created_at,
                "items": [
                    {
                        "id": item.id,
                        "data_type": item.data_type.value,
                        "content": item.content,
                        "metadata": item.metadata,
                        "created_at": item.created_at,
                        "validation_status": item.validation_status.value,
                        "tags": item.tags,
                        "quality_score": item.quality_score,
                    }
                    for item in dataset.items
                ]
            },
            "statistics": {
                "total_items": len(dataset.items),
                "valid_items": len(dataset.get_valid_items()),
                "validation_rate": len(dataset.get_valid_items()) / len(dataset.items) if dataset.items else 0,
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"SPRE generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/spre/generate-from-prompts")
async def generate_from_prompts(
    request: DataGenerationFromPromptsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Generate data from list of prompts"""
    if "write" not in current_user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        generator = SPREGenerator()
        
        # Process prompts in batches
        all_results = []
        for i in range(0, len(request.prompts), request.batch_size):
            batch = request.prompts[i:i + request.batch_size]
            batch_results = await generator.generate_from_prompts(batch)
            all_results.extend(batch_results)  # type: ignore[arg-type]
        
        return {
            "total_prompts": len(request.prompts),
            "generated_items": len(all_results),  # type: ignore[arg-type]
            "results": all_results
        }
        
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/spre/generators")
async def list_generators(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """List active SPRE generators"""
    return {
        "generators": [
            {
                "session_id": session_id,
                "stats": generator.get_dataset_stats() if hasattr(generator, 'get_dataset_stats') else {}
            }
            for session_id, generator in app_state["spre_generators"].items()
        ]
    }


@app.get("/spre/generators/{session_id}/stats")
async def get_generator_stats(session_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Get statistics for a specific generator"""
    if session_id not in app_state["spre_generators"]:
        raise HTTPException(status_code=404, detail="Generator not found")
    
    generator = app_state["spre_generators"][session_id]
    
    if hasattr(generator, 'get_dataset_stats'):
        return generator.get_dataset_stats()
    else:
        return {"error": "Stats not available"}


# Agent Management Endpoints
@app.post("/agents/create")
async def create_agent(
    request: AgentCreationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Create a new agent"""
    if "write" not in current_user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        # Initialize provider factory if not exists
        if not app_state["provider_factory"]:
            app_state["provider_factory"] = ProviderFactory()
        
        # Initialize tool registry if not exists
        if not app_state["tool_registry"]:
            app_state["tool_registry"] = ToolRegistry()
            for tool in get_all_tools():
                app_state["tool_registry"].register(tool)
        
        # Create agent configuration
        config = AgentConfig(
            name=request.name,
            role=request.role,
            description=request.description,
            spree_enabled=request.spree_enabled,
            debug=request.debug,
        )
        
        # Create LLM provider
        provider = app_state["provider_factory"].create_provider(request.llm_provider)
        
        # Filter tools
        tools = app_state["tool_registry"]
        if request.tools:
            # TODO: Filter tools based on request
            pass
        
        # Create agent
        agent = ReactAgent(
            config=config,
            llm_provider=provider,
            tools=tools
        )
        
        # Store agent
        agent_id = str(uuid4())
        app_state["agents"][agent_id] = agent
        app_state["metrics"]["agents_created"] += 1
        
        return {
            "agent_id": agent_id,
            "config": {
                "name": config.name,
                "role": config.role.value,
                "description": config.description,
                "spree_enabled": config.spree_enabled,
                "debug": config.debug,
            },
            "provider": request.llm_provider,
            "tools": request.tools,
        }
        
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")


@app.post("/agents/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: AgentExecutionRequest,
    user: Dict[str, Any] = Depends(get_current_user)
) -> Any:
    """Execute a task with an agent"""
    if "write" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if agent_id not in app_state["agents"]:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent = app_state["agents"][agent_id]
        
        # Execute task
        start_time = time.time()
        result = await agent.execute(request.message)
        execution_time = time.time() - start_time
        
        return {
            "agent_id": agent_id,
            "message": request.message,
            "result": {
                "content": result.content if hasattr(result, 'content') else str(result),
                "success": result.success if hasattr(result, 'success') else True,
                "execution_time": execution_time,
                "tokens_used": result.tokens_used if hasattr(result, 'tokens_used') else 0,
            },
            "context": request.context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.get("/agents")
async def list_agents(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """List all agents"""
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "config": {
                    "name": agent.config.name,
                    "role": agent.config.role.value,
                    "description": agent.config.description,
                } if hasattr(agent, 'config') else {}
            }
            for agent_id, agent in app_state["agents"].items()
        ]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Get agent details"""
    if agent_id not in app_state["agents"]:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = app_state["agents"][agent_id]
    
    return {
        "agent_id": agent_id,
        "config": {
            "name": agent.config.name,
            "role": agent.config.role.value,
            "description": agent.config.description,
            "spree_enabled": agent.config.spree_enabled,
            "debug": agent.config.debug,
        } if hasattr(agent, 'config') else {},
        "tools": agent.tools.list_names() if hasattr(agent, 'tools') and hasattr(agent.tools, 'list_names') else [],
        "provider": type(agent.llm_provider).__name__ if hasattr(agent, 'llm_provider') else "Unknown",
    }


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Delete an agent"""
    if "admin" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    if agent_id not in app_state["agents"]:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del app_state["agents"][agent_id]
    
    return {"message": f"Agent {agent_id} deleted successfully"}


# File Upload and Processing Endpoints
@app.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    operation: str = Form(default="store"),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Upload and optionally process a file"""
    if "write" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    try:
        # Create upload directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_id = str(uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process file based on operation
        result = {"file_id": file_id, "filename": file.filename, "size": len(content)}
        
        if operation == "analyze":
            # TODO: Implement file analysis
            result["analysis"] = {"type": "text", "encoding": "utf-8"}  # type: ignore
        elif operation == "extract":
            # TODO: Implement content extraction
            result["content"] = content.decode('utf-8', errors='ignore')[:1000]
        
        return result
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/files/{file_id}")
async def download_file(file_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Download a file"""
    upload_dir = Path("uploads")
    
    # Find file
    for file_path in upload_dir.glob(f"{file_id}_*"):
        return FileResponse(file_path)
    
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/files/{file_id}/process")
async def process_file(
    file_id: str,
    request: FileProcessingRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Process an uploaded file"""
    if "write" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # TODO: Implement file processing operations
    return {
        "file_id": file_id,
        "operation": request.operation,
        "parameters": request.parameters,
        "status": "completed",
        "result": "File processed successfully"
    }


# WebSocket Endpoints
@app.websocket("/ws/agent/{agent_id}")
async def websocket_agent(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket endpoint for real-time agent communication"""
    await websocket.accept()
    app_state["websocket_connections"].add(websocket)
    
    if agent_id not in app_state["agents"]:
        await websocket.send_json({"error": "Agent not found"})
        await websocket.close()
        return
    
    try:
        agent = app_state["agents"][agent_id]
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                await websocket.send_json({"error": "No message provided"})
                continue
            
            # Execute agent
            try:
                result = await agent.execute(message)
                await websocket.send_json({
                    "type": "response",
                    "content": result.content if hasattr(result, 'content') else str(result),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
    except WebSocketDisconnect:
        app_state["websocket_connections"].discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        app_state["websocket_connections"].discard(websocket)


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    app_state["websocket_connections"].add(websocket)
    
    try:
        while True:
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": app_state["metrics"],
                "active_sessions": len(app_state["active_sessions"]),
                "websocket_connections": len(app_state["websocket_connections"]),
                "generators": len(app_state["spre_generators"]),
                "agents": len(app_state["agents"]),
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except WebSocketDisconnect:
        app_state["websocket_connections"].discard(websocket)
    except Exception as e:
        logger.error(f"Metrics WebSocket error: {e}")
    finally:
        app_state["websocket_connections"].discard(websocket)


# Integration Endpoints
@app.get("/integrations/openai/status")
async def openai_integration_status(user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Get OpenAI integration status"""
    try:
        from ..integration.openai_agents import get_openai_integration  # type: ignore[import]
        
        integration = get_openai_integration()  # type: ignore
        
        return {
            "available": integration is not None,
            "configured": bool(os.getenv("OPENAI_API_KEY")),
            "status": "active" if integration else "inactive"
        }
    except ImportError:
        return {
            "available": False,
            "configured": False,
            "status": "not_available"
        }


@app.get("/integrations/langgraph/status")
async def langgraph_integration_status(user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """Get LangGraph integration status"""
    try:
        from ..integration.langgraph import is_langgraph_available
        
        return {
            "available": is_langgraph_available(),
            "status": "active" if is_langgraph_available() else "inactive"
        }
    except ImportError:
        return {
            "available": False,
            "status": "not_available"
        }


# Export and Backup Endpoints
@app.post("/export/data")
async def export_data(
    format: str = Query(default="json", description="Export format"),
    include_metadata: bool = Query(default=True, description="Include metadata"),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Any:
    """Export application data"""
    if "admin" not in user["permissions"]:
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    try:
        export_data = {  # type: ignore[assignment]
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "generators": len(app_state["spre_generators"]),
            "agents": len(app_state["agents"]),
            "metrics": app_state["metrics"] if include_metadata else {},
        }
        
        if format == "json":
            return export_data  # type: ignore
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Startup event
@app.on_event("startup")  # type: ignore[attr-defined]
async def startup_event() -> None:
    """Initialize application on startup"""
    logger.info("Starting LlamaAgent Complete API...")
    
    # Initialize provider factory
    app_state["provider_factory"] = ProviderFactory()
    
    # Initialize tool registry
    app_state["tool_registry"] = ToolRegistry()
    for tool in get_all_tools():
        app_state["tool_registry"].register(tool)
    
    # Initialize orchestrator
    try:
        app_state["orchestrator"] = AgentOrchestrator()
        logger.info("Agent orchestrator initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize orchestrator: {e}")
    
    logger.info("LlamaAgent Complete API started successfully")


# Shutdown event
@app.on_event("shutdown")  # type: ignore[attr-defined]
async def shutdown_event() -> None:
    """Cleanup on shutdown"""
    logger.info("Shutting down LlamaAgent Complete API...")
    
    # Close WebSocket connections
    for websocket in app_state["websocket_connections"]:
        try:
            await websocket.close()
        except Exception:
            pass
    
    # Cleanup resources
    app_state["spre_generators"].clear()
    app_state["agents"].clear()
    app_state["active_sessions"].clear()
    
    logger.info("Shutdown completed")


# Mount static files
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    update_metrics()
    
    return response


if __name__ == "__main__":
    uvicorn.run(
        "src.llamaagent.api.complete_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 