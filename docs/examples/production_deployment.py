"""
Production Deployment Example

This example shows how to deploy LlamaAgent in a production environment
with monitoring, caching, security, and scalability features.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llamaagent import Agent
from llamaagent.api import AgentRequest, AgentResponse
from llamaagent.cache import CacheConfig, HierarchicalCache, LLMCache
from llamaagent.llm import LiteLLMProvider
from llamaagent.memory import VectorMemory
from llamaagent.monitoring import HealthChecker, PrometheusMonitor
from llamaagent.security import APIKeyAuth, InputValidator, JWTAuth, RateLimiter
from llamaagent.tools import CodeExecutionTool, WebSearchTool


# Configuration
class Config:
    """Production configuration"""
    # API Settings
    API_TITLE = "LlamaAgent Production API"
    API_VERSION = "1.0.0"
    API_PREFIX = "/api/v1"
    
    # Model Settings
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
    
    # Cache Settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Database Settings
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/llamaagent")
    
    # Security Settings
    API_KEY_HEADER = "X-API-Key"
    JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # Monitoring
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")


# Global instances
config = Config()
monitor = PrometheusMonitor(port=config.METRICS_PORT)
health_checker = HealthChecker()
rate_limiter = RateLimiter(
    requests_per_minute=config.RATE_LIMIT_REQUESTS,
    window_size=config.RATE_LIMIT_WINDOW
)
input_validator = InputValidator(
    max_length=10000,
    forbidden_patterns=["<script", "javascript:", "onclick"]
)


# Cache initialization
cache_config = CacheConfig(
    primary_backend="memory",
    secondary_backend="disk",
    redis_url=config.REDIS_URL,
    default_ttl=config.CACHE_TTL,
    enable_compression=True,
    enable_warming=True
)
cache_manager = HierarchicalCache(cache_config)


# Authentication
api_key_auth = APIKeyAuth(header_name=config.API_KEY_HEADER)
jwt_auth = JWTAuth(secret=config.JWT_SECRET)


# Agent factory with fallback
class AgentFactory:
    """Factory for creating agents with fallback support"""
    
    @staticmethod
    async def create_agent(
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> Agent:
        """Create an agent with production features"""
        
        # Primary LLM with fallback
        primary_llm = LiteLLMProvider(
            model=model or config.PRIMARY_MODEL,
            temperature=0.7,
            timeout=30,
            retry_count=3
        )
        
        # Fallback LLM
        fallback_llm = LiteLLMProvider(
            model=config.FALLBACK_MODEL,
            temperature=0.7,
            timeout=30
        )
        
        # Create LLM with fallback
        llm = primary_llm.with_fallback(fallback_llm)
        
        # Initialize cache if enabled
        llm_cache = None
        if use_cache:
            llm_cache = LLMCache(
                cache_backend="redis",
                redis_url=config.REDIS_URL,
                ttl=config.CACHE_TTL,
                enable_semantic_cache=True,
                similarity_threshold=0.95
            )
        
        # Create agent
        agent = Agent(
            name="ProductionAgent",
            llm=llm,
            tools=[
                WebSearchTool(max_results=5),
                CodeExecutionTool(sandbox=True, timeout=10)
            ],
            memory=VectorMemory(
                embedding_model="all-MiniLM-L6-v2",
                persist_directory="./vector_store"
            ),
            cache=llm_cache,
            monitor=monitor,
            middleware=[
                rate_limiter,
                input_validator
            ],
            system_prompt="""You are a helpful AI assistant in a production environment.
            Follow these guidelines:
            1. Provide accurate, helpful responses
            2. Use tools when appropriate
            3. Be concise but thorough
            4. Admit when you don't know something
            5. Prioritize user safety and privacy"""
        )
        
        return agent


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("Starting LlamaAgent Production API...")
    
    # Initialize monitoring
    await monitor.start()
    
    # Initialize cache background tasks
    await cache_manager.start_background_tasks()
    
    # Warm cache with common queries
    common_queries = [
        "Hello",
        "What can you help me with?",
        "How do I use this API?"
    ]
    await cache_manager.warm_cache(common_queries)
    
    # Register health checks
    health_checker.register_check("cache", cache_manager.health_check)
    health_checker.register_check("database", check_database_health)
    
    print("LlamaAgent API started successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down LlamaAgent API...")
    
    # Stop background tasks
    await cache_manager.stop_background_tasks()
    await monitor.stop()
    
    print("LlamaAgent API shut down complete.")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
monitor.instrument_app(app)


# Dependency for getting authenticated user
async def get_current_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_auth)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not api_key:
        # Try JWT auth
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        user = await jwt_auth.verify_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user
    
    # Verify API key
    user = await api_key_auth.verify_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    results = await health_checker.check_all()
    
    status_code = 200 if all(r["healthy"] for r in results.values()) else 503
    
    return {
        "status": "healthy" if status_code == 200 else "unhealthy",
        "checks": results
    }


# Metrics endpoint (Prometheus format)
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return monitor.export_metrics()


# Main chat endpoint
@app.post(f"{config.API_PREFIX}/chat")
async def chat(
    request: AgentRequest,
    user: Dict[str, Any] = Depends(get_current_user)
) -> AgentResponse:
    """Main chat endpoint with full production features"""
    
    # Track request
    with monitor.track_request("chat", user_id=user.get("id")):
        try:
            # Create agent for this request
            agent = await AgentFactory.create_agent(
                model=request.model,
                use_cache=not request.disable_cache
            )
            
            # Set user context
            agent.set_context({
                "user_id": user.get("id"),
                "session_id": request.session_id,
                "metadata": request.metadata
            })
            
            # Execute request
            response = await agent.run(
                request.message,
                stream=False,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            # Track metrics
            monitor.track_tokens(response.usage.get("total_tokens", 0))
            monitor.track_cost(response.cost)
            
            return AgentResponse(
                message=response.content,
                usage=response.usage,
                cost=response.cost,
                model=response.model,
                tools_used=response.tools_used,
                cached=response.cached
            )
            
        except Exception as e:
            monitor.track_error(str(e))
            raise HTTPException(status_code=500, detail=str(e))


# Streaming chat endpoint
@app.post(f"{config.API_PREFIX}/chat/stream")
async def chat_stream(
    request: AgentRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Streaming chat endpoint"""
    
    async def stream_generator():
        """Generate streaming response"""
        try:
            # Create agent
            agent = await AgentFactory.create_agent(
                model=request.model,
                use_cache=False  # Disable cache for streaming
            )
            
            # Set context
            agent.set_context({
                "user_id": user.get("id"),
                "session_id": request.session_id
            })
            
            # Stream response
            async for chunk in agent.run_stream(
                request.message,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ):
                yield f"data: {chunk.json()}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


# Batch processing endpoint
@app.post(f"{config.API_PREFIX}/batch")
async def batch_chat(
    requests: List[AgentRequest],
    user: Dict[str, Any] = Depends(get_current_user)
) -> List[AgentResponse]:
    """Process multiple requests in batch"""
    
    # Validate batch size
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Batch size limited to 10")
    
    # Create agent once for efficiency
    agent = await AgentFactory.create_agent()
    
    # Process requests in parallel
    tasks = []
    for req in requests:
        tasks.append(
            agent.run(
                req.message,
                max_tokens=req.max_tokens,
                temperature=req.temperature
            )
        )
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Format responses
    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            results.append(AgentResponse(
                message=f"Error: {str(response)}",
                error=True
            ))
        else:
            results.append(AgentResponse(
                message=response.content,
                usage=response.usage,
                cost=response.cost,
                model=response.model,
                tools_used=response.tools_used
            ))
    
    return results


# Admin endpoints
@app.get(f"{config.API_PREFIX}/admin/stats")
async def admin_stats(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system statistics (admin only)"""
    
    # Check admin permission
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Gather statistics
    stats = {
        "cache": await cache_manager.get_stats(),
        "monitoring": monitor.get_stats(),
        "agents": {
            "active": Agent.get_active_count(),
            "total_created": Agent.get_total_created()
        },
        "health": await health_checker.check_all()
    }
    
    return stats


@app.post(f"{config.API_PREFIX}/admin/cache/clear")
async def clear_cache(
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Clear all caches (admin only)"""
    
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    await cache_manager.clear()
    
    return {"message": "Cache cleared successfully"}


# Utility functions
async def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    try:
        # Implement actual database check
        # For now, return mock healthy status
        return {"healthy": True, "latency_ms": 5}
    except Exception as e:
        return {"healthy": False, "error": str(e)}


# WebSocket endpoint for real-time chat
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket(f"{config.API_PREFIX}/ws")
async def websocket_chat(
    websocket: WebSocket,
    api_key: str
):
    """WebSocket endpoint for real-time chat"""
    
    # Verify API key
    user = await api_key_auth.verify_key(api_key)
    if not user:
        await websocket.close(code=1008, reason="Unauthorized")
        return
    
    await websocket.accept()
    
    # Create agent for this connection
    agent = await AgentFactory.create_agent()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process message
            response = await agent.run(
                data.get("message", ""),
                stream=True
            )
            
            # Stream response
            async for chunk in response:
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk.content
                })
            
            # Send completion
            await websocket.send_json({
                "type": "complete",
                "usage": response.usage,
                "cost": response.cost
            })
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user.get('id')}")


# Main entry point
if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "production_deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        workers=4,  # Number of worker processes
        log_level="info",
        access_log=True
    )