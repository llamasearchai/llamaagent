#!/usr/bin/env python3
"""
LlamaAgent Master Program - Complete OpenAI Agents SDK Integration

A comprehensive command-line interface and FastAPI server for running agents with
OpenAI integration, budget tracking, complete experiment management, automated testing,
and production deployment capabilities.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our core components
try:
    from llamaagent.agents.base import AgentConfig
    from llamaagent.agents.react import ReactAgent
    from llamaagent.integration.openai_agents import (
        OpenAIAgentMode,
        OpenAIIntegrationConfig,
        create_openai_integration,
    )
    from llamaagent.llm.providers.openai_provider import OpenAIProvider
    from llamaagent.llm.factory import create_provider
    from llamaagent.tools.base import ToolRegistry
    from llamaagent.tools.calculator import CalculatorTool
    from llamaagent.tools.python_repl import PythonREPLTool
    from llamaagent.types import TaskInput
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False
    OPENAI_AGENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

# Global state
app_state = {
    "agents": {},
    "integrations": {},
    "active_tasks": {},
    "system_metrics": {
        "requests_processed": 0,
        "total_cost": 0.0,
        "uptime_start": time.time()
    }
}

# Pydantic models for API
class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""
    name: str = Field(..., description="Agent name")
    provider: str = Field("mock", description="LLM provider")
    model: str = Field("gpt-4o-mini", description="Model name")
    budget_limit: float = Field(100.0, description="Budget limit in USD")
    tools: List[str] = Field(default_factory=list, description="Tool names to enable")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")

class TaskRequest(BaseModel):
    """Request to execute a task."""
    agent_name: str = Field(..., description="Agent name")
    task: str = Field(..., description="Task to execute")
    mode: str = Field("hybrid", description="Execution mode (openai, native, hybrid)")

class TaskResponse(BaseModel):
    """Response from task execution."""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class BudgetStatus(BaseModel):
    """Budget status response."""
    budget_limit: float
    current_cost: float
    remaining_budget: float
    total_calls: int

class MasterProgramManager:
    """Main manager for the master program."""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.integrations: Dict[str, Any] = {}
        self.tools_registry = ToolRegistry()
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup available tools."""
        if IMPORTS_AVAILABLE:
            calculator_tool = CalculatorTool()
            python_repl_tool = PythonREPLTool()
            self.tools_registry.register(calculator_tool)
            self.tools_registry.register(python_repl_tool)
    
    async def create_agent(self, config: AgentCreateRequest) -> Dict[str, Any]:
        """Create a new agent with OpenAI integration."""
        if not IMPORTS_AVAILABLE:
            return {"success": False, "error": "Required imports not available"}
        
        try:
            # Create LLM provider
            if config.provider == "openai" and config.openai_api_key:
                llm_provider = OpenAIProvider(
                    model_name=config.model,
                    api_key=config.openai_api_key,
                    temperature=0.1
                )
            else:
                llm_provider = create_provider("mock")
            
            # Create agent config
            agent_config = AgentConfig(
                name=config.name,
                temperature=0.1
            )
            
            # Create agent
            agent = ReactAgent(
                config=agent_config,
                llm_provider=llm_provider,
                tools=self.tools_registry
            )
            
            # Create OpenAI integration
            if config.provider == "openai" and OPENAI_AGENTS_AVAILABLE:
                integration = create_openai_integration(
                    openai_api_key=config.openai_api_key,
                    model_name=config.model,
                    budget_limit=config.budget_limit
                )
                adapter = integration.register_agent(agent)
                self.integrations[config.name] = integration
            
            self.agents[config.name] = agent
            app_state["agents"][config.name] = agent
            app_state["integrations"][config.name] = self.integrations.get(config.name)
            
            return {
                "success": True,
                "agent_name": config.name,
                "provider": config.provider,
                "model": config.model,
                "integration_enabled": config.name in self.integrations
            }
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_task(self, request: TaskRequest) -> TaskResponse:
        """Execute a task on an agent."""
        if request.agent_name not in self.agents:
            return TaskResponse(
                task_id="error",
                status="failed",
                error="Agent not found"
            )
        
        try:
            agent = self.agents[request.agent_name]
            integration = self.integrations.get(request.agent_name)
            
            task_input = TaskInput(
                id=f"task_{int(time.time())}",
                task=request.task
            )
            
            # Execute based on mode
            if request.mode == "openai" and integration and OPENAI_AGENTS_AVAILABLE:
                adapter = integration.get_adapter(request.agent_name)
                if adapter:
                    result = await adapter.run_task(task_input)
                else:
                    result = await agent.execute_task(task_input)
            else:
                result = await agent.execute_task(task_input)
            
            # Update metrics
            app_state["system_metrics"]["requests_processed"] += 1
            
            return TaskResponse(
                task_id=result.task_id,
                status=result.status.value,
                result=result.result.__dict__ if result.result else None,
                completed_at=result.completed_at.isoformat() if result.completed_at else None,
                metadata=getattr(result.result, 'metadata', {}) if result.result else None
            )
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return TaskResponse(
                task_id=request.task if hasattr(request, 'task') else "error",
                status="failed",
                error=str(e)
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - app_state["system_metrics"]["uptime_start"]
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "agents_count": len(self.agents),
            "integrations_count": len(self.integrations),
            "requests_processed": app_state["system_metrics"]["requests_processed"],
            "total_cost": app_state["system_metrics"]["total_cost"],
            "openai_agents_available": OPENAI_AGENTS_AVAILABLE,
            "imports_available": IMPORTS_AVAILABLE,
            "tools_available": self.tools_registry.list_tools() if hasattr(self.tools_registry, 'list_tools') else []
        }

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    console.print("[bold green]Starting LlamaAgent Master Program API...[/bold green]")
    yield
    console.print("[bold red]Shutting down LlamaAgent Master Program API...[/bold red]")

app = FastAPI(
    title="LlamaAgent Master Program API",
    description="Complete LlamaAgent system with OpenAI Agents SDK integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize manager
manager = MasterProgramManager()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LlamaAgent Master Program API",
        "version": "1.0.0",
        "author": "Nik Jois <nikjois@llamasearch.ai>",
        "openai_agents_available": OPENAI_AGENTS_AVAILABLE,
        "imports_available": IMPORTS_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "agents": "/agents",
            "tasks": "/tasks",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return manager.get_system_status()

@app.post("/agents", response_model=Dict[str, Any])
async def create_agent(config: AgentCreateRequest):
    """Create a new agent."""
    return await manager.create_agent(config)

@app.get("/agents")
async def list_agents():
    """List all agents."""
    agents_info = []
    for name, agent in manager.agents.items():
        integration = manager.integrations.get(name)
        budget_status = integration.get_budget_status() if integration else {}
        
        agents_info.append({
            "name": name,
            "description": getattr(agent, 'description', ''),
            "tools": manager.tools_registry.list_tools() if hasattr(manager.tools_registry, 'list_tools') else [],
            "budget_status": budget_status,
            "integration_enabled": name in manager.integrations
        })
    
    return {"agents": agents_info}

@app.post("/tasks", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task on an agent."""
    return await manager.execute_task(request)

@app.get("/status")
async def get_status():
    """Get system status."""
    return manager.get_system_status()

@app.get("/budget/{agent_name}", response_model=BudgetStatus)
async def get_budget_status(agent_name: str):
    """Get budget status for an agent."""
    if agent_name not in manager.integrations:
        raise HTTPException(status_code=404, detail="Agent integration not found")
    
    integration = manager.integrations[agent_name]
    budget_data = integration.get_budget_status()
    
    return BudgetStatus(**budget_data)

# CLI Interface
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """LlamaAgent Master Program CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def server(host, port, reload):
    """Start the FastAPI server."""
    console.print(f"[bold green]Starting LlamaAgent Master Program Server on {host}:{port}[/bold green]")
    uvicorn.run(
        "master_program:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@cli.command()
@click.option('--openai-key', help='OpenAI API key')
@click.option('--model', default='gpt-4o-mini', help='Model to use')
@click.option('--budget', default=10.0, help='Budget limit')
def demo(openai_key, model, budget):
    """Run a demonstration of the system."""
    async def run_demo():
        console.print("[bold blue]LlamaAgent Master Program Demo[/bold blue]")
        
        # Create manager
        demo_manager = MasterProgramManager()
        
        # Create agent
        config = AgentCreateRequest(
            name="demo_agent",
            provider="openai" if openai_key else "mock",
            model=model,
            budget_limit=budget,
            openai_api_key=openai_key
        )
        
        result = await demo_manager.create_agent(config)
        if not result["success"]:
            console.print(f"[red]Failed to create agent: {result['error']}[/red]")
            return
        
        console.print(f"[green]Created agent: {result['agent_name']}[/green]")
        
        # Execute demo tasks
        demo_tasks = [
            "What is artificial intelligence?",
            "Calculate the square root of 144",
            "Explain the benefits of using AI agents"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for i, task in enumerate(demo_tasks):
                task_progress = progress.add_task(f"Executing task {i+1}: {task[:50]}...", total=1)
                
                request = TaskRequest(
                    agent_name="demo_agent",
                    task=task,
                    mode="hybrid"
                )
                
                response = await demo_manager.execute_task(request)
                progress.update(task_progress, completed=1)
                
                console.print(f"\n[bold]Task {i+1}:[/bold] {task}")
                console.print(f"[bold]Status:[/bold] {response.status}")
                if response.result:
                    console.print(f"[bold]Result:[/bold] {response.result.get('data', 'No data')}")
                if response.error:
                    console.print(f"[red]Error:[/red] {response.error}")
                console.print("-" * 80)
        
        # Show system status
        status = demo_manager.get_system_status()
        
        table = Table(title="System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", status["status"])
        table.add_row("Agents", str(status["agents_count"]))
        table.add_row("Requests Processed", str(status["requests_processed"]))
        table.add_row("OpenAI Available", str(status["openai_agents_available"]))
        table.add_row("Imports Available", str(status["imports_available"]))
        
        console.print(table)
    
    asyncio.run(run_demo())

@cli.command()
def test():
    """Run comprehensive tests."""
    console.print("[bold blue]Running LlamaAgent Master Program Tests[/bold blue]")
    
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    console.print(result.stdout)
    if result.stderr:
        console.print(f"[red]{result.stderr}[/red]")
    
    if result.returncode == 0:
        console.print("[bold green]All tests passed![/bold green]")
    else:
        console.print(f"[bold red]Tests failed with exit code {result.returncode}[/bold red]")

@cli.command()
def build():
    """Build and validate the complete system."""
    console.print("[bold blue]Building LlamaAgent Master Program[/bold blue]")
    
    steps = [
        ("Installing dependencies", "pip install -e ."),
        ("Running linting", "python -m flake8 src/ --max-line-length=100 --ignore=E203,W503"),
        ("Running type checking", "python -m mypy src/ --ignore-missing-imports"),
        ("Running tests", "python -m pytest tests/ -v"),
        ("Building Docker image", "docker build -t llamaagent-master:latest ."),
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for step_name, command in steps:
            task = progress.add_task(step_name, total=1)
            
            try:
                result = subprocess.run(
                    command.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=300
                )
                if result.returncode == 0:
                    progress.update(task, completed=1)
                    console.print(f"[green][/green] {step_name}")
                else:
                    console.print(f"[red][/red] {step_name}")
                    console.print(f"[red]{result.stderr}[/red]")
            except subprocess.TimeoutExpired:
                console.print(f"[yellow]⏱[/yellow] {step_name} (timeout)")
            except Exception as e:
                console.print(f"[red][/red] {step_name}: {e}")

@cli.command()
def status():
    """Show system status."""
    async def show_status():
        demo_manager = MasterProgramManager()
        status = demo_manager.get_system_status()
        
        layout = Layout()
        
        status_table = Table(title="LlamaAgent Master Program Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        
        status_table.add_row("System", status["status"])
        status_table.add_row("OpenAI Agents SDK", "Available" if status["openai_agents_available"] else "Not Available")
        status_table.add_row("Imports", "Available" if status["imports_available"] else "Not Available")
        status_table.add_row("Active Agents", str(status["agents_count"]))
        status_table.add_row("Integrations", str(status["integrations_count"]))
        status_table.add_row("Requests Processed", str(status["requests_processed"]))
        
        console.print(status_table)
    
    asyncio.run(show_status())

if __name__ == "__main__":
    cli() 