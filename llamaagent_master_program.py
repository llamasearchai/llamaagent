#!/usr/bin/env python3
"""
LlamaAgent Master Program - Complete AI Agent System with Dynamic Task Planning and Subagent Spawning

This is the ultimate master program that integrates:
- OpenAI Agents SDK compatibility
- Dynamic task planning and decomposition
- Intelligent subagent spawning
- Hierarchical agent management
- Complete task execution pipeline
- Production-ready deployment capabilities

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.tree import Tree

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import core components with error handling
try:
    from llamaagent.agents.base import AgentConfig, AgentRole, BaseAgent
    from llamaagent.agents.react import ReactAgent
    from llamaagent.integration.openai_agents_complete import (
        OpenAIAgentsManager,
        create_openai_agent_manager,
        AgentState as OpenAIAgentState,
    )
    from llamaagent.llm.factory import create_provider
    from llamaagent.planning.task_planner import (
        Task,
        TaskDecomposer,
        TaskPlan,
        TaskPlanner,
        TaskPriority,
        TaskStatus,
    )
    from llamaagent.spawning.agent_spawner import (
        AgentHierarchy,
        AgentRelationship,
        AgentSpawner,
        SpawnConfig,
    )
    from llamaagent.tools import ToolRegistry, get_all_tools
    from llamaagent.types import TaskInput, TaskOutput, TaskResult, TaskStatus as TypeTaskStatus
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

# Global configuration
MASTER_CONFIG = {
    "max_agents": 100,
    "max_concurrent_tasks": 50,
    "default_timeout": 300.0,
    "enable_auto_spawn": True,
    "enable_dynamic_planning": True,
    "enable_openai_integration": True,
    "default_model": "gpt-4o-mini",
    "agent_memory_mb": 512,
    "system_memory_mb": 4096,
}

# Pydantic models for API
class CreateMasterTaskRequest(BaseModel):
    """Request to create a master task with dynamic planning."""
    task_description: str = Field(..., description="High-level task description")
    auto_decompose: bool = Field(True, description="Automatically decompose into subtasks")
    auto_spawn: bool = Field(True, description="Automatically spawn agents for subtasks")
    max_agents: int = Field(10, description="Maximum agents to spawn")
    enable_openai: bool = Field(True, description="Enable OpenAI integration")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    priority: str = Field("medium", description="Task priority: low, medium, high, critical")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentSpawnRequest(BaseModel):
    """Request to spawn a new agent."""
    task: str = Field(..., description="Task for the agent")
    role: str = Field("generalist", description="Agent role")
    parent_id: Optional[str] = Field(None, description="Parent agent ID")
    tools: List[str] = Field(default_factory=list, description="Tools to enable")
    auto_plan: bool = Field(True, description="Auto-generate execution plan")

class SystemStatusResponse(BaseModel):
    """System status response."""
    status: str
    timestamp: str
    active_agents: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_spawns: int
    hierarchy_depth: int
    resource_usage: Dict[str, Any]
    openai_integration: bool

class TaskProgressUpdate(BaseModel):
    """Real-time task progress update."""
    task_id: str
    status: str
    progress: float
    current_step: Optional[str]
    agent_id: Optional[str]
    message: Optional[str]
    timestamp: str


class MasterOrchestrator:
    """
    Master orchestrator that manages the entire agent system.
    
    This is the brain of the operation, coordinating:
    - Task planning and decomposition
    - Agent spawning and lifecycle
    - Resource allocation
    - OpenAI integration
    - Real-time monitoring
    """
    
    def __init__(self):
        """Initialize the master orchestrator."""
        self.task_planner = TaskPlanner()
        self.agent_spawner = AgentSpawner()
        self.hierarchy = AgentHierarchy()
        self.tool_registry = self._initialize_tools()
        self.openai_manager: Optional[OpenAIAgentsManager] = None
        
        # State management
        self.active_plans: Dict[str, TaskPlan] = {}
        self.agent_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_results: Dict[str, Any] = {}
        self.websocket_connections: Set[WebSocket] = set()
        
        # Metrics
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_spawns": 0,
            "start_time": time.time(),
        }
        
        # Initialize OpenAI if configured
        if MASTER_CONFIG["enable_openai_integration"]:
            self._initialize_openai()
    
    def _initialize_tools(self) -> ToolRegistry:
        """Initialize available tools."""
        registry = ToolRegistry()
        
        # Register all available tools
        if IMPORTS_AVAILABLE:
            for tool in get_all_tools():
                registry.register(tool)
        
        logger.info(f"Initialized {len(registry.tools) if hasattr(registry, 'tools') else 0} tools")
        return registry
    
    def _initialize_openai(self):
        """Initialize OpenAI integration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_manager = create_openai_agent_manager(
                    api_key=api_key,
                    model=MASTER_CONFIG["default_model"],
                    enable_tools=True,
                    enable_reasoning=True,
                )
                logger.info("OpenAI integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI integration: {e}")
                self.openai_manager = None
    
    async def create_master_task(self, request: CreateMasterTaskRequest) -> Dict[str, Any]:
        """
        Create a master task with dynamic planning and agent spawning.
        
        This is the main entry point for complex tasks that need to be
        broken down and executed by multiple agents.
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.metrics["total_tasks"] += 1
        
        try:
            # Create high-level task
            master_task = Task(
                id=task_id,
                name=request.task_description[:100],
                description=request.task_description,
                priority=TaskPriority[request.priority.upper()],
                metadata=request.metadata,
            )
            
            # Create execution plan
            if request.auto_decompose:
                plan = await self._create_dynamic_plan(master_task)
            else:
                plan = self.task_planner.create_plan(
                    goal=request.task_description,
                    initial_tasks=[master_task],
                    auto_decompose=False,
                )
            
            self.active_plans[plan.id] = plan
            
            # Spawn agents if requested
            spawned_agents = []
            if request.auto_spawn:
                spawned_agents = await self._spawn_agents_for_plan(
                    plan,
                    max_agents=request.max_agents,
                    enable_openai=request.enable_openai,
                    openai_api_key=request.openai_api_key,
                )
            
            # Start execution
            asyncio.create_task(self._execute_plan(plan))
            
            # Broadcast initial status
            await self._broadcast_progress({
                "task_id": task_id,
                "status": "created",
                "progress": 0.0,
                "message": f"Created master task with {len(plan.tasks)} subtasks",
            })
            
            return {
                "success": True,
                "task_id": task_id,
                "plan_id": plan.id,
                "total_subtasks": len(plan.tasks),
                "spawned_agents": len(spawned_agents),
                "estimated_duration": sum(
                    t.estimated_duration.total_seconds() for t in plan.tasks.values()
                ),
                "execution_order": self._get_execution_preview(plan),
            }
            
        except Exception as e:
            logger.error(f"Failed to create master task: {e}", exc_info=True)
            self.metrics["failed_tasks"] += 1
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
            }
    
    async def _create_dynamic_plan(self, task: Task) -> TaskPlan:
        """Create a dynamic execution plan using AI-powered decomposition."""
        logger.info(f"Creating dynamic plan for task: {task.name}")
        
        # Use task decomposer to break down the task
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose(task)
        
        # If we have OpenAI integration, enhance the plan
        if self.openai_manager and task.metadata.get("use_ai_planning", True):
            enhanced_subtasks = await self._enhance_plan_with_ai(task, subtasks)
            if enhanced_subtasks:
                subtasks = enhanced_subtasks
        
        # Create plan with dependencies
        plan = TaskPlan(
            name=f"Plan for: {task.name}",
            goal=task.description,
            description=f"Dynamic execution plan for: {task.description}",
        )
        
        # Add tasks with dependencies
        for i, subtask in enumerate(subtasks):
            plan.add_task(subtask)
            
            # Add sequential dependencies by default
            if i > 0:
                subtask.add_dependency(subtasks[i-1].id)
        
        # Validate and optimize the plan
        is_valid, errors = self.task_planner.validator.validate(plan)
        if not is_valid:
            logger.warning(f"Plan validation errors: {errors}")
        
        plan = self.task_planner.optimize_plan(plan)
        
        return plan
    
    async def _enhance_plan_with_ai(
        self,
        task: Task,
        initial_subtasks: List[Task]
    ) -> Optional[List[Task]]:
        """Enhance task decomposition using AI."""
        try:
            # Create a temporary agent for planning
            agent_id = f"planner_{uuid.uuid4().hex[:8]}"
            await self.openai_manager.create_agent(
                agent_id=agent_id,
                system_prompt="""You are an expert task planner. Break down complex tasks into 
                clear, actionable subtasks with proper dependencies. Consider resource requirements,
                parallel execution opportunities, and critical paths."""
            )
            
            # Get AI-enhanced plan
            prompt = f"""Task: {task.description}
            
Initial subtasks:
{chr(10).join(f"- {st.name}: {st.description}" for st in initial_subtasks)}

Please enhance this plan by:
1. Identifying missing steps
2. Optimizing task order
3. Identifying parallel execution opportunities
4. Estimating resource requirements

Provide a structured list of subtasks with clear dependencies."""
            
            response = await self.openai_manager.send_message(
                agent_id=agent_id,
                message=prompt
            )
            
            # Parse AI response and create enhanced subtasks
            # (Simplified parsing - in production, use structured output)
            enhanced_subtasks = initial_subtasks  # Fallback to initial
            
            # Clean up planning agent
            await self.openai_manager.delete_agent(agent_id)
            
            return enhanced_subtasks
            
        except Exception as e:
            logger.error(f"Failed to enhance plan with AI: {e}")
            return None
    
    async def _spawn_agents_for_plan(
        self,
        plan: TaskPlan,
        max_agents: int,
        enable_openai: bool,
        openai_api_key: Optional[str]
    ) -> List[str]:
        """Spawn agents for executing a plan."""
        spawned_agents = []
        
        # Get execution order
        execution_levels = self.task_planner.get_execution_order(plan)
        
        # Spawn coordinator agent
        coordinator_config = SpawnConfig(
            agent_config=AgentConfig(
                name=f"Coordinator_{plan.id[:8]}",
                role=AgentRole.COORDINATOR,
                description=f"Coordinating execution of: {plan.goal}",
            ),
            priority=10,
        )
        
        coordinator_result = await self.agent_spawner.spawn_agent(
            task=f"Coordinate: {plan.goal}",
            config=coordinator_config,
        )
        
        if coordinator_result.success:
            spawned_agents.append(coordinator_result.agent_id)
            self.metrics["total_spawns"] += 1
        
        # Spawn worker agents for tasks
        agents_spawned = 0
        for level_tasks in execution_levels:
            if agents_spawned >= max_agents:
                break
            
            for task in level_tasks:
                if agents_spawned >= max_agents:
                    break
                
                # Determine agent role based on task type
                role = self._determine_agent_role(task)
                
                # Create spawn configuration
                worker_config = SpawnConfig(
                    agent_config=AgentConfig(
                        name=f"Worker_{task.id[:8]}",
                        role=role,
                        description=f"Executing: {task.name}",
                        tools=self._determine_required_tools(task),
                    ),
                    parent_id=coordinator_result.agent_id if coordinator_result.success else None,
                    relationship=AgentRelationship.COORDINATOR_SUBORDINATE,
                    priority=task.priority.value,
                )
                
                # Spawn the agent
                result = await self.agent_spawner.spawn_agent(
                    task=task.description,
                    config=worker_config,
                )
                
                if result.success:
                    spawned_agents.append(result.agent_id)
                    self.agent_assignments[task.id] = result.agent_id
                    agents_spawned += 1
                    self.metrics["total_spawns"] += 1
                    
                    # Create OpenAI agent if enabled
                    if enable_openai and self.openai_manager:
                        await self._create_openai_agent_wrapper(
                            result.agent_id,
                            task,
                            openai_api_key
                        )
        
        logger.info(f"Spawned {len(spawned_agents)} agents for plan {plan.id}")
        return spawned_agents
    
    def _determine_agent_role(self, task: Task) -> AgentRole:
        """Determine the appropriate agent role for a task."""
        task_type_to_role = {
            "coding": AgentRole.EXECUTOR,
            "research": AgentRole.RESEARCHER,
            "analysis": AgentRole.ANALYZER,
            "testing": AgentRole.EXECUTOR,
            "deployment": AgentRole.SPECIALIST,
            "planning": AgentRole.PLANNER,
        }
        
        return task_type_to_role.get(task.task_type, AgentRole.GENERALIST)
    
    def _determine_required_tools(self, task: Task) -> List[str]:
        """Determine required tools for a task."""
        tools = []
        
        # Basic tool mapping based on task type
        if task.task_type == "coding":
            tools.extend(["python_repl", "file_reader", "file_writer"])
        elif task.task_type == "research":
            tools.extend(["web_search", "file_reader"])
        elif task.task_type == "analysis":
            tools.extend(["calculator", "python_repl"])
        
        # Add tools from task metadata
        if "required_tools" in task.metadata:
            tools.extend(task.metadata["required_tools"])
        
        return list(set(tools))  # Remove duplicates
    
    async def _create_openai_agent_wrapper(
        self,
        agent_id: str,
        task: Task,
        api_key: Optional[str]
    ):
        """Create OpenAI agent wrapper for enhanced capabilities."""
        if not self.openai_manager:
            return
        
        try:
            system_prompt = f"""You are an AI agent responsible for: {task.description}
            
Task Type: {task.task_type}
Priority: {task.priority.value}
Required Tools: {', '.join(self._determine_required_tools(task))}

Execute this task efficiently and report progress."""
            
            await self.openai_manager.create_agent(
                agent_id=f"openai_{agent_id}",
                system_prompt=system_prompt,
                metadata={
                    "task_id": task.id,
                    "llama_agent_id": agent_id,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create OpenAI wrapper for agent {agent_id}: {e}")
    
    async def _execute_plan(self, plan: TaskPlan):
        """Execute a task plan with dynamic coordination."""
        logger.info(f"Starting execution of plan: {plan.id}")
        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        
        try:
            while len(completed_tasks) + len(failed_tasks) < len(plan.tasks):
                # Get ready tasks
                ready_tasks = plan.get_ready_tasks(completed_tasks)
                
                if not ready_tasks:
                    # Check for deadlock
                    if len(completed_tasks) + len(failed_tasks) < len(plan.tasks):
                        logger.error(f"Deadlock detected in plan {plan.id}")
                        break
                
                # Execute ready tasks in parallel
                execution_tasks = []
                for task in ready_tasks:
                    if task.id not in self.agent_assignments:
                        # Spawn agent on-demand if needed
                        if MASTER_CONFIG["enable_auto_spawn"]:
                            await self._spawn_agent_for_task(task, plan)
                    
                    if task.id in self.agent_assignments:
                        execution_tasks.append(self._execute_single_task(task, plan))
                
                # Wait for tasks to complete
                if execution_tasks:
                    results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                    
                    for task, result in zip(ready_tasks, results):
                        if isinstance(result, Exception):
                            failed_tasks.add(task.id)
                            task.status = TaskStatus.FAILED
                            task.error = str(result)
                        else:
                            completed_tasks.add(task.id)
                            task.status = TaskStatus.COMPLETED
                            task.result = result
                            self.task_results[task.id] = result
                
                # Update progress
                progress = len(completed_tasks) / len(plan.tasks) * 100
                await self._broadcast_progress({
                    "task_id": plan.id,
                    "status": "executing",
                    "progress": progress,
                    "completed": len(completed_tasks),
                    "failed": len(failed_tasks),
                    "total": len(plan.tasks),
                })
            
            # Plan execution complete
            self.metrics["completed_tasks"] += len(completed_tasks)
            self.metrics["failed_tasks"] += len(failed_tasks)
            
            final_status = "completed" if len(failed_tasks) == 0 else "completed_with_errors"
            await self._broadcast_progress({
                "task_id": plan.id,
                "status": final_status,
                "progress": 100.0,
                "completed": len(completed_tasks),
                "failed": len(failed_tasks),
                "message": f"Plan execution {final_status}",
            })
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            await self._broadcast_progress({
                "task_id": plan.id,
                "status": "failed",
                "error": str(e),
            })
    
    async def _spawn_agent_for_task(self, task: Task, plan: TaskPlan):
        """Spawn an agent for a specific task on-demand."""
        config = SpawnConfig(
            agent_config=AgentConfig(
                name=f"OnDemand_{task.id[:8]}",
                role=self._determine_agent_role(task),
                description=f"On-demand execution: {task.name}",
                tools=self._determine_required_tools(task),
            ),
            priority=task.priority.value,
        )
        
        result = await self.agent_spawner.spawn_agent(
            task=task.description,
            config=config,
        )
        
        if result.success:
            self.agent_assignments[task.id] = result.agent_id
            self.metrics["total_spawns"] += 1
    
    async def _execute_single_task(self, task: Task, plan: TaskPlan) -> Any:
        """Execute a single task using assigned agent."""
        agent_id = self.agent_assignments.get(task.id)
        if not agent_id:
            raise ValueError(f"No agent assigned to task {task.id}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Broadcast task start
        await self._broadcast_progress({
            "task_id": task.id,
            "status": "started",
            "agent_id": agent_id,
            "message": f"Starting: {task.name}",
        })
        
        try:
            # Get agent from hierarchy
            node = self.agent_spawner.hierarchy.nodes.get(agent_id)
            if not node:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = node.agent
            
            # Create task input
            task_input = TaskInput(
                id=task.id,
                task=task.description,
                context={
                    "plan_id": plan.id,
                    "task_type": task.task_type,
                    "priority": task.priority.value,
                    "dependencies": [d.task_id for d in task.dependencies],
                }
            )
            
            # Execute task
            result = await agent.execute_task(task_input)
            
            # Update task with results
            task.completed_at = datetime.now()
            task.actual_duration = task.completed_at - task.started_at
            
            # Broadcast completion
            await self._broadcast_progress({
                "task_id": task.id,
                "status": "completed",
                "agent_id": agent_id,
                "message": f"Completed: {task.name}",
                "duration": task.actual_duration.total_seconds(),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")
            task.error = str(e)
            
            await self._broadcast_progress({
                "task_id": task.id,
                "status": "failed",
                "agent_id": agent_id,
                "error": str(e),
            })
            
            raise
    
    def _get_execution_preview(self, plan: TaskPlan) -> List[Dict[str, Any]]:
        """Get execution order preview for a plan."""
        try:
            levels = self.task_planner.get_execution_order(plan)
            preview = []
            
            for i, level_tasks in enumerate(levels):
                preview.append({
                    "level": i + 1,
                    "parallel_tasks": [
                        {
                            "id": t.id,
                            "name": t.name,
                            "type": t.task_type,
                            "priority": t.priority.value,
                        }
                        for t in level_tasks
                    ]
                })
            
            return preview
        except Exception as e:
            logger.error(f"Failed to get execution preview: {e}")
            return []
    
    async def _broadcast_progress(self, update: Dict[str, Any]):
        """Broadcast progress updates to all websocket connections."""
        if not self.websocket_connections:
            return
        
        # Add timestamp
        update["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Convert to TaskProgressUpdate format
        progress_update = TaskProgressUpdate(
            task_id=update.get("task_id", ""),
            status=update.get("status", "unknown"),
            progress=update.get("progress", 0.0),
            current_step=update.get("message"),
            agent_id=update.get("agent_id"),
            message=update.get("error"),
            timestamp=update["timestamp"],
        )
        
        # Broadcast to all connections
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(progress_update.dict())
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        self.websocket_connections -= disconnected
    
    async def get_system_status(self) -> SystemStatusResponse:
        """Get comprehensive system status."""
        hierarchy_stats = self.agent_spawner.hierarchy.get_hierarchy_stats()
        resource_usage = self.agent_spawner._resource_monitor.get_usage()
        
        # Count tasks by status
        running_tasks = sum(
            1 for plan in self.active_plans.values()
            for task in plan.tasks.values()
            if task.status == TaskStatus.RUNNING
        )
        
        completed_tasks = sum(
            1 for plan in self.active_plans.values()
            for task in plan.tasks.values()
            if task.status == TaskStatus.COMPLETED
        )
        
        failed_tasks = sum(
            1 for plan in self.active_plans.values()
            for task in plan.tasks.values()
            if task.status == TaskStatus.FAILED
        )
        
        return SystemStatusResponse(
            status="operational",
            timestamp=datetime.now(timezone.utc).isoformat(),
            active_agents=hierarchy_stats["active_agents"],
            running_tasks=running_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_spawns=self.metrics["total_spawns"],
            hierarchy_depth=hierarchy_stats["max_depth"],
            resource_usage=resource_usage,
            openai_integration=self.openai_manager is not None,
        )
    
    def get_hierarchy_visualization(self) -> str:
        """Get text visualization of agent hierarchy."""
        tree = Tree("Featured [bold yellow]Agent Hierarchy[/bold yellow]")
        
        def add_node(parent_tree, node_id: str, depth: int = 0):
            if depth > 10:  # Prevent infinite recursion
                return
            
            node = self.agent_spawner.hierarchy.nodes.get(node_id)
            if not node:
                return
            
            # Format node info
            status = "🟢" if node.is_active else ""
            role = node.agent.config.role.value if hasattr(node.agent, 'config') else "unknown"
            
            node_text = f"{status} {node.agent_id[:8]} [{role}]"
            if node_id in self.agent_assignments.values():
                task_id = next(tid for tid, aid in self.agent_assignments.items() if aid == node_id)
                node_text += f" LIST: {task_id[:8]}"
            
            # Add to tree
            subtree = parent_tree.add(node_text)
            
            # Add children
            for child_id in node.children:
                add_node(subtree, child_id, depth + 1)
        
        # Add root agents
        for root_id in self.agent_spawner.hierarchy.root_agents:
            add_node(tree, root_id)
        
        return tree


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    console.print("[bold green]Starting LlamaAgent Master Program...[/bold green]")
    
    # Initialize orchestrator
    app.state.orchestrator = MasterOrchestrator()
    
    # Show startup banner
    banner = Panel.fit(
        """[bold cyan]LlamaAgent Master Program[/bold cyan]
        
Complete AI Agent System with:
• Dynamic Task Planning & Decomposition
• Intelligent Subagent Spawning  
• OpenAI Agents SDK Integration
• Hierarchical Agent Management
• Real-time Progress Monitoring

[dim]By Nik Jois <nikjois@llamasearch.ai>[/dim]""",
        border_style="cyan"
    )
    console.print(banner)
    
    yield
    
    # Cleanup
    console.print("[bold red]Shutting down LlamaAgent Master Program...[/bold red]")
    
    # Terminate all agents
    for agent_id in list(app.state.orchestrator.agent_spawner.hierarchy.nodes.keys()):
        await app.state.orchestrator.agent_spawner.terminate_agent(agent_id)

app = FastAPI(
    title="LlamaAgent Master Program API",
    description="Complete AI Agent System with Dynamic Planning and Spawning",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "LlamaAgent Master Program",
        "version": "2.0.0",
        "description": "Complete AI Agent System",
        "features": [
            "Dynamic Task Planning",
            "Intelligent Agent Spawning",
            "OpenAI Integration",
            "Hierarchical Management",
            "Real-time Monitoring"
        ],
        "author": "Nik Jois <nikjois@llamasearch.ai>",
        "endpoints": {
            "create_task": "/api/v1/tasks",
            "spawn_agent": "/api/v1/agents/spawn",
            "system_status": "/api/v1/status",
            "hierarchy": "/api/v1/hierarchy",
            "websocket": "/ws",
            "docs": "/docs"
        }
    }

@app.post("/api/v1/tasks")
async def create_task(request: CreateMasterTaskRequest):
    """Create a master task with dynamic planning."""
    orchestrator: MasterOrchestrator = app.state.orchestrator
    return await orchestrator.create_master_task(request)

@app.post("/api/v1/agents/spawn")
async def spawn_agent(request: AgentSpawnRequest):
    """Manually spawn a new agent."""
    orchestrator: MasterOrchestrator = app.state.orchestrator
    
    config = SpawnConfig(
        agent_config=AgentConfig(
            name=f"Manual_{uuid.uuid4().hex[:8]}",
            role=AgentRole[request.role.upper()],
            tools=request.tools,
        ),
        parent_id=request.parent_id,
    )
    
    result = await orchestrator.agent_spawner.spawn_agent(
        task=request.task,
        config=config,
    )
    
    return {
        "success": result.success,
        "agent_id": result.agent_id,
        "error": result.error,
        "spawn_time": result.spawn_time,
    }

@app.get("/api/v1/status", response_model=SystemStatusResponse)
async def get_status():
    """Get comprehensive system status."""
    orchestrator: MasterOrchestrator = app.state.orchestrator
    return await orchestrator.get_system_status()

@app.get("/api/v1/hierarchy")
async def get_hierarchy():
    """Get agent hierarchy visualization."""
    orchestrator: MasterOrchestrator = app.state.orchestrator
    
    hierarchy_stats = orchestrator.agent_spawner.hierarchy.get_hierarchy_stats()
    
    # Build hierarchy tree structure
    def build_tree(node_id: str) -> Dict[str, Any]:
        node = orchestrator.agent_spawner.hierarchy.nodes.get(node_id)
        if not node:
            return {}
        
        return {
            "id": node.agent_id,
            "name": node.agent.config.name if hasattr(node.agent, 'config') else "Unknown",
            "role": node.agent.config.role.value if hasattr(node.agent, 'config') else "unknown",
            "active": node.is_active,
            "children": [build_tree(child_id) for child_id in node.children],
            "metrics": {
                "memory_mb": node.memory_usage_mb,
                "api_calls": node.api_calls_made,
                "execution_time": node.total_execution_time,
            }
        }
    
    hierarchy_tree = [
        build_tree(root_id) 
        for root_id in orchestrator.agent_spawner.hierarchy.root_agents
    ]
    
    return {
        "hierarchy": hierarchy_tree,
        "stats": hierarchy_stats,
        "visualization": orchestrator.get_hierarchy_visualization(),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    orchestrator: MasterOrchestrator = app.state.orchestrator
    orchestrator.websocket_connections.add(websocket)
    
    try:
        # Send initial status
        status = await orchestrator.get_system_status()
        await websocket.send_json({"type": "status", "data": status.dict()})
        
        # Keep connection alive
        while True:
            # Wait for messages (ping/pong)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        orchestrator.websocket_connections.discard(websocket)


# CLI Interface
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """LlamaAgent Master Program CLI - Complete AI Agent System."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def server(host, port, reload):
    """Start the Master Program API server."""
    console.print(f"[bold green]Starting server on {host}:{port}[/bold green]")
    uvicorn.run(
        "llamaagent_master_program:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@cli.command()
@click.argument('task', required=True)
@click.option('--auto-spawn', is_flag=True, default=True, help='Auto-spawn agents')
@click.option('--max-agents', default=10, help='Maximum agents to spawn')
@click.option('--openai-key', help='OpenAI API key')
def execute(task, auto_spawn, max_agents, openai_key):
    """Execute a task with dynamic planning and agent spawning."""
    async def run_task():
        orchestrator = MasterOrchestrator()
        
        # Set OpenAI key if provided
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            orchestrator._initialize_openai()
        
        # Create task request
        request = CreateMasterTaskRequest(
            task_description=task,
            auto_decompose=True,
            auto_spawn=auto_spawn,
            max_agents=max_agents,
            enable_openai=bool(openai_key),
            openai_api_key=openai_key,
        )
        
        # Execute with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task_progress = progress.add_task("Creating master task...", total=100)
            
            # Create task
            result = await orchestrator.create_master_task(request)
            
            if not result["success"]:
                console.print(f"[red]Failed to create task: {result.get('error')}[/red]")
                return
            
            progress.update(task_progress, description="Executing plan...", completed=10)
            
            # Monitor execution
            plan_id = result["plan_id"]
            while plan_id in orchestrator.active_plans:
                await asyncio.sleep(1)
                
                # Get progress
                plan = orchestrator.active_plans[plan_id]
                completed = sum(1 for t in plan.tasks.values() if t.status == TaskStatus.COMPLETED)
                total = len(plan.tasks)
                
                progress.update(
                    task_progress,
                    description=f"Executing tasks ({completed}/{total})...",
                    completed=10 + (completed / total * 80)
                )
                
                # Check if all done
                if completed + sum(1 for t in plan.tasks.values() if t.status == TaskStatus.FAILED) >= total:
                    break
            
            progress.update(task_progress, description="Complete!", completed=100)
        
        # Show results
        console.print("\n[bold]Execution Summary:[/bold]")
        
        status = await orchestrator.get_system_status()
        
        table = Table(title="Task Execution Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Subtasks", str(result["total_subtasks"]))
        table.add_row("Spawned Agents", str(result["spawned_agents"]))
        table.add_row("Completed Tasks", str(status.completed_tasks))
        table.add_row("Failed Tasks", str(status.failed_tasks))
        table.add_row("Total Spawns", str(status.total_spawns))
        
        console.print(table)
        
        # Show hierarchy
        console.print("\n[bold]Agent Hierarchy:[/bold]")
        console.print(orchestrator.get_hierarchy_visualization())
    
    asyncio.run(run_task())

@cli.command()
def demo():
    """Run an interactive demonstration."""
    async def run_demo():
        console.print("[bold cyan]LlamaAgent Master Program Demo[/bold cyan]\n")
        
        orchestrator = MasterOrchestrator()
        
        demo_tasks = [
            "Build a web scraper that extracts product information from e-commerce sites",
            "Analyze customer sentiment from social media data and generate insights",
            "Create a machine learning pipeline for predictive maintenance",
        ]
        
        for i, task_desc in enumerate(demo_tasks, 1):
            console.print(f"\n[bold]Demo Task {i}:[/bold] {task_desc}")
            
            request = CreateMasterTaskRequest(
                task_description=task_desc,
                auto_decompose=True,
                auto_spawn=True,
                max_agents=5,
                priority="high",
            )
            
            result = await orchestrator.create_master_task(request)
            
            if result["success"]:
                console.print(f"PASS Created plan with {result['total_subtasks']} subtasks")
                console.print(f"PASS Spawned {result['spawned_agents']} agents")
                
                # Show execution preview
                console.print("\n[dim]Execution Preview:[/dim]")
                for level in result["execution_order"][:3]:  # Show first 3 levels
                    console.print(f"  Level {level['level']}: {len(level['parallel_tasks'])} parallel tasks")
            else:
                console.print(f"FAIL Failed: {result.get('error')}")
        
        # Final status
        status = await orchestrator.get_system_status()
        console.print(f"\n[bold]Final System Status:[/bold]")
        console.print(f"• Active Agents: {status.active_agents}")
        console.print(f"• Total Spawns: {status.total_spawns}")
        console.print(f"• Hierarchy Depth: {status.hierarchy_depth}")
    
    asyncio.run(run_demo())

@cli.command()
def monitor():
    """Start real-time system monitor."""
    async def run_monitor():
        orchestrator = MasterOrchestrator()
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(
            Panel("[bold cyan]LlamaAgent Master Program Monitor[/bold cyan]", border_style="cyan")
        )
        
        with Live(layout, console=console, refresh_per_second=1) as live:
            while True:
                try:
                    # Get status
                    status = await orchestrator.get_system_status()
                    
                    # Update main panel
                    status_table = Table(show_header=False, box=None)
                    status_table.add_column("Metric", style="cyan")
                    status_table.add_column("Value", style="green")
                    
                    status_table.add_row("Active Agents", str(status.active_agents))
                    status_table.add_row("Running Tasks", str(status.running_tasks))
                    status_table.add_row("Completed Tasks", str(status.completed_tasks))
                    status_table.add_row("Failed Tasks", str(status.failed_tasks))
                    status_table.add_row("Total Spawns", str(status.total_spawns))
                    
                    # Resource usage
                    memory_usage = status.resource_usage.get("memory", {})
                    api_usage = status.resource_usage.get("api_calls", {})
                    
                    status_table.add_row("Memory Usage", f"{memory_usage.get('percentage', 0):.1f}%")
                    status_table.add_row("API Calls", f"{api_usage.get('percentage', 0):.1f}%")
                    
                    layout["main"].update(Panel(status_table, title="System Status"))
                    
                    # Update footer
                    layout["footer"].update(
                        Panel(
                            f"[dim]Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to exit[/dim]",
                            border_style="dim"
                        )
                    )
                    
                    await asyncio.sleep(1)
                    
                except KeyboardInterrupt:
                    break
    
    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped[/yellow]")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - show help
        cli.main(['--help'])
    else:
        cli()