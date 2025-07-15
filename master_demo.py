#!/usr/bin/env python3
"""
LlamaAgent Master Program - Comprehensive Demo
Showcases all features of the complete AI agent system
"""

import asyncio
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest

console = Console()


async def demo_basic_task():
    """Demo: Basic task execution."""
    console.print("\n[bold cyan]Demo 1: Basic Task Execution[/bold cyan]")
    console.print("Creating a simple task without decomposition...\n")
    
    orchestrator = MasterOrchestrator()
    
    request = CreateMasterTaskRequest(
        task_description="Calculate the factorial of 10",
        auto_decompose=False,
        auto_spawn=False,
        priority="low"
    )
    
    result = await orchestrator.create_master_task(request)
    
    if result["success"]:
        console.print(f"PASS Task created successfully!")
        console.print(f"   Task ID: {result['task_id']}")
        console.print(f"   Subtasks: {result['total_subtasks']}")
    else:
        console.print(f"FAIL Failed: {result.get('error')}")


async def demo_complex_decomposition():
    """Demo: Complex task with automatic decomposition."""
    console.print("\n[bold cyan]Demo 2: Complex Task Decomposition[/bold cyan]")
    console.print("Creating a complex task that will be automatically decomposed...\n")
    
    orchestrator = MasterOrchestrator()
    
    request = CreateMasterTaskRequest(
        task_description="""Build a complete web application with:
        1. User authentication system
        2. Database integration
        3. REST API endpoints
        4. Frontend interface
        5. Deployment configuration""",
        auto_decompose=True,
        auto_spawn=False,
        priority="high"
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating and decomposing task...", total=100)
        
        result = await orchestrator.create_master_task(request)
        progress.update(task, completed=100)
    
    if result["success"]:
        console.print(f"\nPASS Complex task decomposed successfully!")
        console.print(f"   Original task broken into {result['total_subtasks']} subtasks")
        
        # Show execution order
        console.print("\n[bold]Execution Plan:[/bold]")
        for level in result.get("execution_order", [])[:3]:  # Show first 3 levels
            console.print(f"\n   Level {level['level']}:")
            for task in level['parallel_tasks']:
                console.print(f"     • {task['name']} [{task['priority']}]")


async def demo_agent_spawning():
    """Demo: Dynamic agent spawning and hierarchy."""
    console.print("\n[bold cyan]Demo 3: Dynamic Agent Spawning[/bold cyan]")
    console.print("Creating a task with automatic agent spawning...\n")
    
    orchestrator = MasterOrchestrator()
    
    request = CreateMasterTaskRequest(
        task_description="Analyze sales data from multiple sources and generate comprehensive report",
        auto_decompose=True,
        auto_spawn=True,
        max_agents=5,
        priority="high",
        metadata={
            "require_specialized_agents": True
        }
    )
    
    result = await orchestrator.create_master_task(request)
    
    if result["success"]:
        console.print(f"PASS Task created with agent spawning!")
        console.print(f"   Spawned {result['spawned_agents']} specialized agents")
        
        # Wait a bit for agents to initialize
        await asyncio.sleep(1)
        
        # Show agent hierarchy
        console.print("\n[bold]Agent Hierarchy:[/bold]")
        hierarchy_viz = orchestrator.get_hierarchy_visualization()
        console.print(hierarchy_viz)


async def demo_real_time_monitoring():
    """Demo: Real-time task monitoring."""
    console.print("\n[bold cyan]Demo 4: Real-time Task Monitoring[/bold cyan]")
    console.print("Executing a task with real-time progress monitoring...\n")
    
    orchestrator = MasterOrchestrator()
    
    # Create a multi-step task
    request = CreateMasterTaskRequest(
        task_description="Process dataset: load data, clean it, analyze patterns, generate visualizations",
        auto_decompose=True,
        auto_spawn=True,
        max_agents=4,
        priority="medium"
    )
    
    result = await orchestrator.create_master_task(request)
    
    if result["success"]:
        plan_id = result["plan_id"]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing tasks...", total=result["total_subtasks"])
            
            # Monitor execution
            start_time = time.time()
            while plan_id in orchestrator.active_plans and time.time() - start_time < 10:
                await asyncio.sleep(0.5)
                
                # Get progress
                plan = orchestrator.active_plans.get(plan_id)
                if plan:
                    completed = sum(1 for t in plan.tasks.values() 
                                  if t.status.value in ["completed", "failed"])
                    progress.update(task, completed=completed)
                    
                    if completed >= result["total_subtasks"]:
                        break
        
        # Show final status
        status = await orchestrator.get_system_status()
        
        table = Table(title="Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Tasks", str(status.completed_tasks))
        table.add_row("Failed Tasks", str(status.failed_tasks))
        table.add_row("Active Agents", str(status.active_agents))
        table.add_row("Resource Usage", f"{status.resource_usage['memory']['percentage']:.1f}%")
        
        console.print(table)


async def demo_team_coordination():
    """Demo: Multi-agent team coordination."""
    console.print("\n[bold cyan]Demo 5: Multi-Agent Team Coordination[/bold cyan]")
    console.print("Creating a coordinated team of agents for a complex project...\n")
    
    orchestrator = MasterOrchestrator()
    
    # Create a project that requires multiple specialized agents
    request = CreateMasterTaskRequest(
        task_description="""Develop a machine learning solution:
        1. Research state-of-the-art models
        2. Prepare and preprocess the dataset
        3. Train multiple model architectures
        4. Evaluate and compare performance
        5. Deploy the best model
        6. Create documentation and reports""",
        auto_decompose=True,
        auto_spawn=True,
        max_agents=10,
        priority="critical",
        metadata={
            "project_type": "machine_learning",
            "require_roles": ["coordinator", "researcher", "analyzer", "executor"],
            "enable_parallel_execution": True
        }
    )
    
    result = await orchestrator.create_master_task(request)
    
    if result["success"]:
        console.print(f"PASS Created ML project with {result['spawned_agents']} agents")
        
        # Show team structure
        await asyncio.sleep(1)
        
        hierarchy = orchestrator.agent_spawner.hierarchy
        stats = hierarchy.get_hierarchy_stats()
        
        console.print(f"\n[bold]Team Structure:[/bold]")
        console.print(f"  • Total Agents: {stats['total_agents']}")
        console.print(f"  • Hierarchy Depth: {stats['max_depth']}")
        console.print(f"  • Active Workers: {stats['active_agents']}")
        
        # Show relationships
        console.print(f"\n[bold]Agent Relationships:[/bold]")
        for rel_type, count in stats['relationships'].items():
            if count > 0:
                console.print(f"  • {rel_type}: {count}")


async def main():
    """Run all demos."""
    banner = Panel.fit(
        """[bold cyan]LlamaAgent Master Program - Feature Showcase[/bold cyan]
        
This demo showcases the complete capabilities of the AI agent system:
• Dynamic task planning and decomposition
• Intelligent agent spawning
• Hierarchical team coordination
• Real-time monitoring
• Resource management

[dim]Press Ctrl+C to skip any demo[/dim]""",
        border_style="cyan"
    )
    console.print(banner)
    
    demos = [
        ("Basic Task Execution", demo_basic_task),
        ("Complex Task Decomposition", demo_complex_decomposition),
        ("Dynamic Agent Spawning", demo_agent_spawning),
        ("Real-time Monitoring", demo_real_time_monitoring),
        ("Team Coordination", demo_team_coordination),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            console.print(f"\n{'='*60}")
            console.print(f"[bold green]Demo {i}/{len(demos)}: {name}[/bold green]")
            console.print(f"{'='*60}")
            
            await demo_func()
            
            if i < len(demos):
                console.print("\n[dim]Press Enter to continue to next demo...[/dim]")
                input()
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo skipped[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]Demo error: {e}[/red]")
    
    # Final summary
    console.print("\n" + "="*60)
    console.print("[bold green]Demo Complete![/bold green]")
    console.print("="*60)
    
    console.print("""
The LlamaAgent Master Program is now ready for production use!

To start using the system:

1. Start the API server:
   python3 llamaagent_master_program.py server

2. Execute tasks via CLI:
   python3 llamaagent_master_program.py execute "Your task"

3. Use the API:
   POST http://localhost:8000/api/v1/tasks

For more information, see LLAMAAGENT_MASTER_README.md
""")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")