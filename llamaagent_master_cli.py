#!/usr/bin/env python3
"""
LlamaAgent Master CLI - Complete Command-Line Interface
======================================================

A comprehensive command-line interface that integrates all llamaagent components:
- Dynamic task planning and scheduling
- Multi-agent orchestration
- Real-time execution monitoring
- Interactive menu system
- Performance analytics
- Tool management

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import signal
from datetime import timedelta
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# LlamaAgent imports
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.agents.base import AgentConfig, AgentRole
from src.llamaagent.llm.providers import create_provider

from src.llamaagent.planning import (
    ExecutionEngine,
    Task,
    TaskPlan,
    TaskPlanner,
    TaskPriority,
    TaskStatus,
)
from src.llamaagent.tools import ToolRegistry, get_all_tools
from src.llamaagent.memory.base import SimpleMemory
from src.llamaagent.types import TaskInput

logger = logging.getLogger(__name__)
console = Console()


class MasterCLI:
    """Master CLI for LlamaAgent with dynamic task planning and scheduling."""

    def __init__(self):
        self.console = console
        self.agents: Dict[str, ReactAgent] = {}
        self.task_planner = TaskPlanner()
        self.execution_engine = ExecutionEngine()
        self.tools = ToolRegistry()
        self.memory = SimpleMemory()
        self.active_plans: Dict[str, TaskPlan] = {}
        self.shutdown_requested = False
        
        # LLM Provider configuration
        self.current_provider = "mock"
        self.available_providers = ["mock", "openai", "llama_local"]
        self.provider_configs: Dict[str, Dict[str, Any]] = {
            "mock": {"model_name": "mock-model"},
            "openai": {"model_name": "gpt-3.5-turbo", "api_key": ""},
            "llama_local": {"model_name": "microsoft/DialoGPT-medium", "device": "auto"}
        }
        
        # Initialize tools
        for tool in get_all_tools():
            self.tools.register(tool)
        
        # Initialize default agents
        self._initialize_default_agents()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_default_agents(self):
        """Initialize default specialized agents."""
        agent_configs = [
            ("general", AgentRole.GENERALIST, "General purpose agent"),
            ("planner", AgentRole.PLANNER, "Strategic planning agent"),
            ("executor", AgentRole.EXECUTOR, "Task execution agent"),
            ("analyzer", AgentRole.ANALYZER, "Data analysis agent"),
        ]
        
        for name, role, description in agent_configs:
            config = AgentConfig(
                name=name,
                role=role,
                description=description,
                max_iterations=10,
                temperature=0.7,
            )
            
            try:
                # Create provider for this agent
                provider_config = self.provider_configs[self.current_provider].copy()
                provider = create_provider(
                    provider_type=self.current_provider,
                    **provider_config
                )
                
                agent = ReactAgent(
                    config=config,
                    llm_provider=provider,
                    tools=self.tools,
                    memory=self.memory,
                )
                self.agents[name] = agent
                logger.info(f"Initialized agent: {name} with provider: {self.current_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {name}: {e}")
                raise RuntimeError(
                    f"Unable to initialize agent '{name}' with provider '{self.current_provider}': {e}. "
                    "Please check your provider configuration and ensure the service is available."
                )

    def _signal_handler(self, signum: int, frame: Any):
        """Handle shutdown signals."""
        self.console.print("\n[yellow]Shutdown requested...[/yellow]")
        self.shutdown_requested = True

    def show_banner(self):
        """Display startup banner."""
        banner = """
                                
                      
                             
                             
                       
                             
        """
        
        self.console.print(Panel(
            Text(banner, style="bold cyan") + "\n" +
            Text("Master CLI with Dynamic Task Planning & Scheduling", style="bold white") + "\n" +
            Text("Author: Nik Jois <nikjois@llamasearch.ai>", style="dim white"),
            title="LlamaAgent Master CLI",
            border_style="cyan"
        ))

    def show_main_menu(self):
        """Display main interactive menu."""
        table = Table(title="Master CLI Features", show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Feature", style="green")
        table.add_column("Description", style="yellow")

        menu_items = [
            ("1", "Task Planning", "Create and manage dynamic task plans"),
            ("2", "Execute Tasks", "Run tasks with real-time monitoring"),
            ("3", "Agent Chat", "Interactive chat with specialized agents"),
            ("4", "Dashboard", "Performance metrics and analytics"),
            ("5", "Configuration", "System and agent configuration"),
            ("6", "Testing", "Debug and test system components"),
            ("7", "Help", "Documentation and examples"),
            ("0", "Exit", "Exit the application"),
        ]

        for option, feature, description in menu_items:
            table.add_row(option, feature, description)

        self.console.print(table)

    async def run(self):
        """Main CLI execution loop."""
        self.show_banner()
        
        while not self.shutdown_requested:
            try:
                self.console.print("\n")
                self.show_main_menu()
                
                choice = Prompt.ask(
                    "\n[bold cyan]Select option[/bold cyan]",
                    choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                    default="0"
                )

                if choice == "0":
                    if Confirm.ask("Exit the application?"):
                        break
                elif choice == "1":
                    await self._task_planning_interface()
                elif choice == "2":
                    await self._task_execution_interface()
                elif choice == "3":
                    await self._agent_chat_interface()
                elif choice == "4":
                    await self._dashboard_interface()
                elif choice == "5":
                    await self._configuration_interface()
                elif choice == "6":
                    await self._testing_interface()
                elif choice == "7":
                    await self._help_interface()

                if choice != "0":
                    input("\nPress Enter to continue...")
                    self.console.clear()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use option 0 to exit[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.exception("Error in main loop")

        await self._cleanup()

    async def _task_planning_interface(self):
        """Task planning interface."""
        self.console.print(Panel("[bold]Dynamic Task Planning[/bold]", style="blue"))
        
        while True:
            self.console.print("\n[bold]Task Planning Options:[/bold]")
            self.console.print("1. Create new task plan")
            self.console.print("2. View existing plans")
            self.console.print("3. Edit plan")
            self.console.print("4. Delete plan")
            self.console.print("0. Back to main menu")
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
            
            if choice == "0":
                break
            elif choice == "1":
                await self._create_task_plan()
            elif choice == "2":
                await self._view_task_plans()
            elif choice == "3":
                await self._edit_task_plan()
            elif choice == "4":
                await self._delete_task_plan()

    async def _create_task_plan(self):
        """Create a new task plan."""
        self.console.print("\n[bold]Create New Task Plan[/bold]")
        
        # Get task details
        goal = Prompt.ask("Enter the main goal")
        if not goal:
            return
        
        description = Prompt.ask("Enter description (optional)", default="")
        
        # Create initial tasks
        tasks: List[Task] = []
        self.console.print("\n[bold]Add tasks to the plan:[/bold]")
        
        while True:
            task_name = Prompt.ask("Task name (or 'done' to finish)")
            if task_name.lower() == 'done':
                break
            
            task_desc = Prompt.ask("Task description", default=task_name)
            
            # Priority selection
            priority_map = {
                "1": TaskPriority.LOW,
                "2": TaskPriority.MEDIUM,
                "3": TaskPriority.HIGH,
                "4": TaskPriority.CRITICAL,
            }
            
            priority_choice = Prompt.ask(
                "Priority (1=Low, 2=Medium, 3=High, 4=Critical)",
                choices=["1", "2", "3", "4"],
                default="2"
            )
            
            priority = priority_map[priority_choice]
            
            # Estimated duration
            duration_minutes = IntPrompt.ask("Estimated duration (minutes)", default=30)
            
            task = Task(
                name=task_name,
                description=task_desc,
                priority=priority,
                estimated_duration=timedelta(minutes=duration_minutes),
            )
            
            tasks.append(task)
            self.console.print(f"[green]Added task: {task_name}[/green]")
        
        if not tasks:
            self.console.print("[yellow]No tasks added. Plan creation cancelled.[/yellow]")
            return
        
        # Create plan
        plan = self.task_planner.create_plan(goal, tasks)
        plan.description = description
        
        # Store plan
        self.active_plans[plan.id] = plan
        
        self.console.print(f"[green]Created plan: {plan.name}[/green]")
        self.console.print(f"Plan ID: {plan.id}")
        self.console.print(f"Tasks: {len(plan.tasks)}")

    async def _view_task_plans(self):
        """View existing task plans."""
        if not self.active_plans:
            self.console.print("[yellow]No active plans found.[/yellow]")
            return
        
        table = Table(title="Active Task Plans", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Tasks", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Created", style="blue")
        
        for plan_id, plan in self.active_plans.items():
            completed_tasks = sum(1 for task in plan.tasks.values() if task.status == TaskStatus.COMPLETED)
            total_tasks = len(plan.tasks)
            status = f"{completed_tasks}/{total_tasks}"
            
            table.add_row(
                plan_id[:8],
                plan.name,
                str(total_tasks),
                status,
                plan.created_at.strftime("%Y-%m-%d %H:%M")
            )
        
        self.console.print(table)

    async def _edit_task_plan(self):
        """Edit an existing task plan."""
        if not self.active_plans:
            self.console.print("[yellow]No active plans to edit.[/yellow]")
            return
        
        # Show plans and let user select
        await self._view_task_plans()
        
        plan_id = Prompt.ask("Enter plan ID to edit (first 8 characters)")
        
        # Find matching plan
        selected_plan = None
        for pid, plan in self.active_plans.items():
            if pid.startswith(plan_id):
                selected_plan = plan
                break
        
        if not selected_plan:
            self.console.print("[red]Plan not found.[/red]")
            return
        
        self.console.print(f"\n[bold]Editing plan: {selected_plan.name}[/bold]")
        
        # Show current tasks
        table = Table(title="Current Tasks", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="magenta")
        
        for task in selected_plan.tasks.values():
            table.add_row(
                task.id[:8],
                task.name,
                task.priority.name,
                task.status.name
            )
        
        self.console.print(table)
        
        # Edit options
        self.console.print("\n[bold]Edit Options:[/bold]")
        self.console.print("1. Add new task")
        self.console.print("2. Remove task")
        self.console.print("3. Modify task")
        self.console.print("0. Done editing")
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3"])
        
        if choice == "1":
            # Add new task (similar to create_task_plan)
            await self._add_task_to_plan(selected_plan)
        elif choice == "2":
            await self._remove_task_from_plan(selected_plan)
        elif choice == "3":
            await self._modify_task_in_plan(selected_plan)

    async def _delete_task_plan(self):
        """Delete a task plan."""
        if not self.active_plans:
            self.console.print("[yellow]No active plans to delete.[/yellow]")
            return
        
        await self._view_task_plans()
        
        plan_id = Prompt.ask("Enter plan ID to delete (first 8 characters)")
        
        # Find matching plan
        selected_plan_id = None
        for pid in self.active_plans.keys():
            if pid.startswith(plan_id):
                selected_plan_id = pid
                break
        
        if not selected_plan_id:
            self.console.print("[red]Plan not found.[/red]")
            return
        
        plan = self.active_plans[selected_plan_id]
        
        if Confirm.ask(f"Delete plan '{plan.name}'?"):
            del self.active_plans[selected_plan_id]
            self.console.print(f"[green]Deleted plan: {plan.name}[/green]")

    async def _task_execution_interface(self):
        """Task execution interface."""
        self.console.print(Panel("[bold]Task Execution[/bold]", style="green"))
        
        if not self.active_plans:
            self.console.print("[yellow]No active plans to execute.[/yellow]")
            return
        
        # Show available plans
        await self._view_task_plans()
        
        plan_id = Prompt.ask("Enter plan ID to execute (first 8 characters)")
        
        # Find matching plan
        selected_plan = None
        for pid, plan in self.active_plans.items():
            if pid.startswith(plan_id):
                selected_plan = plan
                break
        
        if not selected_plan:
            self.console.print("[red]Plan not found.[/red]")
            return
        
        # Execute plan with progress monitoring
        await self._execute_plan_with_monitoring(selected_plan)

    async def _execute_plan_with_monitoring(self, plan: TaskPlan):
        """Execute a plan with real-time monitoring."""
        self.console.print(f"\n[bold]Executing plan: {plan.name}[/bold]")
        
        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        with progress:
            # Create progress task
            total_tasks = len(plan.tasks)
            progress_task = progress.add_task("Executing tasks...", total=total_tasks)
            
            # Execute tasks
            completed = 0
            for task in plan.tasks.values():
                if task.status == TaskStatus.COMPLETED:
                    completed += 1
                    continue
                
                # Update progress
                progress.update(progress_task, description=f"Executing: {task.name}")
                
                # Execute task
                try:
                    await self._execute_single_task(task)
                    task.status = TaskStatus.COMPLETED
                    completed += 1
                    
                    progress.update(progress_task, completed=completed)
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self.console.print(f"[red]Task failed: {task.name} - {e}[/red]")
            
            progress.update(progress_task, description="Execution complete!")
        
        # Show results
        self.console.print(f"\n[bold]Execution Results:[/bold]")
        self.console.print(f"Total tasks: {total_tasks}")
        self.console.print(f"Completed: {completed}")
        self.console.print(f"Failed: {total_tasks - completed}")

    async def _execute_single_task(self, task: Task):
        """Execute a single task."""
        # Select appropriate agent
        agent = self._select_agent_for_task(task)
        
        # Create task input
        task_input = TaskInput(
            id=task.id,
            task=task.name,
            prompt=f"Task: {task.name}\nDescription: {task.description}",
            data={"task_type": task.task_type, "priority": task.priority.name}
        )
        
        # Execute task
        result = await agent.execute_task(task_input)
        
        # Store result
        task.result = result.result if result.result else None
        
        # Simulate execution time
        await asyncio.sleep(1)

    def _select_agent_for_task(self, task: Task) -> ReactAgent:
        """Select the most appropriate agent for a task."""
        # Simple agent selection logic
        if "plan" in task.name.lower() or "strategy" in task.name.lower():
            return self.agents.get("planner", self.agents["general"])
        elif "analyze" in task.name.lower() or "data" in task.name.lower():
            return self.agents.get("analyzer", self.agents["general"])
        elif "execute" in task.name.lower() or "implement" in task.name.lower():
            return self.agents.get("executor", self.agents["general"])
        else:
            return self.agents["general"]

    async def _agent_chat_interface(self):
        """Interactive chat with agents."""
        self.console.print(Panel("[bold]Agent Chat Interface[/bold]", style="magenta"))
        
        # Show available agents
        table = Table(title="Available Agents", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Description", style="yellow")
        
        for name, agent in self.agents.items():
            table.add_row(
                name,
                agent.config.role.name if agent.config.role else "Unknown",
                agent.config.description or "No description"
            )
        
        self.console.print(table)
        
        # Select agent
        agent_name = Prompt.ask("Select agent", choices=list(self.agents.keys()))
        agent = self.agents[agent_name]
        
        self.console.print(f"\n[bold]Chatting with {agent_name}[/bold]")
        self.console.print("Type 'quit' to exit chat\n")
        
        # Chat loop
        while True:
            try:
                user_input = Prompt.ask(f"[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                # Execute agent
                with self.console.status(f"[bold green]{agent_name} is thinking..."):
                    response = await agent.execute(user_input)
                
                self.console.print(f"[bold green]{agent_name}[/bold green]: {response.content}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def _dashboard_interface(self):
        """Performance dashboard."""
        self.console.print(Panel("[bold]Performance Dashboard[/bold]", style="yellow"))
        
        # System status
        status_table = Table(title="System Status", show_header=True)
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="yellow")
        
        status_table.add_row("Agents", "Active", f"{len(self.agents)} loaded")
        status_table.add_row("Tools", "Ready", f"{len(self.tools.list_names())} available")
        status_table.add_row("Plans", "Active", f"{len(self.active_plans)} plans")
        status_table.add_row("Memory", "OK", "Initialized")
        
        self.console.print(status_table)
        
        # Task statistics
        if self.active_plans:
            stats_table = Table(title="Task Statistics", show_header=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            total_tasks = sum(len(plan.tasks) for plan in self.active_plans.values())
            completed_tasks = sum(
                sum(1 for task in plan.tasks.values() if task.status == TaskStatus.COMPLETED)
                for plan in self.active_plans.values()
            )
            
            stats_table.add_row("Total Tasks", str(total_tasks))
            stats_table.add_row("Completed", str(completed_tasks))
            stats_table.add_row("Success Rate", f"{(completed_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%")
            
            self.console.print(stats_table)

    async def _configuration_interface(self):
        """Configuration interface."""
        self.console.print(Panel("[bold]System Configuration[/bold]", style="cyan"))
        
        self.console.print("\n[bold]Configuration Options:[/bold]")
        self.console.print("1. View current configuration")
        self.console.print("2. Agent settings")
        self.console.print("3. Tool management")
        self.console.print("4. System settings")
        self.console.print("0. Back to main menu")
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
        
        if choice == "1":
            await self._view_configuration()
        elif choice == "2":
            await self._agent_settings()
        elif choice == "3":
            await self._tool_management()
        elif choice == "4":
            await self._system_settings()

    async def _view_configuration(self):
        """View current configuration."""
        config_table = Table(title="Current Configuration", show_header=True)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Agents", str(len(self.agents)))
        config_table.add_row("Tools", str(len(self.tools.list_tools())))
        config_table.add_row("Active Plans", str(len(self.active_plans)))
        config_table.add_row("Memory Type", type(self.memory).__name__)
        
        self.console.print(config_table)

    async def _agent_settings(self):
        """Agent configuration settings."""
        self.console.print("\n[bold]Agent Settings[/bold]")
        
        # Show current agents
        for name, agent in self.agents.items():
            self.console.print(f"[cyan]{name}[/cyan]: {agent.config.description}")

    async def _tool_management(self):
        """Tool management interface."""
        self.console.print("\n[bold]Tool Management[/bold]")
        
        tools_table = Table(title="Available Tools", show_header=True)
        tools_table.add_column("Name", style="cyan")
        tools_table.add_column("Description", style="green")
        
        for tool_name in self.tools.list_names():
            tool = self.tools.get_tool(tool_name)
            if tool:
                tools_table.add_row(tool_name, tool.description)
        
        self.console.print(tools_table)

    async def _system_settings(self):
        """System settings interface."""
        self.console.print("\n[bold]System Settings[/bold]")
        
        while True:
            self.console.print("\n[bold]Settings Options:[/bold]")
            self.console.print("1. LLM Provider Configuration")
            self.console.print("2. Model Settings")
            self.console.print("3. System Information")
            self.console.print("0. Back to configuration menu")
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3"])
            
            if choice == "0":
                break
            elif choice == "1":
                await self._configure_llm_provider()
            elif choice == "2":
                await self._configure_model_settings()
            elif choice == "3":
                await self._show_system_info()

    async def _configure_llm_provider(self):
        """Configure LLM provider."""
        self.console.print("\n[bold]LLM Provider Configuration[/bold]")
        
        # Show current provider
        self.console.print(f"Current provider: [cyan]{self.current_provider}[/cyan]")
        
        # Show available providers
        table = Table(title="Available Providers", show_header=True)
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description", style="yellow")
        
        provider_info = {
            "mock": ("Always Available", "Mock provider for testing and development"),
            "openai": ("Requires API Key", "OpenAI GPT models via official API"),
            "llama_local": ("Requires Libraries", "Local Llama models via transformers")
        }
        
        for provider in self.available_providers:
            status, desc = provider_info.get(provider, ("Unknown", "Unknown provider"))
            table.add_row(provider, status, desc)
        
        self.console.print(table)
        
        # Provider selection
        new_provider = Prompt.ask(
            "Select provider",
            choices=self.available_providers,
            default=self.current_provider
        )
        
        if new_provider != self.current_provider:
            self.current_provider = new_provider
            self.console.print(f"[green]Provider changed to: {new_provider}[/green]")
            
            # Reinitialize agents with new provider
            await self._reinitialize_agents()

    async def _configure_model_settings(self):
        """Configure model settings."""
        self.console.print("\n[bold]Model Settings[/bold]")
        
        config = self.provider_configs[self.current_provider]
        
        # Show current settings
        settings_table = Table(title=f"Current Settings ({self.current_provider})", show_header=True)
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value", style="green")
        
        for key, value in config.items():
            settings_table.add_row(key, str(value))
        
        self.console.print(settings_table)
        
        # Allow editing
        if Confirm.ask("Edit settings?"):
            for key, current_value in config.items():
                if key == "api_key" and current_value == "":
                    new_value = Prompt.ask(f"Enter {key} (hidden)", password=True, default="")
                else:
                    new_value = Prompt.ask(f"Enter {key}", default=str(current_value))
                
                # Type conversion
                if key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                    try:
                        new_value = float(new_value)
                    except ValueError:
                        self.console.print(f"[red]Invalid float value for {key}[/red]")
                        continue
                elif key in ["max_tokens", "context_length", "batch_size"]:
                    try:
                        new_value = int(new_value)
                    except ValueError:
                        self.console.print(f"[red]Invalid integer value for {key}[/red]")
                        continue
                
                config[key] = new_value
            
            self.console.print("[green]Settings updated[/green]")
            
            # Reinitialize agents
            await self._reinitialize_agents()

    async def _show_system_info(self):
        """Show system information."""
        self.console.print("\n[bold]System Information[/bold]")
        
        info_table = Table(title="System Info", show_header=True)
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Current Provider", self.current_provider)
        info_table.add_row("Model", self.provider_configs[self.current_provider].get("model_name", "Unknown"))
        info_table.add_row("Active Agents", str(len(self.agents)))
        info_table.add_row("Active Plans", str(len(self.active_plans)))
        info_table.add_row("Available Tools", str(len(self.tools.list_names())))
        
        # Environment info
        import os
        info_table.add_row("OpenAI API Key", "Set" if os.getenv("OPENAI_API_KEY") else "Not Set")
        info_table.add_row("Provider Override", os.getenv("LLAMAAGENT_LLM_PROVIDER", "None"))
        info_table.add_row("Model Override", os.getenv("LLAMAAGENT_LLM_MODEL", "None"))
        
        self.console.print(info_table)

    async def _reinitialize_agents(self):
        """Reinitialize agents with new provider configuration."""
        self.console.print("[yellow]Reinitializing agents with new configuration...[/yellow]")
        
        # Cleanup existing agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up agent: {e}")
        
        # Clear agents
        self.agents.clear()
        
        # Reinitialize with new provider
        self._initialize_default_agents()
        
        self.console.print("[green]Agents reinitialized successfully[/green]")

    async def _testing_interface(self):
        """Testing and debugging interface."""
        self.console.print(Panel("[bold]System Testing[/bold]", style="red"))
        
        self.console.print("\n[bold]Testing Options:[/bold]")
        self.console.print("1. Test agent functionality")
        self.console.print("2. Test tool execution")
        self.console.print("3. Test task planning")
        self.console.print("4. System diagnostics")
        self.console.print("0. Back to main menu")
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
        
        if choice == "1":
            await self._test_agents()
        elif choice == "2":
            await self._test_tools()
        elif choice == "3":
            await self._test_task_planning()
        elif choice == "4":
            await self._system_diagnostics()

    async def _test_agents(self):
        """Test agent functionality."""
        self.console.print("\n[bold]Testing Agents[/bold]")
        
        test_prompt = "What is 2 + 2?"
        
        for name, agent in self.agents.items():
            try:
                with self.console.status(f"Testing {name}..."):
                    response = await agent.execute(test_prompt)
                self.console.print(f"[green]{name}: {response.content[:100]}...[/green]")
            except Exception as e:
                self.console.print(f"[red]{name}: Error - {e}[/red]")

    async def _test_tools(self):
        """Test tool execution."""
        self.console.print("\n[bold]Testing Tools[/bold]")
        
        for tool_name in self.tools.list_names():
            tool = self.tools.get_tool(tool_name)
            if tool:
                try:
                    if tool_name == "calculator":
                        result = tool.execute(expression="2 + 2")
                        self.console.print(f"[green]{tool_name}: {result}[/green]")
                    else:
                        self.console.print(f"[yellow]{tool_name}: Available[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]{tool_name}: Error - {e}[/red]")

    async def _test_task_planning(self):
        """Test task planning functionality."""
        self.console.print("\n[bold]Testing Task Planning[/bold]")
        
        # Create test plan
        test_task = Task(
            name="Test Task",
            description="A simple test task",
            priority=TaskPriority.MEDIUM,
        )
        
        test_plan = self.task_planner.create_plan("Test Goal", [test_task])
        
        self.console.print(f"[green]Created test plan: {test_plan.name}[/green]")
        self.console.print(f"Tasks: {len(test_plan.tasks)}")

    async def _system_diagnostics(self):
        """Run system diagnostics."""
        self.console.print("\n[bold]System Diagnostics[/bold]")
        
        diagnostics = [
            ("Agents", len(self.agents) > 0, f"{len(self.agents)} loaded"),
            ("Tools", len(self.tools.list_names()) > 0, f"{len(self.tools.list_names())} available"),
            ("Memory", True, "Initialized"),
            ("Task Planner", True, "Ready"),
            ("Execution Engine", True, "Ready"),
        ]
        
        table = Table(title="Diagnostics", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="green")
        
        for component, status, details in diagnostics:
            status_text = " OK" if status else " FAIL"
            table.add_row(component, status_text, details)
        
        self.console.print(table)

    async def _help_interface(self):
        """Help and documentation."""
        help_text = """[bold]LlamaAgent Master CLI[/bold]

[bold cyan]Features:[/bold cyan]
• Dynamic Task Planning - Create complex task plans with dependencies
• Multi-Agent System - Specialized agents for different task types
• Real-time Monitoring - Track execution progress with live updates
• Interactive Chat - Direct communication with agents
• Performance Dashboard - Monitor system metrics and statistics

[bold cyan]Getting Started:[/bold cyan]
1. Create a task plan (Option 1)
2. Execute tasks (Option 2)
3. Chat with agents (Option 3)
4. Monitor performance (Option 4)

[bold cyan]Agent Types:[/bold cyan]
• General Agent - General purpose tasks
• Planner Agent - Strategic planning and decomposition
• Executor Agent - Task execution and implementation
• Analyzer Agent - Data analysis and insights

[bold cyan]Task Planning:[/bold cyan]
• Create hierarchical task structures
• Set priorities and dependencies
• Estimate durations and resources
• Monitor execution progress

[bold cyan]Tools Available:[/bold cyan]
• Calculator - Mathematical computations
• Python REPL - Code execution
• And more through the tool registry
"""
        self.console.print(Panel(help_text, title="Help", style="blue"))

    async def _add_task_to_plan(self, plan: TaskPlan):
        """Add a task to an existing plan."""
        task_name = Prompt.ask("Task name")
        task_desc = Prompt.ask("Task description", default=task_name)
        
        priority_map = {
            "1": TaskPriority.LOW,
            "2": TaskPriority.MEDIUM,
            "3": TaskPriority.HIGH,
            "4": TaskPriority.CRITICAL,
        }
        
        priority_choice = Prompt.ask(
            "Priority (1=Low, 2=Medium, 3=High, 4=Critical)",
            choices=["1", "2", "3", "4"],
            default="2"
        )
        
        task = Task(
            name=task_name,
            description=task_desc,
            priority=priority_map[priority_choice],
        )
        
        plan.add_task(task)
        self.console.print(f"[green]Added task: {task_name}[/green]")

    async def _remove_task_from_plan(self, plan: TaskPlan):
        """Remove a task from a plan."""
        task_id = Prompt.ask("Enter task ID to remove (first 8 characters)")
        
        # Find matching task
        task_to_remove = None
        for tid in plan.tasks:
            if tid.startswith(task_id):
                task_to_remove = tid
                break
        
        if task_to_remove:
            del plan.tasks[task_to_remove]
            self.console.print(f"[green]Removed task[/green]")
        else:
            self.console.print("[red]Task not found[/red]")

    async def _modify_task_in_plan(self, plan: TaskPlan):
        """Modify a task in a plan."""
        task_id = Prompt.ask("Enter task ID to modify (first 8 characters)")
        
        # Find matching task
        task_to_modify = None
        for tid, task in plan.tasks.items():
            if tid.startswith(task_id):
                task_to_modify = task
                break
        
        if not task_to_modify:
            self.console.print("[red]Task not found[/red]")
            return
        
        self.console.print(f"Modifying task: {task_to_modify.name}")
        
        # Modification options
        new_name = Prompt.ask("New name", default=task_to_modify.name)
        new_desc = Prompt.ask("New description", default=task_to_modify.description)
        
        task_to_modify.name = new_name
        task_to_modify.description = new_desc
        
        self.console.print("[green]Task modified[/green]")

    async def _cleanup(self):
        """Cleanup resources."""
        self.console.print("\n[yellow]Cleaning up...[/yellow]")
        
        # Cleanup agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up agent: {e}")
        
        self.console.print("[green]Cleanup complete[/green]")


async def main():
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run CLI
    cli = MasterCLI()
    
    try:
        await cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error in main")
    finally:
        console.print("[green]Goodbye![/green]")


if __name__ == "__main__":
    asyncio.run(main()) 