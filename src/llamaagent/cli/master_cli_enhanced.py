"""
Enhanced Master CLI for LlamaAgent with Dynamic Task Planning
============================================================

Integrates with existing llamaagent architecture to provide:
- Dynamic task planning and scheduling
- Interactive menu system
- Real-time execution monitoring
- Multi-agent orchestration
- Performance analytics

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
import signal
import time
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

from ..agents import ReactAgent
from ..agents.base import AgentConfig, AgentRole
from ..llm import create_provider
from ..memory.base import SimpleMemory
from ..planning import (
    ExecutionEngine,
    Task,
    TaskPlan,
    TaskPlanner,
    TaskPriority,
    TaskStatus,
)
from ..tools import ToolRegistry, get_all_tools

logger = logging.getLogger(__name__)
console = Console()


class EnhancedMasterCLI:
    """Enhanced Master CLI with dynamic task planning and scheduling."""

    def __init__(self):
        self.console = console
        self.agents: Dict[str, ReactAgent] = {}
        self.task_planner = TaskPlanner()
        self.execution_engine = ExecutionEngine(
            max_concurrent_tasks=3,
            enable_adaptive_execution=True,
            enable_parallel_execution=True,
        )
        self.tools = ToolRegistry()
        self.memory = SimpleMemory()
        self.active_plans: Dict[str, TaskPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.shutdown_requested = False
        
        # Performance metrics
        self.start_time = time.time()
        self.total_tasks = 0
        self.successful_tasks = 0
        
        self._initialize_components()
        self._setup_signal_handlers()

    def _initialize_components(self):
        """Initialize CLI components."""
        try:
            # Initialize tools
            for tool in get_all_tools():
                self.tools.register(tool)
            
            # Create default agents
            self._create_default_agents()
            
            logger.info("Enhanced Master CLI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")

    def _create_default_agents(self):
        """Create default agent configurations."""
        agent_configs = [
            ("GeneralAgent", AgentRole.GENERALIST, True),
            ("PlannerAgent", AgentRole.PLANNER, True),
            ("ExecutorAgent", AgentRole.EXECUTOR, False),
            ("AnalyzerAgent", AgentRole.ANALYZER, True),
        ]
        
        for name, role, spree_enabled in agent_configs:
            config = AgentConfig(
                name=name,
                role=role,
                description=f"{role.name} specialized agent",
                spree_enabled=spree_enabled,
                max_iterations=10,
            )
            
            agent = ReactAgent(
                config=config,
                llm_provider=create_provider("mock"),
                memory=self.memory,
                tools=self.tools,
            )
            
            self.agents[name] = agent

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            self.shutdown_requested = True
            self.console.print("\n[yellow]Shutdown requested...[/yellow]")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def show_banner(self):
        """Display application banner."""
        banner = Text()
        banner.append("LlamaAgent LlamaAgent Enhanced Master CLI\n", style="bold blue")
        banner.append("Dynamic Task Planning & Scheduling\n", style="cyan")
        banner.append("Author: Nik Jois <nikjois@llamasearch.ai>", style="dim")
        
        self.console.print(Panel(banner, title="Welcome", border_style="blue"))

    def show_main_menu(self):
        """Display main interactive menu."""
        table = Table(title="Enhanced Master CLI", show_header=True, header_style="bold magenta")
        table.add_column("Option", style="cyan", width=8)
        table.add_column("Feature", style="green")
        table.add_column("Description", style="yellow")

        menu_items = [
            ("1", "📋 Task Planning", "Create and manage dynamic task plans"),
            ("2", "Execute Execute Tasks", "Run tasks with real-time monitoring"),
            ("3", "Agent Agent Chat", "Interactive chat with specialized agents"),
            ("4", "RESULTS Dashboard", "Performance metrics and analytics"),
            ("5", "Configuration Configuration", "System and agent configuration"),
            ("6", "Testing", "Debug and test system components"),
            ("7", "Help Help", "Documentation and examples"),
            ("0", "🚪 Exit", "Exit the application"),
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
        self.console.print(Panel("📋 Dynamic Task Planning", style="bold blue"))
        
        if self.active_plans:
            self.console.print("\n[bold]Active Plans:[/bold]")
            for plan_id, plan in self.active_plans.items():
                status = self._get_plan_status(plan)
                self.console.print(f"  • {plan.name} ({plan_id[:8]}) - {status}")
        
        options = [
            ("1", "Create New Plan"),
            ("2", "View Plan Details"),
            ("3", "Modify Plan"),
            ("4", "Delete Plan"),
            ("0", "Back"),
        ]
        
        self.console.print("\n[bold]Planning Options:[/bold]")
        for option, description in options:
            self.console.print(f"  {option}. {description}")
        
        choice = Prompt.ask("Select option", choices=[opt[0] for opt in options], default="0")
        
        if choice == "1":
            await self._create_task_plan()
        elif choice == "2":
            await self._view_plan_details()

    async def _create_task_plan(self):
        """Create a new task plan."""
        self.console.print(Panel("Create Create Task Plan", style="bold green"))
        
        goal = Prompt.ask("Enter main goal")
        if not goal:
            return
        
        plan_name = Prompt.ask("Plan name", default=f"Plan: {goal[:30]}")
        
        # Get initial tasks
        self.console.print("\n[bold]Define tasks (empty to finish):[/bold]")
        tasks = []
        
        while True:
            task_desc = Prompt.ask(f"Task {len(tasks) + 1}", default="")
            if not task_desc:
                break
            
            priority = Prompt.ask(
                "Priority (1=Low, 2=Medium, 3=High, 4=Critical)",
                choices=["1", "2", "3", "4"],
                default="2"
            )
            
            duration = IntPrompt.ask("Duration (minutes)", default=15)
            
            task = Task(
                name=task_desc,
                description=task_desc,
                priority=list(TaskPriority)[int(priority) - 1],
                estimated_duration=timedelta(minutes=duration),
            )
            tasks.append(task)
        
        # Create plan
        try:
            with self.console.status("Creating plan..."):
                plan = self.task_planner.create_plan(
                    goal=goal,
                    initial_tasks=tasks,
                    auto_decompose=True,
                )
                plan.name = plan_name
                
                # Optimize plan
                optimized_plan = self.task_planner.optimize_plan(plan)
                self.active_plans[plan.id] = optimized_plan
            
            self.console.print(f"[green]✓ Plan created: {plan.id}[/green]")
            self.console.print(f"Total tasks: {len(plan.tasks)}")
            
            # Show summary
            await self._show_plan_summary(plan)
            
        except Exception as e:
            self.console.print(f"[red]Failed to create plan: {e}[/red]")

    async def _show_plan_summary(self, plan: TaskPlan):
        """Show plan summary."""
        table = Table(title=f"Plan: {plan.name}", show_header=True)
        table.add_column("Task", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Duration", style="green")
        table.add_column("Status", style="magenta")
        
        for task in plan.tasks.values():
            table.add_row(
                task.name,
                task.priority.name,
                f"{task.estimated_duration.total_seconds()/60:.1f}m",
                task.status.name,
            )
        
        self.console.print(table)

    async def _task_execution_interface(self):
        """Task execution interface."""
        self.console.print(Panel("Execute Task Execution", style="bold blue"))
        
        if not self.active_plans:
            self.console.print("[yellow]No active plans to execute[/yellow]")
            return
        
        # Select plan
        plan_choices = {}
        self.console.print("\n[bold]Available Plans:[/bold]")
        for i, (plan_id, plan) in enumerate(self.active_plans.items(), 1):
            self.console.print(f"  {i}. {plan.name}")
            plan_choices[str(i)] = plan_id
        
        choice = Prompt.ask("Select plan", choices=list(plan_choices.keys()), default="1")
        plan = self.active_plans[plan_choices[choice]]
        
        if Confirm.ask(f"Execute '{plan.name}'?"):
            await self._execute_plan_with_monitoring(plan)

    async def _execute_plan_with_monitoring(self, plan: TaskPlan):
        """Execute plan with real-time monitoring."""
        self.console.print(f"\n[bold green]Executing: {plan.name}[/bold green]")
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        
        with progress:
            overall_task = progress.add_task("Overall Progress", total=len(plan.tasks))
            
            try:
                # Simple task executor
                async def task_executor(task: Task) -> str:
                    agent = self._select_agent_for_task(task)
                    response = await agent.execute(task.description)
                    
                    if response.success:
                        return response.content
                    else:
                        raise Exception(f"Task failed: {response.content}")
                
                # Execute plan
                results = await self.execution_engine.execute_plan(plan, task_executor)
                
                # Update metrics
                self.total_tasks += len(results)
                self.successful_tasks += sum(1 for r in results.values() if r.success)
                
                # Show results
                self._show_execution_results(results)
                
            except Exception as e:
                self.console.print(f"[red]Execution failed: {e}[/red]")

    def _select_agent_for_task(self, task: Task) -> ReactAgent:
        """Select appropriate agent for task."""
        task_lower = task.description.lower()
        
        if any(word in task_lower for word in ["plan", "strategy", "design"]):
            return self.agents["PlannerAgent"]
        elif any(word in task_lower for word in ["analyze", "evaluate", "assess"]):
            return self.agents["AnalyzerAgent"]
        elif any(word in task_lower for word in ["execute", "run", "perform"]):
            return self.agents["ExecutorAgent"]
        else:
            return self.agents["GeneralAgent"]

    def _show_execution_results(self, results: Dict[str, Any]):
        """Display execution results."""
        table = Table(title="Execution Results", show_header=True)
        table.add_column("Task", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Duration", style="green")
        table.add_column("Result", style="white")
        
        for task_id, result in results.items():
            status = "✓ Success" if result.success else "✗ Failed"
            duration = f"{result.duration.total_seconds():.1f}s" if result.duration else "N/A"
            result_preview = str(result.result)[:40] + "..." if len(str(result.result)) > 40 else str(result.result)
            
            table.add_row(task_id[:8], status, duration, result_preview)
        
        self.console.print(table)

    async def _agent_chat_interface(self):
        """Interactive chat with agents."""
        self.console.print(Panel("💬 Agent Chat", style="bold blue"))
        
        # Select agent
        agent_names = list(self.agents.keys())
        self.console.print("\n[bold]Available Agents:[/bold]")
        for i, name in enumerate(agent_names, 1):
            agent = self.agents[name]
            self.console.print(f"  {i}. {name} ({agent.config.role.name})")
        
        choice = IntPrompt.ask("Select agent", choices=list(range(1, len(agent_names) + 1)), default=1)
        selected_agent = self.agents[agent_names[choice - 1]]
        
        self.console.print(f"\n[bold green]Chatting with {selected_agent.config.name}[/bold green]")
        self.console.print("[dim]Type 'exit' to end, 'help' for commands[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                elif user_input.lower() == "help":
                    self._show_chat_help()
                    continue
                
                with self.console.status("Thinking..."):
                    response = await selected_agent.execute(user_input)
                
                self.console.print(f"\n[bold green]{selected_agent.config.name}[/bold green]: {response.content}")
                
                if response.trace:
                    self.console.print(f"[dim]Time: {response.execution_time:.2f}s, Tokens: {response.tokens_used}[/dim]")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def _show_chat_help(self):
        """Show chat help."""
        help_text = """[bold]Chat Commands:[/bold]
• exit/quit - End chat
• help - Show this help
• Just type your message to chat!"""
        self.console.print(Panel(help_text, title="Help", style="blue"))

    async def _dashboard_interface(self):
        """Performance dashboard."""
        self.console.print(Panel("RESULTS Performance Dashboard", style="bold blue"))
        
        uptime = time.time() - self.start_time
        success_rate = (self.successful_tasks / max(self.total_tasks, 1)) * 100
        
        metrics_table = Table(title="System Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics = [
            ("Uptime", f"{uptime/3600:.1f} hours"),
            ("Active Agents", str(len(self.agents))),
            ("Active Plans", str(len(self.active_plans))),
            ("Total Tasks", str(self.total_tasks)),
            ("Successful Tasks", str(self.successful_tasks)),
            ("Success Rate", f"{success_rate:.1f}%"),
        ]
        
        for metric, value in metrics:
            metrics_table.add_row(metric, value)
        
        self.console.print(metrics_table)

    async def _configuration_interface(self):
        """System configuration."""
        self.console.print(Panel("Configuration Configuration", style="bold blue"))
        self.console.print("Configuration options will be implemented here.")

    async def _testing_interface(self):
        """Testing and debugging."""
        self.console.print(Panel("Testing & Debug", style="bold blue"))
        
        test_options = [
            ("1", "Test Agent Response"),
            ("2", "Test Task Planning"),
            ("3", "System Diagnostics"),
            ("0", "Back"),
        ]
        
        for option, description in test_options:
            self.console.print(f"  {option}. {description}")
        
        choice = Prompt.ask("Select test", choices=[opt[0] for opt in test_options], default="0")
        
        if choice == "1":
            await self._test_agent_response()
        elif choice == "3":
            await self._system_diagnostics()

    async def _test_agent_response(self):
        """Test agent response."""
        test_prompt = "What is 2 + 2?"
        agent = self.agents["GeneralAgent"]
        
        self.console.print(f"Testing with: {test_prompt}")
        
        with self.console.status("Testing..."):
            response = await agent.execute(test_prompt)
        
        self.console.print(f"Response: {response.content}")
        self.console.print(f"Success: {response.success}")

    async def _system_diagnostics(self):
        """Run system diagnostics."""
        self.console.print("Running diagnostics...")
        
        diagnostics = [
            ("Agents", len(self.agents) > 0, f"{len(self.agents)} loaded"),
            ("Tools", len(self.tools.list_tools()) > 0, f"{len(self.tools.list_tools())} available"),
            ("Memory", self.memory is not None, "Initialized"),
            ("Task Planner", self.task_planner is not None, "Ready"),
            ("Execution Engine", self.execution_engine is not None, "Ready"),
        ]
        
        table = Table(title="Diagnostics", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="green")
        
        for component, status, details in diagnostics:
            status_text = "✓ OK" if status else "✗ FAIL"
            table.add_row(component, status_text, details)
        
        self.console.print(table)

    async def _help_interface(self):
        """Help and documentation."""
        help_text = """[bold]LlamaAgent Enhanced Master CLI[/bold]

[bold cyan]Features:[/bold cyan]
• Dynamic Task Planning - Create complex task plans
• Multi-Agent System - Specialized agents for different tasks
• SPRE Methodology - Strategic Planning & Resourceful Execution
• Real-time Monitoring - Track execution progress
• Interactive Chat - Direct communication with agents

[bold cyan]Getting Started:[/bold cyan]
1. Create a task plan (Option 1)
2. Execute tasks (Option 2)
3. Chat with agents (Option 3)
4. Monitor performance (Option 4)

[bold cyan]Agent Types:[/bold cyan]
• GeneralAgent - General purpose tasks
• PlannerAgent - Strategic planning
• ExecutorAgent - Task execution
• AnalyzerAgent - Data analysis
"""
        self.console.print(Panel(help_text, title="Help", style="blue"))

    def _get_plan_status(self, plan: TaskPlan) -> str:
        """Get plan execution status."""
        completed = sum(1 for task in plan.tasks.values() if task.status == TaskStatus.COMPLETED)
        total = len(plan.tasks)
        return f"{completed}/{total} completed"

    async def _view_plan_details(self):
        """View plan details - placeholder."""
        self.console.print("Plan details view will be implemented here.")

    async def _cleanup(self):
        """Cleanup resources."""
        self.console.print("\n[bold blue]Cleaning up...[/bold blue]")
        
        for agent in self.agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        
        self.console.print("[bold green]Goodbye![/bold green]")


async def main():
    """Main entry point."""
    cli = EnhancedMasterCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main()) 