"""
Master Command-Line Interface for LlamaAgent

Complete, feature-rich CLI with progress bars, menus, and interactive options.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from ..agents import AdvancedReasoningAgent, MultiModalAdvancedAgent, ReactAgent
from ..agents.base import AgentConfig
from ..cache import AdvancedCache, CacheStrategy
from ..core import get_error_handler
from ..llm import create_provider, get_available_providers
from ..optimization import get_optimizer
from ..tools import ToolRegistry, get_all_tools

console = Console()


class LlamaAgentCLI:
    """Master CLI for LlamaAgent framework."""

    def __init__(self) -> None:
        self.console = console
        self.config = self._load_config()
        self.cache = AdvancedCache(strategy=CacheStrategy.HYBRID)
        self.error_handler = get_error_handler()
        self.optimizer = get_optimizer()
        self.current_agent = None
        self.session_history: List[Dict[str, Any]] = []

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config_path = Path.home() / ".llamaagent" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {"default_provider": "openai", "default_model": "gpt-4", "theme": "dark"}

    def main_menu(self) -> None:
        """Display main menu."""
        self.console.clear()
        self.console.print(
            Panel.fit(
                "[bold cyan]LlamaAgent Master CLI[/bold cyan]\n[dim]Advanced AI Agent Framework[/dim]",
                border_style="cyan",
            )
        )

        menu_items = [
            "1. Quick Chat",
            "2. Advanced Reasoning",
            "3. Multi-Modal Analysis",
            "4. Agent Builder",
            "5. Tool Management",
            "6. Performance Monitor",
            "7. Configuration",
            "8. Documentation",
            "9. Exit",
        ]

        for item in menu_items:
            self.console.print(f"  {item}")

        choice = Prompt.ask(
            "\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )

        if choice == "1":
            self.quick_chat()
        elif choice == "2":
            self.advanced_reasoning()
        elif choice == "3":
            self.multimodal_analysis()
        elif choice == "4":
            self.agent_builder()
        elif choice == "5":
            self.tool_management()
        elif choice == "6":
            self.performance_monitor()
        elif choice == "7":
            self.configuration()
        elif choice == "8":
            self.documentation()
        elif choice == "9":
            self.exit_cli()

    def quick_chat(self):
        """Quick chat interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Quick Chat Mode[/bold]"))

        # Select provider
        providers = get_available_providers()
        self.console.print("\nAvailable providers:")
        for i, provider in enumerate(providers, 1):
            self.console.print(f"  {i}. {provider}")

        provider_idx = (
            IntPrompt.ask(
                "Select provider",
                default=(
                    providers.index(self.config["default_provider"]) + 1
                    if self.config["default_provider"] in providers
                    else 1
                ),
            )
            - 1
        )
        provider = providers[provider_idx]

        # Create agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing agent...", total=None)

            llm_provider = create_provider(provider)
            config = AgentConfig(name="QuickChatAgent")
            tools = ToolRegistry()
            for tool in get_all_tools():
                tools.register(tool)

            self.current_agent = ReactAgent(
                config=config, llm_provider=llm_provider, tools=tools
            )

            progress.update(task, completed=True)

        # Chat loop
        self.console.print("\n[dim]Type 'exit' to return to menu[/dim]\n")

        while True:
            message = Prompt.ask("[bold]You[/bold]")

            if message.lower() == "exit":
                break

            # Process message
            with Live(
                Panel("Thinking...", border_style="yellow"),
                console=self.console,
                refresh_per_second=4,
            ) as live:
                start_time = time.time()

                async def process():
                    response = await self.current_agent.execute(message)
                    return response

                response = asyncio.run(process())

                elapsed = time.time() - start_time

                live.update(
                    Panel(
                        f"[bold green]Agent:[/bold green]\n{response.content}\n\n"
                        f"[dim]Time: {elapsed:.2f}s | Tokens: {response.tokens_used}[/dim]",
                        border_style="green",
                    )
                )

            self.session_history.append(
                {"user": message, "agent": response.content, "time": elapsed}
            )

        if Confirm.ask("\nSave conversation?"):
            self._save_conversation()

    def advanced_reasoning(self):
        """Advanced reasoning interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Advanced Reasoning Mode[/bold]"))

        # Create reasoning agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing agent...", total=None)

            llm_provider = create_provider(self.config["default_provider"])
            config = AgentConfig(name="AdvancedReasoner")

            self.current_agent = AdvancedReasoningAgent(
                config=config,
                llm_provider=llm_provider,
            )

            progress.update(task, completed=True)

        # Reasoning interface
        self.console.print("\n[dim]Enter your complex query[/dim]\n")

        query = Prompt.ask("[bold]Query[/bold]")

        # Process with visualization
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            steps = [
                "Decomposing problem",
                "Generating thoughts",
                "Self-correcting",
                "Verifying reasoning",
                "Synthesizing answer",
            ]

            task = progress.add_task("Reasoning...", total=len(steps))

            async def reason():
                trace = await self.current_agent.reason(query)
                return trace

            # Simulate steps
            for step in steps:
                progress.update(task, description=f"[cyan]{step}[/cyan]")
                time.sleep(
                    0.5
                )  # In real implementation, update based on actual progress
                progress.advance(task)

            trace = asyncio.run(reason())

        # Display results
        self._display_reasoning_trace(trace)

    def multimodal_analysis(self):
        """Multi-modal analysis interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Multi-Modal Analysis[/bold]"))

        # Select modalities
        self.console.print("\nSelect input modalities:")
        self.console.print("  1. Text only")
        self.console.print("  2. Text + Image")
        self.console.print("  3. Text + Audio")
        self.console.print("  4. All modalities")

        mode = Prompt.ask("Select mode", choices=["1", "2", "3", "4"])

        inputs: Dict[str, Any] = {}

        # Get text input
        text = Prompt.ask("\n[bold]Text input[/bold]")
        inputs["text"] = text

        # Get other inputs based on mode
        if mode in ["2", "4"]:
            image_path = Prompt.ask("Image path (or press Enter to skip)", default="")
            if image_path and Path(image_path).exists():
                inputs["image"] = image_path

        if mode in ["3", "4"]:
            audio_path = Prompt.ask("Audio path (or press Enter to skip)", default="")
            if audio_path and Path(audio_path).exists():
                inputs["audio"] = audio_path

        # Create multi-modal agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Initializing multi-modal agent...", total=None)

            llm_provider = create_provider(self.config["default_provider"])
            config = AgentConfig(name="MultiModalAnalyzer")

            agent = MultiModalAdvancedAgent(
                config=config,
                llm_provider=llm_provider,
            )

            progress.update(task, completed=True)

        # Analyze
        task_description = Prompt.ask("\n[bold]Analysis task[/bold]")

        with Live(
            Panel("Analyzing multi-modal inputs...", border_style="yellow"),
            console=self.console,
            refresh_per_second=4,
        ) as live:

            async def analyze():
                result = await agent.analyze_multimodal(inputs, task_description)
                return result

            result = asyncio.run(analyze())

        # Display results
        self._display_multimodal_results(result)

    def agent_builder(self):
        """Agent builder interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Agent Builder[/bold]"))

        # Get agent configuration
        name = Prompt.ask("Agent name", default="CustomAgent")

        # Select capabilities
        self.console.print("\nSelect capabilities (comma-separated):")
        self.console.print("  1. Reasoning")
        self.console.print("  2. Tool usage")
        self.console.print("  3. Memory")
        self.console.print("  4. Multi-modal")
        self.console.print("  5. Web search")

        capabilities = Prompt.ask("Capabilities", default="1,2").split(",")

        # Build agent configuration
        agent_config = {
            "name": name,
            "capabilities": [cap.strip() for cap in capabilities],
            "provider": self.config["default_provider"],
            "model": self.config["default_model"],
            "cache_enabled": True,
            "error_recovery": True,
            "performance_optimization": True,
        }

        # Generate agent code
        code = self._generate_agent_code(agent_config)

        # Display generated code
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print("\n[bold]Generated Agent Code:[/bold]")
        self.console.print(syntax)

        # Save option
        if Confirm.ask("\nSave agent code?"):
            filename = Prompt.ask("Filename", default=f"{name.lower()}_agent.py")
            with open(filename, "w") as f:
                f.write(code)
            self.console.print(f"[green]Saved to {filename}[/green]")

    def tool_management(self) -> None:
        """Tool management interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Tool Management[/bold]"))

        # List available tools
        tools = get_all_tools()

        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")

        for tool in tools:
            table.add_row(tool.name, tool.description[:50] + "...", "Active")

        self.console.print(table)

        # Tool actions
        self.console.print("\nActions:")
        self.console.print("  1. View tool details")
        self.console.print("  2. Test tool")
        self.console.print("  3. Create custom tool")
        self.console.print("  4. Back to menu")

        action = Prompt.ask("Select action", choices=["1", "2", "3", "4"])

        if action == "1":
            self._view_tool_details(tools)
        elif action == "2":
            self._test_tool(tools)
        elif action == "3":
            self._create_custom_tool()

    def performance_monitor(self) -> None:
        """Performance monitoring interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Performance Monitor[/bold]"))

        # Get performance stats
        stats = self.optimizer.get_optimization_stats()
        cache_stats = asyncio.run(self.cache.get_stats())
        error_stats = self.error_handler.get_error_statistics()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["header"].update(Panel("[bold]System Performance[/bold]"))

        # Performance metrics
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="white")

        perf_table.add_row("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.2%}")
        perf_table.add_row(
            "Avg Response Time", f"{cache_stats.get('avg_access_time_ms', 0):.2f}ms"
        )
        perf_table.add_row("Total Errors", str(error_stats.get("total_errors", 0)))
        perf_table.add_row(
            "Active Thread Pool", str(stats.get("thread_pool_active", 0))
        )

        layout["body"].update(perf_table)

        # Live monitoring
        with Live(layout, console=self.console, refresh_per_second=1) as live:
            for _ in range(10):  # Monitor for 10 seconds
                time.sleep(1)
                # Update stats in real implementation

    def configuration(self) -> None:
        """Configuration interface."""
        self.console.clear()
        self.console.print(Panel("[bold]Configuration[/bold]"))

        # Display current config
        tree = Tree("[bold]Current Configuration[/bold]")

        for key, value in self.config.items():
            tree.add(f"{key}: {value}")

        self.console.print(tree)

        # Edit configuration
        if Confirm.ask("\nEdit configuration?"):
            key = Prompt.ask("Configuration key")
            if key in self.config:
                old_value = self.config[key]
                new_value = Prompt.ask(f"New value (current: {old_value})")
                self.config[key] = new_value
                self._save_config()
                self.console.print("[green]Configuration updated[/green]")

    def documentation(self) -> None:
        """Display documentation."""
        self.console.clear()
        self.console.print(Panel("[bold]LlamaAgent Documentation[/bold]"))

        docs = """
        [bold]Quick Start:[/bold]
        1. Select 'Quick Chat' for simple conversations
        2. Use 'Advanced Reasoning' for complex problem-solving
        3. Try 'Multi-Modal Analysis' for image/audio understanding

        [bold]Features:[/bold]
        - Multiple LLM providers (OpenAI, Anthropic, etc.)
        - Advanced reasoning strategies
        - Multi-modal capabilities
        - Tool integration
        - Performance optimization
        - Error recovery

        [bold]Configuration:[/bold]
        - Edit ~/.llamaagent/config.json
        - Set environment variables for API keys
        - Use 'Configuration' menu for settings

        [bold]Tips:[/bold]
        - Enable caching for faster responses
        - Use error recovery for reliability
        - Monitor performance in real-time
        """

        self.console.print(docs)

        Prompt.ask("\nPress Enter to continue")

    def exit_cli(self) -> None:
        """Exit CLI gracefully."""
        if self.session_history:
            if Confirm.ask("\nSave session history?"):
                self._save_conversation()

        self.console.print("\n[bold green]Thank you for using LlamaAgent![/bold green]")
        sys.exit(0)

    def _save_conversation(self) -> None:
        """Save conversation history."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"llamaagent_session_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.session_history, f, indent=2)

        self.console.print(f"[green]Conversation saved to {filename}[/green]")

    def _display_reasoning_trace(self, trace) -> None:
        """Display reasoning trace."""
        self.console.print("\n[bold]Reasoning Trace:[/bold]\n")

        # Display thoughts
        tree = Tree("[bold]Thought Process[/bold]")
        for i, thought in enumerate(trace.thoughts[-5:], 1):  # Show last 5 thoughts
            tree.add(f"[cyan]Step {i}:[/cyan] {thought.content[:100]}...")

        self.console.print(tree)

        # Display final answer
        self.console.print(
            Panel(
                f"[bold green]Final Answer:[/bold green]\n{trace.final_answer}\n\n"
                f"[dim]Confidence: {trace.confidence_score:.2%} | "
                f"Time: {trace.total_thinking_time:.2f}s[/dim]",
                border_style="green",
            )
        )

    def _display_multimodal_results(self, result) -> None:
        """Display multi-modal analysis results."""
        self.console.print("\n[bold]Analysis Results:[/bold]\n")

        # Display modality analyses
        for modality, analysis in result.get("modality_analyses", {}).items():
            self.console.print(
                Panel(
                    f"[bold]{modality.title()} Analysis:[/bold]\n{analysis.get('analysis', 'N/A')[:200]}...",
                    border_style="blue",
                )
            )

        # Display synthesis
        self.console.print(
            Panel(
                f"[bold green]Synthesis:[/bold green]\n"
                f"{result.get('synthesis', {}).get('synthesis', 'N/A')}\n\n"
                f"[dim]Confidence: {result.get('confidence', 0):.2%}[/dim]",
                border_style="green",
            )
        )

    def _generate_agent_code(self, config: Dict[str, Any]) -> str:
        """Generate agent code based on configuration."""
        capabilities = config["capabilities"]

        code = f'''"""
Custom Agent: {config["name"]}
Generated by LlamaAgent Agent Builder
"""

from llamaagent.agents import AgentConfig'''

        if "1" in capabilities:
            code += ", AdvancedReasoningAgent"
        if "2" in capabilities:
            code += ", ReactAgent"
        if "3" in capabilities:
            code += ", AgentConfig"
        if "4" in capabilities:
            code += ", MultiModalAdvancedAgent"

        code += '''
from llamaagent.llm import create_provider
from llamaagent.tools import ToolRegistry, get_all_tools

class {config['name']}:
    """Custom agent with selected capabilities."""

    def __init__(self) -> None:
        self.config = AgentConfig(name="{config['name']}")
        self.llm_provider = create_provider("{config['provider']}")
        '''

        if "2" in capabilities:
            code += """
        # Initialize tools
        self.tools = ToolRegistry()
        for tool in get_all_tools():
            self.tools.register(tool)
        """

        if "1" in capabilities:
            code += """
        # Initialize reasoning agent
        self.reasoning_agent = AdvancedReasoningAgent(
            config=self.config,
            llm_provider=self.llm_provider,
        )
        """

        code += '''
    async def process(self, input_text: str) -> str:
        """Process input and return response."""
        # Add your custom logic here
        pass
'''

        return code

    def _view_tool_details(self, tools) -> None:
        """View detailed tool information."""
        tool_names = [t.name for t in tools]
        tool_name = Prompt.ask("Tool name", choices=tool_names)

        tool = next(t for t in tools if t.name == tool_name)

        self.console.print(
            Panel(
                f"[bold]{tool.name}[/bold]\n\n"
                f"[cyan]Description:[/cyan] {tool.description}\n"
                f"[cyan]Parameters:[/cyan] {tool.parameters}\n"
                f"[cyan]Returns:[/cyan] {tool.returns}",
                border_style="cyan",
            )
        )

    def _test_tool(self, tools) -> None:
        """Test a tool interactively."""
        tool_names = [t.name for t in tools]
        tool_name = Prompt.ask("Tool to test", choices=tool_names)

        tool = next(t for t in tools if t.name == tool_name)

        # Get parameters
        params: Dict[str, Any] = {}
        for param_name, param_info in tool.parameters.items():
            if param_info.get("required", False):
                value = Prompt.ask(f"{param_name}")
                params[param_name] = value

        # Execute tool
        with self.console.status("Executing tool..."):
            result = asyncio.run(tool.execute(**params))

        self.console.print(
            Panel(f"[bold green]Result:[/bold green]\n{result}", border_style="green")
        )

    def _create_custom_tool(self) -> None:
        """Create a custom tool interactively."""
        self.console.print("\n[bold]Create Custom Tool[/bold]\n")

        name = Prompt.ask("Tool name")
        description = Prompt.ask("Description")

        # Generate tool code
        code = f'''"""
Custom Tool: {name}
{description}
"""

from llamaagent.tools import Tool

class {name}(Tool):
    """Custom tool implementation."""

    name = "{name}"
    description = "{description}"
    parameters = {{
        "input": {{"type": "string", "required": True}}
    }}
    returns = "string"

    async def execute(self, input: str) -> str:
        """Execute tool logic."""
        # Add your implementation here
        return f"Processed: {{input}}"
'''

        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print(syntax)

        if Confirm.ask("\nSave tool code?"):
            filename = f"{name.lower()}_tool.py"
            with open(filename, "w") as f:
                f.write(code)
            self.console.print(f"[green]Saved to {filename}[/green]")

    def _save_config(self) -> None:
        """Save configuration to file."""
        config_path = Path.home() / ".llamaagent" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)


def main() -> None:
    """Main entry point for master CLI."""
    cli = LlamaAgentCLI()

    try:
        while True:
            cli.main_menu()
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]Interrupted by user[/yellow]")
        cli.exit_cli()
    except Exception as e:
        cli.console.print(f"\n[red]Error: {e}[/red]")
        if cli.config.get("debug", False):
            import traceback

            cli.console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
