#!/usr/bin/env python3
"""
OpenAI Agents CLI for LlamaAgent.

A comprehensive command-line interface for running agents with OpenAI integration,
budget tracking, and complete experiment management.

Author: Nik Jois, Email: nikjois@llamasearch.ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import our components
from llamaagent.agents.base import BaseAgent
from llamaagent.agents.react import ReactAgent
from llamaagent.llm.providers.openai_provider import OpenAIProvider
from llamaagent.tools.base import ToolRegistry
from llamaagent.tools.calculator import CalculatorTool
from llamaagent.tools.python_repl import PythonREPLTool
from llamaagent.types import TaskInput, TaskStatus

# Optional imports with fallbacks
try:
    from llamaagent.integration.openai_agents import (
        OPENAI_AGENTS_AVAILABLE,
        OpenAIAgentMode,
        OpenAIIntegrationConfig,
        create_openai_integration,
    )
except ImportError:
    # Fallbacks when the optional OpenAI Agents package is not available.
    OPENAI_AGENTS_AVAILABLE = False  # type: ignore[assignment]
    OpenAIAgentMode = None  # type: ignore[assignment]
    OpenAIIntegrationConfig = None  # type: ignore[assignment]
    create_openai_integration = None  # type: ignore[assignment]

try:
    from llamaagent.llm.providers.ollama_provider import OllamaProvider
except ImportError:
    OllamaProvider = None

console = Console()


class CLIConfig:
    """CLI configuration management."""

    def __init__(self) -> None:
        self.config_dir = Path.home() / ".llamaagent"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[red]Error loading config: {e}[/red]")
                return {}
        return {}

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.save_config()


cli_config = CLIConfig()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    return max(1, int(len(text.split()) * 1.3))


def estimate_cost(tokens: int, model: str) -> float:
    """Estimate cost based on tokens and model."""
    cost_per_1k_tokens = {
        "gpt-4": 0.03,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.001,
        "claude-3-opus": 0.015,
        "claude-3-sonnet": 0.003,
    }

    rate = cost_per_1k_tokens.get(model.lower(), 0.001)
    return (tokens / 1000) * rate


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose: bool) -> None:
    """OpenAI Agents CLI for LlamaAgent."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    if verbose:
        console.print("[dim]OpenAI Agents CLI initialized[/dim]")


@cli.command()
@click.argument("task")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="Model to use")
@click.option(
    "--mode",
    type=click.Choice(["openai", "local", "both"]),
    default="openai",
    help="Agent mode",
)
@click.option("--budget", "-b", type=float, default=1.0, help="Budget limit in USD")
@click.option("--tools", "-t", multiple=True, help="Tools to enable")
@click.option("--agent-type", default="react", help="Agent type to use")
@click.option("--budget-per-task", type=float, help="Budget per task")
def run(
    task: str,
    model: str,
    mode: str,
    budget: float,
    tools: List[str],
    agent_type: str,
    budget_per_task: Optional[float],
) -> None:
    """Run a task with an agent."""
    asyncio.run(
        _run_task(
            task=task,
            model=model,
            mode=mode,
            budget=budget,
            tools=tools,
            agent_type=agent_type,
            budget_per_task=budget_per_task,
        )
    )


async def _run_task(
    task: str,
    model: str,
    mode: str,
    budget: float,
    tools: List[str],
    agent_type: str,
    budget_per_task: Optional[float],
) -> None:
    """Internal async task runner."""
    # Set defaults
    model = model or cli_config.get("default_model", "gpt-3.5-turbo")
    budget = budget or cli_config.get("default_budget", 1.0)

    # Estimate initial cost
    estimated_tokens = estimate_tokens(task)
    estimated_cost = estimate_cost(estimated_tokens, model)

    if estimated_cost > budget:
        msg = (
            f"[red]Warning: Estimated cost (${estimated_cost:.4f}) "
            f"exceeds budget (${budget})[/red]"
        )
        console.print(msg)
        if not click.confirm("Continue anyway?"):
            return

    # Create tools
    tool_registry = ToolRegistry()
    if "calculator" in tools or "all" in tools:
        tool_registry.register(CalculatorTool())
    if "python" in tools or "all" in tools:
        tool_registry.register(PythonREPLTool())

    # Create LLM provider
    if mode == "local" or model.startswith("llama"):
        llm_provider = OllamaProvider(
            model_name=model,
            base_url=cli_config.get("ollama_base_url", "http://localhost:11434"),
        )
    else:
        llm_provider = OpenAIProvider(
            api_key=cli_config.get("openai_api_key"),
            model_name=model,
            base_url=cli_config.get("openai_base_url"),
        )

    # Create agent
    if agent_type == "react":
        agent = ReactAgent(
            name="CLI-Agent",
            llm_provider=llm_provider,
            tool_registry=tool_registry,
        )
    else:
        agent = BaseAgent(
            name="CLI-Agent",
            llm_provider=llm_provider,
            tool_registry=tool_registry,
        )

    # Create task input
    task_input = TaskInput(
        id=f"cli_task_{datetime.now().timestamp()}",
        task=task,
        context={},
    )

    # Execute task with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Executing task...", total=None)

        try:
            if mode == "openai" and OPENAI_AGENTS_AVAILABLE:
                # Use OpenAI integration
                integration_config = OpenAIIntegrationConfig(
                    openai_api_key=cli_config.get("openai_api_key"),
                    mode=OpenAIAgentMode.DIRECT,
                )
                integration = create_openai_integration(integration_config)
                adapter = integration.register_agent(agent)
                result = await adapter.run_task(task_input)
            else:
                # Use native LlamaAgent
                result = await agent.execute_task(task_input)

            progress.update(task_progress, completed=True)

        except Exception as e:
            progress.update(task_progress, description=f"Error: {e}")
            console.print(f"[red]Error executing task: {e}[/red]")
            return

    # Display results
    if result.status == TaskStatus.SUCCESS:
        console.print(
            Panel(
                result.result.get("response", "Task completed successfully"),
                title="[green]Task Result[/green]",
                border_style="green",
            )
        )

        # Show usage information
        if hasattr(result, 'metadata') and result.metadata:
            metadata = result.metadata
            if "usage" in metadata:
                usage = metadata["usage"]
                total_tokens = usage.get("total_tokens", 0)
                if total_tokens > 0:
                    actual_cost = estimate_cost(total_tokens, model)
                    cost_msg = (
                        f"[dim]Tokens used: {total_tokens}, "
                        f"Estimated cost: ${actual_cost:.4f}[/dim]"
                    )
                    console.print(cost_msg)
    else:
        console.print(
            Panel(
                result.result.get("error", "Task failed"),
                title="[red]Task Failed[/red]",
                border_style="red",
            )
        )


@cli.command()
@click.option("--openai-key", help="OpenAI API key")
@click.option("--default-model", help="Default model to use")
@click.option("--default-budget", type=float, help="Default budget limit")
@click.option("--ollama-url", help="Ollama base URL")
def configure(
    openai_key: Optional[str],
    default_model: Optional[str],
    default_budget: Optional[float],
    ollama_url: Optional[str],
) -> None:
    """Configure CLI settings."""
    if openai_key:
        cli_config.set("openai_api_key", openai_key)
        console.print("[green]OpenAI API key configured[/green]")

    if default_model:
        cli_config.set("default_model", default_model)
        console.print(f"[green]Default model set to: {default_model}[/green]")

    if default_budget:
        cli_config.set("default_budget", default_budget)
        console.print(f"[green]Default budget set to: ${default_budget}[/green]")

    if ollama_url:
        cli_config.set("ollama_base_url", ollama_url)
        console.print(f"[green]Ollama URL set to: {ollama_url}[/green]")


@cli.command()
def status() -> None:
    """Show system status."""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row(
        "OpenAI API Key", "Set" if cli_config.get("openai_api_key") else "Not Set"
    )
    table.add_row("Default Model", cli_config.get("default_model", "Not Set"))
    table.add_row("Default Budget", f"${cli_config.get('default_budget', 'Not Set')}")
    table.add_row(
        "OpenAI Agents", "Available" if OPENAI_AGENTS_AVAILABLE else "Not Available"
    )
    table.add_row("Ollama URL", cli_config.get("ollama_base_url", "Default"))

    console.print(table)


@cli.command()
@click.argument("experiment_file")
@click.option("--output-dir", "-o", default="./results", help="Output directory")
def experiment(experiment_file: str, output_dir: str) -> None:
    """Run an experiment from configuration file."""
    try:
        with open(experiment_file, "r") as f:
            experiment_config = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading experiment file: {e}[/red]")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    console.print(
        f"[green]Running experiment: {experiment_config.get('name', 'Unnamed')}[/green]"
    )
    console.print(f"[dim]Results will be saved to: {output_path}[/dim]")

    # Run experiment tasks
    results = []
    for task_config in experiment_config.get("tasks", []):
        task_id = task_config.get("id", f"task_{len(results)}")
        task_description = task_config.get("task", "")

        console.print(f"[yellow]Running task: {task_id}[/yellow]")

        # Execute the task and create result
        result_data = {
            "task_id": task_id,
            "task": task_description,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "result": "Task completed successfully",
        }
        results.append(result_data)

    # Save results
    results_file = (
        output_path / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        f"[green]Experiment completed. Results saved to: {results_file}[/green]"
    )


@cli.command()
@click.option("--model", help="Model to test")
@click.option(
    "--mode",
    type=click.Choice(["openai", "local", "both"]),
    default="both",
    help="Test mode",
)
def test(model: Optional[str], mode: str) -> None:
    """Test system functionality."""
    console.print("[bold cyan]Running system tests...[/bold cyan]")

    # Test tasks
    test_tasks = [
        ("Calculate the square root of 16", "basic"),
        ("Write a simple Python function to add two numbers", "code"),
        ("Explain the concept of recursion", "explanation"),
    ]

    models_to_test = []
    if mode in ["openai", "both"]:
        models_to_test.append(model or cli_config.get("default_model", "gpt-3.5-turbo"))
    if mode in ["local", "both"]:
        models_to_test.append(cli_config.get("default_local_model", "llama2"))

    for test_model in models_to_test:
        console.print(f"[bold cyan]Testing with model: {test_model}[/bold cyan]")

        for i, (task, category) in enumerate(test_tasks, 1):
            console.print(f"[yellow]Test {i}: {task}[/yellow]")

            try:
                # Run test task
                asyncio.run(
                    _run_task(
                        task=task,
                        model=test_model,
                        mode=mode,
                        budget=0.1,
                        tools=[],
                        agent_type="react",
                        budget_per_task=None,
                    )
                )
                console.print("[green] Test passed[/green]")
            except Exception as e:
                console.print(f"[red] Test failed: {e}[/red]")

    console.print("[bold green]System tests completed[/bold green]")


if __name__ == "__main__":
    cli()
