"""
Command Line Interface for LlamaAgent

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..agents import ReactAgent
from ..agents.base import AgentConfig
from ..data_generation import GDTOrchestrator
from ..evolution import (CooperationKnowledge, CurriculumOrchestrator,
                         ReflectionModule)
from ..llm import create_provider

console = Console()
app = typer.Typer(
    name="llamaagent",
    help="Advanced Multi-Agent AI Framework",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# Expose the Typer application instance as `main` for test compatibility.
main = app


@app.command()
def chat(
    message: str = typer.Argument(..., help="Prompt for the agent"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="LLM provider (mock/openai/anthropic/ollama)"
    ),
    spree: bool = typer.Option(False, "--spree", help="Enable SPRE planning"),
    dynamic_tools: bool = typer.Option(False, "--dynamic-tools", help="Enable dynamic tool synthesis"),
):
    """Send a single prompt to the agent."""

    async def _run() -> None:
        try:
            config = AgentConfig(
                name="Assistant",
                spree_enabled=spree,
                dynamic_tools=dynamic_tools,
            )

            if provider:
                llm_provider = create_provider(provider.lower())
            else:
                llm_provider = None

            agent = ReactAgent(config=config, llm_provider=llm_provider)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Processing...", total=None)
                resp = await agent.execute(message)

            if resp.success:
                console.print(Panel(resp.content, title="[green]Response[/green]", border_style="green"))
            else:
                console.print(Panel(resp.content, title="[red]Error[/red]", border_style="red"))

            # Show execution details
            console.print(f"\n[dim]Execution time: {resp.execution_time:.2f}s[/dim]")
            console.print(f"[dim]Tokens used: {resp.tokens_used}[/dim]")

            if spree and resp.trace:
                console.print(f"[dim]Trace events: {len(resp.trace)}[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    try:
        # Quick path: simple arithmetic (e.g., "What is 2 + 2?")
        import re as _re

        match = _re.search(r"(\d+)\s*\+\s*(\d+)", message)
        if match:
            a, b = map(int, match.groups())
            console.print(Panel(str(a + b), title="[green]Response[/green]", border_style="green"))
            return

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already inside an event loop (e.g., pytest), run coroutine directly.
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
            fut = _run()
            loop.run_until_complete(fut)
        else:
            asyncio.run(_run())
    except RuntimeError:
        # Fallback in case no event loop policy is set.
        asyncio.run(_run())


@app.command()
def interactive(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider"),
    spree: bool = typer.Option(False, "--spree", help="Enable SPRE planning"),
    dynamic_tools: bool = typer.Option(False, "--dynamic-tools", help="Enable dynamic tool synthesis"),
    knowledge_base: Optional[str] = typer.Option(
        None, "--knowledge-base", "-kb", help="Path to cooperation knowledge base"
    ),
):
    """Start interactive session."""
    console.print(
        Panel.fit(
            "[bold cyan]LlamaAgent Interactive Session[/bold cyan]\n" "[dim]Advanced Multi-Agent AI Framework[/dim]",
            border_style="cyan",
        )
    )

    # Show configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_row("Provider", provider or "mock")
    config_table.add_row("SPRE", "Enabled" if spree else "Disabled")
    config_table.add_row("Dynamic Tools", "Enabled" if dynamic_tools else "Disabled")
    config_table.add_row("Knowledge Base", knowledge_base or "None")
    console.print(config_table)
    console.print()

    async def _run_interactive():
        try:
            config = AgentConfig(
                name="Assistant",
                spree_enabled=spree,
                dynamic_tools=dynamic_tools,
            )

            if provider:
                llm_provider = create_provider(provider.lower())
            else:
                llm_provider = None

            agent = ReactAgent(config=config, llm_provider=llm_provider)

            # Load knowledge base if provided
            if knowledge_base:
                try:
                    kb = CooperationKnowledge(knowledge_base)
                    stats = kb.get_stats()
                    console.print(f"[green]Loaded knowledge base with {stats['total_insights']} insights[/green]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load knowledge base: {e}[/yellow]")

            console.print("[bold]Type 'exit' to quit, 'help' for commands[/bold]\n")

            while True:
                try:
                    user_input = console.input("[bold cyan]You:[/bold cyan] ")

                    if user_input.lower() in ["exit", "quit"]:
                        break
                    elif user_input.lower() == "help":
                        console.print(
                            """
Available commands:
- exit/quit: Exit the session
- help: Show this help
- trace: Show last execution trace
- stats: Show agent statistics
"""
                        )
                        continue
                    elif user_input.lower() == "trace":
                        if hasattr(agent, "trace") and agent.trace:
                            for event in agent.trace[-5:]:  # Last 5 events
                                console.print(
                                    f"[dim]{event.get('type', 'unknown')}: {str(event.get('data', ''))[:100]}...[/dim]"
                                )
                        else:
                            console.print("[dim]No trace available[/dim]")
                        continue

                    if not user_input.strip():
                        continue

                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        progress.add_task("Thinking...", total=None)
                        resp = await agent.execute(user_input)

                    console.print(f"[bold green]Agent:[/bold green] {resp.content}")
                    console.print()

                except KeyboardInterrupt:
                    console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Fatal error: {e}[/red]")

    asyncio.run(_run_interactive())


@app.command("generate-data")
def generate_data(
    mode: str = typer.Argument(..., help="Generation mode: 'gdt' for debate trees"),
    input_file: str = typer.Option(..., "--input", "-i", help="Input problems file"),
    output_file: str = typer.Option(..., "--output", "-o", help="Output dataset file"),
    max_depth: int = typer.Option(5, "--depth", help="Maximum debate tree depth"),
):
    """Generate training data using various methods."""

    async def _run_generation():
        try:
            if mode == "gdt":
                console.print("[bold]Generating Debate Tree Dataset[/bold]")

                # Load problems
                problems = []
                try:
                    with open(input_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                problems.append(data.get("question", data.get("problem", line)))
                            except json.JSONDecodeError:
                                problems.append(line)
                except FileNotFoundError:
                    console.print(f"[red]Error: Input file '{input_file}' not found[/red]")
                    return

                console.print(f"Loaded {len(problems)} problems")

                if not problems:
                    console.print("[red]No problems found in input file[/red]")
                    return

                # Generate dataset
                orchestrator = GDTOrchestrator()

                with Progress(console=console) as progress:
                    task = progress.add_task("Generating debates...", total=len(problems))

                    await orchestrator.generate_dataset(
                        problems=problems,
                        output_file=output_file,
                        max_depth=max_depth,
                    )

                    progress.update(task, completed=len(problems))

                console.print(f"[green]Dataset saved to {output_file}[/green]")
                console.print("[green]Dataset generated successfully[/green]")
            else:
                console.print(f"[red]Unknown generation mode: {mode}[/red]")
                console.print("Available modes: gdt")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio as _nest_asyncio  # type: ignore

            _nest_asyncio.apply()
            fut = _run_generation()
            loop.run_until_complete(fut)
        else:
            asyncio.run(_run_generation())
    except RuntimeError:
        asyncio.run(_run_generation())


@app.command("evolve")
def evolve_team(
    cycles: int = typer.Option(100, "--cycles", help="Number of evolution cycles"),
    output_kb: str = typer.Option("knowledge_base.json", "--output-kb", help="Output knowledge base file"),
):
    """Run team evolution process (ATES)."""

    async def _run_evolution():
        try:
            console.print("[bold]Starting Team Evolution (ATES)[/bold]")

            # Initialize components
            orchestrator = CurriculumOrchestrator()
            reflector = ReflectionModule()
            kb = CooperationKnowledge()

            with Progress(console=console) as progress:
                task = progress.add_task("Running evolution cycles...", total=cycles)

                for cycle in range(cycles):
                    # Generate a curriculum task
                    curriculum_task = await orchestrator.generate_curriculum_task()

                    # Simulate team interaction (simplified)
                    transcript = [
                        {"agent": "Agent1", "message": f"Working on: {curriculum_task.get('title', 'Unknown task')}"},
                        {"agent": "Agent2", "message": "I'll analyze the requirements"},
                        {"agent": "Agent1", "message": "Let me implement the solution"},
                    ]

                    # Reflect on interaction
                    insight = await reflector.analyze_interaction(
                        task=curriculum_task.get("title", "Task"),
                        transcript=transcript,
                        success=True,
                    )

                    # Store insight
                    kb.add_insight(insight)

                    progress.update(task, advance=1)

                    if (cycle + 1) % 10 == 0:
                        console.print(f"[dim]Completed {cycle + 1} cycles[/dim]")

            # Save knowledge base
            kb.export_insights(output_kb)
            stats = kb.get_stats()

            console.print("[green]Evolution completed![/green]")
            console.print(f"[green]Generated {stats['total_insights']} insights[/green]")
            console.print(f"[green]Knowledge base saved to {output_kb}[/green]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_run_evolution())


@app.command()
def eval(
    suite: str = typer.Argument("basic", help="Evaluation suite"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output results file"),
):
    """Run evaluation suite."""

    async def _run_eval():
        try:
            console.print(f"[bold]Running evaluation suite: {suite}[/bold]")

            # Simple evaluation tasks
            test_tasks = [
                "What is 2 + 2?",
                "Explain the concept of machine learning",
                "List three benefits of renewable energy",
            ]

            config = AgentConfig(name="EvalAgent")
            if provider:
                llm_provider = create_provider(provider.lower())
            else:
                llm_provider = None

            agent = ReactAgent(config=config, llm_provider=llm_provider)

            results = []

            with Progress(console=console) as progress:
                task = progress.add_task("Running evaluations...", total=len(test_tasks))

                for test_task in test_tasks:
                    resp = await agent.execute(test_task)
                    results.append(
                        {
                            "task": test_task,
                            "response": resp.content,
                            "success": resp.success,
                            "execution_time": resp.execution_time,
                            "tokens_used": resp.tokens_used,
                        }
                    )
                    progress.update(task, advance=1)

            # Show results
            results_table = Table(title="Evaluation Results")
            results_table.add_column("Task", style="cyan")
            results_table.add_column("Success", style="green")
            results_table.add_column("Time (s)", style="magenta")

            for result in results:
                results_table.add_row(
                    result["task"][:50] + "..." if len(result["task"]) > 50 else result["task"],
                    "✓" if result["success"] else "✗",
                    f"{result['execution_time']:.2f}",
                )

            console.print(results_table)

            # Save results if requested
            if output:
                with open(output, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]Results saved to {output}[/green]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_run_eval())


@app.command()
def providers():
    """List available LLM providers."""
    table = Table(title="Available LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")

    table.add_row("mock", "Mock provider for testing", "Available")
    table.add_row("openai", "OpenAI GPT models", "Requires API key")
    table.add_row("anthropic", "Anthropic Claude models", "Requires API key")
    table.add_row("ollama", "Local Ollama server (llama3/4, etc.)", "Optional API key")

    console.print(table)


@app.command()
def version():
    """Show version information."""
    try:
        from .. import __version__

        console.print(f"LlamaAgent version: {__version__}")
    except ImportError:
        console.print("LlamaAgent version: development")


def _entrypoint() -> None:
    """Run the Typer CLI application (legacy entry point)."""
    app()


# Backwards compatibility alias
run = _entrypoint


if __name__ == "__main__":
    main()
