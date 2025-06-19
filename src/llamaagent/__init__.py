import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from ._version import __version__
from .agents import Agent, AgentConfig, AgentRole
from .tools import ToolRegistry, get_all_tools

console = Console()


def _get_or_create_event_loop():
    """Get existing event loop or create new one if needed."""
    try:
        loop = asyncio.get_running_loop()
        return loop, False  # Existing loop, don't close
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, True  # New loop, should close


def _run_async_safe(coro):
    """Run async function safely, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
        # If we're in an event loop, create a task
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running, safe to use asyncio.run
        return asyncio.run(coro)
    except ImportError:
        # nest_asyncio not available, use thread-based approach
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()


@click.group()
@click.version_option(version=__version__)
def cli_main():
    """LlamaAgent - Advanced AI Agent Framework"""
    pass


@cli_main.command()
@click.argument("message")
@click.option("--model", default="gpt-3.5-turbo", help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--spree", is_flag=True, help="Enable SPRE planning mode")
def chat(message: str, model: str, verbose: bool, spree: bool):
    """Chat with the AI agent"""

    async def _run():
        # Create agent configuration
        config = AgentConfig(
            name="CLIAgent",
            role=AgentRole.GENERALIST,
            temperature=0.7,
            max_tokens=2000,
            spree_enabled=spree,
            metadata={"model": model},
        )

        # Create tool registry
        tools = ToolRegistry()
        for tool in get_all_tools():
            tools.register(tool)

        # Create agent with proper config
        agent = Agent(config=config, tools=tools)

        with Live(Spinner("dots", text="Processing..."), console=console, transient=True):
            response = await agent.execute(message)

        console.print(Panel(response.content, title="Response", border_style="blue"))

        if verbose:
            console.print(f"\n[dim]Execution time: {response.execution_time:.2f}s[/dim]")
            console.print(f"[dim]Tokens used: {response.tokens_used}[/dim]")
            console.print(f"[dim]Success: {response.success}[/dim]")
        else:
            console.print(f"\nExecution time: {response.execution_time:.2f}s")
            console.print(f"Tokens used: {response.tokens_used}")

        return response

    result = _run_async_safe(_run())
    return result


@cli_main.command("generate-data")
@click.argument("data_type", type=click.Choice(["gdt", "spre"]))
@click.option("-i", "--input", "input_file", required=True, help="Input file path")
@click.option("-o", "--output", "output_file", required=True, help="Output file path")
@click.option("-n", "--samples", default=100, help="Number of samples to generate")
def generate_data(data_type: str, input_file: str, output_file: str, samples: int):
    """Generate training data"""

    async def _run_generation():
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            console.print(f"[red]Error: Input file {input_file} not found[/red]")
            return False

        console.print(f"Generating {data_type.upper()} dataset...")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output_file}")
        console.print(f"Samples: {samples}")

        if data_type == "gdt":
            from .data_generation.gdt import GDTOrchestrator

            orchestrator = GDTOrchestrator()

            # Read input problems
            with open(input_path, "r") as f:
                content = f.read()

            # Simple problem extraction (you can enhance this)
            problems = [line.strip() for line in content.split("\n") if line.strip()][:samples]

            await orchestrator.generate_dataset(problems, str(output_path))

        elif data_type == "spre":
            from .data_generation.spre import generate_spre_dataset

            await generate_spre_dataset(input_path, output_path, samples)

        console.print("[green]Dataset generated successfully![/green]")
        return True

    success = _run_async_safe(_run_generation())
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
