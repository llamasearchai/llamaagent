#!/usr/bin/env python3
"""
Enhanced interactive CLI with comprehensive features and error handling.

Author: Nik Jois, Email: nikjois@llamasearch.ai
"""

import argparse
import asyncio
import signal
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from ..agents import ReactAgent
from ..config import get_config

# Optional imports with fallbacks
try:
    from ..agents import AgentConfig
except ImportError:
    AgentConfig = None

console = Console()


class InteractiveCLI:
    """Interactive CLI for LlamaAgent with comprehensive features."""

    def __init__(self, spree_enabled: bool = False, debug: bool = False) -> None:
        self.spree_enabled = spree_enabled
        self.debug = debug
        self.config = get_config()
        self.agent: Optional[ReactAgent] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle system signals."""
        console.print(
            "\n[yellow]Received interrupt signal. Shutting down gracefully...[/yellow]"
        )
        self.running = False

    async def initialize(self) -> None:
        """Initialize the CLI and agent."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing LlamaAgent...", total=None)

            try:
                # Create agent with basic configuration
                self.agent = ReactAgent(
                    name="Interactive-Agent",
                    description="Interactive CLI agent",
                )

                progress.update(task, description="Agent initialized successfully")

            except Exception as e:
                progress.update(task, description=f"Failed to initialize: {e}")
                console.print(f"[red]Error initializing agent: {e}[/red]")
                raise

    def display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
[bold blue]Welcome to LlamaAgent Interactive CLI![/bold blue]

Type your questions or commands below. Available commands:
• [bold]/help[/bold] - Show help information
• [bold]/history[/bold] - Show conversation history
• [bold]/clear[/bold] - Clear conversation history
• [bold]/config[/bold] - Show current configuration
• [bold]/status[/bold] - Show agent status
• [bold]/debug[/bold] - Toggle debug mode
• [bold]/spree[/bold] - Toggle SPREE mode
• [bold]/quit[/bold] - Exit the CLI

Ready to chat!
        """
        console.print(Panel(welcome_text, title="LlamaAgent CLI", border_style="blue"))

    async def run(self) -> None:
        """Main interactive loop."""
        await self.initialize()
        self.display_welcome()

        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue

                # Process user message
                await self.process_message(user_input)

            except KeyboardInterrupt:
                if Confirm.ask("Are you sure you want to exit?"):
                    break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if self.debug:
                    console.print_exception()

    async def handle_command(self, command: str) -> None:
        """Handle CLI commands."""
        cmd = command.lower().strip()

        if cmd in ["/help", "/h"]:
            self.show_help()
        elif cmd in ["/history", "/hist"]:
            self.show_history()
        elif cmd in ["/clear", "/c"]:
            self.clear_history()
        elif cmd in ["/config", "/cfg"]:
            self.show_config()
        elif cmd in ["/status", "/s"]:
            await self.show_status()
        elif cmd in ["/debug"]:
            self.toggle_debug()
        elif cmd in ["/spree"]:
            self.toggle_spree()
        elif cmd in ["/quit", "/q"]:
            self.running = False
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Type [bold]/help[/bold] for available commands")

    def show_help(self) -> None:
        """Show help information."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")

        help_table.add_row("/help, /h", "Show this help message")
        help_table.add_row("/history, /hist", "Show conversation history")
        help_table.add_row("/clear, /c", "Clear conversation history")
        help_table.add_row("/config, /cfg", "Show current configuration")
        help_table.add_row("/status, /s", "Show agent status")
        help_table.add_row("/debug", "Toggle debug mode")
        help_table.add_row("/spree", "Toggle SPREE mode")
        help_table.add_row("/quit, /q", "Exit the CLI")

        console.print(help_table)

    def show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            console.print("[yellow]No conversation history available[/yellow]")
            return

        console.print(
            Panel("Conversation History", title="History", border_style="green")
        )

        for i, entry in enumerate(self.conversation_history, 1):
            # User message
            console.print(f"[bold blue]{i}. You:[/bold blue]")
            console.print(entry["user_message"])

            # Agent response
            console.print("[bold green]Agent:[/bold green]")
            if entry["success"]:
                console.print(entry["response"])
            else:
                console.print(f"[red]Error: {entry['error']}[/red]")

            # Metadata
            if entry.get("metadata"):
                console.print(
                    f"[dim]Tokens: {entry['metadata'].get('tokens_used', 0)}, "
                    f"Latency: {entry['metadata'].get('latency_ms', 0):.1f}ms[/dim]"
                )

    def clear_history(self) -> None:
        """Clear conversation history."""
        if Confirm.ask("Are you sure you want to clear the conversation history?"):
            self.conversation_history.clear()
            console.print("[green]Conversation history cleared[/green]")

    def show_config(self) -> None:
        """Show current configuration."""
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")

        config_table.add_row("Debug Mode", "Enabled" if self.debug else "Disabled")
        config_table.add_row(
            "SPREE Mode", "Enabled" if self.spree_enabled else "Disabled"
        )
        config_table.add_row("LLM Provider", self.config.llm.provider)
        config_table.add_row("Model", self.config.llm.model)
        config_table.add_row("Temperature", str(self.config.llm.temperature))

        console.print(config_table)

    async def show_status(self) -> None:
        """Show agent status."""
        status_table = Table(title="Agent Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="white")

        # Agent status
        if self.agent:
            try:
                status = "Active"
                details = f"Model: {self.config.llm.model}"
            except Exception as e:
                status = "Error"
                details = str(e)
        else:
            status = "Not initialized"
            details = "Agent not created"

        status_table.add_row("Agent", status)
        status_table.add_row("Details", details)

        # Tool status
        if hasattr(self.agent, 'tool_registry') and self.agent.tool_registry:
            status_table.add_row("Tools", "All tools loaded")
        else:
            status_table.add_row("Tools", "No tools available")

        # Conversation status
        msg_count = len(self.conversation_history)
        status_table.add_row("Messages", f"{msg_count} in history")

        if msg_count > 0:
            last_msg = self.conversation_history[-1]
            status_table.add_row(
                "Last Message", "Success" if last_msg["success"] else "Error"
            )
        else:
            status_table.add_row("Last Message", "None")

        console.print(status_table)

    def toggle_debug(self) -> None:
        """Toggle debug mode."""
        self.debug = not self.debug
        console.print(
            f"[green]Debug mode {'enabled' if self.debug else 'disabled'}[/green]"
        )

    def toggle_spree(self) -> None:
        """Toggle SPREE mode."""
        self.spree_enabled = not self.spree_enabled
        console.print(
            f"[green]SPREE mode {'enabled' if self.spree_enabled else 'disabled'}[/green]"
        )

    async def process_message(self, user_input: str) -> None:
        """Process user message and generate response."""
        if not self.agent:
            console.print("[red]Agent not initialized. Please restart the CLI.[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Agent is thinking...", total=None)

            try:
                # Create task input
                task_input = {
                    "task": user_input,
                    "context": {
                        "conversation_history": (
                            self.conversation_history[-10:]
                            if len(self.conversation_history) > 10
                            else self.conversation_history
                        ),
                        "spree_enabled": self.spree_enabled,
                        "debug": self.debug,
                    },
                }

                # Execute task
                response = await self.agent.execute_task(task_input)

                progress.update(task, description="Processing response...")

                # Process response
                if response and hasattr(response, 'result'):
                    result_text = response.result.get(
                        "response", "No response generated"
                    )
                    success = response.status.name == "SUCCESS"
                    error_msg = None
                else:
                    result_text = str(response) if response else "No response"
                    success = True
                    error_msg = None

                # Store in history
                history_entry = {
                    "user_message": user_input,
                    "response": result_text,
                    "success": success,
                    "error": error_msg,
                    "metadata": {
                        "tokens_used": getattr(response, 'tokens_used', 0),
                        "latency_ms": getattr(response, 'latency_ms', 0),
                    },
                    "timestamp": asyncio.get_event_loop().time(),
                }

                self.conversation_history.append(history_entry)

                # Display response
                self.display_response(result_text)

                # Display debug info if enabled
                if self.debug:
                    self.display_debug_info(response)

            except Exception as e:
                error_msg = f"Error processing message: {e}"
                console.print(f"[red]{error_msg}[/red]")

                # Store error in history
                history_entry = {
                    "user_message": user_input,
                    "response": "",
                    "success": False,
                    "error": error_msg,
                    "metadata": {"tokens_used": 0, "latency_ms": 0},
                    "timestamp": asyncio.get_event_loop().time(),
                }
                self.conversation_history.append(history_entry)

                if self.debug:
                    console.print_exception()

    def display_response(self, content: str) -> None:
        """Display response with syntax highlighting for code blocks."""
        parts = content.split("```")

        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text
                console.print(part.strip())
            else:
                # Code block
                lines = part.split("\n")
                language = lines[0].strip() if lines else "text"
                code = "\n".join(lines[1:]) if language != "text" else part.strip()

                syntax = Syntax(
                    code, language or "text", theme="monokai", line_numbers=True
                )
                console.print(syntax)

    def display_debug_info(self, response: Any) -> None:
        """Display debug information."""
        if not response:
            return

        debug_table = Table(title="Debug Information")
        debug_table.add_column("Metric", style="cyan")
        debug_table.add_column("Value", style="white")

        debug_table.add_row("Response Type", type(response).__name__)
        debug_table.add_row("Status", getattr(response, 'status', 'Unknown'))
        debug_table.add_row("Tokens Used", str(getattr(response, 'tokens_used', 0)))
        debug_table.add_row("Latency", f"{getattr(response, 'latency_ms', 0):.1f}ms")

        if hasattr(response, 'metadata') and response.metadata:
            for key, value in response.metadata.items():
                debug_table.add_row(f"Metadata: {key}", str(value))

        console.print(debug_table)


async def run_interactive_experiment(
    spree_enabled: bool = False, debug: bool = False
) -> None:
    """Entry-point helper for run_experiment.py.

    The historical public API expects an async run_interactive_experiment
    coroutine that sets up an InteractiveCLI instance and drives its
    event-loop. This wrapper keeps that contract while re-using the
    implementation that already exists inside InteractiveCLI.run.
    """
    cli = InteractiveCLI(spree_enabled=spree_enabled, debug=debug)
    await cli.run()


def main() -> None:
    """Main entry point for the interactive CLI."""
    parser = argparse.ArgumentParser(description="LlamaAgent Interactive CLI")
    parser.add_argument("--spree", action="store_true", help="Enable SPREE mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Run CLI
    cli = InteractiveCLI(spree_enabled=args.spree, debug=args.debug)

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    main()
