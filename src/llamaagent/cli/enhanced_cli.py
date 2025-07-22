#!/usr/bin/env python3
"""
Enhanced Interactive CLI with Progress Bars and Llama Animation
Author: Nik Jois <nikjois@llamasearch.ai>

This module provides:
- Rich console interface with animations
- Progress bars and status indicators
- Interactive chat with llama animations
- Command handling and history
- Statistics and configuration display
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm
    from rich.table import Table
    from rich.text import Text

    rich_available = True
except ImportError:
    rich_available = False

from ..config.settings import AgentConfig
from ..core.agent import ReactAgent

logger = logging.getLogger(__name__)
console = Console() if rich_available else None


class LlamaAnimation:
    """Llama animation states for CLI."""

    def __init__(self):
        self.states = [
            """
         ∩___∩
        (  -.o  )
         > ^ <
        /|   |\\
         |___|
        """,
            """
         ∩___∩
        (  ^.^  )
         > ^ <
        /|   |\\
         |___|
        """,
            """
         ∩___∩
        (  o.-  )
         > ^ <
        /|   |\\
         |___|
        """,
            """
         ∩___∩
        (  ^o^  )
         > ^ <
        /|   |\\
         |___|
        """,
        ]
        self.current_state = 0

    def get_next_state(self) -> str:
        """Get next llama animation state."""
        self.current_state = (self.current_state + 1) % len(self.states)
        return self.states[self.current_state]

    def get_thinking_llama(self) -> str:
        """Get thinking llama animation."""
        return """
         ∩___∩
        (  -.o  )  Thinking...
         > ^ <
        /|   |\\
         |___|
        """

    def get_error_llama(self) -> str:
        """Get error llama animation."""
        return """
         ∩___∩
        (  x.x  )  Oops!
         > ^ <
        /|   |\\
         |___|
        """

    def get_success_llama(self) -> str:
        """Get success llama animation."""
        return """
         ∩___∩
        (  ^^  )  Success!
         > ^ <
        /|   |\\
         |___|
        """


class EnhancedCLI:
    """Enhanced CLI with animations and progress bars."""

    def __init__(self, spree_enabled: bool = False):
        self.spree_enabled = spree_enabled
        self.debug = False
        self.running = True
        self.llama = LlamaAnimation()
        self.conversation_history: List[Dict[str, Any]] = []
        self.agent: Optional[ReactAgent] = None
        self.config: Optional[AgentConfig] = None
        self.total_tokens = 0

        # Initialize agent
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the agent with configuration."""
        try:
            self.config = AgentConfig()
            self.agent = ReactAgent(self.config)
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            if console:
                console.print(f"[red]Failed to initialize agent: {e}[/red]")

    def show_welcome(self) -> None:
        """Show welcome message with llama."""
        if not RICH_AVAILABLE:
            print("Welcome to LlamaAgent Enhanced CLI!")
            return

        welcome_text = """
         ∩___∩
        (  ^^  )  Welcome to LlamaAgent!
         > ^ <     Type /help for commands
        /|   |\\    Type /quit to exit
         |___|
        """

        console.print(
            Panel(
                Align.center(Text(welcome_text, style="bold green")),
                title="LlamaAgent Enhanced CLI",
                border_style="green",
            )
        )

    def show_help(self) -> None:
        """Show help message."""
        if not RICH_AVAILABLE:
            print("Available commands:")
            print("  /help - Show this help message")
            print("  /quit - Exit the CLI")
            print("  /clear - Clear conversation history")
            print("  /config - Show current configuration")
            print("  /status - Show agent status")
            print("  /stats - Show usage statistics")
            print("  /history - Show conversation history")
            print("  /debug - Toggle debug mode")
            return

        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")

        commands = [
            ("/help, /h", "Show this help message"),
            ("/quit, /q", "Exit the CLI"),
            ("/clear, /c", "Clear conversation history"),
            ("/config, /cfg", "Show current configuration"),
            ("/status, /s", "Show agent status"),
            ("/stats", "Show usage statistics"),
            ("/history, /hist", "Show conversation history"),
            ("/debug", "Toggle debug mode"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)

    def show_status(self) -> None:
        """Show agent status."""
        if not RICH_AVAILABLE:
            print("Agent Status:")
            print(
                f"  Model: {self.config.llm.model if self.config else 'Not configured'}"
            )
            print(f"  Debug: {self.debug}")
            print(f"  SPREE: {self.spree_enabled}")
            return

        status_table = Table(title="Agent Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="white")
        status_table.add_column("Details", style="dim")

        # Agent status
        if self.agent:
            status = "PASS Ready"
            details = f"Model: {self.config.llm.model if self.config else 'Unknown'}"
        else:
            status = "FAIL Not initialized"
            details = "Agent not created"

        status_table.add_row("Agent", status, details)
        status_table.add_row(
            "Debug Mode", " Enabled" if self.debug else "Disabled Disabled", ""
        )
        status_table.add_row(
            "SPREE Mode",
            "Enabled Enabled" if self.spree_enabled else "Disabled Disabled",
            "",
        )

        # Tools status
        if self.agent and hasattr(self.agent, 'tools'):
            tools_count = len(self.agent.tools) if self.agent.tools else 0
            status_table.add_row(
                "Tools", f"Tools {tools_count} loaded", "All tools available"
            )
        else:
            status_table.add_row("Tools", "Tools No tools", "No tools available")

        # Conversation status
        msg_count = len(self.conversation_history)
        status_table.add_row(
            "Messages",
            f" {msg_count}",
            "History available" if msg_count > 0 else "No history",
        )

        console.print(status_table)

    def show_statistics(self) -> None:
        """Show usage statistics with visual charts."""
        if not RICH_AVAILABLE:
            print("Usage Statistics:")
            print(f"  Total Messages: {len(self.conversation_history)}")
            print(f"  Total Tokens: {self.total_tokens}")
            return

        stats_table = Table(title="Usage Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_column("Visual", style="green")

        # Calculate statistics
        total_messages = len(self.conversation_history)
        successful_messages = sum(
            1 for entry in self.conversation_history if entry.get("success", False)
        )

        success_rate = (
            (successful_messages / total_messages * 100) if total_messages > 0 else 0
        )

        # Create visual bars
        success_bar = "" * int(success_rate / 10) + "" * (10 - int(success_rate / 10))

        stats_table.add_row("Total Messages", str(total_messages), "")
        stats_table.add_row("Successful", str(successful_messages), "")
        stats_table.add_row("Success Rate", f"{success_rate:.1f}%", success_bar)
        stats_table.add_row("Total Tokens", str(self.total_tokens), "")

        if total_messages > 0:
            avg_tokens = self.total_tokens / total_messages
            stats_table.add_row("Avg Tokens/Msg", f"{avg_tokens:.1f}", "")

        console.print(stats_table)

    def show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            if console:
                console.print("[yellow]No conversation history available[/yellow]")
            else:
                print("No conversation history available")
            return

        if not RICH_AVAILABLE:
            print("Conversation History:")
            for i, entry in enumerate(self.conversation_history, 1):
                print(f"{i}. You: {entry['user_message']}")
                if entry["success"]:
                    print(f"   Agent: {entry['response']}")
                else:
                    print(f"   Error: {entry['error']}")
            return

        console.print(Panel("Conversation History", style="blue"))

        for i, entry in enumerate(self.conversation_history, 1):
            # Timestamp
            if "timestamp" in entry:
                console.print(f"[dim]{entry['timestamp']}[/dim]")

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
                    f"Latency: {entry['metadata'].get('latency_ms', 0):.1f}ms, "
                    f"Time: {entry['metadata'].get('processing_time', 0):.1f}s[/dim]"
                )

            console.print()  # Empty line between entries

    def show_config(self) -> None:
        """Show current configuration."""
        if not RICH_AVAILABLE:
            print("Current Configuration:")
            if self.config:
                print(f"  Model: {self.config.llm.model}")
                print(f"  Provider: {self.config.llm.provider}")
                print(f"  Temperature: {self.config.llm.temperature}")
            else:
                print("  No configuration available")
            return

        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")

        if self.config:
            config_table.add_row("Model", self.config.llm.model)
            config_table.add_row("Provider", self.config.llm.provider)
            config_table.add_row("Temperature", str(self.config.llm.temperature))
            config_table.add_row("Max Tokens", str(self.config.llm.max_tokens))
        else:
            config_table.add_row("Configuration", "Not available")

        console.print(config_table)

    def clear_history(self) -> None:
        """Clear conversation history."""
        if RICH_AVAILABLE:
            if Confirm.ask("Are you sure you want to clear the conversation history?"):
                self.conversation_history.clear()
                console.print("[green]Conversation history cleared[/green]")
        else:
            response = input(
                "Are you sure you want to clear the conversation history? (y/N): "
            )
            if response.lower() in ['y', 'yes']:
                self.conversation_history.clear()
                print("Conversation history cleared")

    def toggle_debug(self) -> None:
        """Toggle debug mode."""
        self.debug = not self.debug
        status = "enabled" if self.debug else "disabled"
        if console:
            console.print(f"[cyan]Debug mode {status}[/cyan]")
        else:
            print(f"Debug mode {status}")

    def handle_command(self, command: str) -> None:
        """Handle CLI commands."""
        cmd = command.strip().lower()

        if cmd in ["/help", "/h"]:
            self.show_help()
        elif cmd in ["/quit", "/q"]:
            self.show_goodbye()
            self.running = False
        elif cmd in ["/clear", "/c"]:
            self.clear_history()
        elif cmd in ["/config", "/cfg"]:
            self.show_config()
        elif cmd in ["/status", "/s"]:
            self.show_status()
        elif cmd in ["/stats"]:
            self.show_statistics()
        elif cmd in ["/history", "/hist"]:
            self.show_history()
        elif cmd.startswith("/debug"):
            self.toggle_debug()
        else:
            if console:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type [bold]/help[/bold] for available commands")
            else:
                print(f"Unknown command: {command}")
                print("Type /help for available commands")

    def show_goodbye(self) -> None:
        """Show goodbye message with llama."""
        if not RICH_AVAILABLE:
            print("Goodbye! Thanks for using LlamaAgent.")
            return

        goodbye_llama = """
         ∩___∩
        (  ^^  )  Goodbye!
         > ^ <     Thanks for using LlamaAgent.
        /|   |\\    See you next time!
         |___|
        """
        console.print(
            Panel(
                Align.center(Text(goodbye_llama, style="bold green")),
                title="See you later!",
                border_style="green",
            )
        )

    def display_debug_info(self, response: Any) -> None:
        """Display debug information."""
        if not self.debug:
            return

        if console:
            console.print("[dim]Debug Info:[/dim]")
            console.print(f"[dim]Response type: {type(response)}[/dim]")
            if hasattr(response, 'model_dump'):
                console.print(f"[dim]Response data: {response.model_dump()}[/dim]")
        else:
            print(f"Debug Info: {type(response)}")

    async def process_message(self, message: str) -> None:
        """Process user message with progress indicators."""
        if not self.agent:
            error_msg = "Agent not initialized"
            if console:
                console.print(f"[red]Error: {error_msg}[/red]")
            else:
                print(f"Error: {error_msg}")
            return

        start_time = time.time()

        try:
            if RICH_AVAILABLE:
                # Show progress with llama animation
                with console.status("[green]Processing your request...") as status:
                    status.update(f"[green]{self.llama.get_thinking_llama()}")
                    response = await self.agent.process_message(message)
            else:
                print("Processing your request...")
                response = await self.agent.process_message(message)

            processing_time = time.time() - start_time

            # Display response
            if response and hasattr(response, 'success') and response.success:
                if console:
                    console.print(f"[green]{response.response}[/green]")
                else:
                    print(response.response)

                # Store successful response
                history_entry = {
                    "user_message": message,
                    "response": response.response,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "tokens_used": getattr(response, 'tokens_used', 0),
                        "processing_time": processing_time,
                        "latency_ms": processing_time * 1000,
                    },
                }

                self.total_tokens += getattr(response, 'tokens_used', 0)
                self.display_debug_info(response)
            else:
                error_msg = getattr(response, 'error', 'Unknown error occurred')
                if console:
                    console.print(f"[red]Error: {error_msg}[/red]")
                else:
                    print(f"Error: {error_msg}")

                # Store error response
                history_entry = {
                    "user_message": message,
                    "error": error_msg,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "processing_time": processing_time,
                        "latency_ms": processing_time * 1000,
                    },
                }

                if self.debug and console:
                    console.print_exception()

            self.conversation_history.append(history_entry)

        except Exception as e:
            error_msg = str(e)
            if console:
                console.print(f"[red]Error: {error_msg}[/red]")
                if self.debug:
                    console.print_exception()
            else:
                print(f"Error: {error_msg}")

            # Store error
            history_entry = {
                "user_message": message,
                "error": error_msg,
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"processing_time": time.time() - start_time},
            }
            self.conversation_history.append(history_entry)

    async def run(self) -> None:
        """Run the enhanced CLI."""
        self.show_welcome()

        while self.running:
            try:
                if console:
                    user_input = console.input("[bold blue]You:[/bold blue] ")
                else:
                    user_input = input("You: ")

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    self.handle_command(user_input)
                else:
                    await self.process_message(user_input)

            except KeyboardInterrupt:
                if console:
                    console.print("\n[yellow]Interrupted by user[/yellow]")
                else:
                    print("\nInterrupted by user")
                break
            except EOFError:
                break
            except Exception as e:
                if console:
                    console.print(f"[red]Unexpected error: {e}[/red]")
                else:
                    print(f"Unexpected error: {e}")
                if self.debug:
                    logger.exception("Unexpected error in CLI")


def main() -> None:
    """Main entry point for the enhanced CLI."""
    parser = argparse.ArgumentParser(description="LlamaAgent Enhanced CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--spree", action="store_true", help="Enable SPREE mode")
    parser.add_argument(
        "--no-rich", action="store_true", help="Disable rich formatting"
    )

    args = parser.parse_args()

    # Override rich availability if requested
    if args.no_rich:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run CLI
    cli = EnhancedCLI(spree_enabled=args.spree)
    cli.debug = args.debug

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
