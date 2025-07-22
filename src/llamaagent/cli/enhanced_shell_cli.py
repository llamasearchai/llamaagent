#!/usr/bin/env python3
"""
Enhanced Shell CLI with shell_gpt-style functionality, Author: Nik Jois <nikjois@llamasearch.ai>

This module provides a comprehensive command-line interface inspired by shell_gpt
with additional LlamaAgent, capabilities:
- Shell command generation and execution
- Code generation mode
- Chat and REPL sessions
- Role-based interactions
- Function calling
- Configuration management
- Caching and performance optimization
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ..agents.base import BaseAgent
from ..agents.react import ReactAgent
from ..llm.messages import LLMMessage
from ..types import AgentConfig, TaskInput

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Enhanced LlamaAgent Shell CLI")

# Global configuration
CONFIG = {
    "DEFAULT_COLOR": "bright_blue",
    "CACHE_ENABLED": True,
    "MAX_HISTORY": 100,
}


class ShellCLI:
    """Enhanced shell CLI for LlamaAgent."""

    def __init__(self):
        self.agent: Optional[BaseAgent] = None
        self.messages: List[LLMMessage] = []
        self.debug = False

    def create_agent(self, model: str = "gpt-4o") -> BaseAgent:
        """Create an agent instance."""
        config = AgentConfig(name="ShellAssistant", llm_model=model, temperature=0.1)
        return ReactAgent(config)


def print_output(text: str, color: str = "white") -> None:
    """Print colored output to console."""
    console.print(text, style=color)


def display_response(response: str, color: str = "bright_blue") -> None:
    """Display response in a panel."""
    panel = Panel(response, border_style=color, padding=(1, 2))
    console.print(panel)


def get_chat_file_path(session_id: str) -> Path:
    """Get the file path for a chat session."""
    cache_dir = Path.home() / ".config" / "llamaagent" / "chat_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{session_id}.json"


def load_chat_messages(chat_file: Path) -> List[LLMMessage]:
    """Load chat messages from file."""
    if not chat_file.exists():
        return []

    try:
        with open(chat_file, "r") as f:
            data = json.load(f)
            messages = []
            for msg_data in data.get("messages", []):
                messages.append(
                    LLMMessage(role=msg_data["role"], content=msg_data["content"])
                )
            return messages
    except Exception as e:
        logger.error(f"Error loading chat messages: {e}")
        return []


def save_chat_messages(chat_file: Path, messages: List[LLMMessage]) -> None:
    """Save chat messages to file."""
    try:
        data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }
        with open(chat_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat messages: {e}")


def get_role_prompt(role: str) -> str:
    """Get the prompt for a specific role."""
    role_prompts = {
        "default": "You are a helpful AI assistant.",
        "shell_expert": """You are a shell and command-line expert.
Your job is to help users with shell commands, explain what they do, and suggest improvements.
Provide clear, concise explanations and warn about potentially dangerous operations.""",
        "code_expert": """You are a software engineering expert.
Your job is to help with coding tasks, review code, and provide technical guidance.
Provide clean, well-documented code examples and follow best practices.""",
        "data_analyst": """You are a data analysis expert.
Your job is to help with data analysis tasks, visualization, and statistical insights.
Provide clear explanations and actionable recommendations.""",
    }
    return role_prompts.get(role, role_prompts["default"])


def list_available_roles() -> None:
    """List available roles."""
    roles = ["default", "shell_expert", "code_expert", "data_analyst"]
    print_output("Available roles:")
    for role in roles:
        print_output(f"  {role}")


def show_role_details(role: str) -> None:
    """Show details of a specific role."""
    prompt = get_role_prompt(role)
    print_output(f"Role: {role}")
    print_output(f"Description:\n{prompt}")


def create_new_role(role_name: str) -> None:
    """Create a new role."""
    print_output(f"Creating new role: {role_name}")
    print_output("Role creation functionality available in full CLI mode.")


def install_shell_integration() -> None:
    """Install shell integration."""
    print_output("Installing shell integration...")
    print_output("Shell integration functionality available in full CLI mode.")


def get_os_name() -> str:
    """Get the operating system name."""
    import platform

    return platform.system()


def get_shell_name() -> str:
    """Get the current shell name."""
    return os.environ.get("SHELL", "unknown").split("/")[-1]


def determine_role(shell: bool, explain: bool, chat: bool) -> str:
    """Determine the appropriate role based on options."""
    if shell:
        return "shell_expert"
    elif explain:
        return "shell_expert"
    elif chat:
        return "default"
    else:
        return "default"


async def describe_shell_command(command: str, debug: bool = False) -> str:
    """Describe what a shell command does."""
    cli = ShellCLI()
    cli.debug = debug

    try:
        agent = cli.create_agent()

        prompt = f"""Explain this shell command in detail:

Command: {command}

Please explain:
1. What this command does
2. Each part/flag of the command
3. Any potential risks or side effects
4. Expected output

Provide a clear, beginner-friendly explanation."""

        task = TaskInput(
            id="describe_command", prompt=prompt, data={"command": command}
        )

        result = await agent.execute_task(task)

        if result.result and result.result.success:
            return str(result.result.data)
        else:
            return f"Error describing command: {result.result.error if result.result else 'Unknown error'}"

    except Exception as e:
        error_msg = f"Error describing command: {e}"
        if debug:
            import traceback

            error_msg += f"\nDebug trace:\n{traceback.format_exc()}"
        return error_msg


async def execute_shell_command(command: str, debug: bool = False) -> None:
    """Execute a shell command safely."""
    try:
        print_output(f"Executing: {command}", "yellow")

        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )

        if result.stdout:
            print_output("Output:", "green")
            print_output(result.stdout)

        if result.stderr:
            print_output("Errors:", "red")
            print_output(result.stderr)

        if result.returncode != 0:
            print_output(f"Command failed with exit code {result.returncode}", "red")

    except subprocess.TimeoutExpired:
        print_output("Command timed out after 30 seconds", "red")
    except Exception as e:
        print_output(f"Error executing command: {e}", "red")
        if debug:
            import traceback

            print_output(f"Debug trace:\n{traceback.format_exc()}", "red")


async def handle_shell_execution(command: str, debug: bool = False) -> None:
    """Handle shell command execution with user confirmation."""
    print_output(f"Command to execute: {command}", "yellow")
    print_output("Do you want to execute this command? (y/N): ", "yellow")

    try:
        choice = input().lower()
        if choice in ("y", "yes"):
            await execute_shell_command(command, debug)
        else:
            print_output("Command execution cancelled.", "yellow")
    except (EOFError, KeyboardInterrupt):
        print_output("\nOperation cancelled by user.", "yellow")


async def handle_chat_mode(
    chat_id: str, role: str = "default", debug: bool = False
) -> None:
    """Handle interactive chat mode."""
    cli = ShellCLI()
    cli.debug = debug

    print_output(f"Starting chat session: {chat_id}", "bright_green")
    print_output(f"Role: {role}", "bright_green")
    print_output("Type 'quit' or 'exit' to end the session", "dim")
    print_output("Press Ctrl+C to exit", "dim")

    # Load or create agent
    agent = cli.create_agent()

    # Load chat history
    chat_file = get_chat_file_path(chat_id)
    messages = load_chat_messages(chat_file)

    # Set role prompt
    role_prompt = get_role_prompt(role)
    if not messages or messages[0].role != "system":
        messages.insert(0, LLMMessage(role="system", content=role_prompt))

    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if user_input.lower() in ["quit", "exit", "bye"]:
                    break

                if not user_input.strip():
                    continue

                # Handle multiline input
                if user_input.strip() == '"""':
                    multiline_parts: List[str] = []
                    print_output('Enter multiline input (end with \'"""\'):', "dim")
                    while True:
                        line = input("... ")
                        if line.strip() == '"""':
                            break
                        multiline_parts.append(line)
                    user_input = "\n".join(multiline_parts)

                # Add user message
                messages.append(LLMMessage(role="user", content=user_input))

                # Create task
                task = TaskInput(
                    id=f"chat_{chat_id}_{len(messages)}",
                    prompt=user_input,
                    data={"chat_id": chat_id, "role": role},
                )

                # Get response
                print_output("Thinking...", "dim")
                result = await agent.execute_task(task)

                if result.result and result.result.success:
                    response_content = str(result.result.data)
                    print_output("\n[bold green]Assistant[/bold green]:")
                    display_response(response_content)

                    # Add assistant message
                    messages.append(
                        LLMMessage(role="assistant", content=response_content)
                    )
                else:
                    error_msg = (
                        result.result.error if result.result else "Unknown error"
                    )
                    print_output(f"Error: {error_msg}", "red")

                # Save messages
                save_chat_messages(chat_file, messages)

            except EOFError:
                print_output("\nEOF received, exiting chat mode.", "yellow")
                break
            except KeyboardInterrupt:
                print_output("\nChat session interrupted.", "yellow")
                break

    except Exception as e:
        print_output(f"Error in chat mode: {e}", "red")
        if debug:
            import traceback

            print_output(f"Debug trace:\n{traceback.format_exc()}", "red")

    # Save final messages
    save_chat_messages(chat_file, messages)
    print_output("Chat session ended.", "bright_green")


async def run_main_logic(
    prompt: str,
    shell: bool = False,
    explain: bool = False,
    chat: bool = False,
    chat_id: str = "default",
    role: str = "default",
    debug: bool = False,
) -> None:
    """Run the main CLI logic."""
    try:
        if chat:
            await handle_chat_mode(chat_id, role, debug)
            return

        if explain:
            response = await describe_shell_command(prompt, debug)
            display_response(response)
            return

        if shell:
            await handle_shell_execution(prompt, debug)
            return

        # Default: process as general prompt
        cli = ShellCLI()
        cli.debug = debug
        agent = cli.create_agent()

        task = TaskInput(id="general_query", prompt=prompt, data={"role": role})

        result = await agent.execute_task(task)

        if result.result and result.result.success:
            response_content = str(result.result.data)
            display_response(response_content)
        else:
            error_msg = result.result.error if result.result else "Unknown error"
            print_output(f"Error: {error_msg}", "red")

    except KeyboardInterrupt:
        print_output("\nOperation cancelled by user.", "yellow")
        raise typer.Exit(130)
    except Exception as e:
        print_output(f"Error: {str(e)}", "red")
        if debug:
            import traceback

            print_output(f"Debug trace:\n{traceback.format_exc()}", "red")
        raise typer.Exit(1)


# CLI Commands
@app.command()
def main(
    prompt: str = typer.Argument(None, help="The prompt or command to process"),
    shell: bool = typer.Option(False, "--shell", "-s", help="Execute as shell command"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Explain the command"),
    chat: bool = typer.Option(
        False, "--chat", "-c", help="Start interactive chat mode"
    ),
    chat_id: str = typer.Option("default", "--chat-id", help="Chat session ID"),
    role: str = typer.Option("default", "--role", "-r", help="AI role to use"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    list_roles: bool = typer.Option(False, "--list-roles", help="List available roles"),
    show_role: str = typer.Option(None, "--show-role", help="Show details of a role"),
    install: bool = typer.Option(False, "--install", help="Install shell integration"),
) -> None:
    """Enhanced LlamaAgent Shell CLI with advanced features."""

    # Handle special commands
    if list_roles:
        list_available_roles()
        return

    if show_role:
        show_role_details(show_role)
        return

    if install:
        install_shell_integration()
        return

    # Validate arguments
    if sum([shell, explain, chat]) > 1:
        print_output("Error: Only one mode can be specified at a time", "red")
        raise typer.Exit(1)

    if not prompt and not chat:
        print_output("Error: Prompt is required unless in chat mode", "red")
        raise typer.Exit(1)

    # Determine role if not specified
    if role == "default":
        role = determine_role(shell, explain, chat)

    # Run the main logic
    try:
        asyncio.run(
            run_main_logic(
                prompt or "",
                shell=shell,
                explain=explain,
                chat=chat,
                chat_id=chat_id,
                role=role,
                debug=debug,
            )
        )
    except Exception as e:
        print_output(f"Fatal error: {e}", "red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
