#!/usr/bin/env python3
"""
Shell Command Generation and Execution System
Author: Nik Jois <nikjois@llamasearch.ai>

This module provides shell command generation, with:
- OS-aware command generation
- Interactive execution prompts ([E]xecute, [D]escribe, [A]bort)
- Safety checks and validation
- Shell integration support
- Command history and caching
"""

import asyncio
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ShellCommandGenerator:
    """Generates OS-aware shell commands using LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.os_info = self._get_os_info()
        self.command_cache: Dict[str, str] = {}

        # Try to initialize LLM provider if available
        try:
            from ..llm.factory import LLMFactory
            from ..types import LLMMessage

            self.factory = LLMFactory()
            self.provider = self.factory.create_provider("openai", model_name=model)
            self.llm_available = True
            self.LLMMessage = LLMMessage
        except ImportError:
            self.llm_available = False
            logger.warning("LLM provider not available, using basic command generation")

    def _get_os_info(self) -> Dict[str, str]:
        """Get operating system information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "shell": os.environ.get(
                "SHELL", "/bin/bash" if os.name == "posix" else "cmd"
            ),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "home": str(Path.home()),
            "cwd": str(Path.cwd()),
        }

    async def generate_command(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a shell command for the given prompt."""
        # Check cache first
        cache_key = f"{prompt}:{self.os_info['system']}"
        if cache_key in self.command_cache:
            logger.info("Using cached command")
            return self.command_cache[cache_key]

        if self.llm_available:
            command = await self._generate_with_llm(prompt, context)
        else:
            command = self._generate_basic_command(prompt, context)

        # Cache the result
        self.command_cache[cache_key] = command

        logger.info(f"Generated command: {command}")
        return command

    async def _generate_with_llm(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate command using LLM."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(prompt, context)

        messages = [
            self.LLMMessage(role="system", content=system_prompt),
            self.LLMMessage(role="user", content=user_prompt),
        ]

        response = await self.provider.complete(
            messages, max_tokens=500, temperature=0.1
        )

        command = self._clean_command_response(response.content)
        return command

    def _generate_basic_command(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate basic commands without LLM."""
        prompt_lower = prompt.lower()

        # Basic command patterns for common tasks
        patterns = {
            "list files": "ls -la" if self.os_info["system"] != "Windows" else "dir",
            "current directory": "pwd" if self.os_info["system"] != "Windows" else "cd",
            "change directory": "cd",
            "copy file": "cp" if self.os_info["system"] != "Windows" else "copy",
            "move file": "mv" if self.os_info["system"] != "Windows" else "move",
            "delete file": "rm" if self.os_info["system"] != "Windows" else "del",
            "create directory": "mkdir",
            "remove directory": "rmdir",
            "find file": "find" if self.os_info["system"] != "Windows" else "dir /s",
            "disk usage": "df -h" if self.os_info["system"] != "Windows" else "dir",
            "process list": (
                "ps aux" if self.os_info["system"] != "Windows" else "tasklist"
            ),
            "network info": (
                "ifconfig" if self.os_info["system"] != "Windows" else "ipconfig"
            ),
            "system info": (
                "uname -a" if self.os_info["system"] != "Windows" else "systeminfo"
            ),
        }

        # Find matching pattern
        for pattern, command in patterns.items():
            if pattern in prompt_lower:
                return command

        # Default fallback
        return f"# Command for: {prompt}"

    def _build_system_prompt(self) -> str:
        """Build system prompt with OS-specific information."""
        return f"""You are an expert system administrator and shell command specialist.
Your job is to generate safe, efficient shell commands for the user's specific environment.

SYSTEM INFORMATION:
- Operating System: {self.os_info['system']} {self.os_info['release']}
- Shell: {self.os_info['shell']}
- Architecture: {self.os_info['machine']}
- Current Directory: {self.os_info['cwd']}

GUIDELINES:
1. Generate ONLY the shell command - no explanations, markdown, or extra text
2. Consider the user's specific OS and shell environment
3. Prefer standard POSIX commands when possible for portability
4. Use safe command practices and avoid destructive operations without explicit confirmation
5. If multiple commands are needed, separate them with appropriate operators (&&, ||)
6. For Windows, use appropriate CMD/PowerShell commands
7. If the request is unclear or potentially dangerous, suggest a safer alternative

SAFETY RULES:
- Never suggest 'rm -rf /' or equivalent destructive commands
- Always use appropriate flags for safety (e.g., -i for interactive mode)
- Prefer specific paths over wildcards when possible
- Use quotes around paths with spaces

Respond with ONLY the command that accomplishes the task."""

    def _build_user_prompt(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt with context."""
        user_prompt = f"Generate a shell command for: {prompt}"

        if context:
            if "current_directory" in context:
                user_prompt += f"\n\nCurrent directory: {context['current_directory']}"

            if "available_files" in context:
                files = ", ".join(context["available_files"][:10])  # Limit to 10 files
                user_prompt += f"\n\nFiles in current directory: {files}"

            if "environment" in context:
                user_prompt += f"\n\nEnvironment info: {context['environment']}"

        return user_prompt

    def _clean_command_response(self, response: str) -> str:
        """Clean and validate the command response."""
        command = response.strip()

        # Remove markdown code blocks
        if command.startswith("```"):
            lines = command.split("\n")
            if len(lines) > 1:
                # Find first non-empty line after ```
                start_idx = 1
                while start_idx < len(lines) and not lines[start_idx].strip():
                    start_idx += 1

                # Find closing ```
                end_idx = len(lines) - 1
                while end_idx > start_idx and not lines[end_idx].strip().startswith(
                    "```"
                ):
                    end_idx -= 1

                if end_idx > start_idx:
                    command = "\n".join(lines[start_idx:end_idx])
                else:
                    command = "\n".join(lines[start_idx:])

        # Remove extra whitespace and newlines
        command = " ".join(command.split())

        return command


class ShellCommandExecutor:
    """Executes shell commands with safety checks and interactive prompts."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.command_history: List[
            Tuple[str, bool, float]
        ] = []  # (command, success, timestamp)
        self.safe_mode = True

    def is_safe_command(self, command: str) -> Tuple[bool, str]:
        """Check if command is safe to execute."""
        if not self.safe_mode:
            return True, "Safe mode disabled"

        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "chmod -R 777 /",
            "dd if=/dev/random",
            "mv / /dev/null",
            "mkfs.",
            "format c:",
            "deltree",
            "rmdir /s /q c:",
            "del /f /s /q c:",
            "> /dev/sda",
            ":(){ :|:& };:",  # Fork bomb
            "cat /dev/urandom",
        ]

        command_lower = command.lower()

        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False, f"Dangerous pattern detected: {pattern}"

        # Check for potentially risky operations
        risky_patterns = [
            ("rm -rf", "Recursive delete operation"),
            ("chmod 777", "Setting overly permissive permissions"),
            ("sudo rm", "Root-level delete operation"),
            ("dd if=", "Direct disk access"),
            ("curl | sh", "Executing downloaded script"),
            ("wget | sh", "Executing downloaded script"),
            ("curl | bash", "Executing downloaded script"),
            ("wget | bash", "Executing downloaded script"),
            ("shutdown", "System shutdown command"),
            ("reboot", "System reboot command"),
            ("halt", "System halt command"),
            ("fdisk", "Disk partitioning"),
            ("parted", "Disk partitioning"),
        ]

        for pattern, warning in risky_patterns:
            if pattern in command_lower:
                return False, f"Risky operation: {warning}"

        return True, "Command appears safe"

    async def execute_interactive(self, command: str) -> bool:
        """Execute command with interactive prompts."""
        print(f"\nGenerated command: {command}")

        # Safety check
        is_safe, safety_msg = self.is_safe_command(command)
        if not is_safe:
            print(f"WARNING:  WARNING: {safety_msg}")
            print("This command will not be executed for safety reasons.")
            return False

        while True:
            try:
                choice = input("\n[E]xecute, [D]escribe, [A]bort: ").lower().strip()

                if choice in ("e", "execute", "y", "yes"):
                    return await self._execute_command(command)
                elif choice in ("d", "describe"):
                    self._describe_command(command)
                elif choice in ("a", "abort", "n", "no"):
                    print("Command execution aborted.")
                    return False
                else:
                    print("Please choose [E]xecute, [D]escribe, or [A]bort")

            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return False

    async def _execute_command(self, command: str) -> bool:
        """Execute the command and return success status."""
        start_time = time.time()

        try:
            if self.debug:
                print(f"Executing: {command}")

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True,
            )

            stdout, stderr = await process.communicate()

            execution_time = time.time() - start_time
            success = process.returncode == 0

            # Display output
            if stdout:
                print("Output:")
                print(stdout)

            if stderr:
                print("Errors:")
                print(stderr)

            if success:
                print(f"PASS Command completed successfully in {execution_time:.2f}s")
            else:
                print(f"FAIL Command failed with exit code {process.returncode}")

            # Record in history
            self.command_history.append((command, success, start_time))

            return success

        except Exception as e:
            print(f"FAIL Execution error: {e}")
            self.command_history.append((command, False, start_time))
            return False

    def _describe_command(self, command: str) -> None:
        """Describe what the command does."""
        descriptions = {
            "ls": "List directory contents",
            "pwd": "Print working directory",
            "cd": "Change directory",
            "cp": "Copy files or directories",
            "mv": "Move/rename files or directories",
            "rm": "Remove files or directories",
            "mkdir": "Create directories",
            "rmdir": "Remove empty directories",
            "cat": "Display file contents",
            "grep": "Search text patterns",
            "find": "Search for files and directories",
            "chmod": "Change file permissions",
            "chown": "Change file ownership",
            "ps": "Show running processes",
            "kill": "Terminate processes",
            "df": "Show disk space usage",
            "du": "Show directory space usage",
            "tar": "Archive/extract files",
            "gzip": "Compress files",
            "wget": "Download files from web",
            "curl": "Transfer data to/from servers",
            "ssh": "Secure shell remote access",
            "scp": "Secure copy over network",
            "rsync": "Synchronize files/directories",
        }

        # Extract first command
        first_command = command.split()[0] if command.split() else ""

        if first_command in descriptions:
            print(f"Command description: {descriptions[first_command]}")
        else:
            print("This command will execute the following:")

        print(f"Full command: {command}")

        # Basic analysis
        if "&&" in command:
            print("Note: Multiple commands chained with AND logic")
        if "||" in command:
            print("Note: Multiple commands with OR logic (fallback)")
        if "|" in command and "||" not in command:
            print("Note: Command uses pipes to chain output")
        if ">" in command:
            print("Note: Output will be redirected to a file")
        if ">>" in command:
            print("Note: Output will be appended to a file")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get command execution history."""
        history = []
        for command, success, timestamp in self.command_history:
            history.append(
                {
                    "command": command,
                    "success": success,
                    "timestamp": timestamp,
                    "formatted_time": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(timestamp)
                    ),
                }
            )
        return history

    def clear_history(self) -> None:
        """Clear command history."""
        self.command_history.clear()

    def toggle_safe_mode(self) -> bool:
        """Toggle safe mode on/off."""
        self.safe_mode = not self.safe_mode
        return self.safe_mode


class ShellCommandInterface:
    """Combined interface for command generation and execution."""

    def __init__(self, model: str = "gpt-4o-mini", debug: bool = False):
        self.generator = ShellCommandGenerator(model)
        self.executor = ShellCommandExecutor(debug)

    async def process_request(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        auto_execute: bool = False,
    ) -> bool:
        """Process a command request from prompt to execution."""
        try:
            # Generate command
            command = await self.generator.generate_command(prompt, context)

            if auto_execute:
                # Execute directly (be careful with this!)
                return await self.executor._execute_command(command)
            else:
                # Interactive execution
                return await self.executor.execute_interactive(command)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print(f"FAIL Error: {e}")
            return False

    def get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return self.generator.os_info

    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.executor.get_history()


def main() -> None:
    """Example usage of the shell command system."""

    async def demo():
        interface = ShellCommandInterface(debug=True)

        print("Shell Command Generation Demo")
        print("=" * 40)

        # Show system info
        sys_info = interface.get_system_info()
        print(f"System: {sys_info['system']} {sys_info['release']}")
        print(f"Shell: {sys_info['shell']}")
        print(f"Current Directory: {sys_info['cwd']}")

        # Example requests
        requests = [
            "list all files in the current directory",
            "show current directory",
            "create a new directory called 'test'",
            "find all Python files",
        ]

        for request in requests:
            print(f"\n--- Request: {request} ---")
            await interface.process_request(request)

        # Show history
        print("\n--- Execution History ---")
        history = interface.get_history()
        for entry in history:
            status = "PASS" if entry["success"] else "FAIL"
            print(f"{status} {entry['formatted_time']}: {entry['command']}")

    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    main()
