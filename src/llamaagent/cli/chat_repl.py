#!/usr/bin/env python3
"""
Chat and REPL System with Persistent Sessions

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides:
- Interactive chat sessions with memory
- REPL (Read-Eval-Print Loop) mode
- Session persistence and management
- Context-aware conversations
- Multi-modal chat support
"""

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..llm.factory import LLMFactory
from ..types import LLMMessage

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Individual chat message."""

    role: str
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """Chat session with persistent state."""

    session_id: str
    name: str
    created_at: float
    updated_at: float
    messages: List[ChatMessage]
    context: Dict[str, Any]
    settings: Dict[str, Any]


class SessionManager:
    """Manages chat sessions and persistence."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.config/llamaagent/sessions")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self, name: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        if name is None:
            name = f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = ChatSession(
            session_id=session_id,
            name=name,
            created_at=time.time(),
            updated_at=time.time(),
            messages=[],
            context=context or {},
            settings={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2000,
                "max_history": 20,
            },
        )

        self.save_session(session)
        return session

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load an existing session."""
        session_file = self.storage_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, "r") as f:
                data = json.load(f)

            # Convert message dictionaries back to ChatMessage objects
            messages = [ChatMessage(**msg) for msg in data.get("messages", [])]

            return ChatSession(
                session_id=data["session_id"],
                name=data["name"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                messages=messages,
                context=data.get("context", {}),
                settings=data.get("settings", {}),
            )
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def save_session(self, session: ChatSession) -> None:
        """Save session to storage."""
        session.updated_at = time.time()
        session_file = self.storage_dir / f"{session.session_id}.json"

        try:
            # Convert to dictionary for JSON serialization
            session_data = asdict(session)

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions: List[Dict[str, Any]] = []

        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)

                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "name": data["name"],
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except Exception as e:
                logger.error(f"Error reading session file {session_file}: {e}")

        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = self.storage_dir / f"{session_id}.json"

        try:
            if session_file.exists():
                session_file.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False


class ChatEngine:
    """Core chat engine with LLM integration."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.factory = LLMFactory()
        self.provider = self.factory.create_provider("openai", model_name=model)
        self.session_manager = SessionManager()

    async def chat(self, session: ChatSession, user_input: str, **kwargs: Any) -> str:
        """Process a chat message and return response."""

        # Add user message to session
        user_message = ChatMessage(
            role="user", content=user_input, timestamp=time.time()
        )
        session.messages.append(user_message)

        # Build conversation history
        messages = self._build_conversation_context(session)

        # Generate response
        try:
            response = await self.provider.complete(
                messages,
                temperature=kwargs.get(
                    "temperature", session.settings.get("temperature", 0.7)
                ),
                max_tokens=kwargs.get(
                    "max_tokens", session.settings.get("max_tokens", 2000)
                ),
            )

            # Add assistant response to session
            assistant_message = ChatMessage(
                role="assistant", content=response.content, timestamp=time.time()
            )
            session.messages.append(assistant_message)

            # Save session
            self.session_manager.save_session(session)

            return response.content

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _build_conversation_context(self, session: ChatSession) -> List[LLMMessage]:
        """Build conversation context from session history."""
        messages: List[LLMMessage] = []

        # Add system message if available in context
        if "system_prompt" in session.context:
            messages.append(
                LLMMessage(role="system", content=session.context["system_prompt"])
            )
        else:
            # Default system message
            messages.append(
                LLMMessage(
                    role="system",
                    content="You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
                )
            )

        # Add conversation history (limit to recent messages to manage token usage)
        max_history = session.settings.get("max_history", 20)
        recent_messages = (
            session.messages[-max_history:]
            if len(session.messages) > max_history
            else session.messages
        )

        for msg in recent_messages:
            messages.append(LLMMessage(role=msg.role, content=msg.content))

        return messages


class REPLInterface:
    """Read-Eval-Print Loop interface for interactive chat."""

    def __init__(self, chat_engine: ChatEngine) -> None:
        self.chat_engine = chat_engine
        self.current_session: Optional[ChatSession] = None
        self.running = False

    async def start(
        self, session_id: Optional[str] = None, system_prompt: Optional[str] = None
    ) -> None:
        """Start the REPL interface."""

        # Load or create session
        if session_id:
            if session_id == "temp":
                # Create temporary session
                self.current_session = self.chat_engine.session_manager.create_session(
                    "Temporary Chat"
                )
            else:
                # Load existing session
                self.current_session = self.chat_engine.session_manager.load_session(
                    session_id
                )
                if not self.current_session:
                    print(f"Session '{session_id}' not found. Creating new session.")
                    self.current_session = (
                        self.chat_engine.session_manager.create_session(session_id)
                    )
        else:
            # Create new session
            self.current_session = self.chat_engine.session_manager.create_session()

        # Set system prompt if provided
        if system_prompt:
            self.current_session.context["system_prompt"] = system_prompt
            self.chat_engine.session_manager.save_session(self.current_session)

        self.running = True

        # Display welcome message
        print(f"LlamaAgent LlamaAgent Chat Session: {self.current_session.name}")
        print(f"Session ID: {self.current_session.session_id}")
        print("Type 'exit', 'quit', or press Ctrl+C to end session")
        print("Type 'help' for available commands")
        print("-" * 50)

        # Main REPL loop
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input("\nUser You: ").strip()

                    if not user_input:
                        continue

                    # Handle special commands
                    if await self._handle_command(user_input):
                        continue

                    # Process chat message
                    print("\nLlamaAgent Assistant: ", end="", flush=True)
                    response = await self.chat_engine.chat(
                        self.current_session, user_input
                    )
                    print(response)

                except KeyboardInterrupt:
                    print("\n\nSession interrupted. Goodbye!")
                    break
                except EOFError:
                    print("\n\nEOF received. Goodbye!")
                    break
                except Exception as e:
                    print(f"\nFAIL Error: {e}")
                    logger.error(f"REPL error: {e}")

        finally:
            self.running = False
            if self.current_session:
                self.chat_engine.session_manager.save_session(self.current_session)
                print(f"\n Session saved: {self.current_session.session_id}")

    async def _handle_command(self, input_text: str) -> bool:
        """Handle special REPL commands."""

        if input_text.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            self.running = False
            return True

        elif input_text.lower() == "help":
            self._show_help()
            return True

        elif input_text.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            return True

        elif input_text.lower() == "history":
            self._show_history()
            return True

        elif input_text.lower() == "info":
            self._show_session_info()
            return True

        elif input_text.startswith("/model "):
            model_name = input_text[7:].strip()
            await self._change_model(model_name)
            return True

        elif input_text.startswith("/temp "):
            if not self.current_session:
                print("No active session")
                return True
            try:
                temp = float(input_text[6:].strip())
                self.current_session.settings["temperature"] = temp
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Invalid temperature value")
            return True

        elif input_text.startswith("/system "):
            if not self.current_session:
                print("No active session")
                return True
            system_prompt = input_text[8:].strip()
            self.current_session.context["system_prompt"] = system_prompt
            print("System prompt updated")
            return True

        return False

    def _show_help(self) -> None:
        """Show available commands."""
        print(
            """
Available commands:
- exit, quit, bye: End the session
- clear: Clear the screen
- history: Show conversation history
- info: Show session information
- help: Show this help message
- /model <name>: Change the AI model
- /temp <value>: Set temperature (0.0-2.0)
- /system <prompt>: Set system prompt
"""
        )

    def _show_history(self) -> None:
        """Show conversation history."""
        if not self.current_session or not self.current_session.messages:
            print("No conversation history yet.")
            return

        print("\n Conversation History:")
        print("-" * 50)

        for msg in self.current_session.messages[-10:]:  # Show last 10 messages
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
            role_icon = "User" if msg.role == "user" else "LlamaAgent"
            print(
                f"[{timestamp}] {role_icon} {msg.role.title()}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}"
            )

    def _show_session_info(self) -> None:
        """Show current session information."""
        if not self.current_session:
            return

        created = datetime.fromtimestamp(self.current_session.created_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        updated = datetime.fromtimestamp(self.current_session.updated_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(
            f"""
Session Information:
- Session ID: {self.current_session.session_id}
- Name: {self.current_session.name}
- Created: {created}
- Updated: {updated}
- Total messages: {len(self.current_session.messages)}
- Current role: {self.current_session.messages[-1].role if self.current_session.messages else 'None'}
- Model: {self.current_session.settings.get('model', 'Unknown')}
- Temperature: {self.current_session.settings.get('temperature', 0.7)}
"""
        )

    async def _change_model(self, model_name: str) -> None:
        """Change the AI model."""
        if not self.current_session:
            print("No active session")
            return

        try:
            # Test if model is available
            test_provider = self.chat_engine.factory.create_provider(
                "openai", model_name=model_name
            )

            self.chat_engine.provider = test_provider
            self.current_session.settings["model"] = model_name
            print(f"Model changed to {model_name}")

        except Exception as e:
            print(f"Error changing model: {e}")


# Utility functions
def list_sessions() -> None:
    """List all available chat sessions."""
    manager = SessionManager()
    sessions = manager.list_sessions()

    if not sessions:
        print("No chat sessions found.")
        return

    print("Sessions Available Chat Sessions:")
    print("-" * 70)
    print(f"{'ID':<36} {'Name':<20} {'Messages':<8} {'Updated'}")
    print("-" * 70)

    for session in sessions:
        session_id = session["session_id"][:8] + "..."
        name = (
            session["name"][:18] + "..."
            if len(session["name"]) > 18
            else session["name"]
        )
        msg_count = session["message_count"]
        updated = datetime.fromtimestamp(session["updated_at"]).strftime("%m/%d %H:%M")

        print(f"{session_id:<36} {name:<20} {msg_count:<8} {updated}")


def show_session_messages(session_id: str) -> None:
    """Show all messages from a specific session."""
    manager = SessionManager()
    session = manager.load_session(session_id)

    if not session:
        print(f"Session '{session_id}' not found.")
        return

    print(f" Messages from session: {session.name}")
    print("-" * 50)

    for msg in session.messages:
        timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        role_icon = "User" if msg.role == "user" else "LlamaAgent"
        print(f"\n[{timestamp}] {role_icon} {msg.role.title()}:")
        print(msg.content)


# Example usage and main function
async def main() -> None:
    """Example usage of the chat system."""

    # Initialize chat engine
    chat_engine = ChatEngine()
    repl = REPLInterface(chat_engine)

    # Start REPL with temporary session
    await repl.start("temp")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
