#!/usr/bin/env python3
"""
Comprehensive tests for the chat REPL system.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llamaagent.cli.chat_repl import (
    ChatEngine,
    ChatMessage,
    ChatSession,
    REPLInterface,
    SessionManager,
)
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import LLMMessage, LLMResponse


class TestChatMessage:
    """Test suite for ChatMessage dataclass."""

    def test_chat_message_creation(self):
        """Test creating a chat message."""
        message = ChatMessage(
            role="user",
            content="Hello world",
            timestamp=time.time(),
            metadata={"test": True}
        )
        
        assert message.role == "user"
        assert message.content == "Hello world"
        assert message.metadata["test"] is True
        assert isinstance(message.timestamp, float)

    def test_chat_message_minimal(self):
        """Test creating a minimal chat message."""
        message = ChatMessage(
            role="assistant",
            content="Response"
        )
        
        assert message.role == "assistant"
        assert message.content == "Response"
        assert message.metadata is None
        assert isinstance(message.timestamp, float)


class TestChatSession:
    """Test suite for ChatSession dataclass."""

    def test_chat_session_creation(self):
        """Test creating a chat session."""
        messages = [
            ChatMessage(role="user", content="Hello", timestamp=time.time()),
            ChatMessage(role="assistant", content="Hi there", timestamp=time.time())
        ]
        
        session = ChatSession(
            session_id="test-session",
            name="Test Session",
            created_at=time.time(),
            updated_at=time.time(),
            messages=messages,
            context={"system_prompt": "You are helpful"},
            settings={"temperature": 0.7}
        )
        
        assert session.session_id == "test-session"
        assert session.name == "Test Session"
        assert len(session.messages) == 2
        assert session.context["system_prompt"] == "You are helpful"
        assert session.settings["temperature"] == 0.7


class TestSessionManager:
    """Test suite for SessionManager."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def session_manager(self, temp_storage):
        """Create session manager with temporary storage."""
        return SessionManager(storage_dir=temp_storage)

    def test_session_manager_initialization(self, temp_storage):
        """Test session manager initialization."""
        manager = SessionManager(storage_dir=temp_storage)
        assert manager.storage_dir == Path(temp_storage)
        assert manager.storage_dir.exists()

    def test_create_session(self, session_manager):
        """Test creating a new session."""
        session = session_manager.create_session(
            name="Test Session",
            context={"system_prompt": "Be helpful"}
        )
        
        assert session.name == "Test Session"
        assert session.context["system_prompt"] == "Be helpful"
        assert session.settings["model"] == "gpt-4o-mini"
        assert session.settings["temperature"] == 0.7
        assert len(session.messages) == 0

    def test_create_session_with_defaults(self, session_manager):
        """Test creating a session with default parameters."""
        session = session_manager.create_session()
        
        assert session.name.startswith("Chat_")
        assert isinstance(session.session_id, str)
        assert len(session.session_id) > 10  # Should be a UUID
        assert session.context == {}
        assert "model" in session.settings

    def test_save_and_load_session(self, session_manager):
        """Test saving and loading a session."""
        # Create and save session
        original_session = session_manager.create_session(name="Test Session")
        original_session.messages.append(
            ChatMessage(role="user", content="Test message", timestamp=time.time())
        )
        session_manager.save_session(original_session)
        
        # Load session
        loaded_session = session_manager.load_session(original_session.session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == original_session.session_id
        assert loaded_session.name == original_session.name
        assert len(loaded_session.messages) == 1
        assert loaded_session.messages[0].content == "Test message"

    def test_load_nonexistent_session(self, session_manager):
        """Test loading a session that doesn't exist."""
        session = session_manager.load_session("nonexistent-id")
        assert session is None

    def test_list_sessions(self, session_manager):
        """Test listing sessions."""
        # Create multiple sessions
        session1 = session_manager.create_session(name="Session 1")
        session2 = session_manager.create_session(name="Session 2")
        
        # Add messages to sessions
        session1.messages.append(
            ChatMessage(role="user", content="Message 1", timestamp=time.time())
        )
        session2.messages.extend([
            ChatMessage(role="user", content="Message 2", timestamp=time.time()),
            ChatMessage(role="assistant", content="Response 2", timestamp=time.time()),
        ])
        
        session_manager.save_session(session1)
        session_manager.save_session(session2)
        
        # List sessions
        sessions = session_manager.list_sessions()
        
        assert len(sessions) == 2
        
        # Check session information
        session_names = {s["name"] for s in sessions}
        assert "Session 1" in session_names
        assert "Session 2" in session_names
        
        # Check message counts
        for session_info in sessions:
            if session_info["name"] == "Session 1":
                assert session_info["message_count"] == 1
            elif session_info["name"] == "Session 2":
                assert session_info["message_count"] == 2

    def test_delete_session(self, session_manager):
        """Test deleting a session."""
        # Create session
        session = session_manager.create_session(name="To Delete")
        session_manager.save_session(session)
        
        # Verify it exists
        assert session_manager.load_session(session.session_id) is not None
        
        # Delete it
        result = session_manager.delete_session(session.session_id)
        assert result is True
        
        # Verify it's gone
        assert session_manager.load_session(session.session_id) is None

    def test_delete_nonexistent_session(self, session_manager):
        """Test deleting a session that doesn't exist."""
        result = session_manager.delete_session("nonexistent-id")
        assert result is False


class TestChatEngine:
    """Test suite for ChatEngine."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        provider = MockProvider(model_name="test-model")
        return provider

    @pytest.fixture
    def chat_engine(self, mock_provider):
        """Create chat engine with mock provider."""
        engine = ChatEngine()
        engine.provider = mock_provider
        return engine

    @pytest.fixture
    def sample_session(self, chat_engine):
        """Create a sample chat session."""
        return chat_engine.session_manager.create_session(
            name="Test Session",
            context={"system_prompt": "You are a helpful assistant"}
        )

    @pytest.mark.asyncio
    async def test_chat_basic(self, chat_engine, sample_session):
        """Test basic chat functionality."""
        # Mock the provider's complete method
        mock_response = LLMResponse(
            content="Hello! How can I help you?",
            model="test-model",
            provider="mock",
            tokens_used=20
        )
        chat_engine.provider.complete = AsyncMock(return_value=mock_response)
        
        response = await chat_engine.chat(sample_session, "Hello")
        
        assert response == "Hello! How can I help you?"
        assert len(sample_session.messages) == 2  # User + Assistant
        assert sample_session.messages[0].role == "user"
        assert sample_session.messages[0].content == "Hello"
        assert sample_session.messages[1].role == "assistant"
        assert sample_session.messages[1].content == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_chat_with_conversation_history(self, chat_engine, sample_session):
        """Test chat with existing conversation history."""
        # Add some history
        sample_session.messages.extend([
            ChatMessage(role="user", content="What's 2+2?", timestamp=time.time()),
            ChatMessage(role="assistant", content="2+2 equals 4.", timestamp=time.time()),
        ])
        
        # Mock the provider
        mock_response = LLMResponse(
            content="The result is 9.",
            model="test-model",
            provider="mock",
            tokens_used=15
        )
        chat_engine.provider.complete = AsyncMock(return_value=mock_response)
        
        response = await chat_engine.chat(sample_session, "What's 3+6?")
        
        # Verify the conversation history was used
        call_args = chat_engine.provider.complete.call_args[0][0]
        assert len(call_args) >= 4  # System + 2 history + new user message
        
        # Check message structure
        user_messages = [msg for msg in call_args if msg.role == "user"]
        assert len(user_messages) == 2
        assert "2+2" in user_messages[0].content
        assert "3+6" in user_messages[1].content

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, chat_engine, sample_session):
        """Test chat error handling."""
        # Mock provider to raise an error
        chat_engine.provider.complete = AsyncMock(side_effect=Exception("API Error"))
        
        response = await chat_engine.chat(sample_session, "Test message")
        
        assert "Sorry, I encountered an error" in response
        assert "API Error" in response

    def test_build_conversation_context_with_system_prompt(self, chat_engine, sample_session):
        """Test building conversation context with system prompt."""
        sample_session.context["system_prompt"] = "You are a math tutor"
        sample_session.messages.extend([
            ChatMessage(role="user", content="Help me", timestamp=time.time()),
            ChatMessage(role="assistant", content="I'll help", timestamp=time.time()),
        ])
        
        messages = chat_engine._build_conversation_context(sample_session)
        
        assert len(messages) >= 3  # System + user + assistant
        assert messages[0].role == "system"
        assert messages[0].content == "You are a math tutor"

    def test_build_conversation_context_default_system(self, chat_engine, sample_session):
        """Test building conversation context with default system prompt."""
        sample_session.messages.append(
            ChatMessage(role="user", content="Hello", timestamp=time.time())
        )
        
        messages = chat_engine._build_conversation_context(sample_session)
        
        assert len(messages) >= 2  # System + user
        assert messages[0].role == "system"
        assert "helpful AI assistant" in messages[0].content

    def test_build_conversation_context_max_history(self, chat_engine, sample_session):
        """Test that conversation context respects max history limit."""
        # Add many messages
        for i in range(30):
            sample_session.messages.extend([
                ChatMessage(role="user", content=f"Message {i}", timestamp=time.time()),
                ChatMessage(role="assistant", content=f"Response {i}", timestamp=time.time()),
            ])
        
        messages = chat_engine._build_conversation_context(sample_session)
        
        # Should have system message + limited history (max_history = 20 by default)
        assert len(messages) <= 21  # 1 system + 20 history


class TestREPLInterface:
    """Test suite for REPLInterface."""

    @pytest.fixture
    def mock_chat_engine(self):
        """Create mock chat engine."""
        engine = Mock()
        engine.session_manager = Mock()
        return engine

    @pytest.fixture
    def repl_interface(self, mock_chat_engine):
        """Create REPL interface with mock engine."""
        return REPLInterface(mock_chat_engine)

    @pytest.mark.asyncio
    async def test_handle_exit_command(self, repl_interface):
        """Test handling exit commands."""
        assert await repl_interface._handle_command("exit") is True
        assert repl_interface.running is False
        
        # Reset and test other exit commands
        repl_interface.running = True
        assert await repl_interface._handle_command("quit") is True
        assert repl_interface.running is False
        
        repl_interface.running = True
        assert await repl_interface._handle_command("bye") is True
        assert repl_interface.running is False

    @pytest.mark.asyncio
    async def test_handle_help_command(self, repl_interface, capsys):
        """Test handling help command."""
        result = await repl_interface._handle_command("help")
        
        assert result is True
        captured = capsys.readouterr()
        assert "Available commands" in captured.out
        assert "exit" in captured.out
        assert "/model" in captured.out

    @pytest.mark.asyncio
    async def test_handle_clear_command(self, repl_interface):
        """Test handling clear command."""
        with patch('os.system') as mock_system:
            result = await repl_interface._handle_command("clear")
            
            assert result is True
            mock_system.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_temp_command(self, repl_interface):
        """Test handling temperature command."""
        # Mock current session
        mock_session = Mock()
        mock_session.settings = {}
        repl_interface.current_session = mock_session
        
        result = await repl_interface._handle_command("/temp 0.9")
        
        assert result is True
        assert mock_session.settings["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_handle_temp_command_invalid(self, repl_interface, capsys):
        """Test handling invalid temperature command."""
        mock_session = Mock()
        mock_session.settings = {}
        repl_interface.current_session = mock_session
        
        result = await repl_interface._handle_command("/temp invalid")
        
        assert result is True
        captured = capsys.readouterr()
        assert "Invalid temperature value" in captured.out

    @pytest.mark.asyncio
    async def test_handle_temp_command_no_session(self, repl_interface, capsys):
        """Test handling temperature command without active session."""
        repl_interface.current_session = None
        
        result = await repl_interface._handle_command("/temp 0.5")
        
        assert result is True
        captured = capsys.readouterr()
        assert "No active session" in captured.out

    @pytest.mark.asyncio
    async def test_handle_system_command(self, repl_interface):
        """Test handling system prompt command."""
        mock_session = Mock()
        mock_session.context = {}
        repl_interface.current_session = mock_session
        
        result = await repl_interface._handle_command("/system You are a coding assistant")
        
        assert result is True
        assert mock_session.context["system_prompt"] == "You are a coding assistant"

    @pytest.mark.asyncio
    async def test_handle_system_command_no_session(self, repl_interface, capsys):
        """Test handling system command without active session."""
        repl_interface.current_session = None
        
        result = await repl_interface._handle_command("/system Test prompt")
        
        assert result is True
        captured = capsys.readouterr()
        assert "No active session" in captured.out

    @pytest.mark.asyncio
    async def test_handle_model_command(self, repl_interface):
        """Test handling model change command."""
        # Mock the chat engine and session
        mock_session = Mock()
        mock_session.settings = {}
        repl_interface.current_session = mock_session
        
        mock_provider = Mock()
        repl_interface.chat_engine.factory = Mock()
        repl_interface.chat_engine.factory.create_provider.return_value = mock_provider
        
        result = await repl_interface._handle_command("/model gpt-4")
        
        assert result is True
        assert mock_session.settings["model"] == "gpt-4"
        repl_interface.chat_engine.factory.create_provider.assert_called_once_with(
            "openai", model_name="gpt-4"
        )

    @pytest.mark.asyncio
    async def test_handle_model_command_error(self, repl_interface, capsys):
        """Test handling model change command with error."""
        mock_session = Mock()
        repl_interface.current_session = mock_session
        
        repl_interface.chat_engine.factory = Mock()
        repl_interface.chat_engine.factory.create_provider.side_effect = Exception("Model not found")
        
        result = await repl_interface._handle_command("/model invalid-model")
        
        assert result is True
        captured = capsys.readouterr()
        assert "Error changing model" in captured.out

    @pytest.mark.asyncio
    async def test_handle_regular_message(self, repl_interface):
        """Test that regular messages are not handled as commands."""
        result = await repl_interface._handle_command("This is a regular message")
        assert result is False

    def test_show_session_info(self, repl_interface, capsys):
        """Test showing session information."""
        # Mock current session
        mock_session = Mock()
        mock_session.session_id = "test-123"
        mock_session.name = "Test Session"
        mock_session.created_at = time.time()
        mock_session.updated_at = time.time()
        mock_session.messages = [Mock(), Mock(), Mock()]  # 3 messages
        mock_session.settings = {"model": "gpt-4", "temperature": 0.7}
        repl_interface.current_session = mock_session
        
        repl_interface._show_session_info()
        
        captured = capsys.readouterr()
        assert "Session Information" in captured.out
        assert "test-123" in captured.out
        assert "Test Session" in captured.out
        assert "Total messages: 3" in captured.out
        assert "Model: gpt-4" in captured.out
        assert "Temperature: 0.7" in captured.out

    def test_show_session_info_no_session(self, repl_interface):
        """Test showing session info when no session exists."""
        repl_interface.current_session = None
        
        # Should not raise an error
        repl_interface._show_session_info()

    def test_show_history(self, repl_interface, capsys):
        """Test showing conversation history."""
        # Mock messages
        messages = [
            ChatMessage(role="user", content="Hello there", timestamp=time.time()),
            ChatMessage(role="assistant", content="Hi! How can I help?", timestamp=time.time()),
            ChatMessage(role="user", content="What's the weather?", timestamp=time.time()),
        ]
        
        mock_session = Mock()
        mock_session.messages = messages
        repl_interface.current_session = mock_session
        
        repl_interface._show_history()
        
        captured = capsys.readouterr()
        assert "Conversation History" in captured.out
        assert "Hello there" in captured.out
        assert "Hi! How can I help?" in captured.out
        assert "What's the weather?" in captured.out

    def test_show_history_empty(self, repl_interface, capsys):
        """Test showing history when no messages exist."""
        mock_session = Mock()
        mock_session.messages = []
        repl_interface.current_session = mock_session
        
        repl_interface._show_history()
        
        captured = capsys.readouterr()
        assert "No conversation history yet" in captured.out

    def test_show_history_no_session(self, repl_interface, capsys):
        """Test showing history when no session exists."""
        repl_interface.current_session = None
        
        repl_interface._show_history()
        
        captured = capsys.readouterr()
        assert "No conversation history yet" in captured.out


class TestUtilityFunctions:
    """Test suite for utility functions."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage with test sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test sessions
            manager = SessionManager(storage_dir=tmpdir)
            
            session1 = manager.create_session(name="Session 1")
            session1.messages.append(
                ChatMessage(role="user", content="Test 1", timestamp=time.time())
            )
            manager.save_session(session1)
            
            session2 = manager.create_session(name="Session 2")
            session2.messages.extend([
                ChatMessage(role="user", content="Test 2", timestamp=time.time()),
                ChatMessage(role="assistant", content="Response 2", timestamp=time.time()),
            ])
            manager.save_session(session2)
            
            yield tmpdir

    def test_list_sessions_function(self, temp_storage, capsys):
        """Test the list_sessions utility function."""
        with patch('src.llamaagent.cli.chat_repl.SessionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock session data
            mock_manager.list_sessions.return_value = [
                {
                    "session_id": "session-1-long-id-here",
                    "name": "Test Session 1",
                    "message_count": 5,
                    "updated_at": time.time()
                },
                {
                    "session_id": "session-2-long-id-here",
                    "name": "Very Long Session Name That Should Be Truncated",
                    "message_count": 10,
                    "updated_at": time.time()
                }
            ]
            
            from src.llamaagent.cli.chat_repl import list_sessions
            list_sessions()
            
            captured = capsys.readouterr()
            assert "Available Chat Sessions" in captured.out
            assert "Test Session 1" in captured.out
            assert "Very Long Session Nam..." in captured.out  # Should be truncated
            assert "session-1..." in captured.out  # ID should be truncated

    def test_list_sessions_empty(self, capsys):
        """Test list_sessions when no sessions exist."""
        with patch('src.llamaagent.cli.chat_repl.SessionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.list_sessions.return_value = []
            
            from src.llamaagent.cli.chat_repl import list_sessions
            list_sessions()
            
            captured = capsys.readouterr()
            assert "No chat sessions found" in captured.out

    def test_show_session_messages_function(self, capsys):
        """Test the show_session_messages utility function."""
        with patch('src.llamaagent.cli.chat_repl.SessionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock session
            mock_session = Mock()
            mock_session.name = "Test Session"
            mock_session.messages = [
                ChatMessage(role="user", content="Hello", timestamp=time.time()),
                ChatMessage(role="assistant", content="Hi there!", timestamp=time.time()),
            ]
            mock_manager.load_session.return_value = mock_session
            
            from src.llamaagent.cli.chat_repl import show_session_messages
            show_session_messages("test-session-id")
            
            captured = capsys.readouterr()
            assert "Messages from session: Test Session" in captured.out
            assert "Hello" in captured.out
            assert "Hi there!" in captured.out

    def test_show_session_messages_not_found(self, capsys):
        """Test show_session_messages when session doesn't exist."""
        with patch('src.llamaagent.cli.chat_repl.SessionManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            mock_manager.load_session.return_value = None
            
            from src.llamaagent.cli.chat_repl import show_session_messages
            show_session_messages("nonexistent-id")
            
            captured = capsys.readouterr()
            assert "Session 'nonexistent-id' not found" in captured.out


class TestIntegrationScenarios:
    """Integration tests for complete REPL scenarios."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_complete_chat_session_workflow(self, temp_storage):
        """Test a complete chat session workflow."""
        # Create chat engine with mock provider
        chat_engine = ChatEngine()
        chat_engine.provider = MockProvider(model_name="test-model")
        chat_engine.session_manager = SessionManager(storage_dir=temp_storage)
        
        # Mock provider responses
        responses = [
            LLMResponse(content="Hello! I'm here to help.", model="test-model", provider="mock"),
            LLMResponse(content="2 + 2 equals 4.", model="test-model", provider="mock"),
            LLMResponse(content="The weather is sunny today.", model="test-model", provider="mock"),
        ]
        chat_engine.provider.complete = AsyncMock(side_effect=responses)
        
        # Create REPL interface
        repl = REPLInterface(chat_engine)
        
        # Simulate starting a session
        session = chat_engine.session_manager.create_session(name="Test Chat")
        repl.current_session = session
        
        # Simulate chat interactions
        response1 = await chat_engine.chat(session, "Hello")
        assert response1 == "Hello! I'm here to help."
        assert len(session.messages) == 2
        
        response2 = await chat_engine.chat(session, "What's 2+2?")
        assert response2 == "2 + 2 equals 4."
        assert len(session.messages) == 4
        
        response3 = await chat_engine.chat(session, "How's the weather?")
        assert response3 == "The weather is sunny today."
        assert len(session.messages) == 6
        
        # Test command handling
        assert await repl._handle_command("/temp 0.9") is True
        assert session.settings["temperature"] == 0.9
        
        assert await repl._handle_command("/system You are a weather expert") is True
        assert session.context["system_prompt"] == "You are a weather expert"
        
        # Verify session persistence
        chat_engine.session_manager.save_session(session)
        loaded_session = chat_engine.session_manager.load_session(session.session_id)
        
        assert loaded_session is not None
        assert len(loaded_session.messages) == 6
        assert loaded_session.settings["temperature"] == 0.9
        assert loaded_session.context["system_prompt"] == "You are a weather expert"

    @pytest.mark.asyncio
    async def test_session_recovery_and_continuation(self, temp_storage):
        """Test session recovery and continuation."""
        # Create initial session
        manager1 = SessionManager(storage_dir=temp_storage)
        session = manager1.create_session(name="Recovery Test")
        
        # Add some conversation history
        session.messages.extend([
            ChatMessage(role="user", content="Start conversation", timestamp=time.time()),
            ChatMessage(role="assistant", content="Conversation started", timestamp=time.time()),
        ])
        session.context["system_prompt"] = "You are helpful"
        session.settings["temperature"] = 0.8
        manager1.save_session(session)
        
        # Simulate restart - create new manager instance
        manager2 = SessionManager(storage_dir=temp_storage)
        
        # Load the session
        recovered_session = manager2.load_session(session.session_id)
        
        assert recovered_session is not None
        assert recovered_session.name == "Recovery Test"
        assert len(recovered_session.messages) == 2
        assert recovered_session.context["system_prompt"] == "You are helpful"
        assert recovered_session.settings["temperature"] == 0.8
        
        # Continue the conversation
        chat_engine = ChatEngine()
        chat_engine.provider = MockProvider(model_name="test-model")
        chat_engine.session_manager = manager2
        
        mock_response = LLMResponse(
            content="Continuing our conversation",
            model="test-model",
            provider="mock"
        )
        chat_engine.provider.complete = AsyncMock(return_value=mock_response)
        
        response = await chat_engine.chat(recovered_session, "Continue please")
        
        assert response == "Continuing our conversation"
        assert len(recovered_session.messages) == 4

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, temp_storage):
        """Test error handling and recovery scenarios."""
        chat_engine = ChatEngine()
        chat_engine.provider = MockProvider(model_name="test-model")
        chat_engine.session_manager = SessionManager(storage_dir=temp_storage)
        
        session = chat_engine.session_manager.create_session(name="Error Test")
        
        # Test API error handling
        chat_engine.provider.complete = AsyncMock(side_effect=Exception("API Error"))
        
        response = await chat_engine.chat(session, "This should fail")
        
        assert "Sorry, I encountered an error" in response
        assert "API Error" in response
        
        # Verify session state is still valid
        assert len(session.messages) == 1  # Only user message added
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "This should fail"
        
        # Test recovery after error
        mock_response = LLMResponse(
            content="I'm back online",
            model="test-model",
            provider="mock"
        )
        chat_engine.provider.complete = AsyncMock(return_value=mock_response)
        
        response = await chat_engine.chat(session, "Are you working now?")
        
        assert response == "I'm back online"
        assert len(session.messages) == 3  # Previous user + new user + assistant

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, temp_storage):
        """Test concurrent access to sessions."""
        manager = SessionManager(storage_dir=temp_storage)
        
        # Create session
        session = manager.create_session(name="Concurrent Test")
        manager.save_session(session)
        
        # Simulate concurrent access
        async def modify_session(session_id: str, message_content: str):
            loaded_session = manager.load_session(session_id)
            if loaded_session:
                loaded_session.messages.append(
                    ChatMessage(role="user", content=message_content, timestamp=time.time())
                )
                manager.save_session(loaded_session)
        
        # Run concurrent modifications
        await asyncio.gather(
            modify_session(session.session_id, "Message 1"),
            modify_session(session.session_id, "Message 2"),
            modify_session(session.session_id, "Message 3"),
        )
        
        # Load final session and verify
        final_session = manager.load_session(session.session_id)
        
        # Note: Due to file-based storage, last write wins
        # In a production system, you'd want proper concurrency control
        assert final_session is not None
        assert len(final_session.messages) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 