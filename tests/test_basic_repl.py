#!/usr/bin/env python3
"""
Basic tests for the chat REPL system without complex dependencies.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest


def test_chat_message_basic():
    """Test ChatMessage dataclass creation."""
    from src.llamaagent.cli.chat_repl import ChatMessage
    
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


def test_chat_session_basic():
    """Test ChatSession dataclass creation."""
    from src.llamaagent.cli.chat_repl import ChatSession, ChatMessage
    
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


def test_session_manager_basic():
    """Test SessionManager basic functionality."""
    from src.llamaagent.cli.chat_repl import SessionManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=tmpdir)
        
        # Test session creation
        session = manager.create_session(name="Test Session")
        assert session.name == "Test Session"
        assert isinstance(session.session_id, str)
        
        # Test session saving and loading
        manager.save_session(session)
        loaded_session = manager.load_session(session.session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == session.session_id
        assert loaded_session.name == session.name


@pytest.mark.asyncio
async def test_chat_engine_basic():
    """Test ChatEngine basic functionality."""
    from src.llamaagent.cli.chat_repl import ChatEngine
    from src.llamaagent.llm.providers.mock_provider import MockProvider
    from src.llamaagent.types import LLMResponse
    from unittest.mock import AsyncMock
    
    # Create chat engine with mock provider
    chat_engine = ChatEngine()
    chat_engine.provider = MockProvider(model_name="test-model")
    
    # Create a test session
    session = chat_engine.session_manager.create_session(name="Test")
    
    # Mock the provider's complete method
    mock_response = LLMResponse(
        content="Hello! How can I help you?",
        model="test-model",
        provider="mock",
        tokens_used=20
    )
    chat_engine.provider.complete = AsyncMock(return_value=mock_response)
    
    # Test chat
    response = await chat_engine.chat(session, "Hello")
    
    assert response == "Hello! How can I help you?"
    assert len(session.messages) == 2  # User + Assistant
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Hello! How can I help you?"


@pytest.mark.asyncio 
async def test_repl_interface_commands():
    """Test REPL interface command handling."""
    from src.llamaagent.cli.chat_repl import REPLInterface
    from unittest.mock import Mock
    
    # Create mock chat engine
    mock_engine = Mock()
    mock_engine.session_manager = Mock()
    
    repl = REPLInterface(mock_engine)
    
    # Test exit command
    assert await repl._handle_command("exit") is True
    assert repl.running is False
    
    # Reset and test help command
    repl.running = True
    result = await repl._handle_command("help")
    assert result is True
    
    # Test that regular messages are not handled as commands
    result = await repl._handle_command("This is a regular message")
    assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 