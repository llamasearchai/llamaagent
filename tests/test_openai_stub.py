"""
Tests for the OpenAI API stub.

This module tests the mock OpenAI implementation to ensure it works correctly
for development and testing purposes.
"""

import pytest

from src.llamaagent.integration._openai_stub import (
    MockAsyncOpenAIClient,
    MockAuthenticationError,
    MockChatCompletion,
    MockEmbedding,
    MockModeration,
    MockOpenAIClient,
    install_openai_stub,
    uninstall_openai_stub,
)


class TestOpenAIStub:
    """Test the OpenAI stub functionality."""

    def setup_method(self):
        """Set up test method."""
        # Ensure clean state
        uninstall_openai_stub()

    def teardown_method(self):
        """Tear down test method."""
        # Clean up after tests
        uninstall_openai_stub()

    def test_install_uninstall_stub(self):
        """Test installing and uninstalling the stub."""
        # Install stub
        mock_openai = install_openai_stub()
        assert mock_openai is not None

        # Check that openai is in sys.modules
        import sys

        assert "openai" in sys.modules

        # Import should give us the mock
        import openai

        assert openai.OpenAI == MockOpenAIClient

        # Uninstall
        uninstall_openai_stub()
        assert "openai" not in sys.modules

    def test_mock_client_initialization(self):
        """Test MockOpenAIClient initialization."""
        install_openai_stub()
        import openai

        # Test with normal API key
        client = openai.OpenAI(api_key="sk-12345")
        assert client.api_key == "sk-12345"
        assert hasattr(client, "chat")
        assert hasattr(client, "embeddings")
        assert hasattr(client, "moderations")

        # Test with test API key (should set flag for auth failure)
        client2 = openai.OpenAI(api_key="test-key")
        assert client2._should_fail_auth is True

    def test_chat_completions(self):
        """Test mock chat completions."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Test basic completion
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )

        assert isinstance(response, MockChatCompletion)
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert (
            response.choices[0].message.content
            == "This is a mock response from OpenAI stub."
        )

        # Test with specific keywords
        response2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Can you calculate something?"}],
        )
        assert "42" in response2.choices[0].message.content

        response3 = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Write some code please"}],
        )
        assert "```python" in response3.choices[0].message.content
        assert "Hello, World!" in response3.choices[0].message.content

    def test_embeddings(self):
        """Test mock embeddings."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Test single input
        response = client.embeddings.create(
            model="text-embedding-ada-002", input="Test text"
        )

        assert isinstance(response, MockEmbedding)
        assert response.model == "text-embedding-ada-002"
        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 1536
        assert all(isinstance(x, float) for x in response.data[0].embedding)

        # Test multiple inputs
        response2 = client.embeddings.create(
            model="text-embedding-ada-002", input=["Text 1", "Text 2", "Text 3"]
        )

        assert len(response2.data) == 3
        for i, item in enumerate(response2.data):
            assert item.index == i
            assert len(item.embedding) == 1536

    def test_moderations(self):
        """Test mock moderations."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Test clean content
        response = client.moderations.create(
            input="This is a friendly message about puppies"
        )

        assert isinstance(response, MockModeration)
        assert len(response.results) == 1
        assert response.results[0].flagged is False

        # Test problematic content
        response2 = client.moderations.create(
            input="This message contains violence and hate"
        )

        assert response2.results[0].flagged is True
        assert response2.results[0].categories.violence is True
        assert response2.results[0].categories.hate is True
        assert response2.results[0].category_scores.violence == 0.8
        assert response2.results[0].category_scores.hate == 0.8

    def test_response_serialization(self):
        """Test that responses can be serialized to dict."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Test chat completion serialization
        chat_response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )
        chat_dict = chat_response.to_dict()
        assert isinstance(chat_dict, dict)
        assert chat_dict["model"] == "gpt-4"
        assert "choices" in chat_dict
        assert "usage" in chat_dict

        # Test embedding serialization
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002", input="Test"
        )
        embedding_dict = embedding_response.to_dict()
        assert isinstance(embedding_dict, dict)
        assert embedding_dict["model"] == "text-embedding-ada-002"
        assert "data" in embedding_dict
        assert "usage" in embedding_dict

        # Test moderation serialization
        moderation_response = client.moderations.create(input="Test")
        moderation_dict = moderation_response.to_dict()
        assert isinstance(moderation_dict, dict)
        assert moderation_dict["model"] == "text-moderation-latest"
        assert "results" in moderation_dict

    @pytest.mark.asyncio
    async def test_async_client(self):
        """Test async OpenAI client."""
        install_openai_stub()
        import openai

        client = openai.AsyncOpenAI(api_key="sk-12345")
        assert isinstance(client, MockAsyncOpenAIClient)

        # Test async chat completion
        response = await client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )
        assert isinstance(response, MockChatCompletion)

        # Test async embeddings
        embedding_response = await client.embeddings.create(
            model="text-embedding-ada-002", input="Test"
        )
        assert isinstance(embedding_response, MockEmbedding)

        # Test async moderations
        moderation_response = await client.moderations.create(input="Test")
        assert isinstance(moderation_response, MockModeration)

    def test_exception_classes(self):
        """Test mock exception classes."""
        install_openai_stub()
        import openai

        # Test creating exceptions
        auth_error = openai.AuthenticationError("Invalid API key")
        assert isinstance(auth_error, MockAuthenticationError)
        assert auth_error.message == "Invalid API key"

        rate_error = openai.RateLimitError("Rate limit exceeded")
        assert isinstance(rate_error, openai.RateLimitError)

        conn_error = openai.APIConnectionError("Connection failed")
        assert isinstance(conn_error, openai.APIConnectionError)

        api_error = openai.APIError("Generic API error")
        assert isinstance(api_error, openai.APIError)

    def test_deterministic_embeddings(self):
        """Test that embeddings are deterministic based on input."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Same input should produce same embeddings
        response1 = client.embeddings.create(
            model="text-embedding-ada-002", input="Test text"
        )
        response2 = client.embeddings.create(
            model="text-embedding-ada-002", input="Test text"
        )

        assert response1.data[0].embedding == response2.data[0].embedding

        # Different input should produce different embeddings
        response3 = client.embeddings.create(
            model="text-embedding-ada-002", input="Different text"
        )

        assert response1.data[0].embedding != response3.data[0].embedding

    def test_token_usage_calculation(self):
        """Test that token usage is calculated reasonably."""
        install_openai_stub()
        import openai

        client = openai.OpenAI(api_key="sk-12345")

        # Test chat completion token usage
        long_message = "This is a very long message " * 20
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": long_message}]
        )

        assert response.usage.prompt_tokens > 10
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # Test embedding token usage
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002", input=long_message
        )

        assert embedding_response.usage.prompt_tokens > 10
        assert (
            embedding_response.usage.total_tokens
            == embedding_response.usage.prompt_tokens
        )
