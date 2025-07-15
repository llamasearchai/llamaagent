"""
Integration tests for the OpenAI stub with the LlamaAgent system.

This module tests that the OpenAI stub integrates properly with
the existing codebase and testing infrastructure.

Note: The OpenAIProvider uses httpx for direct API calls, so it doesn't
interact with the openai library stub. These tests focus on direct usage
of the stub and potential future integrations.
"""

import pytest

from src.llamaagent.integration._openai_stub import (
    install_openai_stub,
    uninstall_openai_stub,
)


class TestOpenAIStubIntegration:
    """Test OpenAI stub integration with LlamaAgent."""

    def setup_method(self):
        """Set up test method."""
        uninstall_openai_stub()

    def teardown_method(self):
        """Tear down test method."""
        uninstall_openai_stub()

    def test_stub_prevents_real_api_calls(self):
        """Test that the stub actually prevents real API calls."""
        install_openai_stub()

        import openai

        # Create a client with a fake API key
        client = openai.OpenAI(api_key="sk-fake-key-that-would-fail")

        # This should work with the stub (would fail with real API)
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Test"}]
        )

        assert response is not None
        assert (
            response.choices[0].message.content
            == "This is a mock response from OpenAI stub."
        )

    def test_stub_with_direct_openai_usage(self):
        """Test the stub with direct OpenAI library usage."""
        install_openai_stub()

        import openai

        # Create multiple clients
        client1 = openai.OpenAI(api_key="sk-test-1")
        client2 = openai.OpenAI(api_key="sk-test-2")

        # Both should work with the stub
        response1 = client1.chat.completions.create(
            messages=[{"role": "user", "content": "calculate"}], model="gpt-3.5-turbo"
        )
        response2 = client2.chat.completions.create(
            messages=[{"role": "user", "content": "code"}], model="gpt-4"
        )

        assert "42" in response1.choices[0].message.content
        assert "Hello, World!" in response2.choices[0].message.content

    @pytest.mark.asyncio
    async def test_async_operations_with_stub(self):
        """Test async operations with the stub."""
        install_openai_stub()

        import openai

        client = openai.AsyncOpenAI(api_key="sk-test-async")

        # Test async chat completion
        response = await client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Test async"}]
        )

        assert response is not None
        assert isinstance(response.choices[0].message.content, str)

    def test_stub_state_isolation(self):
        """Test that installing/uninstalling stub doesn't affect state."""
        # First, use without stub (would need real API key in production)
        # We'll mock this part since we don't want to make real calls in tests

        # Install stub
        install_openai_stub()
        import openai

        client1 = openai.OpenAI(api_key="sk-test")
        response1 = client1.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Test 1"}]
        )

        # Uninstall stub
        uninstall_openai_stub()

        # Reinstall stub
        install_openai_stub()
        import openai as openai2

        client2 = openai2.OpenAI(api_key="sk-test")
        response2 = client2.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Test 2"}]
        )

        # Both responses should be valid
        assert response1.choices[0].message.content is not None
        assert response2.choices[0].message.content is not None

    def test_embedding_operations_with_provider(self):
        """Test embedding operations through provider with stub."""
        install_openai_stub()

        # Assuming the provider has an embed method
        # We'll test the raw OpenAI client since provider might not expose embeddings
        import openai

        client = openai.OpenAI(api_key="sk-test-embeddings")

        # Single embedding
        response = client.embeddings.create(
            model="text-embedding-ada-002", input="Test text for embedding"
        )

        assert len(response.data) == 1
        assert len(response.data[0].embedding) == 1536

        # Batch embeddings
        batch_response = client.embeddings.create(
            model="text-embedding-ada-002", input=["Text 1", "Text 2", "Text 3"]
        )

        assert len(batch_response.data) == 3
        for item in batch_response.data:
            assert len(item.embedding) == 1536

    def test_error_scenarios_with_stub(self):
        """Test error scenarios are handled properly with stub."""
        install_openai_stub()

        import openai

        # Test creating various exception types
        auth_error = openai.AuthenticationError("Test auth error")
        assert isinstance(auth_error, Exception)
        assert auth_error.message == "Test auth error"

        rate_error = openai.RateLimitError("Test rate limit")
        assert isinstance(rate_error, Exception)

        conn_error = openai.APIConnectionError("Test connection error")
        assert isinstance(conn_error, Exception)

        api_error = openai.APIError("Test API error")
        assert isinstance(api_error, Exception)
