"""
Unit tests for LLM providers.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio

import pytest

from src.llamaagent.llm.factory import LLMFactory
from src.llamaagent.llm.providers.mock_provider import MockProvider
from src.llamaagent.types import LLMMessage


class TestLLMProviders:
    """Test suite for LLM providers."""

    def test_mock_provider_initialization(self):
        """Test mock provider can be initialized."""
        provider = MockProvider(model_name="test-model")
        assert provider is not None
        assert provider.model_name == "test-model"

    def test_llm_factory_creates_mock_provider(self):
        """Test LLM factory can create mock provider."""
        factory = LLMFactory()
        provider = factory.get_provider("mock")
        assert provider is not None
        assert isinstance(provider, MockProvider)

    def test_llm_factory_fails_without_api_key(self):
        """Test LLM factory fails properly without API key."""
        factory = LLMFactory()
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            factory.get_provider("openai")

    def test_mock_provider_generates_response(self):
        """Test mock provider generates responses."""
        provider = MockProvider(model_name="test-model")

        # Use the async generate_response method
        async def test_async():
            response = await provider.generate_response("Test prompt")
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            return response

        response = asyncio.run(test_async())
        assert response.provider == "mock"

    def test_provider_error_handling(self):
        """Test provider error handling."""
        provider = MockProvider(model_name="test-model")

        # Test with valid input - mock provider should handle this gracefully
        async def test_async():
            messages = [LLMMessage(role="user", content="test input")]
            response = await provider.complete(messages)
            assert response is not None
            assert response.content is not None

        asyncio.run(test_async())
