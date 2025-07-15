"""
Unit tests for LLM providers.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
from unittest.mock import Mock, patch
from src.llamaagent.llm.factory import LLMFactory
from src.llamaagent.llm.providers.mock_provider import MockProvider


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
        with pytest.raises(ValueError, match="API key not properly configured"):
            factory.get_provider("openai")
    
    def test_mock_provider_generates_response(self):
        """Test mock provider generates responses."""
        provider = MockProvider(model_name="test-model")
        response = provider.generate("Test prompt")
        assert response is not None
        assert len(response) > 0
    
    def test_provider_error_handling(self):
        """Test provider error handling."""
        provider = MockProvider(model_name="test-model")
        # Test with invalid input
        with pytest.raises(Exception):
            provider.generate(None)
