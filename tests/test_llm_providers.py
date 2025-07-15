import os
from unittest.mock import AsyncMock, patch

import pytest

from llm.factory import ProviderFactory
from llm.models import LLMMessage, LLMResponse
from llm.providers.mlx_provider import MlxProvider
from llm.providers.mock_provider import MockProvider
from llm.providers.ollama_provider import OllamaProvider
from llm.providers.openai_provider import OpenAIProvider


class TestLLMMessage:
    """Test LLM message validation and immutability."""

    def test_valid_message_creation(self):
        """Test creating valid messages."""
        msg = LLMMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_invalid_role_raises_error(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            LLMMessage(role="invalid", content="Hello")  # type: ignore

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            LLMMessage(role="user", content="")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            LLMMessage(role="user", content="   ")

    def test_message_immutability(self):
        """Test that messages are immutable."""
        msg = LLMMessage(role="user", content="Hello")
        with pytest.raises(AttributeError):
            msg.role = "assistant"  # type: ignore
        with pytest.raises(AttributeError):
            msg.content = "Goodbye"  # type: ignore


class TestLLMResponse:
    """Test LLM response structure and validation."""

    def test_response_creation(self):
        """Test creating responses with all fields."""
        response = LLMResponse(
            content="Hello!",
            model="test-model",
            provider="test",
            tokens_used=10,
            metadata={"key": "value", "latency_ms": 150.5},
        )

        assert response.content == "Hello!"
        assert response.tokens_used == 10
        assert response.model == "test-model"
        assert response.metadata["latency_ms"] == 150.5
        assert response.provider == "test"
        assert response.metadata["key"] == "value"

    def test_response_defaults(self):
        """Test response with default values."""
        response = LLMResponse(content="Hello!", model="test", provider="test")
        assert response.tokens_used == 0
        assert response.model == "test"
        assert response.provider == "test"
        assert response.metadata == {}

    def test_response_immutability(self):
        """Test that responses are immutable."""
        response = LLMResponse(content="Hello!", model="test", provider="test")
        with pytest.raises(AttributeError):
            response.content = "Goodbye"  # type: ignore


class TestMockProvider:
    """Comprehensive tests for MockProvider."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider for testing."""
        return MockProvider(simulate_latency=False)

    @pytest.fixture
    def messages(self):
        """Sample messages for testing."""
        return [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Calculate 2 + 2"),
        ]

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_provider, messages):
        """Test basic completion functionality."""
        response = await mock_provider.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.tokens_used > 0
        assert response.model == "mock-gpt-4"
        assert response.provider == "mock"
        assert response.metadata["simulated"] is True

    @pytest.mark.asyncio
    async def test_math_response_context(self, mock_provider):
        """Test contextual math responses."""
        messages = [LLMMessage(role="user", content="Calculate the compound interest")]
        response = await mock_provider.complete(messages)

        assert "calculation" in response.content.lower()
        assert "42" in response.content  # Mock math result

    @pytest.mark.asyncio
    async def test_programming_response_context(self, mock_provider):
        """Test contextual programming responses."""
        messages = [LLMMessage(role="user", content="Write a Python function")]
        response = await mock_provider.complete(messages)

        assert "python" in response.content.lower()
        assert "def " in response.content
        assert "```" in response.content

    @pytest.mark.asyncio
    async def test_planning_response_context(self, mock_provider):
        """Test contextual planning responses."""
        messages = [LLMMessage(role="user", content="Create a strategic plan")]
        response = await mock_provider.complete(messages)

        assert "Phase" in response.content
        assert (
            "strategy" in response.content.lower() or "plan" in response.content.lower()
        )

    @pytest.mark.asyncio
    async def test_predefined_responses(self):
        """Test predefined response functionality."""
        custom_responses = {"test input": "test output"}
        provider = MockProvider(responses=custom_responses, simulate_latency=False)

        messages = [LLMMessage(role="user", content="test input")]
        response = await provider.complete(messages)

        assert response.content == "test output"

    @pytest.mark.asyncio
    async def test_failure_simulation(self):
        """Test failure simulation."""
        provider = MockProvider(failure_rate=1.0, simulate_latency=False)
        messages = [LLMMessage(role="user", content="test")]

        with pytest.raises(Exception, match="Mock failure"):
            await provider.complete(messages)

    @pytest.mark.asyncio
    async def test_call_counting(self, mock_provider, messages):
        """Test call counting functionality."""
        assert mock_provider.call_count == 0

        await mock_provider.complete(messages)
        assert mock_provider.call_count == 1

        await mock_provider.complete(messages)
        assert mock_provider.call_count == 2

        mock_provider.reset_call_count()
        assert mock_provider.call_count == 0

    @pytest.mark.asyncio
    async def test_health_check(self, mock_provider):
        """Test health check."""
        assert await mock_provider.health_check() is True


class TestOllamaProvider:
    """Comprehensive tests for OllamaProvider."""

    @pytest.fixture
    def ollama_provider(self):
        """Create Ollama provider for testing."""
        return OllamaProvider(
            base_url="http://localhost:11434", model="llama3.2:3b", timeout=30.0
        )

    @pytest.fixture
    def messages(self):
        """Sample messages for testing."""
        return [LLMMessage(role="user", content="Hello, how are you?")]

    @pytest.mark.asyncio
    async def test_provider_initialization(self, ollama_provider):
        """Test provider initialization."""
        assert ollama_provider.base_url == "http://localhost:11434"
        assert ollama_provider.model == "llama3.2:3b"
        assert ollama_provider.timeout == 30.0

    @pytest.mark.asyncio
    async def test_url_normalization(self):
        """Test URL normalization."""
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_successful_completion(self, mock_post, ollama_provider, messages):
        """Test successful completion with mocked HTTP."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello! I'm doing well, thank you."},
            "eval_count": 15,
            "eval_duration": 1000000,
            "total_duration": 2000000,
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = await ollama_provider.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! I'm doing well, thank you."
        assert response.model == "llama3.2:3b"
        assert response.provider == "ollama"
        assert response.tokens_used > 0
        assert "eval_count" in response.metadata

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_retry_logic(self, mock_post, ollama_provider, messages):
        """Test retry logic on connection failures."""
        # First two calls fail, third succeeds
        mock_post.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            AsyncMock(
                **{
                    "json.return_value": {"message": {"content": "Success"}},
                    "raise_for_status.return_value": None,
                }
            ),
        ]

        response = await ollama_provider.complete(messages)
        assert response.content == "Success"
        assert mock_post.call_count == 3

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_max_retries_exceeded(self, mock_post, ollama_provider, messages):
        """Test behavior when max retries are exceeded."""
        mock_post.side_effect = Exception("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
            await ollama_provider.complete(messages)

        assert mock_post.call_count == ollama_provider.retry_attempts

    @pytest.mark.asyncio
    async def test_empty_messages_validation(self, ollama_provider):
        """Test validation of empty messages."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            await ollama_provider.complete([])

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_success(self, mock_get, ollama_provider):
        """Test successful health check."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:3b"}, {"name": "other:model"}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = await ollama_provider.health_check()
        assert result is True

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.get")
    async def test_health_check_model_not_found(self, mock_get, ollama_provider):
        """Test health check when model is not available."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"models": [{"name": "other:model"}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = await ollama_provider.health_check()
        assert result is False

    def test_token_estimation(self, ollama_provider):
        """Test token estimation logic."""
        content = "Hello world"
        messages = [LLMMessage(role="user", content="Hi")]

        tokens = ollama_provider._estimate_tokens(content, messages)
        expected = (len("Hi") + len("Hello world")) // 4
        assert tokens == expected


class TestProviderFactory:
    """Test the provider factory functionality."""

    def setup_method(self):
        """Clear factory cache before each test."""
        ProviderFactory.clear_cache()

    def test_create_mock_provider_default(self):
        """Test creating default mock provider."""
        provider = ProviderFactory.create_provider()
        assert isinstance(provider, MockProvider)

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = ProviderFactory.create_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_create_mlx_provider(self):
        """Test creating MLX provider."""
        provider = ProviderFactory.create_provider("mlx")
        assert isinstance(provider, MlxProvider)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_provider(self):
        """Test creating OpenAI provider with API key."""
        provider = ProviderFactory.create_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_create_openai_provider_no_key(self):
        """Test creating OpenAI provider without API key raises error."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            ProviderFactory.create_provider("openai")

    def test_provider_caching(self):
        """Test that providers are cached properly."""
        provider1 = ProviderFactory.create_provider("mock")
        provider2 = ProviderFactory.create_provider("mock")
        assert provider1 is provider2

    def test_force_new_provider(self):
        """Test forcing creation of new provider instance."""
        provider1 = ProviderFactory.create_provider("mock")
        provider2 = ProviderFactory.create_provider("mock", force_new=True)
        assert provider1 is not provider2

    @patch.dict(os.environ, {"LLAMAAGENT_LLM_PROVIDER": "ollama"})
    def test_environment_provider_detection(self):
        """Test auto-detection of provider from environment."""
        provider = ProviderFactory.create_provider()
        assert isinstance(provider, OllamaProvider)

    def test_provider_with_custom_config(self):
        """Test creating provider with custom configuration."""
        provider = ProviderFactory.create_provider(
            "ollama", model="custom:model", timeout=60.0
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "custom:model"
        assert provider.timeout == 60.0

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health checking all cached providers."""
        # Create some providers
        ProviderFactory.create_provider("mock")
        ProviderFactory.create_provider("ollama")

        results = await ProviderFactory.health_check_all()
        assert len(results) == 2
        assert all(isinstance(v, bool) for v in results.values())


class TestMLXProvider:
    """Test MLX provider with fallback logic."""

    @pytest.fixture
    def mlx_provider(self):
        """Create MLX provider for testing."""
        return MlxProvider(model="llama3.2:3b")

    @pytest.fixture
    def messages(self):
        """Sample messages for testing."""
        return [LLMMessage(role="user", content="Hello")]

    @pytest.mark.asyncio
    async def test_fallback_to_ollama(self, mlx_provider, messages):
        """Test fallback to Ollama when MLX is unavailable."""
        # MLX should fallback to Ollama or mock
        response = await mlx_provider.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.provider in ["mlx", "ollama", "mock"]

    @pytest.mark.asyncio
    async def test_health_check(self, mlx_provider):
        """Test MLX health check."""
        result = await mlx_provider.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_mlx_initialization(self, mlx_provider):
        """Test MLX provider initialization."""
        assert mlx_provider.model == "llama3.2:3b"
        assert hasattr(mlx_provider, "fallback_provider")

    @pytest.mark.asyncio
    async def test_empty_messages_validation(self, mlx_provider):
        """Test validation of empty messages."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            await mlx_provider.complete([])


class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(api_key="test-key", model="gpt-4", timeout=30.0)

    @pytest.fixture
    def messages(self):
        """Sample messages for testing."""
        return [LLMMessage(role="user", content="Hello, how are you?")]

    def test_provider_initialization(self, openai_provider):
        """Test provider initialization."""
        assert openai_provider.model == "gpt-4"
        assert hasattr(openai_provider, "client")

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_successful_completion(self, mock_openai, openai_provider, messages):
        """Test successful completion with mocked OpenAI client."""
        # Mock the OpenAI client and response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="Hello! I'm doing well, thank you."))
        ]
        mock_response.usage = AsyncMock(total_tokens=25)
        mock_response.model = "gpt-4"

        mock_client.chat.completions.create.return_value = mock_response

        response = await openai_provider.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! I'm doing well, thank you."
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.tokens_used == 25

    @pytest.mark.asyncio
    async def test_empty_messages_validation(self, openai_provider):
        """Test validation of empty messages."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            await openai_provider.complete([])

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_api_error_handling(self, mock_openai, openai_provider, messages):
        """Test handling of OpenAI API errors."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Simulate an API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await openai_provider.complete(messages)

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_health_check(self, mock_openai, openai_provider):
        """Test health check functionality."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Mock successful model list response
        mock_client.models.list.return_value = AsyncMock(
            data=[AsyncMock(id="gpt-4"), AsyncMock(id="gpt-3.5-turbo")]
        )

        result = await openai_provider.health_check()
        assert result is True


class TestProviderIntegration:
    """Integration tests for provider interactions."""

    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """Test switching between different providers."""
        # Test that we can create different providers
        mock_provider = ProviderFactory.create_provider("mock")
        ollama_provider = ProviderFactory.create_provider("ollama", force_new=True)

        assert isinstance(mock_provider, MockProvider)
        assert isinstance(ollama_provider, OllamaProvider)
        assert mock_provider is not ollama_provider

    @pytest.mark.asyncio
    async def test_provider_error_fallback(self):
        """Test that providers handle errors gracefully."""
        provider = MockProvider(failure_rate=0.5, simulate_latency=False)
        messages = [LLMMessage(role="user", content="test")]

        # Test multiple attempts - some should succeed, some fail
        results = []
        for _ in range(10):
            try:
                await provider.complete(messages)
                results.append("success")
            except Exception:
                results.append("failure")

        # Should have both successes and failures with 50% failure rate
        assert "success" in results
        assert "failure" in results

    @pytest.mark.asyncio
    async def test_concurrent_provider_calls(self):
        """Test concurrent calls to the same provider."""
        import asyncio

        provider = MockProvider(simulate_latency=False)
        messages = [LLMMessage(role="user", content="test")]

        # Make 5 concurrent calls
        tasks = [provider.complete(messages) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(isinstance(r, LLMResponse) for r in responses)
        assert provider.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
