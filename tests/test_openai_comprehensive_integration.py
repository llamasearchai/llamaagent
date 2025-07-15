"""
Comprehensive test suite for OpenAI integration.

Tests all OpenAI model types, APIs, tools, and endpoints including:
- Reasoning models (o-series)
- Chat models
- Image generation
- Text-to-speech
- Transcription
- Embeddings
- Moderation
- FastAPI endpoints
- Budget tracking
- Error handling

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.llamaagent.api.openai_comprehensive_api import app

# Import modules to test
from src.llamaagent.integration.openai_comprehensive import (
    BudgetExceededError,
    OpenAIComprehensiveConfig,
    OpenAIComprehensiveIntegration,
    OpenAIModelType,
    OpenAIUsageTracker,
    create_comprehensive_openai_integration,
)
from src.llamaagent.tools.openai_tools import (
    OPENAI_TOOLS,
    OpenAIComprehensiveTool,
    OpenAIEmbeddingsTool,
    OpenAIImageGenerationTool,
    OpenAIModerationTool,
    OpenAIReasoningTool,
    OpenAITextToSpeechTool,
    OpenAITranscriptionTool,
    create_all_openai_tools,
    create_openai_tool,
)


class TestOpenAIUsageTracker:
    """Test usage tracking functionality."""

    def test_usage_tracker_initialization(self):
        """Test usage tracker initialization."""
        tracker = OpenAIUsageTracker(budget_limit=50.0)

        assert tracker.budget_limit == 50.0
        assert tracker.total_cost == 0.0
        assert tracker.request_count == 0
        assert len(tracker.usage_by_model) == 0
        assert len(tracker.usage_log) == 0

    def test_usage_tracking(self):
        """Test usage tracking functionality."""
        tracker = OpenAIUsageTracker(budget_limit=10.0)

        # Add usage
        tracker.add_usage(
            model="gpt-4o-mini", input_tokens=100, output_tokens=50, operation="test"
        )

        assert tracker.request_count == 1
        assert tracker.total_cost > 0
        assert "gpt-4o-mini" in tracker.usage_by_model
        assert len(tracker.usage_log) == 1

        # Get summary
        summary = tracker.get_usage_summary()
        assert summary["total_requests"] == 1
        assert summary["total_cost"] > 0
        assert summary["remaining_budget"] < 10.0

    def test_budget_exceeded_error(self):
        """Test budget exceeded error."""
        tracker = OpenAIUsageTracker(budget_limit=0.001)  # Very low limit

        with pytest.raises(BudgetExceededError):
            tracker.add_usage(model="gpt-4o", input_tokens=1000, output_tokens=1000)

    def test_cost_estimation(self):
        """Test cost estimation for different models."""
        tracker = OpenAIUsageTracker()

        # Test reasoning model
        cost_reasoning = tracker._estimate_cost("o3-mini", 1000, 500)
        assert cost_reasoning > 0

        # Test flagship model
        cost_flagship = tracker._estimate_cost("gpt-4o", 1000, 500)
        assert cost_flagship > cost_reasoning

        # Test unknown model (should use default)
        cost_unknown = tracker._estimate_cost("unknown-model", 1000, 500)
        assert cost_unknown > 0


class TestOpenAIComprehensiveIntegration:
    """Test comprehensive OpenAI integration."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = AsyncMock()

        # Mock chat completion
        completion_response = Mock()
        completion_response.id = "test-id"
        completion_response.object = "chat.completion"
        completion_response.created = 1234567890
        completion_response.model = "gpt-4o-mini"
        completion_response.choices = [Mock()]
        completion_response.choices[0].index = 0
        completion_response.choices[0].message = Mock()
        completion_response.choices[0].message.role = "assistant"
        completion_response.choices[0].message.content = "Test response"
        completion_response.choices[0].message.tool_calls = None
        completion_response.choices[0].finish_reason = "stop"
        completion_response.usage = Mock()
        completion_response.usage.prompt_tokens = 10
        completion_response.usage.completion_tokens = 20
        completion_response.usage.total_tokens = 30

        client.chat.completions.create.return_value = completion_response

        # Mock image generation
        image_response = Mock()
        image_response.created = 1234567890
        image_response.data = [Mock()]
        image_response.data[0].url = "https://example.com/image.png"
        image_response.data[0].b64_json = None
        image_response.data[0].revised_prompt = "Revised prompt"

        client.images.generate.return_value = image_response

        # Mock embeddings
        embedding_response = Mock()
        embedding_response.object = "list"
        embedding_response.data = [Mock()]
        embedding_response.data[0].object = "embedding"
        embedding_response.data[0].index = 0
        embedding_response.data[0].embedding = [0.1] * 1536
        embedding_response.model = "text-embedding-3-large"
        embedding_response.usage = Mock()
        embedding_response.usage.prompt_tokens = 5
        embedding_response.usage.total_tokens = 5

        client.embeddings.create.return_value = embedding_response

        # Mock moderation
        moderation_response = Mock()
        moderation_response.id = "mod-test"
        moderation_response.model = "text-moderation-latest"
        moderation_response.results = [Mock()]
        moderation_response.results[0].flagged = False
        moderation_response.results[0].categories = Mock()
        moderation_response.results[0].category_scores = Mock()

        client.moderations.create.return_value = moderation_response

        # Mock TTS
        client.audio.speech.create.return_value = Mock()
        client.audio.speech.create.return_value.content = b"fake audio data"

        # Mock transcription
        transcription_response = Mock()
        transcription_response.text = "Transcribed text"

        client.audio.transcriptions.create.return_value = transcription_response

        # Mock models list
        models_response = Mock()
        models_response.data = [
            Mock(id="gpt-4o-mini", object="model", created=123, owned_by="openai"),
            Mock(id="dall-e-3", object="model", created=123, owned_by="openai"),
        ]

        client.models.list.return_value = models_response

        return client

    @pytest.fixture
    def integration(self, mock_openai_client):
        """Create integration with mocked client."""
        config = OpenAIComprehensiveConfig(api_key="test-key", budget_limit=100.0)

        integration = OpenAIComprehensiveIntegration(config)
        integration.client = mock_openai_client

        return integration

    @pytest.mark.asyncio
    async def test_chat_completion(self, integration):
        """Test chat completion functionality."""
        messages = [{"role": "user", "content": "Hello"}]

        result = await integration.chat_completion(
            messages=messages, model="gpt-4o-mini"
        )

        assert result["choices"][0]["message"]["content"] == "Test response"
        assert result["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_image_generation(self, integration):
        """Test image generation functionality."""
        result = await integration.generate_image(
            prompt="A beautiful landscape", model="dall-e-3"
        )

        assert len(result["data"]) == 1
        assert result["data"][0]["url"] == "https://example.com/image.png"

    @pytest.mark.asyncio
    async def test_embeddings_creation(self, integration):
        """Test embeddings creation."""
        result = await integration.create_embeddings(
            input_texts=["Hello world"], model="text-embedding-3-large"
        )

        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) == 1536

    @pytest.mark.asyncio
    async def test_content_moderation(self, integration):
        """Test content moderation."""
        result = await integration.moderate_content(
            input_text=["This is safe content"], model="text-moderation-latest"
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["flagged"] is False

    @pytest.mark.asyncio
    async def test_text_to_speech(self, integration):
        """Test text-to-speech conversion."""
        result = await integration.text_to_speech(
            input_text="Hello world", model="tts-1"
        )

        assert result == b"fake audio data"

    @pytest.mark.asyncio
    async def test_audio_transcription(self, integration):
        """Test audio transcription."""
        # Create fake audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"fake audio data")
            tmp_file_path = tmp_file.name

        try:
            result = await integration.transcribe_audio(
                audio_file=tmp_file_path, model="whisper-1"
            )

            assert result["text"] == "Transcribed text"
        finally:
            os.unlink(tmp_file_path)

    @pytest.mark.asyncio
    async def test_health_check(self, integration):
        """Test health check functionality."""
        health = await integration.health_check()

        assert health["api_accessible"] is True
        assert "model_types_available" in health
        assert "budget_status" in health

    def test_model_configuration(self, integration):
        """Test model configuration functionality."""
        # Test known model
        config = integration.get_model_config("gpt-4o-mini")
        assert config.model_name == "gpt-4o-mini"
        assert config.model_type == OpenAIModelType.COST_OPTIMIZED

        # Test unknown model (should return default)
        config = integration.get_model_config("unknown-model")
        assert config.model_name == "unknown-model"

    def test_models_by_type(self, integration):
        """Test getting models by type."""
        reasoning_models = integration.get_models_by_type(OpenAIModelType.REASONING)
        assert "o3-mini" in reasoning_models

        image_models = integration.get_models_by_type(OpenAIModelType.IMAGE_GENERATION)
        assert "dall-e-3" in image_models


class TestOpenAITools:
    """Test OpenAI tools functionality."""

    @pytest.fixture
    def mock_integration(self):
        """Mock integration for tool testing."""
        integration = Mock()
        integration.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {"total_tokens": 30},
            }
        )
        integration.generate_image = AsyncMock(
            return_value={"data": [{"url": "https://example.com/image.png"}]}
        )
        integration.create_embeddings = AsyncMock(
            return_value={
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 5},
            }
        )
        integration.moderate_content = AsyncMock(
            return_value={"results": [{"flagged": False}]}
        )
        integration.text_to_speech = AsyncMock(return_value=b"audio data")
        integration.transcribe_audio = AsyncMock(
            return_value={"text": "Transcribed text"}
        )

        return integration

    @pytest.mark.asyncio
    async def test_reasoning_tool(self, mock_integration):
        """Test reasoning tool."""
        tool = OpenAIReasoningTool(mock_integration)

        result = await tool.aexecute("Solve 2+2")

        assert result["success"] is True
        assert result["response"] == "Test response"
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_image_generation_tool(self, mock_integration):
        """Test image generation tool."""
        tool = OpenAIImageGenerationTool(mock_integration)

        result = await tool.aexecute("A beautiful sunset")

        assert result["success"] is True
        assert len(result["images"]) == 1

    @pytest.mark.asyncio
    async def test_text_to_speech_tool(self, mock_integration):
        """Test text-to-speech tool."""
        tool = OpenAITextToSpeechTool(mock_integration)

        result = await tool.aexecute("Hello world")

        assert result["success"] is True
        assert "audio_path" in result

    @pytest.mark.asyncio
    async def test_transcription_tool(self, mock_integration):
        """Test transcription tool."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(b"fake audio")
            audio_path = tmp_file.name

        try:
            tool = OpenAITranscriptionTool(mock_integration)
            result = await tool.aexecute(audio_path)

            assert result["success"] is True
            assert result["transcription"]["text"] == "Transcribed text"
        finally:
            os.unlink(audio_path)

    @pytest.mark.asyncio
    async def test_embeddings_tool(self, mock_integration):
        """Test embeddings tool."""
        tool = OpenAIEmbeddingsTool(mock_integration)

        result = await tool.aexecute("Test text")

        assert result["success"] is True
        assert len(result["embeddings"]) == 1

    @pytest.mark.asyncio
    async def test_moderation_tool(self, mock_integration):
        """Test moderation tool."""
        tool = OpenAIModerationTool(mock_integration)

        result = await tool.aexecute("Safe content")

        assert result["success"] is True
        assert len(result["moderation_results"]) == 1

    @pytest.mark.asyncio
    async def test_comprehensive_tool(self, mock_integration):
        """Test comprehensive tool."""
        tool = OpenAIComprehensiveTool(mock_integration)

        # Test reasoning operation
        result = await tool.aexecute("reasoning", problem="Test problem")
        assert result["success"] is True

        # Test image generation operation
        result = await tool.aexecute("image_generation", prompt="Test prompt")
        assert result["success"] is True

        # Test invalid operation
        result = await tool.aexecute("invalid_operation")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    def test_tool_creation_utilities(self, mock_integration):
        """Test tool creation utilities."""
        # Test creating specific tool
        tool = create_openai_tool("reasoning", mock_integration)
        assert isinstance(tool, OpenAIReasoningTool)

        # Test creating all tools
        tools = create_all_openai_tools(mock_integration)
        assert len(tools) == len(OPENAI_TOOLS)

        # Test invalid tool type
        with pytest.raises(ValueError):
            create_openai_tool("invalid_tool_type", mock_integration)


class TestOpenAIComprehensiveAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_integration_patch(self):
        """Patch the integration for API tests."""
        mock_integration = Mock()

        # Mock health check
        mock_integration.health_check = AsyncMock(
            return_value={"api_accessible": True, "model_types_available": {}}
        )

        # Mock budget status
        mock_integration.get_budget_status.return_value = {
            "budget_limit": 100.0,
            "total_cost": 10.0,
            "remaining_budget": 90.0,
        }

        # Mock chat completion
        mock_integration.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "API test response"}}],
                "usage": {"total_tokens": 25},
            }
        )

        # Mock available models
        mock_integration.get_available_models = AsyncMock(
            return_value=[{"id": "gpt-4o-mini", "object": "model"}]
        )

        with patch(
            "src.llamaagent.api.openai_comprehensive_api.get_integration"
        ) as mock_get:
            mock_get.return_value = mock_integration
            yield mock_integration

    def test_root_endpoint(self, client, mock_integration_patch):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "LlamaAgent Comprehensive OpenAI API"
        assert "available_model_types" in data
        assert "budget_status" in data

    def test_health_endpoint(self, client, mock_integration_patch):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_accessible"] is True

    def test_budget_endpoint(self, client, mock_integration_patch):
        """Test budget status endpoint."""
        response = client.get("/budget")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "budget_limit" in data["data"]

    def test_models_endpoint(self, client, mock_integration_patch):
        """Test models listing endpoint."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "models" in data["data"]

    def test_chat_completions_endpoint(self, client, mock_integration_patch):
        """Test chat completions endpoint."""
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
        }

        response = client.post("/chat/completions", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "usage" in data

    @patch("src.llamaagent.api.openai_comprehensive_api.get_tool")
    def test_reasoning_endpoint(self, mock_get_tool, client, mock_integration_patch):
        """Test reasoning endpoint."""
        mock_tool = Mock()
        mock_tool.aexecute = AsyncMock(
            return_value={"success": True, "response": "Reasoning result"}
        )
        mock_get_tool.return_value = mock_tool

        payload = {"problem": "Solve this problem", "model": "o3-mini"}

        response = client.post("/reasoning/solve", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("src.llamaagent.api.openai_comprehensive_api.get_tool")
    def test_image_generation_endpoint(
        self, mock_get_tool, client, mock_integration_patch
    ):
        """Test image generation endpoint."""
        mock_tool = Mock()
        mock_tool.aexecute = AsyncMock(
            return_value={
                "success": True,
                "images": [{"url": "https://example.com/image.png"}],
            }
        )
        mock_get_tool.return_value = mock_tool

        payload = {"prompt": "A beautiful landscape", "model": "dall-e-3"}

        response = client.post("/images/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("src.llamaagent.api.openai_comprehensive_api.get_tool")
    def test_embeddings_endpoint(self, mock_get_tool, client, mock_integration_patch):
        """Test embeddings endpoint."""
        mock_tool = Mock()
        mock_tool.aexecute = AsyncMock(
            return_value={"success": True, "embeddings": [{"embedding": [0.1] * 1536}]}
        )
        mock_get_tool.return_value = mock_tool

        payload = {"texts": ["Test text"], "model": "text-embedding-3-large"}

        response = client.post("/embeddings", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @patch("src.llamaagent.api.openai_comprehensive_api.get_tool")
    def test_moderation_endpoint(self, mock_get_tool, client, mock_integration_patch):
        """Test moderation endpoint."""
        mock_tool = Mock()
        mock_tool.aexecute = AsyncMock(
            return_value={"success": True, "moderation_results": [{"flagged": False}]}
        )
        mock_get_tool.return_value = mock_tool

        payload = {"content": ["Safe content"], "model": "text-moderation-latest"}

        response = client.post("/moderations", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_batch_processing_endpoint(self, client, mock_integration_patch):
        """Test batch processing endpoint."""
        payload = [
            {
                "id": "req1",
                "type": "chat",
                "params": {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "model": "gpt-4o-mini",
                },
            },
            {
                "id": "req2",
                "type": "chat",
                "params": {
                    "messages": [{"role": "user", "content": "World"}],
                    "model": "gpt-4o-mini",
                },
            },
        ]

        response = client.post("/batch/process", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "batch_stats" in data["data"]
        assert data["data"]["batch_stats"]["total_requests"] == 2

    def test_usage_summary_endpoint(self, client, mock_integration_patch):
        """Test usage summary endpoint."""
        mock_integration_patch.get_usage_summary.return_value = {
            "total_cost": 15.0,
            "total_requests": 10,
            "usage_by_model": {},
        }

        response = client.get("/usage/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_cost" in data["data"]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_integration(self):
        """Test budget exceeded handling in integration."""
        config = OpenAIComprehensiveConfig(
            api_key="test-key",
            budget_limit=0.001,  # Very low limit
        )

        integration = OpenAIComprehensiveIntegration(config)

        # Mock the client properly for async operations
        with patch.object(integration, "client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=BudgetExceededError("Budget exceeded")
            )

            with pytest.raises(BudgetExceededError):
                await integration.chat_completion([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test API error handling."""
        config = OpenAIComprehensiveConfig(api_key="test-key")
        integration = OpenAIComprehensiveIntegration(config)

        # Mock client to raise exception
        integration.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception):
            await integration.chat_completion([{"role": "user", "content": "test"}])

    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        integration = OpenAIComprehensiveIntegration()

        # Should return default config for unknown model
        config = integration.get_model_config("non-existent-model")
        assert config.model_name == "non-existent-model"
        assert config.model_type == OpenAIModelType.FLAGSHIP_CHAT

    @pytest.mark.asyncio
    async def test_file_handling_errors(self):
        """Test file handling error cases."""
        integration = OpenAIComprehensiveIntegration()

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            await integration.transcribe_audio("non_existent_file.mp3")


class TestPerformanceAndScaling:
    """Test performance and scaling considerations."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        config = OpenAIComprehensiveConfig(api_key="test-key")
        integration = OpenAIComprehensiveIntegration(config)

        # Mock successful responses
        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
            "usage": {"total_tokens": 10},
        }
        integration.client.chat.completions.create = AsyncMock(
            return_value=type("obj", (object,), mock_response)
        )

        # Run multiple concurrent requests
        tasks = []
        for i in range(5):
            task = integration.chat_completion(
                [{"role": "user", "content": f"Test {i}"}]
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)

    def test_memory_usage_tracking(self):
        """Test memory usage in tracking."""
        tracker = OpenAIUsageTracker()

        # Add many usage records
        for i in range(1000):
            tracker.add_usage(
                model="gpt-4o-mini",
                input_tokens=10,
                output_tokens=5,
                operation=f"test_{i}",
            )

        assert tracker.request_count == 1000
        assert len(tracker.usage_log) == 1000

        # Summary should still work efficiently
        summary = tracker.get_usage_summary()
        assert summary["total_requests"] == 1000


# Integration tests that require actual API keys (skip if not available)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("SKIP_OPENAI_TESTS") == "true",
    reason="OpenAI API key not available or tests skipped",
)
class TestRealAPIIntegration:
    """Test real API integration (requires API key)."""

    @pytest.mark.asyncio
    async def test_real_chat_completion(self):
        """Test real chat completion."""
        # This test should only run with a real API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        integration = create_comprehensive_openai_integration(
            budget_limit=1.0  # Low limit for testing
        )

        result = await integration.chat_completion(
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            model="gpt-4o-mini",
            max_tokens=10,
        )

        # Check if 'test' or 'successful' is in the response (more flexible)
        response_text = result["choices"][0]["message"]["content"].lower()
        assert "test" in response_text or "successful" in response_text

    @pytest.mark.asyncio
    async def test_real_embeddings(self):
        """Test real embeddings creation."""
        # This test should only run with a real API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        integration = create_comprehensive_openai_integration(budget_limit=1.0)

        # Ensure client is properly initialized for real API calls
        if integration.client is None:
            pytest.skip("OpenAI client not initialized")

        result = await integration.create_embeddings(
            input_texts=["Test embedding"],
            model="text-embedding-3-small",  # Cheaper model
        )

        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) > 0

    @pytest.mark.asyncio
    async def test_real_moderation(self):
        """Test real content moderation."""
        # This test should only run with a real API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")

        integration = create_comprehensive_openai_integration()

        # Ensure client is properly initialized for real API calls
        if integration.client is None:
            pytest.skip("OpenAI client not initialized")

        result = await integration.moderate_content(
            input_text=["This is safe content for testing."],
            model="text-moderation-latest",
        )

        assert len(result["results"]) == 1
        assert "flagged" in result["results"][0]


class TestMockedAPIIntegration:
    """Test API integration with mocks (no API key required)."""

    @pytest.mark.asyncio
    async def test_mocked_chat_completion(self):
        """Test chat completion with mocked client."""
        # Create integration without API key (will use mock mode)
        integration = create_comprehensive_openai_integration(
            api_key=None, budget_limit=1.0
        )

        # Mock the client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test successful"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        integration.client = mock_client

        # Test
        result = await integration.chat_completion(
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            model="gpt-4o-mini",
            max_tokens=10,
        )

        # Verify - adjust assertion to match actual response structure
        assert hasattr(result, "choices") or "choices" in result
        if hasattr(result, "choices"):
            assert "test successful" in result.choices[0].message.content.lower()
        else:
            assert (
                "test successful" in result["choices"][0]["message"]["content"].lower()
            )

    @pytest.mark.asyncio
    async def test_mocked_embeddings(self):
        """Test embeddings with mocked client."""
        integration = create_comprehensive_openai_integration(
            api_key=None, budget_limit=1.0
        )

        # Mock the client
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_response.usage = Mock(prompt_tokens=5, total_tokens=5)

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        integration.client = mock_client

        # Test
        result = await integration.create_embeddings(
            input_texts=["Test embedding"], model="text-embedding-3-small"
        )

        # Verify
        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) > 0

    @pytest.mark.asyncio
    async def test_mocked_moderation(self):
        """Test moderation with mocked client."""
        integration = create_comprehensive_openai_integration(api_key=None)

        # Mock the client
        mock_response = Mock()
        mock_response.results = [Mock(flagged=False)]

        mock_client = AsyncMock()
        mock_client.moderations.create = AsyncMock(return_value=mock_response)
        integration.client = mock_client

        # Test
        result = await integration.moderate_content(
            input_text=["This is safe content for testing."],
            model="text-moderation-latest",
        )

        # Verify
        assert len(result["results"]) == 1
        assert "flagged" in result["results"][0]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
