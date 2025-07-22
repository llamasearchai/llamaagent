# LlamaAgent Comprehensive OpenAI Integration

A complete, production-ready integration of all OpenAI APIs and model types into the LlamaAgent framework.

**Author:** Nik Jois  
**Email:** nikjois@llamasearch.ai

## Featured Overview

This comprehensive integration provides seamless access to the entire OpenAI ecosystem through a unified interface:

### Supported OpenAI Model Types

| Model Category | Models | Use Cases |
|----------------|---------|-----------|
| **Reasoning Models (o-series)** | o3-mini, o1, o1-mini | Complex problem-solving, mathematical reasoning, multi-step analysis |
| **Flagship Chat Models** | gpt-4o, gpt-4o-2024-11-20 | High-intelligence tasks, complex conversations, professional applications |
| **Cost-Optimized Models** | gpt-4o-mini | Efficient everyday tasks, high-volume applications |
| **Deep Research Models** | gpt-4o | Multi-step research, comprehensive analysis |
| **Realtime Models** | gpt-4o-realtime-preview | Real-time text and audio interactions |
| **Image Generation** | dall-e-3, dall-e-2 | Creative image generation, visual content creation |
| **Text-to-Speech** | tts-1, tts-1-hd | Audio synthesis, voice applications |
| **Transcription** | whisper-1 | Audio-to-text conversion, transcription services |
| **Embeddings** | text-embedding-3-large, text-embedding-3-small | Semantic search, clustering, similarity analysis |
| **Moderation** | text-moderation-latest | Content safety, policy compliance |
| **Legacy Models** | gpt-4, gpt-4-turbo, gpt-3.5-turbo | Backward compatibility |
| **Base Models** | babbage-002, davinci-002 | Fine-tuning foundations |

## LAUNCH: Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llamaagent

# Install with OpenAI integration
pip install -r requirements.txt
pip install -r requirements-openai.txt

# Or use Docker
docker-compose -f docker-compose.openai.yml up
```

### Basic Usage

```python
from llamaagent.integration.openai_comprehensive import create_comprehensive_openai_integration
from llamaagent.tools.openai_tools import create_all_openai_tools

# Initialize integration
integration = create_comprehensive_openai_integration(
    api_key="your-openai-api-key",
    budget_limit=100.0
)

# Create all OpenAI tools
tools = create_all_openai_tools(integration)

# Use reasoning model
reasoning_tool = next(tool for tool in tools if tool.name == "openai_reasoning")
result = await reasoning_tool.aexecute("Solve: What is the 10th Fibonacci number?")
print(result["response"])
```

### Environment Setup

```bash
# Required environment variables
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORGANIZATION="your-org-id"  # Optional
export LLAMAAGENT_BUDGET_LIMIT="100.0"    # Optional, default 100.0
```

## LIST: API Reference

### Integration Classes

#### `OpenAIComprehensiveIntegration`

Main integration class providing access to all OpenAI APIs.

```python
from llamaagent.integration.openai_comprehensive import OpenAIComprehensiveIntegration

integration = OpenAIComprehensiveIntegration(config)

# Chat completion
response = await integration.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini"
)

# Image generation
image = await integration.generate_image(
    prompt="A beautiful sunset",
    model="dall-e-3"
)

# Text-to-speech
audio = await integration.text_to_speech(
    input_text="Hello world",
    model="tts-1",
    voice="alloy"
)

# Transcription
transcript = await integration.transcribe_audio(
    audio_file="audio.mp3",
    model="whisper-1"
)

# Embeddings
embeddings = await integration.create_embeddings(
    input_texts=["Sample text"],
    model="text-embedding-3-large"
)

# Content moderation
moderation = await integration.moderate_content(
    input_text=["Content to check"],
    model="text-moderation-latest"
)
```

### Tools

#### Available Tools

| Tool | Class | Purpose |
|------|-------|---------|
| `openai_reasoning` | `OpenAIReasoningTool` | Advanced reasoning and problem-solving |
| `openai_image_generation` | `OpenAIImageGenerationTool` | Image generation with DALL-E |
| `openai_text_to_speech` | `OpenAITextToSpeechTool` | Text-to-speech conversion |
| `openai_transcription` | `OpenAITranscriptionTool` | Audio transcription |
| `openai_embeddings` | `OpenAIEmbeddingsTool` | Text embeddings generation |
| `openai_moderation` | `OpenAIModerationTool` | Content moderation |
| `openai_comprehensive` | `OpenAIComprehensiveTool` | Access to all APIs |

#### Tool Usage Examples

```python
from llamaagent.tools.openai_tools import create_openai_tool

# Create specific tools
reasoning_tool = create_openai_tool("reasoning")
image_tool = create_openai_tool("image_generation")

# Use tools
reasoning_result = await reasoning_tool.aexecute(
    problem="Calculate the area of a circle with radius 5",
    model="o3-mini"
)

image_result = await image_tool.aexecute(
    prompt="A futuristic city at night",
    model="dall-e-3",
    size="1024x1024",
    quality="hd"
)
```

### REST API Endpoints

#### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information and status |
| `/health` | GET | Comprehensive health check |
| `/budget` | GET | Current budget status |
| `/models` | GET | List available models |

#### Chat and Reasoning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/completions` | POST | Chat completions with all models |
| `/reasoning/solve` | POST | Reasoning model problem solving |

#### Media and Content

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/images/generate` | POST | Generate images with DALL-E |
| `/audio/speech` | POST | Text-to-speech conversion |
| `/audio/transcriptions` | POST | Audio transcription |
| `/embeddings` | POST | Text embeddings generation |
| `/moderations` | POST | Content moderation |

#### Advanced Features

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools/{tool_type}` | POST | Use any OpenAI tool |
| `/batch/process` | POST | Batch processing multiple requests |
| `/usage/summary` | GET | Usage analytics |
| `/usage/by-model` | GET | Usage breakdown by model |

#### API Usage Examples

```bash
# Chat completion
curl -X POST "http://localhost:8000/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "gpt-4o-mini"
  }'

# Image generation
curl -X POST "http://localhost:8000/images/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful landscape",
    "model": "dall-e-3",
    "size": "1024x1024"
  }'

# Text-to-speech
curl -X POST "http://localhost:8000/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model": "tts-1",
    "voice": "alloy"
  }'

# Audio transcription
curl -X POST "http://localhost:8000/audio/transcriptions" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"

# Embeddings
curl -X POST "http://localhost:8000/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Sample text"],
    "model": "text-embedding-3-large"
  }'

# Content moderation
curl -X POST "http://localhost:8000/moderations" \
  -H "Content-Type: application/json" \
  -d '{
    "content": ["Content to check"],
    "model": "text-moderation-latest"
  }'
```

##  Budget Management

### Automatic Budget Tracking

The system automatically tracks usage and costs across all OpenAI APIs:

```python
# Set budget limit
integration = create_comprehensive_openai_integration(
    budget_limit=50.0  # $50 limit
)

# Check budget status
budget_status = integration.get_budget_status()
print(f"Remaining: ${budget_status['remaining_budget']:.2f}")

# Usage summary
usage = integration.get_usage_summary()
print(f"Total requests: {usage['total_requests']}")
print(f"Cost per request: ${usage['cost_per_request']:.4f}")
```

### Cost Estimates (USD, as of 2025)

| Model | Input (per 1K tokens) | Output (per 1K tokens) | Typical Cost |
|-------|----------------------|------------------------|--------------|
| o3-mini | $0.00015 | $0.0006 | $0.001-0.01 |
| gpt-4o | $0.0025 | $0.01 | $0.01-0.10 |
| gpt-4o-mini | $0.00015 | $0.0006 | $0.001-0.01 |
| dall-e-3 | $0.04/image | - | $0.04-0.08 |
| tts-1 | $0.015/1K chars | - | $0.001-0.05 |
| whisper-1 | $0.006/minute | - | $0.01-0.10 |
| text-embedding-3-large | $0.00013 | - | $0.0001-0.001 |
| text-moderation-latest | Free | Free | $0.00 |

##  Docker Deployment

### Quick Start with Docker Compose

```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"
export LLAMAAGENT_BUDGET_LIMIT="100.0"

# Start all services
docker-compose -f docker-compose.openai.yml up -d

# Check service status
docker-compose -f docker-compose.openai.yml ps

# View logs
docker-compose -f docker-compose.openai.yml logs -f llamaagent-openai
```

### Services Included

- **llamaagent-openai**: Main API service
- **postgres**: Database for storage
- **redis**: Caching and session management
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboards
- **nginx**: Reverse proxy and load balancing
- **minio**: File storage for media
- **jaeger**: Distributed tracing
- **worker**: Background task processing
- **scheduler**: Periodic task execution

### Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Main API | http://localhost:8000 | OpenAI API endpoints |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics and monitoring |
| MinIO | http://localhost:9001 | File storage interface |
| Jaeger | http://localhost:16686 | Distributed tracing |

## Results Monitoring and Observability

### Built-in Monitoring

The system includes comprehensive monitoring:

- **Usage Tracking**: Automatic cost and token tracking
- **Performance Metrics**: Response times, success rates
- **Error Monitoring**: Detailed error logging and alerting
- **Budget Alerts**: Warnings when approaching limits

### Grafana Dashboards

Pre-configured dashboards for:
- API Usage and Costs
- Model Performance Comparison
- Error Rates and Types
- Budget Utilization
- System Health

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Budget status
curl http://localhost:8000/budget

# Usage summary
curl http://localhost:8000/usage/summary
```

## Testing Testing

### Running Tests

```bash
# Run all tests
pytest tests/test_openai_comprehensive_integration.py -v

# Run with coverage
pytest tests/test_openai_comprehensive_integration.py --cov=src/llamaagent

# Run specific test categories
pytest tests/test_openai_comprehensive_integration.py::TestOpenAITools -v

# Run tests with real API (requires API key)
OPENAI_API_KEY="your-key" pytest tests/test_openai_comprehensive_integration.py::TestRealAPIIntegration -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **API Tests**: REST endpoint validation
- **Performance Tests**: Load and stress testing
- **Real API Tests**: Live OpenAI API validation (optional)

### Docker Testing

```bash
# Run tests in Docker
docker-compose -f docker-compose.openai.yml --profile testing up test-runner

# View test results
docker-compose -f docker-compose.openai.yml logs test-runner
```

## Target Demo and Examples

### Running the Comprehensive Demo

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the comprehensive demo
python demo_openai_comprehensive.py
```

The demo showcases:
- All model types and APIs
- Budget tracking in action
- Error handling scenarios
- Performance benchmarking
- Cost analysis

### Example Outputs

The demo generates a detailed report including:
- Success rates for each API
- Total costs and token usage
- Performance metrics
- Budget utilization analysis

## Tools Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_ORGANIZATION` | None | OpenAI organization ID |
| `LLAMAAGENT_BUDGET_LIMIT` | 100.0 | Budget limit in USD |
| `LLAMAAGENT_LOG_LEVEL` | INFO | Logging level |
| `LLAMAAGENT_ENABLE_TRACING` | true | Enable distributed tracing |
| `LLAMAAGENT_ENABLE_MONITORING` | true | Enable metrics collection |

### Configuration File

```python
from llamaagent.integration.openai_comprehensive import OpenAIComprehensiveConfig

config = OpenAIComprehensiveConfig(
    api_key="your-api-key",
    budget_limit=100.0,
    timeout=60.0,
    max_retries=3,
    enable_usage_tracking=True,
    enable_cost_warnings=True,
    default_models={
        OpenAIModelType.REASONING: "o3-mini",
        OpenAIModelType.FLAGSHIP_CHAT: "gpt-4o",
        OpenAIModelType.COST_OPTIMIZED: "gpt-4o-mini",
        # ... more model mappings
    }
)
```

## Alert Error Handling

### Common Error Scenarios

1. **Budget Exceeded**
   ```python
   try:
       result = await integration.chat_completion(messages)
   except BudgetExceededError as e:
       logger.error(f"Budget limit reached: {e}")
   ```

2. **Invalid Model**
   ```python
   try:
       result = await integration.chat_completion(messages, model="invalid-model")
   except Exception as e:
       logger.error(f"Model error: {e}")
   ```

3. **API Rate Limits**
   - Automatic retry with exponential backoff
   - Rate limit tracking and warnings
   - Queue-based request management

### Error Response Format

```json
{
  "success": false,
  "error": "Error description",
  "error_type": "BudgetExceededError",
  "timestamp": "2025-01-18T10:30:00Z",
  "request_id": "req_123",
  "usage": {
    "budget_remaining": 45.67,
    "total_cost": 54.33
  }
}
```

## Security Security

### API Key Management

- Environment variable storage
- No hardcoding in source code
- Optional organization-level isolation
- Secure credential injection in Docker

### Rate Limiting

- Built-in rate limit tracking
- Configurable limits per model
- Automatic backoff and retry
- Budget-based throttling

### Input Validation

- Comprehensive input sanitization
- File upload restrictions
- Content length limits
- Model-specific parameter validation

## Performance Performance Optimization

### Async/Await Support

All operations are fully asynchronous:

```python
# Concurrent requests
tasks = [
    integration.chat_completion(msg1),
    integration.chat_completion(msg2),
    integration.create_embeddings(texts)
]
results = await asyncio.gather(*tasks)
```

### Caching

- Redis-based response caching
- Embeddings cache for repeated texts
- Model metadata caching
- Budget calculation optimization

### Batch Processing

```python
# Batch API endpoint
batch_requests = [
    {"type": "chat", "params": {"messages": [...], "model": "gpt-4o-mini"}},
    {"type": "embeddings", "params": {"texts": [...]}},
]

batch_response = await client.post("/batch/process", json=batch_requests)
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd llamaagent

# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest

# Run linters
flake8 src/
black src/
mypy src/
```

### Code Standards

- Type hints for all functions
- Comprehensive docstrings
- 100% test coverage for critical paths
- Follow PEP 8 style guidelines
- Use async/await patterns

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

- **Email**: nikjois@llamasearch.ai
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

##  Roadmap

### Planned Features

- [ ] OpenAI Assistants API integration
- [ ] Fine-tuning workflow support
- [ ] Advanced function calling
- [ ] Multi-modal conversation support
- [ ] Custom model integration
- [ ] Advanced analytics dashboard
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline templates

### Version History

- **v1.0.0**: Initial comprehensive OpenAI integration
- **v1.1.0**: Added Docker support and monitoring
- **v1.2.0**: Enhanced batch processing and caching
- **v1.3.0**: Advanced error handling and recovery

---

**Built with LOVE: by the LlamaAgent team** 