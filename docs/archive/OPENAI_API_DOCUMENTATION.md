# OpenAI Comprehensive API Documentation

This FastAPI implementation provides a complete REST API for all OpenAI models and services through the LlamaAgent framework.

## API Overview

The API is implemented in `/src/llamaagent/api/openai_comprehensive_api.py` and provides endpoints for:

- **Reasoning models** (o-series)
- **Flagship chat models**
- **Cost-optimized models**
- **Deep research models**
- **Image generation models**
- **Text-to-speech models**
- **Transcription models**
- **Embeddings models**
- **Moderation models**
- **Batch processing**
- **Usage analytics**

## Base URL

```
http://localhost:8000
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Status Endpoints

#### 1. Root Endpoint
- **GET** `/`
- Returns API information including version, status, available model types, and tools

#### 2. Health Check
- **GET** `/health`
- Returns comprehensive health status including API accessibility, budget status, and model availability

#### 3. Budget Status
- **GET** `/budget`
- Returns current budget information and usage limits

### Model Management

#### 4. List Models
- **GET** `/models`
- **Query Parameters:**
  - `model_type` (optional): Filter by model type (e.g., "reasoning", "chat", "embeddings")
- Returns available models, optionally filtered by type

### Chat Endpoints

#### 5. Chat Completions
- **POST** `/chat/completions`
- **Request Body:**
  ```json
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Hello!"}
    ],
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": false,
    "tools": []
  }
  ```
- Supports all OpenAI chat models with optional streaming and function calling

### Reasoning Endpoints

#### 6. Reasoning Solve
- **POST** `/reasoning/solve`
- **Request Body:**
  ```json
  {
    "problem": "Solve this complex math problem...",
    "model": "o3-mini",
    "temperature": 0.1,
    "max_tokens": 4000
  }
  ```
- Uses advanced reasoning models for complex problem-solving

### Image Generation

#### 7. Generate Images
- **POST** `/images/generate`
- **Request Body:**
  ```json
  {
    "prompt": "A beautiful sunset over mountains",
    "model": "dall-e-3",
    "size": "1024x1024",
    "quality": "standard",
    "n": 1,
    "style": "vivid"
  }
  ```
- Generates images using DALL-E models

### Audio Endpoints

#### 8. Text-to-Speech
- **POST** `/audio/speech`
- **Request Body:**
  ```json
  {
    "text": "Hello, this is a test",
    "model": "tts-1",
    "voice": "alloy",
    "response_format": "mp3",
    "speed": 1.0
  }
  ```
- Converts text to speech audio

#### 9. Audio Transcription
- **POST** `/audio/transcriptions`
- **Form Data:**
  - `file`: Audio file to transcribe
  - `model`: "whisper-1" (default)
  - `language`: Language code (optional)
  - `prompt`: Context prompt (optional)
  - `response_format`: "json" (default)
  - `temperature`: 0.0 (default)
- Transcribes audio to text

#### 10. Download Speech File
- **GET** `/audio/speech/{file_id}`
- Downloads generated speech audio files

### Embeddings

#### 11. Create Embeddings
- **POST** `/embeddings`
- **Request Body:**
  ```json
  {
    "texts": "Text to embed" or ["Multiple", "texts"],
    "model": "text-embedding-3-large",
    "dimensions": 1536
  }
  ```
- Creates text embeddings for similarity search

### Moderation

#### 12. Content Moderation
- **POST** `/moderations`
- **Request Body:**
  ```json
  {
    "content": "Text to moderate",
    "model": "text-moderation-latest"
  }
  ```
- Checks content for safety and policy compliance

### Batch Processing

#### 13. Batch Process
- **POST** `/batch/process`
- **Request Body:**
  ```json
  [
    {
      "type": "chat",
      "id": "req_1",
      "params": {
        "messages": [...],
        "model": "gpt-4o-mini"
      }
    },
    {
      "type": "embeddings",
      "id": "req_2",
      "params": {
        "texts": ["Text 1", "Text 2"],
        "model": "text-embedding-3-small"
      }
    }
  ]
  ```
- Processes multiple requests in a single batch

### Analytics

#### 14. Usage Summary
- **GET** `/usage/summary`
- Returns comprehensive usage statistics

#### 15. Usage by Model
- **GET** `/usage/by-model`
- Returns usage breakdown by model type

### Tools

#### 16. Generic Tool Endpoint
- **POST** `/tools/{tool_type}`
- **Path Parameters:**
  - `tool_type`: Type of tool (e.g., "comprehensive", "reasoning", "embeddings")
- **Request Body:** Tool-specific parameters
- Provides access to any OpenAI tool with arbitrary parameters

## Response Format

All endpoints return responses in the following format:

```json
{
  "success": true,
  "data": {
    // Response data specific to the endpoint
  },
  "error": null,
  "timestamp": "2024-01-20T12:00:00Z",
  "usage": {
    // Optional usage statistics
  }
}
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid parameters
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Budget exceeded
- **500 Internal Server Error**: Server errors

Error responses include detailed error messages:

```json
{
  "success": false,
  "data": null,
  "error": "Detailed error message",
  "timestamp": "2024-01-20T12:00:00Z"
}
```

## Authentication

The API uses the OpenAI API key configured in the integration. Set your API key through environment variables or configuration.

## CORS Support

The API includes CORS middleware configured to allow all origins, methods, and headers. Adjust the CORS settings in production as needed.

## Running the API

To run the API server:

```bash
python -m llamaagent.api.openai_comprehensive_api
```

Or programmatically:

```python
import uvicorn
from llamaagent.api.openai_comprehensive_api import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Integration with LlamaAgent

The API integrates with:
- `OpenAIComprehensiveIntegration`: Main integration class
- `OpenAIModelType`: Enum of available model types
- `OPENAI_TOOLS`: Registry of available tools
- Various tool implementations for each OpenAI service

## Example Usage

### Python Requests Example

```python
import requests

# Chat completion
response = requests.post(
    "http://localhost:8000/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "model": "gpt-4o-mini"
    }
)
print(response.json())

# Image generation
response = requests.post(
    "http://localhost:8000/images/generate",
    json={
        "prompt": "A futuristic city at night",
        "model": "dall-e-3"
    }
)
print(response.json())
```

### cURL Example

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt-4o-mini"
  }'
```

## Notes

- The API is designed to be extensible and can easily accommodate new OpenAI models and features
- All endpoints include proper error handling and validation
- The batch processing endpoint allows efficient processing of multiple requests
- File uploads (audio transcription) are handled with temporary file management
- The API includes comprehensive logging for debugging and monitoring
