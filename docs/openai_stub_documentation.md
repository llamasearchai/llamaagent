# OpenAI API Stub Documentation

## Overview

The OpenAI API stub (`src/llamaagent/integration/_openai_stub.py`) provides a mock implementation of the OpenAI API for testing and development purposes. This allows developers to:

- Write and test code that uses OpenAI without making real API calls
- Develop offline without incurring API costs
- Create deterministic tests with predictable responses
- Test error handling and edge cases

## Features

### Mock Client Classes

1. **MockOpenAIClient** - Synchronous client that mimics `openai.OpenAI`
2. **MockAsyncOpenAIClient** - Asynchronous client that mimics `openai.AsyncOpenAI`

### Supported Endpoints

#### Chat Completions
- `client.chat.completions.create()` - Generate chat responses
- Returns contextual responses based on input keywords:
  - "calculate" → "The calculation result is 42."
  - "code" → Python code snippet
  - "test successful" → "test successful"
  - Default → "This is a mock response from OpenAI stub."

#### Embeddings
- `client.embeddings.create()` - Generate text embeddings
- Returns deterministic 1536-dimensional vectors
- Supports single text or batch processing
- Embeddings are consistent for the same input

#### Moderations
- `client.moderations.create()` - Check content for policy violations
- Flags content containing keywords: "violence", "hate", "harassment", "self-harm", "sexual"
- Returns appropriate category flags and scores

### Mock Response Objects

All response objects match OpenAI's API format:
- `MockChatCompletion` - Chat completion responses
- `MockEmbedding` - Embedding responses
- `MockModeration` - Moderation responses

Each includes proper token usage tracking and metadata.

## Usage

### Basic Usage

```python
from src.llamaagent.integration._openai_stub import install_openai_stub, uninstall_openai_stub

# Install the stub to intercept OpenAI imports
install_openai_stub()

# Now import and use OpenAI normally
import openai

client = openai.OpenAI(api_key="sk-test-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Clean up when done
uninstall_openai_stub()
```

### Async Usage

```python
install_openai_stub()
import openai

async_client = openai.AsyncOpenAI(api_key="sk-test-key")
response = await async_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Testing Example

```python
def test_my_openai_function():
    install_openai_stub()
    
    # Your code that uses OpenAI
    result = my_function_that_uses_openai()
    
    # Assertions
    assert result is not None
    
    uninstall_openai_stub()
```

## Integration Notes

### LlamaAgent OpenAIProvider

The `OpenAIProvider` class in LlamaAgent uses `httpx` for direct HTTP API calls rather than the `openai` library. Therefore, the stub does not intercept these calls. To test OpenAIProvider, you'll need to mock the httpx requests separately.

### Components Using OpenAI Directly

Any component that imports and uses the `openai` library directly will benefit from this stub. This includes:
- Custom tools or plugins
- Vector stores using OpenAI embeddings
- Content moderation systems
- Any third-party integrations

## Advanced Features

### Deterministic Embeddings

The stub generates deterministic embeddings based on the input text hash, ensuring consistent results across test runs:

```python
# Same input always produces same embedding
embedding1 = client.embeddings.create(input="test").data[0].embedding
embedding2 = client.embeddings.create(input="test").data[0].embedding
assert embedding1 == embedding2
```

### Token Usage Tracking

All responses include realistic token usage calculations:
- Prompt tokens: ~1 token per 4 characters
- Completion tokens: Based on response length
- Total tokens: Sum of prompt and completion

### Error Simulation

While not fully implemented in the current version, the stub infrastructure supports error simulation:
- Authentication errors for specific API key patterns
- Rate limiting simulation
- Connection errors

## Best Practices

1. **Always clean up**: Use `uninstall_openai_stub()` in test teardown
2. **Install before imports**: Install the stub before any code imports `openai`
3. **Use context managers**: Consider wrapping in try/finally for cleanup
4. **Test both sync and async**: Ensure your code works with both client types

## Limitations

1. **No streaming support**: The stub doesn't implement streaming responses
2. **Basic content generation**: Responses are simple and keyword-based
3. **No function calling**: Tool/function calling features aren't implemented
4. **No fine-tuning APIs**: Only core completion/embedding/moderation APIs

## Future Enhancements

Potential improvements for the stub:
1. Streaming response support
2. Function/tool calling simulation
3. More sophisticated response generation
4. Configuration for custom responses
5. Response delay simulation
6. Error injection for resilience testing

## Example Test Suite

See the following test files for comprehensive examples:
- `tests/test_openai_stub.py` - Core stub functionality tests
- `tests/test_openai_stub_integration.py` - Integration with LlamaAgent
- `tests/test_openai_stub_direct.py` - Direct usage examples