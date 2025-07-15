# LlamaAgent Comprehensive Integration Guide

**Author: Nik Jois <nikjois@llamasearch.ai>**

This guide demonstrates the complete LLM and data integration capabilities of LlamaAgent, including multiple LLM providers, advanced data management with SQLite/PostgreSQL/Vector databases, FastAPI endpoints, CLI tools, and Datasette integration.

## Overview

LlamaAgent now provides comprehensive support for:

- **Multiple LLM Providers**: OpenAI, Anthropic, Cohere, Together AI, Ollama
- **Advanced Data Management**: SQLite with FTS, PostgreSQL, Vector databases (ChromaDB, Qdrant)
- **Web API**: FastAPI with authentication, rate limiting, and comprehensive endpoints
- **CLI Tools**: Rich command-line interface with data exploration
- **Data Visualization**: Datasette integration for interactive data exploration
- **Security**: Rate limiting, input validation, authentication
- **Monitoring**: Usage statistics, cost tracking, performance metrics

## Quick Start

### 1. Installation

```bash
# Install with comprehensive dependencies
pip install -e .

# Or install specific optional dependencies
pip install llamaagent[llm,data,api,cli]
```

### 2. Environment Setup

Create a `.env` file:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
TOGETHER_API_KEY=your_together_key_here

# Database Configuration
DATABASE_URL=sqlite:///./llamaagent.db
POSTGRES_URL=postgresql://user:pass@localhost/llamaagent
VECTOR_DB_PATH=./vector_db

# API Configuration
SECRET_KEY=your_secret_key_here
API_HOST=0.0.0.0
API_PORT=8000

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
```

### 3. Initialize a New Project

```bash
# Create a new project with advanced template
llamaagent init my-ai-project --template advanced

cd my-ai-project
```

## LLM Provider Integration

### Command Line Usage

```bash
# List available providers and models
llamaagent llm providers

# Chat with different providers
llamaagent llm chat openai gpt-4 "Explain quantum computing"
llamaagent llm chat anthropic claude-3-sonnet "Write a Python function"
llamaagent llm chat cohere command "Summarize this text"
llamaagent llm chat together llama-2-70b "Generate code"

# Search conversation history
llamaagent llm search "quantum computing" --limit 5

# View usage statistics
llamaagent llm stats

# Start Datasette for data exploration
llamaagent llm serve --port 8001
```

### Python API Usage

```python
import asyncio
from llamaagent.llm.factory import LLMFactory
from llamaagent.storage.database import DatabaseManager

async def main():
    # Initialize LLM factory
    factory = LLMFactory()
    
    # Create provider
    provider = await factory.create_provider("openai", "gpt-4")
    
    # Generate response
    response = await provider.generate_response(
        prompt="Explain machine learning",
        max_tokens=500,
        temperature=0.7
    )
    
    print(f"Response: {response.content}")
    print(f"Cost: ${response.cost:.4f}")
    print(f"Tokens: {response.usage['total_tokens']}")

# Run the example
asyncio.run(main())
```

### Streaming Responses

```python
async def streaming_example():
    provider = await factory.create_provider("openai", "gpt-4")
    
    async for chunk in provider.generate_streaming_response(
        prompt="Write a story about AI",
        max_tokens=1000
    ):
        print(chunk, end="", flush=True)
```

## Data Management

### Database Integration

```python
from llamaagent.storage.database import DatabaseManager, DatabaseConfig

async def database_example():
    # Configure database
    config = DatabaseConfig(
        sqlite_path="my_project.db",
        postgres_url="postgresql://localhost/llamaagent",
        vector_backend="chroma",
        auto_migrate=True
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    
    # Save conversation
    conversation_id = await db_manager.save_conversation(
        provider="openai",
        model="gpt-4",
        prompt="What is AI?",
        response="AI is artificial intelligence...",
        tokens_used=100,
        cost=0.002,
        metadata={"category": "education"}
    )
    
    # Search conversations
    results = await db_manager.search_conversations(
        query="artificial intelligence",
        limit=10,
        provider="openai"
    )
    
    # Get statistics
    stats = await db_manager.get_conversation_stats()
    print(f"Total conversations: {stats['total_conversations']}")
    
    # Export data
    export_path = await db_manager.export_data("conversations", "json")
    print(f"Data exported to: {export_path}")
```

### Vector Database Integration

```python
async def vector_example():
    # Save embeddings
    embedding_id = await db_manager.save_embedding(
        text="Machine learning is a subset of AI",
        embedding=[0.1, 0.2, 0.3, ...],  # Your embedding vector
        model="text-embedding-ada-002",
        metadata={"topic": "ML"}
    )
    
    # Similarity search
    results = await db_manager.similarity_search(
        query_embedding=[0.1, 0.2, 0.3, ...],
        limit=5,
        threshold=0.8
    )
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
```

## FastAPI Web Interface

### Starting the Server

```bash
# Start the API server
llamaagent serve --host 0.0.0.0 --port 8000 --workers 4

# Or with auto-reload for development
llamaagent serve --reload
```

### API Endpoints

The API provides comprehensive endpoints:

- **Authentication**: `POST /auth/login`
- **LLM Chat**: `POST /chat`
- **Streaming Chat**: `POST /chat` (with `stream=true`)
- **Search**: `POST /search`
- **Embeddings**: `POST /embeddings`
- **Similarity Search**: `POST /similarity-search`
- **Statistics**: `GET /stats`
- **Data Export**: `POST /export/{table}`
- **Health Check**: `GET /health`
- **Providers**: `GET /providers`

### Example API Usage

```python
import httpx
import asyncio

async def api_example():
    async with httpx.AsyncClient() as client:
        # Login
        login_response = await client.post(
            "http://localhost:8000/auth/login",
            params={"username": "admin", "password": "admin"}
        )
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Chat
        chat_response = await client.post(
            "http://localhost:8000/chat",
            headers=headers,
            json={
                "provider": "openai",
                "model": "gpt-4",
                "prompt": "Hello, API!",
                "max_tokens": 100,
                "temperature": 0.7
            }
        )
        
        result = chat_response.json()
        print(f"Response: {result['content']}")
        print(f"Cost: ${result['cost']:.4f}")
        
        # Search
        search_response = await client.post(
            "http://localhost:8000/search",
            headers=headers,
            json={"query": "API", "limit": 5}
        )
        
        results = search_response.json()["results"]
        print(f"Found {len(results)} conversations")

asyncio.run(api_example())
```

### JavaScript/Frontend Usage

```javascript
// Authentication
const loginResponse = await fetch('/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'username=admin&password=admin'
});
const {access_token} = await loginResponse.json();

// Chat
const chatResponse = await fetch('/chat', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${access_token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        provider: 'openai',
        model: 'gpt-4',
        prompt: 'Hello from JavaScript!',
        max_tokens: 100
    })
});

const result = await chatResponse.json();
console.log('Response:', result.content);
```

## Data Exploration with Datasette

### Starting Datasette

```bash
# Start Datasette server for data exploration
llamaagent datasette --port 8001 --host 0.0.0.0

# Or use the LLM CLI
llamaagent llm serve --port 8001
```

### Datasette Features

- **Full-text search** on conversations and knowledge base
- **SQL queries** with advanced filtering
- **Data visualization** with charts and graphs
- **Export capabilities** (CSV, JSON)
- **API access** to all data
- **Custom metadata** and descriptions

### Example Datasette Queries

```sql
-- Find expensive conversations
SELECT provider, model, cost, prompt, response 
FROM conversations 
WHERE cost > 0.01 
ORDER BY cost DESC;

-- Usage by provider
SELECT provider, 
       COUNT(*) as conversation_count,
       SUM(cost) as total_cost,
       AVG(tokens_used) as avg_tokens
FROM conversations 
GROUP BY provider;

-- Recent activity
SELECT DATE(timestamp) as date, 
       COUNT(*) as conversations,
       SUM(cost) as daily_cost
FROM conversations 
WHERE timestamp >= date('now', '-30 days')
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Full-text search
SELECT * FROM conversations_fts 
WHERE conversations_fts MATCH 'machine learning OR artificial intelligence'
ORDER BY rank;
```

## Advanced Features

### Rate Limiting and Security

```python
from llamaagent.security import RateLimiter, RateLimitRule, InputValidator

# Rate limiting
rate_limiter = RateLimiter()
rule = RateLimitRule(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_size=10
)

allowed, metadata = await rate_limiter.is_allowed("user123", rule)
if not allowed:
    print(f"Rate limited. Retry after {metadata['retry_after']} seconds")

# Input validation
validator = InputValidator()
result = validator.validate_text_input("User input here")
if not result["is_valid"]:
    print(f"Invalid input: {result['threats']}")
```

### Monitoring and Analytics

```python
# Get comprehensive statistics
stats = await db_manager.get_conversation_stats()

print(f"Total conversations: {stats['total_conversations']}")
print(f"Providers: {list(stats['by_provider'].keys())}")
print(f"Recent activity: {len(stats['recent_activity'])} days")

# Provider performance
for provider, data in stats['by_provider'].items():
    print(f"{provider}: {data['count']} conversations, ${data['total_cost']:.4f}")
```

### Backup and Export

```bash
# Create database backup
llamaagent backup

# Export data in different formats
llamaagent export conversations --format json
llamaagent export embeddings --format csv
```

### Performance Benchmarking

```bash
# Benchmark different providers
llamaagent benchmark --provider openai --model gpt-4 --iterations 10
llamaagent benchmark --provider anthropic --model claude-3-sonnet --iterations 10
llamaagent benchmark --provider cohere --model command --iterations 10
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
TOGETHER_API_KEY=...

# Optional
DATABASE_URL=sqlite:///./llamaagent.db
POSTGRES_URL=postgresql://user:pass@localhost/db
VECTOR_DB_PATH=./vector_db
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
RATE_LIMIT_ENABLED=true
```

### Configuration Files

```python
# config.py
from llamaagent.config.settings import Settings

settings = Settings(
    openai_api_key="your-key",
    database_url="sqlite:///./custom.db",
    vector_backend="chroma",
    rate_limit_enabled=True,
    log_level="INFO"
)
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["llamaagent", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaagent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamaagent
  template:
    metadata:
      labels:
        app: llamaagent
    spec:
      containers:
      - name: llamaagent
        image: llamaagent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: DATABASE_URL
          value: "postgresql://postgres:password@postgres:5432/llamaagent"
```

### Environment-Specific Configurations

```bash
# Development
export ENV=development
export LOG_LEVEL=DEBUG
export RATE_LIMIT_ENABLED=false

# Production
export ENV=production
export LOG_LEVEL=INFO
export RATE_LIMIT_ENABLED=true
export DATABASE_URL=postgresql://...
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all optional dependencies are installed
2. **API Key Issues**: Check environment variables are set correctly
3. **Database Issues**: Verify database permissions and connectivity
4. **Rate Limiting**: Check rate limit configuration
5. **Vector Database**: Ensure ChromaDB or Qdrant is properly configured

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
llamaagent serve --reload

# Check system status
llamaagent status
```

### Performance Optimization

```python
# Database optimization
config = DatabaseConfig(
    sqlite_path="llamaagent.db",
    enable_wal=True,  # Better concurrency
    postgres_pool_size=20,  # Larger pool for high traffic
    vector_backend="qdrant"  # Faster vector operations
)

# API optimization
llamaagent serve --workers 4 --host 0.0.0.0 --port 8000
```

## Contributing

This comprehensive integration demonstrates enterprise-grade LLM and data management capabilities. The system is designed to be:

- **Scalable**: Supports multiple databases and deployment patterns
- **Secure**: Includes authentication, rate limiting, and input validation
- **Observable**: Comprehensive logging, monitoring, and analytics
- **Extensible**: Plugin architecture for new providers and features
- **Production-Ready**: Docker, Kubernetes, and CI/CD support

For questions or contributions, contact Nik Jois <nikjois@llamasearch.ai>.

## License

MIT License - see LICENSE file for details. 