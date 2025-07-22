# Simon Willison's LLM Ecosystem - Complete Integration

**Comprehensive integration of Simon Willison's LLM ecosystem with LlamaAgent**

A fully-featured, production-ready system that brings together Simon's entire LLM toolkit with LlamaAgent's agent framework, providing the most complete LLM integration available.

**Author:** Nik Jois <nikjois@llamasearch.ai>

## Overview

This integration combines:

- **Simon's LLM Core**: Universal interface to 100+ language models
- **Provider Integrations**: OpenAI, Anthropic, Gemini, Mistral, Ollama, and more
- **Powerful Tools**: SQLite, Datasette, Docker, JavaScript execution
- **Data Management**: sqlite-utils, conversation logging, analytics
- **FastAPI Endpoints**: Complete REST API with streaming support
- **Production Deployment**: Docker, monitoring, scaling

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llamaagent

# Install with Simon's ecosystem
pip install -r requirements.txt
pip install -r requirements-openai.txt

# Install Simon's LLM tools
pip install llm llm-anthropic llm-openai-plugin llm-gemini sqlite-utils datasette
```

### 2. Configuration

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Database
LLM_DATABASE_PATH=simon_ecosystem.db

# Security
ENABLE_COMMAND_TOOL=false
```

### 3. Quick Start

```python
from src.llamaagent.llm.simon_ecosystem import SimonLLMEcosystem, SimonEcosystemConfig

# Initialize ecosystem
config = SimonEcosystemConfig()
ecosystem = SimonLLMEcosystem(config)

# Chat with any LLM
response = await ecosystem.chat(
    "Explain quantum computing",
    model="gpt-4o-mini"
)

# Generate embeddings
embeddings = await ecosystem.embed(
    ["Text 1", "Text 2"],
    model="text-embedding-3-small"
)

# Use tools
result = await ecosystem.use_tool(
    "sqlite",
    "query", 
    sql="SELECT COUNT(*) FROM conversations"
)
```

## Features

### Agent Multi-Provider LLM Support

Support for all major LLM providers through Simon's ecosystem:

```python
# OpenAI
await ecosystem.chat("Hello", model="gpt-4o", provider=LLMProvider.OPENAI)

# Anthropic
await ecosystem.chat("Hello", model="claude-3-haiku", provider=LLMProvider.ANTHROPIC)

# Google
await ecosystem.chat("Hello", model="gemini-pro", provider=LLMProvider.GEMINI)

# Local models via Ollama
await ecosystem.chat("Hello", model="llama3.2", provider=LLMProvider.OLLAMA)
```

### BUILD: Powerful Tool Integration

#### SQLite Operations
```python
# Create tables
await ecosystem.use_tool("sqlite", "create_table", 
    table="users", 
    schema={"id": int, "name": str, "email": str}
)

# Insert data
await ecosystem.use_tool("sqlite", "insert",
    table="users",
    data={"id": 1, "name": "John", "email": "john@example.com"}
)

# Query data
results = await ecosystem.use_tool("sqlite", "query",
    sql="SELECT * FROM users WHERE name LIKE 'John%'"
)
```

#### Code Execution
```python
# Python execution
result = await ecosystem.use_tool("python", "run",
    code="print('Hello from Python!')"
)

# JavaScript execution
result = await ecosystem.use_tool("quickjs", "run",
    code="console.log('Hello from JS!')"
)

# Docker execution (safe sandboxing)
result = await ecosystem.use_tool("docker", "run",
    code="print('Hello from Docker!')",
    language="python"
)
```

### Results Data Analytics & Visualization

#### Conversation Analytics
```python
# Get usage statistics
stats = await ecosystem.get_conversation_stats()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total cost: ${stats['total_cost']:.4f}")

# Search conversation history
results = await ecosystem.search_conversations("machine learning", limit=10)
```

#### Data Export
```python
# Export to JSON
await ecosystem.export_data("conversations", "json", "export.json")

# Export to CSV
await ecosystem.export_data("conversations", "csv", "export.csv")
```

### Network FastAPI REST API

Complete REST API with all ecosystem features:

```bash
# Start the API server
python -m src.llamaagent.api.simon_ecosystem_api

# Or with the deployment script
./scripts/simon_ecosystem_deploy.sh development
```

#### API Endpoints

**Chat Completions**
```bash
# Simple chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain AI", "model": "gpt-4o-mini"}'

# Streaming chat
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell a story", "stream": true}'
```

**Embeddings**
```bash
curl -X POST http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": ["Hello", "World"], "model": "text-embedding-3-small"}'
```

**Database Operations**
```bash
curl -X POST http://localhost:8000/database/query \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT COUNT(*) FROM conversations"}'
```

**Code Execution**
```bash
# Python
curl -X POST http://localhost:8000/execute/python \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello World\")"}'

# JavaScript
curl -X POST http://localhost:8000/execute/javascript \
  -H "Content-Type: application/json" \
  -d '{"code": "console.log(\"Hello World\")"}'
```

###  Jupyter Notebooks

Complete cookbook with examples:

```bash
# Start Jupyter
jupyter lab notebooks/02_simon_ecosystem_cookbook.ipynb

# Or with Docker
docker-compose -f docker-compose.simon.yml up jupyter
```

The cookbook covers:
- Multi-provider chat examples
- Database operations
- Code execution
- Analytics and visualization
- Production deployment

###  Docker Deployment

#### Development Deployment
```bash
# Quick start
./scripts/simon_ecosystem_deploy.sh development

# Manual Docker Compose
docker-compose -f docker-compose.simon.yml up -d
```

#### Production Deployment
```bash
# Full production stack
./scripts/simon_ecosystem_deploy.sh production

# Includes:
# - Load balancing with Nginx
# - Monitoring with Prometheus + Grafana
# - Redis caching
# - PostgreSQL analytics
# - Backup services
```

### Performance Monitoring & Analytics

#### Datasette Integration
```bash
# Automatic data exploration
curl http://localhost:8002

# Or start manually
datasette simon_ecosystem.db --host 0.0.0.0 --port 8002
```

#### Prometheus Metrics
```bash
# Health metrics
curl http://localhost:8000/health

# API metrics
curl http://localhost:8000/metrics
```

#### Grafana Dashboards
- Visit http://localhost:3000 (admin/admin123)
- Pre-configured dashboards for LLM usage
- Real-time monitoring

## Architecture

### Core Components

```

   FastAPI Web API     ← REST endpoints

  Simon Tool Registry  ← Tool management

 Simon LLM Ecosystem   ← Core integration

   Provider Layer      ← LLM providers

    Tool Layer         ← SQLite, Docker, etc.

   Storage Layer       ← SQLite, PostgreSQL

```

### Data Flow

1. **Request** → FastAPI endpoint
2. **Validation** → Pydantic models
3. **Routing** → Simon ecosystem
4. **Processing** → LLM providers/tools
5. **Storage** → Database logging
6. **Response** → JSON/Stream

### Tool Architecture

```python
# Base tool interface
class BaseTool:
    async def execute(self, operation: str, **kwargs) -> Any
    async def health_check(self) -> bool

# Specialized tools
SQLiteTool, PythonTool, DockerTool, etc.

# Registry management
SimonToolRegistry.register(tool)
```

## Configuration

### Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...

# Database Configuration
LLM_DATABASE_PATH=/app/data/simon_ecosystem.db
POSTGRES_PASSWORD=secure_password

# Security Settings
ENABLE_COMMAND_TOOL=false  # Disable system commands
JWT_SECRET_KEY=your_secret_key

# Performance Settings
MAX_WORKERS=4
TIMEOUT=120

# Monitoring
LOG_LEVEL=INFO
GRAFANA_PASSWORD=admin123
```

### Advanced Configuration

```python
config = SimonEcosystemConfig(
    # API Configuration
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    
    # Database Settings
    database_path="production.db",
    log_conversations=True,
    
    # Default Models
    default_chat_model="gpt-4o-mini",
    default_embedding_model="text-embedding-3-small",
    
    # Tool Configuration
    enabled_tools=[
        LLMTool.SQLITE,
        LLMTool.PYTHON,
        LLMTool.QUICKJS,
        # LLMTool.DOCKER,  # Enable if needed
        # LLMTool.COMMAND, # Security risk
    ],
    
    # Performance Settings
    docker_timeout=60,
    datasette_port=8001,
)
```

## Production Deployment

### Docker Compose Stack

```yaml
services:
  simon-ecosystem:      # Main API
  datasette:           # Data exploration
  redis:               # Caching
  postgres:            # Analytics storage
  nginx:               # Load balancer
  prometheus:          # Metrics
  grafana:             # Dashboards
  ollama:              # Local models
  jupyter:             # Notebooks
```

### Scaling Configuration

```bash
# Horizontal scaling
docker-compose -f docker-compose.simon.yml up --scale simon-ecosystem=3

# Load balancing
# Nginx automatically distributes load

# Database sharding
# PostgreSQL with read replicas
```

### Security Best Practices

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Tool Permissions**
   - Disable `COMMAND` tool in production
   - Sandbox Docker execution
   - Validate all inputs

3. **Network Security**
   - Use HTTPS in production
   - Configure firewalls
   - VPN for internal services

4. **Data Protection**
   - Encrypt sensitive data
   - Regular backups
   - Access logging

## Performance Optimization

### Caching Strategy

```python
# Redis caching for frequent queries
# Automatic cache invalidation
# Conversation result caching
```

### Database Optimization

```sql
-- Optimized indexes
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX idx_conversations_model ON conversations(model);

-- Partition large tables
-- Regular VACUUM operations
```

### Model Selection

```python
# Cost-effective defaults
default_chat_model="gpt-4o-mini"        # Fast, cheap
default_embedding_model="text-embedding-3-small"  # Efficient

# High-quality options
premium_chat_model="gpt-4o"             # Best quality
premium_embedding_model="text-embedding-3-large"  # Best embeddings
```

## Testing

### Unit Tests
```bash
# Run Simon ecosystem tests
python -m pytest tests/test_simon_ecosystem_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Test with actual API keys (optional)
OPENAI_API_KEY=sk-... python -m pytest tests/test_simon_ecosystem_integration.py::TestSimonEcosystemIntegration -v
```

### Load Testing
```bash
# API load testing
ab -n 1000 -c 10 http://localhost:8000/health

# Database performance
python -m pytest tests/test_simon_ecosystem_integration.py::TestPerformance -v
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Check environment variables
echo $OPENAI_API_KEY

# Verify in logs
docker-compose -f docker-compose.simon.yml logs simon-ecosystem
```

**2. Database Connection Issues**
```bash
# Check database file permissions
ls -la data/simon_ecosystem.db

# Test SQLite connection
sqlite3 data/simon_ecosystem.db ".tables"
```

**3. Docker Issues**
```bash
# Check container status
docker-compose -f docker-compose.simon.yml ps

# View logs
docker-compose -f docker-compose.simon.yml logs -f simon-ecosystem
```

**4. Memory Issues**
```bash
# Monitor resource usage
docker stats

# Adjust memory limits in docker-compose.yml
```

### Debug Mode

```python
# Enable debug logging
config = SimonEcosystemConfig(log_level="DEBUG")

# Detailed error reporting
ecosystem = SimonLLMEcosystem(config)
health = await ecosystem.health_check()
print(json.dumps(health, indent=2))
```

### Performance Monitoring

```bash
# API response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Database query performance
EXPLAIN QUERY PLAN SELECT * FROM conversations WHERE model = 'gpt-4o-mini';
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd llamaagent

# Setup development environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run in development mode
./scripts/simon_ecosystem_deploy.sh development
```

### Code Style

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

### Adding New Tools

```python
class CustomTool(BaseTool):
    async def execute(self, operation: str, **kwargs) -> Any:
        # Implement tool logic
        pass
    
    async def health_check(self) -> bool:
        # Implement health check
        return True

# Register with ecosystem
registry.register(CustomTool())
```

## API Reference

### Complete API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/providers` | List LLM providers |
| GET | `/tools` | List available tools |
| POST | `/chat` | Chat completion |
| POST | `/chat/stream` | Streaming chat |
| POST | `/embeddings` | Generate embeddings |
| POST | `/database/query` | Execute SQL |
| POST | `/execute/python` | Run Python code |
| POST | `/execute/javascript` | Run JavaScript |
| POST | `/search/conversations` | Search history |
| GET | `/analytics/stats` | Usage statistics |
| POST | `/export/data` | Export data |

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **Simon Willison** for the incredible LLM ecosystem
- **OpenAI** for GPT models and APIs
- **Anthropic** for Claude models
- **Google** for Gemini models
- **The open-source community** for tools and libraries

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [Full Documentation](docs/)
- **Examples**: [Jupyter Notebooks](notebooks/)

---

**Simon Willison's LLM Ecosystem Integration - The most comprehensive LLM toolkit available!** 