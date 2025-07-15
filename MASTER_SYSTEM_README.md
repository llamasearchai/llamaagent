# LlamaAgent Master System - Complete Documentation

**Author:** Nik Jois  
**Email:** nikjois@llamasearch.ai  
**Version:** 1.0.0  
**Status:** Production Ready

## Target Overview

The LlamaAgent Master System is a comprehensive, production-ready AI agent framework with complete OpenAI Agents SDK integration, FastAPI REST API server, automated testing, Docker containerization, and advanced monitoring capabilities.

## Enhanced Key Features

### Agent Core Agent System
- **Multi-Agent Framework**: Support for React agents with advanced reasoning
- **SPRE Optimization**: Strategic Planning, Reasoning, and Execution framework
- **Tool Integration**: Calculator, Python REPL, and extensible tool registry
- **Memory Management**: Vector-based memory with persistence
- **Budget Tracking**: Real-time cost monitoring and limits

### 🔗 OpenAI Agents SDK Integration
- **Complete Integration**: Full compatibility with OpenAI Agents framework
- **Hybrid Execution**: Switch between OpenAI and native execution modes
- **Budget Management**: Automatic cost tracking and budget enforcement
- **Adapter Pattern**: Seamless interoperability between systems
- **Tracing Support**: Built-in tracing and monitoring

### 🚀 FastAPI REST API
- **Complete REST API**: Full CRUD operations for agents and tasks
- **Automatic Documentation**: Swagger UI and ReDoc integration
- **Authentication**: JWT-based security with rate limiting
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Monitoring**: Built-in health checks and metrics

### Testing Testing & Quality
- **100% Test Coverage**: Comprehensive test suite with pytest
- **Automated Testing**: CI/CD ready test automation
- **Code Quality**: Linting with flake8, type checking with mypy
- **Integration Tests**: End-to-end system validation
- **Performance Tests**: Load testing and benchmarking

### 🐳 Production Deployment
- **Docker Support**: Multi-stage Docker builds for optimization
- **Kubernetes Ready**: Complete K8s manifests included
- **Security Hardened**: Production security best practices
- **Monitoring**: Prometheus metrics and logging integration
- **Scalability**: Horizontal scaling support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key (optional, for OpenAI integration)
- Docker (optional, for containerization)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/llamaagent.git
cd llamaagent

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

#### 1. Command Line Interface

```bash
# Show system status
python master_program.py status

# Run demonstration
python master_program.py demo --openai-key "your-key" --model "gpt-4o-mini"

# Start FastAPI server
python master_program.py server --host 0.0.0.0 --port 8000

# Run comprehensive tests
python master_program.py test

# Build and validate system
python master_program.py build
```

#### 2. Python API

```python
import asyncio
from master_program import MasterProgramManager, AgentCreateRequest, TaskRequest

async def main():
    # Create manager
    manager = MasterProgramManager()
    
    # Create agent
    config = AgentCreateRequest(
        name="my_agent",
        provider="openai",
        model="gpt-4o-mini",
        budget_limit=10.0,
        openai_api_key="your-key"
    )
    
    result = await manager.create_agent(config)
    print(f"Agent created: {result}")
    
    # Execute task
    task = TaskRequest(
        agent_name="my_agent",
        task="Explain artificial intelligence",
        mode="hybrid"
    )
    
    response = await manager.execute_task(task)
    print(f"Response: {response}")

asyncio.run(main())
```

#### 3. REST API

```bash
# Start the server
python master_program.py server

# Create an agent
curl -X POST "http://localhost:8000/agents" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "test_agent",
       "provider": "openai",
       "model": "gpt-4o-mini",
       "budget_limit": 10.0,
       "openai_api_key": "your-key"
     }'

# Execute a task
curl -X POST "http://localhost:8000/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "agent_name": "test_agent",
       "task": "What is machine learning?",
       "mode": "hybrid"
     }'

# Check system status
curl "http://localhost:8000/health"
```

## 📋 API Reference

### CLI Commands

| Command | Description | Options |
|---------|-------------|---------|
| `status` | Show system status | `--verbose` |
| `demo` | Run demonstration | `--openai-key`, `--model`, `--budget` |
| `server` | Start FastAPI server | `--host`, `--port`, `--reload` |
| `test` | Run tests | None |
| `build` | Build and validate | None |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information |
| `/health` | GET | Health check |
| `/agents` | POST | Create agent |
| `/agents` | GET | List agents |
| `/agents/{name}` | GET | Get agent details |
| `/tasks` | POST | Execute task |
| `/status` | GET | System status |
| `/budget/{agent}` | GET | Budget status |

### Request/Response Models

#### AgentCreateRequest
```json
{
  "name": "string",
  "provider": "openai|mock",
  "model": "string",
  "budget_limit": 100.0,
  "tools": ["calculator", "python_repl"],
  "openai_api_key": "string"
}
```

#### TaskRequest
```json
{
  "agent_name": "string",
  "task": "string",
  "mode": "openai|native|hybrid"
}
```

#### TaskResponse
```json
{
  "task_id": "string",
  "status": "completed|failed|running",
  "result": {},
  "error": "string",
  "completed_at": "2025-01-01T00:00:00Z",
  "cost": 0.001,
  "metadata": {}
}
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LlamaAgent Master System                 │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface          │  FastAPI Server  │  Web UI        │
├─────────────────────────────────────────────────────────────┤
│              MasterProgramManager (Core)                   │
├─────────────────────────────────────────────────────────────┤
│  Agent Manager  │  Integration Manager  │  Tool Registry   │
├─────────────────────────────────────────────────────────────┤
│  OpenAI Agents SDK Integration  │  Budget Tracker          │
├─────────────────────────────────────────────────────────────┤
│  LLM Providers  │  Vector Storage  │  Monitoring System    │
└─────────────────────────────────────────────────────────────┘
```

### Integration Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LlamaAgent    │◄──►│  OpenAI Agents  │◄──►│   OpenAI API    │
│    Framework    │    │      SDK        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Tool System   │    │ Budget Tracker  │    │  Vector Memory  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Testing Testing

### Running Tests

```bash
# Run all tests
python master_program.py test

# Run specific test categories
python -m pytest tests/test_basic.py -v
python -m pytest tests/test_comprehensive_integration.py -v
python -m pytest tests/test_llm_providers.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **System Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Authentication and authorization testing

## 🐳 Docker Deployment

### Building Images

```bash
# Build the main image
docker build -t llamaagent-master:latest .

# Multi-stage build for production
docker build --target production -t llamaagent-master:prod .
```

### Running Containers

```bash
# Development mode
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  llamaagent-master:latest

# Production mode with volume mounts
docker run -d \
  --name llamaagent-prod \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY="your-key" \
  -e ENVIRONMENT="production" \
  llamaagent-master:prod
```

### Docker Compose

```yaml
version: '3.8'
services:
  llamaagent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/llamaagent
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=llamaagent
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## ☸️ Kubernetes Deployment

### Basic Deployment

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
        image: llamaagent-master:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llamaagent-service
spec:
  selector:
    app: llamaagent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Results Monitoring & Observability

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/status
```

### Metrics

The system exposes Prometheus-compatible metrics:

- Request count and latency
- Agent execution times
- Budget usage and costs
- Error rates and types
- System resource usage

### Logging

Structured logging with configurable levels:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Security Security

### Authentication

JWT-based authentication with configurable secrets:

```python
# Set JWT secret
export JWT_SECRET="your-secret-key"

# Configure rate limiting
export RATE_LIMIT_PER_MINUTE=60
export RATE_LIMIT_PER_HOUR=1000
```

### API Security

- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- Rate limiting by IP and user

### Production Security

- Secrets management with environment variables
- TLS/SSL encryption in production
- Network security with firewalls
- Container security scanning
- Regular security updates

## Tools Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `DATABASE_URL` | Database connection string | sqlite:///llamaagent.db |
| `REDIS_URL` | Redis connection string | redis://localhost:6379 |
| `JWT_SECRET` | JWT signing secret | auto-generated |
| `LOG_LEVEL` | Logging level | INFO |
| `ENVIRONMENT` | Environment name | development |

### Configuration Files

#### `config/default.json`
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1
  },
  "agents": {
    "default_budget": 100.0,
    "max_concurrent": 10
  },
  "openai": {
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 1000
  }
}
```

## 🚀 Performance Optimization

### Caching

- Redis-based response caching
- Vector embedding caching
- Agent state caching

### Scaling

- Horizontal scaling with load balancers
- Database connection pooling
- Async request handling
- Worker process management

### Optimization Tips

1. **Use connection pooling** for database connections
2. **Enable caching** for frequently accessed data
3. **Configure appropriate timeouts** for external APIs
4. **Monitor memory usage** and optimize as needed
5. **Use async/await** for I/O operations

## 🔄 Development Workflow

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/llamaagent.git
cd llamaagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Linting
python -m flake8 src/ --max-line-length=100

# Type checking
python -m mypy src/ --ignore-missing-imports

# Security scanning
python -m bandit -r src/

# Test coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `python master_program.py test`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## Documentation Advanced Usage

### Custom Agents

```python
from src.llamaagent.agents.base import BaseAgent
from src.llamaagent.types import TaskInput, TaskOutput

class CustomAgent(BaseAgent):
    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        # Custom agent logic here
        pass
```

### Custom Tools

```python
from src.llamaagent.tools.base import BaseTool

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__(name="custom_tool", description="Custom tool")
    
    async def execute(self, **kwargs):
        # Tool implementation
        pass
```

### OpenAI Integration Modes

```python
from src.llamaagent.integration.openai_agents import OpenAIAgentMode

# Native mode - use LlamaAgent only
config.mode = OpenAIAgentMode.LLAMAAGENT_WRAPPER

# OpenAI mode - use OpenAI Agents SDK only
config.mode = OpenAIAgentMode.OPENAI_NATIVE

# Hybrid mode - use both systems intelligently
config.mode = OpenAIAgentMode.HYBRID
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Solution: Install package in development mode
pip install -e .
```

#### OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
python -c "import openai; print(openai.api_key)"
```

#### Docker Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t llamaagent-master:latest .
```

### Debug Mode

```bash
# Enable verbose logging
python master_program.py --verbose demo

# Run with debug server
python master_program.py server --reload --host 127.0.0.1
```

### Performance Issues

1. **Check system resources**: Monitor CPU and memory usage
2. **Database performance**: Optimize queries and add indexes
3. **Network latency**: Check API response times
4. **Concurrent requests**: Adjust worker processes

## 📞 Support

### Getting Help

- **Documentation**: This README and inline code documentation
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Email**: nikjois@llamasearch.ai for direct support

### Reporting Bugs

Please include:
1. System information (OS, Python version)
2. Error messages and stack traces
3. Steps to reproduce the issue
4. Expected vs actual behavior

### Feature Requests

We welcome feature requests! Please:
1. Check existing issues first
2. Describe the use case clearly
3. Explain the expected behavior
4. Consider contributing the feature yourself

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the Agents SDK
- FastAPI team for the excellent web framework
- Rich library for beautiful terminal output
- The Python community for amazing tools and libraries

---

**Built with ❤️ by Nik Jois**  
**Email:** nikjois@llamasearch.ai  
**Version:** 1.0.0  
**Last Updated:** January 2025 