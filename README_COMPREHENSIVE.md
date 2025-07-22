# LlamaAgent - Complete Agentic AI System

A comprehensive, production-ready agentic AI framework with advanced reasoning, tool integration, and OpenAI Agents SDK support.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities
- **Multi-Agent Orchestration**: Advanced agent spawning and coordination
- **SPRE Data Generation**: Sophisticated Prompting for Reasoning Enhancement
- **OpenAI Agents SDK Integration**: Complete integration with OpenAI's latest agents
- **Advanced Reasoning**: Chain-of-thought and multi-step reasoning
- **Tool Integration**: Comprehensive tool ecosystem with extensible framework
- **Real-time Monitoring**: Performance metrics and health monitoring
- **Production-Ready**: Docker, Kubernetes, and cloud deployment support

### Technical Features
- **FastAPI Endpoints**: RESTful API with WebSocket support
- **Database Integration**: PostgreSQL, Redis, and vector databases
- **Security**: Authentication, authorization, and audit logging
- **Scalability**: Horizontal scaling with load balancing
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Caching**: Multi-level caching for optimal performance
- **Error Handling**: Comprehensive error recovery and logging

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nikjois/llamaagent.git
cd llamaagent

# Install with pip
pip install -e .

# Or use Docker
docker-compose up -d
```

### 2. Basic Usage

```python
from llamaagent import LlamaAgent, AgentConfig, SPREGenerator

# Create an agent
config = AgentConfig(
    name="research_agent",
    role="researcher",
    llm_provider="openai",
    tools=["web_search", "calculator", "code_executor"]
)

agent = LlamaAgent(config)

# Execute a task
result = await agent.execute("Analyze the latest AI research trends")
print(result)

# Generate SPRE dataset
generator = SPREGenerator()
dataset = await generator.generate_dataset(
    name="research_data",
    count=100,
    data_type="conversation"
)
```

### 3. API Usage

```bash
# Start the API server
python -m llamaagent.api.complete_api

# Create an agent via API
curl -X POST "http://localhost:8000/agents/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "assistant",
    "role": "generalist",
    "llm_provider": "openai"
  }'

# Execute a task
curl -X POST "http://localhost:8000/agents/{agent_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest developments in AI?",
    "stream": false
  }'
```

## Installation

### Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- PostgreSQL (for persistent storage)
- Redis (for caching and session management)

### Installation Methods

#### 1. Development Installation

```bash
# Clone repository
git clone https://github.com/nikjois/llamaagent.git
cd llamaagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

#### 2. Production Installation

```bash
# Install from PyPI (when published)
pip install llamaagent

# Or build from source
python build_comprehensive.py
pip install dist/llamaagent-*.whl
```

#### 3. Docker Installation

```bash
# Quick start with Docker Compose
docker-compose -f docker-compose.complete.yml up -d

# Build custom image
docker build -f Dockerfile.complete -t llamaagent:latest .
```

### Environment Configuration

Create a `.env` file:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_ID=your_organization_id

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/llamaagent
REDIS_URL=redis://localhost:6379/0

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key

# Security
JWT_SECRET_KEY=your_secret_key
ENCRYPT_SECRET_KEY=your_encryption_key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

## Usage

### Agent Creation and Management

```python
from llamaagent import LlamaAgent, AgentConfig, AgentRole

# Create specialized agents
research_agent = LlamaAgent(AgentConfig(
    name="researcher",
    role=AgentRole.RESEARCHER,
    tools=["web_search", "pdf_reader", "citation_generator"],
    llm_provider="openai",
    model="gpt-4o"
))

coding_agent = LlamaAgent(AgentConfig(
    name="coder",
    role=AgentRole.CODER,
    tools=["code_executor", "github_api", "documentation_generator"],
    llm_provider="openai",
    model="gpt-4o"
))

# Multi-agent coordination
from llamaagent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
orchestrator.add_agent(research_agent)
orchestrator.add_agent(coding_agent)

# Execute complex task
result = await orchestrator.execute_complex_task(
    "Research the latest FastAPI features and implement a sample API"
)
```

### SPRE Data Generation

```python
from llamaagent.data_generation import SPREGenerator, DataType

generator = SPREGenerator()

# Generate conversational data
conversation_data = await generator.generate_dataset(
    name="customer_support",
    count=500,
    data_type=DataType.CONVERSATION,
    topic="technical support",
    difficulty="medium"
)

# Generate reasoning data
reasoning_data = await generator.generate_dataset(
    name="math_problems",
    count=200,
    data_type=DataType.REASONING,
    topic="calculus",
    difficulty="advanced"
)

# Generate from custom prompts
custom_data = await generator.generate_from_prompts([
    "Explain quantum computing",
    "Write a Python function for binary search",
    "Analyze market trends in renewable energy"
])
```

### OpenAI Agents Integration

```python
from llamaagent.integration import OpenAIAgentsIntegration, OpenAIIntegrationConfig

# Configure OpenAI integration
config = OpenAIIntegrationConfig(
    api_key="your_openai_key",
    model="gpt-4o",
    budget_limit=100.0
)

# Initialize integration
openai_integration = OpenAIAgentsIntegration(config)
await openai_integration.initialize()

# Create OpenAI agent
agent_id = await openai_integration.create_agent(
    name="analysis_agent",
    instructions="You are an expert data analyst",
    tools=[{"type": "code_interpreter"}]
)

# Execute task
result = await openai_integration.execute_task(
    agent_id, 
    "Analyze this dataset and provide insights"
)
```

### Tool Integration

```python
from llamaagent.tools import ToolRegistry, WebSearchTool, CalculatorTool

# Register custom tool
class CustomTool:
    name = "custom_processor"
    description = "Process custom data formats"
    
    async def execute(self, data: str) -> str:
        # Custom processing logic
        return f"Processed: {data}"

# Use tools in agents
tool_registry = ToolRegistry()
tool_registry.register(CustomTool())

agent = LlamaAgent(AgentConfig(
    name="processor",
    tools=["custom_processor", "web_search", "calculator"]
))
```

## API Documentation

### FastAPI Endpoints

The complete API provides comprehensive endpoints for all functionality:

#### Agent Management
- `POST /agents/create` - Create new agent
- `GET /agents` - List all agents
- `GET /agents/{id}` - Get agent details
- `POST /agents/{id}/execute` - Execute agent task
- `DELETE /agents/{id}` - Delete agent

#### SPRE Data Generation
- `POST /spre/generate` - Generate SPRE dataset
- `POST /spre/generate-from-prompts` - Generate from custom prompts
- `GET /spre/generators` - List active generators
- `GET /spre/generators/{id}/stats` - Get generation statistics

#### File Operations
- `POST /files/upload` - Upload files for processing
- `GET /files/{id}` - Download files
- `POST /files/{id}/process` - Process uploaded files

#### Real-time Communication
- `WebSocket /ws/agent/{id}` - Agent communication
- `WebSocket /ws/metrics` - Real-time metrics

#### Monitoring and Health
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /status` - System status

### API Authentication

The API supports JWT-based authentication:

```python
import jwt
import requests

# Get token (implement your auth endpoint)
token = get_auth_token()

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/agents/create",
    headers=headers,
    json={"name": "test_agent"}
)
```

## Docker Deployment

### Complete Production Setup

```bash
# Deploy full stack with monitoring
docker-compose -f docker-compose.complete.yml up -d

# Services included:
# - LlamaAgent API
# - PostgreSQL database
# - Redis cache
# - Qdrant vector database
# - Nginx reverse proxy
# - Prometheus monitoring
# - Grafana dashboards
# - Log aggregation (Loki)
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -k k8s/overlays/production/

# Scale deployment
kubectl scale deployment llamaagent --replicas=5

# Check status
kubectl get pods -l app=llamaagent
```

### Configuration

#### docker-compose.complete.yml
```yaml
version: '3.8'
services:
  llamaagent:
    build:
      context: .
      dockerfile: Dockerfile.complete
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/llamaagent
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: llamaagent
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests
pytest tests/performance/   # Performance tests

# Run comprehensive tests
pytest tests/test_comprehensive_functionality.py -v
```

### Test Configuration

```python
# conftest.py
import pytest
from llamaagent import LlamaAgent
from llamaagent.testing import MockLLMProvider

@pytest.fixture
def mock_agent():
    return LlamaAgent(AgentConfig(
        name="test_agent",
        llm_provider="mock"
    ))

@pytest.fixture
def test_database():
    # Setup test database
    pass
```

### Performance Testing

```bash
# Run benchmarks
python -m pytest tests/performance/ --benchmark-only

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Development

### Development Setup

```bash
# Setup development environment
python build_comprehensive.py

# Install pre-commit hooks
pre-commit install

# Run development server
python -m llamaagent.api.complete_api --reload
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
flake8 src tests
mypy src

# Security scan
bandit -r src
safety check
```

### Building and Packaging

```bash
# Run complete build pipeline
python build_comprehensive.py

# Build specific components
python build_comprehensive.py --skip-tests  # Skip tests
python build_comprehensive.py --skip-docs   # Skip documentation

# Create distribution
python -m build
```

## Architecture

### System Architecture

```
        
   Web UI               API Gateway          Load Balancer 
        
                                                       
         
                                 

                    FastAPI Application                            

   Agent Pool      SPRE Engine    OpenAI Agents    Tool Engine 

                                                       
         
                                         

                    Data Layer                                     

   PostgreSQL         Redis          Qdrant         File Store 

```

### Component Overview

- **Agent Pool**: Manages agent lifecycle and execution
- **SPRE Engine**: Handles data generation and enhancement
- **OpenAI Agents**: Integration with OpenAI's agent platform
- **Tool Engine**: Extensible tool framework
- **Data Layer**: Persistent storage and caching

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation
- Add docstrings to all public functions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Author**: Nik Jois  
**Email**: nikjois@llamasearch.ai  
**GitHub**: [nikjois](https://github.com/nikjois)

## Acknowledgments

- OpenAI for the Agents SDK
- FastAPI for the excellent web framework
- The open-source community for inspiration and tools

---

**LlamaAgent** - Empowering the future of agentic AI systems. 