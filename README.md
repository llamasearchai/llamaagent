<div align="center">

![LlamaAgent Logo](logo.png)

# LlamaAgent

**Advanced AI Agent Framework for Production-Ready Applications**

[![CI/CD](https://github.com/llamasearchai/llamaagent/workflows/CI/CD/badge.svg)](https://github.com/llamasearchai/llamaagent/actions)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/llamasearchai/llamaagent)

*Empowering developers to build intelligent, scalable AI agents with enterprise-grade reliability*

</div>

---

## Overview

LlamaAgent is a comprehensive AI agent framework designed for production environments. It provides a robust foundation for building intelligent agents that can reason, use tools, maintain memory, and integrate seamlessly with modern AI providers.

### Key Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, Cohere, Together AI, and more
- **Advanced Reasoning**: ReAct pattern implementation with chain-of-thought capabilities
- **Tool Integration**: Extensible tool system with calculator, Python REPL, and custom tools
- **Memory Management**: Persistent memory with vector storage capabilities
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **FastAPI Integration**: RESTful API endpoints for web applications
- **Docker Support**: Containerized deployment with Kubernetes manifests
- **Comprehensive Testing**: 38+ tests with 100% pass rate

## Quick Start

### Installation

```bash
# Install from PyPI
pip install llamaagent

# Or install from source
git clone https://github.com/llamasearchai/llamaagent.git
cd llamaagent
pip install -e ".[dev]"
```

### Basic Usage

```python
from llamaagent.agents.react import ReactAgent
from llamaagent.agents.base import AgentConfig
from llamaagent.llm.providers.openai_provider import OpenAIProvider
from llamaagent.types import TaskInput

# Configure the agent
config = AgentConfig(
    name="MyAgent",
    description="A helpful AI assistant",
    tools_enabled=True
)

# Initialize LLM provider
provider = OpenAIProvider(
    model_name="gpt-4",
    api_key="your-api-key"
)

# Create the agent
agent = ReactAgent(config=config, llm_provider=provider)

# Execute a task
task = TaskInput(
    id="task-1",
    task="Calculate the square root of 144 and explain the process"
)

result = await agent.arun(task)
print(result.content)
```

## Architecture

LlamaAgent follows a modular architecture designed for scalability and maintainability:

```
├── agents/          # Agent implementations (ReAct, reasoning chains)
├── llm/            # LLM provider integrations
├── tools/          # Tool system and implementations
├── memory/         # Memory management and storage
├── api/            # FastAPI web interfaces
├── monitoring/     # Observability and metrics
├── security/       # Authentication and validation
└── types/          # Core type definitions
```

## Advanced Features

### Tool System

```python
from llamaagent.tools.calculator import CalculatorTool
from llamaagent.tools.python_repl import PythonREPLTool

# Register custom tools
agent.register_tool(CalculatorTool())
agent.register_tool(PythonREPLTool())
```

### Memory Management

```python
from llamaagent.memory.vector_memory import VectorMemory

# Configure persistent memory
memory = VectorMemory(
    embedding_model="text-embedding-3-large",
    storage_path="./agent_memory"
)
agent.set_memory(memory)
```

### FastAPI Integration

```python
from llamaagent.api.main import create_app

# Create web API
app = create_app()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Configuration File

```yaml
# config/default.yaml
agent:
  name: "ProductionAgent"
  max_iterations: 10
  timeout: 300
  
llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

tools:
  enabled: true
  timeout: 30
  
memory:
  enabled: true
  type: "vector"
  max_entries: 10000
```

## Deployment

### Docker

```bash
# Build the image
docker build -t llamaagent:latest .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key llamaagent:latest
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Docker Compose

```bash
# Full stack deployment
docker-compose up -d
```

## API Reference

### Core Endpoints

- `POST /agents/execute` - Execute agent task
- `GET /agents/{agent_id}/status` - Get agent status
- `POST /tools/execute` - Execute tool directly
- `GET /health` - Health check endpoint

### OpenAI Compatible API

```bash
# Chat completions
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
pytest tests/e2e/ -v          # End-to-end tests
```

## Monitoring and Observability

### Metrics

LlamaAgent provides comprehensive metrics for production monitoring:

- Request/response times
- Success/failure rates
- Token usage and costs
- Agent performance metrics
- Tool execution statistics

### Logging

```python
import logging
from llamaagent.monitoring.logging import setup_logging

# Configure structured logging
setup_logging(level=logging.INFO, format="json")
```

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Detailed diagnostics
curl http://localhost:8000/diagnostics
```

## Security

### Authentication

```python
from llamaagent.security.authentication import APIKeyAuth

# Configure API key authentication
auth = APIKeyAuth(api_keys=["your-secret-key"])
app.add_middleware(auth)
```

### Input Validation

```python
from llamaagent.security.validator import InputValidator

# Validate and sanitize inputs
validator = InputValidator()
safe_input = validator.sanitize(user_input)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaagent.git
cd llamaagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## Examples

### Basic Agent

```python
# examples/basic_agent.py
import asyncio
from llamaagent.agents.react import ReactAgent
from llamaagent.agents.base import AgentConfig
from llamaagent.llm.providers.mock_provider import MockProvider
from llamaagent.types import TaskInput

async def main():
    config = AgentConfig(name="BasicAgent")
    provider = MockProvider(model_name="test-model")
    agent = ReactAgent(config=config, llm_provider=provider)
    
    task = TaskInput(
        id="example-1",
        task="Explain quantum computing in simple terms"
    )
    
    result = await agent.arun(task)
    print(f"Agent Response: {result.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent System

```python
# examples/multi_agent.py
import asyncio
from llamaagent.spawning.agent_spawner import AgentSpawner
from llamaagent.orchestration.adaptive_orchestra import AdaptiveOrchestra

async def main():
    spawner = AgentSpawner()
    orchestra = AdaptiveOrchestra()
    
    # Spawn multiple specialized agents
    research_agent = await spawner.spawn_agent("researcher")
    analysis_agent = await spawner.spawn_agent("analyst")
    writer_agent = await spawner.spawn_agent("writer")
    
    # Orchestrate collaborative task
    result = await orchestra.execute_collaborative_task(
        task="Write a comprehensive report on AI safety",
        agents=[research_agent, analysis_agent, writer_agent]
    )
    
    print(f"Collaborative Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Benchmarks

LlamaAgent includes comprehensive benchmarking against industry standards:

- **GAIA Benchmark**: General AI Assistant evaluation
- **SPRE Evaluation**: Structured Problem Reasoning
- **Custom Benchmarks**: Domain-specific performance testing

```bash
# Run benchmarks
python -m llamaagent.benchmarks.run_all --provider openai --model gpt-4
```

## Roadmap

- [ ] Multi-modal agent support (vision, audio)
- [ ] Advanced reasoning patterns (Tree of Thoughts, Graph of Thoughts)
- [ ] Federated learning capabilities
- [ ] Enhanced security features
- [ ] Performance optimizations
- [ ] Extended tool ecosystem

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://llamaagent.readthedocs.io](https://llamaagent.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/llamasearchai/llamaagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/llamasearchai/llamaagent/discussions)
- **Email**: [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)

## Acknowledgments

Built with love by [Nik Jois](https://github.com/nikjois) and the LlamaSearch AI team.

Special thanks to the open-source community and all contributors who make this project possible.

---

<div align="center">
  <strong>LlamaAgent - Empowering the Future of AI Agents</strong>
</div>