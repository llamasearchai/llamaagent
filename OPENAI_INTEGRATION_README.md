# LlamaAgent + OpenAI Agents SDK Integration

**Complete integration between LlamaAgent and OpenAI's Agents SDK with budget tracking, hybrid execution, and production-ready features.**

**Author:** Nik Jois  
**Email:** nikjois@llamasearch.ai  
**Budget:** $100 for complete system development and testing

---

## LAUNCH: Overview

This project provides a complete, production-ready integration between our LlamaAgent framework and OpenAI's Agents SDK. The integration enables:

- **Seamless interoperability** between LlamaAgent and OpenAI Agents
- **Budget tracking and cost management** for API usage
- **Hybrid execution** supporting both OpenAI models (GPT-4o-mini) and local models (Llama 3.2)
- **Complete CLI and REST API** interfaces
- **Automated testing and benchmarking**
- **Docker containerization** for deployment

##  Architecture

```

                    LlamaAgent System                        

       
     CLI Interface     FastAPI Server     Integration  
                                           Manager     
       

       
   OpenAI Agents      Budget Tracker     Agent         
   SDK Integration                       Registry      
       

       
   OpenAI Provider    Ollama Provider    Tool          
   (GPT-4o-mini)      (Llama 3.2)        Registry      
       

```

## Package Installation

### Prerequisites

- Python 3.11+
- OpenAI API key (for GPT-4o-mini access)
- Ollama (optional, for local models)
- Docker (optional, for containerized deployment)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd llamaagent

# Install dependencies
pip install -r requirements.txt

# Install OpenAI Agents SDK (optional but recommended)
pip install agents

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the simple demo
python simple_openai_demo.py
```

### Full Installation

```bash
# Install all dependencies including development tools
pip install -r requirements.txt
pip install -e .

# Install optional dependencies
pip install agents tiktoken rich click uvicorn fastapi

# Set up Ollama (for local models)
# Install Ollama from https://ollama.ai
ollama pull llama3.2:3b

# Run comprehensive tests
python -m pytest tests/ -v
```

## Target Quick Demo

### Simple Command Line Usage

```bash
# Configure the system
python -m llamaagent.cli.openai_cli configure \
    --openai-key "your-api-key" \
    --model "gpt-4o-mini" \
    --budget 100.0

# Run a simple task
python -m llamaagent.cli.openai_cli run \
    "Explain artificial intelligence in one sentence" \
    --model gpt-4o-mini \
    --budget 1.0

# Check system status
python -m llamaagent.cli.openai_cli status
```

### FastAPI Server

```bash
# Start the server
python -m llamaagent.api.openai_fastapi

# The API will be available at:
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Python API Usage

```python
import asyncio
from llamaagent.integration.openai_agents import create_openai_integration
from llamaagent.agents.react import ReactAgent
from llamaagent.llm.providers.openai_provider import OpenAIProvider
from llamaagent.types import TaskInput

async def main():
    # Create OpenAI integration
    integration = create_openai_integration(
        openai_api_key="your-api-key",
        model_name="gpt-4o-mini",
        budget_limit=10.0
    )
    
    # Create agent
    llm_provider = OpenAIProvider(model_name="gpt-4o-mini")
    agent = ReactAgent(name="MyAgent", llm_provider=llm_provider)
    
    # Register agent
    adapter = integration.register_agent(agent)
    
    # Execute task
    task = TaskInput(
        id="test_task",
        task="What is machine learning?"
    )
    
    result = await adapter.run_task(task)
    print(f"Result: {result.result.data['response']}")
    
    # Check budget
    budget_status = integration.get_budget_status()
    print(f"Cost: ${budget_status['current_cost']:.4f}")

asyncio.run(main())
```

## Tools Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Custom OpenAI endpoint
export OLLAMA_BASE_URL="http://localhost:11434"     # Ollama endpoint
export LLAMAAGENT_BUDGET="100.0"                    # Default budget
export LLAMAAGENT_MODEL="gpt-4o-mini"               # Default model
```

### Configuration File

Create `~/.llamaagent/config.json`:

```json
{
  "openai_api_key": "your-api-key",
  "openai_base_url": null,
  "default_model": "gpt-4o-mini",
  "default_budget": 100.0,
  "ollama_base_url": "http://localhost:11434",
  "default_local_model": "llama3.2:3b",
  "enable_tracing": true,
  "verbose": false
}
```

##  Budget Management

The system includes comprehensive budget tracking:

### Features

- **Real-time cost tracking** for OpenAI API calls
- **Budget limits** with automatic enforcement
- **Cost estimation** before task execution
- **Usage analytics** and reporting
- **Multi-model cost tracking** (OpenAI vs local)

### Usage

```python
# Set budget limit
integration = create_openai_integration(budget_limit=50.0)

# Check budget status
budget_status = integration.get_budget_status()
print(f"Remaining budget: ${budget_status['remaining_budget']}")

# Budget will automatically prevent overspending
```

### Cost Estimates (as of 2025)

| Model | Input Cost | Output Cost | Typical Task Cost |
|-------|------------|-------------|-------------------|
| GPT-4o-mini | $0.15/1M tokens | $0.60/1M tokens | $0.001-0.01 |
| GPT-4o | $2.50/1M tokens | $10.00/1M tokens | $0.01-0.10 |
| Llama 3.2 (local) | Free | Free | $0.00 |

## BUILD: Features

### Core Capabilities

- PASS **OpenAI Agents SDK Integration** - Full compatibility and interoperability
- PASS **Budget Tracking** - Real-time cost monitoring and limits
- PASS **Hybrid Execution** - Switch between OpenAI and local models
- PASS **Tool Integration** - Calculator, Python REPL, and custom tools
- PASS **CLI Interface** - Complete command-line interface
- PASS **REST API** - FastAPI server with Swagger documentation
- PASS **Batch Processing** - Execute multiple tasks efficiently
- PASS **Experiment Management** - Automated testing and benchmarking
- PASS **Docker Support** - Containerized deployment
- PASS **Comprehensive Testing** - Unit tests and integration tests

### Agent Types

1. **ReactAgent** - Reasoning and acting agent with tool usage
2. **BaseAgent** - Simple agent for basic tasks
3. **OpenAI Native** - Direct OpenAI Agents SDK usage
4. **Hybrid Agent** - Automatic fallback between providers

### Supported Models

#### OpenAI Models
- GPT-4o-mini (recommended for cost efficiency)
- GPT-4o
- GPT-4
- GPT-3.5-turbo

#### Local Models (via Ollama)
- Llama 3.2 3B
- Llama 3.2 1B
- Other Ollama-supported models

## Results Usage Examples

### 1. Simple Task Execution

```bash
# Execute a simple task
python -m llamaagent.cli.openai_cli run "What is AI?" --budget 1.0
```

### 2. Tool Usage

```bash
# Use calculator tool
python -m llamaagent.cli.openai_cli run \
    "Calculate the square root of 144" \
    --tools calculator \
    --budget 2.0

# Use Python tool
python -m llamaagent.cli.openai_cli run \
    "Write Python code to sort a list" \
    --tools python \
    --budget 3.0
```

### 3. Batch Processing

```bash
# Create experiment configuration
cat > experiment.json << EOF
{
  "tasks": [
    {"task": "What is machine learning?"},
    {"task": "Explain neural networks"},
    {"task": "Calculate 15 * 23"}
  ],
  "models": ["gpt-4o-mini", "llama3.2:3b"],
  "budget_per_task": 1.0
}
EOF

# Run experiment
python -m llamaagent.cli.openai_cli experiment experiment.json
```

### 4. API Usage

```bash
# Start server
python -m llamaagent.api.openai_fastapi &

# Create agent
curl -X POST "http://localhost:8000/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-agent",
    "model_name": "gpt-4o-mini",
    "budget_limit": 10.0,
    "tools": ["calculator"]
  }'

# Execute task
curl -X POST "http://localhost:8000/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is 2+2?",
    "agent_name": "test-agent"
  }'
```

##  Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t llamaagent .

# Run with environment variables
docker run -e OPENAI_API_KEY="your-key" -p 8000:8000 llamaagent

# Run with docker-compose
docker-compose up -d
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
      - LLAMAAGENT_BUDGET=100.0
    volumes:
      - ./data:/app/data
```

## Testing Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_openai_integration.py -v
python -m pytest tests/test_budget_tracking.py -v

# Run with coverage
python -m pytest tests/ --cov=llamaagent --cov-report=html
```

### Benchmark Tests

```bash
# Run performance benchmarks
python -m pytest tests/test_benchmarks.py -v

# Run GAIA benchmark
python -m llamaagent.benchmarks.gaia_benchmark
```

## Performance Performance

### Benchmarks

| Metric | GPT-4o-mini | Llama 3.2 3B | Notes |
|--------|-------------|---------------|-------|
| Response Time | ~2-5s | ~3-8s | Varies by task complexity |
| Cost per Task | ~$0.001-0.01 | $0.00 | Local models are free |
| Accuracy | High | Good | GPT-4o-mini generally more accurate |
| Throughput | 10-20 req/min | 5-15 req/min | Depends on hardware |

### Optimization Tips

1. **Use GPT-4o-mini** for cost efficiency
2. **Enable local models** for free inference
3. **Set appropriate budgets** to control costs
4. **Use batch processing** for multiple tasks
5. **Cache responses** when possible

## Security Security

### Best Practices

- Store API keys securely (environment variables or secrets management)
- Use budget limits to prevent unexpected costs
- Validate all inputs before processing
- Use HTTPS in production
- Implement rate limiting
- Monitor usage and costs regularly

### Configuration

```python
# Secure configuration example
integration = create_openai_integration(
    openai_api_key=os.getenv("OPENAI_API_KEY"),  # From environment
    budget_limit=float(os.getenv("BUDGET_LIMIT", "100.0")),
    enable_tracing=False,  # Disable in production if needed
    timeout=30.0  # Reasonable timeout
)
```

## LAUNCH: Production Deployment

### Kubernetes

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

### Monitoring

```bash
# Health check endpoint
curl http://localhost:8000/health

# System information
curl http://localhost:8000/system/info

# Budget status
curl http://localhost:8000/budget/agent-name
```

## Documentation API Reference

### CLI Commands

```bash
# Configuration
llamaagent configure --help
llamaagent status

# Task execution
llamaagent run "task" --model gpt-4o-mini --budget 5.0
llamaagent test --model gpt-4o-mini

# Experiments
llamaagent experiment config.json --budget-per-task 1.0
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information |
| `/health` | GET | Health check |
| `/agents` | POST | Create agent |
| `/agents` | GET | List agents |
| `/agents/{name}` | GET | Get agent details |
| `/tasks` | POST | Execute task |
| `/tasks/{id}` | GET | Get task status |
| `/budget/{agent}` | GET | Get budget status |
| `/experiments` | POST | Run experiment |

### Python API

```python
# Core classes
from llamaagent.integration.openai_agents import (
    create_openai_integration,
    OpenAIAgentsIntegration,
    OpenAIAgentAdapter,
    BudgetTracker
)

# Agent types
from llamaagent.agents.react import ReactAgent
from llamaagent.agents.base import BaseAgent

# Providers
from llamaagent.llm.providers.openai_provider import OpenAIProvider
from llamaagent.llm.providers.ollama_provider import OllamaProvider

# Types
from llamaagent.types import TaskInput, TaskOutput, TaskStatus
```

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd llamaagent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8
- Use type hints
- Write comprehensive docstrings
- Add tests for new features
- Update documentation

##  License

This project is licensed under the MIT License. See LICENSE file for details.

##  Support

For questions, issues, or support:

- **Email:** nikjois@llamasearch.ai
- **GitHub Issues:** Create an issue in the repository
- **Documentation:** Check this README and inline documentation

## Target Roadmap

### Completed PASS
- OpenAI Agents SDK integration
- Budget tracking and management
- CLI interface
- REST API
- Docker support
- Comprehensive testing
- Documentation

### Planned 
- Web UI dashboard
- Advanced analytics
- More model providers
- Enhanced security features
- Performance optimizations
- Cloud deployment guides

---

**Built with LOVE: by Nik Jois for the LlamaAgent project**

*This integration enables seamless use of both OpenAI's powerful models and local alternatives, with complete budget control and production-ready features.* 