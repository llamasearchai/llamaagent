# LlamaAgent Framework

A production-ready, highly extensible AI agent framework with multi-provider LLM support, advanced tool integration, and comprehensive evaluation capabilities.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities
- **Multi-Provider LLM Support**: Seamless integration with OpenAI, Anthropic, Cohere, Together, Ollama, and more
- **Advanced Agent Types**: ReAct agents, multimodal agents, reasoning chains
- **Extensible Tool System**: Built-in tools plus easy custom tool creation
- **Memory Management**: Short-term and long-term memory with vector storage
- **Production-Ready**: Comprehensive error handling, logging, and monitoring

### Advanced Features
- **Data Generation**: SPRE and GDT frameworks for synthetic data
- **Benchmarking**: Built-in evaluation with GAIA and custom benchmarks
- **Caching**: Multi-level caching for optimal performance
- **Security**: Rate limiting, input validation, API key management
- **Visualization**: Performance plots and analysis tools
- **API & CLI**: RESTful API and interactive command-line interface

## Quick Start

```python
from llamaagent import ReactAgent
from llamaagent.tools import CalculatorTool

# Create an agent
agent = ReactAgent(
    name="MyAgent",
    model="gpt-4",
    tools=[CalculatorTool()]
)

# Run a query
response = await agent.run("What is 25 * 4?")
print(response.response)  # "100"
```

## Installation

### From PyPI (Recommended)

```bash
pip install llamaagent
```

### From Source

```bash
git clone https://github.com/yourusername/llamaagent.git
cd llamaagent
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For OpenAI support
pip install llamaagent[openai]

# For Anthropic support
pip install llamaagent[anthropic]

# For all providers
pip install llamaagent[all]
```

## Core Concepts

### Agents

Agents are the primary interface for interacting with LLMs. LlamaAgent provides several agent types:

#### ReactAgent
The ReAct (Reasoning and Acting) agent uses a thought-action-observation loop:

```python
from llamaagent import ReactAgent
from llamaagent.llm import OpenAIProvider

agent = ReactAgent(
    name="ResearchAgent",
    provider=OpenAIProvider(api_key="your-key"),
    tools=[WebSearchTool(), CalculatorTool()],
    max_steps=10
)
```

#### Custom Agents
Create your own agent by extending BaseAgent:

```python
from llamaagent.agents import BaseAgent

class CustomAgent(BaseAgent):
    async def run(self, query: str) -> AgentResponse:
        # Your custom logic here
        pass
```

### Tools

Tools extend agent capabilities. Create custom tools easily:

```python
from llamaagent.tools import Tool

@Tool(
    name="weather",
    description="Get current weather for a location"
)
async def get_weather(location: str) -> str:
    # Implementation here
    return f"Sunny in {location}"

# Use with agent
agent = ReactAgent(tools=[get_weather])
```

### Providers

LlamaAgent supports multiple LLM providers with a unified interface:

```python
from llamaagent.llm import LLMFactory

# Create provider
provider = LLMFactory.create_provider(
    "openai",
    api_key="your-key",
    model="gpt-4"
)

# Use with agent
agent = ReactAgent(provider=provider)
```

Supported providers:
- OpenAI (GPT-3.5, GPT-4, etc.)
- Anthropic (Claude 2, Claude 3)
- Cohere (Command, Command-R)
- Together (Open source models)
- Ollama (Local models)
- Mock (For testing)

### Memory

Agents can maintain conversation context and long-term memory:

```python
from llamaagent.memory import VectorMemory

# Create agent with vector memory
memory = VectorMemory(embedding_model="text-embedding-ada-002")
agent = ReactAgent(
    name="MemoryAgent",
    memory=memory,
    enable_memory=True
)

# Memory persists across conversations
response1 = await agent.run("My name is Alice")
response2 = await agent.run("What's my name?")  # Remembers "Alice"
```

## Usage Examples

### Basic Chat

```python
import asyncio
from llamaagent import ReactAgent

async def main():
    agent = ReactAgent(name="ChatBot")

    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break

        response = await agent.run(query)
        print(f"Bot: {response.response}")

asyncio.run(main())
```

### Multi-Tool Agent

```python
from llamaagent import ReactAgent
from llamaagent.tools import CalculatorTool, PythonREPLTool, WebSearchTool

agent = ReactAgent(
    name="AssistantAgent",
    tools=[
        CalculatorTool(),
        PythonREPLTool(sandbox=True),
        WebSearchTool(api_key="your-search-key")
    ],
    verbose=True  # See reasoning process
)

response = await agent.run(
    "Search for the current Bitcoin price, then calculate how much "
    "100 Bitcoin would be worth in USD"
)
```

### Data Generation

```python
from llamaagent.data_generation import SPREGenerator

generator = SPREGenerator()

# Generate reasoning tasks
task = await generator.generate_task(
    domain="mathematics",
    difficulty="medium"
)

print(f"Question: {task['question']}")
print(f"Answer: {task['answer']}")
print(f"Steps: {task['reasoning_steps']}")
```

### Benchmarking

```python
from llamaagent import ReactAgent
from llamaagent.benchmarks import GAIABenchmark

agent = ReactAgent(name="BenchmarkAgent")
benchmark = GAIABenchmark()

results = await benchmark.run_benchmark(
    agent,
    num_tasks=50,
    parallel=True
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Avg time: {results['avg_time']:.2f}s")
```

### API Server

```python
from llamaagent.api import create_app

app = create_app()

# Endpoints available:
# POST /chat - Chat with agent
# GET /agents - List agents
# POST /agents - Create agent
# GET /tools - List available tools

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### CLI Interface

```bash
# Interactive mode
llamaagent chat --model gpt-4

# Single query
llamaagent query "What is the weather today?"

# With tools
llamaagent chat --tools calculator,web_search

# Benchmark
llamaagent benchmark --agent myagent --dataset gaia
```

## API Reference

### Core Classes

#### ReactAgent
```python
class ReactAgent:
    def __init__(
        self,
        name: str,
        provider: Optional[LLMProvider] = None,
        tools: List[BaseTool] = None,
        memory: Optional[BaseMemory] = None,
        max_steps: int = 10,
        verbose: bool = False
    )

    async def run(self, query: str) -> AgentResponse
    async def run_with_context(self, query: str, context: Dict) -> AgentResponse
```

#### AgentResponse
```python
@dataclass
class AgentResponse:
    success: bool
    response: str
    trace: AgentTrace
    metadata: Dict[str, Any]
    error: Optional[str]
```

#### Tool
```python
@Tool(
    name: str,
    description: str,
    parameters: Dict[str, Any]  # JSON Schema
)
async def tool_function(**kwargs) -> Union[str, Dict]
```

### Advanced Features

#### Caching
```python
from llamaagent.cache import cache_result

@cache_result(ttl=3600)
async def expensive_operation(param: str) -> str:
    # Cached for 1 hour
    pass
```

#### Rate Limiting
```python
from llamaagent.security import RateLimiter

limiter = RateLimiter(
    max_requests=100,
    window_seconds=60
)

@limiter.limit
async def api_endpoint():
    pass
```

#### Monitoring
```python
from llamaagent.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track metrics
metrics.record("api_calls", 1)
metrics.record("response_time", 0.5)

# Get report
report = metrics.get_report()
```

## Architecture

### System Overview

```

   API Layer             CLI Layer             SDK Layer





                         Agent Manager





  LLM Providers           Tools                 Memory

```

### Key Components

1. **Agent Manager**: Orchestrates agent lifecycle and execution
2. **Provider Factory**: Creates and manages LLM provider instances
3. **Tool Registry**: Manages available tools and their execution
4. **Memory System**: Handles short-term and long-term memory
5. **Cache Layer**: Provides multi-level caching (memory, disk, Redis)
6. **Security Layer**: Handles authentication, rate limiting, validation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/llamaagent.git
cd llamaagent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=llamaagent

# Run linting
ruff check src/
mypy src/

# Format code
black src/ tests/
```

### Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# Full test suite with coverage
pytest --cov=llamaagent --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LlamaAgent in your research, please cite:

```bibtex
@software{llamaagent2024,
  title = {LlamaAgent: A Production-Ready AI Agent Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llamaagent}
}
```

## Support

- Documentation: [https://llamaagent.readthedocs.io](https://llamaagent.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/yourusername/llamaagent/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/llamaagent/discussions)
- Email: support@llamaagent.ai

---

Built with LOVE: by the LlamaAgent Team
