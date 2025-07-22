# Getting Started with LlamaAgent

Welcome to LlamaAgent, the most comprehensive agent framework for building intelligent AI agents. This guide will help you get started quickly.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Your First Agent](#your-first-agent)
5. [Advanced Usage](#advanced-usage)
6. [Next Steps](#next-steps)

## Installation

### Requirements

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- PostgreSQL (optional, for advanced features)
- Redis (optional, for caching)

### Install from PyPI

```bash
pip install llamaagent
```

### Install from Source

```bash
git clone https://github.com/yourusername/llamaagent.git
cd llamaagent
pip install -e .
```

### Install with Extras

```bash
# Full installation with all features
pip install llamaagent[all]

# Specific features
pip install llamaagent[openai]  # OpenAI integration
pip install llamaagent[semantic]  # Semantic caching
pip install llamaagent[monitoring]  # Monitoring and observability
```

## Quick Start

Here's a simple example to get you started:

```python
from llamaagent import Agent, Tool
from llamaagent.llm import LiteLLMProvider

# Initialize the LLM provider
llm = LiteLLMProvider(model="gpt-4")

# Create a simple calculator tool
@Tool.create
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression"""
    return eval(expression)

# Create an agent
agent = Agent(
    name="MathAgent",
    llm=llm,
    tools=[calculator],
    system_prompt="You are a helpful math assistant."
)

# Use the agent
response = await agent.run("What is 25 * 4 + 10?")
print(response)  # The result is 110
```

## Core Concepts

### 1. Agents

Agents are the core building blocks of LlamaAgent. They combine:
- **LLM Provider**: The language model backend
- **Tools**: Functions the agent can call
- **Memory**: Short and long-term memory systems
- **Prompting**: Advanced prompting strategies

### 2. Tools

Tools are functions that agents can use to interact with the world:

```python
from llamaagent.tools import Tool

@Tool.create
def weather(location: str) -> dict:
    """Get weather for a location"""
    # Implementation here
    return {"temperature": 72, "condition": "sunny"}

# Tools can be async
@Tool.create
async def fetch_data(url: str) -> str:
    """Fetch data from a URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### 3. Memory Systems

LlamaAgent provides multiple memory backends:

```python
from llamaagent.memory import VectorMemory, ConversationMemory

# Vector memory for semantic search
vector_memory = VectorMemory(
    embedding_model="all-MiniLM-L6-v2",
    dimension=384
)

# Conversation memory for chat history
chat_memory = ConversationMemory(max_turns=10)

# Create agent with memory
agent = Agent(
    name="MemoryAgent",
    llm=llm,
    memory=vector_memory,
    conversation_memory=chat_memory
)
```

### 4. LLM Providers

Support for 100+ models through LiteLLM:

```python
from llamaagent.llm import LiteLLMProvider

# OpenAI
openai_llm = LiteLLMProvider(model="gpt-4")

# Anthropic
claude_llm = LiteLLMProvider(model="claude-3-opus")

# Open source models
llama_llm = LiteLLMProvider(model="together_ai/llama-3-70b")

# Local models
local_llm = LiteLLMProvider(model="ollama/llama3.2")
```

## Your First Agent

Let's build a more complex agent that can search the web and answer questions:

```python
from llamaagent import Agent
from llamaagent.tools import WebSearchTool, CalculatorTool
from llamaagent.llm import LiteLLMProvider
from llamaagent.memory import VectorMemory
from llamaagent.prompting import ChainOfThoughtStrategy

# Initialize components
llm = LiteLLMProvider(model="gpt-4", temperature=0.7)
memory = VectorMemory()
cot_strategy = ChainOfThoughtStrategy()

# Create agent with multiple tools
agent = Agent(
    name="ResearchAssistant",
    llm=llm,
    tools=[
        WebSearchTool(),
        CalculatorTool()
    ],
    memory=memory,
    prompting_strategy=cot_strategy,
    system_prompt="""You are a research assistant. 
    Break down complex questions into steps.
    Search for information when needed.
    Provide accurate, well-sourced answers."""
)

# Use the agent
async def main():
    # Simple question
    response = await agent.run(
        "What is the population of Tokyo and how does it compare to New York?"
    )
    print(response)
    
    # Multi-step reasoning
    response = await agent.run(
        "Calculate the population density of Tokyo if its area is 2,194 km²"
    )
    print(response)
    
    # Save conversation
    await agent.save_conversation("research_session.json")

# Run
import asyncio
asyncio.run(main())
```

## Advanced Usage

### 1. Custom Prompting Strategies

```python
from llamaagent.prompting import PromptingStrategy

class ScientificMethodStrategy(PromptingStrategy):
    def format_prompt(self, task: str, context: dict) -> str:
        return f"""
        Apply the scientific method:
        1. Observation: {task}
        2. Hypothesis: Form a hypothesis
        3. Experiment: Design a test
        4. Analysis: Analyze results
        5. Conclusion: Draw conclusions
        
        Context: {context}
        """

agent = Agent(
    name="ScientistAgent",
    llm=llm,
    prompting_strategy=ScientificMethodStrategy()
)
```

### 2. Multi-Agent Systems

```python
from llamaagent.orchestrator import Orchestrator
from llamaagent.agents import ResearchAgent, WriterAgent, CriticAgent

# Create specialized agents
researcher = ResearchAgent(llm=llm)
writer = WriterAgent(llm=llm)
critic = CriticAgent(llm=llm)

# Create orchestrator
orchestrator = Orchestrator(
    agents=[researcher, writer, critic],
    workflow="sequential"  # or "parallel", "hierarchical"
)

# Run complex task
result = await orchestrator.run(
    "Write a comprehensive article about quantum computing"
)
```

### 3. Production Deployment

```python
from llamaagent import Agent
from llamaagent.monitoring import PrometheusMonitor
from llamaagent.cache import LLMCache
from llamaagent.security import RateLimiter, InputValidator

# Production-ready agent
agent = Agent(
    name="ProductionAgent",
    llm=llm,
    # Caching
    cache=LLMCache(
        ttl=3600,
        semantic_threshold=0.95
    ),
    # Monitoring
    monitor=PrometheusMonitor(
        port=9090,
        collect_latency=True,
        collect_tokens=True
    ),
    # Security
    middleware=[
        RateLimiter(requests_per_minute=60),
        InputValidator(max_length=1000)
    ]
)

# Deploy as API
from llamaagent.api import create_app

app = create_app(agent)
app.run(host="0.0.0.0", port=8000)
```

### 4. Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  llamaagent:
    image: ghcr.io/yourusername/llamaagent:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@postgres:5432/llamaagent
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_DB=llamaagent
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  
  redis:
    image: redis:7-alpine
```

## Next Steps

1. **Explore Examples**: Check out the [examples directory](./examples/) for more use cases
2. **Read the API Reference**: Detailed documentation of all classes and functions
3. **Join the Community**: Discord, GitHub Discussions, and more
4. **Contribute**: We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md)

### Useful Resources

- [Architecture Overview](./architecture/overview.md)
- [Advanced Prompting Guide](./guides/prompting.md)
- [Tool Development Guide](./guides/tools.md)
- [Performance Optimization](./guides/performance.md)
- [Security Best Practices](./guides/security.md)

### Getting Help

- Documentation [Documentation](https://docs.llamaagent.ai)
-  [Discord Community](https://discord.gg/llamaagent)
-  [Issue Tracker](https://github.com/yourusername/llamaagent/issues)
-  [Email Support](mailto:support@llamaagent.ai)

Welcome to the LlamaAgent community! LlamaAgentEnhanced