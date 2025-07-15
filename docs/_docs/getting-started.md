---
title: "Getting Started"
permalink: /docs/getting-started/
excerpt: "Quick start guide for LlamaAgent framework"
toc: true
---

# Getting Started with LlamaAgent

Welcome to LlamaAgent! This guide will help you get up and running with the framework quickly.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Basic knowledge of Python and async programming

## Installation

### Quick Installation

```bash
pip install llamaagent
```

### Full Installation with All Features

```bash
pip install llamaagent[all]
```

### Development Installation

```bash
git clone https://github.com/yourusername/llamaagent.git
cd llamaagent
pip install -e ".[dev,all]"
```

## Your First Agent

Let's create a simple agent that can perform mathematical calculations:

```python
import asyncio
from llamaagent import ReactAgent, AgentConfig
from llamaagent.tools import CalculatorTool
from llamaagent.llm import MockProvider

async def main():
    # Create agent configuration
    config = AgentConfig(
        name="MathAgent",
        description="A helpful mathematical assistant",
        tools=["calculator"],
        temperature=0.7,
        max_tokens=2000
    )
    
    # Create the agent
    agent = ReactAgent(
        config=config,
        llm_provider=MockProvider(),
        tools=[CalculatorTool()]
    )
    
    # Execute a task
    response = await agent.execute("What is 25 * 4 + 10?")
    
    print(f"Agent: {response.content}")
    print(f"Success: {response.success}")
    print(f"Execution time: {response.execution_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the Components

### Agent Configuration

The `AgentConfig` class defines how your agent behaves:

```python
config = AgentConfig(
    name="MyAgent",              # Agent identifier
    description="Agent purpose",  # What the agent does
    tools=["calculator"],        # Available tools
    temperature=0.7,             # LLM creativity (0-1)
    max_tokens=2000,            # Maximum response length
    timeout=300.0,              # Execution timeout
    retry_attempts=3,           # Number of retries
    memory_enabled=True,        # Enable conversation memory
    spree_enabled=True,         # Enable SPRE planning
)
```

### LLM Providers

LlamaAgent supports multiple LLM providers:

```python
from llamaagent.llm import (
    OpenAIProvider,
    AnthropicProvider,
    CohereProvider,
    MockProvider
)

# OpenAI (requires API key)
openai_provider = OpenAIProvider(
    api_key="your-openai-key",
    model="gpt-4"
)

# Anthropic (requires API key)
anthropic_provider = AnthropicProvider(
    api_key="your-anthropic-key",
    model="claude-3-sonnet-20240229"
)

# Mock provider (for testing)
mock_provider = MockProvider()
```

### Tools

Tools extend your agent's capabilities:

```python
from llamaagent.tools import (
    CalculatorTool,
    PythonREPLTool,
    WebSearchTool
)

# Available built-in tools
tools = [
    CalculatorTool(),           # Mathematical calculations
    PythonREPLTool(),          # Python code execution
    WebSearchTool(),           # Web search capability
]
```

## Advanced Example

Here's a more advanced example with memory and multiple tools:

```python
import asyncio
from llamaagent import ReactAgent, AgentConfig
from llamaagent.tools import CalculatorTool, PythonREPLTool
from llamaagent.memory import SimpleMemory
from llamaagent.llm import MockProvider

async def advanced_example():
    # Create memory system
    memory = SimpleMemory()
    
    # Create agent configuration
    config = AgentConfig(
        name="AdvancedAgent",
        description="Multi-tool agent with memory",
        tools=["calculator", "python_repl"],
        temperature=0.7,
        max_tokens=4000,
        memory_enabled=True,
        spree_enabled=True
    )
    
    # Create agent with memory and multiple tools
    agent = ReactAgent(
        config=config,
        llm_provider=MockProvider(),
        tools=[CalculatorTool(), PythonREPLTool()],
        memory=memory
    )
    
    # Execute multiple related tasks
    tasks = [
        "Calculate 15% of 200",
        "Create a Python function to calculate compound interest",
        "Use the function to calculate interest on $1000 at 5% for 3 years"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}: {task} ---")
        response = await agent.execute(task)
        print(f"Response: {response.content}")
        print(f"Success: {response.success}")
        
        # Show memory context
        if hasattr(agent.memory, 'get_context'):
            context = agent.memory.get_context()
            print(f"Memory entries: {len(context)}")

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

## CLI Usage

LlamaAgent provides a command-line interface:

```bash
# Start interactive chat
llamaagent chat

# Execute a single task
llamaagent execute "Calculate the square root of 144"

# Start the API server
llamaagent server --port 8000

# Run benchmarks
llamaagent benchmark --help
```

## API Server

Start a REST API server:

```python
from llamaagent.api import create_app
import uvicorn

# Create the FastAPI application
app = create_app()

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then access the API:

```bash
# Health check
curl http://localhost:8000/health

# Execute task
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 2 + 2?", "agent_name": "MathAgent"}'
```

## Configuration

### Environment Variables

```bash
# LLM Provider settings
LLAMAAGENT_PROVIDER=openai
LLAMAAGENT_MODEL=gpt-4
LLAMAAGENT_API_KEY=your-api-key
LLAMAAGENT_TEMPERATURE=0.7

# Database settings
DATABASE_URL=sqlite:///llamaagent.db
REDIS_URL=redis://localhost:6379

# API settings
LLAMAAGENT_API_HOST=0.0.0.0
LLAMAAGENT_API_PORT=8000
```

### Configuration File

Create a `config.yaml` file:

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000

agents:
  default:
    name: DefaultAgent
    tools:
      - calculator
      - python_repl
    memory_enabled: true
    spree_enabled: true

api:
  host: 0.0.0.0
  port: 8000
  cors_enabled: true

database:
  url: sqlite:///llamaagent.db
  
monitoring:
  enabled: true
  prometheus_port: 9090
```

## Error Handling

Always handle errors gracefully:

```python
import asyncio
from llamaagent import ReactAgent, AgentConfig
from llamaagent.llm import MockProvider

async def error_handling_example():
    try:
        config = AgentConfig(name="TestAgent")
        agent = ReactAgent(config=config, llm_provider=MockProvider())
        
        response = await agent.execute("Invalid task that might fail")
        
        if response.success:
            print(f"Success: {response.content}")
        else:
            print(f"Error: {response.error}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(error_handling_example())
```

## Next Steps

Now that you have a basic understanding of LlamaAgent, explore these topics:

1. **[Agents](/docs/agents/)** - Learn about different agent types
2. **[Tools](/docs/tools/)** - Discover available tools and create custom ones
3. **[Memory](/docs/memory/)** - Understand memory systems
4. **[Providers](/docs/providers/)** - Configure different LLM providers
5. **[Deployment](/docs/deployment/)** - Deploy to production

## Common Issues

### Import Errors

If you encounter import errors, ensure you have the correct dependencies:

```bash
pip install llamaagent[all]
```

### API Key Issues

Make sure your API keys are properly configured:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

### Performance Issues

For better performance, consider:

- Using async/await properly
- Enabling caching
- Optimizing tool selection
- Using appropriate model sizes

## Getting Help

- **Documentation**: [Full documentation](/docs/)
- **API Reference**: [API docs](/api/)
- **GitHub Issues**: [Report bugs](https://github.com/yourusername/llamaagent/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/llamaagent/discussions)
- **Email**: [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)

---

Ready to build more advanced agents? Continue with the [User Guide](/docs/user-guide/) or explore specific topics in the documentation. 