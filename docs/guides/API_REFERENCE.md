# LlamaAgent API Reference

## Table of Contents

1. [Core Components](#core-components)
2. [Agents](#agents)
3. [Tools](#tools)
4. [LLM Providers](#llm-providers)
5. [Orchestration](#orchestration)
6. [OpenAI Integration](#openai-integration)
7. [Types and Models](#types-and-models)
8. [Storage](#storage)
9. [API Endpoints](#api-endpoints)

## Core Components

### AgentConfig

Configuration class for creating agents.

```python
from llamaagent import AgentConfig, AgentRole

config = AgentConfig(
    name="MyAgent",                    # Agent name (required)
    role=AgentRole.SPECIALIST,         # Agent role
    description="Agent description",   # Description
    max_iterations=10,                 # Max reasoning iterations
    temperature=0.7,                   # LLM temperature
    max_tokens=2000,                   # Max tokens per response
    timeout=300.0,                     # Execution timeout (seconds)
    retry_attempts=3,                  # Number of retry attempts
    system_prompt=None,                # Custom system prompt
    tools=["calculator", "web_search"], # Available tools
    memory_enabled=True,               # Enable conversation memory
    streaming=False,                   # Enable streaming responses
    spree_enabled=False,               # Enable SPRE planning
    dynamic_tools=False,               # Enable dynamic tool creation
    metadata={}                        # Additional metadata
)
```

### AgentRole

Enumeration of available agent roles:

- `COORDINATOR` - Orchestrates other agents
- `RESEARCHER` - Gathers and analyzes information
- `ANALYZER` - Performs data analysis
- `EXECUTOR` - Executes specific tasks
- `CRITIC` - Reviews and critiques work
- `PLANNER` - Creates execution plans
- `SPECIALIST` - Domain-specific expertise
- `GENERALIST` - General-purpose agent

## Agents

### ReactAgent

The primary agent implementation using ReAct (Reasoning + Acting) pattern.

```python
from llamaagent import ReactAgent
from llamaagent.llm import create_provider

agent = ReactAgent(
    config=config,                     # AgentConfig instance
    tools=tool_registry,               # Optional ToolRegistry
    memory=memory_instance,            # Optional memory instance
    llm_provider=create_provider("openai")  # LLM provider
)

# Execute a task
response = await agent.execute(
    task="Analyze the weather data",
    context={"location": "New York"}
)

# Stream execution
async for chunk in agent.stream_execute(task):
    print(chunk)
```

### BaseAgent

Abstract base class for custom agent implementations.

```python
from llamaagent.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentResponse:
        # Custom implementation
        pass
```

## Tools

### ToolRegistry

Manages available tools for agents.

```python
from llamaagent.tools import ToolRegistry, BaseTool

registry = ToolRegistry()

# Register a tool
registry.register(tool_instance)

# List available tools
tools = registry.list_tools()

# Get specific tool
calculator = registry.get_tool("calculator")
```

### Built-in Tools

#### Calculator
```python
from llamaagent.tools import Calculator

calc = Calculator()
result = await calc.execute("25 * 4 + 10")
```

#### PythonREPL
```python
from llamaagent.tools import PythonREPL

repl = PythonREPL(timeout=30.0)
result = await repl.execute("import math; math.sqrt(16)")
```

### Custom Tools

```python
from llamaagent.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "A custom tool"

    async def execute(self, *args, **kwargs) -> Any:
        # Implementation
        return result
```

## LLM Providers

### Creating Providers

```python
from llamaagent.llm import create_provider

# OpenAI
openai_provider = create_provider(
    "openai",
    api_key="your-key",
    model_name="gpt-4o-mini"
)

# Anthropic
anthropic_provider = create_provider(
    "anthropic",
    api_key="your-key",
    model_name="claude-3-sonnet-20240229"
)

# Local Ollama
ollama_provider = create_provider(
    "ollama",
    base_url="http://localhost:11434",
    model_name="llama3.2:3b"
)

# Mock (for testing)
mock_provider = create_provider("mock")
```

### Provider Methods

```python
# Generate response
response = await provider.generate_response(
    prompt="Hello, world!",
    max_tokens=1000,
    temperature=0.7
)

# Stream response
async for chunk in provider.generate_streaming_response(prompt):
    print(chunk)

# Generate embeddings
embeddings = await provider.embed_text(["text1", "text2"])
```

## Orchestration

### AgentOrchestrator

Coordinates multiple agents in workflows.

```python
from llamaagent.orchestrator import AgentOrchestrator, WorkflowDefinition, WorkflowStep

orchestrator = AgentOrchestrator()

# Register agents
orchestrator.register_agent(agent1)
orchestrator.register_agent(agent2)

# Define workflow
workflow = WorkflowDefinition(
    workflow_id="my_workflow",
    name="Data Analysis Workflow",
    description="Analyze and report on data",
    strategy=OrchestrationStrategy.SEQUENTIAL,
    steps=[
        WorkflowStep(
            step_id="gather",
            agent_name="ResearchAgent",
            task="Gather data from sources",
            timeout=300.0
        ),
        WorkflowStep(
            step_id="analyze",
            agent_name="AnalysisAgent",
            task="Analyze the gathered data",
            dependencies=["gather"]
        )
    ]
)

# Execute workflow
result = await orchestrator.execute_workflow(workflow.workflow_id)
```

### Orchestration Strategies

- `SEQUENTIAL` - Execute steps in order
- `PARALLEL` - Execute independent steps simultaneously
- `HIERARCHICAL` - Coordinator-based execution
- `DEBATE` - Multi-agent debate and synthesis
- `CONSENSUS` - Reach consensus among agents
- `PIPELINE` - Data pipeline execution

## OpenAI Integration

### OpenAIAgentsIntegration

Integrates with OpenAI Agents SDK.

```python
from llamaagent.integration.openai_agents import (
    OpenAIAgentsIntegration,
    OpenAIIntegrationConfig,
    OpenAIAgentMode
)

# Configure integration
config = OpenAIIntegrationConfig(
    mode=OpenAIAgentMode.HYBRID,
    openai_api_key="your-key",
    model_name="gpt-4o-mini",
    budget_limit=100.0,
    enable_tracing=True
)

integration = OpenAIAgentsIntegration(config)

# Register LlamaAgent for OpenAI compatibility
adapter = integration.register_agent(llamaagent_instance)

# Execute task
result = await integration.run_task(
    agent_name="MyAgent",
    task_input=TaskInput(
        id="task_001",
        task="Analyze this data",
        context={"data": [1, 2, 3]}
    )
)

# Check budget
budget_status = integration.get_budget_status()
```

### OpenAI Agent Modes

- `OPENAI_NATIVE` - Use OpenAI Agents SDK directly
- `LLAMAAGENT_WRAPPER` - Wrap LlamaAgent for OpenAI
- `HYBRID` - Use both systems interchangeably

## Types and Models

### TaskInput

```python
from llamaagent.types import TaskInput

task_input = TaskInput(
    id="unique_id",
    task="Task description",
    context={"key": "value"},
    agent_name="AgentName",
    metadata={}
)
```

### TaskOutput

```python
from llamaagent.types import TaskOutput, TaskStatus, TaskResult

output = TaskOutput(
    task_id="unique_id",
    status=TaskStatus.COMPLETED,
    result=TaskResult(
        success=True,
        data={"answer": 42},
        error=None,
        metadata={}
    )
)
```

### AgentResponse

```python
from llamaagent.agents.base import AgentResponse

response = AgentResponse(
    content="Response content",
    success=True,
    messages=[],          # Conversation history
    trace=[],            # Execution trace
    metadata={},         # Additional metadata
    execution_time=1.23, # Execution time in seconds
    tokens_used=150,     # Token count
    plan=None,           # Execution plan (if SPRE enabled)
    error=None           # Error message if failed
)
```

## Storage

### DatabaseManager

```python
from llamaagent.storage import DatabaseManager

db = DatabaseManager(connection_string="postgresql://...")

# Initialize database
await db.initialize()

# Store agent data
await db.store_agent_data(agent_id, data)

# Retrieve agent data
data = await db.get_agent_data(agent_id)

# Store conversation
await db.store_conversation(conversation_id, messages)
```

### VectorMemory

```python
from llamaagent.storage import VectorMemory

memory = VectorMemory(
    embedding_provider=provider,
    dimension=1536
)

# Add memory
await memory.add("Memory content", metadata={"type": "fact"})

# Search memories
results = await memory.search("query", top_k=5)

# Clear memory
await memory.clear()
```

## API Endpoints

### FastAPI Application

```python
from llamaagent.api import create_app

app = create_app()

# Run with uvicorn
# uvicorn llamaagent.api:create_app --factory --reload
```

### Available Endpoints

#### Health Check
```http
GET /health
```

#### Execute Task
```http
POST /v1/agents/{agent_name}/execute
Content-Type: application/json

{
    "task": "Analyze this data",
    "context": {"data": [1, 2, 3]},
    "stream": false
}
```

#### List Agents
```http
GET /v1/agents
```

#### Create Agent
```http
POST /v1/agents
Content-Type: application/json

{
    "name": "NewAgent",
    "role": "specialist",
    "description": "A new agent",
    "config": {
        "temperature": 0.7,
        "max_tokens": 2000
    }
}
```

#### Get Agent Info
```http
GET /v1/agents/{agent_name}
```

#### Delete Agent
```http
DELETE /v1/agents/{agent_name}
```

#### List Workflows
```http
GET /v1/workflows
```

#### Execute Workflow
```http
POST /v1/workflows/{workflow_id}/execute
Content-Type: application/json

{
    "context": {"param": "value"}
}
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/agents/MyAgent/stream');

ws.send(JSON.stringify({
    task: "Analyze this data",
    context: {}
}));

ws.onmessage = (event) => {
    const chunk = JSON.parse(event.data);
    console.log(chunk.content);
};
```

## Environment Variables

```bash
# LLM Provider Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
TOGETHER_API_KEY=your-together-key
COHERE_API_KEY=your-cohere-key

# Database
DATABASE_URL=postgresql://user:pass@localhost/llamaagent

# API Configuration
LLAMAAGENT_HOST=0.0.0.0
LLAMAAGENT_PORT=8000

# Security
JWT_SECRET_KEY=your-secret-key
API_KEY=your-api-key

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Development
DEBUG=false
LOG_LEVEL=INFO
```

## Error Handling

```python
from llamaagent.exceptions import (
    LLMError,
    AuthenticationError,
    BudgetExceededError,
    ToolExecutionError
)

try:
    response = await agent.execute(task)
except LLMError as e:
    print(f"LLM error: {e}")
except AuthenticationError as e:
    print(f"Auth error: {e}")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Agent Configuration**
   - Use appropriate agent roles
   - Set reasonable timeouts
   - Enable retries for resilience
   - Use SPRE for complex tasks

2. **Tool Usage**
   - Register only needed tools
   - Implement proper error handling
   - Set tool timeouts
   - Validate tool inputs

3. **LLM Providers**
   - Use appropriate models for tasks
   - Monitor token usage
   - Implement fallback providers
   - Cache responses when possible

4. **Orchestration**
   - Design workflows carefully
   - Use parallel execution when possible
   - Handle failures gracefully
   - Monitor workflow performance

5. **Production Deployment**
   - Use environment variables for secrets
   - Enable monitoring and logging
   - Implement rate limiting
   - Use connection pooling
   - Enable health checks

## Support

For questions and support:
- Email: nikjois@llamasearch.ai
- GitHub: https://github.com/nikjois/llamaagent
- Documentation: https://llamaagent.readthedocs.io
