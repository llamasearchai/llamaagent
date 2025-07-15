---
title: "API Reference"
permalink: /docs/api-reference/
excerpt: "Complete API reference for LlamaAgent framework"
toc: true
toc_sticky: true
---

# API Reference

Complete reference for all LlamaAgent classes, methods, and functions.

## Core Classes

### AgentConfig

Configuration class for creating agents.

```python
class AgentConfig:
    """Configuration for AI agents."""
    
    def __init__(
        self,
        name: str = "TestAgent",
        role: AgentRole = AgentRole.GENERALIST,
        description: str = "",
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 300.0,
        retry_attempts: int = 3,
        system_prompt: Optional[str] = None,
        tools: List[str] = None,
        memory_enabled: bool = True,
        streaming: bool = False,
        spree_enabled: bool = True,
        dynamic_tools: bool = False,
        llm_provider: Any = None,
        verbose: bool = False,
        debug: bool = False,
        metadata: Dict[str, Any] = None
    ):
        """Initialize agent configuration."""
```

#### Parameters

- **name** (`str`): Agent identifier. Default: "TestAgent"
- **role** (`AgentRole`): Agent role. Default: `AgentRole.GENERALIST`
- **description** (`str`): Agent description. Default: ""
- **max_iterations** (`int`): Maximum reasoning iterations. Default: 10
- **temperature** (`float`): LLM temperature (0-1). Default: 0.7
- **max_tokens** (`int`): Maximum tokens per response. Default: 2000
- **timeout** (`float`): Execution timeout in seconds. Default: 300.0
- **retry_attempts** (`int`): Number of retry attempts. Default: 3
- **system_prompt** (`Optional[str]`): Custom system prompt. Default: None
- **tools** (`List[str]`): Available tools. Default: []
- **memory_enabled** (`bool`): Enable conversation memory. Default: True
- **streaming** (`bool`): Enable streaming responses. Default: False
- **spree_enabled** (`bool`): Enable SPRE planning. Default: True
- **dynamic_tools** (`bool`): Enable dynamic tool creation. Default: False
- **llm_provider** (`Any`): LLM provider instance. Default: None
- **verbose** (`bool`): Verbose logging. Default: False
- **debug** (`bool`): Debug mode. Default: False
- **metadata** (`Dict[str, Any]`): Additional metadata. Default: {}

#### Properties

- **agent_name** (`str`): Backward compatibility property for name

### ReactAgent

The primary agent implementation using ReAct (Reasoning + Acting) pattern.

```python
class ReactAgent(BaseAgent):
    """ReAct agent with reasoning and acting capabilities."""
    
    def __init__(
        self,
        config: AgentConfig,
        llm_provider: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None
    ):
        """Initialize ReactAgent."""
```

#### Parameters

- **config** (`AgentConfig`): Agent configuration
- **llm_provider** (`Optional[Any]`): LLM provider instance
- **tools** (`Optional[List[Any]]`): List of tool instances
- **memory** (`Optional[Any]`): Memory implementation

#### Methods

##### execute

```python
async def execute(
    self,
    task: str,
    context: Optional[Dict[str, Any]] = None
) -> AgentResponse:
    """Execute a task and return response."""
```

**Parameters:**
- **task** (`str`): The task to execute
- **context** (`Optional[Dict[str, Any]]`): Optional context dictionary

**Returns:**
- `AgentResponse`: Execution results

##### execute_task

```python
async def execute_task(self, task_input: Any) -> Any:
    """Execute a task using TaskInput/TaskOutput interface."""
```

**Parameters:**
- **task_input** (`Any`): The task input

**Returns:**
- `TaskOutput`: Task execution results

##### stream_execute

```python
async def stream_execute(
    self,
    task: str,
    context: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """Stream execution results."""
```

**Parameters:**
- **task** (`str`): The task to execute
- **context** (`Optional[Dict[str, Any]]`): Optional context dictionary

**Yields:**
- `str`: String chunks of the response

## Agent Roles

### AgentRole

Enumeration of available agent roles.

```python
class AgentRole(str, Enum):
    """Agent roles for multi-agent systems."""
    
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    CRITIC = "critic"
    PLANNER = "planner"
    SPECIALIST = "specialist"
    TOOL_SPECIFIER = "tool_specifier"
    TOOL_SYNTHESIZER = "tool_synthesizer"
    ORCHESTRATOR = "orchestrator"
    GENERALIST = "generalist"
```

## Response Types

### AgentResponse

Response from agent execution.

```python
@dataclass
class AgentResponse:
    """Agent execution response with full trace."""
    
    content: str
    success: bool = True
    messages: List[AgentMessage] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[ExecutionPlan] = None
```

#### Fields

- **content** (`str`): Main response content
- **success** (`bool`): Whether execution was successful
- **messages** (`List[AgentMessage]`): Message history
- **trace** (`List[Dict[str, Any]]`): Execution trace
- **final_result** (`Optional[str]`): Final result
- **error** (`Optional[str]`): Error message if failed
- **tokens_used** (`int`): Number of tokens used
- **execution_time** (`float`): Execution time in seconds
- **metadata** (`Dict[str, Any]`): Additional metadata
- **plan** (`Optional[ExecutionPlan]`): Execution plan if available

### TaskInput

Input for task execution.

```python
@dataclass
class TaskInput:
    """Input payload for task execution."""
    
    id: str
    task: str
    prompt: Optional[str] = None
    data: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Methods

##### model_dump

```python
def model_dump(self) -> Dict[str, Any]:
    """Return a serializable representation."""
```

### TaskOutput

Output from task execution.

```python
@dataclass
class TaskOutput:
    """Final state of a task once processing has finished."""
    
    task_id: str
    status: TaskStatus
    result: Optional[TaskResult] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

#### Properties

- **success** (`bool`): Whether the task was successful
- **error** (`Optional[str]`): Error message if failed

#### Methods

##### model_dump

```python
def model_dump(self) -> Dict[str, Any]:
    """Return a serializable representation."""
```

## Tools

### Tool

Base class for creating tools.

```python
class Tool(ABC):
    """Abstract base class for tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.CUSTOM,
        version: str = "1.0.0",
        security_level: ToolSecurityLevel = ToolSecurityLevel.PUBLIC
    ):
        """Initialize tool."""
```

#### Methods

##### get_parameters

```python
@abstractmethod
def get_parameters(self) -> List[ToolParameter]:
    """Get tool parameters."""
```

##### execute

```python
@abstractmethod
async def execute(
    self,
    parameters: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None
) -> ToolResult:
    """Execute the tool."""
```

### CalculatorTool

Built-in calculator tool for mathematical operations.

```python
class CalculatorTool(Tool):
    """Calculator tool for mathematical operations."""
    
    def __init__(self):
        """Initialize calculator tool."""
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            category=ToolCategory.COMPUTATION
        )
```

#### Methods

##### execute

```python
async def execute(
    self,
    parameters: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None
) -> ToolResult:
    """Execute mathematical calculation."""
```

**Parameters:**
- **parameters** (`Dict[str, Any]`): Must contain "expression" key
- **context** (`Optional[ToolExecutionContext]`): Execution context

**Returns:**
- `ToolResult`: Calculation result

### PythonREPLTool

Built-in Python REPL tool for code execution.

```python
class PythonREPLTool(Tool):
    """Python REPL tool for code execution."""
    
    def __init__(self, sandbox: bool = True):
        """Initialize Python REPL tool."""
        super().__init__(
            name="python_repl",
            description="Execute Python code in a REPL environment",
            category=ToolCategory.COMPUTATION,
            security_level=ToolSecurityLevel.RESTRICTED
        )
        self.sandbox = sandbox
```

#### Methods

##### execute

```python
async def execute(
    self,
    parameters: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None
) -> ToolResult:
    """Execute Python code."""
```

**Parameters:**
- **parameters** (`Dict[str, Any]`): Must contain "code" key
- **context** (`Optional[ToolExecutionContext]`): Execution context

**Returns:**
- `ToolResult`: Code execution result

## LLM Providers

### LLMFactory

Factory for creating LLM providers.

```python
class LLMFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(
        provider_name: str,
        **kwargs: Any
    ) -> BaseLLMProvider:
        """Create an LLM provider."""
```

**Parameters:**
- **provider_name** (`str`): Name of the provider ("openai", "anthropic", etc.)
- **kwargs** (`Any`): Provider-specific configuration

**Returns:**
- `BaseLLMProvider`: Provider instance

### OpenAIProvider

OpenAI LLM provider.

```python
class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """Initialize OpenAI provider."""
```

#### Methods

##### complete

```python
async def complete(
    self,
    messages: List[Dict[str, str]],
    **kwargs: Any
) -> Dict[str, Any]:
    """Complete a conversation."""
```

### MockProvider

Mock LLM provider for testing.

```python
class MockProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, model_name: str = "mock-gpt-4"):
        """Initialize mock provider."""
```

## Memory Systems

### SimpleMemory

Simple in-memory storage for conversation history.

```python
class SimpleMemory:
    """Simple in-memory conversation storage."""
    
    def __init__(self, max_messages: int = 100):
        """Initialize simple memory."""
```

#### Methods

##### add_message

```python
def add_message(self, message: Dict[str, Any]) -> None:
    """Add a message to memory."""
```

##### get_context

```python
def get_context(self, max_tokens: int = 4000) -> List[Dict[str, Any]]:
    """Get conversation context."""
```

##### clear

```python
def clear(self) -> None:
    """Clear all memory."""
```

## Exceptions

### LlamaAgentError

Base exception for all LlamaAgent errors.

```python
class LlamaAgentError(Exception):
    """Base exception for LlamaAgent."""
```

### AgentExecutionError

Exception raised during agent execution.

```python
class AgentExecutionError(LlamaAgentError):
    """Exception raised during agent execution."""
```

### ToolExecutionError

Exception raised during tool execution.

```python
class ToolExecutionError(LlamaAgentError):
    """Exception raised during tool execution."""
```

### ProviderError

Exception raised by LLM providers.

```python
class ProviderError(LlamaAgentError):
    """Exception raised by LLM providers."""
```

## Utilities

### create_agent

Utility function to create agents.

```python
def create_agent(
    name: str,
    provider: str = "mock",
    tools: Optional[List[str]] = None,
    **kwargs: Any
) -> ReactAgent:
    """Create an agent with default configuration."""
```

**Parameters:**
- **name** (`str`): Agent name
- **provider** (`str`): LLM provider name
- **tools** (`Optional[List[str]]`): Tool names
- **kwargs** (`Any`): Additional configuration

**Returns:**
- `ReactAgent`: Configured agent

### create_tool

Utility function to create tools from functions.

```python
def create_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.CUSTOM
) -> Tool:
    """Create a tool from a function."""
```

**Parameters:**
- **func** (`Callable`): Function to wrap as tool
- **name** (`Optional[str]`): Tool name (defaults to function name)
- **description** (`Optional[str]`): Tool description (defaults to docstring)
- **category** (`ToolCategory`): Tool category

**Returns:**
- `Tool`: Tool instance

## Constants

### Version Information

```python
__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
```

### Default Configuration

```python
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 300.0,
    "retry_attempts": 3,
    "memory_enabled": True,
    "spree_enabled": True
}
```

## Type Hints

### Common Types

```python
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

# Type aliases
AgentID = str
TaskID = str
ToolName = str
ProviderName = str
MessageRole = str
TokenCount = int
Temperature = float
ExecutionTime = float
```

## Examples

### Basic Agent Creation

```python
from llamaagent import ReactAgent, AgentConfig
from llamaagent.tools import CalculatorTool

config = AgentConfig(
    name="MathAgent",
    tools=["calculator"],
    temperature=0.7
)

agent = ReactAgent(
    config=config,
    tools=[CalculatorTool()]
)
```

### Custom Tool Creation

```python
from llamaagent.tools import Tool, ToolCategory

@Tool.create(
    name="weather",
    description="Get current weather",
    category=ToolCategory.WEB_API
)
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Sunny, 72°F in {location}"
```

### Error Handling

```python
from llamaagent.exceptions import AgentExecutionError

try:
    response = await agent.execute("Complex task")
except AgentExecutionError as e:
    print(f"Agent execution failed: {e}")
```

---

For more examples and detailed usage, see the [User Guide](/docs/user-guide/) and [Examples](/docs/examples/). 