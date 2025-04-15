"""
Configuration classes for the LlamaAgent framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for language models used by the agent."""

    provider: str = "anthropic"  # Options: "anthropic", "openai", "llamadb"
    model: str = "claude-3-opus-20240229"  # Model identifier
    temperature: float = 0.7  # Temperature for generation
    max_tokens: int = 2000  # Maximum tokens to generate
    api_key: Optional[str] = None  # API key (will fall back to env vars if None)
    extra_params: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional parameters for the provider


@dataclass
class MemoryConfig:
    """Configuration for the agent's memory system."""

    short_term_limit: int = 100  # Maximum number of items in short-term memory
    long_term_enabled: bool = True  # Whether to use long-term memory
    retrieval_limit: int = 10  # Default number of items to retrieve
    relevance_threshold: float = 0.7  # Minimum relevance score for retrieval
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"  # Model for embeddings
    )
    vector_db_url: Optional[str] = None  # Connection URL for vector database
    vector_db_api_key: Optional[str] = None  # API key for vector database


@dataclass
class ToolConfig:
    """Configuration for the agent's tools."""

    timeout: float = 30.0  # Timeout for tool execution in seconds
    max_retries: int = 3  # Maximum number of retries for failed tools
    parallel_execution: bool = False  # Whether to execute tools in parallel
    allow_fallbacks: bool = True  # Whether to allow fallback tools


@dataclass
class AgentConfig:
    """
    Main configuration for the agent.

    This class combines all the configuration options for the different
    components of the agent, including the language model, memory system,
    tools, and general settings.
    """

    name: str = "LlamaAgent"  # Name of the agent
    description: str = "A flexible agent for orchestrating tasks"  # Description
    version: str = "0.1.0"  # Version of the agent

    # Component configurations
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    tool_config: ToolConfig = field(default_factory=ToolConfig)

    # Agent behavior settings
    verbose: bool = False  # Whether to log details
    max_steps: int = 50  # Maximum number of steps to execute
    max_iterations: int = 5  # Maximum iterations for planning
    allow_self_reflection: bool = True  # Whether to reflect on performance

    # List of available tools (populated by the agent)
    available_tools: List[Any] = field(default_factory=list)

    # Default system prompt for the agent's LLM
    system_prompt: str = """
    You are a helpful and thoughtful assistant that can break down tasks into steps
    and execute them using available tools. You should approach problems methodically,
    explaining your reasoning and being proactive about gathering necessary information.
    
    When users ask you to perform complex tasks:
    1. Break the task down into manageable steps
    2. Use the appropriate tools for each step
    3. Reflect on the results and adjust your approach as needed
    4. Provide clear, concise summaries of what you've learned
    
    Always be helpful, accurate, and honest. If you don't know something, say so rather
    than making up information. Use tools to find information whenever possible rather
    than relying on your built-in knowledge.
    """
