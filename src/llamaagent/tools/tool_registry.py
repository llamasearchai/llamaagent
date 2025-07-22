"""
Tool Registry - Enterprise Dynamic Tool Integration

This module implements advanced tool management capabilities including:
- Dynamic tool registration and discovery
- Tool versioning and compatibility
- Tool execution context and sandboxing
- Performance monitoring and caching
- Security validation and access control

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import math
import sys
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Sized

from pydantic import BaseModel, Field

try:
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
except ImportError:
    # Fallback tracer
    class MockTracer:
        def start_as_current_span(self, name: str) -> 'MockSpan':
            return MockSpan()

    class MockSpan:
        def set_attribute(self, key: str, value: Any) -> None:
            pass

        def __enter__(self) -> 'MockSpan':
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    tracer = MockTracer()

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    COMPUTATION = "computation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    FILE_SYSTEM = "file_system"
    WEB_API = "web_api"
    DATABASE = "database"
    MACHINE_LEARNING = "machine_learning"
    REASONING = "reasoning"
    VISUALIZATION = "visualization"
    SYSTEM = "system"
    CUSTOM = "custom"


class ToolSecurityLevel(Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    PRIVATE = "private"
    ADMIN_ONLY = "admin_only"


@dataclass
class ToolExecutionContext:
    """Context for tool execution"""

    agent_id: str = ""
    session_id: str = ""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    permissions: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30  # seconds
    max_memory_mb: int = 512
    max_cpu_percent: int = 50


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Any] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Tool execution result"""

    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class Tool(ABC):
    """Abstract base class for tools"""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.CUSTOM,
        version: str = "1.0.0",
        security_level: ToolSecurityLevel = ToolSecurityLevel.PUBLIC,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.category = category
        self.version = version
        self.security_level = security_level
        self.created_at = datetime.now(timezone.utc)
        self.usage_count = 0
        self.last_used: Optional[datetime] = None
        self.performance_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameters"""

    @abstractmethod
    async def execute(
        self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        """Execute the tool"""

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate tool parameters"""
        errors: List[str] = []
        tool_params = {p.name: p for p in self.get_parameters()}

        # Check required parameters
        for param in self.get_parameters():
            if param.required and param.name not in parameters:
                errors.append(f"Missing required parameter: {param.name}")

        # Check parameter types and constraints
        for name, value in parameters.items():
            if name in tool_params:
                param = tool_params[name]
                # Validate constraints
                constraint_errors = self._validate_constraints(value, param.constraints)
                errors.extend(constraint_errors)

        return errors

    def _validate_constraints(
        self, value: Any, constraints: Dict[str, Any]
    ) -> List[str]:
        """Validate parameter constraints"""
        errors: List[str] = []

        if "min_value" in constraints and isinstance(value, (int, float)):
            if value < constraints["min_value"]:
                errors.append(
                    f"Value {value} is below minimum {constraints['min_value']}"
                )

        if "max_value" in constraints and isinstance(value, (int, float)):
            if value > constraints["max_value"]:
                errors.append(
                    f"Value {value} is above maximum {constraints['max_value']}"
                )

        if "min_length" in constraints and hasattr(value, "__len__"):
            # Type check for Sized objects
            if isinstance(value, Sized):
                if len(value) < constraints["min_length"]:
                    errors.append(
                        f"Length {len(value)} is below minimum {constraints['min_length']}"
                    )

        if "max_length" in constraints and hasattr(value, "__len__"):
            # Type check for Sized objects
            if isinstance(value, Sized):
                if len(value) > constraints["max_length"]:
                    errors.append(
                        f"Length {len(value)} is above maximum {constraints['max_length']}"
                    )

        if "allowed_values" in constraints:
            if value not in constraints["allowed_values"]:
                errors.append(
                    f"Value {value} not in allowed values: {constraints['allowed_values']}"
                )

        return errors

    def update_stats(self, execution_time: float, success: bool) -> None:
        """Update performance statistics"""
        self.performance_stats["total_executions"] += 1
        self.performance_stats["total_execution_time"] += execution_time
        self.performance_stats["average_execution_time"] = (
            self.performance_stats["total_execution_time"]
            / self.performance_stats["total_executions"]
        )

        if success:
            self.performance_stats["successful_executions"] += 1
        else:
            self.performance_stats["failed_executions"] += 1

        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)


class CalculatorTool(Tool):
    """Calculator tool for mathematical expressions"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions",
            category=ToolCategory.COMPUTATION,
            security_level=ToolSecurityLevel.PUBLIC,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="str",
                description="Mathematical expression to evaluate",
                required=True,
                examples=["2 + 2", "sqrt(16)", "sin(pi/2)", "10 * 5"],
            )
        ]

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        try:
            expression = parameters["expression"]

            # Safe evaluation with limited globals
            safe_globals: Dict[str, Any] = {
                "__builtins__": {},
                "abs": abs,
                "round": round,
                "pow": pow,
                "min": min,
                "max": max,
                "sum": sum,
                "int": int,
                "float": float,
                "math": math,
                "pi": math.pi,
                "e": math.e,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "exp": math.exp,
            }

            result = eval(expression, safe_globals, {})
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_stats(execution_time, True)

            return ToolResult(
                success=True, result=result, execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_stats(execution_time, False)

            return ToolResult(
                success=False, error=str(e), execution_time=execution_time
            )


class PythonREPLTool(Tool):
    """Python REPL tool for code execution"""

    def __init__(self):
        super().__init__(
            name="python_repl",
            description="Execute Python code",
            category=ToolCategory.COMPUTATION,
            security_level=ToolSecurityLevel.RESTRICTED,
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="str",
                description="Python code to execute",
                required=True,
                examples=["print('Hello, World!')", "x = 5\nprint(x * 2)"],
            )
        ]

    async def execute(
        self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        old_stdout = None
        captured_output = None

        try:
            code = parameters["code"]

            # Capture stdout
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output

            # Execute code in restricted environment
            safe_globals: Dict[str, Any] = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "any": any,
                    "all": all,
                    "min": min,
                    "max": max,
                    "sum": sum,
                }
            }

            exec(code, safe_globals, {})

            if old_stdout:
                sys.stdout = old_stdout

            output = captured_output.getvalue() if captured_output else ""
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_stats(execution_time, True)

            return ToolResult(
                success=True, result=output, execution_time=execution_time
            )

        except Exception as e:
            if old_stdout:
                sys.stdout = old_stdout
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.update_stats(execution_time, False)

            return ToolResult(
                success=False, error=str(e), execution_time=execution_time
            )


class ToolRegistry:
    """Advanced tool registry with execution management"""

    def __init__(self, max_concurrent_executions: int = 10):
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[ToolCategory, Set[str]] = defaultdict(set)
        self.execution_cache: Dict[str, ToolResult] = {}
        self.access_control: Dict[str, Set[str]] = defaultdict(set)

        # Execution management
        self.max_concurrent_executions = max_concurrent_executions
        self.active_executions: Dict[str, asyncio.Task[ToolResult]] = {}

        self.logger = logging.getLogger(__name__)

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default built-in tools"""
        default_tools = [CalculatorTool(), PythonREPLTool()]

        for tool in default_tools:
            # We directly add to storage to avoid side-effects during self-test
            self.tools[tool.name] = tool
            self.tool_categories[tool.category].add(tool.name)
            self.logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def register_tool(self, tool: Tool) -> bool:
        """Register a tool in the registry"""
        try:
            if tool.name in self.tools:
                self.logger.warning(f"Tool {tool.name} already exists, overwriting")

            self.tools[tool.name] = tool
            self.tool_categories[tool.category].add(tool.name)

            self.logger.info(f"Registered tool: {tool.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register tool {tool.name}: {e}")
            return False

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry"""
        if tool_name not in self.tools:
            return False

        tool = self.tools[tool_name]
        del self.tools[tool_name]
        self.tool_categories[tool.category].discard(tool_name)

        self.logger.info(f"Unregistered tool: {tool_name}")
        return True

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        security_level: Optional[ToolSecurityLevel] = None,
    ) -> List[Tool]:
        """List available tools"""
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if security_level:
            tools = [t for t in tools if t.security_level == security_level]

        return tools

    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        matches: List[Tool] = []

        for tool in self.tools.values():
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                matches.append(tool)

        return matches

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive tool information"""
        tool = self.get_tool(tool_name)
        if not tool:
            return None

        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "version": tool.version,
            "security_level": tool.security_level.value,
            "created_at": tool.created_at.isoformat(),
            "usage_count": tool.usage_count,
            "last_used": tool.last_used.isoformat() if tool.last_used else None,
            "parameters": [p.model_dump() for p in tool.get_parameters()],
            "performance_stats": tool.performance_stats,
        }

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """Execute a tool with given parameters"""
        with tracer.start_as_current_span("execute_tool") as span:
            span.set_attribute("tool.name", tool_name)

            tool = self.get_tool(tool_name)
            if not tool:
                return ToolResult(success=False, error=f"Tool '{tool_name}' not found")

            # Validate parameters
            validation_errors = tool.validate_parameters(parameters)
            if validation_errors:
                return ToolResult(
                    success=False,
                    error=f"Parameter validation failed: {'; '.join(validation_errors)}",
                )

            # Check execution limits
            if len(self.active_executions) >= self.max_concurrent_executions:
                return ToolResult(
                    success=False, error="Maximum concurrent executions reached"
                )

            # Create execution context if not provided
            if context is None:
                context = ToolExecutionContext()

            # Check cache
            cache_key = self._get_cache_key(tool_name, parameters, context)
            if cache_key in self.execution_cache:
                cached_result = self.execution_cache[cache_key]
                tool.performance_stats["cache_hits"] += 1
                return cached_result

            # Execute tool
            execution_id = context.execution_id

            try:
                task = asyncio.create_task(tool.execute(parameters, context))
                self.active_executions[execution_id] = task

                result = await task

                # Cache successful results
                if result.success:
                    self.execution_cache[cache_key] = result
                else:
                    tool.performance_stats["cache_misses"] += 1

                return result

            except Exception as e:
                return ToolResult(success=False, error=f"Execution failed: {str(e)}")

            finally:
                # Clean up
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]

    def _get_cache_key(
        self, tool_name: str, parameters: Dict[str, Any], context: ToolExecutionContext
    ) -> str:
        """Generate cache key for tool execution"""
        key_data = {
            "tool_name": tool_name,
            "parameters": parameters,
            "agent_id": context.agent_id,
            "session_id": context.session_id,
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))

    def get_categories(self) -> List[ToolCategory]:
        """Get all available categories"""
        return list(self.tool_categories.keys())

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get tools in a specific category"""
        tool_names = self.tool_categories.get(category, set())
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "tools_by_category": {
                cat.value: len(tools) for cat, tools in self.tool_categories.items()
            },
            "active_executions": len(self.active_executions),
            "cache_size": len(self.execution_cache),
        }

    def clear_cache(self) -> None:
        """Clear execution cache"""
        self.execution_cache.clear()
        self.logger.info("Cleared execution cache")

    def export_tools(self) -> Dict[str, Any]:
        """Export tool registry configuration"""
        return {
            "tools": {name: self.get_tool_info(name) for name in self.tools.keys()},
            "categories": {
                cat.value: list(tools) for cat, tools in self.tool_categories.items()
            },
            "stats": self.get_registry_stats(),
        }


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: Tool) -> bool:
    """Register tool in global registry"""
    return get_tool_registry().register_tool(tool)


def get_tool(tool_name: str) -> Optional[Tool]:
    """Get tool from global registry"""
    return get_tool_registry().get_tool(tool_name)


async def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    context: Optional[ToolExecutionContext] = None,
) -> ToolResult:
    """Execute tool from global registry"""
    return await get_tool_registry().execute_tool(tool_name, parameters, context)


__all__ = [
    "Tool",
    "ToolParameter",
    "ToolResult",
    "ToolRegistry",
    "ToolCategory",
    "ToolSecurityLevel",
    "ToolExecutionContext",
    "CalculatorTool",
    "PythonREPLTool",
    "get_tool_registry",
    "register_tool",
    "get_tool",
    "execute_tool",
]
