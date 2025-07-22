#!/usr/bin/env python3
"""
Function Calling System with OpenAI Compatibility
Author: Nik Jois <nikjois@llamasearch.ai>

This module provides:
- OpenAI-compatible function definitions
- Built-in utility functions
- Custom function registration
- Function execution and validation
- Dynamic function discovery
"""

import asyncio
import inspect
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import requests

    requests_available = True
except ImportError:
    requests_available = False

logger = logging.getLogger(__name__)


@dataclass
class FunctionParameter:
    """Function parameter definition."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class FunctionDefinition:
    """OpenAI-compatible function definition."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    function: Optional[Callable] = None


class FunctionManager:
    """Manage and execute functions dynamically."""

    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.function_metadata: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []

        # Register built-in functions
        self._register_builtin_functions()

    def register_function(
        self, name: str, func: Callable, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a function for execution."""
        self.functions[name] = func
        self.function_metadata[name] = metadata or {}

        # Auto-extract metadata from function
        sig = inspect.signature(func)
        self.function_metadata[name].update(
            {
                "name": name,
                "description": func.__doc__ or f"Execute {name}",
                "parameters": self._extract_parameters(sig),
            }
        )

        logger.info(f"Registered function: {name}")

    def _extract_parameters(self, signature: inspect.Signature) -> Dict[str, Any]:
        """Extract parameters from function signature for OpenAI schema."""
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name in ['self', 'cls']:
                continue

            prop_def = {
                "type": self._get_type_string(param.annotation),
                "description": f"Parameter {param_name}",
            }

            # Handle default values
            if param.default != inspect.Parameter.empty:
                prop_def["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop_def

        return {"type": "object", "properties": properties, "required": required}

    def _get_type_string(self, annotation: Any) -> str:
        """Convert Python type annotation to JSON schema type."""
        if annotation == inspect.Parameter.empty:
            return "string"

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle Union types (e.g., Optional[str])
        if hasattr(annotation, '__origin__'):
            if annotation.__origin__ is Union:
                # For Optional types, use the non-None type
                non_none_types = [
                    arg for arg in annotation.__args__ if arg is not type(None)
                ]
                if non_none_types:
                    return self._get_type_string(non_none_types[0])

        return type_mapping.get(annotation, "string")

    def list_functions(self) -> List[str]:
        """List all registered functions."""
        return list(self.functions.keys())

    def get_function_info(self, name: str) -> Dict[str, Any]:
        """Get information about a function."""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found")

        return self.function_metadata[name]

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function schemas."""
        schemas = []
        for func_name, metadata in self.function_metadata.items():
            schema = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": metadata.get("description", ""),
                    "parameters": metadata.get("parameters", {}),
                },
            }
            schemas.append(schema)
        return schemas

    def execute_function(self, name: str, **kwargs) -> Any:
        """Execute a registered function."""
        if name not in self.functions:
            raise ValueError(f"Function '{name}' not found")

        func = self.functions[name]
        start_time = datetime.now()

        try:
            # Validate arguments
            validated_args = self._validate_arguments(name, kwargs)

            # Execute function
            if inspect.iscoroutinefunction(func):
                # Handle async functions
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(func(**validated_args))
                except RuntimeError:
                    # No event loop running
                    result = asyncio.run(func(**validated_args))
            else:
                result = func(**validated_args)

            # Log execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_history.append(
                {
                    "function": name,
                    "arguments": kwargs,
                    "result": str(result)[:100],  # Truncate for logging
                    "success": True,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                }
            )

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            # Log error
            self.execution_history.append(
                {
                    "function": name,
                    "arguments": kwargs,
                    "error": error_msg,
                    "success": False,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                }
            )

            logger.error(f"Error executing function {name}: {e}")
            raise

    def _validate_arguments(
        self, name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate function arguments against schema."""
        metadata = self.function_metadata[name]
        parameters = metadata.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        validated = {}

        # Check required parameters
        for req_param in required:
            if req_param not in arguments:
                raise ValueError(f"Missing required parameter: {req_param}")

        # Validate and convert parameters
        for param_name, value in arguments.items():
            if param_name not in properties:
                logger.warning(f"Unknown parameter {param_name} for function {name}")
                validated[param_name] = value
                continue

            param_schema = properties[param_name]

            # Type validation (basic)
            expected_type = param_schema.get("type", "string")
            if expected_type == "integer" and not isinstance(value, int):
                try:
                    validated[param_name] = int(value)
                except ValueError:
                    raise ValueError(f"Parameter {param_name} must be an integer")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                try:
                    validated[param_name] = float(value)
                except ValueError:
                    raise ValueError(f"Parameter {param_name} must be a number")
            elif expected_type == "boolean" and not isinstance(value, bool):
                if isinstance(value, str):
                    validated[param_name] = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    validated[param_name] = bool(value)
            else:
                validated[param_name] = value

            # Enum validation
            if "enum" in param_schema:
                if value not in param_schema["enum"]:
                    raise ValueError(
                        f"Invalid value for {param_name}. Must be one of: {param_schema['enum']}"
                    )

        # Add default values for missing optional parameters
        for param_name, param_schema in properties.items():
            if param_name not in validated and "default" in param_schema:
                validated[param_name] = param_schema["default"]

        return validated

    def _register_builtin_functions(self) -> None:
        """Register built-in utility functions."""

        # File system functions
        self.register_function(
            name="read_file",
            func=self._read_file,
            metadata={
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        )

        self.register_function(
            name="write_file",
            func=self._write_file,
            metadata={
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        )

        self.register_function(
            name="execute_shell",
            func=self._execute_shell,
            metadata={
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
            },
        )

        self.register_function(
            name="get_current_time",
            func=self._get_current_time,
            metadata={
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "Time format string",
                            "default": "%Y-%m-%d %H:%M:%S",
                        }
                    },
                    "required": [],
                },
            },
        )

        if requests_available:
            self.register_function(
                name="http_request",
                func=self._http_request,
                metadata={
                    "description": "Make an HTTP request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to request"},
                            "method": {
                                "type": "string",
                                "description": "HTTP method",
                                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                                "default": "GET",
                            },
                            "headers": {
                                "type": "object",
                                "description": "HTTP headers",
                            },
                            "data": {"type": "object", "description": "Request data"},
                        },
                        "required": ["url"],
                    },
                },
            )

    # Built-in function implementations
    def _read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            return path.read_text(encoding='utf-8')
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

    def _write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise Exception(f"Error writing file {file_path}: {e}")

    def _execute_shell(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command,
            }
        except subprocess.TimeoutExpired:
            raise Exception(f"Command timed out after {timeout} seconds: {command}")
        except Exception as e:
            raise Exception(f"Error executing command: {e}")

    def _get_current_time(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get the current date and time."""
        return datetime.now().strftime(format)

    def _http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request."""
        if not requests_available:
            raise Exception("requests library not available")

        try:
            response = requests.request(
                method=method, url=url, headers=headers, json=data, timeout=30
            )

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "text": response.text,
                "url": response.url,
            }
        except Exception as e:
            raise Exception(f"HTTP request failed: {e}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get function execution history."""
        return self.execution_history

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()

    def export_functions(self, export_path: str) -> None:
        """Export function definitions to JSON file."""
        export_data = {
            "functions": self.get_function_schemas(),
            "metadata": self.function_metadata,
            "exported_at": datetime.now().isoformat(),
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

    def search_functions(self, query: str) -> List[str]:
        """Search functions by name or description."""
        query = query.lower()
        matches = []

        for name, metadata in self.function_metadata.items():
            if (
                query in name.lower()
                or query in metadata.get("description", "").lower()
            ):
                matches.append(name)

        return matches


def create_function_from_schema(schema: Dict[str, Any]) -> FunctionDefinition:
    """Create a function definition from OpenAI schema."""
    func_info = schema.get("function", {})

    return FunctionDefinition(
        name=func_info.get("name", ""),
        description=func_info.get("description", ""),
        parameters=func_info.get("parameters", {}),
    )


# Example custom functions
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "current_directory": str(Path.cwd()),
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    """Example usage of the function manager."""
    manager = FunctionManager()

    # Register custom functions
    manager.register_function("calculate_sum", calculate_sum)
    manager.register_function("get_system_info", get_system_info)

    # List all functions
    print("Available functions:")
    for func_name in manager.list_functions():
        info = manager.get_function_info(func_name)
        print(f"- {func_name}: {info.get('description', 'No description')}")

    # Test function execution
    print("\nTesting functions:")

    # Test calculation
    result = manager.execute_function("calculate_sum", a=5, b=3)
    print(f"Sum: {result}")

    # Test current time
    time_result = manager.execute_function("get_current_time")
    print(f"Current time: {time_result}")

    # Test system info
    sys_info = manager.execute_function("get_system_info")
    print(f"System info: {sys_info}")

    # Show execution history
    print("\nExecution history:")
    for entry in manager.get_execution_history():
        status = "PASS" if entry["success"] else "FAIL"
        print(f"{status} {entry['function']} ({entry['execution_time']:.3f}s)")


if __name__ == "__main__":
    main()
