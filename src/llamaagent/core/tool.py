"""
Tool implementation for LlamaAgent
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Tool(BaseModel, ABC):
    """
    Base class for tools that agents can use
    """

    name: str
    description: str
    version: str = "0.1.0"
    keywords: List[str] = Field(default_factory=list)
    requires_api_key: bool = False
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def run(self, input: str, **kwargs) -> str:
        """
        Run the tool on the input

        Args:
            input: The input to the tool
            **kwargs: Additional parameters for execution

        Returns:
            str: The output of the tool
        """
        pass

    def _validate_api_key(self) -> None:
        """
        Validate that the API key is present if required
        """
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"Tool {self.name} requires an API key but none was provided"
            )


class FunctionTool(Tool):
    """
    Tool that wraps a function
    """

    func: Callable

    def run(self, input: str, **kwargs) -> str:
        """
        Run the function on the input
        """
        self._validate_api_key()

        logger.debug(f"Running function tool {self.name} with input: {input}")

        # Get function signature to determine how to call it
        sig = inspect.signature(self.func)

        if len(sig.parameters) == 0:
            # Function takes no arguments
            result = self.func()
        elif len(sig.parameters) == 1:
            # Function takes just the input string
            result = self.func(input)
        else:
            # Function takes input and possibly kwargs
            result = self.func(input, **kwargs)

        # Ensure result is a string
        if not isinstance(result, str):
            result = str(result)

        return result


class CommandTool(Tool):
    """
    Tool that runs a shell command
    """

    command_template: str

    def run(self, input: str, **kwargs) -> str:
        """
        Run a shell command with the input
        """
        import shlex
        import subprocess

        self._validate_api_key()

        # Format the command template with the input
        try:
            command = self.command_template.format(input=shlex.quote(input), **kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for command template: {e}")

        logger.debug(f"Running command tool {self.name}: {command}")

        # Run the command
        try:
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed with error: {e.stderr}")


class ApiTool(Tool):
    """
    Tool that makes API requests
    """

    base_url: str
    method: str = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    requires_api_key: bool = True

    def run(self, input: str, **kwargs) -> str:
        """
        Make an API request with the input
        """
        import requests

        self._validate_api_key()

        # Set up headers with API key if provided
        headers = self.headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url.rstrip('/')}/{input.lstrip('/')}"

        logger.debug(f"Making API request to {url} with method {self.method}")

        # Make the request
        try:
            if self.method.upper() == "GET":
                response = requests.get(url, headers=headers, params=kwargs)
            elif self.method.upper() == "POST":
                response = requests.post(url, headers=headers, json=kwargs)
            elif self.method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=kwargs)
            elif self.method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {self.method}")

            response.raise_for_status()

            # Return response as string
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            return response.text

        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")


class WebSearchTool(Tool):
    """
    Tool for web search
    """

    name: str = "web_search"
    description: str = "Search the web for information"
    keywords: List[str] = ["search", "find", "google", "look up", "web"]
    requires_api_key: bool = True
    search_engine: str = "google"  # google, bing, etc.

    def run(self, input: str, **kwargs) -> str:
        """
        Search the web for the input query
        """
        self._validate_api_key()

        logger.debug(f"Searching the web for: {input}")

        # Placeholder for actual implementation
        # In a real implementation, this would use a search API
        return (
            f"Web search results for '{input}': [Placeholder for actual search results]"
        )


class CodeInterpreterTool(Tool):
    """
    Tool for executing code
    """

    name: str = "code_interpreter"
    description: str = "Run code and return the output"
    keywords: List[str] = ["code", "execute", "run", "python", "calculate"]
    supported_languages: List[str] = ["python"]

    def run(self, input: str, **kwargs) -> str:
        """
        Execute code and return the result
        """
        import sys
        from io import StringIO

        logger.debug(f"Executing code: {input}")

        # Very basic implementation - in a real tool, this would need much more security
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        try:
            # Execute the code
            exec(input)
            result = redirected_output.getvalue()
            return result if result else "Code executed successfully with no output."
        except Exception as e:
            return f"Error executing code: {str(e)}"
        finally:
            sys.stdout = old_stdout
