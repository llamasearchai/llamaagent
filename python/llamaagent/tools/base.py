"""
Base class for tools that can be used by the agent.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from ..types import ToolResult

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Base class for all tools.

    Each tool must implement the 'run' method that executes the tool's
    functionality. Tools are used by the agent to interact with external
    systems and perform various tasks.

    Attributes:
        name (str): The name of the tool
        description (str): Description of what the tool does
        max_attempts (int): Maximum number of retry attempts
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_attempts: int = 3,
    ):
        """
        Initialize a tool.

        Args:
            name: The name of the tool (defaults to class name if None)
            description: Description of what the tool does (defaults to docstring if None)
            max_attempts: Maximum number of retry attempts
        """
        self.name = name or self.__class__.__name__
        self.description = (
            description or self.__class__.__doc__ or "No description provided"
        )
        self.max_attempts = max_attempts

        logger.debug(f"Initialized tool: {self.name}")

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the tool's core functionality.

        This method must be implemented by subclasses to provide the
        tool's specific functionality.

        Args:
            *args: Positional arguments to the tool
            **kwargs: Keyword arguments to the tool

        Returns:
            The result of running the tool
        """
        pass

    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """
        Run the tool with error handling and retries.

        This method wraps the '_run' method to provide consistent error
        handling, logging, and retry logic.

        Args:
            *args: Positional arguments to the tool
            **kwargs: Keyword arguments to the tool

        Returns:
            A ToolResult object containing the result or error
        """
        start_time = time.time()
        logger.info(f"Running tool: {self.name}")

        attempt = 0
        last_error = None

        while attempt < self.max_attempts:
            attempt += 1

            try:
                # Run the tool
                result = self._run(*args, **kwargs)

                # Convert result to string if needed
                if result is None:
                    observation = "No result returned"
                elif not isinstance(result, str):
                    observation = str(result)
                else:
                    observation = result

                # Create successful result
                tool_result = ToolResult(
                    tool_name=self.name,
                    observation=observation,
                    success=True,
                    error=None,
                    metadata={
                        "duration": time.time() - start_time,
                        "attempts": attempt,
                        "args": args,
                        "kwargs": kwargs,
                    },
                )

                logger.info(
                    f"Tool {self.name} completed successfully in {attempt} attempts"
                )
                return tool_result

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Tool {self.name} failed on attempt {attempt}/{self.max_attempts}: {e}"
                )

                # If we have more attempts, retry after a short delay
                if attempt < self.max_attempts:
                    delay = 2**attempt  # Exponential backoff
                    logger.debug(f"Retrying after {delay} seconds")
                    time.sleep(delay)

        # If we get here, all attempts failed
        logger.error(f"Tool {self.name} failed after {self.max_attempts} attempts")

        return ToolResult(
            tool_name=self.name,
            observation=f"Tool failed after {self.max_attempts} attempts: {last_error}",
            success=False,
            error=last_error,
            metadata={
                "duration": time.time() - start_time,
                "attempts": attempt,
                "args": args,
                "kwargs": kwargs,
            },
        )
