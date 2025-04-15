"""
Workflow implementation for LlamaAgent
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Step(BaseModel):
    """
    A single step in a workflow
    """

    name: str
    tool: str
    description: Optional[str] = None
    optional: bool = False
    timeout: Optional[int] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"Step '{self.name}' using tool '{self.tool}'"


class Workflow(BaseModel):
    """
    A workflow consisting of a sequence of steps
    """

    name: str
    description: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Step] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def add_step(self, step: Step) -> None:
        """
        Add a step to the workflow
        """
        self.steps.append(step)

    def remove_step(self, step_name: str) -> bool:
        """
        Remove a step from the workflow by name

        Returns:
            bool: True if the step was removed, False if not found
        """
        initial_length = len(self.steps)
        self.steps = [step for step in self.steps if step.name != step_name]
        return len(self.steps) < initial_length

    def get_step(self, step_name: str) -> Optional[Step]:
        """
        Get a step by name

        Returns:
            Optional[Step]: The step if found, None otherwise
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None

    def reorder_steps(self, step_names: List[str]) -> bool:
        """
        Reorder steps based on a list of step names

        Args:
            step_names: List of step names in the desired order

        Returns:
            bool: True if successful, False if any steps were not found
        """
        # Check if all step names are valid
        current_names = {step.name for step in self.steps}
        if set(step_names) != current_names:
            logger.error(f"Cannot reorder: step names don't match existing steps")
            return False

        # Create a mapping of name to step
        steps_map = {step.name: step for step in self.steps}

        # Reorder steps
        try:
            self.steps = [steps_map[name] for name in step_names]
            return True
        except KeyError as e:
            logger.error(f"Error reordering steps: {e}")
            return False

    def __str__(self) -> str:
        result = f"Workflow '{self.name}'"
        if self.description:
            result += f": {self.description}"
        result += f" ({len(self.steps)} steps)"
        return result


class WorkflowFactory:
    """
    Factory class for creating common workflow patterns
    """

    @staticmethod
    def create_empty(name: str, description: Optional[str] = None) -> Workflow:
        """
        Create an empty workflow
        """
        return Workflow(name=name, description=description)

    @staticmethod
    def create_research_workflow() -> Workflow:
        """
        Create a basic research workflow
        """
        workflow = Workflow(
            name="research",
            description="Research a topic and provide a summary",
            keywords=[
                "research",
                "study",
                "investigate",
                "learn about",
                "find information",
            ],
        )

        workflow.add_step(
            Step(
                name="search",
                tool="web_search",
                description="Search for information on the topic",
            )
        )

        workflow.add_step(
            Step(
                name="analyze",
                tool="text_analyzer",
                description="Analyze the search results",
            )
        )

        workflow.add_step(
            Step(
                name="summarize",
                tool="summarizer",
                description="Create a summary of the findings",
            )
        )

        return workflow

    @staticmethod
    def create_code_workflow() -> Workflow:
        """
        Create a basic code generation workflow
        """
        workflow = Workflow(
            name="code_generator",
            description="Generate and test code",
            keywords=["code", "program", "script", "function", "implement"],
        )

        workflow.add_step(
            Step(
                name="design",
                tool="code_designer",
                description="Design the code structure",
            )
        )

        workflow.add_step(
            Step(name="implement", tool="code_writer", description="Implement the code")
        )

        workflow.add_step(
            Step(name="test", tool="code_tester", description="Test the code")
        )

        return workflow

    @staticmethod
    def create_data_analysis_workflow() -> Workflow:
        """
        Create a data analysis workflow
        """
        workflow = Workflow(
            name="data_analysis",
            description="Analyze data and generate insights",
            keywords=[
                "analyze data",
                "data analysis",
                "stats",
                "statistics",
                "dataset",
            ],
        )

        workflow.add_step(
            Step(
                name="prepare",
                tool="data_cleaner",
                description="Clean and prepare the data",
            )
        )

        workflow.add_step(
            Step(name="analyze", tool="data_analyzer", description="Analyze the data")
        )

        workflow.add_step(
            Step(
                name="visualize",
                tool="data_visualizer",
                description="Create visualizations",
            )
        )

        workflow.add_step(
            Step(
                name="summarize",
                tool="insight_generator",
                description="Generate insights from the analysis",
            )
        )

        return workflow
