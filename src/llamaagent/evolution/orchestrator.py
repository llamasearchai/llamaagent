from __future__ import annotations

from typing import List

from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent
from ..llm import MockProvider


class CurriculumOrchestrator:
    """Generates challenging collaboration scenarios."""

    ORCHESTRATOR_PROMPT = """You are an AI curriculum designer. Your goal is to create a challenging problem for a team of AI agents to solve.

The problem must:
1. Require at least two agents to collaborate
2. Test a specific collaborative dynamic
3. Have clear success criteria

Types of collaborative challenges:
- Misinformation correction (one agent provides false info that must be caught)
- Ambiguity resolution (unclear requirements need discussion)
- Resource conflicts (agents must negotiate shared resources)
- Dynamic re-planning (goals change mid-task)
- Information synthesis (agents have partial information)

Create a problem that tests one of these dynamics. Include:
1. The task description
2. The specific collaborative challenge
3. Success criteria
4. Any special constraints

Format as:
TASK: [clear task description]
CHALLENGE: [specific collaborative dynamic being tested]
SUCCESS: [how to measure success]
CONSTRAINTS: [any special rules or limitations]"""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider or MockProvider()

        self.orchestrator = ReactAgent(
            config=AgentConfig(
                name="CurriculumOrchestrator",
                role=AgentRole.ORCHESTRATOR,
                description="Creates collaborative challenges",
            ),
            llm_provider=self.llm,
        )

    async def generate_curriculum_task(self, focus_area: str = "general") -> dict:
        """Generate a curriculum task focused on a specific area."""
        prompt = f"{self.ORCHESTRATOR_PROMPT}\n\nFocus area: {focus_area}"

        response = await self.orchestrator.execute(prompt)

        # Parse the response
        content = response.content
        task_data = self._parse_task_response(content)

        return {
            "task": task_data.get("task", "Collaborate to solve a complex problem"),
            "challenge": task_data.get("challenge", "General collaboration"),
            "success_criteria": task_data.get("success", "Task completed successfully"),
            "constraints": task_data.get("constraints", "None"),
            "focus_area": focus_area,
        }

    def _parse_task_response(self, content: str) -> dict:
        """Parse the orchestrator's response."""
        result = {}
        # Split into sections and parse each one
        sections = content.split("##")
        for section in sections:
            if not section.strip():
                continue
            self._parse_section(section.strip(), result)
        return result

    def _parse_section(self, section: str, result: dict):
        """Parse a single section of the response."""
        if section.startswith("Task:"):
            result["task"] = section.replace("Task:", "").strip()
        elif section.startswith("Sub-tasks:"):
            result["sub_tasks"] = self._parse_list(section.replace("Sub-tasks:", ""))
        elif section.startswith("Dependencies:"):
            result["dependencies"] = self._parse_list(section.replace("Dependencies:", ""))
        elif section.startswith("Agent Assignments:"):
            result["agent_assignments"] = self._parse_assignments(section.replace("Agent Assignments:", ""))

    def _parse_list(self, content: str) -> list:
        """Parse a list from markdown format."""
        return [line.strip()[2:] for line in content.split("\n") if line.strip().startswith("-")]

    def _parse_assignments(self, content: str) -> dict:
        """Parse agent assignments."""
        assignments = {}
        for line in content.split("\n"):
            if ":" in line:
                task, agent = line.split(":", 1)
                assignments[task.strip()] = agent.strip()
        return assignments

    async def generate_curriculum_suite(self, num_tasks: int = 10) -> List[dict]:
        """Generate a suite of curriculum tasks."""
        focus_areas = [
            "misinformation_correction",
            "ambiguity_resolution",
            "resource_conflicts",
            "dynamic_replanning",
            "information_synthesis",
        ]

        tasks = []
        for i in range(num_tasks):
            focus = focus_areas[i % len(focus_areas)]
            task = await self.generate_curriculum_task(focus)
            tasks.append(task)

        return tasks
