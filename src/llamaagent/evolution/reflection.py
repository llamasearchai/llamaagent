"""
Reflection module for analyzing multi-agent interactions and extracting insights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..agents.base import AgentConfig, AgentRole
from ..agents.react import ReactAgent
from ..llm import MockProvider
from ..llm.base import LLMProvider


class ReflectionModule:
    """Analyzes multi-agent interactions and extracts insights."""

    REFLECTION_PROMPT = """You are an expert in multi-agent systems analysis. Analyze the following transcript of an agent team's attempt to solve a task.

Your goal is to identify the single most important reason for their success or failure and distill this into a concise, actionable heuristic for future collaboration.

The heuristic should be:
1. An "if-then" statement or best practice
2. Specific enough to be actionable
3. General enough to apply to similar situations
4. Based on observable agent behaviors

Examples of good heuristics:
- "If the Researcher provides information from an untrusted source, the Critic must verify before other agents use it"
- "When the Coordinator identifies independent subtasks, delegate them to different agents in parallel"
- "If agents provide conflicting information, pause execution and resolve the conflict before proceeding"

Format your response as:
INSIGHT: [single actionable heuristic]
REASONING: [brief explanation of why this insight matters]"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        """Initialize the reflection module."""
        self.llm = llm_provider or MockProvider()

        self.reflector = ReactAgent(
            config=AgentConfig(
                name="Reflector",
                role=AgentRole.CRITIC,
                description="Analyzes team performance",
            ),
            llm_provider=self.llm,
        )

    async def analyze_interaction(
        self, task: str, transcript: List[Dict[str, Any]], success: bool
    ) -> str:
        """Analyze a multi-agent interaction and extract insight."""
        # Format transcript for analysis
        formatted_transcript = self._format_transcript(transcript)

        # Create analysis prompt
        analysis_prompt = f"""{self.REFLECTION_PROMPT}

Task: {task}
Success: {"Yes" if success else "No"}

Transcript:
{formatted_transcript}"""

        response = await self.reflector.execute(analysis_prompt)

        # Extract insight
        insight = self._parse_insight(response.content)
        return insight

    def _format_transcript(self, transcript: List[Dict[str, Any]]) -> str:
        """Format transcript for analysis."""
        formatted: List[str] = []

        for i, message in enumerate(transcript, 1):
            sender = message.get("sender", "Unknown")
            content = message.get("content", "")
            role = message.get("role", "")

            if role:
                formatted.append(f"{i}. [{sender}] ({role}): {content}")
            else:
                formatted.append(f"{i}. {sender}: {content}")

        return "\n".join(formatted)

    def _parse_insight(self, content: str) -> str:
        """Parse insight from reflection response."""
        lines = content.split("\n")

        for line in lines:
            if line.startswith("INSIGHT:"):
                return line[8:].strip()

        # Fallback - return first meaningful line
        for line in lines:
            line = line.strip()
            if line and not line.startswith("REASONING:"):
                return line

        return "No clear insight extracted"

    async def batch_analyze(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Analyze multiple interactions and extract insights."""
        insights: List[str] = []

        for interaction in interactions:
            try:
                insight = await self.analyze_interaction(
                    task=interaction["task"],
                    transcript=interaction["transcript"],
                    success=interaction["success"],
                )
                insights.append(insight)
            except Exception as e:
                print(f"Error analyzing interaction: {e}")
                continue

        return insights

    def filter_insights(self, insights: List[str]) -> List[str]:
        """Filter and deduplicate insights."""
        # Simple deduplication based on similarity
        filtered: List[str] = []

        for insight in insights:
            # Skip very short or generic insights
            if len(insight) < 20:
                continue

            # Check for duplicates (simple string similarity)
            is_duplicate = False
            for existing in filtered:
                if self._similarity(insight, existing) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(insight)

        return filtered

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)
