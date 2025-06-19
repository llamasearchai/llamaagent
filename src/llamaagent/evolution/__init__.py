"""Evolution and learning modules for LlamaAgent."""

from __future__ import annotations

from .knowledge_base import CooperationKnowledge
from .orchestrator import CurriculumOrchestrator
from .reflection import ReflectionModule

__all__ = [
    "CooperationKnowledge",
    "CurriculumOrchestrator",
    "ReflectionModule",
]
