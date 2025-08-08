"""
Advanced Reasoning Module for LlamaAgent

This module implements cutting-edge reasoning patterns including:
- Tree of Thoughts (ToT) for deliberate problem solving
- Graph of Thoughts (GoT) for non-linear reasoning
- Constitutional AI for ethical reasoning and self-critique
- Meta-Reasoning for adaptive strategy selection

Author: Advanced LlamaAgent Development Team
"""

from .cognitive_agent import CognitiveAgent
from .constitutional_ai import Constitution, ConstitutionalAgent, CritiqueSystem
from .graph_of_thoughts import Concept, GraphOfThoughtsAgent, ReasoningGraph
from .meta_reasoning import ConfidenceSystem, MetaCognitiveAgent, StrategySelector
from .tree_of_thoughts import SearchStrategy, ThoughtNode, ThoughtTree, TreeOfThoughtsAgent

__all__ = [
    # Tree of Thoughts
    "TreeOfThoughtsAgent",
    "ThoughtTree",
    "ThoughtNode",
    "SearchStrategy",
    # Graph of Thoughts
    "GraphOfThoughtsAgent",
    "ReasoningGraph",
    "Concept",
    # Constitutional AI
    "ConstitutionalAgent",
    "Constitution",
    "CritiqueSystem",
    # Meta-Reasoning
    "MetaCognitiveAgent",
    "StrategySelector",
    "ConfidenceSystem",
    # Unified Interface
    "CognitiveAgent",
]

# Version info
__version__ = "1.0.0"
__author__ = "LlamaAgent Advanced Reasoning Team"
