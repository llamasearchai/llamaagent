"""
Advanced Reasoning Module for LlamaAgent

This module implements cutting-edge reasoning patterns including:
- Tree of Thoughts (ToT) for deliberate problem solving
- Graph of Thoughts (GoT) for non-linear reasoning  
- Constitutional AI for ethical reasoning and self-critique
- Meta-Reasoning for adaptive strategy selection

Author: Advanced LlamaAgent Development Team
"""

from .graph_of_thoughts import GraphOfThoughtsAgent, ReasoningGraph, Concept
from .meta_reasoning import MetaCognitiveAgent, StrategySelector, ConfidenceSystem
from .tree_of_thoughts import TreeOfThoughtsAgent, ThoughtTree, ThoughtNode, SearchStrategy
from .constitutional_ai import ConstitutionalAgent, Constitution, CritiqueSystem
from .cognitive_agent import CognitiveAgent

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
