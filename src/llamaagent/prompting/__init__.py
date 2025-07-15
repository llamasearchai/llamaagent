"""
Advanced Prompting Techniques for LlamaAgent

This module implements cutting-edge prompting strategies including:
- DSPy optimization
- Chain-of-thought prompting
- Compound prompting
- Self-consistency
- Tree of thoughts
- Reflexion
"""

from .chain_prompting import ChainOfThoughtPrompt, ChainPromptOptimizer
from .compound_prompting import CompoundPromptStrategy, PromptComposer
from .dspy_optimizer import DSPyOptimizer, DSPySignature
from .optimization import OptimizationMetrics, PromptOptimizer
from .prompt_templates import PromptLibrary, PromptTemplate

__all__ = [
    "DSPyOptimizer",
    "DSPySignature",
    "ChainOfThoughtPrompt",
    "ChainPromptOptimizer",
    "CompoundPromptStrategy",
    "PromptComposer",
    "PromptTemplate",
    "PromptLibrary",
    "PromptOptimizer",
    "OptimizationMetrics",
]
