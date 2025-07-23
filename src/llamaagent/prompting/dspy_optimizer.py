"""
DSPy Integration for Advanced Prompt Optimization

Implements DSPy-style prompt optimization with automatic prompt engineering,
few-shot learning, and chain-of-thought reasoning.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..llm.messages import LLMMessage
from ..llm.providers.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class DSPySignature:
    """Defines the signature for a DSPy module"""

    name: str
    description: str
    input_fields: Dict[str, str]
    output_fields: Dict[str, str]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert signature to a prompt template"""
        prompt_parts = [
            f"Task: {self.name}",
            f"Description: {self.description}",
            "",
            "Input fields:",
        ]

        for field, description in self.input_fields.items():
            prompt_parts.append(f"- {field}: {description}")

        prompt_parts.extend(["", "Output fields:"])

        for field, description in self.output_fields.items():
            prompt_parts.append(f"- {field}: {description}")

        if self.constraints:
            prompt_parts.extend(["", "Constraints:"])
            for constraint in self.constraints:
                prompt_parts.append(f"- {constraint}")

        if self.examples:
            prompt_parts.extend(["", "Examples:"])
            for i, example in enumerate(self.examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append("Input:")
                for field, value in example.get("input", {}).items():
                    prompt_parts.append(f"{field}: {value}")
                prompt_parts.append("Output:")
                for field, value in example.get("output", {}).items():
                    prompt_parts.append(f"{field}: {value}")

        return "\n".join(prompt_parts)


@dataclass
class OptimizationResult:
    """Result from DSPy optimization"""

    best_prompt: str
    best_score: float
    best_examples: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DSPyModule(ABC):
    """Base class for DSPy modules"""

    def __init__(
        self, signature: DSPySignature, llm_provider: Optional[BaseLLMProvider] = None
    ):
        self.signature = signature
        self.llm_provider = llm_provider
        self.call_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def forward(self, **kwargs) -> Dict[str, Any]:
        """Forward pass through the module"""
        pass

    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the module callable"""
        result = await self.forward(**kwargs)
        self.call_history.append(
            {
                "input": kwargs,
                "output": result,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )
        return result

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        output = {}

        # Simple parsing - can be enhanced with more sophisticated methods
        lines = response_text.strip().split("\n")
        current_field = None

        for line in lines:
            line = line.strip()
            for field in self.signature.output_fields:
                if line.lower().startswith(f"{field}:"):
                    current_field = field
                    value = line[len(field) + 1 :].strip()
                    output[field] = value
                    break
            if current_field and line:
                # Continuation of previous field
                output[current_field] = output.get(current_field, "") + " " + line

        # Ensure all output fields are present
        for field in self.signature.output_fields:
            if field not in output:
                output[field] = ""

        return output


class ChainOfThought(DSPyModule):
    """Chain-of-thought reasoning module"""

    async def forward(self, **kwargs) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning"""
        # Build the prompt
        prompt_parts = [self.signature.to_prompt(), "", "Current input:"]

        for field, value in kwargs.items():
            if field in self.signature.input_fields:
                prompt_parts.append(f"{field}: {value}")

        prompt_parts.extend(["", "Let's think step by step:", "", "Output:"])

        prompt = "\n".join(prompt_parts)

        # Get LLM response
        if self.llm_provider:
            message = LLMMessage(role="user", content=prompt)
            response = await self.llm_provider.complete([message])

            # Parse the response
            return self._parse_response(response.content)

        # Fallback if no LLM provider
        return {
            field: f"Mock output for {field}" for field in self.signature.output_fields
        }


class Pipeline(DSPyModule):
    """Pipeline module for chaining multiple modules"""

    def __init__(self, signature: DSPySignature, modules: List[DSPyModule]):
        super().__init__(signature)
        self.modules = modules

    async def forward(self, **kwargs) -> Dict[str, Any]:
        """Execute pipeline forward pass"""
        current_output = kwargs

        for module in self.modules:
            current_output = await module(**current_output)

        return current_output


class DSPyOptimizer:
    """Main DSPy optimizer for automatic prompt engineering"""

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    async def compile(
        self,
        module: DSPyModule,
        training_data: List[Dict[str, Any]],
        metric: Callable[[Dict[str, Any], Dict[str, Any]], float],
        num_iterations: int = 5,
        bootstrap_few_shot: int = 3,
    ) -> OptimizationResult:
        """
        Compile/optimize a DSPy module using training data.

        Args:
            module: The DSPy module to optimize
            training_data: List of input/output examples
            metric: Function to evaluate predictions
            num_iterations: Number of optimization iterations
            bootstrap_few_shot: Number of examples for few-shot learning
        """
        best_score = -float("inf")
        best_examples = []
        best_prompt = module.signature.to_prompt()

        for iteration in range(num_iterations):
            # Bootstrap few-shot examples
            if iteration == 0:
                # Use provided examples
                few_shot_examples = training_data[:bootstrap_few_shot]
            else:
                # Use best performing examples from previous iterations
                few_shot_examples = self._select_best_examples(
                    module.call_history, bootstrap_few_shot
                )

            # Update module signature with few-shot examples
            module.signature.examples = few_shot_examples

            # Evaluate on validation set
            validation_scores = []
            predictions = []

            for example in training_data[bootstrap_few_shot:]:
                try:
                    # Make prediction
                    prediction = await module(**example["input"])
                    predictions.append(prediction)

                    # Calculate score
                    score = metric(prediction, example["output"])
                    validation_scores.append(score)

                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}")
                    validation_scores.append(0.0)

            # Calculate average score
            avg_score = (
                np.mean(validation_scores)
                if validation_scores and NUMPY_AVAILABLE
                else 0.0
            )

            # Update best if improved
            if avg_score > best_score:
                best_score = avg_score
                best_examples = few_shot_examples
                best_prompt = module.signature.to_prompt()

            # Adaptive example selection
            if iteration < num_iterations - 1:
                await self._adaptive_example_selection(module, training_data, metric)

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            best_examples=best_examples,
            metadata={
                "iterations": num_iterations,
                "bootstrap_size": bootstrap_few_shot,
                "training_size": len(training_data),
            },
        )

    def _select_best_examples(
        self, call_history: List[Dict[str, Any]], k: int = 3
    ) -> List[Dict[str, Any]]:
        """Select k best examples from call history"""
        if not call_history:
            return []

        # Score all examples
        scored_examples = []
        for call in call_history:
            if "input" in call and "output" in call:
                # Simple scoring based on output length and completeness
                score = len(str(call["output"]))
                scored_examples.append(
                    {"input": call["input"], "output": call["output"], "score": score}
                )

        # Sort by score and return top k
        scored_examples.sort(key=lambda x: x["score"], reverse=True)
        return [
            {"input": ex["input"], "output": ex["output"]} for ex in scored_examples[:k]
        ]

    async def _adaptive_example_selection(
        self, module: DSPyModule, training_data: List[Dict[str, Any]], metric: Callable
    ) -> None:
        """Adaptively select examples based on diversity and performance"""
        # Group examples by output patterns
        output_groups = defaultdict(list)
        for example in training_data:
            # Simple grouping by output length
            output_key = len(str(example.get("output", "")))
            output_groups[output_key].append(example)

        # Select diverse examples from different groups
        diverse_examples = []
        for group in output_groups.values():
            if group:
                diverse_examples.append(group[0])  # Take first from each group

        # Update module with diverse examples
        module.signature.examples = diverse_examples[:5]  # Limit to 5 examples


# Utility functions for common metrics
def exact_match_metric(prediction: Dict[str, Any], target: Dict[str, Any]) -> float:
    """Exact match metric for evaluation"""
    matches = 0
    total = 0

    for key in target:
        if key in prediction:
            total += 1
            if str(prediction[key]).strip() == str(target[key]).strip():
                matches += 1

    return matches / total if total > 0 else 0.0


def semantic_similarity_metric(
    prediction: Dict[str, Any], target: Dict[str, Any]
) -> float:
    """Semantic similarity metric (simplified version)"""
    # In production, this would use embeddings to compute similarity
    # For now, using simple token overlap
    pred_text = " ".join(str(v) for v in prediction.values()).lower()
    target_text = " ".join(str(v) for v in target.values()).lower()

    pred_tokens = set(pred_text.split())
    target_tokens = set(target_text.split())

    if not target_tokens:
        return 0.0

    overlap = len(pred_tokens & target_tokens)
    return overlap / len(target_tokens)


# Example usage and factory functions
def create_qa_module(llm_provider: BaseLLMProvider) -> ChainOfThought:
    """Create a question-answering module"""
    qa_signature = DSPySignature(
        name="Question Answering",
        description="Answer questions based on given context",
        input_fields={
            "context": "The context or background information",
            "question": "The question to answer",
        },
        output_fields={
            "answer": "The answer to the question",
            "confidence": "Confidence level (0-1)",
        },
        constraints=[
            "Answer must be based only on the given context",
            "If answer is not in context, state that clearly",
        ],
    )

    return ChainOfThought(qa_signature, llm_provider)


async def optimize_qa_module(
    llm_provider: BaseLLMProvider, training_data: List[Dict[str, Any]]
) -> OptimizationResult:
    """Optimize a question-answering module"""
    # Create module
    qa_module = create_qa_module(llm_provider)

    # Optimize
    optimizer = DSPyOptimizer(llm_provider)
    result = await optimizer.compile(
        module=qa_module,
        training_data=training_data,
        metric=exact_match_metric,
        num_iterations=3,
        bootstrap_few_shot=2,
    )

    return result


# Example training data format:
EXAMPLE_TRAINING_DATA = [
    {
        "input": {
            "context": "The sky is blue because of light scattering.",
            "question": "Why is the sky blue?",
        },
        "output": {
            "answer": "The sky is blue because of light scattering.",
            "confidence": "0.9",
        },
    },
    {
        "input": {
            "context": "Python is a programming language.",
            "question": "What is Python?",
        },
        "output": {"answer": "Python is a programming language.", "confidence": "1.0"},
    },
]
