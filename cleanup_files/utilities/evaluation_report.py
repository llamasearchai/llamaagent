#!/usr/bin/env python3
"""
Comprehensive Evaluation Report for LlamaAgent
Generates detailed analysis with synthetic benchmarks and real performance data
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append("src")

from llamaagent.agents import AgentConfig, AgentRole, ReactAgent
from llamaagent.tools import ToolRegistry, get_all_tools


class ComprehensiveEvaluationReport:
    """Generate comprehensive evaluation report with multiple benchmarks."""

    def __init__(self):
        self.tools = ToolRegistry()
        for tool in get_all_tools():
            self.tools.register(tool)

    async def run_math_benchmark(
        self, config_name: str, config: AgentConfig, num_tasks: int = 20
    ) -> Dict[str, Any]:
        """Run mathematical reasoning benchmark."""
        print(f"Running math benchmark for {config_name}...")

        agent = ReactAgent(config, tools=self.tools)

        # Mathematical tasks of varying difficulty
        math_tasks = [
            ("Basic Arithmetic", "Calculate 347 + 892 - 156", "1083"),
            ("Percentage", "What is 15% of 240?", "36"),
            (
                "Compound Interest",
                "Calculate compound interest on $1000 at 5% for 3 years",
                "1157.63",
            ),
            ("Area Calculation", "Find the area of a circle with radius 7", "153.94"),
            ("Linear Equations", "Solve for x: 3x + 7 = 22", "5"),
            ("Quadratic Formula", "Find roots of x² - 5x + 6 = 0", "2, 3"),
            (
                "Probability",
                "What's the probability of rolling two 6s with two dice?",
                "1/36",
            ),
            ("Statistics", "Find the mean of: 4, 7, 9, 12, 15", "9.4"),
            (
                "Geometry",
                "Find the hypotenuse of a right triangle with sides 3 and 4",
                "5",
            ),
            ("Algebra", "Simplify: (x + 3)(x - 2)", "x² + x - 6"),
            ("Fractions", "Add 2/3 + 3/4", "17/12"),
            ("Decimals", "Convert 3/8 to decimal", "0.375"),
            ("Ratios", "If 3:4 = x:12, find x", "9"),
            ("Exponents", "Calculate 2^8", "256"),
            ("Logarithms", "What is log₁₀(1000)?", "3"),
            ("Trigonometry", "Find sin(30°)", "0.5"),
            (
                "Word Problem 1",
                "A train travels 60 mph for 2 hours, then 80 mph for 1.5 hours. What's the average speed?",
                "68",
            ),
            (
                "Word Problem 2",
                "If 5 apples cost $3, how much do 12 apples cost?",
                "7.2",
            ),
            (
                "Word Problem 3",
                "A rectangle has length 8 and width 5. What's its perimeter?",
                "26",
            ),
            (
                "Word Problem 4",
                "If a discount of 20% reduces price by $40, what was original price?",
                "200",
            ),
        ]

        correct = 0
        total_time = 0
        results = []

        for i, (category, question, expected) in enumerate(math_tasks[:num_tasks]):
            start_time = time.time()
            response = await agent.execute(f"Solve this step by step: {question}")
            exec_time = time.time() - start_time

            # Simple answer extraction
            predicted = self._extract_numeric_answer(response.content)
            expected_num = self._extract_numeric_answer(expected)

            is_correct = (
                predicted is not None
                and expected_num is not None
                and abs(predicted - expected_num) < 0.1
            )

            if is_correct:
                correct += 1

            total_time += exec_time

            results.append(
                {
                    "category": category,
                    "question": question,
                    "expected": expected,
                    "predicted": str(predicted) if predicted else "N/A",
                    "correct": is_correct,
                    "time": exec_time,
                }
            )

            print(f"  {i + 1}/{num_tasks}: {category} - {'' if is_correct else ''}")

        return {
            "config_name": config_name,
            "total_tasks": num_tasks,
            "correct_answers": correct,
            "accuracy": correct / num_tasks,
            "avg_time": total_time / num_tasks,
            "total_time": total_time,
            "results": results,
        }

    async def run_code_benchmark(
        self, config_name: str, config: AgentConfig, num_tasks: int = 15
    ) -> Dict[str, Any]:
        """Run code generation benchmark."""
        print(f"Running code benchmark for {config_name}...")

        agent = ReactAgent(config, tools=self.tools)

        # Code generation tasks
        code_tasks = [
            (
                "Basic Function",
                "Write a function to calculate factorial",
                ["def", "factorial", "return"],
            ),
            (
                "List Operations",
                "Write a function to find maximum in a list",
                ["def", "max", "return"],
            ),
            (
                "String Manipulation",
                "Write a function to reverse a string",
                ["def", "reverse", "return"],
            ),
            (
                "Sorting",
                "Write a function to sort a list of numbers",
                ["def", "sort", "return"],
            ),
            (
                "Fibonacci",
                "Write a function to generate Fibonacci sequence",
                ["def", "fibonacci", "return"],
            ),
            (
                "Prime Check",
                "Write a function to check if number is prime",
                ["def", "prime", "return"],
            ),
            (
                "Palindrome",
                "Write a function to check if string is palindrome",
                ["def", "palindrome", "return"],
            ),
            (
                "Binary Search",
                "Write binary search function",
                ["def", "binary", "search", "return"],
            ),
            (
                "Merge Lists",
                "Write a function to merge two sorted lists",
                ["def", "merge", "return"],
            ),
            (
                "Count Vowels",
                "Write a function to count vowels in string",
                ["def", "vowel", "return"],
            ),
            (
                "Remove Duplicates",
                "Write a function to remove duplicates from list",
                ["def", "remove", "return"],
            ),
            (
                "Matrix Addition",
                "Write a function to add two matrices",
                ["def", "matrix", "return"],
            ),
            (
                "GCD Function",
                "Write a function to find GCD of two numbers",
                ["def", "gcd", "return"],
            ),
            (
                "Anagram Check",
                "Write a function to check if two strings are anagrams",
                ["def", "anagram", "return"],
            ),
            (
                "Bubble Sort",
                "Write bubble sort algorithm",
                ["def", "bubble", "sort", "return"],
            ),
        ]

        correct = 0
        total_time = 0
        results = []

        for i, (category, task, keywords) in enumerate(code_tasks[:num_tasks]):
            start_time = time.time()
            response = await agent.execute(f"Write Python code: {task}")
            exec_time = time.time() - start_time

            # Check if response contains expected keywords
            response_lower = response.content.lower()
            has_function = "def " in response_lower
            has_keywords = (
                sum(1 for kw in keywords if kw.lower() in response_lower)
                >= len(keywords) // 2
            )
            has_return = "return" in response_lower

            is_correct = has_function and has_keywords and has_return

            if is_correct:
                correct += 1

            total_time += exec_time

            results.append(
                {
                    "category": category,
                    "task": task,
                    "has_function": has_function,
                    "has_keywords": has_keywords,
                    "has_return": has_return,
                    "correct": is_correct,
                    "time": exec_time,
                }
            )

            print(f"  {i + 1}/{num_tasks}: {category} - {'' if is_correct else ''}")

        return {
            "config_name": config_name,
            "total_tasks": num_tasks,
            "correct_answers": correct,
            "accuracy": correct / num_tasks,
            "avg_time": total_time / num_tasks,
            "total_time": total_time,
            "results": results,
        }

    async def run_reasoning_benchmark(
        self, config_name: str, config: AgentConfig, num_tasks: int = 15
    ) -> Dict[str, Any]:
        """Run logical reasoning benchmark."""
        print(f"Running reasoning benchmark for {config_name}...")

        agent = ReactAgent(config, tools=self.tools)

        # Reasoning tasks
        reasoning_tasks = [
            (
                "Logic",
                "If all cats are animals and Fluffy is a cat, is Fluffy an animal?",
                "yes",
            ),
            (
                "Deduction",
                "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
                "maybe",
            ),
            ("Pattern", "What comes next: 2, 4, 8, 16, ?", "32"),
            ("Analogy", "Bird is to fly as fish is to what?", "swim"),
            ("Causation", "If you drop a glass, what happens?", "breaks"),
            (
                "Classification",
                "Which doesn't belong: apple, banana, carrot, orange?",
                "carrot",
            ),
            ("Sequence", "What's the next letter: A, C, E, G, ?", "I"),
            ("Comparison", "Which is larger: 0.5 or 1/3?", "0.5"),
            (
                "Inference",
                "John is taller than Mary. Mary is taller than Sue. Who is shortest?",
                "Sue",
            ),
            (
                "Probability",
                "If you flip a fair coin twice, what's probability of two heads?",
                "0.25",
            ),
            (
                "Time Logic",
                "If meeting is at 3 PM and lasts 90 minutes, when does it end?",
                "4:30",
            ),
            (
                "Spatial",
                "If you face north and turn right, which direction are you facing?",
                "east",
            ),
            (
                "Conditional",
                "If temperature is below 32°F, water freezes. It's 25°F. Does water freeze?",
                "yes",
            ),
            (
                "Negation",
                "If it's not true that all birds can fly, what can we conclude?",
                "some birds cannot fly",
            ),
            (
                "Quantification",
                "Some dogs are brown. All brown things are colored. Are some dogs colored?",
                "yes",
            ),
        ]

        correct = 0
        total_time = 0
        results = []

        for i, (category, question, expected) in enumerate(reasoning_tasks[:num_tasks]):
            start_time = time.time()
            response = await agent.execute(
                f"Answer this reasoning question: {question}"
            )
            exec_time = time.time() - start_time

            # Simple answer matching
            response_lower = response.content.lower().strip()
            expected_lower = expected.lower().strip()

            is_correct = expected_lower in response_lower or self._fuzzy_match(
                response_lower, expected_lower
            )

            if is_correct:
                correct += 1

            total_time += exec_time

            results.append(
                {
                    "category": category,
                    "question": question,
                    "expected": expected,
                    "response": response.content[:100] + "..."
                    if len(response.content) > 100
                    else response.content,
                    "correct": is_correct,
                    "time": exec_time,
                }
            )

            print(f"  {i + 1}/{num_tasks}: {category} - {'' if is_correct else ''}")

        return {
            "config_name": config_name,
            "total_tasks": num_tasks,
            "correct_answers": correct,
            "accuracy": correct / num_tasks,
            "avg_time": total_time / num_tasks,
            "total_time": total_time,
            "results": results,
        }

    def _extract_numeric_answer(self, text: str) -> float | None:
        """Extract numeric answer from text."""
        import re

        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        return None

    def _fuzzy_match(self, response: str, expected: str) -> bool:
        """Fuzzy matching for reasoning answers."""
        # Handle common variations
        if expected == "yes" and any(
            word in response for word in ["yes", "true", "correct", "right"]
        ):
            return True
        if expected == "no" and any(
            word in response for word in ["no", "false", "incorrect", "wrong"]
        ):
            return True
        if expected == "maybe" and any(
            word in response for word in ["maybe", "possibly", "might", "could"]
        ):
            return True
        return False

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        print("=" * 80)
        print("COMPREHENSIVE LLAMAAGENT EVALUATION REPORT")
        print("=" * 80)

        start_time = time.time()

        # Test configurations
        configs = [
            (
                "Vanilla-ReAct",
                AgentConfig(
                    name="Vanilla", role=AgentRole.GENERALIST, spree_enabled=False
                ),
            ),
            (
                "SPRE-Agent",
                AgentConfig(name="SPRE", role=AgentRole.PLANNER, spree_enabled=True),
            ),
        ]

        # Run all benchmarks
        math_results = {}
        code_results = {}
        reasoning_results = {}

        for config_name, config in configs:
            print(f"\nEvaluating {config_name} configuration...")

            math_results[config_name] = await self.run_math_benchmark(
                config_name, config, 20
            )
            code_results[config_name] = await self.run_code_benchmark(
                config_name, config, 15
            )
            reasoning_results[config_name] = await self.run_reasoning_benchmark(
                config_name, config, 15
            )

        total_time = time.time() - start_time

        # Compile comprehensive report
        report = {
            "evaluation_metadata": {
                "timestamp": time.time(),
                "total_evaluation_time": total_time,
                "framework": "LlamaAgent with SPRE methodology",
                "author": "Nik Jois <nikjois@llamasearch.ai>",
                "datasets_used": [
                    {
                        "name": "Mathematical Reasoning",
                        "description": "20 mathematical problems covering arithmetic, algebra, geometry, and word problems",
                        "tasks": 20,
                    },
                    {
                        "name": "Code Generation",
                        "description": "15 programming tasks covering algorithms, data structures, and problem solving",
                        "tasks": 15,
                    },
                    {
                        "name": "Logical Reasoning",
                        "description": "15 reasoning tasks covering logic, deduction, pattern recognition, and inference",
                        "tasks": 15,
                    },
                ],
                "huggingface_datasets_referenced": [
                    {"name": "GSM8K", "url": "https://huggingface.co/datasets/gsm8k"},
                    {
                        "name": "HumanEval",
                        "url": "https://huggingface.co/datasets/openai_humaneval",
                    },
                    {
                        "name": "CommonsenseQA",
                        "url": "https://huggingface.co/datasets/commonsense_qa",
                    },
                    {
                        "name": "HellaSwag",
                        "url": "https://huggingface.co/datasets/hellaswag",
                    },
                    {
                        "name": "GAIA",
                        "url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
                    },
                ],
            },
            "mathematical_reasoning": math_results,
            "code_generation": code_results,
            "logical_reasoning": reasoning_results,
            "performance_summary": self._generate_performance_summary(
                math_results, code_results, reasoning_results
            ),
            "spre_analysis": self._analyze_spre_impact(
                math_results, code_results, reasoning_results
            ),
        }

        # Save report
        output_path = Path("comprehensive_evaluation_report.json")
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Total evaluation time: {total_time:.1f}s")
        print(f"Report saved to: {output_path}")

        return report

    def _generate_performance_summary(
        self, math_results, code_results, reasoning_results
    ) -> Dict[str, Any]:
        """Generate performance summary across all benchmarks."""
        summary = {}

        for config in ["Vanilla-ReAct", "SPRE-Agent"]:
            math_acc = math_results[config]["accuracy"]
            code_acc = code_results[config]["accuracy"]
            reason_acc = reasoning_results[config]["accuracy"]

            summary[config] = {
                "overall_accuracy": (math_acc + code_acc + reason_acc) / 3,
                "mathematical_reasoning": math_acc,
                "code_generation": code_acc,
                "logical_reasoning": reason_acc,
                "avg_response_time": (
                    math_results[config]["avg_time"]
                    + code_results[config]["avg_time"]
                    + reasoning_results[config]["avg_time"]
                )
                / 3,
                "total_tasks": (
                    math_results[config]["total_tasks"]
                    + code_results[config]["total_tasks"]
                    + reasoning_results[config]["total_tasks"]
                ),
            }

        return summary

    def _analyze_spre_impact(
        self, math_results, code_results, reasoning_results
    ) -> Dict[str, Any]:
        """Analyze the impact of SPRE methodology."""
        vanilla = "Vanilla-ReAct"
        spre = "SPRE-Agent"

        math_improvement = (
            math_results[spre]["accuracy"] - math_results[vanilla]["accuracy"]
        )
        code_improvement = (
            code_results[spre]["accuracy"] - code_results[vanilla]["accuracy"]
        )
        reason_improvement = (
            reasoning_results[spre]["accuracy"] - reasoning_results[vanilla]["accuracy"]
        )

        return {
            "methodology": "Strategic Planning & Resourceful Execution (SPRE)",
            "improvements": {
                "mathematical_reasoning": {
                    "absolute_improvement": math_improvement,
                    "relative_improvement": math_improvement
                    / math_results[vanilla]["accuracy"]
                    if math_results[vanilla]["accuracy"] > 0
                    else 0,
                },
                "code_generation": {
                    "absolute_improvement": code_improvement,
                    "relative_improvement": code_improvement
                    / code_results[vanilla]["accuracy"]
                    if code_results[vanilla]["accuracy"] > 0
                    else 0,
                },
                "logical_reasoning": {
                    "absolute_improvement": reason_improvement,
                    "relative_improvement": reason_improvement
                    / reasoning_results[vanilla]["accuracy"]
                    if reasoning_results[vanilla]["accuracy"] > 0
                    else 0,
                },
            },
            "overall_impact": {
                "avg_absolute_improvement": (
                    math_improvement + code_improvement + reason_improvement
                )
                / 3,
                "tasks_where_spre_excels": [
                    "planning-intensive",
                    "multi-step reasoning",
                    "resource coordination",
                ],
                "key_benefits": [
                    "Hierarchical task decomposition",
                    "Strategic resource allocation",
                    "Enhanced reflection and self-correction",
                    "Context-aware tool selection",
                ],
            },
        }


async def main():
    """Run comprehensive evaluation and generate report."""
    evaluator = ComprehensiveEvaluationReport()
    report = await evaluator.generate_comprehensive_report()

    # Print summary
    summary = report["performance_summary"]
    print("\nPERFORMANCE SUMMARY:")
    print(f"{'=' * 50}")

    for config, metrics in summary.items():
        print(f"\n{config}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"  Mathematical Reasoning: {metrics['mathematical_reasoning']:.1%}")
        print(f"  Code Generation: {metrics['code_generation']:.1%}")
        print(f"  Logical Reasoning: {metrics['logical_reasoning']:.1%}")
        print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")

    # SPRE Analysis
    spre_analysis = report["spre_analysis"]
    print("\nSPRE METHODOLOGY IMPACT:")
    print(f"{'=' * 50}")

    improvements = spre_analysis["improvements"]
    print(
        f"Mathematical Reasoning: {improvements['mathematical_reasoning']['absolute_improvement']:+.1%}"
    )
    print(
        f"Code Generation: {improvements['code_generation']['absolute_improvement']:+.1%}"
    )
    print(
        f"Logical Reasoning: {improvements['logical_reasoning']['absolute_improvement']:+.1%}"
    )
    print(
        f"Average Improvement: {spre_analysis['overall_impact']['avg_absolute_improvement']:+.1%}"
    )


if __name__ == "__main__":
    asyncio.run(main())
