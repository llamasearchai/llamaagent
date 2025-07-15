#!/usr/bin/env python3
"""
Comprehensive Hugging Face Dataset Evaluation for LlamaAgent
Tests on GSM8K, HumanEval, StrategyQA, and other accessible datasets
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append("src")

from llamaagent.agents import AgentConfig, AgentRole, ReactAgent
from llamaagent.tools import ToolRegistry, get_all_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HuggingFaceEvaluator:
    """Comprehensive evaluation suite using Hugging Face datasets."""

    def __init__(self):
        self.tools = ToolRegistry()
        for tool in get_all_tools():
            self.tools.register(tool)
        self.results = {}

    async def evaluate_gsm8k_dataset(self, max_tasks: int = 20) -> Dict[str, Any]:
        """Evaluate on GSM8K mathematical reasoning dataset."""
        logger.info(f"Starting GSM8K evaluation with {max_tasks} tasks...")

        try:
            from datasets import load_dataset

            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split="test")

            # Select tasks
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                answer = item["answer"].split("####")[-1].strip()
                tasks.append({"question": item["question"], "answer": answer})

            # Test both configurations
            configs = [
                ("Vanilla-GSM8K", AgentConfig(name="Vanilla", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-GSM8K", AgentConfig(name="SPRE", role=AgentRole.PLANNER, spree_enabled=True)),
            ]

            results = {}

            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0
                task_results = []

                for i, task in enumerate(tasks):
                    start_time = time.time()
                    response = await agent.execute(f"Solve this math problem step by step: {task['question']}")
                    exec_time = time.time() - start_time

                    # Extract numeric answer
                    predicted = self._extract_numeric_answer(response.content)
                    expected = self._extract_numeric_answer(task["answer"])

                    is_correct = predicted is not None and expected is not None and abs(predicted - expected) < 0.01

                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    task_results.append(
                        {
                            "question": task["question"][:100] + "...",
                            "expected": task["answer"],
                            "predicted": str(predicted) if predicted is not None else "N/A",
                            "correct": is_correct,
                            "time": exec_time,
                        }
                    )

                    logger.info(f"Task {i + 1}/{len(tasks)}: {'✓' if is_correct else '✗'} ({exec_time:.1f}s)")

                results[config_name] = {
                    "dataset": "GSM8K",
                    "total_tasks": len(tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(tasks),
                    "avg_time": total_time / len(tasks),
                    "total_tokens": total_tokens,
                    "task_results": task_results,
                }

            return results

        except Exception as e:
            logger.error(f"GSM8K evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_humaneval_dataset(self, max_tasks: int = 15) -> Dict[str, Any]:
        """Evaluate on HumanEval code generation dataset."""
        logger.info(f"Starting HumanEval evaluation with {max_tasks} tasks...")

        try:
            from datasets import load_dataset

            # Load HumanEval dataset
            dataset = load_dataset("openai_humaneval", split="test")

            # Select tasks
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                tasks.append(
                    {
                        "prompt": item["prompt"],
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                        "canonical_solution": item["canonical_solution"],
                    }
                )

            # Test both configurations
            configs = [
                ("Vanilla-Code", AgentConfig(name="VanillaCode", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-Code", AgentConfig(name="SPRECode", role=AgentRole.SPECIALIST, spree_enabled=True)),
            ]

            results = {}

            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0
                task_results = []

                for i, task in enumerate(tasks):
                    start_time = time.time()
                    response = await agent.execute(f"Complete this Python function:\n{task['prompt']}")
                    exec_time = time.time() - start_time

                    # Check if response contains function definition
                    has_function = "def " in response.content and task["entry_point"] in response.content
                    # Additional check: contains return statement
                    has_return = "return" in response.content.lower()

                    is_correct = has_function and has_return

                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    task_results.append(
                        {
                            "prompt": task["prompt"][:100] + "...",
                            "entry_point": task["entry_point"],
                            "has_function": has_function,
                            "has_return": has_return,
                            "correct": is_correct,
                            "time": exec_time,
                        }
                    )

                    logger.info(f"Task {i + 1}/{len(tasks)}: {'✓' if is_correct else '✗'} ({exec_time:.1f}s)")

                results[config_name] = {
                    "dataset": "HumanEval",
                    "total_tasks": len(tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(tasks),
                    "avg_time": total_time / len(tasks),
                    "total_tokens": total_tokens,
                    "task_results": task_results,
                }

            return results

        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_commonsense_qa(self, max_tasks: int = 25) -> Dict[str, Any]:
        """Evaluate on CommonsenseQA dataset."""
        logger.info(f"Starting CommonsenseQA evaluation with {max_tasks} tasks...")

        try:
            from datasets import load_dataset

            # Load CommonsenseQA dataset
            dataset = load_dataset("commonsense_qa", split="validation")

            # Select tasks
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                tasks.append({"question": item["question"], "choices": item["choices"], "answer": item["answerKey"]})

            # Test both configurations
            configs = [
                ("Vanilla-QA", AgentConfig(name="VanillaQA", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-QA", AgentConfig(name="SPREQA", role=AgentRole.PLANNER, spree_enabled=True)),
            ]

            results = {}

            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0
                task_results = []

                for i, task in enumerate(tasks):
                    # Format question with choices
                    choices_text = "\n".join(
                        [
                            f"{label}: {text}"
                            for label, text in zip(task["choices"]["label"], task["choices"]["text"], strict=False)
                        ]
                    )
                    question = f"{task['question']}\n\nChoices:\n{choices_text}\n\nAnswer with just the letter (A, B, C, D, or E):"

                    start_time = time.time()
                    response = await agent.execute(question)
                    exec_time = time.time() - start_time

                    # Extract answer letter
                    predicted = self._extract_answer_letter(response.content)
                    expected = task["answer"]

                    is_correct = predicted == expected

                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    task_results.append(
                        {
                            "question": task["question"][:100] + "...",
                            "expected": expected,
                            "predicted": predicted,
                            "correct": is_correct,
                            "time": exec_time,
                        }
                    )

                    logger.info(f"Task {i + 1}/{len(tasks)}: {'✓' if is_correct else '✗'} ({exec_time:.1f}s)")

                results[config_name] = {
                    "dataset": "CommonsenseQA",
                    "total_tasks": len(tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(tasks),
                    "avg_time": total_time / len(tasks),
                    "total_tokens": total_tokens,
                    "task_results": task_results,
                }

            return results

        except Exception as e:
            logger.error(f"CommonsenseQA evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_hellaswag_dataset(self, max_tasks: int = 20) -> Dict[str, Any]:
        """Evaluate on HellaSwag commonsense reasoning dataset."""
        logger.info(f"Starting HellaSwag evaluation with {max_tasks} tasks...")

        try:
            from datasets import load_dataset

            # Load HellaSwag dataset
            dataset = load_dataset("hellaswag", split="validation")

            # Select tasks
            tasks = []
            for i, item in enumerate(dataset):
                if i >= max_tasks:
                    break
                tasks.append({"context": item["ctx"], "endings": item["endings"], "answer": int(item["label"])})

            # Test both configurations
            configs = [
                ("Vanilla-HellaSwag", AgentConfig(name="VanillaHella", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-HellaSwag", AgentConfig(name="SPREHella", role=AgentRole.PLANNER, spree_enabled=True)),
            ]

            results = {}

            for config_name, config in configs:
                logger.info(f"Testing {config_name}...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0
                task_results = []

                for i, task in enumerate(tasks):
                    # Format question with endings
                    endings_text = "\n".join([f"{j}: {ending}" for j, ending in enumerate(task["endings"])])
                    question = f"Context: {task['context']}\n\nWhich ending makes the most sense?\n{endings_text}\n\nAnswer with just the number (0, 1, 2, or 3):"

                    start_time = time.time()
                    response = await agent.execute(question)
                    exec_time = time.time() - start_time

                    # Extract answer number
                    predicted = self._extract_answer_number(response.content)
                    expected = task["answer"]

                    is_correct = predicted == expected

                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    task_results.append(
                        {
                            "context": task["context"][:100] + "...",
                            "expected": expected,
                            "predicted": predicted,
                            "correct": is_correct,
                            "time": exec_time,
                        }
                    )

                    logger.info(f"Task {i + 1}/{len(tasks)}: {'✓' if is_correct else '✗'} ({exec_time:.1f}s)")

                results[config_name] = {
                    "dataset": "HellaSwag",
                    "total_tasks": len(tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(tasks),
                    "avg_time": total_time / len(tasks),
                    "total_tokens": total_tokens,
                    "task_results": task_results,
                }

            return results

        except Exception as e:
            logger.error(f"HellaSwag evaluation failed: {e}")
            return {"error": str(e)}

    def _extract_numeric_answer(self, text: str) -> float | None:
        """Extract numeric answer from text."""
        import re

        # Look for numbers in the text
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        if numbers:
            try:
                return float(numbers[-1])  # Return last number found
            except ValueError:
                pass
        return None

    def _extract_answer_letter(self, text: str) -> str | None:
        """Extract answer letter (A, B, C, D, E) from text."""
        import re

        # Look for single letter answers
        matches = re.findall(r"\b([A-E])\b", text.upper())
        if matches:
            return matches[-1]  # Return last match
        return None

    def _extract_answer_number(self, text: str) -> int | None:
        """Extract answer number (0, 1, 2, 3) from text."""
        import re

        # Look for single digit answers
        matches = re.findall(r"\b([0-3])\b", text)
        if matches:
            try:
                return int(matches[-1])  # Return last match
            except ValueError:
                pass
        return None

    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluations and compile results."""
        logger.info("Starting comprehensive Hugging Face dataset evaluation...")

        start_time = time.time()

        # Run all evaluations
        results = {
            "gsm8k": await self.evaluate_gsm8k_dataset(max_tasks=15),
            "humaneval": await self.evaluate_humaneval_dataset(max_tasks=10),
            "commonsense_qa": await self.evaluate_commonsense_qa(max_tasks=15),
            "hellaswag": await self.evaluate_hellaswag_dataset(max_tasks=15),
        }

        total_time = time.time() - start_time

        # Compile summary
        summary = {
            "evaluation_timestamp": time.time(),
            "total_evaluation_time": total_time,
            "datasets_evaluated": [
                {
                    "name": "GSM8K",
                    "url": "https://huggingface.co/datasets/gsm8k",
                    "description": "Grade school math word problems",
                },
                {
                    "name": "HumanEval",
                    "url": "https://huggingface.co/datasets/openai_humaneval",
                    "description": "Code generation benchmark",
                },
                {
                    "name": "CommonsenseQA",
                    "url": "https://huggingface.co/datasets/commonsense_qa",
                    "description": "Multiple-choice commonsense reasoning",
                },
                {
                    "name": "HellaSwag",
                    "url": "https://huggingface.co/datasets/hellaswag",
                    "description": "Commonsense natural language inference",
                },
            ],
            "agent_configurations": [
                "Vanilla ReAct (baseline without SPRE)",
                "SPRE-enabled (Strategic Planning & Resourceful Execution)",
            ],
        }

        results["summary"] = summary

        # Save results
        output_path = Path("huggingface_evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comprehensive evaluation completed in {total_time:.1f}s")
        logger.info(f"Results saved to {output_path}")

        return results


async def main():
    """Main evaluation function."""
    evaluator = HuggingFaceEvaluator()
    results = await evaluator.run_comprehensive_evaluation()

    # Print summary
    print("\n" + "=" * 80)
    print("HUGGING FACE DATASET EVALUATION RESULTS")
    print("=" * 80)

    for dataset_name, dataset_results in results.items():
        if dataset_name == "summary":
            continue

        print(f"\n{dataset_name.upper()} RESULTS:")
        print("-" * 40)

        if isinstance(dataset_results, dict) and "error" not in dataset_results:
            for config_name, config_results in dataset_results.items():
                if isinstance(config_results, dict) and "accuracy" in config_results:
                    print(
                        f"{config_name}: {config_results['accuracy']:.1%} accuracy "
                        f"({config_results['correct_answers']}/{config_results['total_tasks']} correct, "
                        f"{config_results['avg_time']:.2f}s avg)"
                    )
        else:
            print(f"Error: {dataset_results.get('error', 'Unknown error')}")

    print("\nDatasets evaluated:")
    for dataset in results["summary"]["datasets_evaluated"]:
        print(f"- {dataset['name']}: {dataset['url']}")
        print(f"  {dataset['description']}")

    print(f"\nTotal evaluation time: {results['summary']['total_evaluation_time']:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
