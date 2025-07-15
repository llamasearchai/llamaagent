#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for LlamaAgent
Tests multiple Hugging Face datasets and generates detailed analysis
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from src.llamaagent.agents import AgentConfig, AgentRole, ReactAgent
from src.llamaagent.benchmarks.gaia_benchmark import GAIABenchmark
from src.llamaagent.tools import ToolRegistry, get_all_tools

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for LlamaAgent across multiple datasets."""

    def __init__(self):
        self.results = {}
        self.tools = ToolRegistry()
        for tool in get_all_tools():
            self.tools.register(tool)

    async def evaluate_gaia_benchmark(self) -> Dict[str, Any]:
        """Evaluate on official GAIA benchmark from Hugging Face."""
        logger.info("Starting GAIA benchmark evaluation...")

        # Test both basic and SPRE configurations
        configs = [
            ("Vanilla-ReAct", AgentConfig(name="Vanilla", role=AgentRole.GENERALIST, spree_enabled=False)),
            ("SPRE-Agent", AgentConfig(name="SPRE", role=AgentRole.PLANNER, spree_enabled=True)),
        ]

        gaia_results = {}

        for name, config in configs:
            logger.info(f"Testing {name} configuration...")

            agent = ReactAgent(config, tools=self.tools)
            benchmark = GAIABenchmark(subset="validation", max_tasks=20)

            start_time = time.time()
            results = await benchmark.evaluate_agent(agent, shuffle=True)
            evaluation_time = time.time() - start_time

            report = benchmark.generate_report(results)
            report["evaluation_time"] = evaluation_time
            report["agent_config"] = name

            gaia_results[name] = report

            logger.info(
                f"{name}: {report['correct_answers']}/{report['total_tasks']} correct "
                f"({report['overall_accuracy']:.1%}) in {evaluation_time:.1f}s"
            )

        return gaia_results

    async def evaluate_math_dataset(self) -> Dict[str, Any]:
        """Evaluate on mathematics reasoning dataset."""
        logger.info("Starting mathematics evaluation...")

        try:
            from datasets import load_dataset

            # Load GSM8K dataset - grade school math word problems
            dataset = load_dataset("gsm8k", "main", split="test")

            # Select subset for evaluation
            math_tasks = []
            for i, item in enumerate(dataset):
                if i >= 25:  # Limit to 25 tasks for comprehensive evaluation
                    break
                math_tasks.append({"question": item["question"], "answer": item["answer"].split("####")[-1].strip()})

            # Test SPRE vs Vanilla
            configs = [
                ("Vanilla-Math", AgentConfig(name="VanillaMath", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-Math", AgentConfig(name="SPREMath", role=AgentRole.SPECIALIST, spree_enabled=True)),
            ]

            math_results = {}

            for name, config in configs:
                logger.info(f"Testing {name} on GSM8K...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0

                for i, task in enumerate(math_tasks):
                    start_time = time.time()
                    response = await agent.execute(task["question"])
                    exec_time = time.time() - start_time

                    # Extract numeric answer
                    predicted = self._extract_numeric_answer(response.content)
                    expected = self._extract_numeric_answer(task["answer"])

                    is_correct = (
                        abs(predicted - expected) < 0.01 if predicted is not None and expected is not None else False
                    )
                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    logger.info(f"Task {i + 1}/25: {'PASS' if is_correct else 'FAIL'} ({exec_time:.1f}s)")

                math_results[name] = {
                    "dataset": "GSM8K",
                    "total_tasks": len(math_tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(math_tasks),
                    "avg_time": total_time / len(math_tasks),
                    "total_tokens": total_tokens,
                }

            return math_results

        except Exception as e:
            logger.error(f"Math evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_code_dataset(self) -> Dict[str, Any]:
        """Evaluate on code generation dataset."""
        logger.info("Starting code evaluation...")

        try:
            from datasets import load_dataset

            # Load HumanEval dataset - code generation benchmark
            dataset = load_dataset("openai_humaneval", split="test")

            # Select subset for evaluation
            code_tasks = []
            for i, item in enumerate(dataset):
                if i >= 15:  # Limit to 15 tasks
                    break
                code_tasks.append({"prompt": item["prompt"], "test": item["test"], "entry_point": item["entry_point"]})

            configs = [
                ("Vanilla-Code", AgentConfig(name="VanillaCode", role=AgentRole.GENERALIST, spree_enabled=False)),
                ("SPRE-Code", AgentConfig(name="SPRECode", role=AgentRole.SPECIALIST, spree_enabled=True)),
            ]

            code_results = {}

            for name, config in configs:
                logger.info(f"Testing {name} on HumanEval...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0

                for i, task in enumerate(code_tasks):
                    start_time = time.time()
                    response = await agent.execute(f"Complete this Python function:\n{task['prompt']}")
                    exec_time = time.time() - start_time

                    # Simple check: does response contain function definition
                    is_correct = "def " in response.content and task["entry_point"] in response.content
                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    logger.info(f"Task {i + 1}/15: {'PASS' if is_correct else 'FAIL'} ({exec_time:.1f}s)")

                code_results[name] = {
                    "dataset": "HumanEval",
                    "total_tasks": len(code_tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(code_tasks),
                    "avg_time": total_time / len(code_tasks),
                    "total_tokens": total_tokens,
                }

            return code_results

        except Exception as e:
            logger.error(f"Code evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_reasoning_dataset(self) -> Dict[str, Any]:
        """Evaluate on logical reasoning dataset."""
        logger.info("Starting reasoning evaluation...")

        try:
            from datasets import load_dataset

            # Load StrategyQA - multi-hop reasoning dataset
            dataset = load_dataset("wics/strategy-qa", split="train")

            # Select subset for evaluation
            reasoning_tasks = []
            for i, item in enumerate(dataset):
                if i >= 20:  # Limit to 20 tasks
                    break
                if "question" in item and "answer" in item:
                    reasoning_tasks.append({"question": item["question"], "answer": str(item["answer"]).lower()})

            configs = [
                (
                    "Vanilla-Reasoning",
                    AgentConfig(name="VanillaReason", role=AgentRole.GENERALIST, spree_enabled=False),
                ),
                ("SPRE-Reasoning", AgentConfig(name="SPREReason", role=AgentRole.PLANNER, spree_enabled=True)),
            ]

            reasoning_results = {}

            for name, config in configs:
                logger.info(f"Testing {name} on StrategyQA...")

                agent = ReactAgent(config, tools=self.tools)
                correct = 0
                total_time = 0
                total_tokens = 0

                for i, task in enumerate(reasoning_tasks):
                    start_time = time.time()
                    response = await agent.execute(f"Answer with yes or no: {task['question']}")
                    exec_time = time.time() - start_time

                    # Extract yes/no answer
                    predicted = "yes" if "yes" in response.content.lower() else "no"
                    expected = "yes" if task["answer"] == "true" else "no"

                    is_correct = predicted == expected
                    if is_correct:
                        correct += 1

                    total_time += exec_time
                    total_tokens += response.tokens_used

                    logger.info(
                        f"Task {i + 1}/{len(reasoning_tasks)}: {'PASS' if is_correct else 'FAIL'} ({exec_time:.1f}s)"
                    )

                reasoning_results[name] = {
                    "dataset": "StrategyQA",
                    "total_tasks": len(reasoning_tasks),
                    "correct_answers": correct,
                    "accuracy": correct / len(reasoning_tasks),
                    "avg_time": total_time / len(reasoning_tasks),
                    "total_tokens": total_tokens,
                }

            return reasoning_results

        except Exception as e:
            logger.error(f"Reasoning evaluation failed: {e}")
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

    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluations and compile results."""
        logger.info("Starting comprehensive evaluation suite...")

        start_time = time.time()

        # Run all evaluations
        results = {
            "gaia": await self.evaluate_gaia_benchmark(),
            "mathematics": await self.evaluate_math_dataset(),
            "code_generation": await self.evaluate_code_dataset(),
            "logical_reasoning": await self.evaluate_reasoning_dataset(),
        }

        total_time = time.time() - start_time

        # Compile summary
        summary = {
            "evaluation_timestamp": time.time(),
            "total_evaluation_time": total_time,
            "datasets_evaluated": [
                {"name": "GAIA", "url": "https://huggingface.co/datasets/gaia-benchmark/GAIA"},
                {"name": "GSM8K", "url": "https://huggingface.co/datasets/gsm8k"},
                {"name": "HumanEval", "url": "https://huggingface.co/datasets/openai_humaneval"},
                {"name": "StrategyQA", "url": "https://huggingface.co/datasets/wics/strategy-qa"},
            ],
            "agent_configurations": [
                "Vanilla ReAct (baseline)",
                "SPRE-enabled (Strategic Planning & Resourceful Execution)",
            ],
        }

        results["summary"] = summary

        # Save results
        output_path = Path("comprehensive_evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Comprehensive evaluation completed in {total_time:.1f}s")
        logger.info(f"Results saved to {output_path}")

        return results


async def main():
    """Main evaluation function."""
    evaluator = ComprehensiveEvaluator()
    results = await evaluator.run_comprehensive_evaluation()

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
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
                        f"({config_results['correct_answers']}/{config_results['total_tasks']} correct)"
                    )

    print("\nDatasets evaluated:")
    for dataset in results["summary"]["datasets_evaluated"]:
        print(f"- {dataset['name']}: {dataset['url']}")


if __name__ == "__main__":
    asyncio.run(main())
