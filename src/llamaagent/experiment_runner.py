"""
Experiment Runner for LlamaAgent

This module provides a comprehensive experiment runner for evaluating different
AI techniques including SPRE, GDT, and other experimental approaches.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llamaagent.agents.base import AgentConfig
from llamaagent.agents.react import ReactAgent
from llamaagent.data_generation.base import DebateTrace
from llamaagent.data_generation.gdt import GDTOrchestrator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runner for executing various AI experiments and collecting results.
    """

    def __init__(self, output_dir: Path = Path("results")):
        """Initialize the experiment runner."""
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def load_tasks(self) -> Dict[str, str]:
        """Load experimental tasks from text files."""
        tasks = {}

        # Load SPRE tasks
        spre_file = Path("tools-txts/01_MultiStepPlanning.txt")
        tasks["spre"] = (
            spre_file.read_text(encoding='utf-8')
            if spre_file.exists()
            else "Default SPRE planning task"
        )

        # Load GDT tasks
        gdt_file = Path("tools-txts/02_DebateTree.txt")
        tasks["gdt"] = (
            gdt_file.read_text(encoding='utf-8')
            if gdt_file.exists()
            else "Default GDT debate task"
        )

        return tasks

    async def run_spre_experiment(self, task: str) -> Dict[str, Any]:
        """Run SPRE planning experiment with detailed metrics."""
        agent = ReactAgent(config=AgentConfig(name="SPRE Tester", spree_enabled=True))
        start_time = time.time()

        try:
            result = await agent.execute(task)
            duration = time.time() - start_time

            return {
                "technique": "SPRE",
                "task": task,
                "success": getattr(result, 'success', False),
                "duration": duration,
                "tokens_used": getattr(result, 'tokens_used', 0),
                "result": getattr(result, 'content', str(result)),
                "plan_quality": self._calculate_plan_quality(result),
                "step_count": len(getattr(getattr(result, 'plan', None), 'steps', [])),
            }
        except Exception as e:
            logger.error(f"SPRE experiment failed: {str(e)}")
            return {
                "technique": "SPRE",
                "task": task,
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "plan_quality": 0.0,
                "step_count": 0,
            }

    async def run_gdt_experiment(self, problem: str) -> Dict[str, Any]:
        """Run GDT debate experiment with consensus metrics."""
        orchestrator = GDTOrchestrator()
        start_time = time.time()

        try:
            # type: ignore[attribute-access]  # generate_debate_trace assumed to exist
            trace: Optional[DebateTrace] = await orchestrator.generate_debate_trace(problem)  # type: ignore[assignment, var-annotated]
            duration = time.time() - start_time

            if trace is None:
                raise ValueError("No trace generated")

            return {
                "technique": "GDT",
                "problem": problem,
                "success": True,
                "duration": duration,
                "tree_depth": getattr(trace, 'tree_depth', 0),  # type: ignore[arg-type]
                "consensus_reached": self._check_consensus(trace),  # type: ignore[arg-type]
                "dissent_ratio": self._calculate_dissent_ratio(trace),  # type: ignore[arg-type]
                "winning_path": [getattr(node, 'proposal', '') for node in getattr(trace, 'winning_path', [])],  # type: ignore[arg-type]
            }
        except Exception as e:
            logger.error(f"GDT experiment failed: {str(e)}")
            return {
                "technique": "GDT",
                "problem": problem,
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e),
                "tree_depth": 0,
                "consensus_reached": False,
                "dissent_ratio": 0.0,
            }

    def _calculate_plan_quality(self, result: Any) -> float:
        """Calculate plan quality score (0-1)."""
        plan = getattr(result, 'plan', None)
        if not plan:
            return 0.0

        steps = getattr(plan, 'steps', [])
        total_steps = len(steps)
        if total_steps == 0:
            return 0.0

        completed_steps = sum(
            1 for step in steps if getattr(step, 'is_completed', False)
        )
        completeness = completed_steps / total_steps
        dependency_coverage = len(getattr(plan, 'dependencies', [])) / max(
            1, total_steps
        )

        return (completeness + dependency_coverage) / 2

    def _calculate_dissent_ratio(self, trace: DebateTrace) -> float:
        """Calculate dissent ratio in debate tree."""
        if not hasattr(trace, 'total_nodes') or trace.total_nodes == 0:
            return 0.0

        total_nodes = trace.total_nodes
        dissent_nodes = 0

        if hasattr(trace, 'winning_path'):
            dissent_nodes = sum(
                1
                for node in trace.winning_path
                if hasattr(node, 'critique') and node.critique
            )

        return dissent_nodes / total_nodes

    def _check_consensus(self, trace: DebateTrace) -> bool:
        """Check if consensus was reached in the debate."""
        if not hasattr(trace, 'winning_path'):
            return False

        # Simple consensus check: last node in winning path should have no critique
        if trace.winning_path:
            last_node = trace.winning_path[-1]
            return not hasattr(last_node, 'critique') or not last_node.critique

        return False

    async def run_all_experiments(self, num_runs: int = 5) -> List[Dict[str, Any]]:
        """Execute multiple runs of all experimental techniques."""
        tasks = self.load_tasks()
        all_results: List[Dict[str, Any]] = []

        for run in range(num_runs):
            print(f"[INFO] Running experiment batch {run + 1}/{num_runs}")

            # Run SPRE experiments
            spre_result = await self.run_spre_experiment(tasks["spre"])
            spre_result["run"] = run
            all_results.append(spre_result)

            # Run GDT experiments
            gdt_result = await self.run_gdt_experiment(tasks["gdt"])
            gdt_result["run"] = run
            all_results.append(gdt_result)

        self.results = all_results
        self._save_results()
        return all_results

    def _save_results(self) -> None:
        """Persist experiment results to a timestamped JSON file inside the output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"experiment_results_{timestamp}.json"

        with results_path.open("w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2, default=str)

        logger.info(f"Experiment results saved to {results_path}")

    def generate_report(self) -> None:
        """Generate research report from results."""
        if not self.results:
            logger.warning("No results to generate report from")
            return

        # Calculate summary statistics
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.get("success", False))
        success_rate = (
            successful_experiments / total_experiments if total_experiments > 0 else 0
        )

        # Group by technique
        techniques = {}
        for result in self.results:
            technique = result.get("technique", "Unknown")
            if technique not in techniques:
                techniques[technique] = []
            techniques[technique].append(result)

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "success_rate": success_rate,
            "techniques": {},
        }

        for technique, results in techniques.items():
            technique_success = sum(1 for r in results if r.get("success", False))
            technique_success_rate = technique_success / len(results) if results else 0
            avg_duration = (
                sum(r.get("duration", 0) for r in results) / len(results)
                if results
                else 0
            )

            report["techniques"][technique] = {
                "total_runs": len(results),
                "successful_runs": technique_success,
                "success_rate": technique_success_rate,
                "average_duration": avg_duration,
            }

        # Save report
        report_path = self.output_dir / "experiment_report.json"
        with report_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        print(f"[INFO] Experiment report saved to {report_path}")

        # Print summary
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"Total experiments: {total_experiments}")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Overall success rate: {success_rate:.2%}")
        print("\nBy technique:")
        for technique, stats in report["techniques"].items():
            print(
                f"  {technique}: {stats['success_rate']:.2%} ({stats['successful_runs']}/{stats['total_runs']})"
            )


# Example usage
if __name__ == "__main__":
    runner = ExperimentRunner()
    asyncio.run(runner.run_all_experiments())
    runner.generate_report()
