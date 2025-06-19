import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from llamaagent.agents.base import AgentConfig, ExecutionPlan
from llamaagent.agents.react import ReactAgent
from llamaagent.data_generation.base import DebateTrace
from llamaagent.data_generation.gdt import GDTOrchestrator


class ExperimentRunner:
    def __init__(self, output_dir: Path = Path("results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results = []

    async def run_spre_experiment(self, task: str) -> Dict[str, Any]:
        """Run SPRE planning experiment with detailed metrics"""
        agent = ReactAgent(config=AgentConfig(name="SPRE Tester", spree_enabled=True))
        start_time = time.time()
        result = await agent.execute(task)
        duration = time.time() - start_time

        return {
            "technique": "SPRE",
            "task": task,
            "duration": duration,
            "success": result.success,
            "steps": len(result.trace),
            "tokens_used": result.tokens_used,
            "plan_quality": self._calculate_plan_quality(result.plan),
            "result": result.content,
        }

    async def run_gdt_experiment(self, problem: str) -> Dict[str, Any]:
        """Run GDT debate experiment with consensus metrics"""
        orchestrator = GDTOrchestrator()
        start_time = time.time()
        trace = await orchestrator.generate_debate_trace(problem)
        duration = time.time() - start_time

        return {
            "technique": "GDT",
            "problem": problem,
            "duration": duration,
            "consensus_reached": trace.winning_path is not None,
            "dissent_ratio": self._calculate_dissent_ratio(trace),
            "step_count": trace.total_nodes,
            "tree_depth": trace.tree_depth,
            "winning_path": [node.proposal for node in trace.winning_path],
            "tokens_used": 0,
        }

    # ... similar enhanced methods for DTSR and ATES ...

    async def run_all_experiments(self, num_runs: int = 5):
        """Execute multiple runs of all experimental techniques"""
        tasks = self.load_tasks()
        all_results = []

        for _run in range(num_runs):
            experiments = [
                self.run_spre_experiment(tasks["spre"]),
                self.run_gdt_experiment(tasks["gdt"]),
                # ... DTSR and ATES experiments ...
            ]
            results = await asyncio.gather(*experiments)
            all_results.extend(results)

        self.results = all_results
        self._save_results()
        return all_results

    def _calculate_plan_quality(self, plan: Optional[ExecutionPlan]) -> float:
        """Calculate plan quality score (0-1)"""
        if not plan:
            return 0.0
        completeness = sum(1 for step in plan.steps if step.is_completed) / len(plan.steps)
        dependency_coverage = len(plan.dependencies) / max(1, len(plan.steps))
        return (completeness + dependency_coverage) / 2

    def _calculate_dissent_ratio(self, trace: DebateTrace) -> float:
        """Calculate dissent ratio in debate tree"""
        total_nodes = trace.total_nodes
        if total_nodes == 0:
            return 0.0
        dissent_nodes = sum(1 for node in trace.winning_path if node.critique)
        return dissent_nodes / total_nodes

    def _save_results(self) -> None:
        """Persist experiment results to a timestamped JSON file inside the output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"results_{timestamp}.json"
        with results_path.open("w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)
        print(f"[INFO] Experiment results saved to {results_path}")

    # ... other helper methods ...

    def load_tasks(self) -> Dict[str, str]:
        """Load experimental tasks from text files"""
        return {
            "spre": Path("tools-txts/01_MultiStepPlanning.txt").read_text(),
            "gdt": Path("tools-txts/02_DebateTree.txt").read_text(),
            # ... load other tasks ...
        }

    def generate_report(self):
        """Generate research report from results"""
        # Implementation for report generation
        # Includes metrics analysis, visualizations, etc.
        pass


# Example usage
if __name__ == "__main__":
    runner = ExperimentRunner()
    asyncio.run(runner.run_all_experiments())
    runner.generate_report()
