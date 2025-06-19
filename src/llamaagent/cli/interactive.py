#!/usr/bin/env python3
"""
Interactive CLI for LlamaAgent Research Experiment
Author : Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

# ──────────────────────────────── stdlib ────────────────────────────────
import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# ─────────────────────────────── 3rd-party (optional) ───────────────────
if TYPE_CHECKING:  # keep valid annotations even if Rich is missing
    from rich.console import Console as _RichConsole
    from rich.prompt import Prompt as _RichPrompt
else:  # simple stubs so the names are always "types"

    class _RichStub: ...  # noqa: E701

    _RichConsole = _RichPrompt = _RichStub  # type: ignore

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn)
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
except ModuleNotFoundError:  # plain-text fall-back
    Console = Panel = Progress = SpinnerColumn = TextColumn = BarColumn = TimeElapsedColumn = Table = Prompt = Confirm = box = None  # type: ignore

# ─────────────────────────────── internal ───────────────────────────────
from llamaagent.agents import AgentConfig, AgentRole, ReactAgent
from llamaagent.data_generation.spre import SPREDatasetGenerator
from llamaagent.tools import ToolRegistry, get_all_tools

# ═════════════════════════════ helper wrappers ══════════════════════════


def _console() -> Optional[_RichConsole]:
    """Return a Rich console if available."""
    return Console() if Console else None


def _panel(*args: Any, **kwargs: Any) -> Any:
    """Rich Panel wrapper that degrades to plain string."""
    if Panel:
        return Panel(*args, **kwargs)
    return "\n".join(str(a) for a in args)


def _prompt(msg: str, choices: List[str]) -> str:
    """Prompt.ask wrapper safe for plain mode."""
    if Prompt:
        return Prompt.ask(msg, choices=choices)
    return input(f"{msg} {choices}: ").strip()


def _confirm(msg: str) -> bool:
    """Confirm.ask wrapper safe for plain mode."""
    if Confirm:
        return Confirm.ask(msg)
    return input(f"{msg} [y/N]: ").lower().startswith("y")


# ═════════════════════════════ main class ═══════════════════════════════


class InteractiveExperiment:
    """Interactive research experiment runner."""

    def __init__(self) -> None:
        self.console: Optional[_RichConsole] = _console()
        self.results: Dict[str, Any] = {}

    # ──────────────────────── ui primitives ────────────────────────────
    def _print(self, msg: str) -> None:
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def display_banner(self) -> None:
        text = (
            "[bold cyan]LlamaAgent Research Experiment[/bold cyan]\n"
            "[dim]Strategic Planning & Resourceful Execution[/dim]\n"
            "Interactive Demo · Real-time Results · AI Agents"
        )
        if self.console:
            self.console.print(_panel(text, border_style="cyan", title="[bold]Welcome[/bold]"))
        else:
            print("=" * 60, "\n", text, "\n", "=" * 60)

    def menu(self) -> str:
        items = [
            ("1", "Quick Demo", "Basic agent interaction"),
            ("2", "SPRE Planning Demo", "Hierarchical planning"),
            ("3", "Dataset Generation", "Create SPRE dataset"),
            ("4", "Performance Benchmarks", "Compare agents"),
            ("5", "GAIA Evaluation", "Official GAIA benchmark"),
            ("6", "Full Experiment", "End-to-end pipeline"),
            ("7", "View Results", "Last run"),
            ("8", "API Demo", "FastAPI smoke-test"),
            ("9", "Exit", "Quit"),
        ]
        if self.console:
            table = Table(show_header=False, box=box.ROUNDED)  # type: ignore[arg-type]
            for opt, desc, feat in items:
                table.add_row(opt, desc, feat)
            self.console.print(_panel(table, title="[bold cyan]Menu[/bold cyan]", border_style="blue"))
        else:
            for opt, desc, _ in items:
                print(f"{opt}. {desc}")
        return _prompt("Choose an option", [opt for opt, *_ in items])

    # ───────────────────────── demos / steps ───────────────────────────
    async def _build_agent(self, cfg: AgentConfig, spree_enabled: bool = False) -> ReactAgent:
        tools = ToolRegistry()
        for t in get_all_tools():
            tools.register(t)
        # Override SPRE setting based on CLI flag
        cfg.spree_enabled = spree_enabled
        # Pass the tool registry via the correct keyword argument so that the
        # second positional parameter (``llm_provider``) retains its intended
        # semantics.  Using a positional argument here caused a type mismatch
        return ReactAgent(cfg, tools=tools)

    async def quick_demo(self, spree_enabled: bool = False) -> None:
        self._print("\n[bold green]Quick Demo[/bold green]")
        agent = await self._build_agent(
            AgentConfig(name="Demo", role=AgentRole.GENERALIST), spree_enabled=spree_enabled
        )
        for prompt in (
            "15 * 23 + 47 = ?",
            "Square root of 144?",
            "Python function to reverse a string.",
            "Explain ML in simple terms.",
        ):
            res = await agent.execute(prompt)
            self._print(_panel(f"[cyan]{prompt}[/cyan]\n\n{res.content}", border_style="green"))

    async def spre_demo(self, spree_enabled: bool = True) -> None:
        self._print("\n[bold blue]SPRE Planning Demo[/bold blue]")
        agent = await self._build_agent(
            AgentConfig(name="Planner", role=AgentRole.PLANNER), spree_enabled=spree_enabled
        )
        task = "Plan a DS project to predict house prices."
        res = await agent.execute(task)
        steps = getattr(res, "plan", None)
        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps.steps)) if steps else "—"
        self._print(_panel(f"[bold]{task}[/bold]\n\n{res.content}\n\n[italic]Steps:[/italic]\n{steps_str}"))

    async def dataset_demo(self) -> None:
        self._print("\n[bold magenta]Dataset Generation[/bold magenta]")
        out = Path("demo_datasets/spre_demo.json")
        out.parent.mkdir(exist_ok=True)
        await SPREDatasetGenerator(seed=42).generate_dataset(10, out)
        self._print(f"Generated dataset → {out}")

    async def benchmark(self, spree_enabled: bool = False) -> None:
        self._print("\n[bold red]Benchmarks[/bold red]")
        configs = [
            ("Basic", AgentConfig("Basic", AgentRole.GENERALIST), False),
            ("SPRE", AgentConfig("SPRE", AgentRole.PLANNER), True),
            ("Spec", AgentConfig("Spec", AgentRole.SPECIALIST, temperature=0.3), False),
        ]
        tasks = ["123*456", "Capital of France?", "Sort list func", "Explain quantum computing"]
        results: List[Dict[str, Any]] = []
        for name, cfg, use_spree in configs:
            agent = await self._build_agent(cfg, spree_enabled=use_spree or spree_enabled)
            summ: Dict[str, Any] = {"name": name, "tasks": []}
            for t in tasks:
                t0 = time.perf_counter()
                r = await agent.execute(t)
                summ["tasks"].append({"time": time.perf_counter() - t0, "ok": r.success})
            results.append(summ)
        self.results["benchmarks"] = results
        self._print(json.dumps(results, indent=2))

    async def gaia_evaluation(self, spree_enabled: bool = False) -> None:
        self._print("\n[bold yellow]GAIA Benchmark Evaluation[/bold yellow]")
        try:
            from ..benchmarks.gaia_benchmark import GAIABenchmark
            
            # Create SPRE-enabled agent
            config = AgentConfig(
                name="GAIA-Agent",
                role=AgentRole.PLANNER,
                spree_enabled=spree_enabled
            )
            agent = await self._build_agent(config, spree_enabled=spree_enabled)
            
            # Run GAIA evaluation on subset
            benchmark = GAIABenchmark(subset="validation", max_tasks=5)
            self._print("Loading GAIA dataset...")
            await benchmark.load_dataset()
            
            self._print(f"Evaluating {len(benchmark.tasks)} GAIA tasks...")
            results = await benchmark.evaluate_agent(agent, shuffle=True)
            
            # Generate report
            report = benchmark.generate_report(results)
            self._print(f"GAIA Results: {report['correct_answers']}/{report['total_tasks']} correct ({report['overall_accuracy']:.1%})")
            
            # Store results
            self.results["gaia"] = report
            
        except ImportError:
            self._print("[red]GAIA benchmark requires 'datasets' library. Install with: pip install datasets[/red]")
        except Exception as e:
            self._print(f"[red]GAIA evaluation failed: {e}[/red]")

    async def full_experiment(self) -> None:
        await self.dataset_demo()
        await self.benchmark()
        await self.gaia_evaluation()
        Path("experiment_results.json").write_text(json.dumps(self.results, indent=2))
        self._print("[green]Experiment complete.[/green]  Results → experiment_results.json")

    def view_results(self) -> None:
        p = Path("experiment_results.json")
        self._print(p.read_text() if p.exists() else "No previous results.")

    async def api_demo(self) -> None:
        self._print("\n[bold purple]FastAPI Demo[/bold purple]")
        try:
            from fastapi.testclient import TestClient

            from llamaagent.api import app
        except ModuleNotFoundError:
            self._print("FastAPI components not installed.")
            return
        client = TestClient(app)
        self._print(f"/health → {client.get('/health').json()}")
        self._print(f"/chat   → {client.post('/chat', json={'message': 'hi'}).json()}")

    # ───────────────────────────── main loop ────────────────────────────
    async def run(self, spree_enabled: bool = False) -> None:
        self.display_banner()
        if spree_enabled:
            self._print("[bold yellow]SPRE Mode Enabled[/bold yellow] - Strategic Planning & Resourceful Execution")
        while True:
            try:
                choice = self.menu()
                if choice == "1":
                    await self.quick_demo(spree_enabled)
                elif choice == "2":
                    await self.spre_demo(True)  # Always use SPRE for demo
                elif choice == "3":
                    await self.dataset_demo()
                elif choice == "4":
                    await self.benchmark(spree_enabled)
                elif choice == "5":
                    await self.gaia_evaluation(spree_enabled)
                elif choice == "6":
                    await self.full_experiment()
                elif choice == "7":
                    self.view_results()
                elif choice == "8":
                    await self.api_demo()
                elif choice == "9":
                    self._print("Goodbye!")
                    break
                else:
                    self._print("Invalid choice.")
                if self.console and choice != "9":
                    _ = _prompt("\n[dim]Enter to continue[/dim]", [""])  # noqa: F841
                    self.console.clear()
                    self.display_banner()
            except KeyboardInterrupt:
                self._print("\nInterrupted.")
                break
            except Exception as exc:  # pragma: no cover
                self._print(f"[red]Error:[/red] {exc}")


# ────────────────────────────── public helper ────────────────────────────
# Exported function consumed by ``run_experiment.py``.  Keeping the helper in
# this module avoids duplication and provides an explicit public API for
# programmatic launches of the interactive experiment.


async def run_interactive_experiment(spree_enabled: bool = False) -> None:
    """Run the :class:`InteractiveExperiment` event loop.

    This thin wrapper exists so that external launchers (e.g. ``run_experiment.py``)
    can invoke the interactive CLI without accessing private helpers or
    rewriting the startup logic.
    """

    await InteractiveExperiment().run(spree_enabled=spree_enabled)


async def _main() -> None:
    # Delegate to the public helper so that both CLI execution and external
    # imports share the same entrypoint.
    import sys

    spree_enabled = "--spree" in sys.argv
    await run_interactive_experiment(spree_enabled=spree_enabled)


if __name__ == "__main__":
    asyncio.run(_main())
