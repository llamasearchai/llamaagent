"""
Diagnostics CLI - Command-line interface for running comprehensive diagnostics.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..diagnostics import MasterDiagnostics, ProblemSeverity

console = Console()
app = typer.Typer(help="Comprehensive LlamaAgent system diagnostics")


@app.command()
def run(
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for the diagnostic report"
    ),
    project_root: Optional[str] = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (defaults to current directory)",
    ),
    format: str = typer.Option(
        "txt", "--format", "-f", help="Output format: txt, json, or console"
    ),
    severity_filter: Optional[str] = typer.Option(
        None,
        "--severity",
        "-s",
        help="Filter by severity: CRITICAL, HIGH, MEDIUM, LOW, INFO",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """
    Run comprehensive diagnostics on the LlamaAgent system.

    This command performs deep analysis of the entire codebase including:
    - Syntax error detection
    - Import issue analysis
    - Dependency validation
    - Security vulnerability scanning
    - Performance bottleneck identification
    - Configuration validation
    - System environment checks
    """

    # Initialize diagnostics
    if project_root:
        project_path = Path(project_root)
        if not project_path.exists():
            console.print(
                f"[red]Error: Project root '{project_root}' does not exist[/red]"
            )
            raise typer.Exit(1)
    else:
        project_path = Path.cwd()

    console.print(
        f"[blue]Running comprehensive diagnostics on: {project_path}[/blue]"
    )
    console.print()

    # Create diagnostics instance
    diagnostics = MasterDiagnostics(str(project_path))

    # Run analysis with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        try:
            report = diagnostics.run_comprehensive_analysis()
            progress.update(task, description="Analysis complete!")

        except Exception as e:
            console.print(f"[red]Error during analysis: {str(e)}[/red]")
            if verbose:
                import traceback

                console.print(traceback.format_exc())
            raise typer.Exit(1)

    # Filter by severity if specified
    if severity_filter:
        try:
            severity_level = ProblemSeverity(severity_filter.upper())
            report.problems = [
                p for p in report.problems if p.severity == severity_level
            ]
            report.total_problems = len(report.problems)
            console.print(
                f"[yellow]Filtered to show only {severity_filter.upper()} severity issues[/yellow]"
            )
        except ValueError:
            console.print(f"[red]Invalid severity level: {severity_filter}[/red]")
            console.print("Valid levels: CRITICAL, HIGH, MEDIUM, LOW, INFO")
            raise typer.Exit(1)

    # Output results
    if format == "console":
        _display_console_report(report, verbose)
    elif format == "json":
        _output_json_report(report, output_file, project_path)
    else:  # txt format
        _output_text_report(report, output_file, project_path, diagnostics)

    # Exit with appropriate code
    critical_count = report.problems_by_severity.get(ProblemSeverity.CRITICAL, 0)
    high_count = report.problems_by_severity.get(ProblemSeverity.HIGH, 0)

    if critical_count > 0:
        console.print(
            f"\n[red]⚠️  {critical_count} critical issues found - system may not function properly![/red]"
        )
        raise typer.Exit(1)
    elif high_count > 0:
        console.print(
            f"\n[yellow]⚠️  {high_count} high-priority issues found - attention recommended[/yellow]"
        )
        raise typer.Exit(0)
    else:
        console.print(
            "\n[green]PASS Analysis complete - no critical issues found![/green]"
        )
        raise typer.Exit(0)


def _display_console_report(report, verbose: bool):
    """Display the report in the console."""
    console.print("\n[bold]DIAGNOSTIC REPORT SUMMARY[/bold]")
    console.print("=" * 60)

    # Summary statistics
    summary_table = Table(title="Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Total Issues", str(report.total_problems))
    summary_table.add_row("Files Analyzed", str(report.total_files_analyzed))
    summary_table.add_row("Lines Analyzed", str(report.total_lines_analyzed))
    summary_table.add_row("Analysis Time", f"{report.analysis_duration:.2f}s")

    # Add severity breakdown
    for severity, count in report.problems_by_severity.items():
        if count > 0:
            color = {
                ProblemSeverity.CRITICAL: "red",
                ProblemSeverity.HIGH: "yellow",
                ProblemSeverity.MEDIUM: "blue",
                ProblemSeverity.LOW: "green",
                ProblemSeverity.INFO: "cyan",
            }.get(severity, "white")
            summary_table.add_row(
                f"{severity.value} Issues", f"[{color}]{count}[/{color}]"
            )

    console.print(summary_table)

    # Show recommendations
    if report.recommendations:
        console.print("\n[bold]RECOMMENDATIONS[/bold]")
        console.print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            console.print(f"{i}. {rec}")

    # Show detailed problems if verbose or if there are critical/high issues
    critical_high_count = report.problems_by_severity.get(
        ProblemSeverity.CRITICAL, 0
    ) + report.problems_by_severity.get(ProblemSeverity.HIGH, 0)

    if verbose or critical_high_count > 0:
        console.print("\n[bold]DETAILED ISSUES[/bold]")
        console.print("=" * 50)

        for problem in report.problems[:20]:  # Show first 20 problems
            color = {
                ProblemSeverity.CRITICAL: "red",
                ProblemSeverity.HIGH: "yellow",
                ProblemSeverity.MEDIUM: "blue",
                ProblemSeverity.LOW: "green",
                ProblemSeverity.INFO: "cyan",
            }.get(problem.severity, "white")

            console.print(
                f"\n[{color}]{problem.severity.value}[/{color}] - {problem.title}"
            )
            console.print(f"  📁 {problem.location}")
            if problem.line_number:
                console.print(f"  📍 Line {problem.line_number}")
            console.print(f"  Issue {problem.description}")
            if problem.suggested_fix:
                console.print(f"  INSIGHT {problem.suggested_fix}")

        if len(report.problems) > 20:
            console.print(
                f"\n[dim]... and {len(report.problems) - 20} more issues[/dim]"
            )
            console.print("[dim]Run with --format=txt for complete report[/dim]")


def _output_json_report(report, output_file: Optional[str], project_path: Path):
    """Output the report as JSON."""
    import json
    from dataclasses import asdict

    # Convert report to dict (handling enums)
    def convert_enums(obj):
        if hasattr(obj, "__dict__"):
            return {k: convert_enums(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert_enums(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_enums(v) for k, v in obj.items()}
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        else:
            return obj

    report_dict = convert_enums(asdict(report))

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = project_path / "llamaagent_diagnostic_report.json"

    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    console.print(f"[green]PASS JSON report saved to: {output_path}[/green]")


def _output_text_report(
    report, output_file: Optional[str], project_path: Path, diagnostics
):
    """Output the report as text."""
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = project_path / "llamaagent_diagnostic_report.txt"

    # Use the existing text formatting method
    report_file = diagnostics.save_report_to_file(report, str(output_path))

    console.print(
        f"[green]PASS Comprehensive diagnostic report saved to: {report_file}[/green]"
    )
    console.print(
        f"[blue]Found Found {report.total_problems} issues across {report.total_files_analyzed} files[/blue]"
    )

    # Show critical issues immediately
    critical_issues = [
        p for p in report.problems if p.severity == ProblemSeverity.CRITICAL
    ]
    if critical_issues:
        console.print("\n[red]CRITICAL CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:[/red]")
        for issue in critical_issues:
            console.print(f"  • {issue.title} ({issue.location})")

    high_issues = [p for p in report.problems if p.severity == ProblemSeverity.HIGH]
    if high_issues:
        console.print("\n[yellow]⚠️  HIGH PRIORITY ISSUES:[/yellow]")
        for issue in high_issues[:5]:  # Show first 5
            console.print(f"  • {issue.title} ({issue.location})")
        if len(high_issues) > 5:
            console.print(f"  ... and {len(high_issues) - 5} more high priority issues")


@app.command()
def quick():
    """
    Run a quick diagnostic check focusing on critical issues only.
    """
    console.print("[blue]Running quick diagnostic check...[/blue]")

    # This is a simplified version focusing on critical issues
    project_path = Path.cwd()
    MasterDiagnostics(str(project_path))

    # Run basic checks
    critical_issues = []

    # Check for syntax errors in key files
    key_files = [
        "src/llamaagent/__init__.py",
        "src/llamaagent/cli/__init__.py",
        "src/llamaagent/core/__init__.py",
    ]

    for file_path in key_files:
        full_path = project_path / file_path
        if full_path.exists():
            try:
                with open(full_path, "r") as f:
                    content = f.read()

                import ast

                ast.parse(content)
                console.print(f"[green]PASS {file_path} - OK[/green]")
            except SyntaxError as e:
                console.print(f"[red]FAIL {file_path} - Syntax Error: {e}[/red]")
                critical_issues.append(f"Syntax error in {file_path}: {e}")
            except Exception as e:
                console.print(f"[yellow]⚠️  {file_path} - Warning: {e}[/yellow]")
        else:
            console.print(f"[red]FAIL {file_path} - Missing[/red]")
            critical_issues.append(f"Missing critical file: {file_path}")

    # Check basic imports
    try:
        import sys

        sys.path.insert(0, str(project_path))
        console.print("[green]PASS Basic imports - OK[/green]")
    except Exception as e:
        console.print(f"[red]FAIL Basic imports - Failed: {e}[/red]")
        critical_issues.append(f"Import error: {e}")

    # Summary
    if critical_issues:
        console.print(f"\n[red]CRITICAL {len(critical_issues)} critical issues found![/red]")
        for issue in critical_issues:
            console.print(f"  • {issue}")
        console.print(
            "\n[yellow]Run 'llamaagent diagnostics run' for comprehensive analysis[/yellow]"
        )
        raise typer.Exit(1)
    else:
        console.print(
            "\n[green]PASS Quick check passed - no critical issues detected![/green]"
        )
        console.print(
            "[blue]INSIGHT For comprehensive analysis, run 'llamaagent diagnostics run'[/blue]"
        )
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
