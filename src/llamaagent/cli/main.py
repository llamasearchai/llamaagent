from typing import Any, List

#!/usr/bin/env python3
"""
Main CLI for LlamaAgent
Comprehensive command-line interface for LLM and agent management.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..api.main import run_server
from ..config.settings import get_settings
from .llm_cmd import llm_cli

console = Console()
app = typer.Typer(
    name="llamaagent",
    help="LlamaAgent - Comprehensive LLM and AI Agent Management Platform",
    add_completion=False,
)

# Add LLM CLI as a subcommand
app.add_typer(llm_cli, name="llm", help="LLM provider commands")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the FastAPI server."""
    console.print(
        Panel(
            f"Starting LlamaAgent API server at http://{host}:{port}",
            title="Server Starting",
            border_style="green",
        )
    )

    run_server(host=host, port=port, reload=reload, workers=workers)


@app.command()
def datasette(
    port: int = typer.Option(8001, help="Port for Datasette server"),
    host: str = typer.Option("127.0.0.1", help="Host for Datasette server"),
    db_path: str = typer.Option("llamaagent.db", help="Database path"),
):
    """Start Datasette server for data exploration."""

    # Check if database exists
    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        console.print("Run some LLM commands first to create the database.")
        raise typer.Exit(1)

    # Create metadata file for Datasette
    metadata = {
        "title": "LlamaAgent Data Explorer",
        "description": "Explore LLM conversations, embeddings, and analytics",
        "databases": {
            "llamaagent": {
                "tables": {
                    "conversations": {
                        "description": "LLM conversation history with full-text search",
                        "plugins": {
                            "datasette-vega": {
                                "line_chart": {"x": "timestamp", "y": "cost"}
                            }
                        },
                    },
                    "embeddings": {
                        "description": "Text embeddings for similarity search"
                    },
                    "knowledge_base": {
                        "description": "Knowledge base with full-text search"
                    },
                }
            }
        },
        "plugins": {
            "datasette-vega": {},
            "datasette-cluster-map": {},
            "datasette-json-html": {},
        },
    }

    import json

    metadata_path = "datasette-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(
        Panel(
            f"Starting Datasette server at http://{host}:{port}",
            title="Datasette Starting",
            border_style="blue",
        )
    )

    # Start Datasette
    cmd = [
        "datasette",
        db_path,
        "--host",
        host,
        "--port",
        str(port),
        "--metadata",
        metadata_path,
        "--setting",
        "sql_time_limit_ms",
        "10000",
        "--setting",
        "max_returned_rows",
        "10000",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start Datasette: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Datasette server stopped[/yellow]")


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option(
        "basic", help="Project template (basic, advanced, research)"
    ),
):
    """Initialize a new LlamaAgent project."""

    project_path = Path(project_name)
    if project_path.exists():
        console.print(f"[red]Directory {project_name} already exists[/red]")
        raise typer.Exit(1)

    # Create project structure
    project_path.mkdir(parents=True)

    # Create basic structure
    dirs = ["agents", "data", "configs", "logs", "exports"]

    for dir_name in dirs:
        (project_path / dir_name).mkdir()

    # Create configuration files
    config_content = f"""# LlamaAgent Configuration for {project_name}
# Author: Nik Jois <nikjois@llamasearch.ai>

# API Keys (set these in your environment)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here
TOGETHER_API_KEY=your_together_key_here

# Database Configuration
DATABASE_URL=sqlite:///./data/llamaagent.db
VECTOR_DB_PATH=./data/vector_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your_secret_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/llamaagent.log

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
"""

    with open(project_path / ".env", "w") as f:
        f.write(config_content)

    # Create README
    readme_content = f"""# {project_name}

LlamaAgent project initialized with template: {template}

## Quick Start

1. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. Start the API server:
   ```bash
   llamaagent serve
   ```

3. Use the LLM CLI:
   ```bash
   llamaagent llm providers
   llamaagent llm chat openai gpt-4 "Hello, world!"
   ```

4. Explore your data:
   ```bash
   llamaagent datasette
   ```

## Project Structure

- `agents/` - Custom agent implementations
- `data/` - Database and data files
- `configs/` - Configuration files
- `logs/` - Application logs
- `exports/` - Data exports

## Documentation

- API Documentation: http://localhost:8000/docs
- Data Explorer: http://localhost:8001

## Author

Nik Jois <nikjois@llamasearch.ai>
"""

    with open(project_path / "README.md", "w") as f:
        f.write(readme_content)

    # Create example agent (for advanced template)
    if template == "advanced":
        agent_content = '''"""
Example LlamaAgent implementation.
"""

from llamaagent import BaseAgent, LLMProvider

class ExampleAgent(BaseAgent):
    """Example agent that demonstrates LlamaAgent capabilities."""
    
    def __init__(self, llm_provider: LLMProvider) -> None:
        super().__init__(llm_provider)
        self.name = "ExampleAgent"
    
    async def process_task(self, task: str) -> str:
        """Process a task using the LLM."""
        prompt = f"Task: {task}\\n\\nPlease provide a detailed response:"
        
        response = await self.llm_provider.generate_response(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.content
'''

        with open(project_path / "agents" / "example_agent.py", "w") as f:
            f.write(agent_content)

    console.print(
        Panel(
            f"Project '{project_name}' initialized successfully!\\n\\n"
            f"Next steps:\\n"
            f"1. cd {project_name}\\n"
            f"2. Edit .env with your API keys\\n"
            f"3. llamaagent serve",
            title="Project Created",
            border_style="green",
        )
    )


@app.command()
def status() -> None:
    """Show system status and configuration."""

    get_settings()

    # Create status table
    table = Table(title="LlamaAgent System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="yellow")

    # Check API keys
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Cohere": os.getenv("COHERE_API_KEY"),
        "Together": os.getenv("TOGETHER_API_KEY"),
    }

    for provider, key in api_keys.items():
        status = "✓ Configured" if key else "✗ Missing"
        details = "API key set" if key else "Set in environment"
        table.add_row(f"{provider} API", status, details)

    # Check database
    db_path = Path("llamaagent.db")
    db_status = "✓ Exists" if db_path.exists() else "✗ Not found"
    db_details = f"Size: {db_path.stat().st_size if db_path.exists() else 0} bytes"
    table.add_row("Database", db_status, db_details)

    # Check dependencies
    dependencies = [
        ("FastAPI", "fastapi"),
        ("SQLite Utils", "sqlite_utils"),
        ("Datasette", "datasette"),
        ("ChromaDB", "chromadb"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            status = "✓ Installed"
            details = "Available"
        except ImportError:
            status = "✗ Missing"
            details = f"pip install {module}"

        table.add_row(f"{name}", status, details)

    console.print(table)


@app.command()
def benchmark(
    provider: str = typer.Option("openai", help="LLM provider to benchmark"),
    model: str = typer.Option("gpt-3.5-turbo", help="Model to benchmark"),
    iterations: int = typer.Option(10, help="Number of test iterations"),
    prompt: str = typer.Option("What is artificial intelligence?", help="Test prompt"),
):
    """Run performance benchmarks."""

    async def run_benchmark() -> None:
        import time

        from ..llm.factory import LLMFactory

        console.print(
            Panel(
                f"Running benchmark: {provider}/{model}\\nIterations: {iterations}\\nPrompt: {prompt[:50]}...",
                title="Benchmark Starting",
                border_style="yellow",
            )
        )

        factory = LLMFactory()

        try:
            llm_provider = await factory.create_provider(
                provider_type=provider, model_name=model
            )

            times: List[Any] = []
            costs: List[Any] = []
            token_counts: List[Any] = []

            for i in range(iterations):
                console.print(f"[blue]Running iteration {i + 1}/{iterations}...[/blue]")

                start_time = time.time()
                response = await llm_provider.generate_response(
                    prompt=prompt, max_tokens=100, temperature=0.7
                )
                end_time = time.time()

                duration = end_time - start_time
                times.append(duration)

                if response.cost:
                    costs.append(response.cost)

                if response.usage and "total_tokens" in response.usage:
                    token_counts.append(response.usage["total_tokens"])

            # Calculate statistics
            avg_time = sum(times) / len(times)
            avg_cost = sum(costs) / len(costs) if costs else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

            # Display results
            results_table = Table(title="Benchmark Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="magenta")

            results_table.add_row("Provider/Model", f"{provider}/{model}")
            results_table.add_row("Iterations", str(iterations))
            results_table.add_row("Average Response Time", f"{avg_time:.2f}s")
            results_table.add_row("Average Cost", f"${avg_cost:.4f}")
            results_table.add_row("Average Tokens", f"{avg_tokens:.0f}")
            results_table.add_row("Tokens per Second", f"{avg_tokens / avg_time:.1f}")

            console.print(results_table)

        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")

    asyncio.run(run_benchmark())


@app.command()
def version() -> None:
    """Show version information."""

    console.print(
        Panel(
            "LlamaAgent v1.0.0\\n"
            "Comprehensive LLM and AI Agent Management Platform\\n\\n"
            "Author: Nik Jois <nikjois@llamasearch.ai>\\n"
            "License: MIT",
            title="Version Information",
            border_style="blue",
        )
    )


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
