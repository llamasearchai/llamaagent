#!/usr/bin/env python3
"""
LLM Command-Line Interface for LlamaAgent
Comprehensive CLI for interacting with various LLM providers and managing data.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from llamaagent.llm.factory import LLMFactory
logger = logging.getLogger(__name__)


class LLMDatabase:
    """Database for storing LLM interactions"""

    def __init__(self, db_path: str = "llamaagent.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tokens_used INTEGER,
                cost REAL,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS providers (
                name TEXT PRIMARY KEY,
                models TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """
        )

        self.conn.commit()

    def save_conversation(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save a conversation to the database"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations 
            (timestamp, provider, model, prompt, response, tokens_used, cost, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now(timezone.utc).isoformat(),
                provider,
                model,
                prompt,
                response,
                tokens_used,
                cost,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        cursor = self.conn.cursor()

        # Total stats
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_requests,
                SUM(tokens_used) as total_tokens,
                SUM(cost) as total_cost
            FROM conversations
        """
        )
        total_stats = cursor.fetchone()

        # Per provider stats
        cursor.execute(
            """
            SELECT 
                provider,
                COUNT(*) as requests,
                SUM(tokens_used) as tokens,
                SUM(cost) as cost,
                AVG(cost) as avg_cost
            FROM conversations
            GROUP BY provider
        """
        )
        provider_stats = cursor.fetchall()

        return {
            "total": {
                "requests": total_stats[0] or 0,
                "tokens": total_stats[1] or 0,
                "cost": total_stats[2] or 0.0,
            },
            "by_provider": [
                {
                    "provider": row[0],
                    "requests": row[1],
                    "tokens": row[2] or 0,
                    "cost": row[3] or 0.0,
                    "avg_cost": row[4] or 0.0,
                }
                for row in provider_stats
            ],
        }

    def export_to_csv(self, output_path: str):
        """Export conversations to CSV"""
        df = pd.read_sql_query("SELECT * FROM conversations", self.conn)
        df.to_csv(output_path, index=False)

    def close(self):
        """Close database connection"""
        self.conn.close()


class LLMCli:
    """LLM Command Line Interface"""

    def __init__(self):
        self.console = Console()
        self.db = LLMDatabase()
        self.factory = LLMFactory()
        self.providers = self._get_available_providers()

    def _get_available_providers(self) -> Dict[str, List[str]]:
        """Get available providers and their models"""
        providers = {}

        # OpenAI
        providers["openai"] = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

        # Anthropic
        providers["anthropic"] = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
        ]

        # Together AI
        providers["together"] = [
            "meta-llama/Llama-2-70b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        ]

        # Cohere
        providers["cohere"] = ["command", "command-light", "command-nightly"]

        # Ollama (if available)
        try:
            providers["ollama"] = self._get_ollama_models()
        except Exception as e:
            logger.error(f"Error: {e}")

        return providers

    def _get_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            import ollama

            models = ollama.list()
            return [model["name"] for model in models.get("models", [])]
        except:
            return []

    async def chat(
        self, provider: str, model: str, system_prompt: Optional[str] = None
    ):
        """Interactive chat session"""
        self.console.print(
            f"[bold green]Starting chat with {provider}/{model}[/bold green]"
        )
        if system_prompt:
            self.console.print(f"[dim]System: {system_prompt}[/dim]")

        # Create provider
        try:
            llm = self.factory.create(provider, model=model)
        except Exception as e:
            self.console.print(f"[red]Error creating provider: {e}[/red]")
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.lower() in ["/exit", "/quit"]:
                break
            elif user_input.lower() == "/clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                self.console.print("[yellow]Conversation cleared[/yellow]")
                continue
            elif user_input.lower() == "/save":
                filename = Prompt.ask("Save to file", default="conversation.json")
                with open(filename, "w") as f:
                    json.dump(messages, f, indent=2)
                self.console.print(f"[green]Saved to {filename}[/green]")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Get response
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Thinking...", total=None)

                try:
                    response = await llm.complete(messages)
                    assistant_msg = response.content

                    # Save to database
                    self.db.save_conversation(
                        provider=provider,
                        model=model,
                        prompt=user_input,
                        response=assistant_msg,
                        tokens_used=getattr(response, "tokens_used", None),
                        cost=getattr(response, "cost", None),
                    )

                    messages.append({"role": "assistant", "content": assistant_msg})

                    progress.update(task, completed=True)

                    # Display response
                    self.console.print(
                        f"\n[bold green]Assistant[/bold green]: {assistant_msg}"
                    )

                    # Show usage if available
                    usage_text = ""
                    if hasattr(response, "tokens_used") and response.tokens_used:
                        usage_text += f"Tokens: {response.tokens_used} | "
                    if hasattr(response, "cost") and response.cost:
                        usage_text += f"Cost: ${response.cost:.4f} | "

                    if usage_text:
                        self.console.print(f"[dim]{usage_text[:-3]}[/dim]")

                except Exception as e:
                    progress.update(task, completed=True)
                    self.console.print(f"[red]Error: {e}[/red]")

    def list_providers(self):
        """List available providers and models"""
        table = Table(title="Available LLM Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Models", style="green")

        for provider, models in self.providers.items():
            models_str = ", ".join(models[:3])  # Show first 3 models
            if len(models) > 3:
                models_str += f" (+{len(models) - 3} more)"
            table.add_row(provider, models_str)

        self.console.print(table)

    def show_statistics(self):
        """Show usage statistics"""
        stats = self.db.get_statistics()

        # Total stats
        table = Table(title="LLM Usage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(stats["total"]["requests"]))
        table.add_row("Total Tokens", f"{stats['total']['tokens']:,}")
        table.add_row("Total Cost", f"${stats['total']['cost']:.2f}")

        self.console.print(table)

        # Per provider stats
        if stats["by_provider"]:
            provider_table = Table(title="Usage by Provider")
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Requests", style="yellow")
            provider_table.add_column("Tokens", style="green")
            provider_table.add_column("Total Cost", style="red")
            provider_table.add_column("Avg Cost", style="magenta")

            for provider_stat in stats["by_provider"]:
                provider_table.add_row(
                    provider_stat["provider"],
                    str(provider_stat["requests"]),
                    f"{provider_stat['tokens']:,}",
                    f"${provider_stat['cost']:.2f}",
                    f"${provider_stat['avg_cost']:.4f}",
                )

            self.console.print(provider_table)

    def export_data(self, format: str = "csv"):
        """Export conversation data"""
        if format == "csv":
            output_path = Prompt.ask("Output file", default="conversations.csv")
            self.db.export_to_csv(output_path)
            self.console.print(f"[green]Exported to {output_path}[/green]")
        elif format == "json":
            output_path = Prompt.ask("Output file", default="conversations.json")
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT * FROM conversations")
            rows = cursor.fetchall()

            data = []
            for row in rows:
                data.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "provider": row[2],
                        "model": row[3],
                        "prompt": row[4],
                        "response": row[5],
                        "tokens_used": row[6],
                        "cost": row[7],
                        "metadata": json.loads(row[8]) if row[8] else None,
                    }
                )

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            self.console.print(f"[green]Exported to {output_path}[/green]")


@click.group()
def cli():
    """LlamaAgent LLM Command Line Interface"""


@cli.command()
@click.option(
    "--provider", "-p", required=True, help="LLM provider (openai, anthropic, etc.)"
)
@click.option("--model", "-m", help="Model name (defaults to provider's default)")
@click.option("--system", "-s", help="System prompt")
def chat(provider: str, model: Optional[str], system: Optional[str]):
    """Start an interactive chat session"""
    llm_cli = LLMCli()

    # Use default model if not specified
    if not model:
        provider_models = llm_cli.providers.get(provider, [])
        if not provider_models:
            click.echo(f"Provider '{provider}' not found or has no models")
            return
        model = provider_models[0]

    asyncio.run(llm_cli.chat(provider, model, system))


@cli.command()
def list():
    """List available providers and models"""
    llm_cli = LLMCli()
    llm_cli.list_providers()


@cli.command()
def stats():
    """Show usage statistics"""
    llm_cli = LLMCli()
    llm_cli.show_statistics()


@cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export format",
)
def export(format: str):
    """Export conversation history"""
    llm_cli = LLMCli()
    llm_cli.export_data(format)


@cli.command()
@click.argument("prompt")
@click.option("--provider", "-p", default="openai", help="LLM provider")
@click.option("--model", "-m", help="Model name")
@click.option("--output", "-o", help="Output file (optional)")
def query(prompt: str, provider: str, model: Optional[str], output: Optional[str]):
    """Make a single query to an LLM"""
    llm_cli = LLMCli()

    # Use default model if not specified
    if not model:
        provider_models = llm_cli.providers.get(provider, [])
        if not provider_models:
            click.echo(f"Provider '{provider}' not found or has no models")
            return
        model = provider_models[0]

    async def run_query():
        try:
            llm = llm_cli.factory.create(provider, model=model)
            response = await llm.complete([{"role": "user", "content": prompt}])

            # Save to database
            llm_cli.db.save_conversation(
                provider=provider,
                model=model,
                prompt=prompt,
                response=response.content,
                tokens_used=getattr(response, "tokens_used", None),
                cost=getattr(response, "cost", None),
            )

            # Output response
            if output:
                with open(output, "w") as f:
                    f.write(response.content)
                click.echo(f"Response saved to {output}")
            else:
                click.echo(response.content)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    asyncio.run(run_query())


if __name__ == "__main__":
    cli()
