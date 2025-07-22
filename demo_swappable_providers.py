#!/usr/bin/env python3
"""
LlamaAgent Swappable Provider Demo
=================================

Comprehensive demonstration of the swappable LLM provider system:
- Mock provider for testing
- OpenAI provider with API key configuration
- Local Llama provider with transformers
- Provider switching and configuration
- End-to-end task execution with different providers

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import os
import logging
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# LlamaAgent imports
from src.llamaagent.agents.react import ReactAgent
from src.llamaagent.agents.base import AgentConfig, AgentRole
from src.llamaagent.llm.providers import (
    create_provider, 
    ProviderType, 
    LLMConfig,
    MockProvider
)
from src.llamaagent.tools import ToolRegistry, get_all_tools
from src.llamaagent.memory.base import SimpleMemory
from src.llamaagent.types import TaskInput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class ProviderDemo:
    """Demo class for testing swappable providers."""
    
    def __init__(self):
        self.console = console
        self.providers: Dict[str, Any] = {}
        self.agents: Dict[str, ReactAgent] = {}
        self.tools = ToolRegistry()
        self.memory = SimpleMemory()
        
        # Initialize tools
        for tool in get_all_tools():
            self.tools.register(tool)
    
    def show_banner(self):
        """Display demo banner."""
        banner = """
    
                        LlamaAgent Swappable Provider Demo                         
                                                                                   
      • Mock Provider - Always available for testing                              
      • OpenAI Provider - GPT models via official API                             
      • Local Llama Provider - Transformers-based local models                    
                                                                                   
      Author: Nik Jois <nikjois@llamasearch.ai>                                   
    
        """
        
        self.console.print(Panel(
            banner,
            title="Provider Demo",
            border_style="cyan"
        ))
    
    async def initialize_providers(self):
        """Initialize all available providers."""
        self.console.print("\n[bold]Initializing Providers...[/bold]")
        
        provider_configs = [
            {
                "name": "mock",
                "provider_type": "mock",
                "model_name": "mock-model"
            },
            {
                "name": "openai",
                "provider_type": "openai",
                "model_name": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "mock-key-for-demo")
            }
        ]
        
        # Initialize providers
        for provider_info in provider_configs:
            try:
                provider = create_provider(**provider_info)
                
                # Check if provider has health_check method, otherwise assume available
                if hasattr(provider, 'health_check'):
                    available = await provider.health_check()
                else:
                    available = True
                
                if available:
                    self.providers[provider_info["name"]] = provider
                    self.console.print(f" [green]{provider_info['name']} provider initialized[/green]")
                else:
                    self.console.print(f" [yellow]{provider_info['name']} provider not available[/yellow]")
                    
            except Exception as e:
                self.console.print(f" [red]{provider_info['name']} provider failed: {e}[/red]")
        
        # Always ensure mock provider is available
        if "mock" not in self.providers:
            self.providers["mock"] = create_provider("mock", model_name="mock-model")
            self.console.print(" [green]Mock provider added as fallback[/green]")
    
    async def create_test_agents(self):
        """Create test agents with different providers."""
        self.console.print("\n[bold]Creating Test Agents...[/bold]")
        
        agent_configs = [
            ("mock_agent", "mock", "Agent using mock provider"),
            ("openai_agent", "openai", "Agent using OpenAI provider")
        ]
        
        for agent_name, provider_name, description in agent_configs:
            if provider_name in self.providers:
                try:
                    config = AgentConfig(
                        name=agent_name,
                        role=AgentRole.GENERALIST,
                        description=description,
                        max_iterations=5,
                        temperature=0.7
                    )
                    
                    agent = ReactAgent(
                        config=config,
                        llm_provider=self.providers[provider_name],
                        tools=self.tools,
                        memory=self.memory
                    )
                    
                    self.agents[agent_name] = agent
                    self.console.print(f" [green]Created {agent_name} with {provider_name} provider[/green]")
                    
                except Exception as e:
                    self.console.print(f" [red]Failed to create {agent_name}: {e}[/red]")
            else:
                self.console.print(f" [yellow]Skipping {agent_name} - {provider_name} not available[/yellow]")
    
    async def run_provider_tests(self):
        """Run tests with different providers."""
        self.console.print("\n[bold]Running Provider Tests...[/bold]")
        
        test_scenarios = [
            {
                "name": "Simple Math",
                "task": "What is 15 + 27?",
                "expected_type": "calculation"
            },
            {
                "name": "Creative Writing",
                "task": "Write a short poem about artificial intelligence",
                "expected_type": "creative"
            },
            {
                "name": "Problem Solving",
                "task": "How would you organize a small library?",
                "expected_type": "analytical"
            },
            {
                "name": "Code Generation",
                "task": "Write a Python function to calculate fibonacci numbers",
                "expected_type": "technical"
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            self.console.print(f"\n[bold cyan]Test: {scenario['name']}[/bold cyan]")
            self.console.print(f"Task: {scenario['task']}")
            
            scenario_results = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn(f"Testing {agent_name}..."),
                        console=self.console
                    ) as progress:
                        task = progress.add_task("processing", total=None)
                        
                        # Execute task
                        response = await agent.execute(scenario['task'])
                        
                        scenario_results[agent_name] = {
                            "success": response.success,
                            "content": response.content[:100] + "..." if len(response.content) > 100 else response.content,
                            "tokens": response.tokens_used,
                            "time": response.execution_time
                        }
                        
                        progress.update(task, completed=True)
                        
                        status = "" if response.success else ""
                        self.console.print(f"  {status} {agent_name}: {response.content[:50]}...")
                        
                except Exception as e:
                    scenario_results[agent_name] = {
                        "success": False,
                        "error": str(e),
                        "content": f"Error: {e}",
                        "tokens": 0,
                        "time": 0
                    }
                    self.console.print(f"   {agent_name}: Error - {e}")
            
            results[scenario['name']] = scenario_results
        
        return results
    
    async def show_performance_comparison(self, results: Dict[str, Any]):
        """Show performance comparison between providers."""
        self.console.print("\n[bold]Performance Comparison[/bold]")
        
        # Create comparison table
        table = Table(title="Provider Performance Summary", show_header=True)
        table.add_column("Test Scenario", style="cyan")
        table.add_column("Mock Provider", style="green")
        table.add_column("OpenAI Provider", style="blue")
        table.add_column("Llama Local", style="magenta")
        
        for scenario_name, scenario_results in results.items():
            row = [scenario_name]
            
            for provider in ["mock_agent", "openai_agent", "llama_agent"]:
                if provider in scenario_results:
                    result = scenario_results[provider]
                    if result["success"]:
                        status = f" ({result['tokens']} tokens)"
                    else:
                        status = " Failed"
                else:
                    status = "N/A"
                
                row.append(status)
            
            table.add_row(*row)
        
        self.console.print(table)
        
        # Show detailed statistics
        stats_table = Table(title="Detailed Statistics", show_header=True)
        stats_table.add_column("Provider", style="cyan")
        stats_table.add_column("Success Rate", style="green")
        stats_table.add_column("Avg Tokens", style="yellow")
        stats_table.add_column("Avg Time (s)", style="blue")
        
        for provider in ["mock_agent", "openai_agent", "llama_agent"]:
            successes = 0
            total_tests = 0
            total_tokens = 0
            total_time = 0
            
            for scenario_results in results.values():
                if provider in scenario_results:
                    result = scenario_results[provider]
                    total_tests += 1
                    if result["success"]:
                        successes += 1
                        total_tokens += result["tokens"]
                        total_time += result["time"]
            
            if total_tests > 0:
                success_rate = f"{(successes/total_tests)*100:.1f}%"
                avg_tokens = f"{total_tokens/total_tests:.1f}" if total_tests > 0 else "0"
                avg_time = f"{total_time/total_tests:.2f}" if total_tests > 0 else "0"
                
                stats_table.add_row(provider, success_rate, avg_tokens, avg_time)
        
        self.console.print(stats_table)
    
    async def demonstrate_provider_switching(self):
        """Demonstrate switching between providers."""
        self.console.print("\n[bold]Provider Switching Demo[/bold]")
        
        # Create a single agent and switch its provider
        config = AgentConfig(
            name="switchable_agent",
            role=AgentRole.GENERALIST,
            description="Agent that can switch providers",
            max_iterations=3
        )
        
        test_task = "Explain the concept of machine learning in simple terms"
        
        for provider_name, provider in self.providers.items():
            self.console.print(f"\n[bold]Testing with {provider_name} provider:[/bold]")
            
            try:
                # Create agent with this provider
                agent = ReactAgent(
                    config=config,
                    llm_provider=provider,
                    tools=self.tools,
                    memory=self.memory
                )
                
                # Execute task
                response = await agent.execute(test_task)
                
                self.console.print(f"Response: {response.content[:200]}...")
                self.console.print(f"Success: {response.success}, Tokens: {response.tokens_used}")
                
                # Cleanup
                await agent.cleanup()
                
            except Exception as e:
                self.console.print(f"[red]Error with {provider_name}: {e}[/red]")
    
    async def show_provider_capabilities(self):
        """Show capabilities of each provider."""
        self.console.print("\n[bold]Provider Capabilities[/bold]")
        
        capabilities_table = Table(title="Provider Feature Matrix", show_header=True)
        capabilities_table.add_column("Feature", style="cyan")
        capabilities_table.add_column("Mock", style="green")
        capabilities_table.add_column("OpenAI", style="blue")
        capabilities_table.add_column("Llama Local", style="magenta")
        
        features = [
            ("Always Available", "", "API Key Required", "Libraries Required"),
            ("Streaming Support", "", "", "Simulated"),
            ("Cost", "Free", "Pay per token", "Hardware cost"),
            ("Latency", "Low", "Network dependent", "Hardware dependent"),
            ("Privacy", "Complete", "Sent to OpenAI", "Complete"),
            ("Customization", "Limited", "Limited", "Full control"),
            ("Model Selection", "Fixed", "Multiple GPT models", "Any HF model"),
            ("Offline Support", "", "", "")
        ]
        
        for feature, mock, openai, llama in features:
            capabilities_table.add_row(feature, mock, openai, llama)
        
        self.console.print(capabilities_table)
    
    async def cleanup(self):
        """Cleanup all resources."""
        self.console.print("\n[bold]Cleaning up...[/bold]")
        
        # Cleanup agents
        for agent in self.agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up agent: {e}")
        
        # Cleanup providers
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up provider: {e}")
        
        self.console.print("[green]Cleanup complete[/green]")
    
    async def run_demo(self):
        """Run the complete demo."""
        try:
            self.show_banner()
            
            await self.initialize_providers()
            await self.create_test_agents()
            
            self.show_provider_capabilities()
            
            results = await self.run_provider_tests()
            await self.show_performance_comparison(results)
            
            await self.demonstrate_provider_switching()
            
            self.console.print("\n[bold green]Demo completed successfully![/bold green]")
            
        except Exception as e:
            self.console.print(f"\n[bold red]Demo failed: {e}[/bold red]")
            logger.exception("Demo error")
        finally:
            await self.cleanup()


async def main():
    """Main demo entry point."""
    demo = ProviderDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 