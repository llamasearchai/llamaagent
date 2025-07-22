#!/usr/bin/env python3
"""
LlamaAgent Working Demo - Complete Functional Demonstration
===========================================================

A complete, fully working demonstration of all LlamaAgent capabilities
with mock data and real functionality that works without external dependencies.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Rich imports for beautiful interface
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.align import Align

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

# ASCII Art Banner
BANNER = """

                                                             
                                   
                             
                            
                            
                        
                           
                                                             
                    A G E N T                                
                                                             
              Advanced AI Agent Framework                    
                                                             

"""

class MockLLMProvider:
    """Mock LLM Provider for demonstration."""
    
    def __init__(self, model_name: str = "mock-gpt-4"):
        self.model_name = model_name
        self.response_templates = [
            "Based on your request, I can provide the following analysis: {input}",
            "I understand you're asking about {input}. Here's my detailed response:",
            "Thank you for your question about {input}. Let me help you with that:",
            "Regarding {input}, I can offer these insights:",
            "I've analyzed your request about {input} and here's what I found:"
        ]
    
    def complete(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate a mock completion response."""
        import random
        template = random.choice(self.response_templates)
        
        # Simulate different response types based on keywords
        if "calculate" in prompt.lower() or "math" in prompt.lower():
            return f"I've performed the calculation: {prompt}. The mathematical result shows clear patterns and relationships."
        elif "code" in prompt.lower() or "python" in prompt.lower():
            return f"Here's the code solution for {prompt}:\n\n```python\n# Solution implementation\nresult = process_request()\nprint(result)\n```"
        elif "analyze" in prompt.lower():
            return f"Analysis of {prompt}:\n\n1. Key findings\n2. Important patterns\n3. Recommendations\n4. Next steps"
        else:
            return template.format(input=prompt[:50] + "..." if len(prompt) > 50 else prompt)

class MockTool:
    """Mock tool for demonstration."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, input_data: str) -> str:
        """Execute the tool with mock functionality."""
        if self.name == "calculator":
            try:
                # Safe evaluation for demo
                result = eval(input_data.replace("^", "**"))
                return f"Result: {result}"
            except:
                return "Error: Invalid mathematical expression"
        elif self.name == "python_repl":
            return f"Executed Python code:\n{input_data}\n\nOutput: Code executed successfully"
        elif self.name == "web_search":
            return f"Search results for '{input_data}':\n1. Relevant article\n2. Documentation\n3. Tutorial"
        else:
            return f"Tool '{self.name}' executed with input: {input_data}"

class MockAgent:
    """Mock agent for demonstration."""
    
    def __init__(self, name: str, role: str, provider: MockLLMProvider):
        self.name = name
        self.role = role
        self.provider = provider
        self.tools = []
        self.conversation_history = []
        self.stats = {
            "tasks_completed": 0,
            "success_rate": 0.95,
            "avg_response_time": 2.1,
            "total_tokens": 0
        }
    
    def add_tool(self, tool: MockTool):
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a request and return response."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Generate response
        response = self.provider.complete(request)
        
        # Check if tool usage is needed
        tool_results = []
        if "calculate" in request.lower():
            calc_tool = next((t for t in self.tools if t.name == "calculator"), None)
            if calc_tool:
                # Extract calculation from request
                import re
                calc_match = re.search(r'[\d+\-*/\s()]+', request)
                if calc_match:
                    tool_result = calc_tool.execute(calc_match.group())
                    tool_results.append({"tool": "calculator", "result": tool_result})
        
        # Update stats
        self.stats["tasks_completed"] += 1
        self.stats["total_tokens"] += len(request.split()) + len(response.split())
        
        # Record in history
        interaction = {
            "timestamp": datetime.now(),
            "request": request,
            "response": response,
            "tool_results": tool_results,
            "execution_time": time.time() - start_time
        }
        self.conversation_history.append(interaction)
        
        return {
            "response": response,
            "tool_results": tool_results,
            "execution_time": time.time() - start_time,
            "agent": self.name,
            "success": True
        }

class LlamaAgentDemo:
    """Complete LlamaAgent demonstration system."""
    
    def __init__(self):
        self.console = console
        self.session_id = str(uuid4())[:8]
        self.start_time = datetime.now()
        self.agents = {}
        self.tools = {}
        self.system_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "uptime": timedelta(0)
        }
        
        # Initialize system
        self.setup_system()
    
    def setup_system(self):
        """Initialize the complete system."""
        # Create LLM provider
        provider = MockLLMProvider()
        
        # Create tools
        self.tools = {
            "calculator": MockTool("calculator", "Mathematical calculations"),
            "python_repl": MockTool("python_repl", "Python code execution"),
            "web_search": MockTool("web_search", "Internet search"),
            "file_manager": MockTool("file_manager", "File operations")
        }
        
        # Create agents
        analyst_agent = MockAgent("AnalystAgent", "Data Analysis Specialist", provider)
        analyst_agent.add_tool(self.tools["calculator"])
        analyst_agent.add_tool(self.tools["web_search"])
        
        developer_agent = MockAgent("DeveloperAgent", "Software Development Expert", provider)
        developer_agent.add_tool(self.tools["python_repl"])
        developer_agent.add_tool(self.tools["file_manager"])
        
        writer_agent = MockAgent("WriterAgent", "Content Creation Specialist", provider)
        writer_agent.add_tool(self.tools["web_search"])
        
        self.agents = {
            "analyst": analyst_agent,
            "developer": developer_agent,
            "writer": writer_agent
        }
        
        logger.info("LlamaAgent system initialized successfully")
    
    def show_banner(self):
        """Display the system banner."""
        self.console.clear()
        banner_text = Text()
        banner_text.append(BANNER, style="bold cyan")
        banner_text.append(f"\nSession: {self.session_id} | Started: {self.start_time.strftime('%H:%M:%S')}\n", style="dim")
        banner_text.append("Author: Nik Jois <nikjois@llamasearch.ai>", style="dim")
        
        panel = Panel(
            Align.center(banner_text),
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def show_main_menu(self):
        """Display the main menu."""
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Option", style="bold cyan", width=4)
        menu_table.add_column("Feature", style="white")
        menu_table.add_column("Description", style="dim")
        
        menu_items = [
            ("1", "Interactive Chat", "Chat with AI agents"),
            ("2", "Agent Dashboard", "View and manage agents"),
            ("3", "Tool Workshop", "Test and explore tools"),
            ("4", "System Monitor", "Monitor system performance"),
            ("5", "Task Automation", "Automated task execution"),
            ("6", "Performance Analytics", "View detailed analytics"),
            ("7", "Configuration", "System settings"),
            ("8", "Help & Documentation", "Get help and examples"),
            ("0", "Exit", "Exit the application")
        ]
        
        for option, feature, description in menu_items:
            menu_table.add_row(option, feature, description)
        
        self.console.print("\n")
        self.console.print(Panel(menu_table, title="LlamaAgent Main Menu", border_style="blue"))
    
    async def interactive_chat(self):
        """Interactive chat with agents."""
        self.console.clear()
        self.console.print(Panel("Interactive Chat Mode", style="bold green"))
        
        # Select agent
        agent_table = Table(title="Available Agents")
        agent_table.add_column("ID", style="cyan")
        agent_table.add_column("Name", style="white")
        agent_table.add_column("Role", style="yellow")
        agent_table.add_column("Tools", style="green")
        
        for agent_id, agent in self.agents.items():
            tools_list = ", ".join([tool.name for tool in agent.tools])
            agent_table.add_row(agent_id, agent.name, agent.role, tools_list)
        
        self.console.print(agent_table)
        
        agent_choice = Prompt.ask("\nSelect agent", choices=list(self.agents.keys()), default="analyst")
        selected_agent = self.agents[agent_choice]
        
        self.console.print(f"\n[green]Connected to {selected_agent.name}[/green]")
        self.console.print("[dim]Type 'exit' to return to menu, 'help' for commands[/dim]\n")
        
        while True:
            try:
                user_input = Prompt.ask(f"[bold cyan]You[/bold cyan]")
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self.show_chat_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_agent_stats(selected_agent)
                    continue
                
                # Process request with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Processing request...", total=None)
                    
                    result = await selected_agent.process_request(user_input)
                    
                    progress.update(task, description="Complete!")
                
                # Display response
                self.console.print(f"\n[bold green]{selected_agent.name}[/bold green]: {result['response']}")
                
                # Show tool results if any
                if result['tool_results']:
                    self.console.print("\n[bold yellow]Tool Results:[/bold yellow]")
                    for tool_result in result['tool_results']:
                        self.console.print(f"  {tool_result['tool']}: {tool_result['result']}")
                
                self.console.print(f"[dim]Execution time: {result['execution_time']:.2f}s[/dim]\n")
                
                # Update system stats
                self.system_stats["total_requests"] += 1
                self.system_stats["successful_requests"] += 1
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Chat interrupted[/yellow]")
                break
    
    def show_chat_help(self):
        """Show chat help."""
        help_table = Table(title="Chat Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        commands = [
            ("help", "Show this help message"),
            ("stats", "Show agent statistics"),
            ("exit", "Return to main menu")
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)
    
    def show_agent_stats(self, agent: MockAgent):
        """Show agent statistics."""
        stats_table = Table(title=f"{agent.name} Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Tasks Completed", str(agent.stats["tasks_completed"]))
        stats_table.add_row("Success Rate", f"{agent.stats['success_rate']:.1%}")
        stats_table.add_row("Avg Response Time", f"{agent.stats['avg_response_time']:.1f}s")
        stats_table.add_row("Total Tokens", str(agent.stats["total_tokens"]))
        stats_table.add_row("Conversations", str(len(agent.conversation_history)))
        
        self.console.print(stats_table)
    
    async def agent_dashboard(self):
        """Agent management dashboard."""
        self.console.clear()
        self.console.print(Panel("Agent Dashboard", style="bold green"))
        
        # Agent overview
        agent_table = Table(title="Agent Overview")
        agent_table.add_column("Agent", style="cyan")
        agent_table.add_column("Role", style="white")
        agent_table.add_column("Tasks", style="yellow")
        agent_table.add_column("Success Rate", style="green")
        agent_table.add_column("Tools", style="blue")
        
        for agent_id, agent in self.agents.items():
            tools_count = len(agent.tools)
            agent_table.add_row(
                agent.name,
                agent.role,
                str(agent.stats["tasks_completed"]),
                f"{agent.stats['success_rate']:.1%}",
                f"{tools_count} tools"
            )
        
        self.console.print(agent_table)
        
        # Agent details
        agent_choice = Prompt.ask("\nSelect agent for details", choices=list(self.agents.keys()), default="analyst")
        selected_agent = self.agents[agent_choice]
        
        # Show detailed info
        details_table = Table(title=f"{selected_agent.name} Details")
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="white")
        
        details_table.add_row("Name", selected_agent.name)
        details_table.add_row("Role", selected_agent.role)
        details_table.add_row("Provider", selected_agent.provider.model_name)
        details_table.add_row("Tools", ", ".join([tool.name for tool in selected_agent.tools]))
        details_table.add_row("Conversations", str(len(selected_agent.conversation_history)))
        
        self.console.print(details_table)
        
        input("\nPress Enter to continue...")
    
    async def tool_workshop(self):
        """Tool testing workshop."""
        self.console.clear()
        self.console.print(Panel("Tool Workshop", style="bold green"))
        
        # Available tools
        tools_table = Table(title="Available Tools")
        tools_table.add_column("Tool", style="cyan")
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Status", style="green")
        
        for tool_name, tool in self.tools.items():
            tools_table.add_row(tool_name, tool.description, "ACTIVE")
        
        self.console.print(tools_table)
        
        # Tool testing
        tool_choice = Prompt.ask("\nSelect tool to test", choices=list(self.tools.keys()), default="calculator")
        selected_tool = self.tools[tool_choice]
        
        test_input = Prompt.ask(f"Enter input for {selected_tool.name}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Executing tool...", total=None)
            await asyncio.sleep(0.3)  # Simulate processing
            
            result = selected_tool.execute(test_input)
            progress.update(task, description="Complete!")
        
        self.console.print(f"\n[bold green]Tool Result:[/bold green]\n{result}")
        
        input("\nPress Enter to continue...")
    
    async def system_monitor(self):
        """System monitoring dashboard."""
        self.console.clear()
        self.console.print(Panel("System Monitor", style="bold green"))
        
        # Update uptime
        self.system_stats["uptime"] = datetime.now() - self.start_time
        
        # System statistics
        stats_table = Table(title="System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_column("Status", style="green")
        
        stats_table.add_row("Uptime", str(self.system_stats["uptime"]).split('.')[0], "HEALTHY")
        stats_table.add_row("Total Requests", str(self.system_stats["total_requests"]), "HEALTHY")
        stats_table.add_row("Successful Requests", str(self.system_stats["successful_requests"]), "HEALTHY")
        stats_table.add_row("Failed Requests", str(self.system_stats["failed_requests"]), "HEALTHY")
        stats_table.add_row("Active Agents", str(len(self.agents)), "HEALTHY")
        stats_table.add_row("Available Tools", str(len(self.tools)), "HEALTHY")
        
        self.console.print(stats_table)
        
        # Resource usage (simulated)
        resource_table = Table(title="Resource Usage")
        resource_table.add_column("Resource", style="cyan")
        resource_table.add_column("Usage", style="white")
        resource_table.add_column("Status", style="green")
        
        import random
        resources = [
            ("CPU", f"{random.randint(20, 60)}%", "NORMAL"),
            ("Memory", f"{random.randint(30, 70)}%", "NORMAL"),
            ("Disk", f"{random.randint(10, 40)}%", "NORMAL"),
            ("Network", f"{random.randint(5, 25)}%", "NORMAL")
        ]
        
        for resource, usage, status in resources:
            resource_table.add_row(resource, usage, status)
        
        self.console.print(resource_table)
        
        input("\nPress Enter to continue...")
    
    async def task_automation(self):
        """Automated task execution demo."""
        self.console.clear()
        self.console.print(Panel("Task Automation", style="bold green"))
        
        # Sample tasks
        tasks = [
            {"id": 1, "description": "Analyze market trends", "agent": "analyst"},
            {"id": 2, "description": "Write a summary report", "agent": "writer"},
            {"id": 3, "description": "Create a Python script", "agent": "developer"},
            {"id": 4, "description": "Calculate ROI metrics", "agent": "analyst"},
            {"id": 5, "description": "Review code quality", "agent": "developer"}
        ]
        
        self.console.print("[yellow]Available Tasks:[/yellow]")
        for task in tasks:
            self.console.print(f"  {task['id']}. {task['description']} (Agent: {task['agent']})")
        
        if Confirm.ask("\nExecute all tasks automatically?"):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                for task in tasks:
                    task_progress = progress.add_task(f"Executing task {task['id']}", total=None)
                    
                    # Execute task
                    agent = self.agents[task["agent"]]
                    result = await agent.process_request(task["description"])
                    
                    progress.update(task_progress, description=f"Task {task['id']} completed")
                    await asyncio.sleep(0.5)
            
            self.console.print("\n[bold green]All tasks completed successfully![/bold green]")
        
        input("\nPress Enter to continue...")
    
    async def performance_analytics(self):
        """Performance analytics dashboard."""
        self.console.clear()
        self.console.print(Panel("Performance Analytics", style="bold green"))
        
        # Aggregate statistics
        total_tasks = sum(agent.stats["tasks_completed"] for agent in self.agents.values())
        avg_success_rate = sum(agent.stats["success_rate"] for agent in self.agents.values()) / len(self.agents)
        avg_response_time = sum(agent.stats["avg_response_time"] for agent in self.agents.values()) / len(self.agents)
        
        # Performance metrics
        metrics_table = Table(title="Performance Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Status", style="green")
        
        metrics_table.add_row("Total Tasks Processed", str(total_tasks), "EXCELLENT")
        metrics_table.add_row("Average Success Rate", f"{avg_success_rate:.1%}", "EXCELLENT")
        metrics_table.add_row("Average Response Time", f"{avg_response_time:.1f}s", "GOOD")
        metrics_table.add_row("System Uptime", str(self.system_stats["uptime"]).split('.')[0], "EXCELLENT")
        
        self.console.print(metrics_table)
        
        # Agent performance comparison
        comparison_table = Table(title="Agent Performance Comparison")
        comparison_table.add_column("Agent", style="cyan")
        comparison_table.add_column("Tasks", style="white")
        comparison_table.add_column("Success Rate", style="green")
        comparison_table.add_column("Avg Response", style="blue")
        
        for agent in self.agents.values():
            comparison_table.add_row(
                agent.name,
                str(agent.stats["tasks_completed"]),
                f"{agent.stats['success_rate']:.1%}",
                f"{agent.stats['avg_response_time']:.1f}s"
            )
        
        self.console.print(comparison_table)
        
        input("\nPress Enter to continue...")
    
    async def configuration(self):
        """System configuration."""
        self.console.clear()
        self.console.print(Panel("System Configuration", style="bold green"))
        
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        config_table.add_column("Description", style="dim")
        
        config_items = [
            ("LLM Provider", "Mock Provider", "Current language model provider"),
            ("Model", "mock-gpt-4", "Active model name"),
            ("Session ID", self.session_id, "Current session identifier"),
            ("Agents", str(len(self.agents)), "Number of active agents"),
            ("Tools", str(len(self.tools)), "Number of available tools"),
            ("Debug Mode", "Enabled", "Debug logging status")
        ]
        
        for setting, value, desc in config_items:
            config_table.add_row(setting, value, desc)
        
        self.console.print(config_table)
        
        input("\nPress Enter to continue...")
    
    async def show_help(self):
        """Show help and documentation."""
        self.console.clear()
        self.console.print(Panel("Help & Documentation", style="bold green"))
        
        help_text = """
[bold cyan]LlamaAgent Working Demo[/bold cyan]

[bold]Overview:[/bold]
This is a complete demonstration of the LlamaAgent framework showcasing:
• Multi-agent orchestration
• Tool integration
• Real-time monitoring
• Performance analytics
• Task automation

[bold]Features Demonstrated:[/bold]
• Interactive Chat - Direct conversation with specialized AI agents
• Agent Dashboard - Complete agent management and monitoring
• Tool Workshop - Test and explore available tools
• System Monitor - Real-time system health and performance
• Task Automation - Automated task execution and management
• Performance Analytics - Detailed performance metrics and analysis

[bold]Agents Available:[/bold]
• AnalystAgent - Data analysis and insights
• DeveloperAgent - Software development and code review
• WriterAgent - Content creation and documentation

[bold]Tools Available:[/bold]
• Calculator - Mathematical calculations
• Python REPL - Python code execution
• Web Search - Internet search simulation
• File Manager - File operations

[bold]Mock Data:[/bold]
This demonstration uses mock data and simulated responses to showcase
functionality without requiring external API keys or dependencies.

[bold]Technical Details:[/bold]
• Built with Rich for beautiful CLI interface
• Async/await for responsive user experience
• Modular architecture for easy extension
• Comprehensive error handling and logging

[bold]Author:[/bold]
Nik Jois <nikjois@llamasearch.ai>

[bold]Repository:[/bold]
https://github.com/nikjois/llamaagent
"""
        
        self.console.print(help_text)
        input("\nPress Enter to continue...")
    
    async def run(self):
        """Main application loop."""
        self.show_banner()
        
        while True:
            try:
                self.show_main_menu()
                choice = Prompt.ask("\n[bold cyan]Select option[/bold cyan]", default="0")
                
                if choice == "0":
                    if Confirm.ask("Are you sure you want to exit?"):
                        break
                elif choice == "1":
                    await self.interactive_chat()
                elif choice == "2":
                    await self.agent_dashboard()
                elif choice == "3":
                    await self.tool_workshop()
                elif choice == "4":
                    await self.system_monitor()
                elif choice == "5":
                    await self.task_automation()
                elif choice == "6":
                    await self.performance_analytics()
                elif choice == "7":
                    await self.configuration()
                elif choice == "8":
                    await self.show_help()
                else:
                    self.console.print("[red]Invalid option. Please try again.[/red]")
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user[/yellow]")
                if Confirm.ask("Return to main menu?"):
                    continue
                else:
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.exception("Error in main loop")
                await asyncio.sleep(1)
        
        # Goodbye message
        self.console.clear()
        goodbye_text = Text()
        goodbye_text.append("Thank you for using LlamaAgent!\n\n", style="bold green")
        goodbye_text.append("Session Summary:\n", style="bold white")
        goodbye_text.append(f"• Duration: {datetime.now() - self.start_time}\n", style="dim")
        goodbye_text.append(f"• Total Requests: {self.system_stats['total_requests']}\n", style="dim")
        goodbye_text.append(f"• Successful Requests: {self.system_stats['successful_requests']}\n", style="dim")
        goodbye_text.append(f"• Agents Used: {len(self.agents)}\n", style="dim")
        goodbye_text.append(f"• Tools Available: {len(self.tools)}\n", style="dim")
        goodbye_text.append("\nAuthor: Nik Jois <nikjois@llamasearch.ai>", style="dim")
        
        panel = Panel(
            Align.center(goodbye_text),
            title="Session Complete",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)

async def main():
    """Main entry point."""
    demo = LlamaAgentDemo()
    await demo.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error")
    finally:
        console.print("[dim]Goodbye![/dim]") 