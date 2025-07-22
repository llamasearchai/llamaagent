#!/usr/bin/env python3
"""
LlamaAgent Master Program System Validation
Comprehensive validation to ensure the system is production-ready
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Color codes for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


class SystemValidator:
    """Validates the entire LlamaAgent Master Program system."""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
        
    def print_banner(self):
        """Print validation banner."""
        print(f"{BLUE}")
        print("")
        print("         LlamaAgent Master Program Validator               ")
        print("                                                           ")
        print("    Comprehensive System Validation & Health Check         ")
        print("")
        print(f"{NC}\n")
    
    def check(self, name: str, condition: bool, error_msg: str = "", warning: bool = False):
        """Add a check result."""
        if condition:
            self.results.append((name, True, ""))
            print(f"{GREEN}PASS {name}{NC}")
        else:
            if warning:
                self.warnings.append((name, error_msg))
                print(f"{YELLOW}WARNING:  {name}: {error_msg}{NC}")
            else:
                self.errors.append((name, error_msg))
                print(f"{RED}FAIL {name}: {error_msg}{NC}")
            self.results.append((name, False, error_msg))
    
    async def validate_imports(self):
        """Validate all critical imports."""
        print(f"\n{YELLOW}1. Validating Imports...{NC}")
        
        # Core imports
        try:
            from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest
            self.check("Master Program imports", True)
        except Exception as e:
            self.check("Master Program imports", False, str(e))
        
        # Agent imports
        try:
            from llamaagent.agents.base import AgentConfig, BaseAgent
            from llamaagent.agents.react import ReactAgent
            self.check("Agent system imports", True)
        except Exception as e:
            self.check("Agent system imports", False, str(e))
        
        # Planning imports
        try:
            from llamaagent.planning.task_planner import TaskPlanner, Task
            self.check("Task planning imports", True)
        except Exception as e:
            self.check("Task planning imports", False, str(e))
        
        # Spawning imports
        try:
            from llamaagent.spawning.agent_spawner import AgentSpawner, SpawnConfig
            self.check("Agent spawning imports", True)
        except Exception as e:
            self.check("Agent spawning imports", False, str(e))
        
        # Tools imports
        try:
            from llamaagent.tools import ToolRegistry, get_all_tools
            self.check("Tools system imports", True)
        except Exception as e:
            self.check("Tools system imports", False, str(e))
        
        # OpenAI integration
        try:
            from llamaagent.integration.openai_agents_complete import OpenAIAgentsManager
            self.check("OpenAI integration imports", True)
        except Exception as e:
            self.check("OpenAI integration imports", False, str(e))
    
    async def validate_configuration(self):
        """Validate configuration files."""
        print(f"\n{YELLOW}2. Validating Configuration...{NC}")
        
        # Check config directory
        config_dir = Path("config")
        self.check("Config directory exists", config_dir.exists())
        
        # Check master config
        master_config = config_dir / "master_config.yaml"
        self.check("Master config exists", master_config.exists())
        
        # Check environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        self.check(
            "OpenAI API key", 
            bool(openai_key), 
            "Not set - OpenAI features will be disabled",
            warning=True
        )
        
        # Check Python version
        import sys
        py_version = sys.version_info
        self.check(
            "Python version",
            py_version >= (3, 8),
            f"Python 3.8+ required, found {py_version.major}.{py_version.minor}"
        )
    
    async def validate_functionality(self):
        """Validate core functionality."""
        print(f"\n{YELLOW}3. Validating Core Functionality...{NC}")
        
        try:
            from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest
            
            # Create orchestrator
            orchestrator = MasterOrchestrator()
            self.check("Orchestrator creation", True)
            
            # Test task creation
            request = CreateMasterTaskRequest(
                task_description="Test task validation",
                auto_decompose=True,
                auto_spawn=False,
                max_agents=1,
                enable_openai=False
            )
            
            result = await orchestrator.create_master_task(request)
            self.check("Task creation", result.get("success", False), result.get("error", ""))
            
            # Test system status
            status = await orchestrator.get_system_status()
            self.check("System status", status is not None)
            
            # Test hierarchy
            hierarchy = orchestrator.get_hierarchy_visualization()
            self.check("Hierarchy visualization", True)
            
        except Exception as e:
            self.check("Core functionality", False, str(e))
    
    async def validate_agent_spawning(self):
        """Validate agent spawning system."""
        print(f"\n{YELLOW}4. Validating Agent Spawning...{NC}")
        
        try:
            from llamaagent.spawning.agent_spawner import AgentSpawner, SpawnConfig
            from llamaagent.agents.base import AgentConfig, AgentRole
            
            spawner = AgentSpawner()
            
            # Test spawning
            config = SpawnConfig(
                agent_config=AgentConfig(
                    name="ValidationAgent",
                    role=AgentRole.GENERALIST
                )
            )
            
            result = await spawner.spawn_agent(
                task="Validation test",
                config=config
            )
            
            self.check("Agent spawning", result.success, result.error or "")
            
            # Check hierarchy
            stats = spawner.hierarchy.get_hierarchy_stats()
            self.check("Hierarchy tracking", stats["total_agents"] > 0)
            
            # Cleanup
            if result.success:
                await spawner.terminate_agent(result.agent_id)
            
        except Exception as e:
            self.check("Agent spawning system", False, str(e))
    
    async def validate_task_planning(self):
        """Validate task planning system."""
        print(f"\n{YELLOW}5. Validating Task Planning...{NC}")
        
        try:
            from llamaagent.planning.task_planner import TaskPlanner, Task
            
            planner = TaskPlanner()
            
            # Create a test plan
            plan = planner.create_plan(
                goal="Validate task planning system",
                auto_decompose=True
            )
            
            self.check("Plan creation", plan is not None)
            self.check("Plan validation", plan.is_valid, str(plan.validation_errors))
            
            # Test execution order
            try:
                execution_order = planner.get_execution_order(plan)
                self.check("Execution order", len(execution_order) > 0)
            except Exception as e:
                self.check("Execution order", False, str(e))
            
        except Exception as e:
            self.check("Task planning system", False, str(e))
    
    async def validate_tools(self):
        """Validate tools system."""
        print(f"\n{YELLOW}6. Validating Tools System...{NC}")
        
        try:
            from llamaagent.tools import ToolRegistry, get_all_tools
            
            # Get all tools
            tools = get_all_tools()
            self.check("Tool loading", len(tools) > 0, f"Found {len(tools)} tools")
            
            # Test tool registry
            registry = ToolRegistry()
            for tool in tools:
                registry.register(tool)
            
            self.check("Tool registration", len(registry.list_tools()) > 0)
            
            # Test calculator tool
            calc_tool = registry.get("calculator")
            if calc_tool:
                try:
                    result = calc_tool.execute(expression="2 + 2")
                    self.check("Calculator tool", result == "4", f"Got {result}")
                except Exception as e:
                    self.check("Calculator tool", False, str(e))
            else:
                self.check("Calculator tool", False, "Not found", warning=True)
            
        except Exception as e:
            self.check("Tools system", False, str(e))
    
    async def validate_api_endpoints(self):
        """Validate API endpoints (if server is running)."""
        print(f"\n{YELLOW}7. Validating API Endpoints...{NC}")
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Check if server is running
                try:
                    async with session.get("http://localhost:8000/") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.check("API root endpoint", True)
                            
                            # Check other endpoints
                            endpoints = [
                                ("/api/v1/status", "Status endpoint"),
                                ("/api/v1/hierarchy", "Hierarchy endpoint"),
                                ("/docs", "API documentation")
                            ]
                            
                            for endpoint, name in endpoints:
                                try:
                                    async with session.get(f"http://localhost:8000{endpoint}") as resp:
                                        self.check(name, resp.status in [200, 307])
                                except:
                                    self.check(name, False, "Failed to connect")
                        else:
                            self.check("API server", False, "Server returned non-200 status", warning=True)
                except:
                    self.check("API server", False, "Not running (start with: python3 llamaagent_master_program.py server)", warning=True)
                    
        except ImportError:
            self.check("API validation", False, "aiohttp not installed", warning=True)
        except Exception as e:
            self.check("API endpoints", False, str(e), warning=True)
    
    async def validate_performance(self):
        """Validate performance metrics."""
        print(f"\n{YELLOW}8. Validating Performance...{NC}")
        
        try:
            from llamaagent_master_program import MasterOrchestrator, CreateMasterTaskRequest
            
            orchestrator = MasterOrchestrator()
            
            # Test task creation speed
            start = time.time()
            request = CreateMasterTaskRequest(
                task_description="Performance test",
                auto_decompose=True,
                auto_spawn=True,
                max_agents=5,
                enable_openai=False
            )
            
            result = await orchestrator.create_master_task(request)
            elapsed = time.time() - start
            
            self.check(
                "Task creation performance",
                elapsed < 5.0,
                f"Took {elapsed:.2f}s (should be < 5s)",
                warning=elapsed > 2.0
            )
            
            # Check resource usage
            status = await orchestrator.get_system_status()
            resource_usage = status.resource_usage
            
            memory_percent = resource_usage["memory"]["percentage"]
            self.check(
                "Memory usage",
                memory_percent < 80,
                f"{memory_percent:.1f}% used",
                warning=memory_percent > 50
            )
            
        except Exception as e:
            self.check("Performance validation", False, str(e))
    
    def generate_report(self):
        """Generate validation report."""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}VALIDATION REPORT{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        total_checks = len(self.results)
        passed_checks = sum(1 for _, success, _ in self.results if success)
        
        # Summary
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {GREEN}{passed_checks}{NC}")
        print(f"Failed: {RED}{len(self.errors)}{NC}")
        print(f"Warnings: {YELLOW}{len(self.warnings)}{NC}")
        
        # Success rate
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        color = GREEN if success_rate >= 90 else YELLOW if success_rate >= 70 else RED
        print(f"\nSuccess Rate: {color}{success_rate:.1f}%{NC}")
        
        # Errors
        if self.errors:
            print(f"\n{RED}Errors:{NC}")
            for name, error in self.errors:
                print(f"  • {name}: {error}")
        
        # Warnings
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{NC}")
            for name, warning in self.warnings:
                print(f"  • {name}: {warning}")
        
        # Overall status
        print(f"\n{BLUE}{'='*60}{NC}")
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print(f"{GREEN}PASS SYSTEM IS FULLY OPERATIONAL!{NC}")
                print(f"\nThe LlamaAgent Master Program is ready for use.")
            else:
                print(f"{GREEN}PASS SYSTEM IS OPERATIONAL{NC}")
                print(f"\nThe system is working but has some warnings to address.")
        else:
            print(f"{RED}FAIL SYSTEM HAS CRITICAL ISSUES{NC}")
            print(f"\nPlease fix the errors before running in production.")
        
        # Save report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed": len(self.errors),
            "warnings": len(self.warnings),
            "success_rate": success_rate,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.results
        }
        
        with open("validation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: validation_report.json")
    
    async def run_validation(self):
        """Run complete system validation."""
        self.print_banner()
        
        # Run all validations
        await self.validate_imports()
        await self.validate_configuration()
        await self.validate_functionality()
        await self.validate_agent_spawning()
        await self.validate_task_planning()
        await self.validate_tools()
        await self.validate_api_endpoints()
        await self.validate_performance()
        
        # Generate report
        self.generate_report()


async def main():
    """Run the system validator."""
    validator = SystemValidator()
    await validator.run_validation()


if __name__ == "__main__":
    asyncio.run(main())