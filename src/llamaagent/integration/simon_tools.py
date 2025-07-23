"""
Simon Tools Integration Module

Provides integration with Simon's ecosystem tools:
- LLM chat and embedding tools
- Data exploration with Datasette
- SQLite query execution
- Docker code execution
- Command line tools
- JSON processing with jq

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tools.base import Tool

logger = logging.getLogger(__name__)


class SimonEcosystemConfig:
    """Configuration for Simon's ecosystem integration"""

    def __init__(
        self,
        llm_model: str = "gpt-4",
        datasette_port: int = 8001,
        docker_timeout: int = 30,
        command_timeout: int = 10,
        enable_embedding: bool = True,
        enable_docker: bool = True,
    ):
        self.llm_model = llm_model
        self.datasette_port = datasette_port
        self.docker_timeout = docker_timeout
        self.command_timeout = command_timeout
        self.enable_embedding = enable_embedding
        self.enable_docker = enable_docker


class LLMChatTool(Tool):
    """Tool for LLM chat using Simon's ecosystem"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="llm_chat", description="Chat with LLM using Simon's ecosystem"
        )
        self.config = config

    async def execute(
        self, prompt: str, model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute LLM chat"""
        try:
            # Use configured model or override
            model_name = model or self.config.llm_model

            # Mock LLM response - replace with actual Simon ecosystem integration
            response = f"Mock response to: {prompt} (using {model_name})"

            return {
                "success": True,
                "response": response,
                "model": model_name,
                "tokens_used": len(prompt.split()) * 2,
            }

        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return {"success": False, "error": str(e)}


class LLMEmbeddingTool(Tool):
    """Tool for creating embeddings using Simon's ecosystem"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="llm_embedding",
            description="Create embeddings using Simon's LLM tools",
        )
        self.config = config

    async def execute(
        self, text: str, model: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute embedding creation"""
        try:
            if not self.config.enable_embedding:
                return {
                    "success": False,
                    "error": "Embedding disabled in configuration",
                }

            # Mock embedding - replace with actual Simon ecosystem integration
            embedding = [0.1] * 768  # Mock 768-dimensional embedding

            return {
                "success": True,
                "embedding": embedding,
                "dimensions": len(embedding),
                "text_length": len(text),
            }

        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return {"success": False, "error": str(e)}


class DatasetteExplorer(Tool):
    """Tool for exploring data using Datasette"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="datasette_explore", description="Explore data using Datasette"
        )
        self.config = config

    async def execute(
        self,
        database: str,
        table: Optional[str] = None,
        sql: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute Datasette operation"""
        try:
            port = self.config.datasette_port

            # Mock Datasette response - replace with actual integration
            if sql:
                result = {"sql": sql, "rows": [{"mock": "data"}], "truncated": False}
            elif table:
                result = {
                    "table": table,
                    "rows": [{"id": 1, "name": "Mock data"}],
                    "count": 1,
                }
            else:
                result = {
                    "database": database,
                    "tables": ["mock_table1", "mock_table2"],
                }

            return {
                "success": True,
                "result": result,
                "server_url": f"http://localhost:{port}",
            }

        except Exception as e:
            logger.error(f"Datasette operation failed: {e}")
            return {"success": False, "error": str(e)}


class SQLiteQueryTool(Tool):
    """Tool for querying SQLite databases using Simon's sqlite-utils"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="sqlite_query", description="Query SQLite databases using sqlite-utils"
        )
        self.config = config

    async def execute(
        self, query: str, database_path: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute SQLite query"""
        try:
            if database_path:
                # Use external database
                cmd = ["sqlite-utils", "query", database_path, query, "--json"]
            else:
                # Use in-memory database
                cmd = ["sqlite-utils", "memory", query, "--json"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.command_timeout
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    return {
                        "success": True,
                        "data": data,
                        "row_count": len(data) if isinstance(data, list) else 1,
                    }
                except json.JSONDecodeError:
                    return {"success": True, "data": result.stdout, "raw_output": True}
            else:
                return {
                    "success": False,
                    "error": result.stderr or "Query execution failed",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Query execution timed out"}
        except Exception as e:
            logger.error(f"SQLite query failed: {e}")
            return {"success": False, "error": str(e)}


class DockerExecutionTool(Tool):
    """Tool for executing code in Docker containers"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="docker_execute", description="Execute code in Docker containers"
        )
        self.config = config

    async def execute(
        self, code: str, language: str = "python", image: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute code in Docker"""
        try:
            if not self.config.enable_docker:
                return {
                    "success": False,
                    "error": "Docker execution disabled in configuration",
                }

            # Determine Docker image
            if not image:
                image_map = {
                    "python": "python:3.11-slim",
                    "javascript": "node:18-slim",
                    "bash": "ubuntu:22.04",
                }
                image = image_map.get(language, "python:3.11-slim")

            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                mode='w', suffix=f'.{language}', delete=False
            ) as f:
                f.write(code)
                code_file = f.name

            try:
                # Run code in Docker
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{code_file}:/code.{language}",
                    "--timeout",
                    str(self.config.docker_timeout),
                    image,
                    "sh",
                    "-c",
                    (
                        f"cd / && python /code.{language}"
                        if language == "python"
                        else f"cd / && cat /code.{language}"
                    ),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.docker_timeout,
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None,
                    "exit_code": result.returncode,
                    "language": language,
                    "image": image,
                }

            finally:
                # Clean up temporary file
                Path(code_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Code execution timed out"}
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return {"success": False, "error": str(e)}


class CommandExecutionTool(Tool):
    """Tool for executing system commands"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(name="command_execute", description="Execute system commands")
        self.config = config

    async def execute(
        self, command: str, shell: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """Execute system command"""
        try:
            # Execute command with timeout
            if shell:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                cmd_parts = command.split()
                process = await asyncio.create_subprocess_exec(
                    *cmd_parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.command_timeout
                )

                return {
                    "success": process.returncode == 0,
                    "output": stdout.decode() if stdout else "",
                    "error": stderr.decode() if stderr else None,
                    "exit_code": process.returncode,
                    "command": command,
                }

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {"success": False, "error": "Command execution timed out"}

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}


class JQProcessorTool(Tool):
    """Tool for processing JSON with jq"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(name="jq_process", description="Process JSON data with jq")
        self.config = config

    async def execute(self, data: Any, filter_expr: str, **kwargs) -> Dict[str, Any]:
        """Process JSON with jq"""
        try:
            # Convert data to JSON string if needed
            if isinstance(data, str):
                json_data = data
            else:
                json_data = json.dumps(data)

            # Execute jq command
            cmd = ["jq", filter_expr]

            process = subprocess.run(
                cmd,
                input=json_data,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
            )

            if process.returncode == 0:
                try:
                    result = json.loads(process.stdout)
                    return {"success": True, "result": result, "filter": filter_expr}
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "result": process.stdout.strip(),
                        "raw_output": True,
                    }
            else:
                return {
                    "success": False,
                    "error": process.stderr or "jq processing failed",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "jq processing timed out"}
        except Exception as e:
            logger.error(f"jq processing failed: {e}")
            return {"success": False, "error": str(e)}


class JavaScriptTool(Tool):
    """Tool for executing JavaScript code"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="javascript_execute",
            description="Execute JavaScript code using Node.js",
        )
        self.config = config

    async def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        """Execute JavaScript code"""
        try:
            # Create temporary file for JavaScript
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                js_file = f.name

            try:
                # Execute with Node.js
                cmd = ["node", js_file]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.command_timeout,
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else None,
                    "exit_code": result.returncode,
                }

            finally:
                # Clean up temporary file
                Path(js_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "JavaScript execution timed out"}
        except Exception as e:
            logger.error(f"JavaScript execution failed: {e}")
            return {"success": False, "error": str(e)}


class ConversationSearchTool(Tool):
    """Tool for searching conversation history"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="conversation_search", description="Search conversation history"
        )
        self.config = config

    async def execute(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Search conversations"""
        try:
            # Mock conversation search - replace with actual implementation
            results = [
                {
                    "id": "conv_1",
                    "content": f"Mock conversation containing '{query}'",
                    "timestamp": "2024-01-01T10:00:00Z",
                    "score": 0.95,
                },
                {
                    "id": "conv_2",
                    "content": f"Another conversation about {query}",
                    "timestamp": "2024-01-01T11:00:00Z",
                    "score": 0.87,
                },
            ][:limit]

            return {
                "success": True,
                "results": results,
                "total_found": len(results),
                "query": query,
            }

        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return {"success": False, "error": str(e)}


class EcosystemStatsTool(Tool):
    """Tool for getting ecosystem statistics"""

    def __init__(self, config: SimonEcosystemConfig):
        super().__init__(
            name="ecosystem_stats", description="Get Simon ecosystem statistics"
        )
        self.config = config

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Get ecosystem statistics"""
        try:
            # Mock stats - replace with actual ecosystem integration
            stats = {
                "llm_model": self.config.llm_model,
                "datasette_port": self.config.datasette_port,
                "docker_enabled": self.config.enable_docker,
                "embedding_enabled": self.config.enable_embedding,
                "uptime": "2h 30m",
                "requests_today": 145,
                "tools_available": 9,
            }

            return {"success": True, "stats": stats}

        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
            return {"success": False, "error": str(e)}


class SimonToolRegistry:
    """Registry for Simon's ecosystem tools"""

    def __init__(self, config: Optional[SimonEcosystemConfig] = None):
        self.config = config or SimonEcosystemConfig()
        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all Simon ecosystem tools"""
        tools = [
            LLMChatTool(self.config),
            LLMEmbeddingTool(self.config),
            DatasetteExplorer(self.config),
            SQLiteQueryTool(self.config),
            DockerExecutionTool(self.config),
            CommandExecutionTool(self.config),
            JQProcessorTool(self.config),
            JavaScriptTool(self.config),
            ConversationSearchTool(self.config),
            EcosystemStatsTool(self.config),
        ]

        for tool in tools:
            self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())

    def get_registry(self) -> Dict[str, Tool]:
        """Get the complete tool registry"""
        return self.tools.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all tools"""
        results = {}

        for name, tool in self.tools.items():
            try:
                # Simple health check - try to access the tool
                results[name] = {
                    "status": "healthy",
                    "name": tool.name,
                    "description": tool.description,
                }
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}

        return {
            "total_tools": len(self.tools),
            "healthy_tools": sum(
                1 for r in results.values() if r["status"] == "healthy"
            ),
            "results": results,
        }


def create_simon_tools(
    config: Optional[SimonEcosystemConfig] = None,
) -> SimonToolRegistry:
    """Create Simon tools registry"""
    return SimonToolRegistry(config)


async def example_usage() -> None:
    """Example usage of Simon tools"""
    # Create tools
    simon_tools = create_simon_tools()
    registry = simon_tools.get_registry()

    # Get LLM chat tool
    chat_tool = registry.get("llm_chat")
    if chat_tool:
        result = await chat_tool.execute(
            prompt="What is the capital of France?", model="gpt-4"
        )
        print(f"Chat result: {result}")

    # Get SQLite query tool
    sql_tool = registry.get("sqlite_query")
    if sql_tool:
        result = await sql_tool.execute(query="SELECT 'Hello, World!' as message")
        print(f"SQL result: {result}")

    # Get ecosystem stats
    stats_tool = registry.get("ecosystem_stats")
    if stats_tool:
        result = await stats_tool.execute()
        print(f"Stats: {result}")

    # Health check
    health = await simon_tools.health_check()
    print(f"Health check: {health}")


if __name__ == "__main__":
    asyncio.run(example_usage())
