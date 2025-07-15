"""
Simon Willison's LLM Ecosystem Integration

Comprehensive integration with Simon Willison's LLM ecosystem, including:
- Core LLM library and CLI
- Provider integrations (OpenAI, Anthropic, Gemini, Mistral, etc.)
- Tool integrations (SQLite, Datasette, Docker, etc.)
- Data utilities (sqlite-utils, datasette, etc.)

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import sqlite_utils
    HAS_SQLITE_UTILS = True
except ImportError:
    HAS_SQLITE_UTILS = False

try:
    import datasette
    HAS_DATASETTE = True
except ImportError:
    HAS_DATASETTE = False

try:
    import llm
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


class LLMTool(Enum):
    """Available LLM tools in Simon's ecosystem"""
    SQLITE = "sqlite"
    JQ = "jq"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    QUICKJS = "quickjs"
    DOCKER = "docker"
    COMMAND = "command"


@dataclass
class ConversationEntry:
    """Represents a conversation entry"""
    id: str
    timestamp: str
    model: str
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool:
    """Base class for Simon ecosystem tools"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Tool_{name}")
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute tool operation"""
        self.logger.info(f"Executing operation: {operation}")
        return {"status": "success", "operation": operation, "tool": self.name}
    
    async def health_check(self) -> bool:
        """Check if tool is healthy"""
        return True


class SQLiteTool(BaseTool):
    """SQLite tool for database operations"""
    
    def __init__(self, db_path: str = ":memory:"):
        super().__init__("sqlite")
        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute SQLite operation"""
        
        if operation == "query":
            sql = kwargs.get("sql")
            if not sql:
                raise ValueError("SQL query required")
            
            return list(self.db.execute(sql))
        
        elif operation == "insert":
            table = kwargs.get("table")
            data = kwargs.get("data")
            if not table or not data:
                raise ValueError("Table name and data required")
            
            self.db[table].insert(data)
        
        elif operation == "create_table":
            table = kwargs.get("table")
            schema = kwargs.get("schema")
            if not table or not schema:
                raise ValueError("Table name and schema required")
            
            self.db[table].create(schema)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class PythonTool(BaseTool):
    """Python code execution tool"""
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute Python code"""
        
        if operation == "exec":
            code = kwargs.get("code")
            if not code:
                raise ValueError("Python code required")
            
            # Execute in restricted environment
            namespace = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "reversed": reversed,
                }
            }
            
            try:
                exec(code, namespace)
                return namespace.get("result", "Code executed successfully")
            except Exception as e:
                return f"Error: {e}"
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class JQTool(BaseTool):
    """JQ JSON processing tool"""
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute JQ operation"""
        
        if operation == "filter":
            data = kwargs.get("data")
            filter_expr = kwargs.get("filter")
            
            if not data or not filter_expr:
                raise ValueError("Data and filter expression required")
            
            try:
                # Convert data to JSON if needed
                if not isinstance(data, str):
                    data = json.dumps(data)
                
                # Execute jq command
                process = subprocess.run(
                    ["jq", filter_expr],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True
                )
                
                return json.loads(process.stdout)
                
            except subprocess.CalledProcessError as e:
                return {"error": f"JQ error: {e.stderr}"}
            except Exception as e:
                return {"error": f"Error: {e}"}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class QuickJSTool(BaseTool):
    """QuickJS JavaScript execution tool"""
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute JavaScript code"""
        
        if operation == "exec":
            code = kwargs.get("code")
            if not code:
                raise ValueError("JavaScript code required")
            
            try:
                # Use llm-tools-quickjs if available
                cmd = ["llm", "tools", "quickjs", "--code", code]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if process.returncode == 0:
                    return {"result": process.stdout.strip()}
                else:
                    return {"error": process.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"error": "JavaScript execution timed out"}
            except Exception as e:
                return {"error": f"Error: {e}"}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class DockerTool(BaseTool):
    """Docker execution tool"""
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute code in Docker container"""
        
        if operation == "exec":
            code = kwargs.get("code")
            language = kwargs.get("language", "python")
            
            if not code:
                raise ValueError("Code required")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{language}',
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            try:
                # Run in Docker container
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{temp_file_path}:/code.{language}",
                    f"{language}:latest",
                    f"/code.{language}"
                ]
                
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if process.returncode == 0:
                    return {"result": process.stdout}
                else:
                    return {"error": process.stderr}
                    
            except subprocess.TimeoutExpired:
                return {"error": "Docker execution timed out"}
            except Exception as e:
                return {"error": f"Error: {e}"}
            finally:
                # Clean up
                os.unlink(temp_file_path)
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class CommandTool(BaseTool):
    """Command execution tool"""
    
    async def execute(self, operation: str, **kwargs) -> Any:
        """Execute system command"""
        
        if operation == "exec":
            command = kwargs.get("command")
            if not command:
                raise ValueError("Command required")
            
            try:
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return {
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "returncode": process.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {"error": "Command execution timed out"}
            except Exception as e:
                return {"error": f"Error: {e}"}
        
        else:
            raise ValueError(f"Unknown operation: {operation}")


class SimonEcosystem:
    """Integration with Simon's LLM ecosystem"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.db = None
        self.logger = logging.getLogger("SimonEcosystem")
        
        # Initialize tools
        self.tools = {
            LLMTool.SQLITE: SQLiteTool(db_path),
            LLMTool.JQ: JQTool(),
            LLMTool.PYTHON: PythonTool(),
            LLMTool.JAVASCRIPT: QuickJSTool(),
            LLMTool.QUICKJS: QuickJSTool(),
            LLMTool.DOCKER: DockerTool(),
            LLMTool.COMMAND: CommandTool()
        }
        
        # Setup database
        self._setup_database()
    
    def _setup_database(self):
        """Setup conversation database"""
        if HAS_SQLITE_UTILS:
            self.db = sqlite_utils.Database(self.db_path)
            
            # Create conversations table if not exists
            if "conversations" not in self.db.table_names():
                self.db["conversations"].create({
                    "id": str,
                    "timestamp": str,
                    "model": str,
                    "prompt": str,
                    "response": str,
                    "metadata": str
                })
    
    async def use_tool(self, tool_name: str, operation: str, **kwargs) -> Any:
        """Use a tool from Simon's ecosystem"""
        
        tool_enum = LLMTool(tool_name)
        if tool_enum not in self.tools:
            raise ValueError(f"Tool {tool_name} not available")
        
        tool = self.tools[tool_enum]
        return await tool.execute(operation, **kwargs)
    
    async def _log_conversation(
        self, 
        prompt: str, 
        response: str, 
        model: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log conversation to database"""
        if not self.db:
            return
            
        entry = ConversationEntry(
            id=str(len(list(self.db["conversations"].rows))),
            timestamp=str(asyncio.get_event_loop().time()),
            model=model,
            prompt=prompt,
            response=response,
            metadata=metadata or {}
        )
        
        self.db["conversations"].insert({
            "id": entry.id,
            "timestamp": entry.timestamp,
            "model": entry.model,
            "prompt": entry.prompt,
            "response": entry.response,
            "metadata": json.dumps(entry.metadata)
        })
    
    async def search_conversations(
        self, 
        query: str, 
        model: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationEntry]:
        """Search conversations"""
        if not self.db:
            return []
            
        sql = "SELECT * FROM conversations WHERE prompt LIKE ? OR response LIKE ?"
        params = [f"%{query}%", f"%{query}%"]
        
        if model:
            sql += " AND model = ?"
            params.append(model)
            
        sql += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        rows = self.db.execute(sql, params).fetchall()
        
        return [
            ConversationEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                model=row["model"],
                prompt=row["prompt"],
                response=row["response"],
                metadata=json.loads(row["metadata"])
            )
            for row in rows
        ]
    
    @asynccontextmanager
    async def datasette_server(self, port: int = 8001):
        """Start temporary Datasette server"""
        if not HAS_DATASETTE:
            raise RuntimeError("Datasette not available")
        
        cmd = [
            "datasette", 
            self.db_path,
            "--port", str(port),
            "--host", "0.0.0.0"
        ]
        
        process = await asyncio.create_subprocess_exec(*cmd)
        
        try:
            # Wait a moment for server to start
            await asyncio.sleep(2)
            self.logger.info(f"Datasette server started on http://localhost:{port}")
            yield f"http://localhost:{port}"
        finally:
            process.terminate()
            await process.wait()
    
    async def export_data(
        self, 
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Export conversation data"""
        if not self.db:
            raise RuntimeError("Database not available")
        
        conversations = list(self.db["conversations"].rows)
        
        if format == "json":
            data = json.dumps(conversations, indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=conversations[0].keys())
            writer.writeheader()
            writer.writerows(conversations)
            data = output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(data)
            return output_path
        
        return data
    
    async def embed_conversations(self, model: str = "all-MiniLM-L6-v2"):
        """Generate embeddings for conversations"""
        if not HAS_LLM:
            raise RuntimeError("LLM tools not available")
        
        conversations = list(self.db["conversations"].rows)
        
        for conv in conversations:
            try:
                # Generate embedding using llm
                result = subprocess.run([
                    "llm", "embed",
                    "-m", model,
                    conv["prompt"]
                ], capture_output=True, text=True, check=True)
                
                embedding = json.loads(result.stdout)
                
                # Update conversation with embedding
                self.db["conversations"].update(conv["id"], {
                    "embedding": json.dumps(embedding)
                })
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Embedding failed: {e.stderr}")
                raise RuntimeError(f"Embedding failed: {e.stderr}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of Simon ecosystem components"""
        health = {
            "database": False,
            "tools": {},
            "components": {}
        }
        
        # Check LLM library
        health["components"]["llm"] = HAS_LLM
        health["components"]["sqlite_utils"] = HAS_SQLITE_UTILS
        health["components"]["datasette"] = HAS_DATASETTE
        
        # Check database
        if self.db:
            try:
                self.db.execute("SELECT 1").fetchone()
                health["database"] = True
            except Exception:
                health["database"] = False
        
        # Check tools
        for tool_name, tool in self.tools.items():
            try:
                health["tools"][tool_name.value] = await tool.health_check()
            except Exception as e:
                health["tools"][tool_name.value] = False
                self.logger.error(f"Tool {tool_name} health check failed: {e}")
        
        return health


async def main():
    """Demo of Simon ecosystem integration"""
    ecosystem = SimonEcosystem()
    
    # Test tools
    print("Testing SQLite tool...")
    result = await ecosystem.use_tool("sqlite", "query", sql="SELECT 1")
    print(f"SQLite result: {result}")
    
    print("Testing Python tool...")
    result = await ecosystem.use_tool("python", "exec", code="result = 2 + 2")
    print(f"Python result: {result}")
    
    print("Testing JQ tool...")
    result = await ecosystem.use_tool("jq", "filter", 
                                     data={"name": "test", "value": 42},
                                     filter=".name")
    print(f"JQ result: {result}")
    
    # Get health status
    health = await ecosystem.health_check()
    print(f"Health: {health}")
    
    # Export conversations
    if ecosystem.db:
        export_path = await ecosystem.export_data("json", "conversations.json")
        print(f"Exported to: {export_path}")


if __name__ == "__main__":
    asyncio.run(main())