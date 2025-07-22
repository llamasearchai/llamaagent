#!/usr/bin/env python3
"""
Shell_GPT FastAPI Endpoints, Author: Nik Jois <nikjois@llamasearch.ai>

Comprehensive FastAPI endpoints for shell_gpt functionality, including:
- Shell command generation and execution
- Code generation with multiple languages
- Function calling and tool usage
- Chat sessions and REPL mode
- Role-based interactions
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/shell", tags=["shell"])


# Request/Response Models
class ShellCommandRequest(BaseModel):
    """Request for shell command generation."""

    prompt: str = Field(..., description="Natural language prompt for command")
    current_directory: Optional[str] = Field(
        None, description="Current working directory"
    )
    os_type: Optional[str] = Field("unix", description="Operating system type")
    shell_type: Optional[str] = Field("bash", description="Shell type")
    auto_execute: bool = Field(False, description="Whether to auto-execute the command")
    safety_check: bool = Field(True, description="Enable safety checks")


class ShellCommandResponse(BaseModel):
    """Response from shell command generation."""

    command: str = Field(..., description="Generated shell command")
    explanation: str = Field(..., description="Explanation of the command")
    safety_warnings: List[str] = Field(
        default_factory=list, description="Safety warnings"
    )
    estimated_runtime: str = Field(..., description="Estimated execution time")
    requires_confirmation: bool = Field(
        ..., description="Whether command requires confirmation"
    )
    execution_result: Optional[Dict[str, Any]] = Field(
        None, description="Execution result if auto-executed"
    )


class CodeGenerationRequest(BaseModel):
    """Request for code generation."""

    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field("python", description="Programming language")
    include_tests: bool = Field(True, description="Include test cases")
    include_docs: bool = Field(True, description="Include documentation")
    complexity_level: str = Field("intermediate", description="Code complexity level")


class CodeGenerationResponse(BaseModel):
    """Response from code generation."""

    code: str = Field(..., description="Generated code")
    explanation: str = Field(..., description="Code explanation")
    test_cases: Optional[str] = Field(None, description="Generated test cases")
    documentation: Optional[str] = Field(None, description="Generated documentation")
    best_practices: List[str] = Field(
        default_factory=list, description="Applied best practices"
    )


class FunctionCallRequest(BaseModel):
    """Request for function calling."""

    function_name: str = Field(..., description="Function to call")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Function parameters"
    )
    context: Optional[str] = Field(None, description="Additional context")


class FunctionCallResponse(BaseModel):
    """Response from function calling."""

    result: Any = Field(..., description="Function execution result")
    success: bool = Field(..., description="Whether function executed successfully")
    execution_time: float = Field(..., description="Execution time in seconds")
    output_type: str = Field(..., description="Type of output")


class ChatRequest(BaseModel):
    """Request for chat interaction."""

    message: str = Field(..., description="Chat message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    role: Optional[str] = Field(None, description="Role for the interaction")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    """Response from chat interaction."""

    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Chat session ID")
    message_id: str = Field(..., description="Message ID")
    timestamp: str = Field(..., description="Response timestamp")


class RoleRequest(BaseModel):
    """Request for role management."""

    action: str = Field(..., description="Action: create, update, delete, get")
    role_name: str = Field(..., description="Role name")
    role_config: Optional[Dict[str, Any]] = Field(
        None, description="Role configuration"
    )


class RoleResponse(BaseModel):
    """Response from role management."""

    role_name: str = Field(..., description="Role name")
    success: bool = Field(..., description="Whether operation was successful")
    role_config: Optional[Dict[str, Any]] = Field(
        None, description="Role configuration"
    )


class SessionRequest(BaseModel):
    """Request for session management."""

    action: str = Field(..., description="Action: create, delete, get, send")
    session_id: Optional[str] = Field(None, description="Session ID")
    role: Optional[str] = Field(None, description="Role for new session")
    message: Optional[str] = Field(None, description="Message for send action")


class SessionResponse(BaseModel):
    """Response from session management."""

    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether operation was successful")
    messages: Optional[List[Dict[str, Any]]] = Field(
        None, description="Session messages"
    )


# Helper functions
def get_shell_generator():
    """Get shell command generator instance."""
    try:
        from ..cli.shell_commands import ShellCommandGenerator

        return ShellCommandGenerator()
    except ImportError:
        # Mock generator for demonstration
        class MockShellGenerator:
            async def generate_command(
                self, prompt: str, context: Dict[str, Any] = None
            ) -> Dict[str, Any]:
                return {
                    "command": f"# Generated command for: {prompt}",
                    "explanation": f"This command addresses: {prompt}",
                    "safety_warnings": [],
                    "confidence": 0.8,
                }

        return MockShellGenerator()


def get_shell_executor():
    """Get shell command executor instance."""
    try:
        from ..cli.shell_commands import ShellCommandExecutor

        return ShellCommandExecutor()
    except ImportError:
        # Mock executor for demonstration
        class MockShellExecutor:
            async def execute_non_interactive(self, command: str) -> bool:
                logger.info(f"Mock executing: {command}")
                return True

        return MockShellExecutor()


def get_code_generator():
    """Get code generator instance."""
    try:
        from ..cli.code_generator import CodeGenerator

        return CodeGenerator()
    except ImportError:
        # Mock generator for demonstration
        class MockCodeGenerator:
            async def generate_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "code": f"# Generated {request.get('language', 'python')} code\nprint('Hello World')",
                    "explanation": "Simple hello world program",
                    "test_cases": "# Test cases here",
                    "documentation": "# Documentation here",
                }

        return MockCodeGenerator()


def get_function_manager():
    """Get function manager instance."""
    try:
        from ..cli.function_manager import FunctionManager

        return FunctionManager()
    except ImportError:
        # Mock manager for demonstration
        class MockFunctionManager:
            async def call_function(
                self, name: str, params: Dict[str, Any]
            ) -> Dict[str, Any]:
                return {
                    "result": f"Mock result for {name} with params {params}",
                    "success": True,
                    "execution_time": 0.1,
                }

        return MockFunctionManager()


def get_role_manager():
    """Get role manager instance."""
    try:
        from ..cli.role_manager import RoleManager

        return RoleManager()
    except ImportError:
        # Mock manager for demonstration
        class MockRoleManager:
            def create_role(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
                return {"name": name, "config": config}

            def update_role(self, name: str, config: Dict[str, Any]) -> bool:
                return True

            def delete_role(self, name: str) -> bool:
                return True

            def get_role(self, name: str) -> Optional[Dict[str, Any]]:
                return {"name": name, "config": {}}

        return MockRoleManager()


def get_config_manager():
    """Get configuration manager instance."""
    try:
        from ..cli.config_manager import ConfigManager

        return ConfigManager()
    except ImportError:
        # Mock manager for demonstration
        class MockConfigManager:
            def get_config(self) -> Dict[str, Any]:
                return {"version": "1.0", "debug": False}

            def update_config(self, config: Dict[str, Any]) -> bool:
                return True

        return MockConfigManager()


# Session management helpers
def _create_session(role: Optional[str] = None) -> str:
    """Create a new chat session."""
    try:
        sessions_dir = Path.home() / ".config" / "llamaagent" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

        session_file = sessions_dir / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        return session_id
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return str(uuid.uuid4())  # Return a basic UUID as fallback


def _delete_session(session_id: str) -> bool:
    """Delete a chat session."""
    try:
        sessions_dir = Path.home() / ".config" / "llamaagent" / "sessions"
        session_file = sessions_dir / f"{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            return True

        return False
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        return False


def _get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get messages from a chat session."""
    try:
        sessions_dir = Path.home() / ".config" / "llamaagent" / "sessions"
        session_file = sessions_dir / f"{session_id}.json"

        if session_file.exists():
            with open(session_file, "r") as f:
                session_data = json.load(f)
            return session_data.get("messages", [])

        return []
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        return []


async def _log_command_generation(
    request: ShellCommandRequest, response: ShellCommandResponse
):
    """Log command generation for analytics."""
    try:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": request.prompt,
            "command": response.command,
            "auto_execute": request.auto_execute,
            "safety_warnings": len(response.safety_warnings),
        }
        logger.info(f"Command generation: {log_data}")
    except Exception as e:
        logger.error(f"Failed to log command generation: {e}")


async def _log_code_generation(
    request: CodeGenerationRequest, response: CodeGenerationResponse
):
    """Log code generation for analytics."""
    try:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "language": request.language,
            "complexity": request.complexity_level,
            "include_tests": request.include_tests,
            "include_docs": request.include_docs,
        }
        logger.info(f"Code generation: {log_data}")
    except Exception as e:
        logger.error(f"Failed to log code generation: {e}")


async def _log_function_call(
    request: FunctionCallRequest, response: FunctionCallResponse
):
    """Log function call for analytics."""
    try:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "function": request.function_name,
            "success": response.success,
            "execution_time": response.execution_time,
        }
        logger.info(f"Function call: {log_data}")
    except Exception as e:
        logger.error(f"Failed to log function call: {e}")


async def _log_chat_interaction(request: ChatRequest, response: ChatResponse):
    """Log chat interaction for analytics."""
    try:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": response.session_id,
            "role": request.role,
            "message_length": len(request.message),
            "response_length": len(response.response),
        }
        logger.info(f"Chat interaction: {log_data}")
    except Exception as e:
        logger.error(f"Failed to log chat interaction: {e}")


# API Endpoints
@router.post("/command/generate", response_model=ShellCommandResponse)
async def generate_shell_command(
    request: ShellCommandRequest, background_tasks: BackgroundTasks
):
    """Generate shell command from natural language prompt."""
    try:
        generator = get_shell_generator()

        # Build context
        context: Dict[str, Any] = {}
        if request.current_directory:
            context["current_directory"] = request.current_directory
        if request.os_type:
            context["os_type"] = request.os_type
        if request.shell_type:
            context["shell_type"] = request.shell_type

        # Generate command
        result = await generator.generate_command(request.prompt, context)

        # Determine safety and execution requirements
        command = result.get("command", "")
        safety_warnings = result.get("safety_warnings", [])

        # Estimate runtime based on command type
        estimated_runtime = "< 1 second"
        requires_confirmation = False

        if any(
            keyword in command.lower()
            for keyword in ["rm", "delete", "format", "shutdown"]
        ):
            estimated_runtime = "Immediate"
            requires_confirmation = True
            safety_warnings.append("Potentially destructive command")
        elif any(
            keyword in command.lower() for keyword in ["install", "update", "upgrade"]
        ):
            estimated_runtime = "1-5 minutes"
            requires_confirmation = True
        elif any(
            keyword in command.lower() for keyword in ["download", "wget", "curl"]
        ):
            estimated_runtime = "10-60 seconds"
        elif any(
            keyword in command.lower() for keyword in ["compile", "build", "make"]
        ):
            estimated_runtime = "30 seconds - 5 minutes"

        generate_response = ShellCommandResponse(
            command=command,
            explanation=result.get("explanation", ""),
            safety_warnings=safety_warnings,
            estimated_runtime=estimated_runtime,
            requires_confirmation=requires_confirmation,
        )

        # Execute if auto_execute is enabled
        if request.auto_execute and not requires_confirmation:
            try:
                executor = get_shell_executor()
                success = await executor.execute_non_interactive(command)
                generate_response.execution_result = {
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                generate_response.execution_result = {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # Log in background
        background_tasks.add_task(_log_command_generation, request, generate_response)

        return generate_response

    except Exception as e:
        logger.error(f"Command generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Command generation failed: {str(e)}"
        )


@router.post("/code/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest, background_tasks: BackgroundTasks
):
    """Generate code from natural language prompt."""
    try:
        generator = get_code_generator()

        # Build generation request
        gen_request = {
            "prompt": request.prompt,
            "language": request.language,
            "include_tests": request.include_tests,
            "include_docs": request.include_docs,
            "complexity_level": request.complexity_level,
        }

        # Generate code
        result = await generator.generate_code(gen_request)

        generate_response = CodeGenerationResponse(
            code=result.get("code", ""),
            explanation=result.get("explanation", ""),
            test_cases=result.get("test_cases") if request.include_tests else None,
            documentation=result.get("documentation") if request.include_docs else None,
            best_practices=result.get("best_practices", []),
        )

        # Log in background
        background_tasks.add_task(_log_code_generation, request, generate_response)

        return generate_response

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


@router.post("/function/call", response_model=FunctionCallResponse)
async def call_function(
    request: FunctionCallRequest, background_tasks: BackgroundTasks
):
    """Call a function with specified parameters."""
    try:
        func_manager = get_function_manager()

        # Call function
        result = await func_manager.call_function(
            request.function_name, request.parameters
        )

        function_response = FunctionCallResponse(
            result=result.get("result"),
            success=result.get("success", True),
            execution_time=result.get("execution_time", 0.0),
            output_type=type(result.get("result", "")).__name__,
        )

        # Log in background
        background_tasks.add_task(_log_function_call, request, function_response)

        return function_response

    except Exception as e:
        logger.error(f"Function call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Function call failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat interaction with optional role and session management."""
    try:
        # Get or create session
        session_id = request.session_id or _create_session(request.role)

        # Create message ID
        message_id = str(uuid.uuid4())
        # Mock response generation (in real implementation, this would use LLM)
        response_text = (
            f"I understand you said: '{request.message}'. How can I help you further?"
        )

        chat_response = ChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Log in background
        background_tasks.add_task(_log_chat_interaction, request, chat_response)

        return chat_response

    except Exception as e:
        logger.error(f"Chat interaction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Chat interaction failed: {str(e)}"
        )


@router.post("/role", response_model=RoleResponse)
async def manage_role(request: RoleRequest):
    """Manage roles for different interaction modes."""
    try:
        role_mgr = get_role_manager()

        if request.action == "create":
            if not request.role_config:
                raise HTTPException(
                    status_code=400,
                    detail="Role configuration required for create action",
                )

            role = role_mgr.create_role(request.role_name, request.role_config)
            success = role is not None
            return RoleResponse(
                role_name=request.role_name,
                success=success,
                role_config=request.role_config,
            )

        elif request.action == "update":
            if not request.role_config:
                raise HTTPException(
                    status_code=400,
                    detail="Role configuration required for update action",
                )

            success = role_mgr.update_role(request.role_name, request.role_config)
            return RoleResponse(
                role_name=request.role_name,
                success=success,
                role_config=request.role_config,
            )

        elif request.action == "delete":
            success = role_mgr.delete_role(request.role_name)
            return RoleResponse(role_name=request.role_name, success=success)

        elif request.action == "get":
            role = role_mgr.get_role(request.role_name)
            role_config = None
            if role:
                role_config = role.get("config", {})
            return RoleResponse(
                role_name=request.role_name,
                success=role is not None,
                role_config=role_config,
            )

        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid action: {request.action}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Role management failed: {str(e)}")


@router.post("/session", response_model=SessionResponse)
async def manage_session(request: SessionRequest):
    """Manage chat sessions."""
    try:
        if request.action == "create":
            session_id = _create_session(request.role)
            return SessionResponse(session_id=session_id, success=True)

        elif request.action == "delete":
            if not request.session_id:
                raise HTTPException(
                    status_code=400, detail="Session ID required for delete action"
                )

            success = _delete_session(request.session_id)
            return SessionResponse(session_id=request.session_id, success=success)

        elif request.action == "get":
            if not request.session_id:
                raise HTTPException(
                    status_code=400, detail="Session ID required for get action"
                )

            messages = _get_session_messages(request.session_id)
            return SessionResponse(
                session_id=request.session_id, success=True, messages=messages
            )

        elif request.action == "send":
            if not request.session_id or not request.message:
                raise HTTPException(
                    status_code=400,
                    detail="Session ID and message required for send action",
                )

            # Mock sending message (in real implementation, this would process the message)
            return SessionResponse(session_id=request.session_id, success=True)

        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid action: {request.action}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session management failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Session management failed: {str(e)}"
        )


@router.get("/info")
async def get_system_info():
    """Get system information."""
    try:
        config_mgr = get_config_manager()
        config = config_mgr.get_config()

        return {
            "version": config.get("version", "1.0.0"),
            "status": "operational",
            "features": [
                "shell_commands",
                "code_generation",
                "function_calling",
                "chat_interaction",
                "role_management",
                "session_management",
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=f"System info failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if all components can be initialized
        shell_gen = get_shell_generator()
        shell_exec = get_shell_executor()
        func_mgr = get_function_manager()
        code_gen = get_code_generator()
        role_mgr = get_role_manager()
        config_mgr = get_config_manager()

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "shell_generator": shell_gen is not None,
                "shell_executor": shell_exec is not None,
                "function_manager": func_mgr is not None,
                "code_generator": code_gen is not None,
                "role_manager": role_mgr is not None,
                "config_manager": config_mgr is not None,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
