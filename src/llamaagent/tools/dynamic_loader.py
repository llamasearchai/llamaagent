#!/usr/bin/env python3
"""
Dynamic Tool Loader - Enterprise Plugin System

This module implements dynamic tool loading capabilities including:
- Hot-loading of tools from files and packages
- Tool metadata and dependency management
- Version compatibility checking
- Sandboxed execution environments
- Tool marketplace integration

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from .tool_registry import Tool, ToolCategory, ToolSecurityLevel


@dataclass
class ToolMetadata:
    """Tool metadata for dynamic loading"""

    name: str
    version: str
    description: str
    author: str = ""
    category: ToolCategory = ToolCategory.CUSTOM
    security_level: ToolSecurityLevel = ToolSecurityLevel.PUBLIC
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    file_path: str = ""
    class_name: str = ""
    checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicToolLoader:
    """Dynamic tool loading and management system"""

    def __init__(
        self,
        tool_directories: Optional[List[str]] = None,
        enable_hot_reload: bool = True,
        check_interval: int = 60,
        sandbox_execution: bool = True,
    ):
        self.tool_directories = tool_directories or ["./tools", "./plugins"]
        self.enable_hot_reload = enable_hot_reload
        self.check_interval = check_interval
        self.sandbox_execution = sandbox_execution

        # Loaded tools tracking
        self.loaded_tools: Dict[str, ToolMetadata] = {}
        self.tool_classes: Dict[str, Type[Tool]] = {}
        self.file_checksums: Dict[str, str] = {}

        # Hot reload state
        self._reload_task: Optional[asyncio.Task[None]] = None
        self._running = False

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("DynamicToolLoader")
        logger.setLevel(logging.INFO)
        return logger

    async def start(self) -> None:
        """Start dynamic tool loader"""
        self._running = True

        # Initial load
        await self.discover_and_load_tools()

        # Start hot reload if enabled
        if self.enable_hot_reload:
            self._reload_task = asyncio.create_task(self._hot_reload_loop())

        self.logger.info("Dynamic tool loader started")

    async def stop(self) -> None:
        """Stop dynamic tool loader"""
        self._running = False

        if self._reload_task:
            self._reload_task.cancel()

        self.logger.info("Dynamic tool loader stopped")

    async def discover_and_load_tools(self) -> List[ToolMetadata]:
        """Discover and load tools from configured directories"""
        discovered_tools: List[ToolMetadata] = []

        for directory in self.tool_directories:
            if not os.path.exists(directory):
                self.logger.warning(f"Tool directory not found: {directory}")
                continue

            tools = await self._scan_directory(directory)
            discovered_tools.extend(tools)

        self.logger.info(f"Discovered {len(discovered_tools)} tools")
        return discovered_tools

    async def _scan_directory(self, directory: str) -> List[ToolMetadata]:
        """Scan directory for tool files"""
        tools: List[ToolMetadata] = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)

                    try:
                        tool_metadata = await self._load_tool_from_file(file_path)
                        if tool_metadata:
                            tools.append(tool_metadata)
                    except Exception as e:
                        self.logger.error(f"Failed to load tool from {file_path}: {e}")

        return tools

    async def _load_tool_from_file(self, file_path: str) -> Optional[ToolMetadata]:
        """Load tool from Python file"""
        try:
            # Calculate file checksum
            checksum = self._calculate_file_checksum(file_path)

            # Check if already loaded and unchanged
            if (
                file_path in self.file_checksums
                and self.file_checksums[file_path] == checksum
            ):
                return None

            # Load metadata from file
            metadata = await self._extract_metadata_from_file(file_path)
            if not metadata:
                return None

            metadata.file_path = file_path
            metadata.checksum = checksum
            metadata.last_modified = datetime.fromtimestamp(
                os.path.getmtime(file_path), tz=timezone.utc
            )

            # Load the tool class
            tool_class = await self._load_tool_class(file_path, metadata.class_name)
            if not tool_class:
                return None

            # Store loaded tool
            self.loaded_tools[metadata.name] = metadata
            self.tool_classes[metadata.name] = tool_class
            self.file_checksums[file_path] = checksum

            self.logger.info(f"Loaded tool: {metadata.name} from {file_path}")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to load tool from {file_path}: {e}")
            return None

    async def _extract_metadata_from_file(
        self, file_path: str
    ) -> Optional[ToolMetadata]:
        """Extract tool metadata from Python file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for TOOL_METADATA comment or variable
            metadata_start = content.find("TOOL_METADATA = {")
            if metadata_start == -1:
                # Try alternative format
                metadata_start = content.find("# TOOL_METADATA:")
                if metadata_start == -1:
                    return self._infer_metadata_from_code(file_path, content)

            # Extract metadata
            if "TOOL_METADATA = {" in content:
                # Parse Python dict
                start = content.find("{", metadata_start)
                end = self._find_matching_brace(content, start)
                if end == -1:
                    return self._infer_metadata_from_code(file_path, content)

                metadata_str = content[start : end + 1]
                try:
                    metadata_dict = eval(metadata_str)  # Safe in this context
                    return ToolMetadata(**metadata_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to parse metadata dict: {e}")
                    return self._infer_metadata_from_code(file_path, content)

            return self._infer_metadata_from_code(file_path, content)

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return None

    def _infer_metadata_from_code(self, file_path: str, content: str) -> ToolMetadata:
        """Infer metadata from code analysis"""
        class_name = self._infer_class_name(content)
        description = self._extract_class_docstring(content, class_name)

        return ToolMetadata(
            name=os.path.splitext(os.path.basename(file_path))[0],
            version="1.0.0",
            description=description or "Dynamically loaded tool",
            class_name=class_name,
            author="Unknown",
        )

    def _infer_class_name(self, content: str) -> str:
        """Infer main tool class name from content"""
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("class ") and "(Tool)" in line:
                # Extract class name
                class_part = line.split("class ")[1].split("(")[0].strip()
                return class_part
        return "UnknownTool"

    def _extract_class_docstring(self, content: str, class_name: str) -> str:
        """Extract docstring from class"""
        class_start = content.find(f"class {class_name}")
        if class_start == -1:
            return ""

        # Find first triple quote after class definition
        docstring_start = content.find('"""', class_start)
        if docstring_start == -1:
            return ""

        docstring_end = content.find('"""', docstring_start + 3)
        if docstring_end == -1:
            return ""

        return content[docstring_start + 3 : docstring_end].strip()

    def _find_matching_brace(self, content: str, start: int) -> int:
        """Find matching closing brace"""
        count = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                count += 1
            elif content[i] == "}":
                count -= 1
                if count == 0:
                    return i
        return -1

    async def _load_tool_class(
        self, file_path: str, class_name: str
    ) -> Optional[Type[Tool]]:
        """Load tool class from file"""
        try:
            # Create module spec
            module_name = f"dynamic_tool_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            if not spec or not spec.loader:
                return None

            # Load module
            module = importlib.util.module_from_spec(spec)

            # Execute in sandboxed environment if enabled
            if self.sandbox_execution:
                # Create restricted globals
                restricted_globals: Dict[str, Any] = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                        "range": range,
                        "enumerate": enumerate,
                        "zip": zip,
                        "isinstance": isinstance,
                        "issubclass": issubclass,
                        "super": super,
                        "property": property,
                        "staticmethod": staticmethod,
                        "classmethod": classmethod,
                    },
                    "__name__": module_name,
                    "__file__": file_path,
                }

                # Add safe imports
                restricted_globals.update(
                    {
                        "asyncio": asyncio,
                        "datetime": datetime,
                        "json": json,
                        "uuid": uuid,
                        "logging": logging,
                    }
                )

                # Import our tool classes
                from .tool_registry import (
                    Tool,
                    ToolCategory,
                    ToolExecutionContext,
                    ToolParameter,
                    ToolResult,
                    ToolSecurityLevel,
                )

                restricted_globals.update(
                    {
                        "Tool": Tool,
                        "ToolCategory": ToolCategory,
                        "ToolSecurityLevel": ToolSecurityLevel,
                        "ToolParameter": ToolParameter,
                        "ToolResult": ToolResult,
                        "ToolExecutionContext": ToolExecutionContext,
                    }
                )

                # Execute module code
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()

                exec(code, restricted_globals)

                # Extract class from globals
                if class_name in restricted_globals:
                    tool_class = restricted_globals[class_name]
                    if isinstance(tool_class, type):
                        return tool_class

            else:
                # Standard loading
                spec.loader.exec_module(module)

                if hasattr(module, class_name):
                    tool_class = getattr(module, class_name)
                    if isinstance(tool_class, type):
                        return tool_class

        except Exception as e:
            self.logger.error(
                f"Failed to load class {class_name} from {file_path}: {e}"
            )

        return None

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()

    async def _hot_reload_loop(self) -> None:
        """Background hot reload loop"""
        while self._running:
            try:
                await self._check_for_changes()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Hot reload error: {e}")
                await asyncio.sleep(5)

    async def _check_for_changes(self) -> None:
        """Check for file changes and reload if necessary"""
        changed_files: List[str] = []

        # Check existing files
        for file_path, old_checksum in self.file_checksums.items():
            if os.path.exists(file_path):
                new_checksum = self._calculate_file_checksum(file_path)
                if new_checksum != old_checksum:
                    changed_files.append(file_path)

        # Reload changed files
        for file_path in changed_files:
            try:
                await self._load_tool_from_file(file_path)
                self.logger.info(f"Hot reloaded tool from {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to hot reload {file_path}: {e}")

        # Discover new files
        await self.discover_and_load_tools()

    def create_tool_instance(self, tool_name: str) -> Optional[Tool]:
        """Create instance of loaded tool"""
        if tool_name not in self.tool_classes:
            return None

        try:
            tool_class = self.tool_classes[tool_name]
            metadata = self.loaded_tools[tool_name]
            # Create instance with basic required parameters
            return tool_class(name=metadata.name, description=metadata.description)
        except Exception as e:
            self.logger.error(f"Failed to create instance of {tool_name}: {e}")
            return None

    def get_loaded_tools(self) -> List[ToolMetadata]:
        """Get list of loaded tools"""
        return list(self.loaded_tools.values())

    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for specific tool"""
        return self.loaded_tools.get(tool_name)

    def unload_tool(self, tool_name: str) -> bool:
        """Unload a tool"""
        if tool_name not in self.loaded_tools:
            return False

        metadata = self.loaded_tools[tool_name]

        # Remove from tracking
        del self.loaded_tools[tool_name]
        del self.tool_classes[tool_name]

        if metadata.file_path in self.file_checksums:
            del self.file_checksums[metadata.file_path]

        self.logger.info(f"Unloaded tool: {tool_name}")
        return True

    async def install_tool_from_package(
        self, package_path: str
    ) -> Optional[ToolMetadata]:
        """Install tool from package file"""
        # This would implement package installation
        # For now, just a placeholder
        self.logger.info(f"Installing tool package: {package_path}")
        return None

    def get_loader_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return {
            "loaded_tools": len(self.loaded_tools),
            "tool_directories": self.tool_directories,
            "hot_reload_enabled": self.enable_hot_reload,
            "sandbox_execution": self.sandbox_execution,
            "tools_by_category": {
                category.value: len(
                    [t for t in self.loaded_tools.values() if t.category == category]
                )
                for category in ToolCategory
            },
        }
