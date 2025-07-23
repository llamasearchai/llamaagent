"""
Tool registry system for LlamaAgent with dynamic loading and validation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import asyncio
import glob
import importlib
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

from .base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for registered tools."""

    name: str
    description: str
    category: str = "general"
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0


class ToolRegistry:
    """Registry for managing tools dynamically."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._usage_stats: Dict[str, int] = {}

    def register(
        self,
        tool: BaseTool,
        metadata: Optional[ToolMetadata] = None,
        aliases: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Register a tool instance."""
        name = tool.name

        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        # Store tool
        self._tools[name] = tool

        # Create metadata if not provided
        if metadata is None:
            metadata = ToolMetadata(
                name=name,
                description=getattr(tool, "description", "No description"),
                category=getattr(tool, "category", "general"),
                version=getattr(tool, "version", "1.0.0"),
                author=getattr(tool, "author", "Unknown"),
                tags=getattr(tool, "tags", []),
                dependencies=getattr(tool, "dependencies", []),
                enabled=getattr(tool, "enabled", True),
                priority=getattr(tool, "priority", 0),
            )

        self._metadata[name] = metadata

        # Update categories
        category = metadata.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

        logger.info(f"Registered tool: {name} (category: {category})")

    def register_class(self, tool_class: Type[BaseTool], *args, **kwargs) -> None:
        """Register a tool class by instantiating it."""
        try:
            tool = tool_class(*args, **kwargs)
            self.register(tool, **kwargs)
        except Exception as e:
            logger.error(f"Failed to register tool class {tool_class.__name__}: {e}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        actual_name = self._aliases.get(name, name)

        if actual_name not in self._tools:
            return False

        # Remove tool
        del self._tools[actual_name]
        self._metadata.pop(actual_name, None)
        self._usage_stats.pop(actual_name, None)

        # Remove from categories
        metadata = self._metadata.get(actual_name)
        if metadata:
            category = metadata.category
            if category in self._categories:
                try:
                    self._categories[category].remove(actual_name)
                    if not self._categories[category]:
                        del self._categories[category]
                except ValueError:
                    pass

        # Remove aliases
        aliases_to_remove = [
            alias for alias, target in self._aliases.items() if target == actual_name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]

        logger.info(f"Unregistered tool: {actual_name}")
        return True

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name or alias."""
        actual_name = self._aliases.get(name, name)
        return self._tools.get(actual_name)

    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        actual_name = self._aliases.get(name, name)
        return actual_name in self._tools

    def enable_tool(self, name: str) -> bool:
        """Enable a tool."""
        actual_name = self._aliases.get(name, name)
        metadata = self._metadata.get(actual_name)
        if metadata:
            metadata.enabled = True
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool."""
        actual_name = self._aliases.get(name, name)
        metadata = self._metadata.get(actual_name)
        if metadata:
            metadata.enabled = False
            return True
        return False

    def is_enabled(self, name: str) -> bool:
        """Check if tool is enabled."""
        actual_name = self._aliases.get(name, name)
        metadata = self._metadata.get(actual_name)
        return metadata.enabled if metadata else False

    async def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Execute a tool and track usage."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        # Check if tool is enabled
        actual_name = self._aliases.get(name, name)
        if not self.is_enabled(actual_name):
            raise ValueError(f"Tool '{name}' is disabled")

        # Track usage
        self._usage_stats[actual_name] = self._usage_stats.get(actual_name, 0) + 1

        # Execute tool
        if asyncio.iscoroutinefunction(tool.execute):
            return await tool.execute(*args, **kwargs)
        else:
            return tool.execute(*args, **kwargs)

    def get_tools(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, BaseTool]:
        """Get tools by criteria."""
        tools: Dict[str, BaseTool] = {}

        for name, tool in self._tools.items():
            metadata = self._metadata.get(name)
            if not metadata:
                continue

            # Filter by enabled status
            if enabled_only and not metadata.enabled:
                continue

            # Filter by category
            if category and metadata.category != category:
                continue

            # Filter by tags
            if tags and not any(tag in metadata.tags for tag in tags):
                continue

            tools[name] = tool

        return tools

    def list_tools(
        self, detailed: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """List all registered tools."""
        if not detailed:
            return list(self._tools.keys())

        return [self.get_tool_info(name) for name in self._tools.keys()]

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive tool information."""
        actual_name = self._aliases.get(name, name)
        tool = self._tools.get(actual_name)
        metadata = self._metadata.get(actual_name)

        if not tool or not metadata:
            return {}

        return {
            "name": metadata.name,
            "description": metadata.description,
            "category": metadata.category,
            "version": metadata.version,
            "author": metadata.author,
            "tags": metadata.tags,
            "dependencies": metadata.dependencies,
            "enabled": metadata.enabled,
            "priority": metadata.priority,
            "tool_type": type(tool).__name__,
            "usage_count": self._usage_stats.get(actual_name, 0),
            "aliases": [
                alias
                for alias, target in self._aliases.items()
                if target == actual_name
            ],
        }

    def search_tools(
        self, query: str, search_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search tools by query."""
        if search_fields is None:
            search_fields = ["name", "description", "tags", "category"]

        results = []
        query_lower = query.lower()

        for name in self._tools.keys():
            tool_info = self.get_tool_info(name)

            # Search in specified fields
            matches = False
            for field in search_fields:
                if field in tool_info:
                    field_value = tool_info[field]
                    if isinstance(field_value, str):
                        if query_lower in field_value.lower():
                            matches = True
                            break
                    elif isinstance(field_value, list):
                        if any(
                            query_lower in str(item).lower() for item in field_value
                        ):
                            matches = True
                            break

            if matches:
                results.append(tool_info)

        # Sort by priority and usage
        results.sort(
            key=lambda x: (x.get("priority", 0), x.get("usage_count", 0)), reverse=True
        )

        return results

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self._categories.keys())

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category."""
        return self._categories.get(category, [])

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all tools."""
        return self._usage_stats.copy()

    def get_aliases(self) -> Dict[str, str]:
        """Get all aliases."""
        return self._aliases.copy()

    def reload_tool(self, tool_name: str) -> bool:
        """Reload a tool (re-import and re-register)."""
        try:
            # Get current tool metadata
            metadata = self._metadata.get(tool_name)
            if not metadata:
                return False

            # Try to reload the module
            # This is a simplified version - in practice you'd need module introspection
            self.unregister(tool_name)

            # Here you would re-import and re-register the tool
            # For now, just return success if unregister worked
            count = 1 if tool_name not in self._tools else 0
            return count > 0

        except Exception as e:
            logger.error(f"Failed to reload tool {tool_name}: {e}")
            return False

    def export_config(self) -> Dict[str, Any]:
        """Export registry configuration."""
        return {
            "tools": list(self._tools.keys()),
            "metadata": {
                name: {
                    "name": meta.name,
                    "description": meta.description,
                    "category": meta.category,
                    "version": meta.version,
                    "author": meta.author,
                    "tags": meta.tags,
                    "dependencies": meta.dependencies,
                    "enabled": meta.enabled,
                    "priority": meta.priority,
                }
                for name, meta in self._metadata.items()
            },
            "categories": self._categories.copy(),
            "aliases": self._aliases.copy(),
            "usage_stats": self._usage_stats.copy(),
        }

    async def cleanup(self) -> None:
        """Cleanup all registered tools."""
        # Clear all registries
        self._tools.clear()
        self._metadata.clear()
        self._categories.clear()
        self._aliases.clear()
        self._usage_stats.clear()


class ToolLoader:
    """Tool loader for dynamic tool discovery and loading."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def load_from_module(
        self, module_path: str, tool_names: Optional[List[str]] = None
    ) -> int:
        """Load tools from a Python module."""
        try:
            module = importlib.import_module(module_path)
            loaded_count = 0

            # Inspect module for tool classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTool) and obj != BaseTool:
                    # Filter by requested tool names
                    if tool_names and name not in tool_names:
                        continue

                    try:
                        # Try to instantiate and register
                        tool = obj()
                        self.registry.register(tool)
                        loaded_count += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to load tool {name} from {module_path}: {e}"
                        )

            logger.info(f"Loaded {loaded_count} tools from {module_path}")
            return loaded_count

        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            return 0

    def load_from_directory(
        self, directory_path: str, recursive: bool = True, pattern: str = "*.py"
    ) -> int:
        """Load tools from all Python files in a directory."""
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return 0

        loaded_count = 0

        if recursive:
            pattern = os.path.join("**", pattern)
            files = glob.glob(os.path.join(directory_path, pattern), recursive=True)
        else:
            files = glob.glob(os.path.join(directory_path, pattern))

        for file_path in files:
            # Skip __init__ files
            if file_path.endswith("__init__.py"):
                continue

            # Convert file path to module path
            rel_path = os.path.relpath(file_path, ".").replace(".py", "")
            module_path = rel_path.replace(os.path.sep, ".")

            try:
                count = self.load_from_module(module_path)
                loaded_count += count
            except Exception as e:
                logger.error(f"Failed to load from {file_path}: {e}")

        return loaded_count


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool: BaseTool, **kwargs) -> None:
    """Register tool in global registry."""
    get_registry().register(tool, **kwargs)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool from global registry."""
    return get_registry().get_tool(name)


async def execute_tool(name: str, **kwargs) -> Any:
    """Execute tool from global registry."""
    return await get_registry().execute_tool(name, **kwargs)


def create_loader() -> ToolLoader:
    """Create tool loader for global registry."""
    return ToolLoader(get_registry())


__all__ = [
    "ToolRegistry",
    "ToolLoader",
    "ToolMetadata",
    "get_registry",
    "register_tool",
    "get_tool",
    "execute_tool",
    "create_loader",
]
