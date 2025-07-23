#!/usr/bin/env python3
"""
Role Management System with JSON Configuration
Author: Nik Jois <nikjois@llamasearch.ai>

This module provides:
- Predefined roles with specialized prompts
- Custom role creation and management
- JSON-based role configuration
- Role templates and inheritance
- Dynamic role switching
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Role:
    """Role configuration."""

    name: str
    description: str
    system_prompt: str
    category: str = "general"
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    author: str = "system"
    version: str = "1.0"


class RoleManager:
    """Manage roles and their configurations."""

    def __init__(self, roles_dir: Optional[str] = None):
        if roles_dir is None:
            self.roles_dir = Path.home() / ".config" / "llamaagent" / "roles"
        else:
            self.roles_dir = Path(roles_dir)
        self.roles_dir.mkdir(parents=True, exist_ok=True)

        # Initialize role collections
        self.builtin_roles: Dict[str, Role] = {}
        self.custom_roles: Dict[str, Role] = {}
        self.history: List[Dict[str, Any]] = []

        # Load roles
        self._create_builtin_roles()
        self.custom_roles.update(self._load_custom_roles())

    def _create_builtin_roles(self) -> None:
        """Create built-in role configurations."""
        builtin_roles = {
            "default": Role(
                name="default",
                description="General-purpose AI assistant",
                system_prompt="You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries. Be concise but comprehensive in your explanations.",
                category="general",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["general", "assistant"],
            ),
            "programmer": Role(
                name="programmer",
                description="Expert programmer proficient in multiple languages",
                system_prompt="You are an expert programmer with deep knowledge of multiple programming languages, frameworks, and best practices. Focus on writing clean, efficient, and well-documented code. Guidelines: - Generate only code unless explanations are requested - Follow best practices for the chosen language - Include necessary imports and dependencies - Write production-ready code - Add comments for complex logic - Consider error handling and edge cases",
                category="programming",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["programming", "code", "development"],
            ),
            "researcher": Role(
                name="researcher",
                description="Academic researcher and information analyst",
                system_prompt="You are an expert academic researcher with strong analytical skills. Your expertise includes conducting thorough research, analyzing information, and synthesizing findings. Your research capabilities: - Literature review and source evaluation - Methodology design and analysis - Citation and reference management - Critical thinking and hypothesis formation - Data interpretation and conclusions - Academic writing and reporting. Maintain high standards for evidence and logical reasoning. Clearly distinguish between established facts and speculation.",
                category="academic",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["research", "academic", "analysis"],
            ),
            "tutor": Role(
                name="tutor",
                description="Patient and knowledgeable educator",
                system_prompt="You are a patient, knowledgeable educator who excels at explaining complex concepts in simple terms. Adapt your teaching style to different learning preferences and provide step-by-step guidance. Focus on understanding rather than memorization.",
                category="education",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["education", "tutoring", "learning"],
            ),
            "writer": Role(
                name="writer",
                description="Creative writing assistant and storyteller",
                system_prompt="You are a skilled creative writer and storyteller with expertise in various genres and writing styles. Focus on compelling narratives, character development, and attention to narrative structure. Adapt your writing style to match the requested genre and tone. Your capabilities include: - Story development and plot creation - Character development and dialogue - Creative writing techniques and style advice - Poetry and prose composition - Content adaptation across different formats - Creative brainstorming and idea generation",
                category="creative",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["writing", "creative", "storytelling"],
            ),
            "analyst": Role(
                name="analyst",
                description="Expert data analyst and statistician",
                system_prompt="You are an expert data analyst with strong statistical background and analytical thinking. Focus on data-driven insights and evidence-based conclusions. Your capabilities include: - Statistical analysis and interpretation - Data visualization recommendations - Pattern recognition and trend analysis - Hypothesis testing and validation - Data cleaning and preprocessing advice - Machine learning model suggestions. Provide clear explanations of your analysis methodology and findings.",
                category="data",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["analysis", "statistics", "data"],
            ),
            "translator": Role(
                name="translator",
                description="Professional translator and language expert",
                system_prompt="You are a professional translator with expertise in multiple languages and cultural nuances. Your translation approach: - Maintain the original meaning and intent - Consider cultural context and idioms - Preserve the appropriate tone and style - Provide alternative translations when ambiguous - Explain cultural references when necessary - Adapt formatting and structure appropriately",
                category="language",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["translation", "language", "culture"],
            ),
            "consultant": Role(
                name="consultant",
                description="Strategic business consultant and advisor",
                system_prompt="You are a strategic business consultant with expertise in business analysis, strategic planning, and implementation feasibility. Structure your advice clearly with executive summaries when appropriate. Focus on practical, actionable recommendations.",
                category="business",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["business", "strategy", "consulting"],
            ),
            "debugger": Role(
                name="debugger",
                description="Expert code debugger and problem solver",
                system_prompt="You are an expert at debugging code and solving technical problems. Focus on systematic problem-solving approaches and clear explanations of issues and solutions.",
                category="technical",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["debugging", "troubleshooting", "technical"],
            ),
            "shell": Role(
                name="shell",
                description="Expert system administrator and shell command specialist",
                system_prompt="You are an expert system administrator who provides efficient shell commands for the user's operating system. Guidelines: - Generate only the shell command that accomplishes the task - Ask for clarification if the request is ambiguous - Provide commands that are safe and efficient - Consider the user's operating system context",
                category="technical",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["shell", "system", "commands"],
            ),
            "math": Role(
                name="math",
                description="Mathematics tutor specializing in step-by-step explanations",
                system_prompt="You are a mathematics tutor who excels at breaking down complex mathematical problems into clear, step-by-step explanations. Focus on helping users understand the underlying concepts and problem-solving strategies.",
                category="education",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["mathematics", "tutoring", "education"],
            ),
            "visualizer": Role(
                name="visualizer",
                description="Data visualization expert",
                system_prompt="You are an expert in data visualization and creating compelling visual representations of information. Focus on choosing appropriate chart types, color schemes, and layouts that effectively communicate insights.",
                category="data",
                author="Nik Jois <nikjois@llamasearch.ai>",
                tags=["visualization", "data", "charts"],
            ),
        }

        self.builtin_roles = builtin_roles

    def _load_custom_roles(self) -> Dict[str, Role]:
        """Load custom roles from JSON files."""
        roles = {}

        for role_file in self.roles_dir.glob("*.json"):
            try:
                with open(role_file, "r") as f:
                    role_data = json.load(f)

                role = Role(**role_data)
                roles[role.name] = role

            except Exception as e:
                logger.error(f"Error loading role from {role_file}: {e}")

        return roles

    def _save_role_to_file(self, role: Role) -> None:
        """Save role to JSON file."""
        role_file = self.roles_dir / f"{role.name}.json"

        try:
            with open(role_file, "w") as f:
                json.dump(asdict(role), f, indent=2)
            logger.info(f"Role '{role.name}' saved to {role_file}")
        except Exception as e:
            logger.error(f"Error saving role '{role.name}': {e}")

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        # Check built-in roles first
        if name in self.builtin_roles:
            return self.builtin_roles[name]

        # Check custom roles
        if name in self.custom_roles:
            return self.custom_roles[name]

        return None

    def list_roles(self, category: Optional[str] = None) -> List[Role]:
        """List all available roles, optionally filtered by category."""
        all_roles = list(self.builtin_roles.values()) + list(self.custom_roles.values())

        if category:
            return [role for role in all_roles if role.category == category]

        return all_roles

    def create_role(
        self,
        name: str,
        description: str,
        system_prompt: str,
        category: str = "custom",
        **kwargs,
    ) -> Role:
        """Create a new custom role."""
        if name in self.custom_roles:
            raise ValueError(f"Role '{name}' already exists")

        role = Role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            category=category,
            author=kwargs.get("author", "user"),
            **kwargs,
        )

        self.custom_roles[name] = role
        self._save_role_to_file(role)

        # Log to history
        self.history.append(
            {"action": "create", "role": name, "timestamp": time.time()}
        )

        return role

    def update_role(self, name: str, **updates) -> bool:
        """Update an existing role."""
        if name not in self.custom_roles:
            return False

        role = self.custom_roles[name]

        # Update fields
        for field, value in updates.items():
            if hasattr(role, field):
                setattr(role, field, value)

        role.updated_at = time.time()
        self._save_role_to_file(role)

        # Log to history
        self.history.append(
            {
                "action": "update",
                "role": name,
                "updates": updates,
                "timestamp": time.time(),
            }
        )

        return True

    def delete_role(self, name: str) -> bool:
        """Delete a custom role."""
        if name not in self.custom_roles:
            return False

        # Remove from memory
        del self.custom_roles[name]

        # Remove file
        role_file = self.roles_dir / f"{name}.json"
        if role_file.exists():
            role_file.unlink()

        # Log to history
        self.history.append(
            {"action": "delete", "role": name, "timestamp": time.time()}
        )

        return True

    def duplicate_role(self, source_name: str, new_name: str) -> Optional[Role]:
        """Duplicate an existing role with a new name."""
        source_role = self.get_role(source_name)
        if not source_role:
            return None

        if new_name in self.custom_roles:
            raise ValueError(f"Role '{new_name}' already exists")

        # Create new role based on source
        new_role = Role(
            name=new_name,
            description=source_role.description,
            system_prompt=source_role.system_prompt,
            category=source_role.category,
            parameters=source_role.parameters.copy(),
            tags=source_role.tags.copy(),
            author="user",
        )

        self.custom_roles[new_name] = new_role
        self._save_role_to_file(new_role)

        # Log to history
        self.history.append(
            {
                "action": "duplicate",
                "source": source_name,
                "new_role": new_name,
                "timestamp": time.time(),
            }
        )

        return new_role

    def import_role(self, file_path: str, overwrite: bool = False) -> Optional[Role]:
        """Import a role from a JSON file."""
        try:
            with open(file_path, "r") as f:
                role_data = json.load(f)

            role = Role(**role_data)

            if role.name in self.custom_roles and not overwrite:
                raise ValueError(
                    f"Role '{role.name}' already exists. Use overwrite=True to replace."
                )

            self.custom_roles[role.name] = role
            self._save_role_to_file(role)

            # Log to history
            self.history.append(
                {
                    "action": "import",
                    "role": role.name,
                    "source": file_path,
                    "timestamp": time.time(),
                }
            )

            return role

        except Exception as e:
            logger.error(f"Error importing role from {file_path}: {e}")
            return None

    def export_role(self, name: str, export_path: str) -> bool:
        """Export a role to a file."""
        role = self.get_role(name)
        if not role:
            return False

        try:
            with open(export_path, "w") as f:
                json.dump(asdict(role), f, indent=2)

            # Log to history
            self.history.append(
                {
                    "action": "export",
                    "role": name,
                    "destination": export_path,
                    "timestamp": time.time(),
                }
            )

            return True
        except Exception as e:
            logger.error(f"Error exporting role '{name}': {e}")
            return False

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        categories = set()
        for role in self.list_roles():
            categories.add(role.category)
        return sorted(list(categories))

    def search_roles(self, query: str) -> List[Role]:
        """Search roles by name, description, or tags."""
        query = query.lower()
        results = []

        for role in self.list_roles():
            if (
                query in role.name.lower()
                or query in role.description.lower()
                or any(query in tag.lower() for tag in role.tags)
            ):
                results.append(role)

        return results

    def get_role_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a role."""
        role = self.get_role(name)
        if not role:
            return None

        return {
            "name": role.name,
            "description": role.description,
            "category": role.category,
            "tags": role.tags,
            "author": role.author,
            "version": role.version,
            "created_at": role.created_at,
            "updated_at": role.updated_at,
            "prompt_length": len(role.system_prompt),
            "parameters": role.parameters,
            "is_builtin": role.name in self.builtin_roles,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get role management history."""
        return self.history

    def validate_role(self, role: Role) -> List[str]:
        """Validate a role configuration."""
        issues = []

        if not role.name:
            issues.append("Role name is required")

        if not role.description:
            issues.append("Role description is required")

        if not role.system_prompt:
            issues.append("System prompt is required")

        if len(role.system_prompt) < 10:
            issues.append("System prompt is too short")

        if not role.category:
            issues.append("Category is required")

        return issues


def create_custom_role_interactive() -> None:
    """Interactive custom role creation."""
    print("SYSTEM Custom Role Creator")
    print("=" * 30)

    manager = RoleManager()

    # Get role details
    name = input("Role name: ").strip()
    if not name:
        print("FAIL Role name is required")
        return

    description = input("Role description: ").strip()
    if not description:
        print("FAIL Role description is required")
        return

    print("\nEnter the system prompt (press Enter twice to finish):")
    prompt_lines = []
    while True:
        line = input()
        if line == "" and prompt_lines and prompt_lines[-1] == "":
            break
        prompt_lines.append(line)

    system_prompt = "\n".join(prompt_lines).strip()
    if not system_prompt:
        print("FAIL System prompt is required")
        return

    # Optional fields
    category = input("Category [custom]: ").strip() or "custom"
    tags = input("Tags (comma-separated): ").strip()
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    try:
        custom_role = manager.create_role(
            name=name,
            description=description,
            system_prompt=system_prompt,
            category=category,
            tags=tag_list,
            author="user",
        )

        print(f"\nPASS Created custom role: {custom_role.name}")
        print(f" Category: {custom_role.category}")
        print(
            f"TAG:  Tags: {', '.join(custom_role.tags) if custom_role.tags else 'None'}"
        )
        print(f"Parameters Parameters: {json.dumps(custom_role.parameters, indent=2)}")

    except ValueError as e:
        print(f"FAIL Role creation failed: {e}")


def main() -> None:
    """Example usage of the role manager."""
    manager = RoleManager()

    # List all roles
    print("Available roles:")
    for role in manager.list_roles():
        print(f"- {role.name}: {role.description}")

    # Get a specific role
    programmer_role = manager.get_role("programmer")
    if programmer_role:
        print(f"\nProgrammer role prompt: {programmer_role.system_prompt[:100]}...")

    # Search roles
    code_roles = manager.search_roles("code")
    print(f"\nRoles related to 'code': {[r.name for r in code_roles]}")


if __name__ == "__main__":
    main()
