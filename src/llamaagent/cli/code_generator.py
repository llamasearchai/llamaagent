#!/usr/bin/env python3
"""
Advanced Code Generation System, Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..llm.factory import LLMFactory
from ..types import LLMMessage

logger = logging.getLogger(__name__)


class ProgrammingLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    AUTO = "auto"


@dataclass
class CodeGenerationRequest:
    """Request for code generation."""

    prompt: str
    language: ProgrammingLanguage
    framework: Optional[str] = None
    style: str = "production"
    context: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedCode:
    """Generated code result."""

    code: str
    language: ProgrammingLanguage
    filename: str
    dependencies: List[str]
    metadata: Dict[str, Any]


class CodeGenerator:
    """Advanced code generation system."""

    def __init__(self, model: str = "gpt-4") -> None:
        self.factory = LLMFactory()
        self.provider = self.factory.create_provider("openai", model_name=model)
        self.history: List[Dict[str, Any]] = []

    async def generate_code(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate code based on the request."""

        # Auto-detect language if needed
        if request.language == ProgrammingLanguage.AUTO:
            request.language = self._detect_language(request.prompt)

        # Build the prompt
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        # Generate main code
        start_time = time.time()
        response = await self.provider.complete(messages)

        # Extract code from response
        code = self._extract_code_from_response(
            response.content if hasattr(response, 'content') else str(response)
        )

        # Generate filename
        filename = self._generate_filename(request.prompt, request.language)

        # Extract dependencies
        dependencies = self._extract_dependencies(code, request.language)

        # Create metadata
        metadata = {
            "generation_time": time.time() - start_time,
            "model": (
                self.provider.model_name
                if hasattr(self.provider, 'model_name')
                else "unknown"
            ),
            "prompt_tokens": len(system_prompt.split()) + len(user_prompt.split()),
            "style": request.style,
            "framework": request.framework,
        }

        result = GeneratedCode(
            code=code,
            language=request.language,
            filename=filename,
            dependencies=dependencies,
            metadata=metadata,
        )

        # Store in history
        self.history.append(
            {"request": request, "result": result, "timestamp": time.time()}
        )

        return result

    def _detect_language(self, prompt: str) -> ProgrammingLanguage:
        """Auto-detect programming language from prompt."""
        prompt_lower = prompt.lower()

        language_indicators = {
            ProgrammingLanguage.PYTHON: [
                "python",
                "py",
                "django",
                "flask",
                "pandas",
                "numpy",
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                "javascript",
                "js",
                "node",
                "react",
                "vue",
                "angular",
            ],
            ProgrammingLanguage.TYPESCRIPT: ["typescript", "ts", "angular", "nest"],
            ProgrammingLanguage.JAVA: ["java", "spring", "maven", "gradle"],
            ProgrammingLanguage.RUST: ["rust", "cargo", "tokio"],
            ProgrammingLanguage.GO: ["go", "golang", "gin", "gorilla"],
            ProgrammingLanguage.HTML: ["html", "webpage", "web page", "markup"],
            ProgrammingLanguage.CSS: ["css", "style", "stylesheet"],
            ProgrammingLanguage.SQL: ["sql", "database", "query", "select"],
            ProgrammingLanguage.BASH: ["bash", "shell", "script", "command"],
        }

        for language, indicators in language_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                return language

        return ProgrammingLanguage.PYTHON

    def _build_system_prompt(self, request: CodeGenerationRequest) -> str:
        """Build system prompt for code generation."""
        return f"""You are an expert {request.language.value} developer.
Generate high-quality, production-ready code that follows best practices.

REQUIREMENTS:
1. Generate clean, readable code
2. Include proper error handling
3. Add appropriate comments
4. Follow language conventions
5. Use modern language features
6. Include necessary imports
7. Provide ONLY the code without explanations

LANGUAGE: {request.language.value}
STYLE: {request.style}
{f"FRAMEWORK: {request.framework}" if request.framework else ""}"""

    def _build_user_prompt(self, request: CodeGenerationRequest) -> str:
        """Build user prompt for code generation."""
        prompt = f"Generate {request.language.value} code for: {request.prompt}"

        if request.context and "requirements" in request.context:
            prompt += "\n\nRequirements:\n"
            for req in request.context["requirements"]:
                prompt += f"- {req}\n"

        return prompt

    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean code from LLM response."""
        content = response.strip()

        if "```" in content:
            lines = content.split("\n")
            in_code_block = False
            code_lines = []

            for line in lines:
                if line.strip().startswith("```"):
                    if in_code_block:
                        break
                    else:
                        in_code_block = True
                        continue

                if in_code_block:
                    code_lines.append(line)

            return "\n".join(code_lines)

        return content

    def _generate_filename(self, prompt: str, language: ProgrammingLanguage) -> str:
        """Generate appropriate filename."""
        extensions = {
            ProgrammingLanguage.PYTHON: ".py",
            ProgrammingLanguage.JAVASCRIPT: ".js",
            ProgrammingLanguage.TYPESCRIPT: ".ts",
            ProgrammingLanguage.JAVA: ".java",
            ProgrammingLanguage.RUST: ".rs",
            ProgrammingLanguage.GO: ".go",
            ProgrammingLanguage.HTML: ".html",
            ProgrammingLanguage.CSS: ".css",
            ProgrammingLanguage.SQL: ".sql",
            ProgrammingLanguage.BASH: ".sh",
        }

        extension = extensions.get(language, ".txt")

        # Generate base name from prompt
        words = prompt.lower().split()
        name = "_".join([w for w in words[:3] if len(w) > 2])
        if not name:
            name = "generated_code"

        return f"{name}{extension}"

    def _extract_dependencies(
        self, code: str, language: ProgrammingLanguage
    ) -> List[str]:
        """Extract dependencies from code."""
        dependencies = []
        lines = code.split("\n")

        if language == ProgrammingLanguage.PYTHON:
            for line in lines:
                line = line.strip()
                if line.startswith("import ", "from "):
                    if line.startswith("import "):
                        module = line[7:].split()[0].split(".")[0]
                    else:
                        module = line.split()[1].split(".")[0]
                    if module not in ["os", "sys", "json", "time", "math", "random"]:
                        dependencies.append(module)

        elif language == ProgrammingLanguage.JAVASCRIPT:
            for line in lines:
                line = line.strip()
                if line.startswith("import ", "const ", "require("):
                    if "from" in line:
                        parts = line.split("from")
                        if len(parts) > 1:
                            module = parts[1].strip().strip("'\"").strip(";")
                            if not module.startswith("./"):
                                dependencies.append(module)

        return list(set(dependencies))

    def get_history(self) -> List[Dict[str, Any]]:
        """Get generation history."""
        return self.history

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data."""
        result = {
            "processed_at": time.time(),
            "input": input_data,
            "status": "processed",
        }

        self.history.append(result)
        return result


# Example usage
async def main() -> None:
    """Example usage of the code generator."""
    generator = CodeGenerator()

    request = CodeGenerationRequest(
        prompt="Create a function that calculates the factorial of a number",
        language=ProgrammingLanguage.PYTHON,
        style="production",
        context={
            "requirements": ["Handle edge cases", "Include type hints", "Add docstring"]
        },
    )

    result = await generator.generate_code(request)
    print(f"Generated {result.filename}")
    print(result.code)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
