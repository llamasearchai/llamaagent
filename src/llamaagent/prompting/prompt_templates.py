"""
Advanced Prompt Templates and Library

Provides a comprehensive collection of optimized prompt templates
for various tasks and a dynamic template system.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TemplateType(Enum):
    """Types of prompt templates"""

    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    CONVERSATIONAL = "conversational"
    INSTRUCTION = "instruction"


@dataclass
class PromptTemplate:
    """Advanced prompt template with dynamic variable substitution"""

    name: str
    template: str
    type: TemplateType
    variables: List[str]
    description: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        # Check all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Advanced formatting with nested templates
        formatted = self.template

        # Handle conditional sections
        formatted = self._process_conditionals(formatted, kwargs)

        # Handle loops
        formatted = self._process_loops(formatted, kwargs)

        # Replace variables
        for var, value in kwargs.items():
            pattern = rf"\{{{{\s*{var}\s*\}}}}"
            formatted = re.sub(pattern, str(value), formatted)

        # Add examples if requested
        if kwargs.get("include_examples", False) and self.examples:
            formatted += self._format_examples()

        # Add constraints if requested
        if kwargs.get("include_constraints", False) and self.constraints:
            formatted += self._format_constraints()

        return formatted.strip()

    def _process_conditionals(self, template: str, variables: Dict[str, Any]) -> str:
        """Process conditional sections in template"""
        # Pattern: {% if variable %} content {% endif %}
        pattern = r"{%\s*if\s+(\w+)\s*%}(.*?){%\s*endif\s*%}"

        def replace_conditional(match):
            var_name = match.group(1)
            content = match.group(2)
            if variables.get(var_name):
                return content
            return ""

        return re.sub(pattern, replace_conditional, template, flags=re.DOTALL)

    def _process_loops(self, template: str, variables: Dict[str, Any]) -> str:
        """Process loop sections in template"""
        # Pattern: {% for item in items %} content {% endfor %}
        pattern = r"{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%}(.*?){%\s*endfor\s*%}"

        def replace_loop(match):
            item_var = match.group(1)
            list_var = match.group(2)
            content = match.group(3)

            items = variables.get(list_var, [])
            if not isinstance(items, list):
                return ""

            results = []
            for item in items:
                item_content = content
                # Replace item variable
                item_content = re.sub(
                    rf"\{{{{\s*{item_var}\s*\}}}}", str(item), item_content
                )
                results.append(item_content)

            return "\n".join(results)

        return re.sub(pattern, replace_loop, template, flags=re.DOTALL)

    def _format_examples(self) -> str:
        """Format examples section"""
        if not self.examples:
            return ""

        example_text = "\n\nExamples:\n"
        for i, example in enumerate(self.examples, 1):
            example_text += f"\nExample {i}:\n"
            for key, value in example.items():
                example_text += f"{key}: {value}\n"

        return example_text

    def _format_constraints(self) -> str:
        """Format constraints section"""
        if not self.constraints:
            return ""

        constraint_text = "\n\nConstraints:\n"
        for constraint in self.constraints:
            constraint_text += f"- {constraint}\n"

        return constraint_text

    def validate(self, **kwargs) -> bool:
        """Validate that all required variables are provided"""
        return all(var in kwargs for var in self.variables)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return {
            "name": self.name,
            "template": self.template,
            "type": self.type.value,
            "variables": self.variables,
            "description": self.description,
            "examples": self.examples,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary"""
        return cls(
            name=data["name"],
            template=data["template"],
            type=TemplateType(data["type"]),
            variables=data["variables"],
            description=data.get("description", ""),
            examples=data.get("examples", []),
            constraints=data.get("constraints", []),
            metadata=data.get("metadata", {}),
        )


class PromptLibrary:
    """Library of optimized prompt templates"""

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize with default optimized templates"""

        # Chain-of-Thought Reasoning
        self.add_template(
            PromptTemplate(
                name="chain_of_thought",
                template="""Let's solve this step by step.

Problem: {{problem}}

{% if context %}
Context: {{context}}
{% endif %}

I'll break this down into logical steps:

1. First, let me understand what we're trying to solve...
2. Next, I'll identify the key components...
3. Then, I'll work through the solution...
4. Finally, I'll verify and conclude...

Step-by-step solution:""",
                type=TemplateType.REASONING,
                variables=["problem"],
                description="Standard chain-of-thought reasoning template",
                examples=[
                    {
                        "problem": "Calculate 15% of 80",
                        "solution": "15% of 80 = 0.15 × 80 = 12",
                    }
                ],
            )
        )

        # Self-Consistency Template
        self.add_template(
            PromptTemplate(
                name="self_consistency",
                template="""I'll solve this problem using multiple approaches to ensure accuracy.

Problem: {{problem}}

Approach 1: {{approach1_name}}
{{approach1_solution}}

Approach 2: {{approach2_name}}
{{approach2_solution}}

{% if approach3_name %}
Approach 3: {{approach3_name}}
{{approach3_solution}}
{% endif %}

Comparing all approaches, the most consistent answer is:""",
                type=TemplateType.REASONING,
                variables=[
                    "problem",
                    "approach1_name",
                    "approach1_solution",
                    "approach2_name",
                    "approach2_solution",
                ],
                description="Self-consistency reasoning with multiple approaches",
            )
        )

        # Tree of Thoughts Template
        self.add_template(
            PromptTemplate(
                name="tree_of_thoughts",
                template="""Let's explore different solution paths for this problem.

Problem: {{problem}}

I'll consider multiple branches of reasoning:

Branch A: {{branch_a_description}}
- Pros: {{branch_a_pros}}
- Cons: {{branch_a_cons}}
- Next steps: {{branch_a_next}}

Branch B: {{branch_b_description}}
- Pros: {{branch_b_pros}}
- Cons: {{branch_b_cons}}
- Next steps: {{branch_b_next}}

{% if branch_c_description %}
Branch C: {{branch_c_description}}
- Pros: {{branch_c_pros}}
- Cons: {{branch_c_cons}}
- Next steps: {{branch_c_next}}
{% endif %}

Evaluating all branches, the most promising path is:""",
                type=TemplateType.REASONING,
                variables=[
                    "problem",
                    "branch_a_description",
                    "branch_a_pros",
                    "branch_a_cons",
                    "branch_a_next",
                    "branch_b_description",
                    "branch_b_pros",
                    "branch_b_cons",
                    "branch_b_next",
                ],
                description="Tree of thoughts exploration template",
            )
        )

        # Reflexion Template
        self.add_template(
            PromptTemplate(
                name="reflexion",
                template="""Let me solve this problem and then reflect on my solution.

Problem: {{problem}}

Initial Solution:
{{initial_solution}}

Reflection:
- What assumptions did I make?
- Are there any errors or oversights?
- Could the solution be improved?
- What alternative approaches exist?

Refined Solution:
Based on my reflection, here's an improved solution:""",
                type=TemplateType.REASONING,
                variables=["problem", "initial_solution"],
                description="Reflexion-based problem solving with self-critique",
            )
        )

        # Code Generation Template
        self.add_template(
            PromptTemplate(
                name="code_generation",
                template="""Task: {{task}}
Language: {{language}}

{% if requirements %}
Requirements:
{% for req in requirements %}
- {{req}}
{% endfor %}
{% endif %}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{constraint}}
{% endfor %}
{% endif %}

Here's my implementation:

```{{language}}
# Implementation
{{code}}
```

{% if include_tests %}
Test cases:
```{{language}}
{{tests}}
```
{% endif %}""",
                type=TemplateType.CODE_GENERATION,
                variables=["task", "language"],
                description="Template for generating code with requirements and tests",
            )
        )

        # Analysis Template
        self.add_template(
            PromptTemplate(
                name="comprehensive_analysis",
                template="""I'll provide a comprehensive analysis of {{subject}}.

{% if data %}
Data provided:
{{data}}
{% endif %}

Analysis Framework:
1. Overview and Context
2. Key Components
3. Strengths and Opportunities
4. Challenges and Risks
5. Recommendations

Detailed Analysis:

1. Overview and Context:
{{overview}}

2. Key Components:
{% for component in components %}
- {{component}}
{% endfor %}

3. Strengths and Opportunities:
{{strengths}}

4. Challenges and Risks:
{{challenges}}

5. Recommendations:
{{recommendations}}

Summary:
{{summary}}""",
                type=TemplateType.ANALYSIS,
                variables=["subject"],
                description="Comprehensive analysis template with structured framework",
            )
        )

        # Few-Shot Learning Template
        self.add_template(
            PromptTemplate(
                name="few_shot_learning",
                template="""I'll learn from these examples to solve the problem.

{% for example in examples %}
Example {{loop.index}}:
Input: {{example.input}}
Output: {{example.output}}
{% if example.explanation %}
Explanation: {{example.explanation}}
{% endif %}

{% endfor %}

Now, let me solve the new problem:
Input: {{input}}

Following the pattern from the examples:
Output:""",
                type=TemplateType.REASONING,
                variables=["input"],
                description="Few-shot learning template with examples",
            )
        )

        # Role-Based Template
        self.add_template(
            PromptTemplate(
                name="role_based",
                template="""As {{role}} with expertise in {{expertise}}, I'll address your request.

{% if background %}
My Background:
{{background}}
{% endif %}

Request: {{request}}

Professional Response:
Drawing from my experience in {{expertise}}, here's my analysis:

{{response}}

{% if recommendations %}
Recommendations:
{% for rec in recommendations %}
{{loop.index}}. {{rec}}
{% endfor %}
{% endif %}""",
                type=TemplateType.CONVERSATIONAL,
                variables=["role", "expertise", "request"],
                description="Role-based response template",
            )
        )

        # Structured Extraction Template
        self.add_template(
            PromptTemplate(
                name="structured_extraction",
                template="""Extract structured information from the following text.

Text: {{text}}

Extract the following fields:
{% for field in fields %}
- {{field.name}}: {{field.description}}
{% endfor %}

Extracted Information:
```json
{
{% for field in fields %}
  "{{field.name}}": {{field.value}}{% if not loop.last %},{% endif %}
{% endfor %}
}
```""",
                type=TemplateType.EXTRACTION,
                variables=["text"],
                description="Template for extracting structured data from text",
            )
        )

        # Creative Writing Template
        self.add_template(
            PromptTemplate(
                name="creative_writing",
                template="""Genre: {{genre}}
Style: {{style}}
Theme: {{theme}}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{constraint}}
{% endfor %}
{% endif %}

Title: {{title}}

{{content}}

{% if include_metadata %}
---
Word Count: {{word_count}}
Tone: {{tone}}
Target Audience: {{audience}}
{% endif %}""",
                type=TemplateType.CREATIVE,
                variables=["genre", "style", "theme"],
                description="Creative writing template with style guidelines",
            )
        )

    def add_template(self, template: PromptTemplate):
        """Add a template to the library"""
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name"""
        return self.templates.get(name)

    def get_templates_by_type(
        self, template_type: TemplateType
    ) -> List[PromptTemplate]:
        """Get all templates of a specific type"""
        return [t for t in self.templates.values() if t.type == template_type]

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

    def search_templates(self, query: str) -> List[PromptTemplate]:
        """Search templates by name or description"""
        query_lower = query.lower()
        results = []

        for template in self.templates.values():
            if (
                query_lower in template.name.lower()
                or query_lower in template.description.lower()
            ):
                results.append(template)

        return results

    def create_custom_template(
        self, name: str, base_template: str, template_text: str, **kwargs
    ) -> PromptTemplate:
        """Create a custom template based on an existing one"""
        base = self.get_template(base_template)
        if not base:
            raise ValueError(f"Base template '{base_template}' not found")

        # Extract variables from template text
        variables = re.findall(r"\{\{\s*(\w+)\s*\}\}", template_text)

        custom_template = PromptTemplate(
            name=name,
            template=template_text,
            type=base.type,
            variables=list(set(variables)),
            description=kwargs.get(
                "description", f"Custom template based on {base_template}"
            ),
            examples=kwargs.get("examples", base.examples),
            constraints=kwargs.get("constraints", base.constraints),
            metadata={**base.metadata, **kwargs.get("metadata", {})},
        )

        self.add_template(custom_template)
        return custom_template

    def export_library(self) -> Dict[str, Any]:
        """Export library as JSON-serializable dictionary"""
        return {
            "version": "1.0",
            "templates": {
                name: template.to_dict() for name, template in self.templates.items()
            },
        }

    def import_library(self, data: Dict[str, Any]):
        """Import templates from dictionary"""
        for name, template_data in data.get("templates", {}).items():
            template = PromptTemplate.from_dict(template_data)
            self.add_template(template)

    def optimize_template(
        self,
        template_name: str,
        test_cases: List[Dict[str, Any]],
        metric: Callable[[str, str], float],
    ) -> PromptTemplate:
        """Optimize a template based on test cases and metric"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        # Simple optimization - can be extended with more sophisticated methods
        best_score = 0.0
        best_template = template

        # Try variations
        variations = self._generate_template_variations(template)

        for variation in variations:
            total_score = 0.0
            for test_case in test_cases:
                try:
                    output = variation.format(**test_case["input"])
                    score = metric(output, test_case["expected"])
                    total_score += score
                except Exception:
                    continue

            avg_score = total_score / len(test_cases) if test_cases else 0
            if avg_score > best_score:
                best_score = avg_score
                best_template = variation

        return best_template

    def _generate_template_variations(
        self, template: PromptTemplate
    ) -> List[PromptTemplate]:
        """Generate variations of a template for optimization"""
        variations = [template]

        # Add instruction variations
        instruction_prefixes = [
            "Let's think step by step.",
            "I'll analyze this carefully.",
            "Breaking this down systematically:",
        ]

        for prefix in instruction_prefixes:
            variation = PromptTemplate(
                name=f"{template.name}_var",
                template=f"{prefix}\n\n{template.template}",
                type=template.type,
                variables=template.variables,
                description=template.description,
                examples=template.examples,
                constraints=template.constraints,
                metadata=template.metadata,
            )
            variations.append(variation)

        return variations


# Utility functions
def create_template_from_examples(
    name: str,
    examples: List[Dict[str, str]],
    template_type: TemplateType = TemplateType.REASONING,
) -> PromptTemplate:
    """Create a template from examples"""
    if not examples:
        raise ValueError("At least one example is required")

    # Extract common pattern
    first_example = examples[0]
    variables = list(first_example.keys())

    # Build template
    template_text = "Given the following:\n\n"
    for var in variables:
        if var != "output":
            template_text += f"{var}: {{{{{var}}}}}\n"

    template_text += "\nBased on the examples:\n\n"

    return PromptTemplate(
        name=name,
        template=template_text,
        type=template_type,
        variables=[v for v in variables if v != "output"],
        examples=examples,
        description=f"Template created from {len(examples)} examples",
    )
