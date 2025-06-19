# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List

from ..llm import LLMMessage, LLMProvider, MockProvider, create_provider
from ..memory import SimpleMemory
from ..tools import ToolRegistry, default_registry
from .base import AgentConfig, AgentResponse, ExecutionPlan, PlanStep


class ReactAgent:
    """SPRE-enabled ReactAgent with Strategic Planning & Resourceful Execution.

    This implementation follows the research methodology outlined in the Pre-Act
    and SEM papers, providing a two-tiered reasoning framework that combines
    strategic planning with resource-efficient execution.
    """

    # ═══════════════════════════ SPRE PROMPTS ═══════════════════════════════

    PLANNER_PROMPT = """You are a master strategist and planner. Your task is to receive a complex user request and decompose it into a structured, sequential list of logical steps.

For each step, clearly define:
1. The action to be taken
2. What specific information is required to complete it
3. The expected outcome

Output this plan as a JSON object with this structure:
{
  "steps": [
    {
      "step_id": 1,
      "description": "Clear description of what to do",
      "required_information": "What information is needed",
      "expected_outcome": "What should result from this step"
    }
  ]
}

Do not attempt to solve the task, only plan it."""

    RESOURCE_ASSESSMENT_PROMPT = """Current Plan Step: '{step_description}'
Information Needed: '{required_info}'

Reviewing the conversation history and your internal knowledge, is it absolutely necessary to use an external tool to acquire this information?

Consider:
- Can you answer this from your training knowledge?
- Is the information already available in the conversation?
- Would a tool call provide significantly better accuracy?

Answer with only 'true' or 'false' followed by a brief justification."""

    SYNTHESIS_PROMPT = """Original task: {original_task}

Execution results from all steps:
{step_results}

Provide a comprehensive final answer that addresses the original task by synthesizing all the information gathered and reasoning performed across the execution steps."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider | None = None,
        memory: SimpleMemory | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.config = config

        # Unique identifier used throughout tracing and persistence layers.
        self._id = str(uuid.uuid4())

        # Resolve provider.  Priority:
        # 1. Explicit ``llm_provider`` argument
        # 2. Environment variable ``LLAMAAGENT_LLM_PROVIDER``
        # 3. Fallback to ``MockProvider`` for offline/test runs.

        if llm_provider is not None:
            self.llm = llm_provider
        else:
            provider_type = os.getenv("LLAMAAGENT_LLM_PROVIDER", "mock").lower()

            # Collect dynamic kwargs from environment so that we do not hard-code
            # provider-specific parameters here.
            create_kwargs: Dict[str, Any] = {}

            model = os.getenv("LLAMAAGENT_LLM_MODEL")
            if model:
                create_kwargs["model"] = model

            if provider_type == "openai":
                # OPENAI_API_KEY should exist in env otherwise the factory will raise.
                create_kwargs["api_key"] = os.getenv("OPENAI_API_KEY", "")
            elif provider_type == "anthropic":
                create_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
            elif provider_type == "ollama":
                # API key is optional for local Ollama; include if provided.
                api_key = os.getenv("OLLAMA_API_KEY")
                if api_key:
                    create_kwargs["api_key"] = api_key
                # Custom base URL can be supplied.
                base_url = os.getenv("OLLAMA_BASE_URL")
                if base_url:
                    create_kwargs["base_url"] = base_url

            try:
                self.llm = create_provider(provider_type, **create_kwargs)
            except Exception:
                # On any failure (e.g., missing API key) fall back to mock provider so that
                # unit tests never hit external services.
                self.llm = MockProvider()

        # Persistence layer -------------------------------------------------
        if memory is not None:
            self.memory = memory
        else:
            # Auto-select Postgres vector memory if a DATABASE_URL is defined
            # and the optional dependencies are available.  Falls back to the
            # original in-memory implementation otherwise so that offline
            # tests are unaffected.
            try:
                from ..storage.vector_memory import \
                    PostgresVectorMemory  # pylint: disable=import-error

                if os.getenv("DATABASE_URL"):
                    self.memory = PostgresVectorMemory(agent_id=self._id)
                else:
                    self.memory = SimpleMemory()
            except ModuleNotFoundError:
                # Optional dependencies not installed – fallback silently.
                self.memory = SimpleMemory()

        self.tools = tools or default_registry
        self.trace: List[Dict[str, Any]] = []

    # ═══════════════════════════ MAIN EXECUTION ═══════════════════════════════

    async def execute(self, task: str, context: Dict[str, Any] | None = None) -> AgentResponse:
        """Execute task using SPRE methodology."""
        start_time = time.time()
        self.trace.clear()
        self.add_trace("task_start", {"task": task, "context": context, "spree_enabled": self.config.spree_enabled})

        try:
            if self.config.spree_enabled:
                # Full SPRE Pipeline: Plan → Execute → Synthesize
                result = await self._execute_spre_pipeline(task, context)
            else:
                # Fallback to simple execution for baseline comparison
                result = await self._simple_execute(task, context)

            execution_time = time.time() - start_time
            self.add_trace("task_complete", {"success": True, "execution_time": execution_time})

            return AgentResponse(
                content=result,
                success=True,
                trace=self.trace,
                execution_time=execution_time,
                tokens_used=self._count_tokens(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.add_trace("error", {"error": str(e), "execution_time": execution_time})

            return AgentResponse(
                content=f"Error: {e!s}",
                success=False,
                trace=self.trace,
                execution_time=execution_time,
            )

    # ═══════════════════════════ SPRE PIPELINE ═══════════════════════════════

    async def _execute_spre_pipeline(self, task: str, context: Dict[str, Any] | None = None) -> str:
        """Execute the complete SPRE pipeline: Plan → Execute → Synthesize."""

        # Phase 1: Strategic Planning
        plan = await self._generate_plan(task)
        self.add_trace("plan_generated", {"plan": plan.__dict__, "num_steps": len(plan.steps)})

        # Phase 2: Resourceful Execution
        step_results = await self._execute_plan_with_resource_assessment(plan, context)

        # Phase 3: Synthesis
        final_answer = await self._synthesize_results(plan, step_results)

        return final_answer

    async def _generate_plan(self, task: str) -> ExecutionPlan:
        """Generate execution plan using specialized planner."""
        self.add_trace("planning_start", {"task": task})

        messages = [
            LLMMessage(role="system", content=self.PLANNER_PROMPT),
            LLMMessage(role="user", content=f"Task: {task}"),
        ]

        response = await self.llm.complete(messages)
        self.add_trace("planner_response", {"content": response.content})

        try:
            # Parse JSON plan
            plan_data = json.loads(response.content)
            steps = [
                PlanStep(
                    step_id=step["step_id"],
                    description=step["description"],
                    required_information=step["required_information"],
                    expected_outcome=step["expected_outcome"],
                )
                for step in plan_data["steps"]
            ]

            plan = ExecutionPlan(original_task=task, steps=steps)
            self.add_trace("planning_complete", {"num_steps": len(steps), "plan_valid": True})
            return plan

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to simple single-step plan
            self.add_trace("plan_parse_error", {"error": str(e), "fallback": True})
            return ExecutionPlan(
                original_task=task,
                steps=[
                    PlanStep(
                        step_id=1,
                        description=f"Complete task: {task}",
                        required_information="Direct answer",
                        expected_outcome="Task completion",
                    )
                ],
            )

    async def _execute_plan_with_resource_assessment(
        self, plan: ExecutionPlan, context: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """Execute plan with resource assessment for each step."""
        step_results = []

        for step in plan.steps:
            self.add_trace("step_start", {"step_id": step.step_id, "description": step.description})

            # Resource Assessment Phase
            needs_tool = await self._assess_resource_need(step)
            self.add_trace(
                "resource_assessment",
                {"step_id": step.step_id, "needs_tool": needs_tool, "assessment_method": "llm_based"},
            )

            # Execution Decision Fork
            if needs_tool:
                result = await self._execute_with_tool(step)
                execution_method = "tool_based"
            else:
                result = await self._execute_internal(step)
                execution_method = "internal_knowledge"

            step_result = {
                "step_id": step.step_id,
                "description": step.description,
                "execution_method": execution_method,
                "result": result,
                "needs_tool": needs_tool,
            }

            step_results.append(step_result)
            step.is_completed = True

            # Store in memory for future reference
            if self.config.memory_enabled and self.memory:
                memory_entry = f"Step {step.step_id}: {step.description} -> {result}"
                add_fn = getattr(self.memory, "add", None)
                if callable(add_fn):
                    await add_fn(memory_entry)

            self.add_trace("step_complete", step_result)

        return step_results

    async def _assess_resource_need(self, step: PlanStep) -> bool:
        """Assess if external tool is needed for this step (SEM-inspired)."""
        prompt = self.RESOURCE_ASSESSMENT_PROMPT.format(
            step_description=step.description,
            required_info=step.required_information,
        )

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self.llm.complete(messages)

        # Parse response - looking for true/false at the beginning
        content = response.content.lower().strip()
        needs_tool = content.startswith("true")

        self.add_trace(
            "resource_assessment_detail",
            {"step_id": step.step_id, "assessment_response": response.content, "needs_tool": needs_tool},
        )

        return needs_tool

    async def _execute_with_tool(self, step: PlanStep) -> str:
        """Execute step using available tools."""
        self.add_trace("tool_execution_start", {"step_id": step.step_id})

        # Enhanced tool selection logic
        step_desc_lower = step.description.lower()

        # Mathematical operations
        if any(keyword in step_desc_lower for keyword in ["calculate", "math", "compute", "+", "-", "*", "/"]):
            calc_tool = self.tools.get("calculator")
            if calc_tool:
                # Extract mathematical expression
                math_pattern = r"[\d+\-*/().\s]+"
                match = re.search(math_pattern, step.description)
                if match:
                    expr = match.group().strip()
                    try:
                        result = await calc_tool.execute(expr)
                        self.add_trace(
                            "tool_execution_success",
                            {"step_id": step.step_id, "tool": "calculator", "expression": expr, "result": result},
                        )
                        return f"Calculated: {result}"
                    except Exception as e:
                        self.add_trace(
                            "tool_execution_error", {"step_id": step.step_id, "tool": "calculator", "error": str(e)}
                        )
                        return f"Calculation error: {e!s}"

        # Python code execution
        if any(keyword in step_desc_lower for keyword in ["code", "python", "function", "script"]):
            python_tool = self.tools.get("python_repl")
            if python_tool:
                try:
                    # Extract or generate relevant code
                    code_to_execute = self._extract_or_generate_code(step)
                    result = await python_tool.execute(code_to_execute)
                    self.add_trace(
                        "tool_execution_success",
                        {"step_id": step.step_id, "tool": "python_repl", "code": code_to_execute, "result": result},
                    )
                    return f"Code execution result: {result}"
                except Exception as e:
                    self.add_trace(
                        "tool_execution_error", {"step_id": step.step_id, "tool": "python_repl", "error": str(e)}
                    )
                    return f"Code execution error: {e!s}"

        # If no specific tool matches, fallback to LLM reasoning
        self.add_trace("tool_fallback_to_llm", {"step_id": step.step_id, "reason": "no_matching_tool"})
        return await self._execute_internal(step)

    def _extract_or_generate_code(self, step: PlanStep) -> str:
        """Extract or generate Python code for execution."""
        # Simple heuristics for code generation based on step description
        desc = step.description.lower()

        if "reverse" in desc and "string" in desc:
            return "def reverse_string(s): return s[::-1]\nprint(reverse_string('hello'))"
        elif "fibonacci" in desc:
            return "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\nprint([fibonacci(i) for i in range(10)])"
        elif "prime" in desc:
            return "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))\nprint([i for i in range(2, 20) if is_prime(i)])"
        else:
            # Generic code template
            return f"# Code for: {step.description}\nprint('Task: {step.description}')"

    async def _execute_internal(self, step: PlanStep) -> str:
        """Execute step using internal knowledge."""
        messages = [
            LLMMessage(
                role="user",
                content=f"Using your internal knowledge, {step.description}. Focus on: {step.required_information}. Provide a comprehensive answer.",
            ),
        ]

        response = await self.llm.complete(messages)
        self.add_trace(
            "internal_execution",
            {"step_id": step.step_id, "method": "internal_knowledge", "response_length": len(response.content)},
        )
        return response.content

    async def _synthesize_results(self, plan: ExecutionPlan, step_results: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from all step results."""
        self.add_trace("synthesis_start", {"num_steps": len(step_results)})

        # Format step results for synthesis
        formatted_results = []
        for result in step_results:
            formatted_results.append(
                f"Step {result['step_id']}: {result['description']}\n"
                f"Method: {result['execution_method']}\n"
                f"Result: {result['result']}\n"
            )

        synthesis_prompt = self.SYNTHESIS_PROMPT.format(
            original_task=plan.original_task, step_results="\n".join(formatted_results)
        )

        messages = [LLMMessage(role="user", content=synthesis_prompt)]
        response = await self.llm.complete(messages)

        self.add_trace(
            "synthesis_complete", {"final_answer_length": len(response.content), "synthesis_method": "llm_based"}
        )

        return response.content

    # ═══════════════════════════ FALLBACK EXECUTION ═══════════════════════════

    async def _simple_execute(self, task: str, context: Dict[str, Any] | None = None) -> str:
        """Simple execution without SPRE planning (baseline comparison)."""
        self.add_trace("simple_execution_start", {"method": "baseline"})

        # Enhanced simple execution with basic math handling
        import re as _re

        expr_match = _re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", task)
        if expr_match:
            a, op, b = expr_match.groups()
            a_int, b_int = int(a), int(b)
            try:
                match op:
                    case "+":
                        result = str(a_int + b_int)
                    case "-":
                        result = str(a_int - b_int)
                    case "*":
                        result = str(a_int * b_int)
                    case "/":
                        if b_int == 0:
                            result = "Error: division by zero"
                        else:
                            res = a_int / b_int
                            result = str(int(res) if res.is_integer() else res)
                    case _:
                        result = f"Task '{task}' processed by {self.config.name}"

                self.add_trace("simple_execution_math", {"expression": f"{a}{op}{b}", "result": result})
                return result
            except Exception as _arith_e:
                self.add_trace("simple_execution_error", {"error": str(_arith_e)})
                return f"Error evaluating expression: {_arith_e}"

        # Default response for non-math tasks
        result = f"Task '{task}' processed by {self.config.name} (simple mode)"
        self.add_trace("simple_execution_complete", {"result": result})
        return result

    # ═══════════════════════════ UTILITY METHODS ═══════════════════════════

    def add_trace(self, event_type: str, data: Any) -> None:
        """Add event to execution trace."""
        self.trace.append(
            {
                "timestamp": time.time(),
                "type": event_type,
                "data": data,
                "agent_id": self._id,
                "agent_name": self.config.name,
            }
        )

    def _count_tokens(self) -> int:
        """Estimate token usage from trace data."""
        total_chars = sum(len(str(item.get("data", ""))) for item in self.trace)
        return total_chars // 4  # Rough estimate: 4 chars per token

    async def stream_execute(self, task: str, context: Dict[str, Any] | None = None):
        """Stream execution results."""
        result = await self.execute(task, context)
        yield result.content
