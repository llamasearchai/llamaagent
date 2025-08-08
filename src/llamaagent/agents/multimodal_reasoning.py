"""
Advanced multimodal reasoning agent for processing text, images, and structured data.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from ..llm.factory import LLMFactory
from ..llm.messages import LLMMessage, LLMResponse
from ..types import TaskInput, TaskOutput, TaskResult, TaskStatus
from .base import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

# Check for vision capabilities
try:
    import PIL.Image  # noqa: F401

    vision_available = True
except ImportError:
    vision_available = False


class ReasoningMode(Enum):
    """Different reasoning modes for multimodal processing."""

    VISUAL_FIRST = "visual_first"
    TEXT_FIRST = "text_first"
    PARALLEL = "parallel"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REFLECTION = "reflection"


@dataclass
class ModalityInput:
    """Input data for different modalities."""

    text: Optional[str] = None
    images: Optional[List[Any]] = None
    structured_data: Optional[Dict[str, Any]] = None


class AdvancedMultimodalAgent(BaseAgent):
    """Advanced Multimodal Reasoning Agent with cutting-edge capabilities."""

    def __init__(
        self,
        name: str = "MultimodalReasoningAgent",
        reasoning_mode: ReasoningMode = ReasoningMode.PARALLEL,
        enable_reflection: bool = True,
        config: Optional[AgentConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AgentConfig(
                name=name, llm_provider=self._create_default_provider(), **kwargs
            )

        super().__init__(config=config)
        self.reasoning_mode = reasoning_mode
        self.enable_reflection = enable_reflection
        self.vision_model = "gpt-4o"
        self.reasoning_model = "gpt-4o"

    def _create_default_provider(self) -> Any:
        """Create a concrete provider with multimodal capabilities."""
        factory = LLMFactory()
        return factory.create_provider("openai")

    async def execute_task(self, task_input: TaskInput) -> TaskOutput:
        """Execute multimodal reasoning task."""
        try:
            logger.info(f"Executing multimodal task: {task_input.id}")

            # Parse input into modalities
            input_data = self._parse_input(task_input)

            # Execute reasoning based on mode
            result = await self._execute_reasoning(input_data)

            # Apply reflection if enabled
            if self.enable_reflection:
                result = await self._apply_reflection(result, input_data)

            # Create task result
            task_result = TaskResult(success=True, data=result.content)

            return TaskOutput(
                task_id=task_input.id, status=TaskStatus.COMPLETED, result=task_result
            )

        except Exception as e:
            logger.error(f"Multimodal reasoning error: {e}")
            return TaskOutput(
                task_id=task_input.id,
                status=TaskStatus.FAILED,
                result=TaskResult(success=False, error=str(e)),
            )

    def _parse_input(self, task_input: TaskInput) -> ModalityInput:
        """Parse task input into different modalities."""
        text_input = None
        images = None
        structured_data = None

        if hasattr(task_input, 'data') and isinstance(task_input.data, dict):
            data = cast(Dict[str, Any], task_input.data)
            text_input = data.get("text")
            images = data.get("images")
            structured_data = data.get("structured_data")

        if text_input is None:
            text_input = task_input.prompt if hasattr(task_input, 'prompt') else None

        return ModalityInput(
            text=text_input, images=images, structured_data=structured_data
        )

    async def _execute_reasoning(self, input_data: ModalityInput) -> LLMResponse:
        """Execute reasoning based on selected mode."""
        if self.reasoning_mode == ReasoningMode.VISUAL_FIRST:
            return await self._visual_first_reasoning(input_data)
        elif self.reasoning_mode == ReasoningMode.TEXT_FIRST:
            return await self._text_first_reasoning(input_data)
        elif self.reasoning_mode == ReasoningMode.CHAIN_OF_THOUGHT:
            return await self._chain_of_thought_reasoning(input_data)
        elif self.reasoning_mode == ReasoningMode.REFLECTION:
            return await self._reflection_reasoning(input_data)
        else:
            return await self._parallel_reasoning(input_data)

    async def _visual_first_reasoning(self, input_data: ModalityInput) -> LLMResponse:
        """Process visual information first, then integrate with text."""
        results = []

        # Analyze images first
        if input_data.images and vision_available:
            visual_result = await self._analyze_images(input_data.images)
            results.append(visual_result)

        # Then analyze text with visual context
        if input_data.text:
            text_prompt = (
                f"Based on the visual analysis, now analyze: {input_data.text}"
            )
            text_result = await self._analyze_text(text_prompt)
            results.append(text_result)

        # Integrate results
        if len(results) > 1:
            return await self._integrate_results(results)
        elif results:
            return results[0]
        else:
            return LLMResponse(content="No valid input provided")

    async def _text_first_reasoning(self, input_data: ModalityInput) -> LLMResponse:
        """Process text information first, then integrate with visuals."""
        results = []

        # Analyze text first
        if input_data.text:
            text_result = await self._analyze_text(input_data.text)
            results.append(text_result)

        # Then analyze images with text context
        if input_data.images and vision_available:
            visual_prompt = (
                "Based on the text analysis, now analyze the provided images"
            )
            visual_result = await self._analyze_images(input_data.images)
            results.append(visual_result)

        # Integrate results
        if len(results) > 1:
            return await self._integrate_results(results)
        elif results:
            return results[0]
        else:
            return LLMResponse(content="No valid input provided")

    async def _parallel_reasoning(self, input_data: ModalityInput) -> LLMResponse:
        """Process multiple modalities in parallel."""
        tasks: List[Any] = []

        if input_data.text:
            tasks.append(self._analyze_text(input_data.text))

        if input_data.images and vision_available:
            tasks.append(self._analyze_images(input_data.images))

        if input_data.structured_data:
            tasks.append(self._analyze_structured_data(input_data.structured_data))

        if not tasks:
            return LLMResponse(content="No valid input provided")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results = [r for r in results if isinstance(r, LLMResponse)]

        if len(successful_results) > 1:
            return await self._integrate_results(successful_results)
        elif successful_results:
            return successful_results[0]
        else:
            return LLMResponse(
                content="Analysis failed: Unable to process the provided input"
            )

    async def _chain_of_thought_reasoning(
        self, input_data: ModalityInput
    ) -> LLMResponse:
        """Execute chain-of-thought reasoning."""
        cot_prompt = self._build_cot_prompt(input_data)
        return await self._analyze_text(cot_prompt)

    async def _reflection_reasoning(self, input_data: ModalityInput) -> LLMResponse:
        """Execute reasoning with built-in reflection."""
        # First pass
        initial_result = await self._parallel_reasoning(input_data)

        # Reflection
        reflected_result = await self._apply_reflection(initial_result, input_data)

        return reflected_result

    async def _analyze_text(self, text: str) -> LLMResponse:
        """Analyze text using LLM."""
        message = LLMMessage(
            role="user", content=f"Analyze and provide insights: {text}"
        )
        if self.config.llm_provider is not None:
            return await self.config.llm_provider.complete([message])
        else:
            return LLMResponse(content="No LLM provider available")

    async def _analyze_images(self, images: List[Any]) -> LLMResponse:
        """Analyze images using vision capabilities."""
        if not images or not vision_available:
            return LLMResponse(
                content="No images available for analysis or vision not supported"
            )

        # For now, simulate image analysis (would integrate with actual vision API)
        image_description = f"Processing {len(images)} image(s) for visual analysis"

        message = LLMMessage(
            role="user", content=f"Analyze these images: {image_description}"
        )
        if self.config.llm_provider is not None:
            return await self.config.llm_provider.complete([message])
        else:
            return LLMResponse(content="No vision provider available")

    async def _analyze_structured_data(self, data: Dict[str, Any]) -> LLMResponse:
        """Analyze structured data."""
        data_summary = f"Structured data with {len(data)} fields: {list(data.keys())}"

        message = LLMMessage(
            role="user", content=f"Analyze this structured data: {data_summary}"
        )
        if self.config.llm_provider is not None:
            return await self.config.llm_provider.complete([message])
        else:
            return LLMResponse(
                content="No provider available for structured data analysis"
            )

    async def _integrate_results(self, results: List[LLMResponse]) -> LLMResponse:
        """Integrate multiple reasoning results."""
        combined_content = "\n\n".join([r.content for r in results])
        integration_prompt = f"""
Please integrate these analysis results into a coherent response:

{combined_content}

Provide a unified, comprehensive answer:
"""

        return await self._analyze_text(integration_prompt)

    async def _apply_reflection(
        self, result: LLMResponse, input_data: ModalityInput
    ) -> LLMResponse:
        """Apply reflection to improve reasoning quality."""
        reflection_prompt = f"""
Original Result: {result.content}

Please reflect on this result and consider:
1. Are there any gaps in the reasoning?
2. Could alternative perspectives strengthen the analysis?
3. What additional insights could be valuable?

Provide an improved, more comprehensive response:
"""

        reflected_result = await self._analyze_text(reflection_prompt)
        return reflected_result

    def _build_cot_prompt(self, input_data: ModalityInput) -> str:
        """Build chain-of-thought prompt."""
        prompt_parts = ["Let's think step by step about this problem:"]

        if input_data.text:
            prompt_parts.append(f"Text Input: {input_data.text}")

        if input_data.images:
            prompt_parts.append(f"Images: {len(input_data.images)} image(s) to analyze")

        if input_data.structured_data:
            prompt_parts.append(f"Structured Data: {input_data.structured_data}")

        prompt_parts.extend(
            [
                "",
                "Step 1: Understanding the problem",
                "Step 2: Analyzing each component",
                "Step 3: Identifying relationships",
                "Step 4: Synthesizing insights",
                "Step 5: Drawing conclusions",
            ]
        )

        return "\n".join(prompt_parts)


# Convenience function for creating multimodal agents
def create_multimodal_agent(
    name: str = "MultimodalAgent",
    reasoning_mode: ReasoningMode = ReasoningMode.PARALLEL,
    enable_reflection: bool = True,
    **kwargs: Any,
) -> AdvancedMultimodalAgent:
    """Create a multimodal reasoning agent."""
    return AdvancedMultimodalAgent(
        name=name,
        reasoning_mode=reasoning_mode,
        enable_reflection=enable_reflection,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        agent = create_multimodal_agent(reasoning_mode=ReasoningMode.PARALLEL)

        task = TaskInput(
            id="test_multimodal",
            prompt="Analyze this multimodal input",
            data={
                "text": "Explain the relationship between AI and machine learning",
                "structured_data": {"topic": "AI", "complexity": "intermediate"},
            },
        )

        result = await agent.execute_task(task)
        print(f"Result: {result}")

    asyncio.run(main())
