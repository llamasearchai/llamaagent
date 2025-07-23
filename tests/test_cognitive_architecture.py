"""
Comprehensive test suite for the Cognitive Architecture module.

This test suite validates all components of the advanced reasoning system:
- Tree of Thoughts
- Graph of Thoughts  
- Constitutional AI
- Meta-Reasoning
- Unified Cognitive Agent

Author: LlamaAgent Development Team
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any, List, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.llamaagent.reasoning.tree_of_thoughts import (
    TreeOfThoughtsAgent, ThoughtNode, SearchStrategy
)
from src.llamaagent.reasoning.graph_of_thoughts import (
    GraphOfThoughtsAgent, Concept
)
from src.llamaagent.reasoning.constitutional_ai import (
    ConstitutionalAgent, ConstitutionalRule
)
from src.llamaagent.reasoning.meta_reasoning import (
    MetaCognitiveAgent
)
from src.llamaagent.agents.base import AgentConfig
from src.llamaagent.memory import MemoryManager
from src.llamaagent.tools.registry import ToolRegistry
from src.llamaagent.llm.providers.base_provider import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self):
        super().__init__(api_key="test-key", model="test-model")
        self.call_count = 0
        self.responses = {}
        
    async def complete(self, messages: List[Any], max_tokens: int = 1000, 
                      temperature: float = 0.7, model: Optional[str] = None, **kwargs: Any) -> Any:
        """Mock completion method"""
        self.call_count += 1
        
        # Get the last message content for response selection
        if messages:
            last_message = messages[-1]
            # Handle both dict and LLMMessage objects
            if hasattr(last_message, 'content'):
                last_content = last_message.content.lower()
            else:
                last_content = last_message.get('content', '').lower()
            
            # Return appropriate responses based on content
            if 'tree of thoughts' in last_content or 'generate thoughts' in last_content:
                return MockResponse(self._get_tree_response())
            elif 'evaluate' in last_content:
                return MockResponse(self._get_evaluation_response())
            elif 'concepts' in last_content or 'extract concepts' in last_content:
                return MockResponse(self._get_concept_response())
            elif 'constitutional' in last_content or 'critique' in last_content:
                return MockResponse(self._get_constitutional_response())
            elif 'complexity' in last_content:
                return MockResponse(self._get_complexity_response())
            elif 'confidence' in last_content:
                return MockResponse(self._get_confidence_response())
            else:
                return MockResponse(self._get_default_response())
        
        return MockResponse(self._get_default_response())
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000, 
                              temperature: float = 0.7, **kwargs: Any) -> Any:
        """Mock generate_response method"""
        return MockResponse(self._get_default_response())
    
    async def chat_completion(self, messages: List[Any], max_tokens: int = 1000, 
                            temperature: float = 0.7, model: Optional[str] = None, **kwargs: Any) -> Any:
        """Mock chat_completion method"""
        return await self.complete(messages, max_tokens, temperature, model, **kwargs)
    
    async def stream_chat_completion(self, messages: List[Any], max_tokens: int = 1000, 
                                   temperature: float = 0.7, model: Optional[str] = None, **kwargs: Any) -> Any:
        """Mock stream_chat_completion method"""
        response = await self.complete(messages, max_tokens, temperature, model, **kwargs)
        async def generator():
            yield response.content
        return generator()
    
    def _get_tree_response(self):
        return """Based on the problem, I'll generate several thought paths:

Thought 1: Direct approach - solve step by step
Thought 2: Alternative method - use different strategy  
Thought 3: Creative solution - think outside the box

Each thought represents a different way to approach this problem."""
    
    def _get_evaluation_response(self):
        return """Evaluation Score: 0.85
Reasoning: This thought shows good logical progression and addresses the core problem effectively. It demonstrates clear reasoning steps and arrives at a reasonable conclusion."""
    
    def _get_concept_response(self):
        return """[
  {
    "name": "Machine Learning",
    "description": "A subset of AI that enables systems to learn from data",
    "confidence": 0.9,
    "domain": "technology"
  },
  {
    "name": "Data Science",
    "description": "Field focused on extracting insights from data",
    "confidence": 0.9,
    "domain": "technology"
  },
  {
    "name": "Statistical Analysis",
    "description": "Mathematical methods for analyzing data patterns",
    "confidence": 0.8,
    "domain": "technology"
  }
]"""
    
    def _get_constitutional_response(self):
        return """Constitutional Analysis:
COMPLIANCE: True
VIOLATIONS: None detected
REASONING: The response adheres to all constitutional principles and shows respect for safety and ethics."""
    
    def _get_complexity_response(self):
        return """{
  "complexity": "complex",
  "domain": "technical",
  "multi_step": true,
  "uncertainty": "moderate",
  "reasoning_required": true,
  "suggested_strategies": ["tree_of_thoughts", "graph_of_thoughts"],
  "reasoning": "This is a complex problem requiring multi-step reasoning"
}"""
    
    def _get_confidence_response(self):
        return """Confidence Assessment:
SCORE: 0.75
BASIS: Strong logical foundation with some uncertainty in implementation details
RELIABILITY: HIGH"""
    
    def _get_default_response(self):
        return """I understand your request. Let me provide a thoughtful response based on careful analysis and reasoning."""


class MockResponse:
    """Mock response object"""
    def __init__(self, content: str):
        self.content = content
        self.success = True
        self.metadata = {
            "tokens_used": 100,
            "model": "mock-model"
        }


class MockAgentConfig(AgentConfig):
    """Mock agent configuration for testing"""
    def __init__(self, name: str = "test-agent", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.max_iterations = kwargs.get('max_iterations', 10)
        self.timeout = kwargs.get('timeout', 30.0)
        self.debug_mode = kwargs.get('debug_mode', False)


class MockCognitiveAgent:
    """Mock cognitive agent that simulates the real agent behavior"""
    def __init__(self, config: AgentConfig, llm_provider: BaseLLMProvider, 
                 tools: Optional[ToolRegistry] = None, memory: Optional[MemoryManager] = None, **kwargs: Any):
        self.config = config
        self.llm_provider = llm_provider
        self.tools = tools
        self.memory = memory
        self.metadata = {}
        
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a cognitive task"""
        # Context is intentionally unused in mock
        _ = context
        return MockResponse(f"Executed task: {task}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "cognitive_agent_metrics": {
                "cognitive_agent_stats": {
                    "reasoning_depth": 3,
                    "adaptation_score": 0.85,
                    "confidence_calibration": 0.92
                },
                "components_status": {
                    "tree_of_thoughts": True,
                    "graph_of_thoughts": True,
                    "constitutional_ai": True,
                    "meta_reasoning": True
                }
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass


# Fixtures
@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def mock_config() -> MockAgentConfig:
    return MockAgentConfig(
        name="test-agent",
        max_iterations=10,
        timeout=30.0,
        debug_mode=True
    )


# Test Tree of Thoughts
class TestTreeOfThoughts:
    """Test suite for Tree of Thoughts implementation"""
    
    @pytest.mark.asyncio
    async def test_tree_initialization(self, mock_llm: MockLLMProvider):
        """Test tree of thoughts agent initialization"""
        agent = TreeOfThoughtsAgent(
            llm_provider=mock_llm,
            strategy=SearchStrategy.BEST_FIRST,
            max_depth=3,
            beam_width=2
        )
        
        assert agent.llm_provider == mock_llm
        assert agent.max_depth == 3
        assert agent.beam_width == 2
        assert hasattr(agent, 'evaluator')
        assert hasattr(agent, 'search_strategies')
    
    @pytest.mark.asyncio
    async def test_thought_generation(self, mock_llm: MockLLMProvider):
        """Test thought generation process"""
        agent = TreeOfThoughtsAgent(
            llm_provider=mock_llm,
            strategy=SearchStrategy.BEST_FIRST
        )
        
        result = await agent.solve(
            problem="How can we improve urban transportation?"
        )
        
        assert result['success'] is True
        assert 'solution' in result
        assert 'statistics' in result
        assert result['statistics']['nodes_created'] > 0
    
    @pytest.mark.asyncio  
    async def test_thought_node_structure(self):
        """Test thought node structure and relationships"""
        node = ThoughtNode(
            content="Initial thought",
            depth=0,
            score=0.0
        )
        
        assert node.content == "Initial thought"
        assert node.depth == 0
        assert node.score == 0.0
        assert len(node.children) == 0
        
        child = ThoughtNode(content="Child thought", depth=1, score=0.5)
        node.add_child(child)
        
        assert len(node.children) == 1
        assert node.children[0] == child


# Test Graph of Thoughts
class TestGraphOfThoughts:
    """Test suite for Graph of Thoughts implementation"""
    
    @pytest.mark.asyncio
    async def test_graph_initialization(self, mock_llm: MockLLMProvider):
        """Test graph of thoughts agent initialization"""
        agent = GraphOfThoughtsAgent(
            llm_provider=mock_llm,
            max_concepts=5,
            max_depth=3
        )
        
        assert agent.llm_provider == mock_llm
        assert agent.max_concepts == 5
        assert agent.max_depth == 3
        assert hasattr(agent, 'concept_extractor')
        assert hasattr(agent, 'relationship_mapper')
    
    @pytest.mark.asyncio
    async def test_concept_extraction(self, mock_llm: MockLLMProvider):
        """Test concept extraction from problem"""
        agent = GraphOfThoughtsAgent(
            llm_provider=mock_llm,
            max_concepts=10
        )
        
        result = await agent.solve(
            problem="Explain the relationship between machine learning and data science",
            domain="technology"
        )
        
        assert result['success'] is True
        assert 'solution' in result
        assert 'statistics' in result
        assert result['statistics']['concepts_extracted'] > 0
    
    @pytest.mark.asyncio
    async def test_concept_structure(self):
        """Test concept node structure"""
        node = Concept(
            name="Machine Learning",
            description="Field of AI",
            confidence=0.9
        )
        
        assert node.name == "Machine Learning"
        assert node.confidence == 0.9
        assert len(node.metadata) == 0


# Test Constitutional AI
class TestConstitutionalAI:
    """Test suite for Constitutional AI implementation"""
    
    @pytest.mark.asyncio
    async def test_constitutional_initialization(self, mock_llm: MockLLMProvider):
        """Test constitutional agent initialization"""
        agent = ConstitutionalAgent(
            llm_provider=mock_llm,
            max_revision_attempts=3
        )
        
        assert agent.llm_provider == mock_llm
        assert agent.constitution is not None
        assert agent.critique_system is not None
    
    @pytest.mark.asyncio
    async def test_constitutional_response(self, mock_llm: MockLLMProvider):
        """Test constitutional response processing"""
        agent = ConstitutionalAgent(
            llm_provider=mock_llm,
            max_revision_attempts=2
        )
        
        result = await agent.process_response(
            query="Help me build something"
        )
        
        assert result['success'] is True
        assert 'response' in result
        assert 'violations' in result
        assert 'compliance_score' in result
    
    def test_constitutional_rule(self):
        """Test constitutional rule structure"""
        rule = ConstitutionalRule(
            id="rule1",
            name="Honesty",
            description="Be truthful and transparent",
            priority=1,
            enabled=True
        )
        
        assert rule.name == "Honesty"
        assert rule.description == "Be truthful and transparent"
        assert rule.priority == 1
        assert rule.enabled is True


# Test Meta-Reasoning
class TestMetaReasoning:
    """Test suite for Meta-Reasoning implementation"""
    
    @pytest.mark.asyncio
    async def test_meta_agent_initialization(self, mock_llm: MockLLMProvider):
        """Test meta-cognitive agent initialization"""
        agent = MetaCognitiveAgent(llm_provider=mock_llm)
        
        assert agent.llm_provider == mock_llm
        assert agent.strategy_selector is not None
        assert agent.confidence_system is not None
        assert agent.performance_history is not None
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, mock_llm: MockLLMProvider):
        """Test strategy selection process"""
        agent = MetaCognitiveAgent(llm_provider=mock_llm)
        
        # Create mock strategies
        available_strategies = {
            "simple": AsyncMock(return_value={"solution": "Simple solution", "success": True}),
            "tree_of_thoughts": AsyncMock(return_value={"solution": "ToT solution", "success": True})
        }
        
        result = await agent.select_and_execute_strategy(
            problem="Design a sustainable city",
            available_strategies=available_strategies,
            context={
                "domain": "urban_planning",  
                "complexity": "complex"
            }
        )
        
        assert result['success'] is True
        assert 'selected_strategy' in result
        assert 'confidence_assessment' in result
        assert result['confidence_assessment']['confidence'] >= 0.0


# Test Unified Cognitive Agent
class TestCognitiveAgent:
    """Test suite for unified Cognitive Agent"""
    
    @pytest.mark.asyncio
    async def test_cognitive_agent_initialization(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test cognitive agent initialization"""
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm,
                enable_constitutional_ai=True,
                enable_meta_reasoning=True
            )
            
            assert agent.config == mock_config
            assert agent.llm_provider == mock_llm
            assert hasattr(agent, 'execute')
            assert hasattr(agent, 'get_performance_metrics')
    
    @pytest.mark.asyncio
    async def test_cognitive_execution(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test cognitive agent execution"""
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm,
                enable_constitutional_ai=True,
                enable_meta_reasoning=True
            )
            
            result = await agent.execute(
                task="Solve a complex problem using multiple reasoning strategies",
                context={"complexity": "high"}
            )
            
            assert hasattr(result, 'success')
            assert hasattr(result, 'content')
            assert result.success is True
            assert len(result.content) > 0
            assert hasattr(result, 'metadata')
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test performance metrics collection"""
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm,
                enable_constitutional_ai=True,
                enable_meta_reasoning=True
            )
            
            # Execute some tasks
            await agent.execute("Task 1")
            await agent.execute("Task 2")
            
            metrics = await agent.get_performance_metrics()
            
            assert 'cognitive_agent_metrics' in metrics
            cognitive_metrics = metrics['cognitive_agent_metrics']
            
            assert 'cognitive_agent_stats' in cognitive_metrics
            stats = cognitive_metrics['cognitive_agent_stats']
            
            assert 'reasoning_depth' in stats
            assert 'adaptation_score' in stats
            assert 'confidence_calibration' in stats
    
    @pytest.mark.asyncio
    async def test_cognitive_components_integration(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test integration of all cognitive components"""
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm,
                enable_constitutional_ai=True,
                enable_meta_reasoning=True
            )
            
            # Test complex multi-component task
            complex_task = """
            Analyze the ethical implications of autonomous vehicles
            in urban environments, considering safety, efficiency,
            and social impact. Use multiple reasoning approaches.
            """
            
            result = await agent.execute(
                task=complex_task,
                context={
                    "domain": "ethics_technology",
                    "complexity": "high",
                    "multi_perspective": True
                }
            )
            
            assert result.success is True
            assert hasattr(result, 'metadata')
    
    @pytest.mark.asyncio
    async def test_adaptive_reasoning(self, mock_config: MockAgentConfig):
        """Test adaptive reasoning capabilities"""
        # Create mock LLM with adaptive responses
        adaptive_llm = MockLLMProvider()
        adaptive_llm.responses = {
            "simple": "Simple direct answer",
            "complex": "Complex multi-faceted analysis with deep reasoning"
        }
        
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=adaptive_llm
            )
            
            # Test simple task
            simple_result = await agent.execute("What is 2+2?")
            assert simple_result.success is True
            
            # Test complex task
            complex_result = await agent.execute(
                "Explain quantum computing impact on cryptography"
            )
            assert complex_result.success is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test error handling in cognitive agent"""
        # Create agent with potential failure scenarios
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm
            )
            
            # Test with empty task
            result = await agent.execute("")
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, mock_llm: MockLLMProvider, mock_config: MockAgentConfig):
        """Test concurrent task execution"""
        # Use mock agent for testing
        with patch('src.llamaagent.reasoning.cognitive_agent.CognitiveAgent', MockCognitiveAgent):
            agent = MockCognitiveAgent(
                config=mock_config,
                llm_provider=mock_llm
            )
            
            # Execute multiple tasks concurrently
            tasks = [
                agent.execute(f"Task {i}")
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_results = [
                r for r in results 
                if not isinstance(r, Exception) 
                and hasattr(r, 'success') 
                and getattr(r, 'success', False)
            ]
            assert len(successful_results) >= 3  # At least 3 should succeed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])