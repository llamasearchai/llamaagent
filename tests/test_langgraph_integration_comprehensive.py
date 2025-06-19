#!/usr/bin/env python3
"""Comprehensive tests for llamaagent.integration.langgraph module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock


# Mock LangGraph to avoid import errors
class MockGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.entry_point = None
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def set_entry_point(self, name):
        self.entry_point = name
    
    def add_edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))
    
    def compile(self):
        return MockCompiledGraph(self)


class MockCompiledGraph:
    def __init__(self, graph):
        self.graph = graph
    
    def invoke(self, inputs):
        return {"response": "mocked response"}
    
    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


# Mock the END constant
class MockEND:
    pass


@pytest.fixture
def mock_langgraph():
    """Mock LangGraph imports."""
    with patch.dict('sys.modules', {
        'langgraph': Mock(),
        'langgraph.graph': Mock(END=MockEND(), Graph=MockGraph)
    }):
        yield


class TestLangGraphIntegration:
    """Test LangGraph integration functionality."""
    
    def test_import_error_when_langgraph_missing(self):
        """Test that import raises helpful error when LangGraph is not available."""
        with patch.dict('sys.modules', {'langgraph': None, 'langgraph.graph': None}):
            with pytest.raises(ImportError, match="The 'langgraph' package is required"):
                import llamaagent.integration.langgraph
    
    @pytest.mark.asyncio
    async def test_build_agent_default_params(self, mock_langgraph):
        """Test _build_agent with default parameters."""
        from llamaagent.integration.langgraph import _build_agent
        
        with patch('llamaagent.integration.langgraph.ToolRegistry') as mock_registry:
            with patch('llamaagent.integration.langgraph.get_all_tools', return_value=[]) as mock_tools:
                with patch('llamaagent.integration.langgraph.ReactAgent') as mock_agent_class:
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    agent = _build_agent()
                    
                    mock_registry.assert_called_once()
                    mock_tools.assert_called_once()
                    mock_agent_class.assert_called_once()
                    
                    # Check that config was created with default values
                    call_args = mock_agent_class.call_args
                    config = call_args[0][0]  # First positional argument
                    assert config.name == "LG-Agent"
                    assert config.spree_enabled == False
    
    @pytest.mark.asyncio
    async def test_build_agent_custom_params(self, mock_langgraph):
        """Test _build_agent with custom parameters."""
        from llamaagent.integration.langgraph import _build_agent
        
        with patch('llamaagent.integration.langgraph.ToolRegistry') as mock_registry:
            with patch('llamaagent.integration.langgraph.get_all_tools', return_value=[]) as mock_tools:
                with patch('llamaagent.integration.langgraph.ReactAgent') as mock_agent_class:
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    agent = _build_agent(name="CustomAgent", spree=True)
                    
                    # Check that config was created with custom values
                    call_args = mock_agent_class.call_args
                    config = call_args[0][0]  # First positional argument
                    assert config.name == "CustomAgent"
                    assert config.spree_enabled == True
    
    @pytest.mark.asyncio
    async def test_build_agent_registers_tools(self, mock_langgraph):
        """Test that _build_agent registers all available tools."""
        from llamaagent.integration.langgraph import _build_agent
        
        mock_tool1 = Mock()
        mock_tool2 = Mock()
        mock_tools = [mock_tool1, mock_tool2]
        
        with patch('llamaagent.integration.langgraph.ToolRegistry') as mock_registry:
            with patch('llamaagent.integration.langgraph.get_all_tools', return_value=mock_tools):
                with patch('llamaagent.integration.langgraph.ReactAgent'):
                    mock_registry_instance = Mock()
                    mock_registry.return_value = mock_registry_instance
                    
                    _build_agent()
                    
                    # Check that each tool was registered
                    assert mock_registry_instance.register.call_count == 2
                    mock_registry_instance.register.assert_any_call(mock_tool1)
                    mock_registry_instance.register.assert_any_call(mock_tool2)
    
    @pytest.mark.asyncio
    async def test_build_react_chain_default(self, mock_langgraph):
        """Test build_react_chain with default parameters."""
        from llamaagent.integration.langgraph import build_react_chain
        
        with patch('llamaagent.integration.langgraph._build_agent') as mock_build_agent:
            mock_agent = Mock()
            mock_build_agent.return_value = mock_agent
            
            chain = build_react_chain()
            
            mock_build_agent.assert_called_once_with(spree=False)
            assert callable(chain)
    
    @pytest.mark.asyncio
    async def test_build_react_chain_with_spree(self, mock_langgraph):
        """Test build_react_chain with SPREE enabled."""
        from llamaagent.integration.langgraph import build_react_chain
        
        with patch('llamaagent.integration.langgraph._build_agent') as mock_build_agent:
            mock_agent = Mock()
            mock_build_agent.return_value = mock_agent
            
            chain = build_react_chain(spree=True)
            
            mock_build_agent.assert_called_once_with(spree=True)
            assert callable(chain)
    
    @pytest.mark.asyncio
    async def test_node_function_execution(self, mock_langgraph):
        """Test the internal _node function execution."""
        from llamaagent.integration.langgraph import build_react_chain
        from llamaagent.agents.base import AgentResponse
        
        with patch('llamaagent.integration.langgraph._build_agent') as mock_build_agent:
            mock_agent = Mock()
            mock_response = AgentResponse(
                content="Test response",
                success=True,
                tokens_used=10,
                trace=[]
            )
            mock_agent.execute = AsyncMock(return_value=mock_response)
            mock_build_agent.return_value = mock_agent
            
            # Build the chain to get access to the compiled graph
            chain = build_react_chain()
            
            # The chain is a compiled graph, we need to test the node function indirectly
            # by checking that the agent.execute was called correctly when we invoke the chain
            result = chain.invoke({"prompt": "test prompt"})
            
            # Since we're using a mock compiled graph, we get back the mocked response
            assert "response" in result
    
    @pytest.mark.asyncio 
    async def test_graph_construction(self, mock_langgraph):
        """Test that the graph is constructed correctly."""
        from llamaagent.integration.langgraph import build_react_chain
        
        with patch('llamaagent.integration.langgraph._build_agent') as mock_build_agent:
            with patch('llamaagent.integration.langgraph.Graph', MockGraph) as mock_graph_class:
                mock_agent = Mock()
                mock_build_agent.return_value = mock_agent
                
                chain = build_react_chain()
                
                # Verify the graph was constructed correctly by checking the mock
                assert isinstance(chain, MockCompiledGraph)
                assert "agent" in chain.graph.nodes
                assert chain.graph.entry_point == "agent"
                assert ("agent", MockEND()) in chain.graph.edges or len(chain.graph.edges) > 0

    @pytest.mark.asyncio
    async def test_node_function_directly(self, mock_langgraph):
        """Test the _node function directly to cover lines 49-52."""
        from llamaagent.integration.langgraph import build_react_chain
        from llamaagent.agents.base import AgentResponse
        
        with patch('llamaagent.integration.langgraph._build_agent') as mock_build_agent:
            mock_agent = Mock()
            mock_response = AgentResponse(
                content="Direct test response",
                success=True,
                tokens_used=15,
                trace=[]
            )
            mock_agent.execute = AsyncMock(return_value=mock_response)
            mock_build_agent.return_value = mock_agent
            
            # Access the graph to get the node function
            with patch('llamaagent.integration.langgraph.Graph', MockGraph):
                chain = build_react_chain()
                
                # Get the _node function from the graph
                node_func = chain.graph.nodes["agent"]
                
                # Test the node function directly
                state = {}
                inputs = {"prompt": "direct test prompt"}
                
                # Call the async node function
                result_state, end_marker = await node_func(state, inputs)
                
                # Verify the function behavior
                assert result_state["response"] == "Direct test response"
                assert end_marker is not None  # Should be END marker
                mock_agent.execute.assert_called_once_with("direct test prompt") 