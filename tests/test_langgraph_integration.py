import pytest


def test_langgraph_chain_compiles():
    try:
        from llamaagent.integration.langgraph import build_react_chain
    except ImportError:
        pytest.skip("LangGraph not installed")

    chain = build_react_chain()
    assert callable(chain.invoke) 