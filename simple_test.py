#!/usr/bin/env python3
"""
Simple LlamaAgent Core Test

This script tests the core working components of LlamaAgent directly.
"""

import asyncio

def test_llm_factory():
    """Test LLM Factory directly."""
    print("Testing LLM Factory...")
    try:
        from src.llamaagent.llm.factory import LLMFactory
        from src.llamaagent.llm.messages import LLMMessage
        
        # Create factory
        factory = LLMFactory()
        print("PASS LLM Factory created")
        
        # Get available providers
        providers = factory.get_available_providers()
        print(f"PASS Available providers: {providers}")
        
        # Create mock provider
        provider = factory.create_provider('mock')
        print("PASS Mock provider created")
        
        return provider
    except Exception as e:
        print(f"FAIL LLM Factory test failed: {e}")
        return None

async def test_provider_functionality(provider):
    """Test provider functionality."""
    print("\nTesting Provider Functionality...")
    try:
        from src.llamaagent.llm.messages import LLMMessage
        
        # Test basic completion
        message = LLMMessage(role='user', content='Hello, world!')
        response = await provider.complete([message])
        print(f"PASS Basic completion: {response.content[:50]}...")
        
        # Test streaming
        print("PASS Testing streaming...")
        chunks = []
        async for chunk in provider.stream_chat_completion([message]):
            chunks.append(chunk)
        print(f"PASS Streaming worked: {len(chunks)} chunks")
        
        # Test embeddings
        texts = ["Hello", "World"]
        embeddings = await provider.embed_text(texts)
        print(f"PASS Embeddings: {len(embeddings['embeddings'])} vectors")
        
        return True
    except Exception as e:
        print(f"FAIL Provider test failed: {e}")
        return False

def test_tools():
    """Test tools system."""
    print("\nTesting Tools System...")
    try:
        from src.llamaagent.tools.calculator import CalculatorTool
        from src.llamaagent.tools.base import ToolRegistry
        
        # Create calculator
        calc = CalculatorTool()
        print("PASS Calculator tool created")
        
        # Test calculations
        result = calc.execute(expression="2 + 2")
        print(f"PASS 2 + 2 = {result}")
        
        result = calc.execute(expression="10 * 5")
        print(f"PASS 10 * 5 = {result}")
        
        # Test registry
        registry = ToolRegistry()
        registry.register(calc)
        print("PASS Tool registry working")
        
        return True
    except Exception as e:
        print(f"FAIL Tools test failed: {e}")
        return False

def test_types():
    """Test type system."""
    print("\nTesting Type System...")
    try:
        from src.llamaagent.types import AgentConfig, LLMMessage, LLMResponse
        
        # Test AgentConfig
        config = AgentConfig(
            agent_name='test-agent',
            model_name='mock-model',
            temperature=0.7
        )
        print("PASS AgentConfig created")
        
        # Test LLMMessage
        message = LLMMessage(role='user', content='Test message')
        print("PASS LLMMessage created")
        
        # Test LLMResponse
        response = LLMResponse(
            content='Test response',
            model='mock-model',
            provider='mock'
        )
        print("PASS LLMResponse created")
        
        return True
    except Exception as e:
        print(f"FAIL Types test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Analyzing LlamaAgent Core Functionality Test")
    print("=" * 50)
    
    results = {}
    
    # Test LLM Factory
    provider = test_llm_factory()
    results['llm_factory'] = provider is not None
    
    # Test Provider
    if provider:
        provider_ok = await test_provider_functionality(provider)
        results['provider'] = provider_ok
    else:
        results['provider'] = False
    
    # Test Tools
    tools_ok = test_tools()
    results['tools'] = tools_ok
    
    # Test Types
    types_ok = test_types()
    results['types'] = types_ok
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS Test Summary")
    print("=" * 50)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"Tests run: {total}")
    print(f"Tests passed: {passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test, status in results.items():
        icon = "PASS" if status else "FAIL"
        print(f"  {icon} {test.replace('_', ' ').title()}")
    
    if passed == total:
        print("\nSUCCESS ALL TESTS PASSED!")
        print("SUCCESS Core LlamaAgent functionality is working!")
        print("SUCCESS System is ready for use!")
    else:
        print(f"\n⚠️  {passed}/{total} tests passed")
        print("Some components need attention.")

if __name__ == "__main__":
    asyncio.run(main()) 