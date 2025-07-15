"""
Example usage of the OpenAI API stub for testing and development.

This example demonstrates how to use the mock OpenAI implementation
to develop and test code without making real API calls.
"""

import asyncio
from src.llamaagent.integration._openai_stub import install_openai_stub, uninstall_openai_stub


def basic_usage_example():
    """Basic example of using the OpenAI stub."""
    print("=== Basic OpenAI Stub Usage ===\n")
    
    # Install the stub to intercept OpenAI imports
    install_openai_stub()
    
    # Now we can import and use OpenAI normally
    import openai
    
    # Create a client (no real API key needed)
    client = openai.OpenAI(api_key="sk-test-key-123")
    
    # Make a chat completion request
    print("1. Chat Completion Example:")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you write some code for me?"}
        ]
    )
    print(f"Response: {response.choices[0].message.content}\n")
    
    # Generate embeddings
    print("2. Embeddings Example:")
    embedding_response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="This is a test sentence for embedding."
    )
    print(f"Embedding dimensions: {len(embedding_response.data[0].embedding)}")
    print(f"First 5 values: {embedding_response.data[0].embedding[:5]}\n")
    
    # Check content moderation
    print("3. Moderation Example:")
    moderation_response = client.moderations.create(
        input="This is a friendly message about coding."
    )
    print(f"Content flagged: {moderation_response.results[0].flagged}")
    
    # Test with problematic content
    moderation_response2 = client.moderations.create(
        input="This message contains violence and hate speech."
    )
    print(f"Problematic content flagged: {moderation_response2.results[0].flagged}")
    print(f"Categories: {[cat for cat, val in moderation_response2.results[0].categories.__dict__.items() if val]}\n")
    
    # Clean up
    uninstall_openai_stub()
    print("Stub uninstalled successfully!\n")


async def async_usage_example():
    """Example of using the async OpenAI client."""
    print("=== Async OpenAI Stub Usage ===\n")
    
    # Install the stub
    install_openai_stub()
    
    import openai
    
    # Create an async client
    async_client = openai.AsyncOpenAI(api_key="sk-test-key-456")
    
    # Make async requests
    print("1. Async Chat Completion:")
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Calculate something for me"}]
    )
    print(f"Response: {response.choices[0].message.content}\n")
    
    # Batch embedding requests
    print("2. Batch Async Embeddings:")
    texts = ["First text", "Second text", "Third text"]
    embedding_tasks = [
        async_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        for text in texts
    ]
    
    embeddings = await asyncio.gather(*embedding_tasks)
    for i, emb in enumerate(embeddings):
        print(f"Text {i+1} embedding: {emb.data[0].embedding[:3]}...")
    
    # Clean up
    uninstall_openai_stub()
    print("\nAsync example completed!\n")


def integration_example():
    """Example of integrating the stub with existing code."""
    print("=== Integration Example ===\n")
    
    # Install stub before any OpenAI imports in your code
    install_openai_stub()
    
    # Your existing code that uses OpenAI
    from openai import OpenAI
    
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment using OpenAI (mocked)."""
        client = OpenAI(api_key="sk-prod-key")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the sentiment of the text."},
                {"role": "user", "content": text}
            ]
        )
        
        return response.choices[0].message.content
    
    # Test the function
    result = analyze_sentiment("I love programming with Python!")
    print(f"Sentiment analysis result: {result}\n")
    
    # Token usage tracking
    def process_documents(documents: list) -> dict:
        """Process multiple documents and track token usage."""
        client = OpenAI(api_key="sk-prod-key")
        total_tokens = 0
        results = []
        
        for doc in documents:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Summarize: {doc}"}]
            )
            
            total_tokens += response.usage.total_tokens
            results.append(response.choices[0].message.content)
        
        return {
            "summaries": results,
            "total_tokens": total_tokens
        }
    
    docs = ["Document 1 content", "Document 2 content", "Document 3 content"]
    result = process_documents(docs)
    print(f"Processed {len(docs)} documents")
    print(f"Total tokens used: {result['total_tokens']}\n")
    
    # Clean up
    uninstall_openai_stub()


def error_handling_example():
    """Example of error handling with the stub."""
    print("=== Error Handling Example ===\n")
    
    install_openai_stub()
    import openai
    
    # Test authentication error
    try:
        # Using "test-key" or "test_api" triggers auth error
        client = openai.OpenAI(api_key="test-key")
        # In a real implementation, this would raise AuthenticationError
        print("Note: Authentication errors would be raised here in full implementation\n")
    except openai.AuthenticationError as e:
        print(f"Authentication failed: {e}")
    
    # Test with valid key
    client = openai.OpenAI(api_key="sk-valid-key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "test successful"}]
    )
    print(f"Success response: {response.choices[0].message.content}\n")
    
    uninstall_openai_stub()


if __name__ == "__main__":
    # Run all examples
    basic_usage_example()
    
    # Run async example
    asyncio.run(async_usage_example())
    
    integration_example()
    error_handling_example()
    
    print("All examples completed successfully!")