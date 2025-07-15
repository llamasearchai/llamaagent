"""
Direct tests for the OpenAI stub functionality.

This module tests components that directly use the openai library
and can benefit from the stub implementation.
"""

import pytest

from src.llamaagent.integration._openai_stub import (
    install_openai_stub,
    uninstall_openai_stub,
)


class TestOpenAIStubDirect:
    """Test direct usage of OpenAI stub."""

    def setup_method(self):
        """Set up test method."""
        uninstall_openai_stub()

    def teardown_method(self):
        """Tear down test method."""
        uninstall_openai_stub()

    def test_code_that_uses_openai_directly(self):
        """Test code that imports and uses openai directly."""
        install_openai_stub()

        # Example function that uses OpenAI
        def generate_summary(text: str) -> str:
            """Generate a summary using OpenAI."""
            import openai

            client = openai.OpenAI(api_key="sk-prod-key-123")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Summarize the following text."},
                    {"role": "user", "content": text},
                ],
            )
            return response.choices[0].message.content

        # Test the function
        summary = generate_summary("This is a long text about testing.")
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_embedding_generation_code(self):
        """Test code that generates embeddings."""
        install_openai_stub()

        def get_embeddings(texts: list) -> list:
            """Get embeddings for a list of texts."""
            import openai

            client = openai.OpenAI(api_key="sk-embed-key")
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=texts
            )
            return [item.embedding for item in response.data]

        # Test embedding generation
        texts = ["Hello world", "Testing embeddings", "OpenAI stub"]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_async_code_with_openai(self):
        """Test async code that uses OpenAI."""
        install_openai_stub()

        async def async_chat(message: str) -> str:
            """Async chat function using OpenAI."""
            import openai

            client = openai.AsyncOpenAI(api_key="sk-async-key")
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": message}]
            )
            return response.choices[0].message.content

        # Test async function
        result = await async_chat("Tell me a joke")
        assert result is not None
        assert isinstance(result, str)

    def test_moderation_workflow(self):
        """Test a content moderation workflow."""
        install_openai_stub()

        def moderate_content(texts: list) -> dict:
            """Moderate a list of texts."""
            import openai

            client = openai.OpenAI(api_key="sk-mod-key")
            results = {}

            for text in texts:
                response = client.moderations.create(input=text)
                results[text] = {
                    "flagged": response.results[0].flagged,
                    "categories": [
                        cat
                        for cat, val in response.results[0].categories.__dict__.items()
                        if val
                    ],
                }

            return results

        # Test moderation
        texts = [
            "This is a friendly message",
            "This contains violence and hate",
            "Normal business communication",
        ]
        results = moderate_content(texts)

        assert results["This is a friendly message"]["flagged"] is False
        assert results["This contains violence and hate"]["flagged"] is True
        assert "violence" in results["This contains violence and hate"]["categories"]
        assert "hate" in results["This contains violence and hate"]["categories"]

    def test_integration_with_vector_memory(self):
        """Test integration with components that might use OpenAI embeddings."""
        install_openai_stub()

        class SimpleVectorStore:
            """Simple vector store that uses OpenAI embeddings."""

            def __init__(self):
                import openai

                self.client = openai.OpenAI(api_key="sk-vector-key")
                self.vectors = {}

            def add_text(self, text: str, metadata: dict = None):
                """Add text with its embedding."""
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002", input=text
                )
                embedding = response.data[0].embedding
                self.vectors[text] = {
                    "embedding": embedding,
                    "metadata": metadata or {},
                }

            def search(self, query: str, top_k: int = 5):
                """Search for similar texts."""
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002", input=query
                )
                query_embedding = response.data[0].embedding

                # Simple cosine similarity
                results = []
                for text, data in self.vectors.items():
                    score = sum(
                        a * b
                        for a, b in zip(
                            query_embedding, data["embedding"], strict=False
                        )
                    )
                    results.append((score, text, data["metadata"]))

                results.sort(reverse=True)
                return results[:top_k]

        # Test the vector store
        store = SimpleVectorStore()
        store.add_text("Python programming", {"category": "tech"})
        store.add_text("Machine learning basics", {"category": "ai"})
        store.add_text("Cooking recipes", {"category": "food"})

        results = store.search("artificial intelligence", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[0], float) for r in results)  # scores
        assert all(isinstance(r[1], str) for r in results)  # texts

    def test_retry_logic_with_stub(self):
        """Test retry logic when using the stub."""
        install_openai_stub()

        def resilient_completion(prompt: str, max_retries: int = 3) -> str:
            """Completion with retry logic."""
            import time

            import openai

            client = openai.OpenAI(api_key="sk-retry-key")

            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4", messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.1)  # Brief delay before retry

            return "Failed after all retries"

        # Test the function
        result = resilient_completion("Test prompt")
        assert result is not None
        assert isinstance(result, str)

    def test_streaming_simulation(self):
        """Test that streaming can be simulated with the stub."""
        install_openai_stub()

        def stream_response(prompt: str) -> list:
            """Simulate streaming response collection."""
            import openai

            client = openai.OpenAI(api_key="sk-stream-key")

            # Note: Our stub doesn't implement streaming, but we can test
            # the non-streaming fallback
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=False,  # Explicitly not streaming
            )

            # Simulate chunking the response
            content = response.choices[0].message.content
            chunks = [content[i : i + 10] for i in range(0, len(content), 10)]
            return chunks

        # Test streaming simulation
        chunks = stream_response("Generate text")
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Reconstruct the full response
        full_response = "".join(chunks)
        assert "mock response" in full_response
