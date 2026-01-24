"""
Tool Compass - Embedder Module
Handles embedding generation via Ollama's nomic-embed-text model.
"""

import httpx
import numpy as np
from typing import List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text dimension


class Embedder:
    """
    Async embedder using Ollama's nomic-embed-text model.
    Optimized for tool description embedding.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = EMBEDDING_MODEL,
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check if our model is available (with or without :latest tag)
                return any(self.model in m for m in models)
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def pull_model(self) -> bool:
        """Pull the embedding model if not present."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": self.model},
                timeout=300.0,  # Model download can take time
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed (tool description, query, etc.)

        Returns:
            numpy array of shape (EMBEDDING_DIM,)
        """
        client = await self._get_client()

        # Add task prefix for better retrieval (nomic-embed-text recommendation)
        prefixed_text = f"search_document: {text}"

        response = await client.post(
            "/api/embed", json={"model": self.model, "input": prefixed_text}
        )

        if response.status_code != 200:
            raise RuntimeError(f"Embedding failed: {response.text}")

        data = response.json()
        embedding = np.array(data["embeddings"][0], dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Uses search_query prefix for better retrieval.

        Args:
            query: Search query (user intent)

        Returns:
            numpy array of shape (EMBEDDING_DIM,)
        """
        client = await self._get_client()

        # Query prefix for retrieval tasks
        prefixed_query = f"search_query: {query}"

        response = await client.post(
            "/api/embed", json={"model": self.model, "input": prefixed_query}
        )

        if response.status_code != 200:
            raise RuntimeError(f"Embedding failed: {response.text}")

        data = response.json()
        embedding = np.array(data["embeddings"][0], dtype=np.float32)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (len(texts), EMBEDDING_DIM)
        """
        # Ollama doesn't support true batching, so we parallelize
        tasks = [self.embed(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return np.stack(embeddings)


# Synchronous wrapper for non-async contexts
class SyncEmbedder:
    """Synchronous wrapper around async Embedder."""

    def __init__(self, **kwargs):
        self._async_embedder = Embedder(**kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run(self, coro):
        """Run coroutine in event loop."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def health_check(self) -> bool:
        return self._run(self._async_embedder.health_check())

    def embed(self, text: str) -> np.ndarray:
        return self._run(self._async_embedder.embed(text))

    def embed_query(self, query: str) -> np.ndarray:
        return self._run(self._async_embedder.embed_query(query))

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self._run(self._async_embedder.embed_batch(texts))

    def close(self):
        self._run(self._async_embedder.close())
        if self._loop and not self._loop.is_closed():
            self._loop.close()


if __name__ == "__main__":
    # Quick test
    async def test():
        embedder = Embedder()

        print("Checking Ollama health...")
        healthy = await embedder.health_check()
        print(f"Ollama available: {healthy}")

        if healthy:
            print("\nGenerating test embedding...")
            emb = await embedder.embed("Read file contents from disk")
            print(f"Embedding shape: {emb.shape}")
            print(f"Embedding norm: {np.linalg.norm(emb):.4f}")

            print("\nGenerating query embedding...")
            query_emb = await embedder.embed_query("I need to read a file")
            print(f"Query embedding shape: {query_emb.shape}")

            # Test similarity
            similarity = np.dot(emb, query_emb)
            print(f"Similarity: {similarity:.4f}")

        await embedder.close()

    asyncio.run(test())
