"""
Tests for Tool Compass embedder module.

Tests Ollama embedding generation, both async and sync wrappers.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import httpx
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from embedder import (
    Embedder,
    SyncEmbedder,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)


# =============================================================================
# Embedder Initialization Tests
# =============================================================================


class TestEmbedderInit:
    """Test Embedder initialization."""

    def test_default_initialization(self):
        """Should initialize with defaults."""
        embedder = Embedder()

        assert embedder.base_url == OLLAMA_BASE_URL
        assert embedder.model == EMBEDDING_MODEL
        assert embedder.timeout == 30.0
        assert embedder._client is None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        embedder = Embedder(
            base_url="http://custom:8080",
            model="custom-model",
            timeout=60.0,
        )

        assert embedder.base_url == "http://custom:8080"
        assert embedder.model == "custom-model"
        assert embedder.timeout == 60.0


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestHTTPClient:
    """Test HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Should create client on first access."""
        embedder = Embedder()

        client = await embedder._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await embedder.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Should reuse existing client."""
        embedder = Embedder()

        client1 = await embedder._get_client()
        client2 = await embedder._get_client()

        assert client1 is client2

        await embedder.close()

    @pytest.mark.asyncio
    async def test_get_client_recreates_if_closed(self):
        """Should recreate client if previous was closed."""
        embedder = Embedder()

        client1 = await embedder._get_client()
        await client1.aclose()

        client2 = await embedder._get_client()

        assert client2 is not client1

        await embedder.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Should close HTTP client."""
        embedder = Embedder()

        await embedder._get_client()
        await embedder.close()

        assert embedder._client is None


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test Ollama health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Should return True when model is available."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "llama2"},
            ]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(self):
        """Should return False when model not found."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2"}]}

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self):
        """Should return False on connection error."""
        embedder = Embedder()

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_get_client.return_value = mock_client

            result = await embedder.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_server_error(self):
        """Should return False on server error."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 500

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.health_check()

            assert result is False


# =============================================================================
# Model Pull Tests
# =============================================================================


class TestPullModel:
    """Test model pulling."""

    @pytest.mark.asyncio
    async def test_pull_model_success(self):
        """Should return True on successful pull."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.pull_model()

            assert result is True
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_pull_model_failure(self):
        """Should return False on pull failure."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.pull_model()

            assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_exception(self):
        """Should return False on exception."""
        embedder = Embedder()

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            mock_get_client.return_value = mock_client

            result = await embedder.pull_model()

            assert result is False


# =============================================================================
# Embed Tests
# =============================================================================


class TestEmbed:
    """Test single text embedding."""

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Should return normalized embedding."""
        embedder = Embedder()

        # Create a mock embedding
        mock_embedding = np.random.randn(EMBEDDING_DIM).tolist()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [mock_embedding]}

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed("Test text for embedding")

            assert isinstance(result, np.ndarray)
            assert result.shape == (EMBEDDING_DIM,)
            assert result.dtype == np.float32
            # Check normalized (unit length)
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 0.0001

    @pytest.mark.asyncio
    async def test_embed_adds_prefix(self):
        """Should add search_document prefix."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await embedder.embed("Test text")

            # Check that search_document prefix was added
            call_args = mock_client.post.call_args
            assert "search_document:" in call_args[1]["json"]["input"]

    @pytest.mark.asyncio
    async def test_embed_failure_raises(self):
        """Should raise on embedding failure."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(RuntimeError, match="Embedding failed"):
                await embedder.embed("Test text")


# =============================================================================
# Embed Query Tests
# =============================================================================


class TestEmbedQuery:
    """Test query embedding."""

    @pytest.mark.asyncio
    async def test_embed_query_success(self):
        """Should return normalized query embedding."""
        embedder = Embedder()

        mock_embedding = np.random.randn(EMBEDDING_DIM).tolist()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [mock_embedding]}

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed_query("find files")

            assert isinstance(result, np.ndarray)
            assert result.shape == (EMBEDDING_DIM,)

    @pytest.mark.asyncio
    async def test_embed_query_uses_query_prefix(self):
        """Should use search_query prefix."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await embedder.embed_query("search query")

            call_args = mock_client.post.call_args
            assert "search_query:" in call_args[1]["json"]["input"]


# =============================================================================
# Embed Batch Tests
# =============================================================================


class TestEmbedBatch:
    """Test batch embedding."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """Should return batch of embeddings."""
        embedder = Embedder()

        texts = ["text 1", "text 2", "text 3"]

        # Mock individual embed calls
        async def mock_embed(text):
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with patch.object(embedder, "embed", new=mock_embed):
            result = await embedder.embed_batch(texts)

            assert isinstance(result, np.ndarray)
            assert result.shape == (3, EMBEDDING_DIM)

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Should handle empty list gracefully or raise clear error."""
        embedder = Embedder()

        async def mock_embed(text):
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with patch.object(embedder, "embed", new=mock_embed):
            # Empty list raises ValueError from numpy - this is expected behavior
            with pytest.raises(ValueError, match="need at least one array"):
                await embedder.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_parallel(self):
        """Should process embeddings in parallel."""
        embedder = Embedder()

        call_times = []

        async def mock_embed(text):
            import time

            call_times.append(time.time())
            await asyncio.sleep(0.01)  # Small delay
            return np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with patch.object(embedder, "embed", new=mock_embed):
            await embedder.embed_batch(["a", "b", "c"])

            # All calls should have started at approximately the same time
            # (within a small window, indicating parallel execution)
            if len(call_times) >= 2:
                time_spread = max(call_times) - min(call_times)
                # Parallel calls should start within 0.05s of each other
                assert time_spread < 0.05


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Test embedding normalization."""

    @pytest.mark.asyncio
    async def test_embed_normalizes_output(self):
        """Embedding should be unit normalized."""
        embedder = Embedder()

        # Create non-normalized embedding
        raw_embedding = [1.0] * EMBEDDING_DIM  # Length = sqrt(768) ~ 27.7

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [raw_embedding]}

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed("test")

            # Should be normalized to unit length
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 0.0001

    @pytest.mark.asyncio
    async def test_embed_handles_zero_norm(self):
        """Should handle zero-norm edge case."""
        embedder = Embedder()

        # Zero embedding
        zero_embedding = [0.0] * EMBEDDING_DIM

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [zero_embedding]}

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed("test")

            # Should not crash, may return zeros
            assert isinstance(result, np.ndarray)


# =============================================================================
# SyncEmbedder Tests
# =============================================================================


class TestSyncEmbedder:
    """Test synchronous embedder wrapper."""

    def test_sync_embedder_init(self):
        """Should initialize with async embedder."""
        embedder = SyncEmbedder(base_url="http://test:8080")

        assert embedder._async_embedder is not None
        assert embedder._async_embedder.base_url == "http://test:8080"

    def test_sync_embedder_health_check(self):
        """Should wrap async health_check."""
        embedder = SyncEmbedder()

        with patch.object(
            embedder._async_embedder, "health_check", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = True

            result = embedder.health_check()

            assert result is True

    def test_sync_embedder_embed(self):
        """Should wrap async embed."""
        embedder = SyncEmbedder()
        expected = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with patch.object(
            embedder._async_embedder, "embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = expected

            result = embedder.embed("test text")

            assert np.array_equal(result, expected)

    def test_sync_embedder_embed_query(self):
        """Should wrap async embed_query."""
        embedder = SyncEmbedder()
        expected = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with patch.object(
            embedder._async_embedder, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = expected

            result = embedder.embed_query("search query")

            assert np.array_equal(result, expected)

    def test_sync_embedder_embed_batch(self):
        """Should wrap async embed_batch."""
        embedder = SyncEmbedder()
        expected = np.random.randn(3, EMBEDDING_DIM).astype(np.float32)

        with patch.object(
            embedder._async_embedder, "embed_batch", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = expected

            result = embedder.embed_batch(["a", "b", "c"])

            assert np.array_equal(result, expected)

    def test_sync_embedder_close(self):
        """Should close async embedder and loop."""
        embedder = SyncEmbedder()

        with patch.object(
            embedder._async_embedder, "close", new_callable=AsyncMock
        ) as mock_close:
            embedder.close()

            mock_close.assert_called_once()


# =============================================================================
# Integration Tests (marked)
# =============================================================================


@pytest.mark.integration
class TestEmbedderIntegration:
    """Integration tests requiring running Ollama server."""

    @pytest.mark.asyncio
    async def test_real_health_check(self):
        """Test against real Ollama server."""
        embedder = Embedder()

        result = await embedder.health_check()

        # Should return True if Ollama is running with model
        # or False if not available
        assert isinstance(result, bool)

        await embedder.close()

    @pytest.mark.asyncio
    async def test_real_embedding(self):
        """Test real embedding generation."""
        embedder = Embedder()

        if not await embedder.health_check():
            pytest.skip("Ollama not available")

        result = await embedder.embed("Test document for embedding")

        assert result.shape == (EMBEDDING_DIM,)
        assert abs(np.linalg.norm(result) - 1.0) < 0.0001

        await embedder.close()

    @pytest.mark.asyncio
    async def test_real_similarity(self):
        """Test embedding similarity for related texts."""
        embedder = Embedder()

        if not await embedder.health_check():
            pytest.skip("Ollama not available")

        # Similar texts should have high similarity
        emb1 = await embedder.embed("Read file contents from disk")
        emb2 = await embedder.embed("Get file data from filesystem")
        emb3 = await embedder.embed("Generate image from text prompt")

        sim_related = np.dot(emb1, emb2)
        sim_unrelated = np.dot(emb1, emb3)

        # Related texts should be more similar
        assert sim_related > sim_unrelated

        await embedder.close()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_embed_empty_string(self):
        """Should handle empty string."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed("")

            assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_embed_unicode_text(self):
        """Should handle Unicode text."""
        embedder = Embedder()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed(
                "Unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00"
            )

            assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_embed_very_long_text(self):
        """Should handle very long text."""
        embedder = Embedder()

        long_text = "word " * 10000

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed(long_text)

            assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_embed_special_characters(self):
        """Should handle special characters."""
        embedder = Embedder()

        special_text = 'Tab:\t Newline:\n Quote:" Backslash:\\ Null:\x00'

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [np.random.randn(EMBEDDING_DIM).tolist()]
        }

        with patch.object(embedder, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await embedder.embed(special_text)

            assert isinstance(result, np.ndarray)
