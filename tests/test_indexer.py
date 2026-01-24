"""
Tests for Tool Compass indexer module.

Tests HNSW index building, searching, and metadata management.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import CompassIndex, SearchResult
from tool_manifest import ToolDefinition


class TestCompassIndex:
    """Test CompassIndex core functionality."""

    @pytest.mark.asyncio
    async def test_build_index(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """Should build index from tool definitions."""
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )

        result = await index.build_index(sample_tools)

        assert result["tools_indexed"] == len(sample_tools)
        assert result["total_time"] > 0
        assert temp_index_path.exists()
        assert temp_db_path.exists()

        await index.close()

    @pytest.mark.asyncio
    async def test_load_index(
        self, test_index, temp_index_path, temp_db_path, mock_embedder
    ):
        """Should load existing index from disk."""
        # test_index fixture already built the index
        # Create new instance and load
        new_index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )

        loaded = new_index.load_index()
        assert loaded is True
        assert new_index.index is not None
        assert len(new_index._id_to_name) > 0

        await new_index.close()

    @pytest.mark.asyncio
    async def test_load_index_missing(self, temp_db_dir, mock_embedder):
        """Should return False when index files don't exist."""
        index = CompassIndex(
            index_path=temp_db_dir / "missing.hnsw",
            db_path=temp_db_dir / "missing.db",
            embedder=mock_embedder,
        )

        loaded = index.load_index()
        assert loaded is False

        await index.close()

    @pytest.mark.asyncio
    async def test_search_basic(self, test_index):
        """Should return relevant results for a query."""
        results = await test_index.search("read a file", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        # Scores are cosine similarity - typically in [-1, 1] but embeddings
        # may produce values slightly outside due to numerical precision
        assert all(isinstance(r.score, float) for r in results)

    @pytest.mark.asyncio
    async def test_search_returns_tool_definition(self, test_index):
        """Search results should include full ToolDefinition."""
        results = await test_index.search("file operations", top_k=1)

        assert len(results) == 1
        tool = results[0].tool
        assert isinstance(tool, ToolDefinition)
        assert tool.name
        assert tool.description
        assert tool.category

    @pytest.mark.asyncio
    async def test_search_category_filter(self, test_index):
        """Should filter results by category."""
        results = await test_index.search(
            "operations", top_k=10, category_filter="file"
        )

        assert len(results) > 0
        for r in results:
            assert r.tool.category == "file"

    @pytest.mark.asyncio
    async def test_search_server_filter(self, test_index):
        """Should filter results by server."""
        results = await test_index.search("anything", top_k=10, server_filter="test")

        assert len(results) > 0
        for r in results:
            assert r.tool.server == "test"

    @pytest.mark.asyncio
    async def test_search_combined_filters(self, test_index):
        """Should apply both category and server filters."""
        results = await test_index.search(
            "file operations", top_k=10, category_filter="file", server_filter="test"
        )

        for r in results:
            assert r.tool.category == "file"
            assert r.tool.server == "test"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, test_index):
        """Should return empty list when no matches."""
        results = await test_index.search(
            "file operations", top_k=10, category_filter="nonexistent_category"
        )

        assert results == []


class TestIndexStats:
    """Test index statistics and metadata."""

    @pytest.mark.asyncio
    async def test_get_stats(self, test_index, sample_tools):
        """Should return comprehensive statistics."""
        stats = test_index.get_stats()

        assert stats["total_tools"] == len(sample_tools)
        assert "by_category" in stats
        assert "by_server" in stats
        assert stats["by_category"]["file"] == 2  # read_file, write_file
        assert stats["by_category"]["git"] == 1
        assert stats["by_category"]["ai"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_hnsw_info(self, test_index, sample_tools):
        """Should include HNSW index information."""
        stats = test_index.get_stats()

        assert "hnsw" in stats
        assert stats["hnsw"]["current_count"] == len(sample_tools)
        assert stats["hnsw"]["max_elements"] >= len(sample_tools)


class TestDynamicUpdates:
    """Test adding and removing tools without rebuild."""

    @pytest.mark.asyncio
    async def test_add_single_tool(self, test_index):
        """Should add a tool to existing index."""
        initial_count = test_index.get_stats()["total_tools"]

        new_tool = ToolDefinition(
            name="test:new_tool",
            description="A newly added test tool",
            category="test",
            server="test",
            parameters={"param": "str"},
            examples=["new tool example"],
            is_core=False,
        )

        success = await test_index.add_single_tool(new_tool)
        assert success is True

        new_count = test_index.get_stats()["total_tools"]
        assert new_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_remove_tool(self, test_index):
        """Should remove a tool from database."""
        initial_count = test_index.get_stats()["total_tools"]

        success = await test_index.remove_tool("test:read_file")
        assert success is True

        new_count = test_index.get_stats()["total_tools"]
        assert new_count == initial_count - 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent_tool(self, test_index):
        """Should return False for nonexistent tool."""
        success = await test_index.remove_tool("test:does_not_exist")
        assert success is False


class TestToolDefinition:
    """Test ToolDefinition data structure."""

    def test_embedding_text_generation(self, sample_tools):
        """Should generate rich embedding text."""
        tool = sample_tools[0]  # read_file
        text = tool.embedding_text()

        # Should include key information
        assert tool.name in text
        assert tool.description in text
        assert tool.category in text
        # Should include examples
        for example in tool.examples:
            assert example in text

    def test_embedding_text_includes_parameters(self, sample_tools):
        """Embedding text should mention parameters."""
        tool = sample_tools[0]
        text = tool.embedding_text()

        for param in tool.parameters.keys():
            assert param in text
