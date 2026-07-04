"""
Tests for Tool Compass indexer module.

Tests HNSW index building, searching, and metadata management.
"""

from unittest.mock import patch

import numpy as np
import pytest

from indexer import CompassIndex, SearchResult
from embedder import EMBEDDING_DIM
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


class TestEmbeddingCacheSelfHeal:
    """SC-002 regression: a corrupt/truncated embedding_cache BLOB whose
    `dim` column still equals EMBEDDING_DIM must be treated as a cache MISS
    (and deleted), not crash the rebuild.

    _cache_get guarded reshape() only on the column-dim value, never the
    actual BLOB byte length. A row with dim==EMBEDDING_DIM but a
    truncated/corrupt BLOB made np.frombuffer(...).reshape(dim) raise
    ValueError. Because _cache_get runs inside build_index's BEGIN IMMEDIATE
    transaction, that uncaught ValueError rolled back and re-raised EVERY
    rebuild forever — defeating the documented self-heal. The fix validates
    len(blob) == dim * 4 and, on mismatch, deletes the bad row + reports a
    miss so the next pass re-populates it.
    """

    @pytest.mark.asyncio
    async def test_cache_get_truncated_blob_is_a_miss(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """Directly probe _cache_get: a dim-OK but wrong-length BLOB returns
        None (miss) and the bad row is deleted (self-heal).
        """
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        await index.build_index(sample_tools)

        text = sample_tools[0].embedding_text()
        text_hash = index._compute_text_hash(text)

        # Corrupt the stored BLOB to a truncated length while leaving the
        # `dim` column at the valid EMBEDDING_DIM — this is the exact shape
        # the column-dim check fails to catch.
        bad_blob = np.zeros(EMBEDDING_DIM - 5, dtype=np.float32).tobytes()
        with index._db_write_lock:
            index.db.execute(
                "UPDATE embedding_cache SET vector = ? WHERE text_hash = ?",
                (bad_blob, text_hash),
            )
            index.db.commit()

        # Must NOT raise ValueError; must report a miss (None).
        result = index._cache_get(text_hash)
        assert result is None, "corrupt-length BLOB must be treated as a miss"

        # The bad row must have been deleted (self-heal).
        row = index.db.execute(
            "SELECT COUNT(*) AS c FROM embedding_cache WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()
        assert row["c"] == 0, "corrupt cache row should be deleted on miss"

        await index.close()

    @pytest.mark.asyncio
    async def test_rebuild_self_heals_corrupt_cache_row(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """End-to-end: a corrupt cache row must not crash build_index — the
        rebuild treats it as a miss, re-embeds, and completes successfully.
        """
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        # First build populates the cache legitimately.
        await index.build_index(sample_tools)

        # Corrupt ONE cache row: keep dim == EMBEDDING_DIM, truncate the BLOB.
        text = sample_tools[1].embedding_text()
        text_hash = index._compute_text_hash(text)
        bad_blob = np.zeros(EMBEDDING_DIM - 3, dtype=np.float32).tobytes()
        with index._db_write_lock:
            index.db.execute(
                "UPDATE embedding_cache SET vector = ? WHERE text_hash = ?",
                (bad_blob, text_hash),
            )
            index.db.commit()

        # On the OLD code, _cache_get's reshape() raised ValueError inside the
        # BEGIN IMMEDIATE txn and this rebuild crashed (and would crash
        # forever). The fix must let the rebuild succeed.
        result = await index.build_index(sample_tools)
        assert result["tools_indexed"] == len(sample_tools)

        # And search still works after the self-heal.
        results = await index.search("read a file", top_k=3)
        assert len(results) > 0

        await index.close()


class TestBuildIndexAtomicRollbackWithCacheMiss:
    """IDX-COMPOSED-002: a cache write during build_index must NOT commit the
    open BEGIN IMMEDIATE transaction mid-rebuild.

    build_index wraps DELETE FROM tools + per-tool INSERTs + HNSW save in a
    single BEGIN IMMEDIATE txn whose documented purpose is atomic rollback:
    'Only commit AFTER HNSW save succeeds, so a failure leaves the previous DB
    state intact (no orphan SQLite rows).' But on a cache MISS the loop calls
    _cache_put, which issued self.db.commit() on the SHARED connection —
    prematurely committing the DELETE + INSERTs. If the subsequent HNSW save
    then failed, the tool rows were already committed with a stale/missing HNSW
    index → DB/HNSW divergence, silently degrading recall until the next
    successful full rebuild.

    The fix defers cache writes made during a rebuild and flushes them AFTER
    build_index's own commit; so when the HNSW save fails, rollback genuinely
    restores the previous tools table.
    """

    @pytest.mark.asyncio
    async def test_hnsw_save_failure_after_cache_miss_leaves_tools_unchanged(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """Force an HNSW-save failure partway through a rebuild that had a
        cache miss; the tools table must reflect the PREVIOUS state (rows NOT
        committed by the cache write)."""
        import hnswlib

        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )

        # First build: legitimate, populates the tools table + embedding cache.
        await index.build_index(sample_tools)

        pre_names = {
            r["name"]
            for r in index.db.execute("SELECT name FROM tools").fetchall()
        }
        assert pre_names == {t.name for t in sample_tools}, "precondition"
        pre_count = len(pre_names)

        # A DIFFERENT tool set with a BRAND-NEW embedding_text -> guaranteed
        # cache MISS -> _cache_put runs during the open rebuild txn.
        new_tools = [
            ToolDefinition(
                name="test:brand_new_alpha",
                description="A brand new alpha tool never seen before",
                category="other",
                server="test",
                parameters={"x": "str"},
                examples=["totally novel embedding text alpha"],
                is_core=False,
            ),
            ToolDefinition(
                name="test:brand_new_beta",
                description="A brand new beta tool never seen before",
                category="other",
                server="test",
                parameters={"y": "int"},
                examples=["totally novel embedding text beta"],
                is_core=False,
            ),
        ]

        # Make the HNSW index created DURING this rebuild fail on save_index,
        # AFTER the DELETE/INSERT and the cache-miss _cache_put have run.
        class _FailingSaveIndex(hnswlib.Index):
            def save_index(self, *args, **kwargs):
                raise RuntimeError("simulated HNSW save failure")

        # The rebuild must raise (save fails inside the txn) and roll back.
        with patch.object(hnswlib, "Index", _FailingSaveIndex):
            with pytest.raises(RuntimeError, match="simulated HNSW save failure"):
                await index.build_index(new_tools)

        # THE INVARIANT: rollback restored the previous tools table — the
        # cache write did NOT prematurely commit the DELETE + new INSERTs.
        post_names = {
            r["name"]
            for r in index.db.execute("SELECT name FROM tools").fetchall()
        }
        assert post_names == pre_names, (
            "build_index rollback must leave the previous tools table intact; "
            "a cache-miss _cache_put committed the open rebuild transaction"
        )
        post_count = index.db.execute(
            "SELECT COUNT(*) AS c FROM tools"
        ).fetchone()["c"]
        assert post_count == pre_count
        # The failed rebuild's brand-new rows must NOT be present.
        assert "test:brand_new_alpha" not in post_names
        assert "test:brand_new_beta" not in post_names

        await index.close()

    @pytest.mark.asyncio
    async def test_successful_rebuild_still_persists_cache_entries(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """The deferral must not lose cache entries on the happy path: after a
        successful rebuild with misses, the embedding_cache is populated (so a
        subsequent rebuild registers hits)."""
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )

        # First build populates cache via deferred-then-flushed puts.
        await index.build_index(sample_tools)

        cache_size = index.db.execute(
            "SELECT COUNT(*) AS c FROM embedding_cache"
        ).fetchone()["c"]
        assert cache_size == len(sample_tools), (
            "deferred cache puts must be flushed after a successful rebuild"
        )

        # Second identical build should be all cache HITS (proves the flushed
        # entries are readable and keyed correctly).
        hits_before = index._cache_hits
        await index.build_index(sample_tools)
        assert index._cache_hits >= hits_before + len(sample_tools)

        await index.close()


class TestGetToolByIdMalformedJson:
    """GW-A-002 sibling: a tools-table row with malformed JSON in the
    `parameters`/`examples` columns must be skipped-with-defaults inside
    _get_tool_by_id, not raise JSONDecodeError.

    _get_tool_by_id runs per-result inside search(); without the guard a
    single corrupt row raised and poisoned the ENTIRE result set (every
    result silently degraded to lexical) instead of dropping the one bad
    field. The fix falls back to {}/[] for the corrupt column and keeps the
    rest of the catalog searchable.
    """

    @pytest.mark.asyncio
    async def test_get_tool_by_id_bad_parameters_json_returns_defaults(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """A row with invalid JSON in `parameters` returns a ToolDefinition
        with parameters={} (and examples=[] when also corrupt), never a raise.
        """
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        await index.build_index(sample_tools)

        # Corrupt the parameters column for one existing tool while leaving at
        # least one VALID row untouched. Capture its row id for direct probing.
        with index._db_write_lock:
            index.db.execute(
                "UPDATE tools SET parameters = ?, examples = ? WHERE name = ?",
                ("{not json", "[also broken", "test:read_file"),
            )
            index.db.commit()
        row = index.db.execute(
            "SELECT id FROM tools WHERE name = ?", ("test:read_file",)
        ).fetchone()
        bad_id = row["id"]

        # Direct probe: must NOT raise JSONDecodeError; degrades to {} / [].
        tool = index._get_tool_by_id(bad_id)
        assert tool is not None
        assert tool.name == "test:read_file"
        assert tool.parameters == {}, (
            "malformed parameters JSON must fall back to {}"
        )
        assert tool.examples == [], (
            "malformed examples JSON must fall back to []"
        )

        await index.close()

    @pytest.mark.asyncio
    async def test_search_survives_one_corrupt_row(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """End-to-end: with one corrupt row present, a search that surfaces a
        DIFFERENT (valid) row still returns normally — the corrupt row does
        not poison the whole search.
        """
        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        await index.build_index(sample_tools)

        # Corrupt git_status's parameters; read_file stays valid.
        with index._db_write_lock:
            index.db.execute(
                "UPDATE tools SET parameters = ? WHERE name = ?",
                ("{not json", "test:git_status"),
            )
            index.db.commit()

        # A search must not raise, even if the corrupt row is among candidates.
        results = await index.search("read a file", top_k=5)
        assert len(results) > 0
        # Any surfaced result is a well-formed ToolDefinition (the corrupt row,
        # if surfaced, degrades to {} rather than raising).
        for r in results:
            assert isinstance(r.tool, ToolDefinition)
            assert isinstance(r.tool.parameters, dict)

        await index.close()


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


class TestCacheKeyIncorporatesProvider:
    """BE-FT-PE-001: the embedding cache key must fold in the PROVIDER name and
    base_url (in addition to the model) so switching embedding_provider /
    embedding_base_url can't return a stale cross-provider vector for the same
    text. The dim self-heal is unaffected — it keys on EMBEDDING_DIM + BLOB
    byte length, not this hash.
    """

    def _index_with_embedder(self, embedder, temp_index_path, temp_db_path):
        return CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=embedder,
        )

    def test_hash_differs_across_provider_names(
        self, temp_index_path, temp_db_path
    ):
        """Same text + same base_url + same model but DIFFERENT provider name
        -> different cache key."""
        from unittest.mock import Mock

        e1 = Mock()
        e1.provider_name = "ollama"
        e1.base_url = "http://localhost:11434"
        e1.model = "nomic-embed-text"

        e2 = Mock()
        e2.provider_name = "openai"
        e2.base_url = "http://localhost:11434"
        e2.model = "nomic-embed-text"

        idx1 = self._index_with_embedder(e1, temp_index_path, temp_db_path)
        idx2 = self._index_with_embedder(
            e2, temp_index_path.with_suffix(".2.hnsw"),
            temp_db_path.with_suffix(".2.db"),
        )

        text = "read a file from disk"
        assert idx1._compute_text_hash(text) != idx2._compute_text_hash(text)

    def test_hash_differs_across_base_urls(
        self, temp_index_path, temp_db_path
    ):
        """Same provider + model but DIFFERENT base_url -> different key
        (two OpenAI-compatible endpoints can return different vectors)."""
        from unittest.mock import Mock

        e1 = Mock()
        e1.provider_name = "openai"
        e1.base_url = "http://server-a:1234"
        e1.model = "text-embedding-3-small"

        e2 = Mock()
        e2.provider_name = "openai"
        e2.base_url = "http://server-b:1234"
        e2.model = "text-embedding-3-small"

        idx1 = self._index_with_embedder(e1, temp_index_path, temp_db_path)
        idx2 = self._index_with_embedder(
            e2, temp_index_path.with_suffix(".2.hnsw"),
            temp_db_path.with_suffix(".2.db"),
        )

        text = "read a file from disk"
        assert idx1._compute_text_hash(text) != idx2._compute_text_hash(text)

    def test_hash_stable_for_same_provider(
        self, temp_index_path, temp_db_path, mock_embedder
    ):
        """Identical embedder identity -> identical key (cache hits still
        work)."""
        idx = self._index_with_embedder(
            mock_embedder, temp_index_path, temp_db_path
        )
        text = "read a file from disk"
        assert idx._compute_text_hash(text) == idx._compute_text_hash(text)

    def test_hash_degrades_gracefully_without_provider_name(
        self, temp_index_path, temp_db_path
    ):
        """A pre-seam / mock embedder exposing only base_url + model must not
        crash _compute_text_hash (provider_name missing -> 'unknown')."""
        from unittest.mock import Mock

        e = Mock(spec=["base_url", "model"])
        e.base_url = "http://localhost:11434"
        e.model = "nomic-embed-text"
        idx = self._index_with_embedder(e, temp_index_path, temp_db_path)
        # Must produce a stable hex digest, not raise.
        h = idx._compute_text_hash("some text")
        assert isinstance(h, str)
        assert len(h) == 64
