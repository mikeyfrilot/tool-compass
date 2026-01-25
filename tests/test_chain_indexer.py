"""
Tests for Tool Compass chain indexer module.

Tests tool chain/workflow indexing, searching, and management.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sqlite3

from chain_indexer import (
    ChainIndexer,
    ToolChain,
    ChainSearchResult,
    get_chain_indexer,
    EMBEDDING_DIM,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    embedder = Mock()

    def mock_embed(text: str) -> np.ndarray:
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)

    async def async_embed(text: str) -> np.ndarray:
        return mock_embed(text)

    async def async_embed_query(query: str) -> np.ndarray:
        return mock_embed(f"query: {query}")

    embedder.embed = AsyncMock(side_effect=async_embed)
    embedder.embed_query = AsyncMock(side_effect=async_embed_query)
    embedder.close = AsyncMock()

    return embedder


@pytest.fixture
def temp_chain_dir(tmp_path):
    """Temporary directory for chain database and index."""
    return tmp_path


@pytest.fixture
def chain_indexer(mock_embedder, temp_chain_dir):
    """Create chain indexer with temp storage."""
    # Create the database first with required schema
    db_path = temp_chain_dir / "test_chains.db"
    db = sqlite3.connect(str(db_path))
    db.execute("""
        CREATE TABLE IF NOT EXISTS tool_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_name TEXT UNIQUE NOT NULL,
            chain_tools TEXT NOT NULL,
            description TEXT,
            use_count INTEGER DEFAULT 0,
            is_auto_detected INTEGER DEFAULT 0,
            embedding_text TEXT,
            last_used_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    db.close()

    with patch("chain_indexer.ANALYTICS_DB_PATH", db_path):
        with patch("chain_indexer.CHAIN_INDEX_PATH", temp_chain_dir / "chains.hnsw"):
            with patch("chain_indexer.DB_DIR", temp_chain_dir):
                indexer = ChainIndexer(
                    embedder=mock_embedder,
                    analytics=None,
                    top_chains_cache_size=5,
                )
                yield indexer
                indexer.close()


@pytest.fixture
def sample_chains():
    """Sample tool chains for testing."""
    return [
        ToolChain(
            id=1,
            name="file_modify",
            tools=["bridge:read_file", "bridge:write_file"],
            description="Read and modify files",
            use_count=10,
            is_auto_detected=False,
            embedding_text="Workflow: file modify | Steps: read, write",
        ),
        ToolChain(
            id=2,
            name="git_commit",
            tools=["bridge:git_status", "bridge:git_add", "bridge:git_commit"],
            description="Git commit workflow",
            use_count=5,
            is_auto_detected=False,
            embedding_text="Workflow: git commit | Steps: status, add, commit",
        ),
        ToolChain(
            id=3,
            name="code_analysis",
            tools=["doc:scan", "doc:report"],
            description="Analyze code and generate report",
            use_count=3,
            is_auto_detected=True,
            embedding_text="Workflow: code analysis | Steps: scan, report",
        ),
    ]


# =============================================================================
# ToolChain Dataclass Tests
# =============================================================================


class TestToolChainDataclass:
    """Test ToolChain dataclass."""

    def test_tool_chain_creation(self):
        """Should create chain with all fields."""
        chain = ToolChain(
            id=1,
            name="test_chain",
            tools=["tool1", "tool2"],
            description="Test workflow",
            use_count=0,
            is_auto_detected=False,
        )

        assert chain.id == 1
        assert chain.name == "test_chain"
        assert chain.tools == ["tool1", "tool2"]
        assert chain.description == "Test workflow"
        assert chain.use_count == 0
        assert chain.is_auto_detected is False
        assert chain.embedding is None

    def test_tool_chain_with_embedding(self):
        """Should store embedding array."""
        embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        chain = ToolChain(
            id=1,
            name="test",
            tools=["tool1"],
            description="Test",
            use_count=0,
            is_auto_detected=False,
            embedding=embedding,
        )

        assert chain.embedding is not None
        assert chain.embedding.shape == (EMBEDDING_DIM,)


class TestChainSearchResult:
    """Test ChainSearchResult dataclass."""

    def test_search_result_creation(self):
        """Should create search result."""
        chain = ToolChain(
            id=1,
            name="test",
            tools=["t1"],
            description="",
            use_count=0,
            is_auto_detected=False,
        )
        result = ChainSearchResult(chain=chain, score=0.85)

        assert result.chain == chain
        assert result.score == 0.85


# =============================================================================
# Embedding Text Generation Tests
# =============================================================================


class TestEmbeddingTextGeneration:
    """Test embedding text generation."""

    def test_create_chain_embedding_text(self, chain_indexer, sample_chains):
        """Should generate rich embedding text."""
        chain = sample_chains[0]  # file_modify

        text = chain_indexer.create_chain_embedding_text(chain)

        assert "Workflow: file modify" in text
        assert "read file" in text.lower() or "read" in text.lower()
        assert "write file" in text.lower() or "write" in text.lower()
        assert chain.description in text

    def test_create_chain_embedding_text_includes_tools(
        self, chain_indexer, sample_chains
    ):
        """Should include tool names in embedding text."""
        chain = sample_chains[1]  # git_commit

        text = chain_indexer.create_chain_embedding_text(chain)

        for tool in chain.tools:
            assert tool in text


# =============================================================================
# Database Loading Tests
# =============================================================================


class TestDatabaseLoading:
    """Test loading chains from database."""

    @pytest.mark.asyncio
    async def test_load_chains_empty_db(self, chain_indexer):
        """Should return empty list for empty database."""
        chains = await chain_indexer.load_chains_from_db()

        assert chains == []

    @pytest.mark.asyncio
    async def test_load_chains_with_data(self, chain_indexer):
        """Should load chains from database."""
        # Insert test data
        db = chain_indexer._get_db()
        db.execute(
            """
            INSERT INTO tool_chains (chain_name, chain_tools, description, use_count, is_auto_detected, embedding_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                "test_chain",
                '["tool1", "tool2"]',
                "Test description",
                5,
                0,
                "test embedding text",
            ),
        )
        db.commit()

        chains = await chain_indexer.load_chains_from_db()

        assert len(chains) == 1
        assert chains[0].name == "test_chain"
        assert chains[0].tools == ["tool1", "tool2"]
        assert chains[0].use_count == 5

    @pytest.mark.asyncio
    async def test_load_chains_sorted_by_use_count(self, chain_indexer):
        """Should return chains sorted by use count descending."""
        db = chain_indexer._get_db()
        db.executemany(
            """
            INSERT INTO tool_chains (chain_name, chain_tools, description, use_count, is_auto_detected)
            VALUES (?, ?, ?, ?, ?)
        """,
            [
                ("low_use", '["t1"]', "", 1, 0),
                ("high_use", '["t2"]', "", 100, 0),
                ("mid_use", '["t3"]', "", 50, 0),
            ],
        )
        db.commit()

        chains = await chain_indexer.load_chains_from_db()

        assert chains[0].name == "high_use"
        assert chains[1].name == "mid_use"
        assert chains[2].name == "low_use"


# =============================================================================
# Index Building Tests
# =============================================================================


class TestIndexBuilding:
    """Test HNSW index building."""

    @pytest.mark.asyncio
    async def test_build_chain_index_empty(self, chain_indexer):
        """Should handle empty chain list."""
        await chain_indexer.build_chain_index([])

        # Should not crash, index may be None or empty

    @pytest.mark.asyncio
    async def test_build_chain_index_with_chains(self, chain_indexer, sample_chains):
        """Should build index from chains."""
        await chain_indexer.build_chain_index(sample_chains)

        assert chain_indexer.index is not None
        assert chain_indexer.index.get_current_count() == len(sample_chains)

    @pytest.mark.asyncio
    async def test_build_chain_index_generates_embeddings(
        self, chain_indexer, mock_embedder
    ):
        """Should generate embeddings for chains without them."""
        chains = [
            ToolChain(
                id=1,
                name="test",
                tools=["tool1"],
                description="Test",
                use_count=0,
                is_auto_detected=False,
                embedding=None,
            )
        ]

        await chain_indexer.build_chain_index(chains)

        # Embedder should have been called
        mock_embedder.embed.assert_called()

    @pytest.mark.asyncio
    async def test_build_chain_index_refreshes_cache(
        self, chain_indexer, sample_chains
    ):
        """Should refresh cache after building."""
        # Insert chains to DB first
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)

        # Cache should be populated
        assert len(chain_indexer._chain_cache) > 0


# =============================================================================
# Index Loading Tests
# =============================================================================


class TestIndexLoading:
    """Test loading existing index."""

    @pytest.mark.asyncio
    async def test_load_chain_index_no_file(self, chain_indexer):
        """Should return False when index file doesn't exist."""
        result = await chain_indexer.load_chain_index()

        assert result is False

    @pytest.mark.asyncio
    async def test_load_chain_index_success(self, chain_indexer, sample_chains):
        """Should load existing index."""
        # First build an index
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)

        # Reset indexer state
        chain_indexer.index = None
        chain_indexer._id_to_chain = {}

        # Load the index
        result = await chain_indexer.load_chain_index()

        assert result is True
        assert chain_indexer.index is not None


# =============================================================================
# Search Tests
# =============================================================================


class TestChainSearch:
    """Test chain search functionality."""

    @pytest.mark.asyncio
    async def test_search_chains_no_index(self, chain_indexer):
        """Should return empty results when no index."""
        results = await chain_indexer.search_chains("file operations")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_chains_basic(self, chain_indexer, sample_chains):
        """Should return search results."""
        # Build index first
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)

        # Search with low confidence to ensure we get results
        results = await chain_indexer.search_chains(
            "file operations", top_k=3, min_confidence=0.0
        )

        # Results may be empty with mock embeddings, just verify types
        assert isinstance(results, list)
        assert all(isinstance(r, ChainSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_chains_respects_top_k(self, chain_indexer, sample_chains):
        """Should limit results to top_k."""
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)

        results = await chain_indexer.search_chains("workflow", top_k=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_chains_filters_by_confidence(
        self, chain_indexer, sample_chains
    ):
        """Should filter results below min_confidence."""
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)

        results = await chain_indexer.search_chains("random query", min_confidence=0.99)

        # All results should be above threshold
        for r in results:
            assert r.score >= 0.99


# =============================================================================
# Chain Management Tests
# =============================================================================


class TestChainManagement:
    """Test adding, retrieving, and managing chains."""

    @pytest.mark.asyncio
    async def test_add_chain(self, chain_indexer):
        """Should add chain to database."""
        chain = await chain_indexer.add_chain(
            name="new_workflow",
            tools=["tool1", "tool2", "tool3"],
            description="A new workflow",
            is_auto_detected=False,
        )

        assert chain.name == "new_workflow"
        assert chain.tools == ["tool1", "tool2", "tool3"]
        assert chain.id is not None

    @pytest.mark.asyncio
    async def test_add_chain_generates_description(self, chain_indexer):
        """Should generate description if not provided."""
        chain = await chain_indexer.add_chain(
            name="auto_desc",
            tools=["bridge:read_file", "bridge:write_file"],
        )

        assert chain.description
        assert (
            "read" in chain.description.lower() or "write" in chain.description.lower()
        )

    @pytest.mark.asyncio
    async def test_add_chain_to_existing_index(self, chain_indexer, sample_chains):
        """Should add chain to existing index."""
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.build_chain_index(sample_chains)
        initial_count = chain_indexer.index.get_current_count()

        await chain_indexer.add_chain(
            name="new_chain",
            tools=["new_tool"],
            description="New chain",
        )

        # Index should have one more item
        assert chain_indexer.index.get_current_count() == initial_count + 1

    @pytest.mark.asyncio
    async def test_get_chain_not_found(self, chain_indexer):
        """Should return None for unknown chain."""
        result = await chain_indexer.get_chain("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_chain_from_db(self, chain_indexer):
        """Should retrieve chain from database."""
        db = chain_indexer._get_db()
        db.execute(
            """
            INSERT INTO tool_chains (chain_name, chain_tools, description, use_count, is_auto_detected)
            VALUES (?, ?, ?, ?, ?)
        """,
            ("db_chain", '["tool1"]', "From DB", 5, 0),
        )
        db.commit()

        result = await chain_indexer.get_chain("db_chain")

        assert result is not None
        assert result.name == "db_chain"
        assert result.use_count == 5

    @pytest.mark.asyncio
    async def test_get_chain_from_cache(self, chain_indexer, sample_chains):
        """Should retrieve chain from cache first."""
        chain_indexer._chain_cache = sample_chains

        result = await chain_indexer.get_chain("file_modify")

        assert result is not None
        assert result.name == "file_modify"


# =============================================================================
# Usage Tracking Tests
# =============================================================================


class TestUsageTracking:
    """Test chain usage recording."""

    @pytest.mark.asyncio
    async def test_record_chain_use(self, chain_indexer):
        """Should increment use count."""
        db = chain_indexer._get_db()
        db.execute(
            """
            INSERT INTO tool_chains (chain_name, chain_tools, description, use_count, is_auto_detected)
            VALUES (?, ?, ?, ?, ?)
        """,
            ("tracked_chain", '["tool1"]', "Test", 5, 0),
        )
        db.commit()

        await chain_indexer.record_chain_use("tracked_chain")

        # Check updated count
        row = db.execute(
            "SELECT use_count FROM tool_chains WHERE chain_name = ?", ("tracked_chain",)
        ).fetchone()
        assert row[0] == 6

    @pytest.mark.asyncio
    async def test_record_chain_use_updates_cache(self, chain_indexer, sample_chains):
        """Should update cached chain count."""
        chain_indexer._chain_cache = sample_chains
        original_count = sample_chains[0].use_count

        await chain_indexer.record_chain_use("file_modify")

        assert chain_indexer._chain_cache[0].use_count == original_count + 1


# =============================================================================
# Cache Tests
# =============================================================================


class TestChainCache:
    """Test chain caching functionality."""

    @pytest.mark.asyncio
    async def test_refresh_chain_cache(self, chain_indexer, sample_chains):
        """Should populate cache with top chains."""
        db = chain_indexer._get_db()
        for chain in sample_chains:
            db.execute(
                """
                INSERT INTO tool_chains (id, chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    chain.id,
                    chain.name,
                    json.dumps(chain.tools),
                    chain.description,
                    chain.use_count,
                    chain.is_auto_detected,
                ),
            )
        db.commit()

        await chain_indexer.refresh_chain_cache()

        assert len(chain_indexer._chain_cache) > 0
        # Should be sorted by use_count
        counts = [c.use_count for c in chain_indexer._chain_cache]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.asyncio
    async def test_refresh_chain_cache_respects_limit(self, chain_indexer):
        """Should limit cache to top_chains_cache_size."""
        db = chain_indexer._get_db()
        for i in range(10):
            db.execute(
                """
                INSERT INTO tool_chains (chain_name, chain_tools, description, use_count, is_auto_detected)
                VALUES (?, ?, ?, ?, ?)
            """,
                (f"chain_{i}", '["tool"]', f"Chain {i}", i, 0),
            )
        db.commit()

        await chain_indexer.refresh_chain_cache()

        assert len(chain_indexer._chain_cache) <= chain_indexer.top_chains_cache_size

    def test_get_cached_chains(self, chain_indexer, sample_chains):
        """Should return cached chains."""
        chain_indexer._chain_cache = sample_chains

        result = chain_indexer.get_cached_chains()

        assert result == sample_chains


# =============================================================================
# Default Chains Tests
# =============================================================================


class TestDefaultChains:
    """Test seeding default chains."""

    @pytest.mark.asyncio
    async def test_seed_default_chains(self, chain_indexer):
        """Should seed predefined chains."""
        await chain_indexer.seed_default_chains()

        chains = await chain_indexer.load_chains_from_db()

        # Should have created some chains
        assert len(chains) > 0

        # Check for expected chains
        names = [c.name for c in chains]
        expected_names = ["file_modify", "git_commit", "code_analysis"]
        for name in expected_names:
            assert name in names

    @pytest.mark.asyncio
    async def test_seed_default_chains_idempotent(self, chain_indexer):
        """Should not duplicate chains on repeated calls."""
        await chain_indexer.seed_default_chains()
        first_count = len(await chain_indexer.load_chains_from_db())

        await chain_indexer.seed_default_chains()
        second_count = len(await chain_indexer.load_chains_from_db())

        assert first_count == second_count


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Test cleanup operations."""

    def test_close(self, chain_indexer):
        """Should close database connection."""
        db = chain_indexer._get_db()
        assert db is not None

        chain_indexer.close()

        assert chain_indexer._db is None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern."""

    def test_get_chain_indexer_creates_instance(self, mock_embedder, temp_chain_dir):
        """Should create instance on first call."""
        import chain_indexer as ci

        ci._chain_indexer_instance = None

        db_path = temp_chain_dir / "test.db"
        db = sqlite3.connect(str(db_path))
        db.execute("""
            CREATE TABLE IF NOT EXISTS tool_chains (
                id INTEGER PRIMARY KEY,
                chain_name TEXT UNIQUE,
                chain_tools TEXT,
                description TEXT,
                use_count INTEGER DEFAULT 0,
                is_auto_detected INTEGER DEFAULT 0,
                embedding_text TEXT,
                last_used_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()
        db.close()

        with patch("chain_indexer.ANALYTICS_DB_PATH", db_path):
            with patch(
                "chain_indexer.CHAIN_INDEX_PATH", temp_chain_dir / "chains.hnsw"
            ):
                with patch("chain_indexer.DB_DIR", temp_chain_dir):
                    indexer = get_chain_indexer(mock_embedder, None)

                    assert indexer is not None
                    assert isinstance(indexer, ChainIndexer)

                    indexer.close()

    def test_get_chain_indexer_returns_same_instance(
        self, mock_embedder, temp_chain_dir
    ):
        """Should return same instance on subsequent calls."""
        import chain_indexer as ci

        ci._chain_indexer_instance = None

        db_path = temp_chain_dir / "test.db"
        db = sqlite3.connect(str(db_path))
        db.execute("""
            CREATE TABLE IF NOT EXISTS tool_chains (
                id INTEGER PRIMARY KEY,
                chain_name TEXT UNIQUE,
                chain_tools TEXT,
                description TEXT,
                use_count INTEGER DEFAULT 0,
                is_auto_detected INTEGER DEFAULT 0,
                embedding_text TEXT,
                last_used_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()
        db.close()

        with patch("chain_indexer.ANALYTICS_DB_PATH", db_path):
            with patch(
                "chain_indexer.CHAIN_INDEX_PATH", temp_chain_dir / "chains.hnsw"
            ):
                with patch("chain_indexer.DB_DIR", temp_chain_dir):
                    indexer1 = get_chain_indexer(mock_embedder, None)
                    indexer2 = get_chain_indexer(mock_embedder, None)

                    assert indexer1 is indexer2

                    indexer1.close()
