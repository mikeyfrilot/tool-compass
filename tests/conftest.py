"""
Tool Compass - Pytest Fixtures and Configuration

Provides shared fixtures for testing the semantic search gateway.
Based on FastMCP testing best practices:
https://gofastmcp.com/patterns/testing
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CompassConfig, StdioBackend
from tool_manifest import ToolDefinition


# =============================================================================
# Async Support
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_config() -> CompassConfig:
    """Minimal test configuration without real backends."""
    return CompassConfig(
        backends={},  # No backends for unit tests
        embedding_model="nomic-embed-text",
        ollama_url="http://localhost:11434",
        index_dir="./test_db",
        auto_sync=False,
        default_top_k=5,
        min_confidence=0.3,
        progressive_disclosure=True,
        analytics_enabled=False,
        chain_indexing_enabled=False,
    )


@pytest.fixture
def test_config_with_backends() -> CompassConfig:
    """Test configuration with mock backend definitions."""
    return CompassConfig(
        backends={
            "test_backend": StdioBackend(
                command="python",
                args=["-c", "print('mock')"],
                env={},
            ),
        },
        auto_sync=False,
        analytics_enabled=True,
        chain_indexing_enabled=True,
    )


# =============================================================================
# Tool Definition Fixtures
# =============================================================================


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    """Sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="test:read_file",
            description="Read contents of a file from disk",
            category="file",
            server="test",
            parameters={"filepath": "str"},
            examples=["read file", "get file contents", "open document"],
            is_core=True,
        ),
        ToolDefinition(
            name="test:write_file",
            description="Write content to a file on disk",
            category="file",
            server="test",
            parameters={"filepath": "str", "content": "str"},
            examples=["write file", "save content", "create file"],
            is_core=True,
        ),
        ToolDefinition(
            name="test:git_status",
            description="Show the working tree status",
            category="git",
            server="test",
            parameters={"repo_path": "str?"},
            examples=["git status", "show changes", "list modifications"],
            is_core=False,
        ),
        ToolDefinition(
            name="test:generate_image",
            description="Generate an image from a text prompt using AI",
            category="ai",
            server="test",
            parameters={"prompt": "str", "size": "str?"},
            examples=["create image", "text to image", "generate artwork"],
            is_core=False,
        ),
        ToolDefinition(
            name="test:search_docs",
            description="Search through document collection",
            category="search",
            server="test",
            parameters={"query": "str", "limit": "int?"},
            examples=["search documents", "find text", "lookup content"],
            is_core=False,
        ),
    ]


# =============================================================================
# Mock Embedder
# =============================================================================


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic vectors."""
    embedder = Mock()

    # Create deterministic embeddings based on text hash
    def mock_embed(text: str) -> np.ndarray:
        # Use hash of text to create reproducible embedding
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        vec = np.random.randn(768).astype(np.float32)
        return vec / np.linalg.norm(vec)  # Normalize

    async def async_embed(text: str) -> np.ndarray:
        return mock_embed(text)

    async def async_embed_batch(texts: list[str]) -> np.ndarray:
        return np.array([mock_embed(t) for t in texts])

    async def async_embed_query(query: str) -> np.ndarray:
        return mock_embed(f"search_query: {query}")

    async def health_check() -> bool:
        return True

    embedder.embed = AsyncMock(side_effect=async_embed)
    embedder.embed_batch = AsyncMock(side_effect=async_embed_batch)
    embedder.embed_query = AsyncMock(side_effect=async_embed_query)
    embedder.health_check = AsyncMock(side_effect=health_check)
    embedder.close = AsyncMock()

    return embedder


# =============================================================================
# Temporary Database Fixtures
# =============================================================================


@pytest.fixture
def temp_db_dir(tmp_path: Path) -> Path:
    """Temporary directory for test databases."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


@pytest.fixture
def temp_index_path(temp_db_dir: Path) -> Path:
    """Path for temporary HNSW index."""
    return temp_db_dir / "test_compass.hnsw"


@pytest.fixture
def temp_db_path(temp_db_dir: Path) -> Path:
    """Path for temporary SQLite database."""
    return temp_db_dir / "test_tools.db"


# =============================================================================
# Index Fixtures
# =============================================================================


@pytest.fixture
async def test_index(temp_index_path, temp_db_path, mock_embedder, sample_tools):
    """Pre-built test index with sample tools."""
    from indexer import CompassIndex

    index = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=mock_embedder,
    )

    await index.build_index(sample_tools)

    yield index

    await index.close()


# =============================================================================
# Analytics Fixtures
# =============================================================================


@pytest.fixture
def temp_analytics_db(temp_db_dir: Path) -> Path:
    """Path for temporary analytics database."""
    return temp_db_dir / "test_analytics.db"


@pytest.fixture
def test_analytics(temp_analytics_db):
    """Test analytics instance with temp database."""
    from analytics import CompassAnalytics

    analytics = CompassAnalytics(
        db_path=temp_analytics_db,
        hot_cache_size=5,
        chain_min_occurrences=2,
    )

    yield analytics

    analytics.close()


# =============================================================================
# Backend Client Fixtures
# =============================================================================


@pytest.fixture
def mock_backend_manager():
    """Mock BackendManager for gateway tests."""
    manager = Mock()
    manager._backends = {}
    manager.connect_backend = AsyncMock(return_value=True)
    manager.connect_all = AsyncMock(return_value={})
    manager.disconnect_all = AsyncMock()
    manager.get_all_tools = Mock(return_value=[])
    manager.get_backend_tools = Mock(return_value=[])
    manager.get_tool_schema = Mock(return_value=None)
    manager.execute_tool = AsyncMock(return_value={"success": True, "result": "ok"})
    manager.get_stats = Mock(
        return_value={
            "configured_backends": [],
            "connected_backends": [],
            "total_tools": 0,
            "tools_by_backend": {},
        }
    )

    return manager


# =============================================================================
# Chain Indexer Fixtures
# =============================================================================


@pytest.fixture
def temp_chain_db(temp_db_dir: Path) -> Path:
    """Path for temporary chain database."""
    import sqlite3

    db_path = temp_db_dir / "test_chains.db"

    # Create required tables
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

    return db_path


@pytest.fixture
def test_chain_indexer(mock_embedder, temp_chain_db, temp_db_dir):
    """Test chain indexer with temp storage."""
    from chain_indexer import ChainIndexer

    chain_index_path = temp_db_dir / "test_chains.hnsw"

    with patch("chain_indexer.ANALYTICS_DB_PATH", temp_chain_db):
        with patch("chain_indexer.CHAIN_INDEX_PATH", chain_index_path):
            with patch("chain_indexer.DB_DIR", temp_db_dir):
                indexer = ChainIndexer(
                    embedder=mock_embedder,
                    analytics=None,
                    top_chains_cache_size=5,
                )
                yield indexer
                indexer.close()


# =============================================================================
# Sync Manager Fixtures
# =============================================================================


@pytest.fixture
def temp_sync_db(temp_db_dir: Path) -> Path:
    """Path for temporary sync state database."""
    return temp_db_dir / "test_sync.db"


@pytest.fixture
def test_sync_manager(
    test_config,
    mock_backend_manager,
    temp_sync_db,
    temp_index_path,
    temp_db_path,
    mock_embedder,
    sample_tools,
):
    """Test sync manager with mocks."""
    from sync_manager import SyncManager
    from indexer import CompassIndex

    # Create a real index for sync manager
    index = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=mock_embedder,
    )

    with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
        manager = SyncManager(
            config=test_config,
            index=index,
            backends=mock_backend_manager,
        )
        yield manager
        manager.close()


# =============================================================================
# Integration Test Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services (Ollama)"
    )
    config.addinivalue_line("markers", "slow: marks tests that take a long time to run")
