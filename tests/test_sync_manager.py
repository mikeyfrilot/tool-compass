"""
Tests for Tool Compass sync manager module.

Tests backend change detection, hash computation, and sync operations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from sync_manager import SyncManager, get_sync_manager
from config import CompassConfig, StdioBackend
from backend_client_simple import ToolInfo


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_sync_db(tmp_path):
    """Temporary database for sync state."""
    return tmp_path / "test_sync.db"


@pytest.fixture
def mock_config():
    """Mock configuration with backends."""
    return CompassConfig(
        backends={
            "backend1": StdioBackend(command="python", args=["-m", "server1"]),
            "backend2": StdioBackend(command="python", args=["-m", "server2"]),
        }
    )


@pytest.fixture
def mock_index():
    """Mock CompassIndex with production-shape sync + async split.

    TS-B-001 fix: previously this fixture was a bare ``AsyncMock()``, which
    made *every* attribute access on the mock return an awaitable. That broke
    the diff-sync code path (``_get_backend_tool_names``) which is SYNCHRONOUS
    and calls ``self.index.db.execute(...).fetchall()`` — under AsyncMock,
    that chain returned unawaited coroutines, raising RuntimeWarning at every
    sync_manager test and silently bypassing the IDX-FT-004 'removed tools'
    branch entirely.

    Fix: use Mock(spec=CompassIndex) so attribute access matches the real API
    surface (and unknown attributes raise AttributeError, not return another
    Mock). The two known-async methods are explicitly AsyncMock; the sync
    .db handle has a real execute(...).fetchall() chain that returns [].
    """
    from indexer import CompassIndex

    index = Mock(spec=CompassIndex)
    index.build_index = AsyncMock(return_value={"tools_indexed": 5})
    index.add_single_tool = AsyncMock(return_value=True)

    # Sync .db handle — mirrors sqlite3.Connection.execute(...).fetchall().
    # _get_backend_tool_names iterates the rows; default to no prior tools.
    db = Mock()
    db.execute.return_value.fetchall.return_value = []
    index.db = db
    return index


@pytest.fixture
def mock_backends():
    """Mock BackendManager."""
    backends = Mock()
    backends._backends = {}
    backends.connect_backend = AsyncMock(return_value=True)
    backends.connect_all = AsyncMock(return_value={"backend1": True, "backend2": True})
    backends.get_backend_tools = Mock(return_value=[])
    return backends


@pytest.fixture
def sync_manager(mock_config, mock_index, mock_backends, temp_sync_db):
    """Create sync manager with mocks."""
    with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
        manager = SyncManager(
            config=mock_config,
            index=mock_index,
            backends=mock_backends,
        )
        yield manager
        manager.close()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSyncManagerInit:
    """Test SyncManager initialization."""

    def test_init_with_dependencies(
        self, mock_config, mock_index, mock_backends, temp_sync_db
    ):
        """Should initialize with all dependencies."""
        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(mock_config, mock_index, mock_backends)

            assert manager.config == mock_config
            assert manager.index == mock_index
            assert manager.backends == mock_backends
            assert manager._db is None  # Lazy init
            assert manager._polling_task is None

            manager.close()

    def test_db_directory_created(
        self, mock_config, mock_index, mock_backends, tmp_path
    ):
        """Should create db directory if not exists."""
        db_path = tmp_path / "subdir" / "test.db"

        with patch("sync_manager.ANALYTICS_DB_PATH", db_path):
            manager = SyncManager(mock_config, mock_index, mock_backends)
            # Access db to trigger creation
            manager._get_db()
            assert db_path.parent.exists()
            manager.close()


# =============================================================================
# Database Tests
# =============================================================================


class TestSyncManagerDatabase:
    """Test database operations."""

    def test_get_db_creates_connection(self, sync_manager):
        """Should create database connection on first access."""
        assert sync_manager._db is None

        db = sync_manager._get_db()

        assert db is not None
        assert sync_manager._db is db

    def test_init_sync_table_creates_schema(self, sync_manager):
        """Should create sync state table."""
        db = sync_manager._get_db()

        # Check table exists
        cursor = db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='backend_sync_state'
        """)
        assert cursor.fetchone() is not None

    def test_init_sync_table_has_columns(self, sync_manager):
        """Should have expected columns."""
        db = sync_manager._get_db()

        cursor = db.execute("PRAGMA table_info(backend_sync_state)")
        columns = {row[1] for row in cursor.fetchall()}

        expected = {
            "backend_name",
            "tool_count",
            "tool_hash",
            "last_sync_at",
            "sync_status",
        }
        assert expected.issubset(columns)

    def test_close_releases_connection(self, sync_manager):
        """Should close database connection."""
        db = sync_manager._get_db()
        assert db is not None

        sync_manager.close()

        assert sync_manager._db is None


# =============================================================================
# Hash Computation Tests
# =============================================================================


class TestHashComputation:
    """Test tool hash computation."""

    def test_compute_hash_empty_list(self, sync_manager):
        """Should handle empty tool list."""
        hash_val = sync_manager._compute_tool_hash([])

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32
        # Empty hash should be consistent
        assert hash_val == sync_manager._compute_tool_hash([])

    def test_compute_hash_single_tool(self, sync_manager):
        """Should compute hash for single tool."""
        tools = [ToolInfo("read", "backend:read", "Read", "backend", {})]

        hash_val = sync_manager._compute_tool_hash(tools)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32

    def test_compute_hash_deterministic(self, sync_manager):
        """Should produce same hash for same tools."""
        tools = [
            ToolInfo("read", "backend:read", "Read", "backend", {}),
            ToolInfo("write", "backend:write", "Write", "backend", {}),
        ]

        hash1 = sync_manager._compute_tool_hash(tools)
        hash2 = sync_manager._compute_tool_hash(tools)

        assert hash1 == hash2

    def test_compute_hash_order_independent(self, sync_manager):
        """Should produce same hash regardless of order (sorted internally)."""
        tools1 = [
            ToolInfo("read", "backend:read", "Read", "backend", {}),
            ToolInfo("write", "backend:write", "Write", "backend", {}),
        ]
        tools2 = [
            ToolInfo("write", "backend:write", "Write", "backend", {}),
            ToolInfo("read", "backend:read", "Read", "backend", {}),
        ]

        hash1 = sync_manager._compute_tool_hash(tools1)
        hash2 = sync_manager._compute_tool_hash(tools2)

        assert hash1 == hash2

    def test_compute_hash_different_for_different_tools(self, sync_manager):
        """Should produce different hashes for different tools."""
        tools1 = [ToolInfo("read", "backend:read", "Read", "backend", {})]
        tools2 = [ToolInfo("write", "backend:write", "Write", "backend", {})]

        hash1 = sync_manager._compute_tool_hash(tools1)
        hash2 = sync_manager._compute_tool_hash(tools2)

        assert hash1 != hash2


# =============================================================================
# Stored Hash Tests
# =============================================================================


class TestStoredHash:
    """Test stored hash operations."""

    @pytest.mark.asyncio
    async def test_get_stored_hash_not_found(self, sync_manager):
        """Should return None for unknown backend."""
        result = await sync_manager.get_stored_hash("unknown_backend")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stored_hash_found(self, sync_manager):
        """Should return stored hash."""
        # Insert test data
        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("test_backend", 5, "abc123hash"),
        )
        db.commit()

        result = await sync_manager.get_stored_hash("test_backend")

        assert result == "abc123hash"


# =============================================================================
# Change Detection Tests
# =============================================================================


class TestChangeDetection:
    """Test backend change detection."""

    @pytest.mark.asyncio
    async def test_check_backend_changes_not_connected(self, sync_manager):
        """Should try to connect if not connected."""
        sync_manager.backends.is_backend_connected = Mock(return_value=False)
        sync_manager.backends.connect_backend = AsyncMock(return_value=False)

        result = await sync_manager.check_backend_changes("backend1")

        assert result is False
        sync_manager.backends.connect_backend.assert_called_once_with("backend1")

    @pytest.mark.asyncio
    async def test_check_backend_changes_no_tools(self, sync_manager):
        """Should return False when backend has no tools."""
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=[])

        result = await sync_manager.check_backend_changes("backend1")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_backend_changes_first_sync(self, sync_manager):
        """Should detect changes on first sync (no stored hash)."""
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )

        result = await sync_manager.check_backend_changes("backend1")

        assert result is True  # Changed because no previous hash

    @pytest.mark.asyncio
    async def test_check_backend_changes_no_change(self, sync_manager):
        """Should detect no changes when hash matches."""
        tools = [ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {})]
        stored_hash = sync_manager._compute_tool_hash(tools)

        # Store the hash
        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("backend1", 1, stored_hash),
        )
        db.commit()

        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=tools)

        result = await sync_manager.check_backend_changes("backend1")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_backend_changes_tool_added(self, sync_manager):
        """Should detect when tools are added."""
        old_tools = [ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {})]
        new_tools = [
            ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ToolInfo("tool2", "backend1:tool2", "Tool 2", "backend1", {}),
        ]
        stored_hash = sync_manager._compute_tool_hash(old_tools)

        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("backend1", 1, stored_hash),
        )
        db.commit()

        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=new_tools)

        result = await sync_manager.check_backend_changes("backend1")

        assert result is True


# =============================================================================
# Tool Categorization Tests
# =============================================================================


class TestToolCategorization:
    """Test automatic tool categorization."""

    @pytest.fixture
    def categorize(self, sync_manager):
        """Get categorization function."""
        return sync_manager._categorize_tool

    def test_categorize_file_tools(self, categorize):
        """Should categorize file-related tools."""
        assert categorize("read_file", "Read file contents") == "file"
        assert categorize("write_data", "Write data to disk") == "file"
        assert categorize("list_directory", "List directory") == "file"

    def test_categorize_git_tools(self, categorize):
        """Should categorize git-related tools."""
        assert categorize("git_status", "Show git status") == "git"
        assert categorize("commit_changes", "Commit changes") == "git"
        assert categorize("create_branch", "Create branch") == "git"

    def test_categorize_database_tools(self, categorize):
        """Should categorize database tools."""
        assert categorize("db_query", "Execute query") == "database"
        assert categorize("sql_execute", "Run SQL") == "database"
        assert categorize("database_stats", "Get stats") == "database"

    def test_categorize_search_tools(self, categorize):
        """Should categorize search tools."""
        assert categorize("search_docs", "Search documents") == "search"
        # Note: "find_files" matches "file" first due to ordering in _categorize_tool
        assert categorize("find_text", "Find text") == "search"
        assert categorize("lookup_value", "Lookup value") == "search"

    def test_categorize_ai_tools(self, categorize):
        """Should categorize AI tools."""
        assert categorize("generate_image", "Generate image") == "ai"
        assert categorize("comfy_status", "ComfyUI status") == "ai"
        assert categorize("video_create", "Create video") == "ai"

    def test_categorize_analysis_tools(self, categorize):
        """Should categorize analysis tools."""
        assert categorize("scan_code", "Scan for issues") == "analysis"
        assert categorize("analyze_data", "Analyze data") == "analysis"
        assert categorize("health_check", "Health check") == "analysis"

    def test_categorize_other(self, categorize):
        """Should default to 'other' for unknown tools."""
        assert categorize("mysterious_tool", "Does something") == "other"


# =============================================================================
# Sync If Needed Tests
# =============================================================================


class TestSyncIfNeeded:
    """Test sync_if_needed method."""

    @pytest.mark.asyncio
    async def test_sync_if_needed_no_changes(self, sync_manager):
        """Should report unchanged when no changes detected."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = False

            results = await sync_manager.sync_if_needed()

            assert results["backend1"] == "unchanged"
            assert results["backend2"] == "unchanged"

    @pytest.mark.asyncio
    async def test_sync_if_needed_with_changes(self, sync_manager):
        """Should sync changed backends."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = [
                True,
                False,
            ]  # backend1 changed, backend2 unchanged

            with patch.object(
                sync_manager, "_rebuild_for_backends", new_callable=AsyncMock
            ) as mock_rebuild:
                results = await sync_manager.sync_if_needed()

                mock_rebuild.assert_called_once_with(["backend1"])
                assert results["backend1"] == "synced"
                assert results["backend2"] == "unchanged"

    @pytest.mark.asyncio
    async def test_sync_if_needed_handles_errors(self, sync_manager):
        """Should report errors gracefully."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = Exception("Connection failed")

            results = await sync_manager.sync_if_needed()

            assert "error" in results["backend1"]
            assert "error" in results["backend2"]


# =============================================================================
# Full Sync Tests
# =============================================================================


class TestFullSync:
    """Test full_sync method."""

    @pytest.mark.asyncio
    async def test_full_sync_no_backends(self, sync_manager):
        """Should handle no connected backends.

        MSYNC-A-001 (masking-test repair): the empty branch must STILL call
        build_index([]) so the prior index is genuinely cleared. Previously
        this test only asserted status/tools_indexed — both of which were
        already correct while the index was left live — so it never failed on
        the bug. Now we assert build_index was actually invoked with an empty
        tool list.
        """
        sync_manager.backends.connect_all = AsyncMock(return_value={})

        result = await sync_manager.full_sync()

        assert result["status"] == "no_tools"
        assert result["tools_indexed"] == 0
        # The index must be emptied — build_index([]) clears tools + HNSW.
        sync_manager.index.build_index.assert_called_once()
        called_arg = sync_manager.index.build_index.call_args.args[0]
        assert called_arg == [], "empty full_sync must rebuild with []"

    @pytest.mark.asyncio
    async def test_full_sync_with_tools(self, sync_manager):
        """Should sync all tools from connected backends."""
        sync_manager.backends.connect_all = AsyncMock(return_value={"backend1": True})
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo(
                    "tool1",
                    "backend1:tool1",
                    "Tool 1",
                    "backend1",
                    {"type": "object", "properties": {"arg": {"type": "string"}}},
                ),
                ToolInfo("tool2", "backend1:tool2", "Tool 2", "backend1", {}),
            ]
        )

        result = await sync_manager.full_sync()

        assert result["status"] == "complete"
        assert result["tools_indexed"] == 2
        sync_manager.index.build_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_sync_all_backends_gone_empties_index(
        self, mock_config, mock_backends, temp_sync_db, tmp_path
    ):
        """MSYNC-A-001 (full-mechanism regression): when every backend is gone,
        the index must be GENUINELY emptied — not just reported as no_tools.

        This drives a REAL CompassIndex (build_index([]) needs no embedder for
        the empty branch) so we observe the actual side effect: the SQLite
        tools table is truncated. Pre-fix, full_sync returned no_tools WITHOUT
        calling build_index([]), leaving stale rows live — compass kept routing
        to dead tools. This test fails against that bug.
        """
        from indexer import CompassIndex

        # Real index pointed at temp files; backends report ZERO tools (all
        # disconnected/empty) but connect_all says they connected.
        index = CompassIndex(
            index_path=tmp_path / "compass.hnsw",
            db_path=tmp_path / "tools.db",
        )
        index._init_db()

        # Seed two stale tool rows as if a prior sync had populated them.
        with index._db_write_lock:
            index.db.execute(
                "INSERT INTO tools (name, description, category, server, "
                "parameters, examples, is_core, embedding_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("backend1:dead", "stale", "other", "backend1", "{}", "[]", 0, "x"),
            )
            index.db.execute(
                "INSERT INTO tools (name, description, category, server, "
                "parameters, examples, is_core, embedding_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("backend2:dead", "stale", "other", "backend2", "{}", "[]", 0, "x"),
            )
            index.db.commit()

        pre = index.db.execute("SELECT COUNT(*) AS c FROM tools").fetchone()["c"]
        assert pre == 2, "precondition: stale rows present"

        mock_backends.connect_all = AsyncMock(
            return_value={"backend1": True, "backend2": True}
        )
        mock_backends.get_backend_tools = Mock(return_value=[])  # all gone

        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(mock_config, index, mock_backends)
            try:
                result = await manager.full_sync()
            finally:
                manager.close()

        assert result["status"] == "no_tools"
        assert result["tools_indexed"] == 0

        # The actual fix: the tools table is genuinely empty now.
        post = index.db.execute("SELECT COUNT(*) AS c FROM tools").fetchone()["c"]
        assert post == 0, "all-backends-gone full_sync must empty the index"
        await index.close()


# =============================================================================
# Sync Status Tests
# =============================================================================


class TestSyncStatus:
    """Test get_sync_status method."""

    @pytest.mark.asyncio
    async def test_get_sync_status_empty(self, sync_manager):
        """Should show never synced for new backends."""
        result = await sync_manager.get_sync_status()

        assert "backends" in result
        assert result["backends"]["backend1"]["status"] == "never_synced"
        assert result["backends"]["backend2"]["status"] == "never_synced"

    @pytest.mark.asyncio
    async def test_get_sync_status_with_data(self, sync_manager):
        """Should show sync status from database."""
        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("backend1", 5, "abc123def456"),
        )
        db.commit()

        result = await sync_manager.get_sync_status()

        assert result["backends"]["backend1"]["tool_count"] == 5
        assert result["backends"]["backend1"]["status"] == "synced"
        assert result["backends"]["backend1"]["tool_hash"] == "abc123de..."  # Truncated

    @pytest.mark.asyncio
    async def test_get_sync_status_polling_status(self, sync_manager):
        """Should report polling status."""
        result = await sync_manager.get_sync_status()

        assert result["polling_active"] is False


# =============================================================================
# Background Polling Tests
# =============================================================================


class TestBackgroundPolling:
    """Test background polling functionality."""

    @pytest.mark.asyncio
    async def test_start_background_polling(self, sync_manager):
        """Should start polling task."""
        await sync_manager.start_background_polling(interval_seconds=1)

        assert sync_manager._polling_task is not None
        assert not sync_manager._polling_task.done()

        await sync_manager.stop_background_polling()

    @pytest.mark.asyncio
    async def test_start_polling_already_running(self, sync_manager):
        """Should warn if polling already running."""
        await sync_manager.start_background_polling(interval_seconds=1)
        first_task = sync_manager._polling_task

        await sync_manager.start_background_polling(interval_seconds=1)

        # Should still be the same task
        assert sync_manager._polling_task is first_task

        await sync_manager.stop_background_polling()

    @pytest.mark.asyncio
    async def test_stop_background_polling(self, sync_manager):
        """Should stop and clean up polling task."""
        await sync_manager.start_background_polling(interval_seconds=1)

        await sync_manager.stop_background_polling()

        assert sync_manager._polling_task is None

    @pytest.mark.asyncio
    async def test_stop_polling_not_running(self, sync_manager):
        """Should handle stopping when not running."""
        await sync_manager.stop_background_polling()

        # Should not raise


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test singleton pattern."""

    def test_get_sync_manager_creates_instance(
        self, mock_config, mock_index, mock_backends, temp_sync_db
    ):
        """Should create instance on first call."""
        import sync_manager as sm

        sm._sync_manager_instance = None

        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = get_sync_manager(mock_config, mock_index, mock_backends)

            assert manager is not None
            assert isinstance(manager, SyncManager)

            manager.close()

    def test_get_sync_manager_returns_same_instance(
        self, mock_config, mock_index, mock_backends, temp_sync_db
    ):
        """Should return same instance on subsequent calls."""
        import sync_manager as sm

        sm._sync_manager_instance = None

        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager1 = get_sync_manager(mock_config, mock_index, mock_backends)
            manager2 = get_sync_manager(mock_config, mock_index, mock_backends)

            assert manager1 is manager2

            manager1.close()


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestRebuildForBackends:
    """Test _rebuild_for_backends method."""

    @pytest.mark.asyncio
    async def test_rebuild_with_tools(self, sync_manager):
        """Should rebuild index with tools from backends."""
        tools = [
            ToolInfo(
                "tool1",
                "backend1:tool1",
                "Tool 1",
                "backend1",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            ToolInfo("tool2", "backend1:tool2", "Tool 2", "backend1", {}),
        ]
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=tools)
        sync_manager.index.add_single_tool = AsyncMock()

        await sync_manager._rebuild_for_backends(["backend1"])

        # Should have added tools to index
        assert sync_manager.index.add_single_tool.call_count == 2

        # Should have updated database
        db = sync_manager._get_db()
        cursor = db.execute(
            "SELECT * FROM backend_sync_state WHERE backend_name = ?", ("backend1",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == 2  # tool_count

    @pytest.mark.asyncio
    async def test_rebuild_no_tools(self, sync_manager):
        """Should handle backend with no tools."""
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=[])
        sync_manager.index.add_single_tool = AsyncMock()

        await sync_manager._rebuild_for_backends(["backend1"])

        # Should not call add_single_tool
        sync_manager.index.add_single_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_rebuild_with_list_param_type(self, sync_manager):
        """Should handle list parameter types (e.g., [string, null])."""
        tools = [
            ToolInfo(
                "tool1",
                "backend1:tool1",
                "Tool 1",
                "backend1",
                {
                    "type": "object",
                    "properties": {"optional_arg": {"type": ["string", "null"]}},
                },
            ),
        ]
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=tools)
        sync_manager.index.add_single_tool = AsyncMock()

        await sync_manager._rebuild_for_backends(["backend1"])

        # Should have added tool
        sync_manager.index.add_single_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebuild_without_qualified_name(self, sync_manager):
        """Should handle tools without colon in qualified name."""
        tools = [
            ToolInfo("simple_tool", "simple_tool", "Simple tool", "backend1", {}),
        ]
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=tools)
        sync_manager.index.add_single_tool = AsyncMock()

        await sync_manager._rebuild_for_backends(["backend1"])

        sync_manager.index.add_single_tool.assert_called_once()


class TestOrphanVectorCompaction:
    """MSYNC-A-002: periodic full rebuild to reclaim orphaned HNSW vectors.

    The incremental diff path removes vanished tools via index.remove_tool,
    which only deletes the SQLite row — the HNSW vector is orphaned and keeps
    consuming the fixed search candidate window. After enough removal-bearing
    cycles, _rebuild_for_backends must force a real build_index() rebuild to
    compact the index, rather than only ever calling add_single_tool.
    """

    def _set_old_names(self, sync_manager, names):
        """Make _get_backend_tool_names report these as currently-indexed."""
        rows = [{"name": n} for n in names]
        sync_manager.index.db.execute.return_value.fetchall.return_value = rows

    @pytest.mark.asyncio
    async def test_no_rebuild_below_threshold(self, sync_manager):
        """A single removal cycle must stay incremental (no full rebuild)."""
        # Old index had a tool that has now vanished -> one removal this cycle.
        self._set_old_names(sync_manager, ["backend1:gone"])
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("kept", "backend1:kept", "Kept", "backend1", {}),
            ]
        )
        sync_manager.index.remove_tool = AsyncMock(return_value=True)
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)
        sync_manager.index.build_index = AsyncMock(return_value={"tools_indexed": 1})

        await sync_manager._rebuild_for_backends(["backend1"])

        # One removal happened, but threshold not reached -> incremental.
        sync_manager.index.remove_tool.assert_awaited_once_with("backend1:gone")
        sync_manager.index.add_single_tool.assert_awaited()
        sync_manager.index.build_index.assert_not_called()
        assert sync_manager._incremental_cycles_since_rebuild == 1

    @pytest.mark.asyncio
    async def test_full_rebuild_after_threshold_cycles(self, sync_manager):
        """After K removal-bearing cycles, a real build_index rebuild fires.

        This is the core MSYNC-A-002 regression: without the periodic rebuild,
        orphan vectors accumulate forever and build_index is NEVER called from
        the incremental path. Drive enough cycles to cross the threshold and
        assert exactly one full rebuild happens, and the counter resets.

        BE-COMPACT-002: this test previously asserted ``len(rebuilt_arg) == 1``
        with a single-backend mock, which FALSELY certified the 'full tool set'
        invariant — it never exercised the multi-backend case where the
        table-truncating build_index would delete an idle backend's tools. It
        now wires BOTH configured backends and asserts the rebuild carries the
        FULL connected catalog (see also TestCompactionPreservesAllBackends).
        """
        # backend1 churns (one tool vanished this cycle); backend2 is a stable,
        # idle-but-connected backend whose single tool must survive compaction.
        current = {
            "backend1": [ToolInfo("kept", "backend1:kept", "Kept", "backend1", {})],
            "backend2": [ToolInfo("b2t", "backend2:b2t", "B2 Tool", "backend2", {})],
        }
        sync_manager.backends.get_backend_tools = Mock(
            side_effect=lambda name: current.get(name, [])
        )
        sync_manager.backends.is_backend_connected = Mock(return_value=True)

        old_names = {
            "backend1": ["backend1:kept", "backend1:gone"],
            "backend2": ["backend2:b2t"],
        }

        def fake_execute(query, params=()):
            backend = params[0] if params else None
            rows = [{"name": n} for n in old_names.get(backend, [])]
            result = Mock()
            result.fetchall = Mock(return_value=rows)
            return result

        sync_manager.index.db.execute.side_effect = fake_execute
        sync_manager.index.remove_tool = AsyncMock(return_value=True)
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)
        sync_manager.index.build_index = AsyncMock(return_value={"tools_indexed": 2})

        threshold = sync_manager._rebuild_after_incremental_cycles

        # Run threshold cycles, each removing the vanished tool from backend1.
        for _ in range(threshold):
            await sync_manager._rebuild_for_backends(["backend1"])

        # Exactly one full rebuild on the cycle that crossed the threshold,
        # over the FULL connected catalog (both backends), counter reset.
        sync_manager.index.build_index.assert_called_once()
        rebuilt_arg = sync_manager.index.build_index.call_args.args[0]
        rebuilt_names = {t.name for t in rebuilt_arg}
        assert rebuilt_names == {"backend1:kept", "backend2:b2t"}, (
            "compaction rebuild must carry the full connected tool set, not "
            "just the changed subset"
        )
        assert sync_manager._incremental_cycles_since_rebuild == 0

    @pytest.mark.asyncio
    async def test_cycles_without_removals_do_not_count(self, sync_manager):
        """Pure-add cycles (no removals) never trip the compaction counter."""
        # Old names == new names -> nothing removed.
        self._set_old_names(sync_manager, ["backend1:kept"])
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("kept", "backend1:kept", "Kept", "backend1", {}),
            ]
        )
        sync_manager.index.remove_tool = AsyncMock(return_value=True)
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)
        sync_manager.index.build_index = AsyncMock(return_value={"tools_indexed": 1})

        for _ in range(sync_manager._rebuild_after_incremental_cycles + 5):
            await sync_manager._rebuild_for_backends(["backend1"])

        # No removals ever -> counter stays at 0, no forced rebuild.
        sync_manager.index.remove_tool.assert_not_called()
        sync_manager.index.build_index.assert_not_called()
        assert sync_manager._incremental_cycles_since_rebuild == 0


class TestCompactionPreservesAllBackends:
    """BE-COMPACT-001 / BE-COMPACT-002: the orphan-vector compaction rebuild
    must operate over the CURRENT FULL tool set across EVERY connected backend
    — not just the changed subset passed to _rebuild_for_backends.

    build_index is a FULL table-truncating replace (DELETE FROM tools + rebuild
    HNSW from only the passed tools). When sync_if_needed calls
    _rebuild_for_backends(changed_backends) with only the churned backend and
    the compaction threshold trips, passing the changed-subset all_tools to
    build_index silently deletes every OTHER connected backend's tools →
    compass() returns "no matching tool" for them until a manual full_sync.

    The regression drives ONLY backend1's churn to the compaction threshold
    while backend2 sits idle-but-connected, and asserts the compaction rebuild
    argument contains BOTH backends' tools (backend2 survives).
    """

    def _wire_two_backends(self, sync_manager):
        """Configure backend1 (churning) + backend2 (idle-but-connected).

        - get_backend_tools returns each backend's CURRENT tool list.
        - index.db.execute(SELECT ... WHERE server = ?) returns the OLD indexed
          names per backend so backend1 shows one removal every cycle (which is
          what advances the compaction counter) while backend2 is stable.
        """
        # Current tools reported by each connected backend.
        current = {
            "backend1": [ToolInfo("kept", "backend1:kept", "Kept", "backend1", {})],
            "backend2": [ToolInfo("b2t", "backend2:b2t", "B2 Tool", "backend2", {})],
        }
        sync_manager.backends.get_backend_tools = Mock(
            side_effect=lambda name: current.get(name, [])
        )
        sync_manager.backends.is_backend_connected = Mock(return_value=True)

        # Old indexed name-sets per backend. backend1 also has ":gone" that
        # vanished this cycle -> a removal every cycle -> counter advances.
        old_names = {
            "backend1": ["backend1:kept", "backend1:gone"],
            "backend2": ["backend2:b2t"],
        }

        def fake_execute(query, params=()):
            # _get_backend_tool_names binds (backend_name,) as params.
            backend = params[0] if params else None
            rows = [{"name": n} for n in old_names.get(backend, [])]
            result = Mock()
            result.fetchall = Mock(return_value=rows)
            return result

        sync_manager.index.db.execute.side_effect = fake_execute

        sync_manager.index.remove_tool = AsyncMock(return_value=True)
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)
        sync_manager.index.build_index = AsyncMock(return_value={"tools_indexed": 2})

    @pytest.mark.asyncio
    async def test_compaction_rebuild_includes_all_connected_backends(
        self, sync_manager
    ):
        """When compaction fires from a single-backend change, the rebuild's
        tool set must still contain the idle backend2's tools."""
        self._wire_two_backends(sync_manager)

        threshold = sync_manager._rebuild_after_incremental_cycles

        # sync_if_needed only rebuilds the CHANGED subset — here only backend1.
        for _ in range(threshold):
            await sync_manager._rebuild_for_backends(["backend1"])

        sync_manager.index.build_index.assert_called_once()
        rebuilt = sync_manager.index.build_index.call_args.args[0]
        rebuilt_names = {t.name for t in rebuilt}

        # The bug: rebuilt would be {"backend1:kept"} only, nuking backend2.
        assert "backend1:kept" in rebuilt_names
        assert "backend2:b2t" in rebuilt_names, (
            "compaction rebuild dropped an idle backend's tools — build_index "
            "truncates the tools table, so backend2 would be silently deleted"
        )
        assert sync_manager._incremental_cycles_since_rebuild == 0


class TestCrossThreadReadLock:
    """MSYNC-A-003: cross-thread reads of the indexer DB hold its write lock.

    _get_backend_tool_names and full_sync's baseline read self.index.db on the
    event-loop thread. Without holding the indexer's _db_write_lock, a
    concurrent worker-thread write can raise sqlite3.ProgrammingError. The fix
    acquires that same lock for the reads.
    """

    @pytest.mark.asyncio
    async def test_get_backend_tool_names_holds_index_write_lock(self, sync_manager):
        """The read must be performed while holding index._db_write_lock."""
        import threading

        real_lock = threading.Lock()
        sync_manager.index._db_write_lock = real_lock

        held_during_execute = {"value": False}
        orig_fetchall = Mock(return_value=[{"name": "backend1:t"}])

        def fake_execute(*args, **kwargs):
            # When the read runs, the indexer write lock must be held so a
            # concurrent worker write would block instead of racing.
            held_during_execute["value"] = real_lock.locked()
            result = Mock()
            result.fetchall = orig_fetchall
            return result

        sync_manager.index.db.execute.side_effect = fake_execute

        names = sync_manager._get_backend_tool_names("backend1")

        assert names == {"backend1:t"}
        assert held_during_execute["value"], (
            "_get_backend_tool_names must hold index._db_write_lock during the read"
        )
        # Lock released afterward (no leak).
        assert not real_lock.locked()

    @pytest.mark.asyncio
    async def test_full_sync_baseline_read_holds_index_write_lock(self, sync_manager):
        """full_sync's baseline cross-thread read must hold the index lock."""
        import threading

        real_lock = threading.Lock()
        sync_manager.index._db_write_lock = real_lock

        lock_states = []

        def fake_execute(query, *args, **kwargs):
            if "SELECT name" in query and "tools" in query:
                lock_states.append(real_lock.locked())
            result = Mock()
            result.fetchall = Mock(return_value=[])
            return result

        sync_manager.index.db.execute.side_effect = fake_execute
        sync_manager.backends.connect_all = AsyncMock(return_value={})

        await sync_manager.full_sync()

        # At least one baseline SELECT ran with the index write lock held.
        assert any(lock_states), (
            "full_sync baseline read must hold index._db_write_lock"
        )
        assert not real_lock.locked()


class TestSyncIfNeededErrors:
    """Test error handling in sync_if_needed."""

    @pytest.mark.asyncio
    async def test_sync_if_needed_rebuild_error(self, sync_manager):
        """Should handle rebuild errors gracefully."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = True  # Backend changed

            with patch.object(
                sync_manager, "_rebuild_for_backends", new_callable=AsyncMock
            ) as mock_rebuild:
                mock_rebuild.side_effect = Exception("Rebuild failed")

                results = await sync_manager.sync_if_needed()

                # Should report error
                assert "sync_error" in results["backend1"]
                assert "sync_error" in results["backend2"]


class TestToolCategorizationEdgeCases:
    """Additional categorization edge cases."""

    @pytest.fixture
    def categorize(self, sync_manager):
        """Get categorization function."""
        return sync_manager._categorize_tool

    def test_categorize_project_tools(self, categorize):
        """Should categorize project-related tools."""
        assert categorize("list_projects", "List all projects") == "project"
        assert categorize("session_info", "Get session info") == "project"
        assert categorize("content_manager", "Manage content") == "project"

    def test_categorize_system_tools(self, categorize):
        """Should categorize system-related tools."""
        assert categorize("service_status", "Get service status") == "system"
        # Note: health matches "analysis" first due to order


class TestGetSyncStatusAdditional:
    """Additional get_sync_status tests."""

    @pytest.mark.asyncio
    async def test_get_sync_status_multiple_backends(self, sync_manager):
        """Should return status for all backends."""
        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("backend1", 5, "hash1"),
        )
        db.commit()

        result = await sync_manager.get_sync_status()

        assert "backends" in result
        # backend1 has sync data, backend2 should show never_synced
        assert result["backends"]["backend1"]["status"] == "synced"
        assert result["backends"]["backend2"]["status"] == "never_synced"


class TestCloseMethod:
    """Test SyncManager.close() method."""

    def test_close_database_connection(self, sync_manager):
        """Should close database connection."""
        # Ensure db is initialized
        _ = sync_manager._get_db()
        assert sync_manager._db is not None

        sync_manager.close()

        assert sync_manager._db is None

    def test_close_when_not_initialized(self, sync_manager):
        """Should handle close when db not initialized."""
        sync_manager._db = None

        # Should not raise
        sync_manager.close()


class TestFullSyncEdgeCases:
    """Additional full_sync edge cases."""

    @pytest.mark.asyncio
    async def test_full_sync_partial_connection(self, sync_manager):
        """Should handle partial backend connections AND surface the failure.

        CONTRACT (masking-test repair): this test previously asserted only
        tools_indexed/status, so it never exercised the partial-failure
        contract the cli relies on. full_sync() must now return
        connected_backends + failed_backends derived from connect_results so
        _cmd_sync can warn on partial failure. A configured backend that
        failed to connect MUST appear in failed_backends.
        """
        sync_manager.backends.connect_all = AsyncMock(
            return_value={
                "backend1": True,
                "backend2": False,  # Failed to connect
            }
        )
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )

        result = await sync_manager.full_sync()

        # Should still sync available tools
        assert result["tools_indexed"] == 1
        assert result["status"] == "complete"
        # CONTRACT: the partial failure is visible in the return shape.
        assert result["connected_backends"] == ["backend1"]
        assert result["failed_backends"] == ["backend2"]
        # backends_synced still reflects every attempted backend.
        assert set(result["backends_synced"]) == {"backend1", "backend2"}

    @pytest.mark.asyncio
    async def test_full_sync_clears_old_data(self, sync_manager):
        """Should clear old index data before full sync."""
        sync_manager.backends.connect_all = AsyncMock(return_value={"backend1": True})
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )

        await sync_manager.full_sync()

        # Should call build_index which clears and rebuilds
        sync_manager.index.build_index.assert_called_once()


# =============================================================================
# Stage-C Humanization: durable per-backend health + honest add-failure logging
# =============================================================================


class TestPersistedBackendHealthDEG02:
    """DEG-02: backend_sync_state.sync_status is the only durable per-backend
    health surface, but it was only ever written 'synced' — and that 'synced'
    was written BEFORE the rebuild that can still fail. These tests prove the
    'error' state is now persisted and that 'synced' lands only after the
    index work actually succeeds.
    """

    async def _status(self, sync_manager, backend):
        st = await sync_manager.get_sync_status()
        return st["backends"][backend]["status"]

    @pytest.mark.asyncio
    async def test_sync_if_needed_persists_check_error(self, sync_manager):
        """A failed change-check must write sync_status='error' (was: only
        ever returned in-memory; the durable surface stayed stale)."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = Exception("connection refused")

            await sync_manager.sync_if_needed()

        # The persisted, triage-visible surface now reflects the failure.
        assert await self._status(sync_manager, "backend1") == "error"
        st = await sync_manager.get_sync_status()
        assert "connection refused" in (st["backends"]["backend1"]["last_error"] or "")

    @pytest.mark.asyncio
    async def test_sync_if_needed_persists_rebuild_error(self, sync_manager):
        """A failed rebuild must write sync_status='error' for the changed
        backend, not leave a stale 'synced'."""
        with patch.object(
            sync_manager, "check_backend_changes", new_callable=AsyncMock
        ) as mock_check:
            mock_check.side_effect = [True, False]
            with patch.object(
                sync_manager, "_rebuild_for_backends", new_callable=AsyncMock
            ) as mock_rebuild:
                mock_rebuild.side_effect = Exception("rebuild boom")

                await sync_manager.sync_if_needed()

        assert await self._status(sync_manager, "backend1") == "error"

    @pytest.mark.asyncio
    async def test_rebuild_marks_error_when_adds_fail(self, sync_manager):
        """DEG-01 + DEG-02: when add_single_tool returns False (Ollama-down /
        breaker-open), the backend must NOT be marked 'synced'."""
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )
        # Every add fails — embedder/index unavailable.
        sync_manager.index.add_single_tool = AsyncMock(return_value=False)

        await sync_manager._rebuild_for_backends(["backend1"])

        # Pre-fix this row would read 'synced' (written before the loop).
        assert await self._status(sync_manager, "backend1") == "error"

    @pytest.mark.asyncio
    async def test_rebuild_marks_synced_only_after_success(self, sync_manager):
        """The terminal 'synced' write lands only after a clean add loop."""
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)

        await sync_manager._rebuild_for_backends(["backend1"])

        assert await self._status(sync_manager, "backend1") == "synced"

    @pytest.mark.asyncio
    async def test_full_sync_connect_failure_persists_error(self, sync_manager):
        """A configured backend that fails to connect during full_sync is
        recorded as 'error' (durable), and a recovered backend is 'synced'."""
        sync_manager.backends.connect_all = AsyncMock(
            return_value={"backend1": True, "backend2": False}
        )
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )

        await sync_manager.full_sync()

        assert await self._status(sync_manager, "backend1") == "synced"
        assert await self._status(sync_manager, "backend2") == "error"


class TestAddFailureLoggingDEG01:
    """DEG-01: _rebuild_for_backends discarded the add_single_tool bool and
    logged an unconditional 'Added N' success even when adds failed.
    """

    @pytest.mark.asyncio
    async def test_partial_add_failure_warns_not_success(self, sync_manager, caplog):
        """A partial add must emit a WARNING and NOT the unconditional
        'Added N tools' success line."""
        import logging

        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
                ToolInfo("tool2", "backend1:tool2", "Tool 2", "backend1", {}),
            ]
        )
        # First add succeeds, second fails.
        sync_manager.index.add_single_tool = AsyncMock(side_effect=[True, False])

        with caplog.at_level(logging.WARNING, logger="sync_manager"):
            await sync_manager._rebuild_for_backends(["backend1"])

        text = caplog.text
        assert "1/2 tool add(s) failed" in text
        # The misleading happy-path line must be suppressed on partial failure.
        assert "Added 2 tools" not in text

    @pytest.mark.asyncio
    async def test_clean_add_logs_success(self, sync_manager, caplog):
        """A fully-successful add still logs the 'Added N tools' line."""
        import logging

        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(
            return_value=[
                ToolInfo("tool1", "backend1:tool1", "Tool 1", "backend1", {}),
            ]
        )
        sync_manager.index.add_single_tool = AsyncMock(return_value=True)

        with caplog.at_level(logging.INFO, logger="sync_manager"):
            await sync_manager._rebuild_for_backends(["backend1"])

        assert "Added 1 tools" in caplog.text


class TestUnreachableVsGoneDEG03:
    """DEG-03: full_sync rebuilds from only the backends that connected, so a
    transiently-unreachable backend's tools get dropped — and were logged via
    'globally_removed' as if intentional. The fix distinguishes
    'removed because gone' from 'removed because unreachable'.
    """

    @pytest.mark.asyncio
    async def test_unreachable_drop_warns_not_removed(self, sync_manager, caplog):
        """When a backend that previously had rows fails to connect, the drop
        of its tools is WARNED as unreachable, not logged as a clean removal."""
        import logging

        # Baseline index has rows for both backends.
        sync_manager.index.db.execute.return_value.fetchall.return_value = [
            {"name": "backend1:t1", "server": "backend1"},
            {"name": "backend2:t1", "server": "backend2"},
        ]
        # backend1 connects (and still reports its tool); backend2 unreachable.
        sync_manager.backends.connect_all = AsyncMock(
            return_value={"backend1": True, "backend2": False}
        )

        def tools_for(name):
            if name == "backend1":
                return [ToolInfo("t1", "backend1:t1", "T1", "backend1", {})]
            return []

        sync_manager.backends.get_backend_tools = Mock(side_effect=tools_for)

        with caplog.at_level(logging.WARNING, logger="sync_manager"):
            await sync_manager.full_sync()

        # backend2:t1 is dropped because backend2 was UNREACHABLE — the log
        # must say so (warning), and must NOT claim it as a clean removal.
        assert "UNREACHABLE" in caplog.text
        assert "backend2" in caplog.text


# =============================================================================
# v2.5.0 FEAT-01 — schema fidelity: full inputSchema carried onto raw_schema
# =============================================================================


class TestSchemaFidelityFEAT01:
    """FEAT-01: _tool_infos_to_definitions collapses each tool's inputSchema to
    a flat {param:type} dict and threw the rest away. The full inputSchema is
    available at collection time (ToolInfo.input_schema); it must be preserved
    verbatim onto ToolDefinition.raw_schema (the Wave-A field) while the
    collapsed `parameters` view is kept unchanged (embedding_text + back-compat
    depend on it).
    """

    _RICH_SCHEMA = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "mode": {"type": "string", "enum": ["r", "rb"]},
            "limit": {"type": ["integer", "null"]},
        },
        "required": ["path"],
    }

    def test_raw_schema_carried_from_tool_info(self, sync_manager):
        """A tool with a rich inputSchema -> its ToolDefinition.raw_schema
        carries the FULL schema (required/enum/description all preserved)."""
        tools = [
            ToolInfo(
                "read_file",
                "backend1:read_file",
                "Read a file",
                "backend1",
                self._RICH_SCHEMA,
            ),
        ]

        defs = sync_manager._tool_infos_to_definitions(tools)

        assert len(defs) == 1
        d = defs[0]
        # The full schema is preserved verbatim, not the collapsed view.
        assert d.raw_schema == self._RICH_SCHEMA
        assert d.raw_schema["required"] == ["path"]
        assert d.raw_schema["properties"]["mode"]["enum"] == ["r", "rb"]
        assert (
            d.raw_schema["properties"]["path"]["description"] == "File path to read"
        )

    def test_collapsed_parameters_unchanged(self, sync_manager):
        """The collapsed `parameters` view is kept exactly as before —
        {param: type}, with list types "/"-joined."""
        tools = [
            ToolInfo(
                "read_file",
                "backend1:read_file",
                "Read a file",
                "backend1",
                self._RICH_SCHEMA,
            ),
        ]

        defs = sync_manager._tool_infos_to_definitions(tools)
        d = defs[0]

        assert d.parameters == {
            "path": "string",
            "mode": "string",
            "limit": "integer/null",
        }

    def test_raw_schema_none_when_no_schema(self, sync_manager):
        """A tool with an empty/absent inputSchema -> raw_schema is None
        (nothing to preserve), collapsed parameters is {}."""
        tools = [
            ToolInfo("noargs", "backend1:noargs", "No args", "backend1", {}),
        ]

        defs = sync_manager._tool_infos_to_definitions(tools)
        d = defs[0]

        assert d.raw_schema is None
        assert d.parameters == {}


# =============================================================================
# v2.5.0 FEAT-02 — change detection hashes description + schema, not just names
# =============================================================================


class TestChangeDetectionFingerprintFEAT02:
    """FEAT-02: _compute_tool_hash hashed sorted tool NAMES only, so a backend
    that revised a tool's description/params (same name) read as 'unchanged'
    and never re-synced. The fingerprint must now hash
    (qualified_name, description, canonical_json(input_schema)) tuples so any
    description OR schema change trips a re-sync — and canonicalize the schema
    so dict key reordering doesn't cause spurious hash churn.
    """

    def test_changed_description_changes_hash(self, sync_manager):
        """Same names, but a changed description -> DIFFERENT hash
        (previously identical -> the bug that skipped the re-sync)."""
        before = [
            ToolInfo("t1", "backend1:t1", "Original description", "backend1", {}),
        ]
        after = [
            ToolInfo("t1", "backend1:t1", "REVISED description", "backend1", {}),
        ]

        assert (
            sync_manager._compute_tool_hash(before)
            != sync_manager._compute_tool_hash(after)
        )

    def test_changed_input_schema_changes_hash(self, sync_manager):
        """Same names + description, but a changed inputSchema -> DIFFERENT
        hash."""
        before = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Same description",
                "backend1",
                {"type": "object", "properties": {"a": {"type": "string"}}},
            ),
        ]
        after = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Same description",
                "backend1",
                {"type": "object", "properties": {"a": {"type": "integer"}}},
            ),
        ]

        assert (
            sync_manager._compute_tool_hash(before)
            != sync_manager._compute_tool_hash(after)
        )

    def test_identical_tool_sets_same_hash(self, sync_manager):
        """Two structurally-identical tool sets still produce the SAME hash —
        no spurious churn."""
        a = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Desc one",
                "backend1",
                {"type": "object", "properties": {"a": {"type": "string"}}},
            ),
            ToolInfo("t2", "backend1:t2", "Desc two", "backend1", {}),
        ]
        b = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Desc one",
                "backend1",
                {"type": "object", "properties": {"a": {"type": "string"}}},
            ),
            ToolInfo("t2", "backend1:t2", "Desc two", "backend1", {}),
        ]

        assert (
            sync_manager._compute_tool_hash(a)
            == sync_manager._compute_tool_hash(b)
        )

    def test_schema_key_reordering_does_not_change_hash(self, sync_manager):
        """Canonicalization: reordering dict keys in the inputSchema must NOT
        change the hash (json.dumps(..., sort_keys=True))."""
        ordered = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Desc",
                "backend1",
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "int"}},
                    "required": ["a"],
                },
            ),
        ]
        reordered = [
            ToolInfo(
                "t1",
                "backend1:t1",
                "Desc",
                "backend1",
                {
                    "required": ["a"],
                    "properties": {"b": {"type": "int"}, "a": {"type": "string"}},
                    "type": "object",
                },
            ),
        ]

        assert (
            sync_manager._compute_tool_hash(ordered)
            == sync_manager._compute_tool_hash(reordered)
        )

    @pytest.mark.asyncio
    async def test_check_backend_changes_detects_description_revision(
        self, sync_manager
    ):
        """End-to-end: a stored hash for the OLD description, then the backend
        revises the description -> check_backend_changes reports changed."""
        old_tools = [
            ToolInfo("t1", "backend1:t1", "Original", "backend1", {}),
        ]
        stored_hash = sync_manager._compute_tool_hash(old_tools)

        db = sync_manager._get_db()
        db.execute(
            """
            INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
        """,
            ("backend1", 1, stored_hash),
        )
        db.commit()

        # Same name, revised description.
        new_tools = [
            ToolInfo("t1", "backend1:t1", "REVISED", "backend1", {}),
        ]
        sync_manager.backends.is_backend_connected = Mock(return_value=True)
        sync_manager.backends.get_backend_tools = Mock(return_value=new_tools)

        result = await sync_manager.check_backend_changes("backend1")

        assert result is True, (
            "a same-name description revision must trip change detection"
        )


# =============================================================================
# v2.5.0 FEAT-06 (index-time half) — filter denied tools at sync
# =============================================================================


class TestToolFilteringAtSyncFEAT06:
    """FEAT-06 (index half): _tool_infos_to_definitions must apply the resolving
    backend's allow_tools/deny_tools globs so denied tools are NEVER indexed.
    Match the BARE tool name; deny wins over allow; empty allow_tools = allow
    all; a non-empty allow_tools = only-those. (The execute() boundary half is
    owned by sibling B2.)
    """

    def _backend_names(self, defs):
        return {d.name for d in defs}

    def test_deny_glob_excludes_matching_tool(self, mock_index, mock_backends, temp_sync_db):
        """deny_tools=['*secret*'] must not index a tool named 'read_secret'."""
        config = CompassConfig(
            backends={
                "backend1": StdioBackend(
                    command="python",
                    args=[],
                    deny_tools=["*secret*"],
                ),
            }
        )
        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(config, mock_index, mock_backends)
            try:
                tools = [
                    ToolInfo("read_secret", "backend1:read_secret", "Secret", "backend1", {}),
                    ToolInfo("read_file", "backend1:read_file", "File", "backend1", {}),
                ]
                defs = manager._tool_infos_to_definitions(tools)
            finally:
                manager.close()

        names = self._backend_names(defs)
        assert "backend1:read_secret" not in names
        assert "backend1:read_file" in names

    def test_allow_glob_indexes_only_matching(self, mock_index, mock_backends, temp_sync_db):
        """allow_tools=['read_*'] must index only read_* tools."""
        config = CompassConfig(
            backends={
                "backend1": StdioBackend(
                    command="python",
                    args=[],
                    allow_tools=["read_*"],
                ),
            }
        )
        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(config, mock_index, mock_backends)
            try:
                tools = [
                    ToolInfo("read_file", "backend1:read_file", "R", "backend1", {}),
                    ToolInfo("read_dir", "backend1:read_dir", "R", "backend1", {}),
                    ToolInfo("write_file", "backend1:write_file", "W", "backend1", {}),
                ]
                defs = manager._tool_infos_to_definitions(tools)
            finally:
                manager.close()

        names = self._backend_names(defs)
        assert names == {"backend1:read_file", "backend1:read_dir"}

    def test_deny_wins_over_allow(self, mock_index, mock_backends, temp_sync_db):
        """A tool matching BOTH allow and deny is denied (deny precedence)."""
        config = CompassConfig(
            backends={
                "backend1": StdioBackend(
                    command="python",
                    args=[],
                    allow_tools=["read_*"],
                    deny_tools=["*secret*"],
                ),
            }
        )
        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(config, mock_index, mock_backends)
            try:
                tools = [
                    ToolInfo("read_secret", "backend1:read_secret", "S", "backend1", {}),
                    ToolInfo("read_file", "backend1:read_file", "F", "backend1", {}),
                ]
                defs = manager._tool_infos_to_definitions(tools)
            finally:
                manager.close()

        names = self._backend_names(defs)
        assert "backend1:read_secret" not in names
        assert "backend1:read_file" in names

    def test_empty_lists_index_everything(self, sync_manager):
        """Back-compat: empty allow_tools + empty deny_tools index everything.

        (The default mock_config backends carry empty filter lists.)"""
        tools = [
            ToolInfo("read_secret", "backend1:read_secret", "S", "backend1", {}),
            ToolInfo("anything", "backend1:anything", "A", "backend1", {}),
        ]

        defs = sync_manager._tool_infos_to_definitions(tools)

        assert self._backend_names(defs) == {
            "backend1:read_secret",
            "backend1:anything",
        }

    def test_filter_uses_resolving_backend_config(self, mock_index, mock_backends, temp_sync_db):
        """The filter is resolved from the tool's OWN backend config — a tool
        whose qualified name resolves to backend2 uses backend2's deny list,
        not backend1's."""
        config = CompassConfig(
            backends={
                "backend1": StdioBackend(command="python", args=[], deny_tools=["*secret*"]),
                "backend2": StdioBackend(command="python", args=[]),  # no filters
            }
        )
        with patch("sync_manager.ANALYTICS_DB_PATH", temp_sync_db):
            manager = SyncManager(config, mock_index, mock_backends)
            try:
                tools = [
                    # backend2's secret tool is NOT denied (backend2 has no deny list).
                    ToolInfo("read_secret", "backend2:read_secret", "S", "backend2", {}),
                ]
                defs = manager._tool_infos_to_definitions(tools)
            finally:
                manager.close()

        assert self._backend_names(defs) == {"backend2:read_secret"}
