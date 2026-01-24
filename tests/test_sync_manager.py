"""
Tests for Tool Compass sync manager module.

Tests backend change detection, hash computation, and sync operations.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sync_manager import SyncManager, get_sync_manager
from config import CompassConfig, StdioBackend
from backend_client import ToolInfo


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
    """Mock CompassIndex."""
    index = AsyncMock()
    index.build_index = AsyncMock(return_value={"tools_indexed": 5})
    index.add_single_tool = AsyncMock(return_value=True)
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
        sync_manager.backends.connect_backend = AsyncMock(return_value=False)

        result = await sync_manager.check_backend_changes("backend1")

        assert result is False
        sync_manager.backends.connect_backend.assert_called_once_with("backend1")

    @pytest.mark.asyncio
    async def test_check_backend_changes_no_tools(self, sync_manager):
        """Should return False when backend has no tools."""
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
        sync_manager.backends.get_backend_tools = Mock(return_value=[])

        result = await sync_manager.check_backend_changes("backend1")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_backend_changes_first_sync(self, sync_manager):
        """Should detect changes on first sync (no stored hash)."""
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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

        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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

        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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
        """Should handle no connected backends."""
        sync_manager.backends.connect_all = AsyncMock(return_value={})

        result = await sync_manager.full_sync()

        assert result["status"] == "no_tools"
        assert result["tools_indexed"] == 0

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
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
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
        sync_manager.backends._backends = {"backend1": Mock(is_connected=True)}
        sync_manager.backends.get_backend_tools = Mock(return_value=tools)
        sync_manager.index.add_single_tool = AsyncMock()

        await sync_manager._rebuild_for_backends(["backend1"])

        sync_manager.index.add_single_tool.assert_called_once()


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
        """Should handle partial backend connections."""
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
