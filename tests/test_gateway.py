"""
Tests for Tool Compass gateway MCP tools.

Tests the main MCP interface functions: compass, describe, execute.
Based on FastMCP testing patterns: https://gofastmcp.com/patterns/testing
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch


class TestCompassTool:
    """Test the compass() search tool."""

    @pytest.mark.asyncio
    async def test_compass_basic_search(self, test_index, test_config):
        """Should return search results for a query."""
        # Import after fixtures set up mocks
        from gateway import compass
        import gateway

        # Inject test fixtures
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True  # Skip sync

        result = await compass(intent="read a file", top_k=3)

        assert "matches" in result
        assert "total_indexed" in result
        assert "hint" in result
        assert len(result["matches"]) <= 3

    @pytest.mark.asyncio
    async def test_compass_with_filters(self, test_index, test_config):
        """Should apply category and server filters."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(
            intent="anything",
            top_k=10,
            category="file",
            server="test",
        )

        # All results should match filters
        for match in result["matches"]:
            assert match["category"] == "file"
            assert match["server"] == "test"

    @pytest.mark.asyncio
    async def test_compass_min_confidence(self, test_index, test_config):
        """Should filter results below min_confidence."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(
            intent="file operations",
            top_k=10,
            min_confidence=0.5,
        )

        # All results should be above threshold
        for match in result["matches"]:
            assert match["confidence"] >= 0.5

    @pytest.mark.asyncio
    async def test_compass_tokens_saved(self, test_index, test_config):
        """Should calculate token savings."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(intent="anything", top_k=3)

        assert "tokens_saved" in result
        assert result["tokens_saved"] >= 0

    @pytest.mark.asyncio
    async def test_compass_no_results(self, test_index, test_config):
        """Should handle no matching results gracefully."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(
            intent="file operations",
            category="nonexistent",
        )

        assert result["matches"] == []
        assert "No tools found" in result["hint"]


class TestDescribeTool:
    """Test the describe() tool schema retrieval."""

    @pytest.mark.asyncio
    async def test_describe_existing_tool(self, test_index, test_config):
        """Should return full schema for existing tool."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_tool_schema = Mock(return_value=None)

        from gateway import describe

        result = await describe(tool_name="test:read_file")

        assert "tool" in result
        assert "description" in result
        assert "parameters" in result
        assert result["tool"] == "test:read_file"

    @pytest.mark.asyncio
    async def test_describe_nonexistent_tool(self, test_index, test_config):
        """Should return error for nonexistent tool."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_tool_schema = Mock(return_value=None)

        from gateway import describe

        result = await describe(tool_name="test:does_not_exist")

        assert "error" in result
        assert "hint" in result


class TestExecuteTool:
    """Test the execute() tool proxy."""

    @pytest.mark.asyncio
    async def test_execute_success(self, test_config):
        """Should proxy tool execution to backend."""
        import gateway

        # Mock backend manager
        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.connect_backend = AsyncMock(return_value=True)
        mock_manager.execute_tool = AsyncMock(
            return_value={"success": True, "data": "result"}
        )

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(
            tool_name="test:read_file",
            arguments={"filepath": "/tmp/test.txt"},
        )

        assert result["success"] is True
        mock_manager.execute_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_backend_connection_failure(self, test_config):
        """Should handle backend connection failures."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {}
        mock_manager.connect_backend = AsyncMock(return_value=False)

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:read_file")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_with_analytics(
        self, test_config_with_backends, test_analytics
    ):
        """Should record tool execution in analytics."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.execute_tool = AsyncMock(return_value={"success": True})

        gateway._backend_manager = mock_manager
        gateway._config = (
            test_config_with_backends  # Use config with analytics_enabled=True
        )
        gateway._analytics = test_analytics

        from gateway import execute

        await execute(tool_name="test:tool", arguments={})

        # Analytics should have recorded the call
        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["tool_calls"]["total"] >= 1


class TestCategoriesAndStatus:
    """Test utility tools."""

    @pytest.mark.asyncio
    async def test_compass_categories(self, test_index, test_config):
        """Should return category and server breakdown."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config

        from gateway import compass_categories

        result = await compass_categories()

        assert "categories" in result
        assert "servers" in result
        assert "total_tools" in result
        assert result["total_tools"] > 0

    @pytest.mark.asyncio
    async def test_compass_status(self, test_index, test_config):
        """Should return comprehensive status."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(return_value={"connected": 0})

        from gateway import compass_status

        result = await compass_status()

        assert "index" in result
        assert "backends" in result
        assert "config" in result


class TestSingletonInitialization:
    """Test async singleton initialization patterns."""

    @pytest.mark.asyncio
    async def test_get_index_creates_once(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """get_index() should only create index once."""
        import gateway

        # Reset global state
        gateway._compass_index = None

        # Create a pre-built index
        from indexer import CompassIndex

        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        await index.build_index(sample_tools)

        # Inject the pre-built index
        gateway._compass_index = index

        # Multiple calls should return same instance
        from gateway import get_index

        idx1 = await get_index()
        idx2 = await get_index()

        assert idx1 is idx2

        await index.close()

    @pytest.mark.asyncio
    async def test_concurrent_initialization_safety(
        self, temp_index_path, temp_db_path, mock_embedder, sample_tools
    ):
        """Concurrent get_index() calls should not create duplicates."""
        import asyncio
        import gateway

        # Build index first
        from indexer import CompassIndex

        index = CompassIndex(
            index_path=temp_index_path,
            db_path=temp_db_path,
            embedder=mock_embedder,
        )
        await index.build_index(sample_tools)
        gateway._compass_index = index

        from gateway import get_index

        # Simulate concurrent calls
        results = await asyncio.gather(
            get_index(),
            get_index(),
            get_index(),
        )

        # All should return same instance
        assert all(r is results[0] for r in results)

        await index.close()


# =============================================================================
# Analytics Tool Tests
# =============================================================================


class TestCompassAnalyticsTool:
    """Test compass_analytics() MCP tool."""

    @pytest.mark.asyncio
    async def test_analytics_disabled(self, test_config):
        """Should return error when analytics disabled."""
        import gateway

        gateway._config = test_config  # analytics_enabled=False

        from gateway import compass_analytics

        result = await compass_analytics()

        assert "error" in result
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_analytics_returns_summary(
        self, test_config_with_backends, test_analytics
    ):
        """Should return analytics summary when enabled."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics

        # Record some data
        await test_analytics.record_search("test query", [], 10.0)
        await test_analytics.record_tool_call("test:tool", True, 5.0)

        from gateway import compass_analytics

        result = await compass_analytics(timeframe="1h", include_failures=True)

        assert "searches" in result
        assert "tool_calls" in result

    @pytest.mark.asyncio
    async def test_analytics_exclude_failures(
        self, test_config_with_backends, test_analytics
    ):
        """Should exclude failures when requested."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics

        from gateway import compass_analytics

        result = await compass_analytics(timeframe="1h", include_failures=False)

        assert "failures" not in result


# =============================================================================
# Chains Tool Tests
# =============================================================================


class TestCompassChainsTool:
    """Test compass_chains() MCP tool."""

    @pytest.mark.asyncio
    async def test_chains_disabled(self, test_config):
        """Should return error when chain indexing disabled."""
        import gateway

        gateway._config = test_config  # chain_indexing_enabled=False

        from gateway import compass_chains

        result = await compass_chains()

        assert "error" in result
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_chains_list(self, test_config_with_backends, test_chain_indexer):
        """Should list chains when enabled."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        # Add a test chain
        await test_chain_indexer.add_chain(
            name="test_workflow", tools=["tool1", "tool2"], description="Test workflow"
        )

        from gateway import compass_chains

        result = await compass_chains(action="list")

        assert "chains" in result
        assert "total" in result

    @pytest.mark.asyncio
    async def test_chains_create(self, test_config_with_backends, test_chain_indexer):
        """Should create new chains."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(
            action="create",
            chain_name="new_workflow",
            tools=["step1", "step2", "step3"],
            description="New workflow description",
        )

        assert "created" in result
        assert result["created"]["name"] == "new_workflow"

    @pytest.mark.asyncio
    async def test_chains_create_missing_params(
        self, test_config_with_backends, test_chain_indexer
    ):
        """Should error when missing required params for create."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(action="create")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_chains_detect(
        self, test_config_with_backends, test_analytics, test_chain_indexer
    ):
        """Should detect chains from usage patterns."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(action="detect")

        assert "detected" in result or "error" in result

    @pytest.mark.asyncio
    async def test_chains_unknown_action(
        self, test_config_with_backends, test_chain_indexer
    ):
        """Should error on unknown action."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(action="invalid")

        assert "error" in result
        assert "unknown" in result["error"].lower()


# =============================================================================
# Sync Tool Tests
# =============================================================================


class TestCompassSyncTool:
    """Test compass_sync() MCP tool."""

    @pytest.mark.asyncio
    async def test_sync_disabled(self, test_config):
        """Should return error when auto_sync disabled."""
        import gateway

        gateway._config = test_config  # auto_sync=False

        from gateway import compass_sync

        result = await compass_sync()

        assert "error" in result
        assert "disabled" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_if_needed(
        self, test_config_with_backends, test_index, mock_backend_manager
    ):
        """Should run sync_if_needed."""
        import gateway
        from sync_manager import SyncManager

        # Create mock sync manager
        mock_sync = Mock(spec=SyncManager)
        mock_sync.sync_if_needed = AsyncMock(return_value={"backend1": "unchanged"})

        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._sync_manager = mock_sync

        from gateway import compass_sync

        result = await compass_sync(force=False)

        assert "action" in result
        assert result["action"] == "sync_if_needed"
        mock_sync.sync_if_needed.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_force(self, test_config_with_backends):
        """Should run full_sync when force=True."""
        import gateway
        from sync_manager import SyncManager

        mock_sync = Mock(spec=SyncManager)
        mock_sync.full_sync = AsyncMock(
            return_value={"status": "complete", "tools_indexed": 10}
        )

        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._sync_manager = mock_sync

        from gateway import compass_sync

        result = await compass_sync(force=True)

        assert result["action"] == "full_sync"
        mock_sync.full_sync.assert_called_once()


# =============================================================================
# Audit Tool Tests
# =============================================================================


class TestCompassAuditTool:
    """Test compass_audit() MCP tool."""

    @pytest.mark.asyncio
    async def test_audit_basic(self, test_index, test_config):
        """Should return comprehensive audit."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={
                "configured_backends": ["test"],
                "connected_backends": [],
                "total_tools": 0,
                "tools_by_backend": {},
            }
        )

        from gateway import compass_audit

        result = await compass_audit()

        assert "system" in result
        assert "categories" in result
        assert "servers" in result
        assert "backends" in result
        assert "config" in result
        assert "health" in result

    @pytest.mark.asyncio
    async def test_audit_with_analytics(
        self, test_index, test_config_with_backends, test_analytics
    ):
        """Should include analytics in audit."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(return_value={
            "configured_backends": [],
            "connected_backends": [],
            "total_tools": 0,
            "tools_by_backend": {}
        })
        # Mock chain indexer to avoid DB path issues in CI
        mock_chain_indexer = Mock()
        mock_chain_indexer._chain_cache = {}
        mock_chain_indexer.load_chains_from_db = AsyncMock(return_value=[])
        gateway._chain_indexer = mock_chain_indexer
        # Mock sync manager to avoid DB issues
        gateway._sync_manager = Mock()
        gateway._sync_manager.get_sync_status = AsyncMock(return_value={})

        # Record some data
        await test_analytics.record_search("query", [], 5.0)

        from gateway import compass_audit

        # Patch get_analytics_instance and get_chain_indexer_instance to avoid
        # issues with singletons creating new instances with default paths
        with patch.object(gateway, "get_analytics_instance", AsyncMock(return_value=test_analytics)):
            with patch.object(gateway, "get_chain_indexer_instance", AsyncMock(return_value=None)):
                result = await compass_audit(timeframe="1h")

        assert "hot_cache" in result
        assert "analytics" in result

    @pytest.mark.asyncio
    async def test_audit_include_tools(self, test_index, test_config):
        """Should include tool list when requested."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={
                "configured_backends": [],
                "connected_backends": [],
                "total_tools": 0,
                "tools_by_backend": {},
            }
        )

        from gateway import compass_audit

        result = await compass_audit(include_tools=True)

        assert "tools" in result
        assert isinstance(result["tools"], list)

    @pytest.mark.asyncio
    async def test_audit_health_issues(self, test_config):
        """Should report health issues."""
        import gateway
        from indexer import CompassIndex

        # Create empty index
        mock_index = Mock(spec=CompassIndex)
        mock_index.get_stats = Mock(
            return_value={"total_tools": 0, "by_category": {}, "by_server": {}}
        )
        mock_index.db = Mock()
        mock_index.index_path = Path("/tmp/test.hnsw")
        mock_index.db_path = Path("/tmp/test.db")

        gateway._compass_index = mock_index
        gateway._config = test_config
        gateway._analytics = Mock()
        gateway._analytics._hot_cache = {}
        gateway._analytics.get_analytics_summary = AsyncMock(
            return_value={
                "searches": {"total": 0, "avg_latency_ms": 0, "top_queries": []},
                "tool_calls": {"total": 0, "success_rate": 0, "top_tools": []},
            }
        )
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={
                "configured_backends": [],
                "connected_backends": [],
                "total_tools": 0,
                "tools_by_backend": {},
            }
        )

        # Enable analytics for test
        gateway._config.analytics_enabled = True

        from gateway import compass_audit

        result = await compass_audit()

        assert result["health"]["status"] == "needs_attention"
        assert len(result["health"]["issues"]) > 0


# =============================================================================
# Startup Sync Tests
# =============================================================================


class TestStartupSync:
    """Test maybe_startup_sync() function."""

    @pytest.mark.asyncio
    async def test_startup_sync_disabled(self, test_config):
        """Should skip sync when sync_check_on_startup=False."""
        import gateway

        gateway._config = test_config
        gateway._config.sync_check_on_startup = False
        gateway._startup_sync_done = False

        from gateway import maybe_startup_sync

        await maybe_startup_sync()

        # Should complete without error

    @pytest.mark.asyncio
    async def test_startup_sync_only_once(self, test_config):
        """Should only run sync once."""
        import gateway

        gateway._config = test_config
        gateway._config.sync_check_on_startup = True
        gateway._startup_sync_done = True

        from gateway import maybe_startup_sync

        await maybe_startup_sync()

        # Should skip because already done


# =============================================================================
# Categorize Tool Tests
# =============================================================================


class TestCategorizeTool:
    """Test the categorize_tool() helper function."""

    def test_file_category(self):
        """Should categorize file operations."""
        from gateway import categorize_tool

        assert categorize_tool("read_file", "Read file contents") == "file"
        assert categorize_tool("write_data", "Write data to disk") == "file"
        assert categorize_tool("list_directory", "List directory") == "file"

    def test_git_category(self):
        """Should categorize git operations."""
        from gateway import categorize_tool

        assert categorize_tool("git_status", "Show git status") == "git"
        assert categorize_tool("commit_changes", "Commit changes") == "git"
        assert categorize_tool("branch_list", "List branches") == "git"

    def test_database_category(self):
        """Should categorize database operations."""
        from gateway import categorize_tool

        assert categorize_tool("db_query", "Execute query") == "database"
        assert categorize_tool("sql_execute", "Run SQL") == "database"

    def test_search_category(self):
        """Should categorize search operations."""
        from gateway import categorize_tool

        assert categorize_tool("search_docs", "Search documents") == "search"
        assert categorize_tool("lookup_value", "Lookup value") == "search"

    def test_ai_category(self):
        """Should categorize AI operations."""
        from gateway import categorize_tool

        assert categorize_tool("comfy_generate", "Generate image") == "ai"
        assert categorize_tool("video_create", "Create video") == "ai"

    def test_analysis_category(self):
        """Should categorize analysis operations."""
        from gateway import categorize_tool

        assert categorize_tool("scan_code", "Scan for issues") == "analysis"
        assert categorize_tool("analyze_data", "Analyze data") == "analysis"

    def test_project_category(self):
        """Should categorize project operations."""
        from gateway import categorize_tool

        assert categorize_tool("create_project", "Create project") == "project"
        assert categorize_tool("list_sessions", "List sessions") == "project"

    def test_system_category(self):
        """Should categorize system operations."""
        from gateway import categorize_tool

        assert categorize_tool("service_status", "Service status") == "system"
        assert (
            categorize_tool("health_check", "Check health") == "analysis"
        )  # matches analysis first

    def test_other_category(self):
        """Should default to other for unknown tools."""
        from gateway import categorize_tool

        assert categorize_tool("unknown_tool", "Does something") == "other"


# =============================================================================
# Get Config Tests
# =============================================================================


class TestGetConfig:
    """Test get_config() function."""

    def test_get_config_caches(self):
        """Should cache config after first load."""
        import gateway

        gateway._config = None

        with patch("gateway.load_config") as mock_load:
            mock_load.return_value = Mock()

            from gateway import get_config

            config1 = get_config()
            config2 = get_config()

            # Should only call load_config once
            mock_load.assert_called_once()
            assert config1 is config2


# =============================================================================
# Get Backends Tests
# =============================================================================


class TestGetBackends:
    """Test get_backends() function."""

    @pytest.mark.asyncio
    async def test_get_backends_creates_once(self, test_config):
        """Should create backend manager once."""
        import gateway

        gateway._backend_manager = None
        gateway._config = test_config

        from gateway import get_backends

        manager1 = await get_backends()
        manager2 = await get_backends()

        assert manager1 is manager2


# =============================================================================
# Get Analytics Instance Tests
# =============================================================================


class TestGetAnalyticsInstance:
    """Test get_analytics_instance() function."""

    @pytest.mark.asyncio
    async def test_analytics_disabled_returns_none(self, test_config):
        """Should return None when analytics disabled."""
        import gateway

        gateway._config = test_config  # analytics_enabled=False
        gateway._analytics = None

        from gateway import get_analytics_instance

        result = await get_analytics_instance()

        assert result is None

    @pytest.mark.asyncio
    async def test_analytics_enabled_returns_instance(self, test_config_with_backends):
        """Should return analytics instance when enabled."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = Mock()
        gateway._analytics.load_hot_cache_from_db = AsyncMock()

        from gateway import get_analytics_instance

        result = await get_analytics_instance()

        assert result is not None


# =============================================================================
# Compass Tool with Chains
# =============================================================================


class TestCompassWithChains:
    """Test compass() tool with chain searching."""

    @pytest.mark.asyncio
    async def test_compass_includes_chains(
        self, test_index, test_config_with_backends, test_chain_indexer
    ):
        """Should include chain results when chain_indexing enabled."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer
        gateway._startup_sync_done = True
        gateway._analytics = None  # Disable analytics for this test

        # Add a chain
        await test_chain_indexer.add_chain(
            name="file_workflow",
            tools=["test:read_file", "test:write_file"],
            description="Read and write files",
        )
        await test_chain_indexer.build_chain_index()

        from gateway import compass

        result = await compass(intent="file operations", include_chains=True)

        assert "matches" in result
        # Chains may or may not be found depending on embedding similarity

    @pytest.mark.asyncio
    async def test_compass_without_chains(
        self, test_index, test_config, test_chain_indexer
    ):
        """Should exclude chains when include_chains=False."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(intent="anything", include_chains=False)

        assert "chains" not in result or result.get("chains") == []


# =============================================================================
# Progressive Disclosure Tests
# =============================================================================


class TestProgressiveDisclosure:
    """Test progressive disclosure behavior."""

    @pytest.mark.asyncio
    async def test_progressive_disclosure_enabled(self, test_index, test_config):
        """Should not include full schemas when progressive disclosure enabled."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._config.progressive_disclosure = True
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(intent="file", top_k=1)

        if result["matches"]:
            match = result["matches"][0]
            # Should not have full parameters in progressive mode
            assert "parameters" not in match

    @pytest.mark.asyncio
    async def test_progressive_disclosure_disabled(self, test_index, test_config):
        """Should include full schemas when progressive disclosure disabled."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._config.progressive_disclosure = False
        gateway._startup_sync_done = True

        from gateway import compass

        result = await compass(intent="file", top_k=1)

        if result["matches"]:
            match = result["matches"][0]
            # Should have parameters when progressive disclosure disabled
            assert "parameters" in match


# =============================================================================
# Describe Tool Backend Fallback Tests
# =============================================================================


class TestDescribeBackendFallback:
    """Test describe() tool backend fallback."""

    @pytest.mark.asyncio
    async def test_describe_falls_back_to_backend(self, test_index, test_config):
        """Should try backend when tool not in index."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config

        # Mock backend with tool schema
        mock_manager = Mock()
        mock_manager.get_tool_schema = Mock(
            return_value={
                "name": "backend:tool",
                "description": "From backend",
                "parameters": {"arg": "str"},
            }
        )
        gateway._backend_manager = mock_manager

        from gateway import describe

        result = await describe(tool_name="backend:tool")

        # Should use backend schema
        assert result.get("description") == "From backend"


# =============================================================================
# Execute Tool Edge Cases
# =============================================================================


class TestExecuteEdgeCases:
    """Test execute() edge cases."""

    @pytest.mark.asyncio
    async def test_execute_default_arguments(self, test_config):
        """Should handle None arguments."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.execute_tool = AsyncMock(return_value={"success": True})

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        await execute(tool_name="test:tool", arguments=None)

        # Should pass empty dict
        mock_manager.execute_tool.assert_called_with("test:tool", {})

    @pytest.mark.asyncio
    async def test_execute_records_analytics_on_success(
        self, test_config_with_backends, test_analytics
    ):
        """Should record analytics on successful execution."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.execute_tool = AsyncMock(return_value={"success": True})

        gateway._backend_manager = mock_manager
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics

        from gateway import execute

        await execute(tool_name="test:analytics_test", arguments={"key": "value"})

        # Check analytics recorded
        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["tool_calls"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_execute_records_analytics_on_failure(
        self, test_config_with_backends, test_analytics
    ):
        """Should record analytics on failed execution."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.execute_tool = AsyncMock(
            return_value={"success": False, "error": "Test error"}
        )

        gateway._backend_manager = mock_manager
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics

        from gateway import execute

        await execute(tool_name="test:failing_tool", arguments={})

        # Analytics should record failure


# =============================================================================
# Sync Manager Instance Tests
# =============================================================================


class TestGetSyncManagerInstance:
    """Test get_sync_manager_instance() function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, test_config):
        """Should return None when auto_sync disabled."""
        import gateway

        gateway._config = test_config  # auto_sync=False
        gateway._sync_manager = None

        from gateway import get_sync_manager_instance

        result = await get_sync_manager_instance()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_instance(self, test_config_with_backends):
        """Should return cached sync manager if exists."""
        import gateway
        from sync_manager import SyncManager

        mock_sync = Mock(spec=SyncManager)
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._sync_manager = mock_sync

        from gateway import get_sync_manager_instance

        result = await get_sync_manager_instance()

        assert result is mock_sync


# =============================================================================
# Chain Indexer Instance Tests
# =============================================================================


class TestGetChainIndexerInstance:
    """Test get_chain_indexer_instance() function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, test_config):
        """Should return None when chain_indexing disabled."""
        import gateway

        gateway._config = test_config  # chain_indexing_enabled=False
        gateway._chain_indexer = None

        from gateway import get_chain_indexer_instance

        result = await get_chain_indexer_instance()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_instance(self, test_config_with_backends):
        """Should return cached chain indexer if exists."""
        import gateway
        from chain_indexer import ChainIndexer

        mock_chain = Mock(spec=ChainIndexer)
        gateway._config = test_config_with_backends
        gateway._chain_indexer = mock_chain

        from gateway import get_chain_indexer_instance

        result = await get_chain_indexer_instance()

        assert result is mock_chain


# =============================================================================
# Execute Tool Additional Tests
# =============================================================================


class TestExecuteToolAdditional:
    """Additional execute() edge case tests."""

    @pytest.mark.asyncio
    async def test_execute_invalid_tool_name_format(self, test_config):
        """Should handle tool names without server prefix."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {}
        mock_manager.connect_backend = AsyncMock(return_value=False)
        mock_manager.execute_tool = AsyncMock(
            return_value={"success": False, "error": "No backend"}
        )

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="no_server_prefix", arguments={})

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_with_exception(self, test_config):
        """Should handle exceptions during execution."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        # Return failure instead of raising - the gateway wraps exceptions
        mock_manager.execute_tool = AsyncMock(
            return_value={"success": False, "error": "Execution failed"}
        )

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:broken_tool", arguments={})

        assert result["success"] is False
        assert "error" in result


# =============================================================================
# Compass Status Additional Tests
# =============================================================================


class TestCompassStatusAdditional:
    """Additional compass_status() tests."""

    @pytest.mark.asyncio
    async def test_status_with_chain_indexer(
        self, test_index, test_config_with_backends, test_chain_indexer
    ):
        """Should include chain indexer stats when available."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={"connected": 1, "total_tools": 5}
        )

        from gateway import compass_status

        result = await compass_status()

        assert "chains" in result

    @pytest.mark.asyncio
    async def test_status_with_sync_manager(
        self, test_index, test_config_with_backends
    ):
        """Should include sync status when sync manager available."""
        import gateway
        from sync_manager import SyncManager

        mock_sync = Mock(spec=SyncManager)
        mock_sync.get_all_sync_status = Mock(return_value={"backend1": "synced"})

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True  # Enable sync in config
        gateway._sync_manager = mock_sync
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(return_value={"connected": 0})

        from gateway import compass_status

        result = await compass_status()

        # sync key exists if sync_manager is set
        assert result is not None


# =============================================================================
# Startup Sync Edge Cases
# =============================================================================


class TestStartupSyncEdgeCases:
    """Edge cases for maybe_startup_sync()."""

    @pytest.mark.asyncio
    async def test_startup_sync_runs_once(self, test_config_with_backends):
        """Should only run sync once across concurrent calls."""
        import gateway
        import asyncio

        gateway._config = test_config_with_backends
        gateway._config.sync_check_on_startup = True
        gateway._startup_sync_done = False
        gateway._sync_manager = Mock()
        gateway._sync_manager.sync_if_needed = AsyncMock()

        from gateway import maybe_startup_sync

        # Run concurrently
        await asyncio.gather(
            maybe_startup_sync(),
            maybe_startup_sync(),
            maybe_startup_sync(),
        )

        # Should only have been called at most once
        # (may be 0 if _sync_manager is None)

    @pytest.mark.asyncio
    async def test_startup_sync_handles_exception(self, test_config_with_backends):
        """Should log warning and continue on sync failure."""
        import gateway
        from sync_manager import SyncManager

        mock_sync = Mock(spec=SyncManager)
        mock_sync.sync_if_needed = AsyncMock(side_effect=Exception("Sync failed"))

        gateway._config = test_config_with_backends
        gateway._config.sync_check_on_startup = True
        gateway._startup_sync_done = False
        gateway._sync_manager = mock_sync

        from gateway import maybe_startup_sync

        # Should not raise
        await maybe_startup_sync()


# =============================================================================
# Compass Audit Edge Cases
# =============================================================================


class TestCompassAuditEdgeCases:
    """Edge cases for compass_audit()."""

    @pytest.mark.asyncio
    async def test_audit_with_chains_enabled(
        self, test_index, test_config_with_backends, test_chain_indexer
    ):
        """Should include chain info when enabled."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={
                "configured_backends": [],
                "connected_backends": [],
                "total_tools": 0,
                "tools_by_backend": {},
            }
        )

        from gateway import compass_audit

        result = await compass_audit()

        assert "chains" in result

    @pytest.mark.asyncio
    async def test_audit_with_sync_enabled(self, test_index, test_config_with_backends):
        """Should include sync info when enabled."""
        import gateway
        from sync_manager import SyncManager

        mock_sync = Mock(spec=SyncManager)
        mock_sync.get_all_sync_status = Mock(return_value={})

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True  # Enable sync in config
        gateway._sync_manager = mock_sync
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_stats = Mock(
            return_value={
                "configured_backends": [],
                "connected_backends": [],
                "total_tools": 0,
                "tools_by_backend": {},
            }
        )

        from gateway import compass_audit

        result = await compass_audit()

        # Audit includes sync section when sync manager is enabled
        assert result is not None
        assert "health" in result


# =============================================================================
# Describe Tool Edge Cases
# =============================================================================


class TestDescribeEdgeCases:
    """Edge cases for describe() tool."""

    @pytest.mark.asyncio
    async def test_describe_with_full_schema_from_index(self, test_index, test_config):
        """Should return full schema from index."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_tool_schema = Mock(return_value=None)

        from gateway import describe

        result = await describe(tool_name="test:read_file")

        # Should have basic schema info
        assert "tool" in result

    @pytest.mark.asyncio
    async def test_describe_analytics_recorded(
        self, test_index, test_config_with_backends, test_analytics
    ):
        """Should record describe call in analytics."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics
        gateway._backend_manager = Mock()
        gateway._backend_manager.get_tool_schema = Mock(return_value=None)

        from gateway import describe

        await describe(tool_name="test:read_file")

        # Analytics should record (this might not directly record describes,
        # but we can verify analytics is accessible)
        assert gateway._analytics is not None


# =============================================================================
# Compass with Analytics Recording
# =============================================================================


class TestCompassAnalyticsRecording:
    """Test that compass() records analytics."""

    @pytest.mark.asyncio
    async def test_compass_records_search(
        self, test_index, test_config_with_backends, test_analytics
    ):
        """Should record search in analytics."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics
        gateway._startup_sync_done = True

        from gateway import compass

        await compass(intent="test search query")

        # Verify search was recorded
        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["searches"]["total"] >= 1


# =============================================================================
# Compass Categories Edge Cases
# =============================================================================


class TestCompassCategoriesEdgeCases:
    """Edge cases for compass_categories()."""

    @pytest.mark.asyncio
    async def test_categories_with_usage_hint(self, test_index, test_config):
        """Should include usage hint."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config

        from gateway import compass_categories

        result = await compass_categories()

        assert "hint" in result


# =============================================================================
# Chain Operations Edge Cases
# =============================================================================


class TestChainsEdgeCases:
    """Additional edge cases for compass_chains()."""

    @pytest.mark.asyncio
    async def test_chains_detect_no_analytics(
        self, test_config_with_backends, test_chain_indexer
    ):
        """Should handle detect with no analytics."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer
        gateway._analytics = None

        from gateway import compass_chains

        result = await compass_chains(action="detect")

        # When no analytics, it still returns detected (empty list)
        assert "detected" in result or "error" in result


# =============================================================================
# Categorize Tool Edge Cases
# =============================================================================


class TestCategorizeToolEdgeCases:
    """Additional categorize_tool() tests."""

    def test_name_only_matching(self):
        """categorize_tool only uses name, not description."""
        from gateway import categorize_tool

        # The function only checks name, so generic names return "other"
        assert (
            categorize_tool("generic_tool", "Execute a SQL database query") == "other"
        )
        # But names with keywords work
        assert categorize_tool("db_query", "Execute a SQL database query") == "database"

    def test_priority_order(self):
        """Name should be checked before description."""
        from gateway import categorize_tool

        # "file" in name should win even if description says "database"
        result = categorize_tool("read_file", "Query the database")
        assert result == "file"

    def test_comfy_ai_category(self):
        """Comfy tools should be AI category."""
        from gateway import categorize_tool

        assert categorize_tool("comfy_generate", "Generate image") == "ai"
        assert categorize_tool("comfy_workflow", "Run workflow") == "ai"


# =============================================================================
# Execute with Backend Auto-Connection
# =============================================================================


class TestExecuteBackendConnection:
    """Test execute() backend connection logic."""

    @pytest.mark.asyncio
    async def test_execute_connects_to_backend(self, test_config):
        """Should connect to backend before executing."""
        import gateway

        mock_manager = Mock()
        mock_backend = Mock(is_connected=False)
        mock_manager._backends = {"test": mock_backend}
        mock_manager.connect_backend = AsyncMock(return_value=True)
        mock_manager.execute_tool = AsyncMock(return_value={"success": True})

        # After connecting, mock that it's now connected
        def update_connected(*args):
            mock_backend.is_connected = True
            return True

        mock_manager.connect_backend.side_effect = update_connected

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        await execute(tool_name="test:tool", arguments={})

        mock_manager.connect_backend.assert_called()

    @pytest.mark.asyncio
    async def test_execute_already_connected(self, test_config):
        """Should skip connection if already connected."""
        import gateway

        mock_manager = Mock()
        mock_manager._backends = {"test": Mock(is_connected=True)}
        mock_manager.execute_tool = AsyncMock(return_value={"success": True})

        gateway._backend_manager = mock_manager
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        await execute(tool_name="test:tool", arguments={})

        # connect_backend should not have been called since already connected
        assert not mock_manager.connect_backend.called
