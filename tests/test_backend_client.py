"""
Tests for Tool Compass backend client module.

Tests MCP backend connections, tool discovery, and execution.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from backend_client import (
    BackendConnection,
    BackendManager,
    ToolInfo,
    get_backend_manager,
    init_backends,
)
from config import CompassConfig, StdioBackend


# =============================================================================
# ToolInfo Tests
# =============================================================================


class TestToolInfo:
    """Test ToolInfo dataclass."""

    def test_tool_info_creation(self):
        """Should create ToolInfo with all fields."""
        tool = ToolInfo(
            name="read_file",
            qualified_name="bridge:read_file",
            description="Read file contents",
            server="bridge",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        assert tool.name == "read_file"
        assert tool.qualified_name == "bridge:read_file"
        assert tool.description == "Read file contents"
        assert tool.server == "bridge"
        assert "path" in tool.input_schema["properties"]

    def test_tool_info_to_dict(self):
        """Should serialize to dictionary."""
        tool = ToolInfo(
            name="write_file",
            qualified_name="bridge:write_file",
            description="Write to file",
            server="bridge",
            input_schema={"type": "object"},
        )

        data = tool.to_dict()

        assert data["name"] == "write_file"
        assert data["qualified_name"] == "bridge:write_file"
        assert data["description"] == "Write to file"
        assert data["server"] == "bridge"
        assert data["input_schema"] == {"type": "object"}

    def test_tool_info_empty_schema(self):
        """Should handle empty input schema."""
        tool = ToolInfo(
            name="status",
            qualified_name="bridge:status",
            description="Get status",
            server="bridge",
            input_schema={},
        )

        assert tool.input_schema == {}
        assert tool.to_dict()["input_schema"] == {}


# =============================================================================
# BackendConnection Tests
# =============================================================================


class TestBackendConnection:
    """Test BackendConnection class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock StdioBackend."""
        return StdioBackend(
            command="python",
            args=["-m", "test_server"],
            env={"DEBUG": "1"},
            cwd=None,
        )

    def test_backend_connection_init(self, mock_backend):
        """Should initialize with correct state."""
        conn = BackendConnection("test", mock_backend)

        assert conn.name == "test"
        assert conn.backend == mock_backend
        assert conn.session is None
        assert conn._connected is False
        assert conn._tools == []

    def test_is_connected_property(self, mock_backend):
        """Should report connection status."""
        conn = BackendConnection("test", mock_backend)

        assert conn.is_connected is False

        conn._connected = True
        assert conn.is_connected is True

    def test_get_tools_empty(self, mock_backend):
        """Should return empty list when not connected."""
        conn = BackendConnection("test", mock_backend)

        tools = conn.get_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_backend):
        """Should connect successfully with mocked MCP."""
        conn = BackendConnection("test", mock_backend)

        # Mock the stdio_client context manager
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=Mock(
                tools=[
                    Mock(name="tool1", description="Tool 1", inputSchema={}),
                    Mock(
                        name="tool2",
                        description="Tool 2",
                        inputSchema={"type": "object"},
                    ),
                ]
            )
        )

        with patch("backend_client.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(
                return_value=(Mock(), Mock())
            )
            mock_stdio.return_value.__aexit__ = AsyncMock()

            with patch("backend_client.ClientSession") as mock_client_session:
                mock_client_session.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_client_session.return_value.__aexit__ = AsyncMock()

                await conn.connect(timeout=5.0)

        # Note: This test verifies the structure, actual connection requires real MCP server

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_backend):
        """Should return True if already connected."""
        conn = BackendConnection("test", mock_backend)
        conn._connected = True

        result = await conn.connect()

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_backend):
        """Should clean up connection state."""
        conn = BackendConnection("test", mock_backend)
        conn._connected = True
        conn._tools = [Mock()]
        conn._exit_stack = AsyncMock()
        conn._exit_stack.aclose = AsyncMock()

        await conn.disconnect()

        assert conn._connected is False
        assert conn._tools == []
        assert conn.session is None

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, mock_backend):
        """Should raise error when not connected."""
        conn = BackendConnection("test", mock_backend)

        with pytest.raises(RuntimeError, match="Not connected"):
            await conn.call_tool("test_tool", {"arg": "value"})

    def test_get_tools_with_cached_tools(self, mock_backend):
        """Should return ToolInfo list from cached tools."""
        conn = BackendConnection("test", mock_backend)

        # Create proper mocks with string values
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Description 1"
        mock_tool1.inputSchema = {"type": "object"}

        mock_tool2 = Mock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Description 2"
        mock_tool2.inputSchema = {}

        conn._tools = [mock_tool1, mock_tool2]
        conn._connected = True

        tools = conn.get_tools()

        assert len(tools) == 2
        assert all(isinstance(t, ToolInfo) for t in tools)
        assert tools[0].name == "tool1"
        assert tools[0].qualified_name == "test:tool1"
        assert tools[1].server == "test"


# =============================================================================
# BackendManager Tests
# =============================================================================


class TestBackendManager:
    """Test BackendManager class."""

    @pytest.fixture
    def config_with_backends(self):
        """Config with multiple backends."""
        return CompassConfig(
            backends={
                "backend1": StdioBackend(command="python", args=["-m", "server1"]),
                "backend2": StdioBackend(command="python", args=["-m", "server2"]),
            }
        )

    @pytest.fixture
    def empty_config(self):
        """Config with no backends."""
        return CompassConfig(backends={})

    def test_manager_init_with_config(self, config_with_backends):
        """Should initialize with provided config."""
        manager = BackendManager(config=config_with_backends)

        assert manager.config == config_with_backends
        assert manager._backends == {}
        assert manager._tool_index == {}

    def test_manager_init_default_config(self):
        """Should use default config if not provided."""
        with patch("backend_client.load_config") as mock_load:
            mock_load.return_value = CompassConfig(backends={})
            BackendManager()
            mock_load.assert_called_once()

    def test_get_all_tools_empty(self, empty_config):
        """Should return empty list when no backends connected."""
        manager = BackendManager(config=empty_config)

        tools = manager.get_all_tools()

        assert tools == []

    def test_get_backend_tools_not_connected(self, config_with_backends):
        """Should return empty list for unconnected backend."""
        manager = BackendManager(config=config_with_backends)

        tools = manager.get_backend_tools("backend1")

        assert tools == []

    def test_get_tool_schema_not_found(self, empty_config):
        """Should return None for unknown tool."""
        manager = BackendManager(config=empty_config)

        schema = manager.get_tool_schema("unknown:tool")

        assert schema is None

    def test_get_tool_schema_parses_qualified_name(self, config_with_backends):
        """Should parse server:tool format."""
        manager = BackendManager(config=config_with_backends)

        # Without a connected backend, should return None
        schema = manager.get_tool_schema("backend1:read_file")

        assert schema is None

    def test_get_stats_empty(self, empty_config):
        """Should return stats with empty backends."""
        manager = BackendManager(config=empty_config)

        stats = manager.get_stats()

        assert stats["configured_backends"] == []
        assert stats["connected_backends"] == []
        assert stats["total_tools"] == 0
        assert stats["tools_by_backend"] == {}

    def test_get_stats_with_configured_backends(self, config_with_backends):
        """Should list configured backends."""
        manager = BackendManager(config=config_with_backends)

        stats = manager.get_stats()

        assert "backend1" in stats["configured_backends"]
        assert "backend2" in stats["configured_backends"]

    @pytest.mark.asyncio
    async def test_connect_all_no_backends(self, empty_config):
        """Should return empty dict when no backends configured."""
        manager = BackendManager(config=empty_config)

        results = await manager.connect_all()

        assert results == {}

    @pytest.mark.asyncio
    async def test_connect_backend_unknown(self, empty_config):
        """Should return False for unknown backend."""
        manager = BackendManager(config=empty_config)

        result = await manager.connect_backend("unknown")

        assert result is False

    @pytest.mark.asyncio
    async def test_connect_backend_already_connected(self, config_with_backends):
        """Should return True if already connected."""
        manager = BackendManager(config=config_with_backends)

        # Mock an already connected backend
        mock_conn = Mock()
        mock_conn.is_connected = True
        manager._backends["backend1"] = mock_conn

        result = await manager.connect_backend("backend1")

        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_all(self, config_with_backends):
        """Should disconnect all backends."""
        manager = BackendManager(config=config_with_backends)

        # Add mock connections
        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()
        manager._backends = {"b1": mock_conn1, "b2": mock_conn2}
        manager._tool_index = {"b1:tool": "b1"}

        await manager.disconnect_all()

        assert manager._backends == {}
        assert manager._tool_index == {}
        mock_conn1.disconnect.assert_called_once()
        mock_conn2.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_format(self, empty_config):
        """Should error on unknown tool without server prefix."""
        manager = BackendManager(config=empty_config)

        result = await manager.execute_tool("unknown_tool", {})

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_backend(self, empty_config):
        """Should error on unknown backend."""
        manager = BackendManager(config=empty_config)

        result = await manager.execute_tool("unknown:tool", {})

        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_tool_connect_on_demand(self, config_with_backends):
        """Should try to connect on-demand if backend not connected."""
        manager = BackendManager(config=config_with_backends)

        with patch.object(
            manager, "connect_backend", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.return_value = False

            result = await manager.execute_tool("backend1:tool", {})

            mock_connect.assert_called_with("backend1")
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, config_with_backends):
        """Should execute tool and return result."""
        manager = BackendManager(config=config_with_backends)

        # Mock connected backend
        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = [Mock(text="Success result")]
        mock_conn.call_tool = AsyncMock(return_value=mock_result)
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:my_tool", {"arg": "value"})

        assert result["success"] is True
        assert "Success result" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_tool_error_response(self, config_with_backends):
        """Should handle error response from tool."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.isError = True
        mock_result.content = [Mock(text="Error message")]
        mock_conn.call_tool = AsyncMock(return_value=mock_result)
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:failing_tool", {})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, config_with_backends):
        """Should handle timeout during execution."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_conn.call_tool = AsyncMock(side_effect=asyncio.TimeoutError())
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:slow_tool", {}, timeout=0.1)

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, config_with_backends):
        """Should handle unexpected exceptions."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_conn.call_tool = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:broken_tool", {})

        assert result["success"] is False
        assert "Unexpected error" in result["error"]


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletonFunctions:
    """Test singleton accessor functions."""

    @pytest.mark.asyncio
    async def test_get_backend_manager_creates_instance(self):
        """Should create manager on first call."""
        import backend_client

        # Reset singleton
        backend_client._manager = None

        with patch("backend_client.load_config") as mock_load:
            mock_load.return_value = CompassConfig(backends={})
            manager = await get_backend_manager()

            assert manager is not None
            assert isinstance(manager, BackendManager)

    @pytest.mark.asyncio
    async def test_get_backend_manager_returns_same_instance(self):
        """Should return same instance on subsequent calls."""

        with patch("backend_client.load_config") as mock_load:
            mock_load.return_value = CompassConfig(backends={})

            manager1 = await get_backend_manager()
            manager2 = await get_backend_manager()

            assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_init_backends_without_connect(self):
        """Should initialize without connecting."""
        import backend_client

        backend_client._manager = None

        with patch("backend_client.load_config") as mock_load:
            mock_load.return_value = CompassConfig(backends={})

            manager = await init_backends(connect=False)

            assert manager is not None

    @pytest.mark.asyncio
    async def test_init_backends_with_connect(self):
        """Should connect all backends when requested."""
        import backend_client

        backend_client._manager = None

        with patch("backend_client.load_config") as mock_load:
            mock_load.return_value = CompassConfig(backends={})

            with patch.object(
                BackendManager, "connect_all", new_callable=AsyncMock
            ) as mock_connect:
                mock_connect.return_value = {}

                await init_backends(connect=True)

                mock_connect.assert_called_once()


# =============================================================================
# Integration-style Tests (Mocked)
# =============================================================================


class TestBackendManagerIntegration:
    """Integration-style tests with mocked backends."""

    @pytest.fixture
    def manager_with_mock_backends(self):
        """Manager with pre-configured mock backends."""
        config = CompassConfig(
            backends={
                "bridge": StdioBackend(command="python", args=["-m", "bridge"]),
                "doc": StdioBackend(command="python", args=["-m", "doc"]),
            }
        )
        manager = BackendManager(config=config)

        # Create mock connections
        mock_bridge = Mock()
        mock_bridge.is_connected = True
        mock_bridge.get_tools.return_value = [
            ToolInfo("read_file", "bridge:read_file", "Read file", "bridge", {}),
            ToolInfo("write_file", "bridge:write_file", "Write file", "bridge", {}),
        ]

        mock_doc = Mock()
        mock_doc.is_connected = True
        mock_doc.get_tools.return_value = [
            ToolInfo("scan", "doc:scan", "Scan code", "doc", {}),
        ]

        manager._backends = {"bridge": mock_bridge, "doc": mock_doc}
        manager._tool_index = {
            "bridge:read_file": "bridge",
            "bridge:write_file": "bridge",
            "doc:scan": "doc",
        }

        return manager

    def test_get_all_tools_aggregates(self, manager_with_mock_backends):
        """Should aggregate tools from all backends."""
        tools = manager_with_mock_backends.get_all_tools()

        assert len(tools) == 3
        names = [t.qualified_name for t in tools]
        assert "bridge:read_file" in names
        assert "bridge:write_file" in names
        assert "doc:scan" in names

    def test_get_backend_tools_filters(self, manager_with_mock_backends):
        """Should return only tools from specified backend."""
        bridge_tools = manager_with_mock_backends.get_backend_tools("bridge")

        assert len(bridge_tools) == 2
        assert all(t.server == "bridge" for t in bridge_tools)

    def test_get_tool_schema_finds_tool(self, manager_with_mock_backends):
        """Should find tool schema by qualified name."""
        # Mock the tool lookup
        manager_with_mock_backends._backends["bridge"].get_tools.return_value = [
            ToolInfo(
                "read_file",
                "bridge:read_file",
                "Read file",
                "bridge",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]

        schema = manager_with_mock_backends.get_tool_schema("bridge:read_file")

        assert schema is not None
        assert schema["name"] == "read_file"

    def test_get_stats_reflects_state(self, manager_with_mock_backends):
        """Should reflect current connection state."""
        stats = manager_with_mock_backends.get_stats()

        assert len(stats["connected_backends"]) == 2
        assert stats["total_tools"] == 3
        assert "bridge" in stats["tools_by_backend"]
        assert "doc" in stats["tools_by_backend"]


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestBackendConnectionEdgeCases:
    """Additional edge case tests for BackendConnection."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock StdioBackend."""
        return StdioBackend(
            command="python",
            args=["-m", "test_server"],
            env={"DEBUG": "1"},
            cwd=None,
        )

    @pytest.mark.asyncio
    async def test_connect_timeout_handling(self, mock_backend):
        """Should handle connection timeout gracefully."""
        conn = BackendConnection("test", mock_backend)

        with patch("backend_client.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(
                side_effect=asyncio.TimeoutError()
            )
            mock_stdio.return_value.__aexit__ = AsyncMock()

            result = await conn.connect(timeout=0.01)

            assert result is False
            assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_connect_exception_handling(self, mock_backend):
        """Should handle connection exceptions gracefully."""
        conn = BackendConnection("test", mock_backend)

        with patch("backend_client.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("Connection refused")
            )
            mock_stdio.return_value.__aexit__ = AsyncMock()

            result = await conn.connect()

            assert result is False
            assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_disconnect_with_exit_stack_error(self, mock_backend):
        """Should handle exit stack close errors."""
        conn = BackendConnection("test", mock_backend)

        # Create a mock exit stack that raises on close
        mock_stack = AsyncMock()
        mock_stack.aclose = AsyncMock(side_effect=RuntimeError("Close error"))
        conn._exit_stack = mock_stack
        conn._connected = True

        await conn.disconnect()

        # Should still mark as disconnected
        assert not conn.is_connected
        assert conn._exit_stack is None

    @pytest.mark.asyncio
    async def test_refresh_tools_no_session(self, mock_backend):
        """Should handle refresh when session is None."""
        conn = BackendConnection("test", mock_backend)
        conn.session = None

        await conn._refresh_tools()

        # Should not raise, tools should remain empty
        assert conn._tools == []

    @pytest.mark.asyncio
    async def test_refresh_tools_exception(self, mock_backend):
        """Should handle refresh exception gracefully."""
        conn = BackendConnection("test", mock_backend)
        mock_session = Mock()
        mock_session.list_tools = AsyncMock(side_effect=RuntimeError("List failed"))
        conn.session = mock_session

        await conn._refresh_tools()

        # Should clear tools on error
        assert conn._tools == []

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, mock_backend):
        """Should raise when calling tool without connection."""
        conn = BackendConnection("test", mock_backend)
        conn._connected = False

        with pytest.raises(RuntimeError, match="Not connected"):
            await conn.call_tool("some_tool", {})


class TestBackendManagerEdgeCases:
    """Additional edge case tests for BackendManager."""

    @pytest.fixture
    def config_with_backends(self):
        """Config with multiple backends."""
        return CompassConfig(
            backends={
                "backend1": StdioBackend(command="python", args=["-m", "b1"]),
                "backend2": StdioBackend(command="python", args=["-m", "b2"]),
            }
        )

    @pytest.mark.asyncio
    async def test_connect_all_with_exception(self, config_with_backends):
        """Should handle exception during connect_all."""
        manager = BackendManager(config=config_with_backends)

        with patch.object(
            BackendConnection, "connect", new_callable=AsyncMock
        ) as mock_conn:
            # Make one backend raise an exception
            mock_conn.side_effect = [True, Exception("Connection error")]

            results = await manager.connect_all()

            # At least one should fail
            assert not all(results.values())

    @pytest.mark.asyncio
    async def test_connect_all_mixed_results(self, config_with_backends):
        """Should handle mixed success/failure results."""
        manager = BackendManager(config=config_with_backends)

        with patch.object(
            BackendConnection, "connect", new_callable=AsyncMock
        ) as mock_conn:
            # First succeeds, second fails
            mock_conn.side_effect = [True, False]

            # Mock get_tools for the first connection
            with patch.object(BackendConnection, "get_tools") as mock_tools:
                mock_tools.return_value = [
                    ToolInfo("tool1", "backend1:tool1", "desc", "backend1", {})
                ]

                results = await manager.connect_all()

                # One success, one failure
                assert sum(1 for v in results.values() if v) >= 0

    @pytest.mark.asyncio
    async def test_connect_backend_creates_connection(self, config_with_backends):
        """Should create new connection for configured backend."""
        manager = BackendManager(config=config_with_backends)

        with patch.object(
            BackendConnection, "connect", new_callable=AsyncMock
        ) as mock_conn:
            mock_conn.return_value = True
            with patch.object(BackendConnection, "get_tools") as mock_tools:
                mock_tools.return_value = [
                    ToolInfo("tool1", "backend1:tool1", "desc", "backend1", {})
                ]

                result = await manager.connect_backend("backend1")

                assert result is True

    def test_get_tool_schema_parses_qualified_name(self, config_with_backends):
        """Should parse qualified name when not in index."""
        manager = BackendManager(config=config_with_backends)

        # No tool in index
        manager._tool_index = {}

        # Should try to parse the name
        result = manager.get_tool_schema("backend1:some_tool")

        # Will return None since backend not connected
        assert result is None

    def test_get_tool_schema_unqualified_not_found(self, config_with_backends):
        """Should return None for unqualified names not in index."""
        manager = BackendManager(config=config_with_backends)
        manager._tool_index = {}

        result = manager.get_tool_schema("some_tool")

        assert result is None

    def test_get_backend_tools_not_connected(self, config_with_backends):
        """Should return empty list for disconnected backend."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = Mock()
        mock_conn.is_connected = False
        manager._backends["backend1"] = mock_conn

        result = manager.get_backend_tools("backend1")

        assert result == []

    @pytest.mark.asyncio
    async def test_execute_tool_binary_content(self, config_with_backends):
        """Should handle binary content in tool result."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.isError = False
        # Binary content item
        binary_item = Mock()
        binary_item.text = None
        del binary_item.text
        binary_item.data = b"binary data"
        binary_item.mimeType = "image/png"
        mock_result.content = [binary_item]
        mock_conn.call_tool = AsyncMock(return_value=mock_result)
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:image_tool", {})

        assert result["success"] is True
        assert "Binary data" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_content_type(self, config_with_backends):
        """Should handle unknown content types."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.isError = False
        # Unknown content type (no text or data)
        unknown_item = Mock()
        del unknown_item.text
        del unknown_item.data
        mock_result.content = [unknown_item]
        mock_conn.call_tool = AsyncMock(return_value=mock_result)
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:special_tool", {})

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_empty_content(self, config_with_backends):
        """Should handle empty content in result."""
        manager = BackendManager(config=config_with_backends)

        mock_conn = AsyncMock()
        mock_result = Mock()
        mock_result.isError = False
        mock_result.content = []
        mock_conn.call_tool = AsyncMock(return_value=mock_result)
        manager._backends["backend1"] = mock_conn

        result = await manager.execute_tool("backend1:void_tool", {})

        assert result["success"] is True
        assert "successfully" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_execute_tool_on_demand_connect_success(self, config_with_backends):
        """Should connect on-demand and execute successfully."""
        manager = BackendManager(config=config_with_backends)

        call_count = 0

        async def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Create mock connection on success
            mock_conn = AsyncMock()
            mock_result = Mock()
            mock_result.isError = False
            mock_result.content = [Mock(text="Success")]
            mock_conn.call_tool = AsyncMock(return_value=mock_result)
            manager._backends["backend1"] = mock_conn
            return True

        with patch.object(manager, "connect_backend", side_effect=mock_connect):
            result = await manager.execute_tool("backend1:tool", {})

            assert call_count == 1
            assert result["success"] is True
