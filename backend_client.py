"""
Tool Compass - Backend Client Module
Manages connections to MCP backend servers via stdio subprocess.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult

from config import CompassConfig, StdioBackend, load_config

logger = logging.getLogger(__name__)

# Timeout constants (in seconds)
CONNECTION_TIMEOUT = 30  # Max time to establish backend connection
TOOL_CALL_TIMEOUT = 60  # Max time for a single tool execution


@dataclass
class ToolInfo:
    """Normalized tool information from a backend."""

    name: str  # Original tool name
    qualified_name: str  # server:tool_name format
    description: str
    server: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "description": self.description,
            "server": self.server,
            "input_schema": self.input_schema,
        }


class BackendConnection:
    """Manages a single MCP backend server connection."""

    def __init__(self, name: str, backend: StdioBackend):
        self.name = name
        self.backend = backend
        self.session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._tools: List[Tool] = []
        self._connected = False

    async def connect(self, timeout: Optional[float] = None) -> bool:
        """
        Establish connection to the backend server.

        Args:
            timeout: Connection timeout in seconds. Defaults to CONNECTION_TIMEOUT.
        """
        if self._connected:
            return True

        timeout = timeout or CONNECTION_TIMEOUT

        try:
            logger.info(f"Connecting to backend: {self.name} (timeout={timeout}s)")

            # Build environment
            env = dict(self.backend.env) if self.backend.env else None

            # Create server parameters
            server_params = StdioServerParameters(
                command=self.backend.command,
                args=self.backend.args,
                env=env,
                cwd=self.backend.cwd,
            )

            # Setup connection with timeout protection
            async def _establish_connection():
                self._exit_stack = AsyncExitStack()
                stdio_transport = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read_stream, write_stream = stdio_transport

                self.session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                # Initialize the session
                await self.session.initialize()

                # Cache tools
                await self._refresh_tools()

            await asyncio.wait_for(_establish_connection(), timeout=timeout)
            self._connected = True

            logger.info(f"Connected to {self.name}: {len(self._tools)} tools available")
            return True

        except asyncio.TimeoutError:
            logger.error(f"Connection to {self.name} timed out after {timeout}s")
            await self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            await self.disconnect()
            return False

    async def disconnect(self):
        """Close the connection."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing connection to {self.name}: {e}")
            self._exit_stack = None
        self.session = None
        self._connected = False
        self._tools = []

    async def _refresh_tools(self):
        """Refresh the cached tool list."""
        if not self.session:
            return

        try:
            result = await self.session.list_tools()
            self._tools = result.tools
        except Exception as e:
            logger.error(f"Failed to list tools from {self.name}: {e}")
            self._tools = []

    def get_tools(self) -> List[ToolInfo]:
        """Get normalized tool info list."""
        tools = []
        for tool in self._tools:
            tools.append(
                ToolInfo(
                    name=tool.name,
                    qualified_name=f"{self.name}:{tool.name}",
                    description=tool.description or "",
                    server=self.name,
                    input_schema=tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                )
            )
        return tools

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Call a tool on this backend."""
        if not self.session or not self._connected:
            raise RuntimeError(f"Not connected to backend: {self.name}")

        return await self.session.call_tool(tool_name, arguments)

    @property
    def is_connected(self) -> bool:
        return self._connected


class BackendManager:
    """
    Manages multiple MCP backend connections.
    Acts as the routing layer for tool discovery and execution.
    """

    def __init__(self, config: Optional[CompassConfig] = None):
        self.config = config or load_config()
        self._backends: Dict[str, BackendConnection] = {}
        self._tool_index: Dict[str, str] = {}  # qualified_name -> server_name

    async def connect_all(self, timeout: Optional[float] = None) -> Dict[str, bool]:
        """
        Connect to all configured backends concurrently.

        Args:
            timeout: Per-backend connection timeout. Defaults to CONNECTION_TIMEOUT.

        Returns:
            Dict mapping backend name to connection success status.
        """
        results = {}
        timeout = timeout or CONNECTION_TIMEOUT

        # Build connection tasks for all stdio backends
        tasks = []
        backend_names = []
        connections = []

        for name, backend in self.config.backends.items():
            if isinstance(backend, StdioBackend):
                conn = BackendConnection(name, backend)
                connections.append(conn)
                backend_names.append(name)
                tasks.append(conn.connect(timeout=timeout))
            else:
                logger.warning(
                    f"Backend type not yet supported: {type(backend)} for {name}"
                )
                results[name] = False

        # Run all connections concurrently
        if tasks:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            for name, conn, outcome in zip(backend_names, connections, outcomes):
                if isinstance(outcome, Exception):
                    logger.error(f"Backend {name} connection failed: {outcome}")
                    results[name] = False
                elif outcome:  # True = success
                    results[name] = True
                    self._backends[name] = conn
                    for tool in conn.get_tools():
                        self._tool_index[tool.qualified_name] = name
                else:
                    results[name] = False

        return results

    async def connect_backend(self, name: str, timeout: Optional[float] = None) -> bool:
        """
        Connect to a specific backend.

        Args:
            name: Backend name to connect to.
            timeout: Connection timeout in seconds. Defaults to CONNECTION_TIMEOUT.
        """
        if name in self._backends and self._backends[name].is_connected:
            return True

        backend = self.config.backends.get(name)
        if not backend:
            logger.error(f"Unknown backend: {name}")
            return False

        if isinstance(backend, StdioBackend):
            conn = BackendConnection(name, backend)
            success = await conn.connect(timeout=timeout)
            if success:
                self._backends[name] = conn
                for tool in conn.get_tools():
                    self._tool_index[tool.qualified_name] = name
            return success

        return False

    async def disconnect_all(self):
        """Disconnect from all backends."""
        for conn in self._backends.values():
            await conn.disconnect()
        self._backends.clear()
        self._tool_index.clear()

    def get_all_tools(self) -> List[ToolInfo]:
        """Get all tools from all connected backends."""
        tools = []
        for conn in self._backends.values():
            tools.extend(conn.get_tools())
        return tools

    def get_backend_tools(self, backend_name: str) -> List[ToolInfo]:
        """Get tools from a specific backend."""
        conn = self._backends.get(backend_name)
        if not conn or not conn.is_connected:
            return []
        return conn.get_tools()

    def get_tool_schema(self, qualified_name: str) -> Optional[Dict[str, Any]]:
        """Get the full schema for a specific tool."""
        server_name = self._tool_index.get(qualified_name)
        if not server_name:
            # Try parsing the qualified name
            if ":" in qualified_name:
                server_name, tool_name = qualified_name.split(":", 1)
            else:
                return None

        conn = self._backends.get(server_name)
        if not conn:
            return None

        for tool in conn.get_tools():
            if (
                tool.qualified_name == qualified_name
                or tool.name == qualified_name.split(":")[-1]
            ):
                return tool.to_dict()

        return None

    async def execute_tool(
        self,
        qualified_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool by its qualified name (server:tool_name).

        Args:
            qualified_name: Tool name in 'server:tool' format.
            arguments: Tool arguments.
            timeout: Execution timeout in seconds. Defaults to TOOL_CALL_TIMEOUT.

        Returns:
            Dict with 'success', 'result' or 'error' keys.
        """
        timeout = timeout or TOOL_CALL_TIMEOUT

        # Parse qualified name
        if ":" in qualified_name:
            server_name, tool_name = qualified_name.split(":", 1)
        else:
            # Try to find in index
            server_name = self._tool_index.get(qualified_name)
            tool_name = qualified_name
            if not server_name:
                return {
                    "success": False,
                    "error": f"Tool not found: {qualified_name}. Use format 'server:tool_name'.",
                }

        # Get backend
        conn = self._backends.get(server_name)
        if not conn:
            # Try to connect on-demand
            if server_name in self.config.backends:
                success = await self.connect_backend(server_name)
                if not success:
                    return {
                        "success": False,
                        "error": f"Failed to connect to backend: {server_name}",
                    }
                conn = self._backends.get(server_name)
            else:
                return {
                    "success": False,
                    "error": f"Unknown backend: {server_name}",
                }

        # Execute with timeout protection
        try:
            result = await asyncio.wait_for(
                conn.call_tool(tool_name, arguments), timeout=timeout
            )

            # Parse result
            if result.isError:
                return {
                    "success": False,
                    "error": str(result.content)
                    if result.content
                    else "Tool returned error",
                }

            # Extract content
            content = []
            for item in result.content:
                if hasattr(item, "text"):
                    content.append(item.text)
                elif hasattr(item, "data"):
                    content.append(f"[Binary data: {item.mimeType}]")
                else:
                    content.append(str(item))

            return {
                "success": True,
                "result": "\n".join(content)
                if content
                else "Tool executed successfully",
            }

        except asyncio.TimeoutError:
            logger.error(f"Tool execution timed out after {timeout}s: {qualified_name}")
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout}s",
            }
        except Exception as e:
            logger.error(f"Error executing {qualified_name}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        connected = [name for name, conn in self._backends.items() if conn.is_connected]
        tool_count = len(self._tool_index)

        return {
            "configured_backends": list(self.config.backends.keys()),
            "connected_backends": connected,
            "total_tools": tool_count,
            "tools_by_backend": {
                name: len(conn.get_tools()) for name, conn in self._backends.items()
            },
        }


# Singleton instance
_manager: Optional[BackendManager] = None


async def get_backend_manager() -> BackendManager:
    """Get or create the global backend manager."""
    global _manager
    if _manager is None:
        _manager = BackendManager()
    return _manager


async def init_backends(connect: bool = True) -> BackendManager:
    """Initialize backends, optionally connecting to all."""
    manager = await get_backend_manager()
    if connect:
        await manager.connect_all()
    return manager
