"""
Tool Compass - Simple Backend Client
Uses subprocess directly with JSON-RPC to avoid anyio conflicts.

This module provides a robust, Windows-compatible MCP client that:
- Uses asyncio.create_subprocess_exec directly (avoids anyio task group issues)
- Implements connection pooling with keep-alive
- Has proper error handling and timeouts
- Supports graceful shutdown
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import CompassConfig, StdioBackend, load_config

logger = logging.getLogger(__name__)

# Timeout constants (in seconds)
CONNECTION_TIMEOUT = 10
TOOL_CALL_TIMEOUT = 15
KEEPALIVE_INTERVAL = 30  # Ping backends every 30s to keep connection alive
MAX_RETRIES = 2


@dataclass
class ToolInfo:
    """Normalized tool information from a backend."""
    name: str
    qualified_name: str
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


@dataclass
class ConnectionStats:
    """Track connection health metrics."""
    connected_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    total_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0

    def record_call(self, success: bool, latency_ms: float):
        self.last_used = datetime.now()
        self.total_calls += 1
        if not success:
            self.failed_calls += 1
        # Running average
        self.avg_latency_ms = (self.avg_latency_ms * (self.total_calls - 1) + latency_ms) / self.total_calls


class SimpleBackendConnection:
    """
    Simple MCP backend connection using subprocess directly.
    Avoids anyio task group conflicts by not using the MCP client library.

    Features:
    - Direct asyncio subprocess management
    - Connection keep-alive with periodic pings
    - Automatic reconnection on failure
    - Detailed error handling
    """

    def __init__(self, name: str, backend: StdioBackend):
        self.name = name
        self.backend = backend
        self._process: Optional[asyncio.subprocess.Process] = None
        self._tools: List[Dict[str, Any]] = []
        self._connected = False
        self._request_id = 0
        self._lock = asyncio.Lock()  # Serialize requests to prevent interleaving
        self._stats = ConnectionStats()
        self._stderr_task: Optional[asyncio.Task] = None

    async def connect(self, timeout: Optional[float] = None) -> bool:
        """Establish connection to the backend server."""
        if self._connected and self._process and self._process.returncode is None:
            return True

        timeout = timeout or CONNECTION_TIMEOUT

        try:
            logger.info(f"Connecting to backend: {self.name} (timeout={timeout}s)")

            # Build environment - inherit from current process and add extras
            env = os.environ.copy()
            if self.backend.env:
                env.update(self.backend.env)
            env.update({
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            })

            # Windows-specific: use CREATE_NO_WINDOW to prevent console popups
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            # Start subprocess
            self._process = await asyncio.create_subprocess_exec(
                self.backend.command,
                *self.backend.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.backend.cwd,
                creationflags=creationflags,
            )

            # Start stderr reader task (logs backend errors)
            self._stderr_task = asyncio.create_task(self._read_stderr())

            # Initialize MCP session with timeout
            init_result = await asyncio.wait_for(
                self._send_request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tool-compass", "version": "2.0"}
                }),
                timeout=timeout
            )

            if "error" in init_result:
                raise RuntimeError(f"Initialize failed: {init_result['error']}")

            # Send initialized notification
            await self._send_notification("notifications/initialized")

            # Get tools list
            tools_result = await asyncio.wait_for(
                self._send_request("tools/list", {}),
                timeout=timeout
            )

            if "result" in tools_result and "tools" in tools_result["result"]:
                self._tools = tools_result["result"]["tools"]

            self._connected = True
            self._stats.connected_at = datetime.now()
            self._stats.last_used = datetime.now()
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
        """Close the connection gracefully."""
        self._connected = False

        # Cancel stderr reader
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        # Terminate process
        if self._process:
            try:
                # Try graceful shutdown first
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception as e:
                logger.debug(f"Error during disconnect of {self.name}: {e}")
            self._process = None

        self._tools = []

    async def _read_stderr(self):
        """Read and log stderr from the backend process."""
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                # Log backend stderr at debug level
                logger.debug(f"[{self.name}] {line.decode('utf-8', errors='replace').rstrip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Stderr reader error for {self.name}: {e}")

    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        async with self._lock:  # Serialize requests
            if not self._process or not self._process.stdin or not self._process.stdout:
                raise RuntimeError("Not connected")

            if self._process.returncode is not None:
                raise RuntimeError(f"Process exited with code {self._process.returncode}")

            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params
            }

            # Write request
            request_str = json.dumps(request) + "\n"
            self._process.stdin.write(request_str.encode("utf-8"))
            await self._process.stdin.drain()

            # Read response - handle multiple lines (some responses are multi-line)
            response_line = await self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("No response from backend (EOF)")

            try:
                return json.loads(response_line.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {self.name}: {response_line[:100]}")
                raise RuntimeError(f"Invalid JSON response: {e}")

    async def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send a JSON-RPC notification (no response expected)."""
        async with self._lock:
            if not self._process or not self._process.stdin:
                raise RuntimeError("Not connected")

            notification: Dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": method,
            }
            if params:
                notification["params"] = params

            notification_str = json.dumps(notification) + "\n"
            self._process.stdin.write(notification_str.encode("utf-8"))
            await self._process.stdin.drain()

    def get_tools(self) -> List[ToolInfo]:
        """Get normalized tool info list."""
        tools = []
        for tool in self._tools:
            tools.append(ToolInfo(
                name=tool.get("name", ""),
                qualified_name=f"{self.name}:{tool.get('name', '')}",
                description=tool.get("description", ""),
                server=self.name,
                input_schema=tool.get("inputSchema", {}),
            ))
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on this backend with automatic reconnection."""
        if not self._connected:
            raise RuntimeError(f"Not connected to backend: {self.name}")

        start_time = asyncio.get_event_loop().time()

        try:
            result = await self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            if "error" in result:
                self._stats.record_call(False, latency_ms)
                return {
                    "success": False,
                    "error": result["error"].get("message", str(result["error"]))
                }

            if "result" in result:
                res = result["result"]
                if res.get("isError"):
                    self._stats.record_call(False, latency_ms)
                    content = res.get("content", [])
                    error_text = ""
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                error_text += item["text"]
                    return {
                        "success": False,
                        "error": error_text or "Tool returned error"
                    }

                # Extract text content
                content = []
                for item in res.get("content", []):
                    if isinstance(item, dict) and "text" in item:
                        content.append(item["text"])
                    elif isinstance(item, str):
                        content.append(item)
                    else:
                        content.append(str(item))

                self._stats.record_call(True, latency_ms)
                return {
                    "success": True,
                    "result": "\n".join(content) if content else "Tool executed successfully"
                }

            self._stats.record_call(False, latency_ms)
            return {"success": False, "error": "Invalid response from backend"}

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._stats.record_call(False, latency_ms)

            # Check if process died
            if self._process and self._process.returncode is not None:
                self._connected = False
                logger.warning(f"Backend {self.name} process died, will reconnect on next call")

            raise

    @property
    def is_connected(self) -> bool:
        # Check both flag and process health
        if not self._connected:
            return False
        if self._process and self._process.returncode is not None:
            self._connected = False
            return False
        return True

    @property
    def stats(self) -> ConnectionStats:
        return self._stats


class SimpleBackendManager:
    """
    Manages multiple MCP backend connections using simple subprocess approach.

    Features:
    - Connection pooling with keep-alive
    - Automatic reconnection on failure
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(self, config: Optional[CompassConfig] = None):
        self.config = config or load_config()
        self._backends: Dict[str, SimpleBackendConnection] = {}
        self._tool_index: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def connect_backend(self, name: str, timeout: Optional[float] = None) -> bool:
        """Connect to a specific backend with retry logic."""
        async with self._lock:
            # Check if already connected
            if name in self._backends and self._backends[name].is_connected:
                return True

            backend = self.config.backends.get(name)
            if not backend:
                logger.error(f"Unknown backend: {name}")
                return False

            if not isinstance(backend, StdioBackend):
                logger.error(f"Unsupported backend type for {name}")
                return False

            # Disconnect existing broken connection
            if name in self._backends:
                await self._backends[name].disconnect()

            # Try to connect with retries
            conn = SimpleBackendConnection(name, backend)

            for attempt in range(MAX_RETRIES + 1):
                success = await conn.connect(timeout=timeout)
                if success:
                    self._backends[name] = conn
                    for tool in conn.get_tools():
                        self._tool_index[tool.qualified_name] = name
                    return True

                if attempt < MAX_RETRIES:
                    logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} for backend {name}")
                    await asyncio.sleep(0.5)  # Brief pause before retry

            return False

    async def ensure_connected(self, name: str) -> bool:
        """Ensure a backend is connected, reconnecting if necessary."""
        if name in self._backends and self._backends[name].is_connected:
            return True
        return await self.connect_backend(name)

    async def connect_all(self, timeout: Optional[float] = None) -> Dict[str, bool]:
        """Connect to all configured backends.

        Returns:
            Dict mapping backend name to connection success status.
        """
        results = {}
        for name in self.config.backends.keys():
            try:
                success = await self.connect_backend(name, timeout=timeout)
                results[name] = success
            except Exception as e:
                logger.error(f"Failed to connect to {name}: {e}")
                results[name] = False
        return results

    async def disconnect_all(self):
        """Disconnect from all backends gracefully."""
        async with self._lock:
            tasks = [conn.disconnect() for conn in self._backends.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._backends.clear()
            self._tool_index.clear()

    def get_all_tools(self) -> List[ToolInfo]:
        """Get all tools from all connected backends."""
        tools = []
        for conn in self._backends.values():
            if conn.is_connected:
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
            if ":" in qualified_name:
                server_name = qualified_name.split(":", 1)[0]
            else:
                return None

        conn = self._backends.get(server_name)
        if not conn:
            return None

        for tool in conn.get_tools():
            if tool.qualified_name == qualified_name or tool.name == qualified_name.split(":")[-1]:
                return tool.to_dict()

        return None

    async def execute_tool(
        self,
        qualified_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a tool by its qualified name with automatic reconnection."""
        timeout = timeout or TOOL_CALL_TIMEOUT

        # Parse qualified name
        if ":" in qualified_name:
            server_name, tool_name = qualified_name.split(":", 1)
        else:
            server_name = self._tool_index.get(qualified_name)
            tool_name = qualified_name
            if not server_name:
                return {
                    "success": False,
                    "error": f"Tool not found: {qualified_name}. Use format 'server:tool_name'.",
                }

        # Ensure connected (with automatic reconnection)
        if not await self.ensure_connected(server_name):
            return {
                "success": False,
                "error": f"Failed to connect to backend: {server_name}",
            }

        conn = self._backends.get(server_name)
        if not conn:
            return {
                "success": False,
                "error": f"Backend not available: {server_name}",
            }

        try:
            return await asyncio.wait_for(
                conn.call_tool(tool_name, arguments),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool execution timed out after {timeout}s: {qualified_name}")
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout}s",
            }
        except Exception as e:
            logger.error(f"Error executing {qualified_name}: {e}")

            # Try one reconnect and retry
            logger.info(f"Attempting reconnect to {server_name}...")
            if await self.connect_backend(server_name):
                try:
                    return await asyncio.wait_for(
                        self._backends[server_name].call_tool(tool_name, arguments),
                        timeout=timeout
                    )
                except Exception as retry_error:
                    return {
                        "success": False,
                        "error": f"Retry failed: {retry_error}",
                    }

            return {
                "success": False,
                "error": str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics for all backends."""
        connected = []
        stats_by_backend = {}

        for name, conn in self._backends.items():
            if conn.is_connected:
                connected.append(name)
                stats_by_backend[name] = {
                    "tools": len(conn.get_tools()),
                    "total_calls": conn.stats.total_calls,
                    "failed_calls": conn.stats.failed_calls,
                    "avg_latency_ms": round(conn.stats.avg_latency_ms, 2),
                    "connected_at": conn.stats.connected_at.isoformat() if conn.stats.connected_at else None,
                    "last_used": conn.stats.last_used.isoformat() if conn.stats.last_used else None,
                }

        return {
            "configured_backends": list(self.config.backends.keys()),
            "connected_backends": connected,
            "total_tools": len(self._tool_index),
            "tools_by_backend": {
                name: len(conn.get_tools())
                for name, conn in self._backends.items()
                if conn.is_connected
            },
            "stats": stats_by_backend,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all backends."""
        health = {}
        for name in self.config.backends.keys():
            conn = self._backends.get(name)
            if conn and conn.is_connected:
                health[name] = {
                    "status": "connected",
                    "tools": len(conn.get_tools()),
                    "success_rate": round(
                        (1 - conn.stats.failed_calls / max(conn.stats.total_calls, 1)) * 100, 1
                    ),
                }
            else:
                health[name] = {"status": "disconnected"}
        return health
