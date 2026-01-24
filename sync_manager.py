"""
Tool Compass - Sync Manager
Detects backend changes and triggers index rebuilds.
"""

import asyncio
import hashlib
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from config import CompassConfig
    from indexer import CompassIndex
    from backend_client import BackendManager

logger = logging.getLogger(__name__)

# Use same analytics DB for sync state
ANALYTICS_DB_PATH = Path(__file__).parent / "db" / "compass_analytics.db"


class SyncManager:
    """
    Manages automatic synchronization with backend servers.

    Strategies:
    1. Hash-based change detection: Compare tool list hashes
    2. On-demand refresh: Check on first compass() call per session
    3. Background polling: Optional periodic check (configurable)
    """

    def __init__(
        self, config: "CompassConfig", index: "CompassIndex", backends: "BackendManager"
    ):
        self.config = config
        self.index = index
        self.backends = backends

        self._sync_lock = asyncio.Lock()
        self._last_check: Dict[str, datetime] = {}
        self._polling_task: Optional[asyncio.Task] = None
        self._db: Optional[sqlite3.Connection] = None

        # Ensure db directory exists
        ANALYTICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._db is None:
            self._db = sqlite3.connect(str(ANALYTICS_DB_PATH))
            self._db.row_factory = sqlite3.Row
            self._init_sync_table()
        return self._db

    def _init_sync_table(self):
        """Ensure sync state table exists."""
        db = self._get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS backend_sync_state (
                backend_name TEXT PRIMARY KEY,
                tool_count INTEGER,
                tool_hash TEXT,
                last_sync_at TIMESTAMP,
                sync_status TEXT DEFAULT 'unknown'
            )
        """)
        db.commit()

    def _compute_tool_hash(self, tools: List[Any]) -> str:
        """
        Compute deterministic hash of tool list for change detection.
        Uses sorted tool names to ensure consistency.
        """
        if not tools:
            return hashlib.sha256(b"empty").hexdigest()[:32]

        # Sort by qualified name for deterministic ordering
        sorted_names = sorted([t.qualified_name for t in tools])
        content = json.dumps(sorted_names, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get_stored_hash(self, backend_name: str) -> Optional[str]:
        """Get the stored tool hash for a backend."""
        db = self._get_db()
        row = db.execute(
            "SELECT tool_hash FROM backend_sync_state WHERE backend_name = ?",
            (backend_name,),
        ).fetchone()
        return row["tool_hash"] if row else None

    async def check_backend_changes(self, backend_name: str) -> bool:
        """
        Check if a backend's tools have changed since last sync.
        Returns True if changes detected.
        """
        # Connect to backend if needed
        if (
            backend_name not in self.backends._backends
            or not self.backends._backends[backend_name].is_connected
        ):
            success = await self.backends.connect_backend(backend_name)
            if not success:
                logger.warning(
                    f"Could not connect to backend {backend_name} for sync check"
                )
                return False

        # Get current tools
        tools = self.backends.get_backend_tools(backend_name)
        if not tools:
            return False

        current_hash = self._compute_tool_hash(tools)
        stored_hash = await self.get_stored_hash(backend_name)

        changed = current_hash != stored_hash
        if changed:
            logger.info(
                f"Backend {backend_name} has changed: {stored_hash} -> {current_hash}"
            )

        self._last_check[backend_name] = datetime.now()
        return changed

    async def sync_if_needed(self) -> Dict[str, str]:
        """
        Check all backends and sync if changes detected.
        Returns dict of {backend: status} where status is 'synced', 'unchanged', or 'error'.
        """
        async with self._sync_lock:
            results = {}
            changed_backends = []

            # Check each backend for changes
            for backend_name in self.config.backends.keys():
                try:
                    has_changes = await self.check_backend_changes(backend_name)
                    if has_changes:
                        changed_backends.append(backend_name)
                        results[backend_name] = "changed"
                    else:
                        results[backend_name] = "unchanged"
                except Exception as e:
                    logger.error(f"Error checking backend {backend_name}: {e}")
                    results[backend_name] = f"error: {str(e)[:50]}"

            # If any backends changed, rebuild affected portions
            if changed_backends:
                logger.info(
                    f"Rebuilding index for changed backends: {changed_backends}"
                )
                try:
                    await self._rebuild_for_backends(changed_backends)
                    for name in changed_backends:
                        results[name] = "synced"
                except Exception as e:
                    logger.error(f"Error rebuilding index: {e}")
                    for name in changed_backends:
                        results[name] = f"sync_error: {str(e)[:50]}"

            return results

    async def _rebuild_for_backends(self, backend_names: List[str]):
        """Rebuild index for specified backends."""
        from tool_manifest import ToolDefinition

        # Collect all tools from changed backends
        all_tools = []
        db = self._get_db()

        for backend_name in backend_names:
            tools = self.backends.get_backend_tools(backend_name)
            if not tools:
                continue

            # Convert to ToolDefinition format
            for tool in tools:
                # Parse server and name
                if ":" in tool.qualified_name:
                    server, name = tool.qualified_name.split(":", 1)
                else:
                    server = tool.server

                # Extract parameters from schema
                params = {}
                if tool.input_schema and "properties" in tool.input_schema:
                    for param_name, param_info in tool.input_schema[
                        "properties"
                    ].items():
                        param_type = param_info.get("type", "any")
                        if isinstance(param_type, list):
                            param_type = "/".join(param_type)
                        params[param_name] = param_type

                all_tools.append(
                    ToolDefinition(
                        name=tool.qualified_name,
                        description=tool.description,
                        category=self._categorize_tool(tool.name, tool.description),
                        server=server,
                        parameters=params,
                        examples=[],
                        is_core=False,
                    )
                )

            # Update sync state
            tool_hash = self._compute_tool_hash(tools)
            db.execute(
                """
                INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
                ON CONFLICT(backend_name) DO UPDATE SET
                    tool_count = excluded.tool_count,
                    tool_hash = excluded.tool_hash,
                    last_sync_at = CURRENT_TIMESTAMP,
                    sync_status = 'synced'
            """,
                (backend_name, len(tools), tool_hash),
            )

        db.commit()

        # Rebuild index with new tools
        if all_tools:
            # For now, do incremental add (HNSW supports this)
            # Full rebuild would be: await self.index.build_index(all_tools)
            for tool in all_tools:
                await self.index.add_single_tool(tool)

            logger.info(f"Added {len(all_tools)} tools from backends: {backend_names}")

    def _categorize_tool(self, name: str, description: str) -> str:
        """Infer category from tool name and description."""
        name_lower = name.lower()

        if any(x in name_lower for x in ["file", "read", "write", "directory", "path"]):
            return "file"
        if any(x in name_lower for x in ["git", "commit", "branch", "repo"]):
            return "git"
        if any(x in name_lower for x in ["db_", "sql", "database", "query"]):
            return "database"
        if any(x in name_lower for x in ["search", "find", "lookup"]):
            return "search"
        if any(x in name_lower for x in ["comfy", "image", "generate", "video"]):
            return "ai"
        if any(x in name_lower for x in ["scan", "analyze", "health", "report"]):
            return "analysis"
        if any(x in name_lower for x in ["project", "session", "content"]):
            return "project"
        if any(x in name_lower for x in ["status", "health", "service"]):
            return "system"

        return "other"

    async def full_sync(self) -> Dict[str, Any]:
        """
        Force a full sync from all backends.
        Rebuilds the entire index.
        """
        async with self._sync_lock:
            logger.info("Starting full sync from all backends...")

            # Connect to all backends
            connect_results = await self.backends.connect_all()

            # Collect all tools
            from tool_manifest import ToolDefinition

            all_tools = []
            db = self._get_db()

            for backend_name, connected in connect_results.items():
                if not connected:
                    logger.warning(f"Skipping {backend_name} - not connected")
                    continue

                tools = self.backends.get_backend_tools(backend_name)
                if not tools:
                    continue

                # Convert to ToolDefinition
                for tool in tools:
                    if ":" in tool.qualified_name:
                        server, name = tool.qualified_name.split(":", 1)
                    else:
                        server = tool.server

                    params = {}
                    if tool.input_schema and "properties" in tool.input_schema:
                        for param_name, param_info in tool.input_schema[
                            "properties"
                        ].items():
                            param_type = param_info.get("type", "any")
                            if isinstance(param_type, list):
                                param_type = "/".join(param_type)
                            params[param_name] = param_type

                    all_tools.append(
                        ToolDefinition(
                            name=tool.qualified_name,
                            description=tool.description,
                            category=self._categorize_tool(tool.name, tool.description),
                            server=server,
                            parameters=params,
                            examples=[],
                            is_core=False,
                        )
                    )

                # Update sync state
                tool_hash = self._compute_tool_hash(tools)
                db.execute(
                    """
                    INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'synced')
                    ON CONFLICT(backend_name) DO UPDATE SET
                        tool_count = excluded.tool_count,
                        tool_hash = excluded.tool_hash,
                        last_sync_at = CURRENT_TIMESTAMP,
                        sync_status = 'synced'
                """,
                    (backend_name, len(tools), tool_hash),
                )

            db.commit()

            # Rebuild entire index
            if all_tools:
                result = await self.index.build_index(all_tools)
                logger.info(f"Full sync complete: {len(all_tools)} tools indexed")
                return {
                    "status": "complete",
                    "tools_indexed": len(all_tools),
                    "backends_synced": list(connect_results.keys()),
                    "build_result": result,
                }
            else:
                return {"status": "no_tools", "tools_indexed": 0, "backends_synced": []}

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status for all backends."""
        db = self._get_db()

        cursor = db.execute("""
            SELECT backend_name, tool_count, tool_hash, last_sync_at, sync_status
            FROM backend_sync_state
        """)

        backends = {}
        for row in cursor.fetchall():
            backends[row["backend_name"]] = {
                "tool_count": row["tool_count"],
                "tool_hash": row["tool_hash"][:8] + "..." if row["tool_hash"] else None,
                "last_sync_at": row["last_sync_at"],
                "status": row["sync_status"],
            }

        # Add any backends not yet synced
        for name in self.config.backends.keys():
            if name not in backends:
                backends[name] = {
                    "tool_count": None,
                    "tool_hash": None,
                    "last_sync_at": None,
                    "status": "never_synced",
                }

        return {
            "backends": backends,
            "polling_active": self._polling_task is not None
            and not self._polling_task.done(),
        }

    async def start_background_polling(self, interval_seconds: int = 300):
        """Start background task that polls for changes."""
        if self._polling_task and not self._polling_task.done():
            logger.warning("Background polling already running")
            return

        async def poll_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    results = await self.sync_if_needed()
                    synced = [k for k, v in results.items() if v == "synced"]
                    if synced:
                        logger.info(f"Background sync completed for: {synced}")
                except Exception as e:
                    logger.error(f"Background sync error: {e}")

        self._polling_task = asyncio.create_task(poll_loop())
        logger.info(f"Started background polling every {interval_seconds}s")

    async def stop_background_polling(self):
        """Stop background polling task."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
            logger.info("Stopped background polling")

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None


# Singleton instance
_sync_manager_instance: Optional[SyncManager] = None


def get_sync_manager(
    config: "CompassConfig", index: "CompassIndex", backends: "BackendManager"
) -> SyncManager:
    """Get or create the sync manager singleton."""
    global _sync_manager_instance
    if _sync_manager_instance is None:
        _sync_manager_instance = SyncManager(config, index, backends)
    return _sync_manager_instance
