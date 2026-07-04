"""
Tool Compass - Sync Manager
Detects backend changes and triggers index rebuilds.
"""

import asyncio
import contextlib
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
    from backend_client_simple import SimpleBackendManager as BackendManager

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

        # MSYNC-A-002: the incremental diff path removes vanished tools via
        # index.remove_tool, which only deletes the SQLite row — the HNSW
        # vector is orphaned (hnswlib has no true delete). High-churn backends
        # accumulate orphan vectors that consume the fixed search candidate
        # window, thinning results until a full rebuild. We count incremental
        # cycles that actually removed tools and force a real build_index()
        # rebuild after this many, which atomically rebuilds a clean HNSW.
        self._incremental_cycles_since_rebuild = 0
        self._rebuild_after_incremental_cycles = 20

        # Ensure db directory exists
        ANALYTICS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get or create database connection.

        BE-A-011 + BE-B-010: check_same_thread=False is needed because the
        sync poll_loop runs on the asyncio event loop while the sync DB may
        also be touched from worker threads. WAL + busy_timeout serialize
        across this connection AND the analytics + chain_indexer connections
        without losing throughput.
        """
        if self._db is None:
            self._db = sqlite3.connect(
                str(ANALYTICS_DB_PATH), check_same_thread=False
            )
            self._db.row_factory = sqlite3.Row
            try:
                self._db.execute("PRAGMA busy_timeout = 5000")
                self._db.execute("PRAGMA journal_mode = WAL")
                self._db.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.Error as e:
                logger.debug(f"sqlite PRAGMA setup failed: {e}")
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
        # DEG-02: sync_status is the only durable per-backend health surface
        # (get_sync_status reads it), but it was only ever written 'synced'.
        # last_error / last_error_at let an 'error' row carry *why* it failed
        # so triage isn't blind. Added via ALTER for forward-compat on
        # pre-existing tables; ignore the "duplicate column" error on re-init.
        for column_ddl in (
            "ALTER TABLE backend_sync_state ADD COLUMN last_error TEXT",
            "ALTER TABLE backend_sync_state ADD COLUMN last_error_at TIMESTAMP",
        ):
            try:
                db.execute(column_ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
        db.commit()

    def _mark_backend_error(self, backend_name: str, message: str) -> None:
        """DEG-02: persist a per-backend failure so triage isn't blind.

        Writes sync_status='error' plus a truncated last_error / last_error_at
        without disturbing the cached tool_count/tool_hash (those still reflect
        the last *good* sync). Best-effort: a failure to record the failure
        must never mask the original error, so DB problems are only logged.
        """
        try:
            db = self._get_db()
            db.execute(
                """
                INSERT INTO backend_sync_state
                    (backend_name, sync_status, last_error, last_error_at)
                VALUES (?, 'error', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(backend_name) DO UPDATE SET
                    sync_status = 'error',
                    last_error = excluded.last_error,
                    last_error_at = CURRENT_TIMESTAMP
            """,
                (backend_name, message[:200]),
            )
            db.commit()
        except sqlite3.Error as e:
            logger.debug(f"_mark_backend_error({backend_name}) failed: {e}")

    def _mark_backends_synced(self, backend_names: List[str]) -> None:
        """DEG-02: promote backends to terminal 'synced' after a good rebuild.

        Clears any stale last_error so a backend that previously failed and
        now recovers doesn't keep showing an error. tool_count/tool_hash were
        already written in the 'rebuilding' interim, so we only flip the
        status here.
        """
        if not backend_names:
            return
        try:
            db = self._get_db()
            for name in backend_names:
                db.execute(
                    """
                    UPDATE backend_sync_state
                    SET sync_status = 'synced',
                        last_sync_at = CURRENT_TIMESTAMP,
                        last_error = NULL,
                        last_error_at = NULL
                    WHERE backend_name = ?
                """,
                    (name,),
                )
            db.commit()
        except sqlite3.Error as e:
            logger.debug(f"_mark_backends_synced failed: {e}")

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
        if not self.backends.is_backend_connected(backend_name):
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
                    # DEG-02: persist the failure so get_sync_status surfaces
                    # it — the docstring promised 'error' and it was never
                    # written anywhere durable.
                    self._mark_backend_error(backend_name, f"check failed: {e}")

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
                        # DEG-02: persist the rebuild failure too.
                        self._mark_backend_error(name, f"rebuild failed: {e}")

            return results

    def _index_read_lock(self):
        """Return the indexer's DB write lock as a context manager.

        MSYNC-A-003: cross-thread reads of self.index.db from the event-loop
        thread race with worker-thread index writes (search_sync dispatches
        embedding work to a ThreadPoolExecutor; add_single_tool / remove_tool
        mutate the same connection). The indexer serializes all of its own
        mutations via _db_write_lock; we acquire that SAME lock here so our
        reads can't observe a half-applied write (sqlite3.ProgrammingError on
        a recursive-use connection). If the lock isn't present (older index or
        a test double), fall back to a no-op so behaviour is unchanged.
        """
        lock = getattr(self.index, "_db_write_lock", None)
        if lock is not None:
            return lock
        return contextlib.nullcontext()

    def _get_backend_tool_names(self, backend_name: str) -> set:
        """Return set of currently-indexed tool names for a backend.

        Used by diffing sync (IDX-FT-004) to compute which tools disappeared
        since the last rebuild so we can remove them instead of only adding.
        """
        names: set = set()
        idx_db = getattr(self.index, "db", None)
        if idx_db is None:
            return names
        try:
            # MSYNC-A-003: hold the indexer's write lock for this cross-thread
            # read so a concurrent worker-thread index write can't raise.
            with self._index_read_lock():
                cursor = idx_db.execute(
                    "SELECT name FROM tools WHERE server = ?", (backend_name,)
                )
                rows = cursor.fetchall()
            for row in rows:
                names.add(row["name"])
        except Exception as e:
            logger.debug(f"_get_backend_tool_names({backend_name}) failed: {e}")
        return names

    def _tool_infos_to_definitions(self, tools: List[Any]) -> List[Any]:
        """Convert a backend's ToolInfo list into ToolDefinition objects.

        BE-COMPACT-001: factored out of _rebuild_for_backends / full_sync so
        the compaction path can rebuild the CURRENT FULL tool set (across every
        connected backend) with identical conversion semantics — the qualified
        name stays the globally-unique ToolDefinition.name, server is parsed
        from the "server:tool" prefix, and list-typed param schemas collapse to
        a "/"-joined string.
        """
        from tool_manifest import ToolDefinition

        definitions: List[Any] = []
        for tool in tools:
            # Parse server from the qualified name ("server:tool"). The short
            # name is intentionally dropped — ToolDefinition.name uses
            # tool.qualified_name (fully-qualified) to keep names globally
            # unique across backends.
            if ":" in tool.qualified_name:
                server, _short_name = tool.qualified_name.split(":", 1)
            else:
                # No colon — fall back to the backend-reported server.
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

            definitions.append(
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
        return definitions

    def _collect_all_connected_tools(self) -> List[Any]:
        """Collect ToolDefinitions from EVERY currently-connected backend.

        BE-COMPACT-001: build_index is a FULL table-truncating replace, so the
        orphan-vector compaction rebuild MUST see the whole live catalog, not
        just the changed subset. Iterate all configured backends and gather
        tools from those that are connected (get_backend_tools already returns
        [] for a disconnected backend, so an explicit is_backend_connected
        check keeps us defensive when the manager exposes it).
        """
        all_tools: List[Any] = []
        for backend_name in self.config.backends.keys():
            is_connected = getattr(self.backends, "is_backend_connected", None)
            if callable(is_connected) and not is_connected(backend_name):
                continue
            tools = self.backends.get_backend_tools(backend_name)
            if not tools:
                continue
            all_tools.extend(self._tool_infos_to_definitions(tools))
        return all_tools

    async def _rebuild_for_backends(self, backend_names: List[str]):
        """Rebuild index for specified backends."""
        # Collect all tools from changed backends (ToolDefinition conversion is
        # delegated to _tool_infos_to_definitions, which imports ToolDefinition).
        all_tools = []
        db = self._get_db()

        # Per-backend diff counters (IDX-FT-004).
        diff_stats: Dict[str, Dict[str, int]] = {}

        # DEG-02: backends whose tools we actually staged this pass. We defer
        # the durable sync_status='synced' write until AFTER the index rebuild
        # below succeeds — writing 'synced' before a rebuild that can still
        # fail (DEG-01) is exactly what blinded triage.
        staged_backends: List[str] = []

        for backend_name in backend_names:
            tools = self.backends.get_backend_tools(backend_name)
            if not tools:
                continue

            # Diff: old names in this backend vs. new names — anything
            # that vanished is removed from the index (IDX-FT-004).
            old_names = self._get_backend_tool_names(backend_name)
            new_names = {t.qualified_name for t in tools}
            removed = old_names - new_names
            removed_count = 0
            for name in removed:
                ok = await self.index.remove_tool(name)
                if ok:
                    removed_count += 1

            diff_stats[backend_name] = {
                "added": len(new_names - old_names),
                "updated": len(new_names & old_names),
                "removed": removed_count,
            }
            staged_backends.append(backend_name)

            # Convert to ToolDefinition format (BE-COMPACT-001: shared helper
            # so the compaction full-set path below uses identical semantics).
            all_tools.extend(self._tool_infos_to_definitions(tools))

            # DEG-02: write an interim 'rebuilding' state (not 'synced') with
            # the fresh count/hash. The terminal 'synced' is written only once
            # the index rebuild below actually succeeds.
            tool_hash = self._compute_tool_hash(tools)
            db.execute(
                """
                INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'rebuilding')
                ON CONFLICT(backend_name) DO UPDATE SET
                    tool_count = excluded.tool_count,
                    tool_hash = excluded.tool_hash,
                    last_sync_at = CURRENT_TIMESTAMP,
                    sync_status = 'rebuilding'
            """,
                (backend_name, len(tools), tool_hash),
            )

        db.commit()

        # Per-backend diff summary (IDX-FT-004).
        for bname, stats in diff_stats.items():
            logger.info(
                f"Sync diff for {bname}: {stats['added']} added, "
                f"{stats['updated']} updated, {stats['removed']} removed"
            )

        # MSYNC-A-002: did this incremental cycle remove any tools? Each
        # removal orphans an HNSW vector (remove_tool only deletes the SQLite
        # row), so removals — not adds — are what accumulate cruft. Count
        # cycles that actually removed something and force a periodic full
        # rebuild to compact the HNSW back to a clean state.
        removed_this_cycle = sum(s["removed"] for s in diff_stats.values())
        if removed_this_cycle:
            self._incremental_cycles_since_rebuild += 1

        if (
            all_tools
            and self._incremental_cycles_since_rebuild
            >= self._rebuild_after_incremental_cycles
        ):
            # Enough churn has accumulated — do a real rebuild so orphaned
            # vectors no longer eat the fixed search candidate window.
            #
            # BE-COMPACT-001: build_index is a FULL table-truncating replace
            # (DELETE FROM tools + rebuild HNSW from ONLY the passed tools).
            # `all_tools` here holds ONLY the changed subset (backend_names),
            # so passing it would silently delete every OTHER connected
            # backend's tools — compass() would then return "no matching tool"
            # for them until a manual full_sync. Re-collect the CURRENT FULL
            # catalog across every connected backend and rebuild from that.
            full_tools = self._collect_all_connected_tools()
            logger.info(
                f"Orphan-vector compaction: {self._incremental_cycles_since_rebuild} "
                f"incremental cycles with removals reached threshold "
                f"({self._rebuild_after_incremental_cycles}); doing a full "
                f"build_index rebuild of {len(full_tools)} tool(s) across all "
                f"connected backends (changed subset: {len(all_tools)})."
            )
            await self.index.build_index(full_tools)
            self._incremental_cycles_since_rebuild = 0
            logger.info(
                f"Full rebuild complete: {len(full_tools)} tools across all "
                f"connected backends (triggered by: {backend_names})"
            )
            # DEG-02: rebuild succeeded — promote staged backends to 'synced'.
            self._mark_backends_synced(staged_backends)
        elif all_tools:
            # Normal incremental path — HNSW supports dynamic add.
            # DEG-01: add_single_tool swallows exceptions and returns False on
            # Ollama-down / breaker-open. Count failures instead of discarding
            # the bool, so we don't claim a clean 'Added N' on a partial add.
            failed = 0
            for tool in all_tools:
                ok = await self.index.add_single_tool(tool)
                if not ok:
                    failed += 1

            if failed:
                # DEG-01: do NOT emit the unconditional success line — be
                # honest that some adds failed.
                logger.warning(
                    f"Sync partial: {failed}/{len(all_tools)} tool add(s) "
                    f"failed for backends {backend_names} (index/embedder "
                    f"unavailable?)"
                )
                # DEG-02: a partial add means the index does not reflect these
                # backends — mark them 'error', not 'synced'.
                for name in staged_backends:
                    self._mark_backend_error(
                        name, f"{failed}/{len(all_tools)} tool add(s) failed"
                    )
            else:
                logger.info(
                    f"Added {len(all_tools)} tools from backends: {backend_names}"
                )
                # DEG-02: all adds landed — now it's genuinely synced.
                self._mark_backends_synced(staged_backends)
        elif staged_backends:
            # Tools were staged (removals applied + state written) but nothing
            # remained to add — that's still a successful sync.
            self._mark_backends_synced(staged_backends)

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

            # Connect to all backends. connect_all() iterates every CONFIGURED
            # backend (config.backends.keys()), so connect_results is the full
            # configured set mapped to connect-success.
            connect_results = await self.backends.connect_all()

            # CONTRACT: split the configured set into who connected vs. who
            # failed THIS pass. The cli's _cmd_sync reads failed_backends to
            # warn on partial failure.
            connected_backends = [n for n, ok in connect_results.items() if ok]
            failed_backends = [n for n, ok in connect_results.items() if not ok]

            # Collect all tools
            from tool_manifest import ToolDefinition

            all_tools = []
            db = self._get_db()

            # Pre-compute the OLD name-set across ALL backends so we can log
            # the diff after build_index clears the old rows (IDX-FT-004).
            # build_index already truncates the tools table, so we capture
            # baseline before it runs. We also keep the per-row (name, server)
            # pairs so DEG-03 can tell which dropped tools belonged to a
            # backend that merely failed to connect (unreachable) vs. one that
            # genuinely went away.
            old_all_names: set = set()
            old_name_server: List[tuple] = []
            idx_db = getattr(self.index, "db", None)
            if idx_db is not None:
                try:
                    # MSYNC-A-003: hold the indexer's write lock for this
                    # cross-thread read so a concurrent worker-thread write
                    # can't raise sqlite3.ProgrammingError mid-fetch.
                    with self._index_read_lock():
                        cursor = idx_db.execute("SELECT name, server FROM tools")
                        rows = cursor.fetchall()
                    old_all_names = {row["name"] for row in rows}
                    old_name_server = [(row["name"], row["server"]) for row in rows]
                except Exception as e:
                    logger.debug(f"full_sync: baseline fetch failed: {e}")

            # DEG-02: a configured backend that failed to connect this pass is
            # a durable health signal — persist 'error' so get_sync_status
            # surfaces it instead of leaving a stale 'synced'.
            for backend_name in failed_backends:
                self._mark_backend_error(
                    backend_name, "connect failed during full_sync"
                )

            # Per-backend diff counters (IDX-FT-004).
            diff_stats: Dict[str, Dict[str, int]] = {}

            # DEG-02: backends successfully staged this pass; promoted to
            # 'synced' only after build_index succeeds below.
            staged_backends: List[str] = []

            for backend_name, connected in connect_results.items():
                if not connected:
                    logger.warning(f"Skipping {backend_name} - not connected")
                    continue

                tools = self.backends.get_backend_tools(backend_name)
                if not tools:
                    continue

                # Diff this backend's old vs. new names.
                old_names = self._get_backend_tool_names(backend_name)
                new_names = {t.qualified_name for t in tools}
                diff_stats[backend_name] = {
                    "added": len(new_names - old_names),
                    "updated": len(new_names & old_names),
                    "removed": len(old_names - new_names),
                }

                # Convert to ToolDefinition
                for tool in tools:
                    # See sibling comment in _rebuild_for_backends — short name
                    # is intentionally dropped; ToolDefinition.name uses the
                    # qualified form.
                    if ":" in tool.qualified_name:
                        server, _short_name = tool.qualified_name.split(":", 1)
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

                # DEG-02: interim 'rebuilding' state; promoted to 'synced'
                # only after build_index succeeds below.
                tool_hash = self._compute_tool_hash(tools)
                db.execute(
                    """
                    INSERT INTO backend_sync_state (backend_name, tool_count, tool_hash, last_sync_at, sync_status)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'rebuilding')
                    ON CONFLICT(backend_name) DO UPDATE SET
                        tool_count = excluded.tool_count,
                        tool_hash = excluded.tool_hash,
                        last_sync_at = CURRENT_TIMESTAMP,
                        sync_status = 'rebuilding'
                """,
                    (backend_name, len(tools), tool_hash),
                )
                staged_backends.append(backend_name)

            db.commit()

            # Emit diff summary per backend (IDX-FT-004).
            for bname, stats in diff_stats.items():
                logger.info(
                    f"Sync diff for {bname}: {stats['added']} added, "
                    f"{stats['updated']} updated, {stats['removed']} removed"
                )
            # Also note globally removed tools (tools present last pass but not
            # this one). DEG-03: a full_sync rebuilds from ONLY the backends
            # that connected, so a transiently-unreachable backend's tools get
            # dropped too. Distinguish "removed because the backend is gone"
            # from "removed because the backend was unreachable this pass" so
            # the log doesn't claim an unreachable backend's tools were
            # intentionally retired.
            new_all_names = {t.name for t in all_tools}
            globally_removed = old_all_names - new_all_names
            failed_set = set(failed_backends)
            dropped_unreachable = {
                name
                for (name, server) in old_name_server
                if name in globally_removed and server in failed_set
            }
            dropped_gone = globally_removed - dropped_unreachable
            if dropped_gone:
                logger.info(
                    f"Full sync will drop {len(dropped_gone)} tool(s) "
                    f"from removed backends"
                )
            if dropped_unreachable:
                # Conservative: still rebuild (build_index is the contract),
                # but WARN loudly that these drops are due to unreachable
                # backends, not intentional removal — so a transient outage
                # reads as an outage in the logs, not a retirement.
                logger.warning(
                    f"Full sync is dropping {len(dropped_unreachable)} tool(s) "
                    f"from {len(failed_set & {s for _, s in old_name_server})} "
                    f"backend(s) that failed to connect this pass "
                    f"({sorted(failed_set)}); these were UNREACHABLE, not "
                    f"removed — they will reappear once the backend(s) reconnect."
                )

            # Rebuild entire index
            if all_tools:
                result = await self.index.build_index(all_tools)
                logger.info(f"Full sync complete: {len(all_tools)} tools indexed")
                # DEG-02: rebuild landed — promote staged backends to 'synced'.
                self._mark_backends_synced(staged_backends)
                return {
                    "status": "complete",
                    "tools_indexed": len(all_tools),
                    "backends_synced": list(connect_results.keys()),
                    "connected_backends": connected_backends,
                    "failed_backends": failed_backends,
                    "build_result": result,
                }
            else:
                # MSYNC-A-001: no tools across all backends. We MUST still
                # rebuild with an empty set so build_index([]) atomically
                # clears the tools table + resets the HNSW index. Returning
                # early without this left the prior index live, so compass
                # kept routing to dead tools — and the globally_removed log
                # above claimed they were dropped when nothing was.
                build_result = await self.index.build_index([])
                logger.info("Full sync complete: index emptied (0 tools)")
                self._mark_backends_synced(staged_backends)
                return {
                    "status": "no_tools",
                    "tools_indexed": 0,
                    "backends_synced": list(connect_results.keys()),
                    "connected_backends": connected_backends,
                    "failed_backends": failed_backends,
                    "build_result": build_result,
                }

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status for all backends."""
        db = self._get_db()

        cursor = db.execute("""
            SELECT backend_name, tool_count, tool_hash, last_sync_at,
                   sync_status, last_error, last_error_at
            FROM backend_sync_state
        """)

        backends = {}
        for row in cursor.fetchall():
            backends[row["backend_name"]] = {
                "tool_count": row["tool_count"],
                "tool_hash": row["tool_hash"][:8] + "..." if row["tool_hash"] else None,
                "last_sync_at": row["last_sync_at"],
                "status": row["sync_status"],
                # DEG-02: surface the persisted failure reason so triage can
                # see *why* a backend is in 'error' without reading logs.
                "last_error": row["last_error"],
                "last_error_at": row["last_error_at"],
            }

        # Add any backends not yet synced
        for name in self.config.backends.keys():
            if name not in backends:
                backends[name] = {
                    "tool_count": None,
                    "tool_hash": None,
                    "last_sync_at": None,
                    "status": "never_synced",
                    "last_error": None,
                    "last_error_at": None,
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
            # BE-A-017 + BE-B-012: removed asyncio.shield wrapper around
            # sync_if_needed. Shield was intended to prevent corruption
            # during cancellation, but it fights graceful shutdown — a
            # SIGTERM during a 60s rebuild would wait the full 60s before
            # unwinding, often escalating to SIGKILL and losing the
            # in-flight DB commit. Sync operations are designed to be
            # cancellable: each backend's update commits atomically via
            # BEGIN IMMEDIATE / COMMIT (see indexer.build_index), so a
            # cancellation between backends leaves a coherent state.
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    results = await self.sync_if_needed()
                    synced = [k for k, v in results.items() if v == "synced"]
                    if synced:
                        logger.info(f"Background sync completed for: {synced}")
                except asyncio.CancelledError:
                    logger.info("Background polling cancelled")
                    raise
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
