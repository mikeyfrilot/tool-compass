"""
Tool Compass - Analytics Module
Tracks usage patterns, manages hot tool cache, and detects tool chains.

Schema migration pattern (MCC-B-003):
    The analytics DB tracks its version in the `schema_meta` table under the
    key `schema_version`. Current version: 1. When a future change alters
    table shape, bump the version constant below and add a handler to
    `_run_migrations` like::

        if current_version < 2:
            db.executescript("ALTER TABLE ... ADD COLUMN ...")
            db.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")

    Migrations must be idempotent — `_run_migrations` runs on every open.
"""

import sqlite3
import json
import hashlib
import threading
from collections import deque
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Database path
ANALYTICS_DB_PATH = Path(__file__).parent / "db" / "compass_analytics.db"

# Current analytics schema version. Bump + add a block to _run_migrations
# whenever the table shape changes. See module docstring for the pattern.
CURRENT_SCHEMA_VERSION = 1


@dataclass
class HotToolEntry:
    """Cached data for frequently used tools."""

    tool_name: str
    rank: int
    call_count: int
    embedding: Optional[np.ndarray]  # Pre-loaded 768-dim vector
    schema: Optional[Dict[str, Any]]  # Full input schema
    description: str
    last_called_at: Optional[datetime]


@dataclass
class SearchRecord:
    """Record of a search query."""

    query: str
    top_result: Optional[str]
    result_count: int
    latency_ms: float
    category_filter: Optional[str]
    server_filter: Optional[str]


@dataclass
class ToolCallRecord:
    """Record of a tool execution."""

    tool_name: str
    server: str
    success: bool
    error_message: Optional[str]
    latency_ms: float


class CompassAnalytics:
    """
    Analytics engine for Tool Compass.

    Responsibilities:
    - Track search queries and tool calls
    - Maintain hot tool cache (top 10 most used)
    - Detect and track tool chains/patterns
    - Provide usage statistics
    """

    def __init__(
        self,
        db_path: Path = ANALYTICS_DB_PATH,
        hot_cache_size: int = 10,
        chain_min_occurrences: int = 3,
    ):
        self.db_path = db_path
        self.hot_cache_size = hot_cache_size
        self.chain_min_occurrences = chain_min_occurrences

        self.db: Optional[sqlite3.Connection] = None
        self._hot_cache: Dict[str, HotToolEntry] = {}

        # MCC-B-005: analytics is observability, not core function. If SQLite
        # writes start failing, degrade gracefully — log ONCE, set this flag,
        # and silently no-op future writes rather than bubbling exceptions
        # up through every tool call.
        self._degraded: bool = False
        self._degraded_logged: bool = False

        # Lock for all DB mutations (sqlite3 connection is shared across threads
        # via check_same_thread=False; we serialize writes ourselves).
        self._lock = threading.Lock()

        # MCC-FT-003: dedicated lock for lazy connection initialization. The
        # write lock (self._lock) does NOT cover _get_db()/_init_db, so two
        # threads first-touching the DB concurrently used to race here — each
        # saw self.db is None, each opened a connection and ran the DDL,
        # interleaving CREATE + commit on a shared handle and corrupting
        # transaction state ("cannot commit - no transaction is active"). A
        # separate init lock keeps that serialization orthogonal to the write
        # path (writers never block on init once the connection is published).
        self._init_lock = threading.Lock()

        # ANL-A-003: once close() runs, refuse to lazily reopen. close()
        # takes self._init_lock (the same lock _get_db uses for lazy init)
        # so close ordering serializes against reopen — a concurrent record_*
        # can't null the handle underneath an in-flight op or resurrect the
        # connection after close. Guarded by self._init_lock.
        self._closed: bool = False

        # Session tracking for chain detection. Use a bounded deque so a long
        # session never drops middle items on truncation — patterns are saved
        # before the window slides.
        self._session_id = hashlib.sha256(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        self._session_tool_sequence: Deque[str] = deque(maxlen=20)
        self._call_count_since_refresh = 0

        # Ensure db directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get or create database connection.

        The connection is opened with check_same_thread=False so it can be
        shared across Gradio's thread pool. All mutations are protected by
        self._lock — see record_search / record_tool_call / _save_chain_pattern.

        BE-A-011 + BE-B-010: WAL journal mode lets one writer + many readers
        coexist cleanly. busy_timeout=5000ms makes SQLite back off on
        SQLITE_BUSY rather than fail — the three-module write topology
        (analytics + sync_manager + chain_indexer) needs both to not race
        for the file lock.
        """
        # MCC-FT-003: double-checked locking. The fast path (already
        # initialized) takes no lock. On first touch, serialize under
        # self._init_lock and publish self.db ONLY after the connection is
        # fully built + schema-initialized, so a concurrent caller never
        # observes a half-constructed connection. _init_db operates on the
        # local `conn` (not self.db) precisely so nothing is visible until
        # it is complete.
        if self.db is None:
            with self._init_lock:
                # ANL-A-003: refuse to reopen after close(). Checked INSIDE
                # the init lock so a record racing a close can't slip a fresh
                # connection in after close() set the flag and nulled self.db.
                if self._closed:
                    raise sqlite3.ProgrammingError(
                        "analytics DB is closed; cannot reopen"
                    )
                if self.db is None:
                    conn = sqlite3.connect(
                        str(self.db_path), check_same_thread=False
                    )
                    conn.row_factory = sqlite3.Row
                    self._apply_sqlite_pragmas(conn)
                    self._init_db(conn)
                    self.db = conn
        return self.db

    @staticmethod
    def _apply_sqlite_pragmas(db: sqlite3.Connection) -> None:
        """Apply BE-B-010 pragmas: WAL + busy_timeout.

        Best-effort: WAL setup can fail on shared filesystems (network mounts),
        in which case journal_mode falls back to the default rollback journal
        and busy_timeout still helps. We swallow rather than crash.
        """
        try:
            db.execute("PRAGMA busy_timeout = 5000")
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.Error as e:
            logger.debug(f"sqlite PRAGMA setup failed: {e}")

    def _init_db(self, db: sqlite3.Connection):
        """Initialize analytics database with all tables.

        MCC-FT-003: takes the connection explicitly rather than calling
        _get_db(). During lazy init the connection is not yet published to
        self.db (so a concurrent reader can't see a half-built handle); this
        method must operate on the passed-in `db` for that to hold.
        """
        db.executescript("""
            -- Search query tracking
            CREATE TABLE IF NOT EXISTS search_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_hash TEXT,
                top_result TEXT,
                result_count INTEGER,
                latency_ms REAL,
                category_filter TEXT,
                server_filter TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_search_created ON search_queries(created_at);
            CREATE INDEX IF NOT EXISTS idx_search_top_result ON search_queries(top_result);

            -- Tool execution tracking
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                server TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                latency_ms REAL,
                arguments_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_calls_tool ON tool_calls(tool_name);
            CREATE INDEX IF NOT EXISTS idx_calls_created ON tool_calls(created_at);
            CREATE INDEX IF NOT EXISTS idx_calls_success ON tool_calls(success);

            -- Aggregated usage stats
            CREATE TABLE IF NOT EXISTS tool_usage_stats (
                tool_name TEXT PRIMARY KEY,
                call_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_latency_ms REAL,
                last_called_at TIMESTAMP,
                last_success_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Hot tool cache persistence
            CREATE TABLE IF NOT EXISTS hot_tools (
                tool_name TEXT PRIMARY KEY,
                rank INTEGER NOT NULL,
                call_count INTEGER NOT NULL,
                embedding BLOB,
                schema_json TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Tool chains (workflows)
            CREATE TABLE IF NOT EXISTS tool_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chain_name TEXT UNIQUE,
                chain_tools TEXT NOT NULL,
                description TEXT,
                embedding_text TEXT,
                use_count INTEGER DEFAULT 0,
                last_used_at TIMESTAMP,
                is_auto_detected INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chains_use_count ON tool_chains(use_count DESC);

            -- Chain execution patterns (for auto-detection)
            CREATE TABLE IF NOT EXISTS chain_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                tool_sequence TEXT NOT NULL,
                sequence_hash TEXT UNIQUE,
                occurrence_count INTEGER DEFAULT 1,
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_patterns_hash ON chain_patterns(sequence_hash);
            CREATE INDEX IF NOT EXISTS idx_patterns_count ON chain_patterns(occurrence_count DESC);

            -- Backend sync state
            CREATE TABLE IF NOT EXISTS backend_sync_state (
                backend_name TEXT PRIMARY KEY,
                tool_count INTEGER,
                tool_hash TEXT,
                last_sync_at TIMESTAMP,
                sync_status TEXT DEFAULT 'unknown'
            );

            -- Schema version tracking (MCC-B-003). Future table shape changes
            -- bump schema_version and add a migration in _run_migrations so
            -- analytics state survives upgrades instead of silently breaking.
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        db.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(CURRENT_SCHEMA_VERSION),),
        )
        db.commit()
        self._run_migrations(db)
        logger.info(f"Analytics database initialized at {self.db_path}")

    def get_schema_version(
        self, db: Optional[sqlite3.Connection] = None
    ) -> int:
        """Return the current analytics DB schema version (MCC-B-003).

        Accepts an optional connection so it can be called during lazy init
        (before self.db is published — MCC-FT-003); falls back to _get_db()
        for ordinary public use.
        """
        db = db if db is not None else self._get_db()
        row = db.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row["value"])
        except (KeyError, TypeError, ValueError):
            return 0

    def _run_migrations(self, db: Optional[sqlite3.Connection] = None) -> None:
        """Apply any pending schema migrations.

        Schema version 1 is the initial shape; there is nothing to migrate
        yet. The hook exists so future changes (new columns, new tables,
        value backfills) can land without scattering conditional logic
        across the codebase. See module docstring for the pattern.

        MCC-FT-003: accepts an optional connection so it runs against the
        in-flight connection during lazy init, before self.db is published.
        """
        current = self.get_schema_version(db)
        if current == CURRENT_SCHEMA_VERSION:
            logger.debug(
                f"Analytics schema version {current}; no migrations needed"
            )
            return
        # No migrations registered yet. Future versions add blocks here
        # in strict ascending order and bump schema_meta at the end.
        logger.warning(
            f"Analytics DB at schema v{current}, code expects v{CURRENT_SCHEMA_VERSION}; "
            "no migrations registered yet."
        )

    def _note_degraded(self, method: str, error: Exception) -> None:
        """Mark analytics as degraded and log once per session (MCC-B-005)."""
        self._degraded = True
        if not self._degraded_logged:
            logger.warning(
                f"Analytics write failed ({method}): {error}. "
                "Analytics degraded — subsequent writes disabled this session."
            )
            self._degraded_logged = True

    def get_health(self) -> Dict[str, Any]:
        """Report whether analytics is recording or running in degraded mode.

        MCC-B-005: lets the UI / status tab distinguish "everything fine"
        from "analytics sqlite broke; tool calls still work but metrics are
        not being recorded."
        """
        return {
            "degraded": self._degraded,
            "reason": (
                "sqlite write failure — see earlier log line for details"
                if self._degraded
                else None
            ),
        }

    async def record_search(
        self,
        query: str,
        results: List[Any],  # List of SearchResult
        latency_ms: float,
        category_filter: Optional[str] = None,
        server_filter: Optional[str] = None,
    ):
        """Record a search query for analytics."""
        if self._degraded:
            return
        try:
            db = self._get_db()

            query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:32]
            top_result = results[0].tool.name if results else None
            result_count = len(results)

            with self._lock:
                db.execute(
                    """
                    INSERT INTO search_queries
                    (query, query_hash, top_result, result_count, latency_ms, category_filter, server_filter)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        query,
                        query_hash,
                        top_result,
                        result_count,
                        latency_ms,
                        category_filter,
                        server_filter,
                    ),
                )
                db.commit()

            logger.debug(
                f"Recorded search: '{query[:50]}...' -> {top_result} ({latency_ms:.1f}ms)"
            )
        except sqlite3.Error as e:
            # MCC-B-005: swallow — analytics failures must not break searches.
            self._note_degraded("record_search", e)

    async def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None,
        arguments: Optional[Dict] = None,
    ):
        """
        Record a tool execution.
        Updates usage stats, hot cache tracking, and chain detection.
        """
        if self._degraded:
            return

        # MCC-FT-002: resolve deprecated aliases BEFORE the DB insert so all
        # analytics (call counts, success rates, hot cache, chain detection)
        # stay consistent across renames. Wrapped defensively — analytics
        # must still work in test isolation where tool_manifest may be absent
        # or monkeypatched away.
        try:
            from tool_manifest import get_canonical_name

            tool_name = get_canonical_name(tool_name)
        except ImportError:
            pass

        try:
            db = self._get_db()

            # Parse server from tool name
            server = tool_name.split(":")[0] if ":" in tool_name else "unknown"

            # Hash arguments for pattern detection. Hash the full key/value payload
            # (sorted for stability) so different argument values produce different
            # hashes — earlier revisions hashed only keys and collided on every call.
            args_hash = None
            if arguments:
                args_hash = hashlib.sha256(
                    json.dumps(arguments, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]

            with self._lock:
                # Insert call record
                db.execute(
                    """
                    INSERT INTO tool_calls
                    (tool_name, server, success, error_message, latency_ms, arguments_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        tool_name,
                        server,
                        1 if success else 0,
                        error_message,
                        latency_ms,
                        args_hash,
                    ),
                )

                # Update aggregated stats.
                # NOTE: SET-clause ordering is load-bearing — SQLite evaluates SET
                # clauses left-to-right (see https://sqlite.org/lang_update.html),
                # so the avg_latency_ms computation MUST come BEFORE call_count is
                # incremented. Reversing these lines introduces an off-by-one that
                # drifts the rolling mean.
                db.execute(
                    """
                    INSERT INTO tool_usage_stats (tool_name, call_count, success_count, failure_count, avg_latency_ms, last_called_at, last_success_at)
                    VALUES (?, 1, ?, ?, ?, CURRENT_TIMESTAMP, CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END)
                    ON CONFLICT(tool_name) DO UPDATE SET
                        avg_latency_ms = (avg_latency_ms * call_count + excluded.avg_latency_ms) / (call_count + 1),
                        call_count = call_count + 1,
                        success_count = success_count + excluded.success_count,
                        failure_count = failure_count + excluded.failure_count,
                        last_called_at = CURRENT_TIMESTAMP,
                        last_success_at = CASE WHEN excluded.success_count > 0 THEN CURRENT_TIMESTAMP ELSE last_success_at END,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (
                        tool_name,
                        1 if success else 0,
                        0 if success else 1,
                        latency_ms,
                        1 if success else 0,
                    ),
                )

                db.commit()
        except sqlite3.Error as e:
            # MCC-B-005: analytics failure must not break tool execution.
            self._note_degraded("record_tool_call", e)
            return

        # Track for chain detection. The deque is bounded (maxlen=20) so appends
        # silently drop the OLDEST item — never middle items. Save patterns on
        # every append so nothing is lost when the window slides.
        self._session_tool_sequence.append(tool_name)
        await self._save_chain_pattern()

        # Check if we should refresh hot cache
        self._call_count_since_refresh += 1
        if self._call_count_since_refresh >= 100:
            await self.refresh_hot_cache()
            self._call_count_since_refresh = 0

        logger.debug(
            f"Recorded tool call: {tool_name} ({'OK' if success else 'FAIL'}) {latency_ms:.1f}ms"
        )

    async def _save_chain_pattern(self):
        """Save current tool sequence as a pattern for chain detection."""
        if len(self._session_tool_sequence) < 2:
            return
        if self._degraded:
            return

        try:
            db = self._get_db()

            # Snapshot the deque as a list for subsequence slicing.
            sequence = list(self._session_tool_sequence)

            with self._lock:
                # ANL-A-004: count ONLY the new subsequences ENDING at the
                # just-appended tool (the last element). Re-counting every
                # length-2..5 subsequence of the whole sliding deque on every
                # call re-incremented any n-gram that merely stayed in the
                # window, systematically inflating occurrence_count and
                # tripping detect_chains thresholds on phantom repeats. The
                # newest distinct subsequences introduced by this append are
                # exactly the suffixes of `sequence` of length 2..5.
                n = len(sequence)
                for length in range(2, min(6, n + 1)):
                    subseq = sequence[n - length : n]
                    seq_json = json.dumps(subseq)
                    seq_hash = hashlib.sha256(seq_json.encode()).hexdigest()[:32]

                    # Upsert pattern
                    db.execute(
                        """
                        INSERT INTO chain_patterns (session_id, tool_sequence, sequence_hash, occurrence_count)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(sequence_hash) DO UPDATE SET
                            occurrence_count = occurrence_count + 1,
                            last_seen_at = CURRENT_TIMESTAMP
                        """,
                        (self._session_id, seq_json, seq_hash),
                    )

                db.commit()
        except sqlite3.Error as e:
            # MCC-B-005: chain pattern save failure degrades analytics only.
            self._note_degraded("_save_chain_pattern", e)

    async def prune_old_records(self, retention_days: int) -> int:
        """Delete analytics rows older than the retention window (FEAT-04).

        DELETEs from BOTH append-only tables — tool_calls and search_queries —
        whose created_at is older than `retention_days` days, and returns the
        total number of rows removed. A sibling gateway agent calls this as
        ``await analytics.prune_old_records(config.analytics_retention_days)``.

        retention_days <= 0 -> NO-OP (keep forever), returns 0.

        CA-001 discipline: created_at is stamped by SQLite CURRENT_TIMESTAMP,
        which is UTC. We compute the cutoff INSIDE SQLite via
        ``datetime('now', '-N days')`` so BOTH sides of the comparison are UTC
        — never datetime.now() naive local wall-clock, which would skew the
        cutoff by the host's UTC offset (the exact bug fixed in
        get_analytics_summary).

        MCC-B-005 discipline: wrapped in the same _degraded / try-except the
        other write methods use — a prune failure must degrade analytics, not
        raise up through the caller. Respects the thread-safe lazy connection
        and the write lock (self._lock) the rest of the module uses.
        """
        if self._degraded:
            return 0
        # retention_days <= 0 means "keep forever" — nothing to prune.
        if retention_days <= 0:
            return 0

        total_deleted = 0
        try:
            db = self._get_db()

            # Push the cutoff into SQL so it is evaluated in UTC by SQLite,
            # matching how created_at was written. datetime('now', ?) with a
            # '-N days' modifier yields a UTC 'YYYY-MM-DD HH:MM:SS' bound.
            modifier = f"-{int(retention_days)} days"

            with self._lock:
                cur = db.execute(
                    "DELETE FROM tool_calls "
                    "WHERE created_at < datetime('now', ?)",
                    (modifier,),
                )
                total_deleted += cur.rowcount or 0

                cur = db.execute(
                    "DELETE FROM search_queries "
                    "WHERE created_at < datetime('now', ?)",
                    (modifier,),
                )
                total_deleted += cur.rowcount or 0

                db.commit()
        except sqlite3.Error as e:
            # MCC-B-005: a prune failure degrades analytics; it must never
            # break the caller (the gateway prunes as housekeeping).
            self._note_degraded("prune_old_records", e)
            return 0

        # VACUUM reclaims the pages freed by a large delete. It CANNOT run
        # inside a transaction and takes its own file lock, so run it
        # best-effort AFTER the committed delete and swallow any failure —
        # reclaiming space is a nice-to-have, never worth raising for. Guard on
        # a non-trivial delete so we don't VACUUM the whole DB on every no-op
        # prune. Held under the write lock so it doesn't race another writer.
        if total_deleted > 0:
            try:
                with self._lock:
                    db.execute("VACUUM")
            except sqlite3.Error as e:
                logger.debug(f"VACUUM after prune skipped (non-fatal): {e}")

        if total_deleted:
            logger.info(
                f"Pruned {total_deleted} analytics row(s) older than "
                f"{retention_days} day(s)."
            )
        return total_deleted

    async def refresh_hot_cache(self, embedder=None, index=None):
        """
        Update the hot cache with top N most used tools.
        Optionally load embeddings and schemas if embedder/index provided.
        """
        if self._degraded:
            return []
        try:
            return await self._refresh_hot_cache_impl(embedder, index)
        except sqlite3.Error as e:
            self._note_degraded("refresh_hot_cache", e)
            return []

    async def _refresh_hot_cache_impl(self, embedder=None, index=None):
        """Inner implementation; raises sqlite3.Error — caller wraps."""
        db = self._get_db()

        # Get top tools by call count
        cursor = db.execute(
            """
            SELECT tool_name, call_count, last_called_at
            FROM tool_usage_stats
            ORDER BY call_count DESC
            LIMIT ?
        """,
            (self.hot_cache_size,),
        )

        top_tools = cursor.fetchall()

        new_cache = {}
        with self._lock:
            for rank, row in enumerate(top_tools, 1):
                tool_name = row["tool_name"]

                # Try to get existing embedding from hot_tools table
                existing = db.execute(
                    "SELECT embedding, schema_json, description FROM hot_tools WHERE tool_name = ?",
                    (tool_name,),
                ).fetchone()

                embedding = None
                schema = None
                description = ""

                if existing:
                    if existing["embedding"]:
                        embedding = np.frombuffer(existing["embedding"], dtype=np.float32)
                    if existing["schema_json"]:
                        schema = json.loads(existing["schema_json"])
                    description = existing["description"] or ""

                entry = HotToolEntry(
                    tool_name=tool_name,
                    rank=rank,
                    call_count=row["call_count"],
                    embedding=embedding,
                    schema=schema,
                    description=description,
                    last_called_at=row["last_called_at"],
                )
                new_cache[tool_name] = entry

                # Persist to DB
                embedding_blob = embedding.tobytes() if embedding is not None else None
                schema_json = json.dumps(schema) if schema else None

                db.execute(
                    """
                    INSERT INTO hot_tools (tool_name, rank, call_count, embedding, schema_json, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tool_name) DO UPDATE SET
                        rank = excluded.rank,
                        call_count = excluded.call_count,
                        embedding = COALESCE(excluded.embedding, embedding),
                        schema_json = COALESCE(excluded.schema_json, schema_json),
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (
                        tool_name,
                        rank,
                        row["call_count"],
                        embedding_blob,
                        schema_json,
                        description,
                    ),
                )

            db.commit()
        self._hot_cache = new_cache

        logger.info(f"Refreshed hot cache with {len(new_cache)} tools")
        return list(new_cache.keys())

    def get_hot_tool(self, tool_name: str) -> Optional[HotToolEntry]:
        """Get cached tool data if available in hot cache."""
        return self._hot_cache.get(tool_name)

    def is_hot(self, tool_name: str) -> bool:
        """Check if a tool is in the hot cache."""
        return tool_name in self._hot_cache

    async def detect_chains(self) -> List[Dict[str, Any]]:
        """
        Analyze chain_patterns to find common tool sequences.
        Promotes patterns with enough occurrences to tool_chains.
        """
        if self._degraded:
            return []
        try:
            return await self._detect_chains_impl()
        except sqlite3.Error as e:
            self._note_degraded("detect_chains", e)
            return []

    async def _detect_chains_impl(self) -> List[Dict[str, Any]]:
        """Inner implementation; raises sqlite3.Error — caller wraps."""
        db = self._get_db()

        # Find patterns with enough occurrences
        cursor = db.execute(
            """
            SELECT tool_sequence, sequence_hash, occurrence_count
            FROM chain_patterns
            WHERE occurrence_count >= ?
            ORDER BY occurrence_count DESC
            LIMIT 20
        """,
            (self.chain_min_occurrences,),
        )

        patterns = cursor.fetchall()
        detected_chains = []

        with self._lock:
            for row in patterns:
                tools = json.loads(row["tool_sequence"])

                # Generate chain name from tools
                chain_name = "_to_".join([t.split(":")[-1] for t in tools])[:50]

                # Generate description
                tool_names = [t.split(":")[-1].replace("_", " ") for t in tools]
                description = f"Workflow: {' → '.join(tool_names)}"

                # Check if already exists
                existing = db.execute(
                    "SELECT id FROM tool_chains WHERE chain_name = ?", (chain_name,)
                ).fetchone()

                if not existing:
                    # Create embedding text for semantic search
                    embedding_text = f"Workflow: {chain_name} | Steps: {', '.join(tool_names)} | Tools: {', '.join(tools)}"

                    db.execute(
                        """
                        INSERT INTO tool_chains (chain_name, chain_tools, description, embedding_text, use_count, is_auto_detected)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """,
                        (
                            chain_name,
                            json.dumps(tools),
                            description,
                            embedding_text,
                            row["occurrence_count"],
                        ),
                    )

                    detected_chains.append(
                        {
                            "name": chain_name,
                            "tools": tools,
                            "description": description,
                            "occurrences": row["occurrence_count"],
                        }
                    )

            db.commit()

        if detected_chains:
            logger.info(f"Detected {len(detected_chains)} new tool chains")

        return detected_chains

    async def get_chains(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all stored tool chains."""
        db = self._get_db()

        cursor = db.execute(
            """
            SELECT chain_name, chain_tools, description, use_count, is_auto_detected, last_used_at
            FROM tool_chains
            ORDER BY use_count DESC
            LIMIT ?
        """,
            (limit,),
        )

        chains = []
        for row in cursor.fetchall():
            chains.append(
                {
                    "name": row["chain_name"],
                    "tools": json.loads(row["chain_tools"]),
                    "description": row["description"],
                    "use_count": row["use_count"],
                    "is_auto_detected": bool(row["is_auto_detected"]),
                    "last_used_at": row["last_used_at"],
                }
            )

        return chains

    async def get_analytics_summary(self, timeframe: str = "24h") -> Dict[str, Any]:
        """
        Get comprehensive analytics summary.

        Args:
            timeframe: "1h", "24h", "7d", or "30d"
        """
        db = self._get_db()

        # Parse timeframe
        hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(timeframe, 24)
        # CA-001: created_at is written by SQLite CURRENT_TIMESTAMP, which is UTC.
        # Compute the window bound in UTC too — datetime.now() is naive LOCAL
        # wall-clock, so a naive bound skews every window by the host's UTC
        # offset. strftime drops the tz suffix, yielding a UTC
        # 'YYYY-MM-DD HH:MM:SS' string that matches CURRENT_TIMESTAMP.
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        # Use SQLite-compatible format (YYYY-MM-DD HH:MM:SS) for timestamp comparison
        since_str = since.strftime("%Y-%m-%d %H:%M:%S")

        # Search stats
        search_stats = db.execute(
            """
            SELECT
                COUNT(*) as total_searches,
                AVG(latency_ms) as avg_latency,
                AVG(result_count) as avg_results
            FROM search_queries
            WHERE created_at >= ?
        """,
            (since_str,),
        ).fetchone()

        # Top searched queries
        top_queries = db.execute(
            """
            SELECT query, COUNT(*) as count
            FROM search_queries
            WHERE created_at >= ?
            GROUP BY query_hash
            ORDER BY count DESC
            LIMIT 10
        """,
            (since_str,),
        ).fetchall()

        # Tool call stats
        call_stats = db.execute(
            """
            SELECT
                COUNT(*) as total_calls,
                SUM(success) as successes,
                AVG(latency_ms) as avg_latency
            FROM tool_calls
            WHERE created_at >= ?
        """,
            (since_str,),
        ).fetchone()

        # Top tools by usage
        top_tools = db.execute("""
            SELECT tool_name, call_count, success_count, failure_count, avg_latency_ms
            FROM tool_usage_stats
            ORDER BY call_count DESC
            LIMIT 10
        """).fetchall()

        # Failed tool calls
        failures = db.execute(
            """
            SELECT tool_name, error_message, COUNT(*) as count
            FROM tool_calls
            WHERE success = 0 AND created_at >= ?
            GROUP BY tool_name, error_message
            ORDER BY count DESC
            LIMIT 10
        """,
            (since_str,),
        ).fetchall()

        # Chain stats
        chain_count = db.execute("SELECT COUNT(*) FROM tool_chains").fetchone()[0]

        return {
            "timeframe": timeframe,
            "searches": {
                "total": search_stats["total_searches"] or 0,
                "avg_latency_ms": round(search_stats["avg_latency"] or 0, 1),
                "avg_results": round(search_stats["avg_results"] or 0, 1),
                "top_queries": [
                    {"query": r["query"][:50], "count": r["count"]} for r in top_queries
                ],
            },
            "tool_calls": {
                "total": call_stats["total_calls"] or 0,
                "success_rate": round(
                    (call_stats["successes"] or 0)
                    / max(call_stats["total_calls"] or 1, 1)
                    * 100,
                    1,
                ),
                "avg_latency_ms": round(call_stats["avg_latency"] or 0, 1),
                "top_tools": [
                    {
                        "tool": r["tool_name"],
                        "calls": r["call_count"],
                        "success_rate": round(
                            r["success_count"] / max(r["call_count"], 1) * 100, 1
                        ),
                        "avg_latency_ms": round(r["avg_latency_ms"] or 0, 1),
                    }
                    for r in top_tools
                ],
            },
            "failures": [
                {
                    "tool": r["tool_name"],
                    "error": r["error_message"][:100] if r["error_message"] else None,
                    "count": r["count"],
                }
                for r in failures
            ],
            "chains": {
                "total": chain_count,
                "detected_auto": db.execute(
                    "SELECT COUNT(*) FROM tool_chains WHERE is_auto_detected = 1"
                ).fetchone()[0],
            },
            "hot_cache": {
                "size": len(self._hot_cache),
                "tools": list(self._hot_cache.keys()),
            },
        }

    async def load_hot_cache_from_db(self):
        """Load hot cache from persistent storage on startup."""
        db = self._get_db()

        cursor = db.execute("""
            SELECT tool_name, rank, call_count, embedding, schema_json, description
            FROM hot_tools
            ORDER BY rank
        """)

        # PC-B-001: this runs on the mandatory init path — get_analytics_instance()
        # awaits it unconditionally and compass()/execute() call that UNWRAPPED,
        # so a raw exception here surfaces as a bare traceback through the MCP
        # JSON-RPC envelope (violates the BE-B-004 "handlers never raise"
        # invariant). Persisted blobs can be corrupt: a bad schema_json raises
        # json.JSONDecodeError and a truncated embedding blob raises ValueError
        # from np.frombuffer — neither is a sqlite3.Error, so nothing above
        # catches them. Guard per-row and SKIP a bad row rather than aborting the
        # whole load, matching record_search / record_tool_call which degrade
        # instead of raising. A per-row skip keeps every good row loadable.
        skipped = 0
        for row in cursor.fetchall():
            try:
                embedding = None
                if row["embedding"]:
                    blob = row["embedding"]
                    # SC-002 (indexer._cache_get): validate the blob byte length
                    # before np.frombuffer. float32 == 4 bytes/element; a length
                    # that is not a whole multiple of 4 is a truncated/corrupt
                    # write and would raise ValueError.
                    if len(blob) % 4 != 0:
                        raise ValueError(
                            f"corrupt embedding blob for {row['tool_name']}: "
                            f"{len(blob)} bytes not a multiple of 4"
                        )
                    embedding = np.frombuffer(blob, dtype=np.float32)

                schema = None
                if row["schema_json"]:
                    schema = json.loads(row["schema_json"])

                self._hot_cache[row["tool_name"]] = HotToolEntry(
                    tool_name=row["tool_name"],
                    rank=row["rank"],
                    call_count=row["call_count"],
                    embedding=embedding,
                    schema=schema,
                    description=row["description"] or "",
                    last_called_at=None,
                )
            except (json.JSONDecodeError, ValueError, TypeError, sqlite3.Error) as e:
                # Skip the bad row; a single corrupt persisted blob must not
                # take down the whole hot-cache load (and thus the first
                # compass()/execute() of the process).
                skipped += 1
                logger.warning(
                    f"Skipping corrupt hot_tools row "
                    f"'{row['tool_name'] if 'tool_name' in row.keys() else '?'}': {e}"
                )

        logger.info(
            f"Loaded {len(self._hot_cache)} tools into hot cache from DB"
            + (f" ({skipped} corrupt row(s) skipped)" if skipped else "")
        )

    def close(self):
        """Close database connection.

        ANL-A-003: acquire BOTH locks — self._init_lock serializes close
        against the lazy reopen in _get_db (which guards on self._init_lock),
        and self._lock serializes against in-flight writers. Order is
        init-lock then write-lock, matching no other acquisition order in the
        class (writers never hold the init lock), so no deadlock cycle exists.
        Set self._closed under the init lock so a concurrent record_* observes
        the closed state and refuses to reopen.
        """
        with self._init_lock:
            with self._lock:
                self._closed = True
                if self.db:
                    self.db.close()
                    self.db = None


# Singleton instance
_analytics_instance: Optional[CompassAnalytics] = None


def get_analytics() -> CompassAnalytics:
    """Get or create the analytics singleton."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = CompassAnalytics()
    return _analytics_instance
