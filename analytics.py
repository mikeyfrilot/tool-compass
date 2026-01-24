"""
Tool Compass - Analytics Module
Tracks usage patterns, manages hot tool cache, and detects tool chains.
"""

import sqlite3
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Database path
ANALYTICS_DB_PATH = Path(__file__).parent / "db" / "compass_analytics.db"


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

        # Session tracking for chain detection
        self._session_id = hashlib.sha256(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        self._session_tool_sequence: List[str] = []
        self._call_count_since_refresh = 0

        # Ensure db directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.db is None:
            self.db = sqlite3.connect(str(self.db_path))
            self.db.row_factory = sqlite3.Row
            self._init_db()
        return self.db

    def _init_db(self):
        """Initialize analytics database with all tables."""
        db = self._get_db()

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
        """)
        db.commit()
        logger.info(f"Analytics database initialized at {self.db_path}")

    async def record_search(
        self,
        query: str,
        results: List[Any],  # List of SearchResult
        latency_ms: float,
        category_filter: Optional[str] = None,
        server_filter: Optional[str] = None,
    ):
        """Record a search query for analytics."""
        db = self._get_db()

        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:32]
        top_result = results[0].tool.name if results else None
        result_count = len(results)

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
        db = self._get_db()

        # Parse server from tool name
        server = tool_name.split(":")[0] if ":" in tool_name else "unknown"

        # Hash arguments for pattern detection (without sensitive data)
        args_hash = None
        if arguments:
            args_hash = hashlib.sha256(
                json.dumps(sorted(arguments.keys())).encode()
            ).hexdigest()[:16]

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

        # Update aggregated stats
        db.execute(
            """
            INSERT INTO tool_usage_stats (tool_name, call_count, success_count, failure_count, avg_latency_ms, last_called_at, last_success_at)
            VALUES (?, 1, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(tool_name) DO UPDATE SET
                call_count = call_count + 1,
                success_count = success_count + excluded.success_count,
                failure_count = failure_count + excluded.failure_count,
                avg_latency_ms = (avg_latency_ms * call_count + excluded.avg_latency_ms) / (call_count + 1),
                last_called_at = CURRENT_TIMESTAMP,
                last_success_at = CASE WHEN excluded.success_count > 0 THEN CURRENT_TIMESTAMP ELSE last_success_at END,
                updated_at = CURRENT_TIMESTAMP
        """,
            (
                tool_name,
                1 if success else 0,
                0 if success else 1,
                latency_ms,
                "CURRENT_TIMESTAMP" if success else None,
            ),
        )

        db.commit()

        # Track for chain detection
        self._session_tool_sequence.append(tool_name)
        if len(self._session_tool_sequence) > 20:
            # Keep last 20 tools, save pattern
            await self._save_chain_pattern()
            self._session_tool_sequence = self._session_tool_sequence[-10:]

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

        db = self._get_db()

        # Find subsequences of length 2-5
        for length in range(2, min(6, len(self._session_tool_sequence) + 1)):
            for i in range(len(self._session_tool_sequence) - length + 1):
                subseq = self._session_tool_sequence[i : i + length]
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

    async def refresh_hot_cache(self, embedder=None, index=None):
        """
        Update the hot cache with top N most used tools.
        Optionally load embeddings and schemas if embedder/index provided.
        """
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

        for row in patterns:
            tools = json.loads(row["tool_sequence"])
            row["sequence_hash"]

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
        since = datetime.now() - timedelta(hours=hours)
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

        for row in cursor.fetchall():
            embedding = None
            if row["embedding"]:
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)

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

        logger.info(f"Loaded {len(self._hot_cache)} tools into hot cache from DB")

    def close(self):
        """Close database connection."""
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
