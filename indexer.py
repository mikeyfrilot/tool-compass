"""
Tool Compass - Indexer Module
Builds and manages the HNSW index for semantic tool discovery.
"""

import hnswlib
import sqlite3
import json
import asyncio
import hashlib
import threading
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import time

from embedder import Embedder, EMBEDDING_DIM
from tool_manifest import ToolDefinition, get_all_tools

logger = logging.getLogger(__name__)

# Configuration
DB_DIR = Path(__file__).parent / "db"
HNSW_INDEX_PATH = DB_DIR / "compass.hnsw"
SQLITE_DB_PATH = DB_DIR / "tools.db"

# HNSW Parameters (tuned for ~100-1000 tools). Defaults preserved here;
# CompassConfig (BE-B-008) overrides them at CompassIndex.__init__.
HNSW_M = 16  # Number of connections per element
HNSW_EF_CONSTRUCTION = 200  # Size of dynamic candidate list during construction
HNSW_EF_SEARCH = 50  # Size of dynamic candidate list during search

# BE-B-008: log a one-time warning when corpus crosses this threshold so
# operators consider raising M / ef_search before recall starts drifting.
_HNSW_SCALE_WARN_TOOLS = 5000


@dataclass
class SearchResult:
    """Result from compass search."""

    tool: ToolDefinition
    score: float  # Cosine similarity (higher = better)
    rank: int


class CompassIndex:
    """
    HNSW-based index for semantic tool discovery.

    Architecture:
    - HNSW index stores tool embeddings for O(log n) search
    - SQLite stores tool metadata for retrieval
    - Embedder generates vectors via Ollama
    """

    def __init__(
        self,
        index_path: Path = HNSW_INDEX_PATH,
        db_path: Path = SQLITE_DB_PATH,
        embedder: Optional[Embedder] = None,
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
    ):
        """Initialize CompassIndex.

        BE-B-008: hnsw_m / hnsw_ef_construction / hnsw_ef_search are now
        runtime-tunable via CompassConfig. Defaults preserved; callers pass
        explicit overrides when they have a config in hand.
        """
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.embedder = embedder or Embedder()
        self.hnsw_m = int(hnsw_m) if hnsw_m is not None else HNSW_M
        self.hnsw_ef_construction = (
            int(hnsw_ef_construction)
            if hnsw_ef_construction is not None
            else HNSW_EF_CONSTRUCTION
        )
        self.hnsw_ef_search = (
            int(hnsw_ef_search) if hnsw_ef_search is not None else HNSW_EF_SEARCH
        )

        self.index: Optional[hnswlib.Index] = None
        self.db: Optional[sqlite3.Connection] = None
        self._id_to_name: Dict[int, str] = {}
        # BE-B-008: histogram of returned similarity scores (bounded) to
        # surface recall drift before users complain.
        self._score_samples: deque = deque(maxlen=2000)
        self._scale_warn_emitted = False

        # Embedding cache counters (IDX-FT-003).
        self._cache_hits = 0
        self._cache_misses = 0

        # IDX-COMPOSED-002: while build_index holds its BEGIN IMMEDIATE
        # rebuild transaction, embedding-cache mutations must NOT commit the
        # shared connection (a mid-rebuild commit prematurely persists the
        # DELETE + INSERTs, so a later HNSW-save failure can't roll them back →
        # DB/HNSW divergence). During a rebuild this holds a list of deferred
        # cache ops; _cache_put / _cache_get's self-heal append to it instead
        # of committing, and build_index flushes it AFTER its own commit. When
        # None (the normal case, e.g. add_single_tool), cache writes commit
        # immediately as before.
        self._deferred_cache_ops: Optional[List[tuple]] = None

        # BE-A-003: serialize DB writes across threads. search_sync() dispatches
        # search() to a worker thread via ThreadPoolExecutor when called from
        # inside a running event loop (Gradio, nested MCP). The sqlite3
        # connection is opened with check_same_thread=False (below in _init_db)
        # so cross-thread access is permitted, but concurrent writes would
        # still race; this lock guards mutating execs and commits.
        self._db_write_lock = threading.Lock()

        # Ensure db directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _compute_text_hash(self, text: str) -> str:
        """Compute stable cache key from (text, provider, base_url, model).

        BE-FT-PE-001: with a pluggable embedding backend the same text+model
        can produce DIFFERENT vectors across providers (e.g. ollama
        nomic-embed-text vs an OpenAI-compatible server), and even across
        endpoints of the same provider. Folding the provider NAME and the
        base_url into the key — in addition to the model — guarantees a cache
        entry written by one provider can never be served to another, so
        switching ``embedding_provider`` / ``embedding_base_url`` can't return
        a stale cross-provider vector. The dim self-heal in ``_cache_get`` is
        unaffected (it keys on EMBEDDING_DIM + BLOB byte length, not this hash).

        ``provider_name`` is read defensively: test mocks and any embedder
        predating the seam expose only ``base_url`` / ``model``, so a missing
        attribute degrades to "unknown" rather than raising.
        """
        provider_name = getattr(self.embedder, "provider_name", "unknown")
        base_url = getattr(self.embedder, "base_url", "unknown")
        model = getattr(self.embedder, "model", "unknown")
        payload = f"{text}||{provider_name}||{base_url}||{model}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_get(self, text_hash: str) -> Optional[np.ndarray]:
        """Return cached float32 vector or None.

        On dim mismatch (e.g., stale row from old model), treat as miss and
        delete the bad row so it gets re-populated with the current model.
        """
        if self.db is None:
            return None
        try:
            row = self.db.execute(
                "SELECT vector, dim FROM embedding_cache WHERE text_hash = ?",
                (text_hash,),
            ).fetchone()
        except sqlite3.OperationalError:
            # Table may not exist yet on a freshly-opened legacy DB.
            return None
        if row is None:
            return None
        dim = int(row["dim"])
        if dim != EMBEDDING_DIM:
            # Stale entry from a different-dim model — drop and miss.
            self._delete_cache_row(text_hash)
            return None
        # SC-002: the column-dim check above is NOT sufficient. A row whose
        # dim==EMBEDDING_DIM but whose BLOB byte length is inconsistent
        # (truncated / corrupt write) makes reshape(dim) raise ValueError.
        # Because _cache_get runs inside build_index's BEGIN IMMEDIATE txn,
        # an uncaught ValueError there rolls back and re-raises EVERY rebuild
        # forever, defeating the documented self-heal. Validate the actual
        # byte length (float32 == 4 bytes/element) before reshape; on
        # mismatch, treat as a miss and delete the bad row (mirroring the
        # column-dim-mismatch branch above) so the next pass re-populates it.
        blob = row["vector"]
        if blob is None or len(blob) != dim * 4:
            self._delete_cache_row(text_hash)
            return None
        vector = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        # frombuffer returns a read-only view; copy so hnswlib can use it.
        return vector.copy()

    def _delete_cache_row(self, text_hash: str) -> None:
        """Self-heal delete of a bad embedding_cache row (IDX-COMPOSED-002).

        During a build_index rebuild (``self._deferred_cache_ops`` is a list)
        the delete is DEFERRED — committing here would end build_index's open
        BEGIN IMMEDIATE transaction prematurely. Otherwise it deletes + commits
        immediately, preserving the original self-heal behaviour outside a
        rebuild (e.g. add_single_tool, direct _cache_get probes).
        """
        if self.db is None:
            return
        if self._deferred_cache_ops is not None:
            self._deferred_cache_ops.append(("delete", text_hash))
            return
        with self._db_write_lock:
            self.db.execute(
                "DELETE FROM embedding_cache WHERE text_hash = ?", (text_hash,)
            )
            self.db.commit()

    def _cache_put(
        self, text_hash: str, vector: np.ndarray, dim: int, provider: str
    ) -> None:
        """BLOB-encode and store a vector. No-op if DB is unavailable.

        IDX-COMPOSED-002: during a build_index rebuild the write is DEFERRED
        (appended to ``self._deferred_cache_ops``) and flushed AFTER the
        rebuild's own commit, so it can't prematurely commit the open
        transaction. Outside a rebuild it writes + commits immediately.
        """
        if self.db is None:
            return
        vec_f32 = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._deferred_cache_ops is not None:
            self._deferred_cache_ops.append(
                ("put", text_hash, vec_f32.tobytes(), int(dim), provider)
            )
            return
        try:
            with self._db_write_lock:
                self.db.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache (text_hash, vector, dim, provider)
                    VALUES (?, ?, ?, ?)
                    """,
                    (text_hash, vec_f32.tobytes(), int(dim), provider),
                )
                self.db.commit()
        except sqlite3.OperationalError as e:
            logger.debug(f"embedding_cache put failed: {e}")

    def _flush_deferred_cache_ops_list(self, ops: List[tuple]) -> None:
        """Apply deferred embedding-cache ops AFTER build_index's own commit.

        IDX-COMPOSED-002: cache puts/deletes accumulated during a rebuild are
        applied here in a single committed batch. Called only on the success
        path (after the rebuild's commit); the failure path discards the
        pending ops so a rolled-back rebuild leaves no cache side effects.
        Best-effort: a cache-table problem must never fail the rebuild that
        already succeeded.
        """
        if not ops or self.db is None:
            return
        try:
            with self._db_write_lock:
                for op in ops:
                    if op[0] == "put":
                        _, text_hash, blob, dim, provider = op
                        self.db.execute(
                            """
                            INSERT OR REPLACE INTO embedding_cache
                                (text_hash, vector, dim, provider)
                            VALUES (?, ?, ?, ?)
                            """,
                            (text_hash, blob, dim, provider),
                        )
                    elif op[0] == "delete":
                        _, text_hash = op
                        self.db.execute(
                            "DELETE FROM embedding_cache WHERE text_hash = ?",
                            (text_hash,),
                        )
                self.db.commit()
        except sqlite3.OperationalError as e:
            logger.debug(f"deferred embedding_cache flush failed: {e}")

    def get_cache_stats(self) -> Dict:
        """Return embedding-cache hit/miss/size stats (IDX-FT-003)."""
        size = 0
        if self.db is not None:
            try:
                row = self.db.execute(
                    "SELECT COUNT(*) AS c FROM embedding_cache"
                ).fetchone()
                size = int(row["c"]) if row else 0
            except sqlite3.OperationalError:
                size = 0
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total) if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": size,
            "hit_rate": hit_rate,
        }

    def _init_db(self):
        """Initialize SQLite database for tool metadata."""
        # BE-A-003: check_same_thread=False allows the connection to be used
        # from worker threads (search_sync ThreadPoolExecutor path). Cross-
        # thread mutations are still serialized via self._db_write_lock.
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row

        with self._db_write_lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS tools (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    server TEXT NOT NULL,
                    parameters TEXT,  -- JSON
                    examples TEXT,    -- JSON
                    is_core INTEGER DEFAULT 0,
                    embedding_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
                CREATE INDEX IF NOT EXISTS idx_tools_server ON tools(server);
                CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);

                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.db.commit()

        # Runtime cache hit/miss counters (IDX-FT-003). Reset only on process
        # lifetime — persisted cache entries live across runs.
        if not hasattr(self, "_cache_hits"):
            self._cache_hits = 0
        if not hasattr(self, "_cache_misses"):
            self._cache_misses = 0

    def _load_id_mapping(self):
        """Load ID to name mapping from database."""
        cursor = self.db.execute("SELECT id, name FROM tools")
        self._id_to_name = {row["id"]: row["name"] for row in cursor.fetchall()}

    async def build_index(
        self,
        tools: Optional[List[ToolDefinition]] = None,
        use_cache: bool = True,
    ):
        """
        Build HNSW index from tool definitions.

        Args:
            tools: List of tools to index. Uses manifest if not provided.
            use_cache: When True (default), reuse cached embeddings for any
                tool whose embedding_text (+ provider+model) has been seen
                before. Set False to force a fresh Ollama pass.
        """
        if tools is None:
            tools = get_all_tools()

        logger.info(f"Building index for {len(tools)} tools...")
        start_time = time.time()

        # Initialize database
        self._init_db()

        # Empty tool set: clear state and initialize an empty HNSW index so
        # search() returns [] cleanly (see IDX-A-002 regression).
        if not tools:
            built_at = time.time()
            with self._db_write_lock:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    self.db.execute("DELETE FROM tools")
                    self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
                    # BE-A2-001: allow_replace_deleted=True permits re-adding a
                    # previously-deleted label on the UPDATE path in
                    # add_single_tool. Without it, hnswlib raises on duplicate
                    # labels and silently breaks updates of changed tools.
                    self.index.init_index(
                        max_elements=1000,
                        ef_construction=self.hnsw_ef_construction,
                        M=self.hnsw_m,
                        allow_replace_deleted=True,
                    )
                    self.index.set_ef(self.hnsw_ef_search)
                    self.index.save_index(str(self.index_path))
                    # BE-A-013: persist a wall-clock timestamp so
                    # tool_compass_index_age_seconds can compute real age.
                    self.db.execute(
                        "INSERT OR REPLACE INTO index_meta (key, value) VALUES "
                        "('built_at_unix', ?), ('tool_count', '0')",
                        (str(built_at),),
                    )
                    self.db.commit()
                except Exception:
                    self.db.rollback()
                    raise
            self._id_to_name = {}
            logger.info("build_index completed with 0 tools")
            # BE-A-012: callers (gateway.sync_from_backends) read
            # result['tools_indexed']; previously this branch returned None
            # and the caller TypeError'd on subscription. Return the same
            # dict shape as the populated branch.
            return {
                "tools_indexed": 0,
                "embedding_time": 0.0,
                "total_time": time.time() - start_time,
                "index_path": str(self.index_path),
                "db_path": str(self.db_path),
            }

        # Wrap the DELETE → INSERT → embed → add_items sequence in a single
        # transaction. Only commit AFTER HNSW save succeeds, so a failure
        # leaves the previous DB state intact (no orphan SQLite rows).
        #
        # IDX-COMPOSED-002: route embedding-cache mutations made during this
        # rebuild into a deferred list so _cache_put / _cache_get's self-heal
        # can't commit the shared connection mid-transaction (which would
        # prematurely persist the DELETE + INSERTs and defeat the atomic
        # rollback below). The list is flushed AFTER the rebuild's own commit
        # on success, and discarded on failure. _deferred_cache_ops is always
        # reset to None (post-rebuild cache writes commit immediately again).
        self._deferred_cache_ops = []
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # Clear existing data
            self.db.execute("DELETE FROM tools")

            # Insert tools and collect texts for embedding
            embedding_texts = []
            tool_ids = []

            for i, tool in enumerate(tools):
                embedding_text = tool.embedding_text()
                embedding_texts.append(embedding_text)

                # Insert into SQLite (still inside the open transaction)
                cursor = self.db.execute(
                    """
                    INSERT INTO tools (name, description, category, server, parameters, examples, is_core, embedding_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        tool.name,
                        tool.description,
                        tool.category,
                        tool.server,
                        json.dumps(tool.parameters),
                        json.dumps(tool.examples),
                        1 if tool.is_core else 0,
                        embedding_text,
                    ),
                )
                tool_ids.append(cursor.lastrowid)

            logger.info(f"Inserted {len(tools)} tools into database (uncommitted)")

            # Partition into cache hits vs. misses (IDX-FT-003). When cache is
            # disabled, everything is a "miss" and gets embedded fresh.
            provider = getattr(self.embedder, "base_url", "unknown")
            hashes: List[str] = []
            cached_vecs: Dict[int, np.ndarray] = {}
            miss_indices: List[int] = []
            miss_texts: List[str] = []

            for i, text in enumerate(embedding_texts):
                h = self._compute_text_hash(text) if use_cache else ""
                hashes.append(h)
                hit = self._cache_get(h) if use_cache else None
                if hit is not None:
                    cached_vecs[i] = hit
                    self._cache_hits += 1
                else:
                    miss_indices.append(i)
                    miss_texts.append(text)
                    if use_cache:
                        self._cache_misses += 1

            logger.info(
                f"Embedding cache: {len(cached_vecs)} hits, {len(miss_texts)} misses"
            )

            # Generate embeddings only for misses
            embed_start = time.time()
            if miss_texts:
                logger.info(
                    f"Generating {len(miss_texts)} embeddings via Ollama..."
                )
                miss_embeddings = await self.embedder.embed_batch(miss_texts)
                if miss_embeddings.shape != (len(miss_texts), EMBEDDING_DIM):
                    raise RuntimeError(
                        f"Embedding shape mismatch: got {miss_embeddings.shape}, "
                        f"expected ({len(miss_texts)}, {EMBEDDING_DIM})"
                    )
                # Populate cache for misses
                if use_cache:
                    for j, mi in enumerate(miss_indices):
                        self._cache_put(
                            hashes[mi],
                            miss_embeddings[j],
                            EMBEDDING_DIM,
                            provider,
                        )
            else:
                miss_embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            embed_time = time.time() - embed_start

            # Stitch the full embedding matrix back together
            embeddings = np.zeros((len(tools), EMBEDDING_DIM), dtype=np.float32)
            for i, vec in cached_vecs.items():
                embeddings[i] = vec
            for j, mi in enumerate(miss_indices):
                embeddings[mi] = miss_embeddings[j]
            logger.info(
                f"Assembled {len(embeddings)} embeddings in {embed_time:.2f}s"
            )

            # Validate shape before add_items to fail fast on partial embeds.
            if embeddings.shape != (len(tools), EMBEDDING_DIM):
                raise RuntimeError(
                    f"Embedding shape mismatch: got {embeddings.shape}, "
                    f"expected ({len(tools)}, {EMBEDDING_DIM})"
                )

            # Build HNSW index
            logger.info("Building HNSW index...")
            # BE-B-008: scale warning when corpus is large enough to warrant
            # operator review of the HNSW knobs.
            if len(tools) >= _HNSW_SCALE_WARN_TOOLS and not self._scale_warn_emitted:
                logger.warning(
                    f"Indexing {len(tools)} tools — at this scale, consider "
                    f"reviewing hnsw_m ({self.hnsw_m}), hnsw_ef_construction "
                    f"({self.hnsw_ef_construction}), hnsw_ef_search "
                    f"({self.hnsw_ef_search}) in CompassConfig."
                )
                self._scale_warn_emitted = True

            self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
            # BE-A2-001: allow_replace_deleted=True permits replacing a label
            # marked deleted on the UPDATE path in add_single_tool. Without
            # this flag, hnswlib raises on duplicate labels and silently fails
            # updates of changed tools.
            self.index.init_index(
                max_elements=max(len(tools) * 2, 1000),  # Room to grow
                ef_construction=self.hnsw_ef_construction,
                M=self.hnsw_m,
                allow_replace_deleted=True,
            )

            # Add vectors with tool IDs
            self.index.add_items(embeddings, tool_ids)
            self.index.set_ef(self.hnsw_ef_search)

            # Save index — only after this succeeds do we commit SQLite.
            self.index.save_index(str(self.index_path))

            # Update metadata.
            # BE-A-013: persist both build_time (elapsed seconds; legacy) and
            # built_at_unix (wall-clock timestamp). get_stats() reads
            # built_at_unix so tool_compass_index_age_seconds is accurate.
            self.db.execute(
                """
                INSERT OR REPLACE INTO index_meta (key, value) VALUES
                ('tool_count', ?),
                ('embedding_dim', ?),
                ('hnsw_m', ?),
                ('hnsw_ef_construction', ?),
                ('hnsw_ef_search', ?),
                ('build_time', ?),
                ('built_at_unix', ?)
            """,
                (
                    str(len(tools)),
                    str(EMBEDDING_DIM),
                    str(self.hnsw_m),
                    str(self.hnsw_ef_construction),
                    str(self.hnsw_ef_search),
                    str(time.time() - start_time),
                    str(time.time()),
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            # IDX-COMPOSED-002: discard deferred cache ops — the rebuild rolled
            # back, so its cache writes/self-heals must not be applied.
            self._deferred_cache_ops = None
            logger.error("build_index failed; rolled back SQLite transaction")
            raise
        else:
            # IDX-COMPOSED-002: the rebuild's own commit landed — now (and only
            # now) flush the deferred cache puts/deletes in their own batch.
            pending = self._deferred_cache_ops
            self._deferred_cache_ops = None
            if pending:
                self._flush_deferred_cache_ops_list(pending)

        # Load ID mapping
        self._load_id_mapping()

        total_time = time.time() - start_time
        logger.info(f"Index built in {total_time:.2f}s")

        return {
            "tools_indexed": len(tools),
            "embedding_time": embed_time,
            "total_time": total_time,
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
        }

    def load_index(self) -> bool:
        """
        Load existing index from disk.

        Integrity checks (IDX-B-001):
        - Compare persisted embedding_dim to the code's EMBEDDING_DIM and
          raise RuntimeError on mismatch, so the gateway can degrade to
          lexical search instead of crashing searches silently with bad
          vectors.
        - After load, warn (don't crash) if HNSW count and DB row count
          disagree — user sees degraded recall but the server still runs.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.index_path.exists() or not self.db_path.exists():
            logger.warning("Index files not found")
            return False

        try:
            # Load database
            self._init_db()
            self._load_id_mapping()

            # Pre-load integrity check: read persisted dim/M from index_meta.
            cursor = self.db.execute(
                "SELECT key, value FROM index_meta WHERE key IN ('embedding_dim', 'hnsw_m')"
            )
            meta = {row["key"]: row["value"] for row in cursor.fetchall()}
            saved_dim = meta.get("embedding_dim")
            if saved_dim is not None:
                try:
                    saved_dim_int = int(saved_dim)
                except (TypeError, ValueError):
                    saved_dim_int = None
                if saved_dim_int is not None and saved_dim_int != EMBEDDING_DIM:
                    msg = (
                        f"Index file uses {saved_dim}-dim vectors but code "
                        f"expects {EMBEDDING_DIM}. The embedding model likely "
                        f"changed. Delete {self.index_path} and run sync to "
                        f"rebuild."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

            # Load HNSW index
            self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
            # BE-A2-001: pass allow_replace_deleted=True at load so the
            # restored index supports mark_deleted + replace_deleted on the
            # add_single_tool UPDATE path. Without it, persisted indexes
            # silently revert to default-strict mode after restart.
            self.index.load_index(
                str(self.index_path), allow_replace_deleted=True
            )
            self.index.set_ef(self.hnsw_ef_search)

            # Post-load sanity: HNSW count vs DB mapping. A mismatch hurts
            # recall but isn't fatal — warn and continue. Rebuild via sync
            # will heal this.
            hnsw_count = self.index.get_current_count()
            db_count = len(self._id_to_name)
            if hnsw_count != db_count:
                logger.warning(
                    f"Index integrity: HNSW has {hnsw_count} vectors but DB "
                    f"has {db_count} tools. Search quality may be degraded — "
                    f"rebuild the index to resolve."
                )

            logger.info(f"Loaded index with {len(self._id_to_name)} tools")
            return True

        except RuntimeError:
            # Dim mismatch is a hard error the gateway needs to see.
            raise
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def _get_tool_by_id(self, tool_id: int) -> Optional[ToolDefinition]:
        """Retrieve tool definition by ID."""
        cursor = self.db.execute(
            """
            SELECT name, description, category, server, parameters, examples, is_core
            FROM tools WHERE id = ?
        """,
            (tool_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        # GW-A-002 sibling: guard json.loads on possibly-corrupt tools-table
        # rows. _get_tool_by_id runs per-result inside search(); without this a
        # single malformed row raised JSONDecodeError and poisoned the ENTIRE
        # result set (everything degraded to lexical) instead of dropping the
        # one bad field. Fall back to empty defaults for the corrupt column.
        try:
            parameters = json.loads(row["parameters"]) if row["parameters"] else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("tool %r: malformed parameters JSON; using {}", row["name"])
            parameters = {}
        try:
            examples = json.loads(row["examples"]) if row["examples"] else []
        except (json.JSONDecodeError, TypeError):
            logger.warning("tool %r: malformed examples JSON; using []", row["name"])
            examples = []
        return ToolDefinition(
            name=row["name"],
            description=row["description"],
            category=row["category"],
            server=row["server"],
            parameters=parameters,
            examples=examples,
            is_core=bool(row["is_core"]),
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        server_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for tools matching the query intent.

        Args:
            query: Natural language description of task/intent
            top_k: Number of results to return
            category_filter: Optional category to filter by
            server_filter: Optional server to filter by

        Returns:
            List of SearchResult ordered by relevance
        """
        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() or build_index() first."
            )

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Guard against empty index — knn_query crashes on k=0 or k > count.
        count = self.index.get_current_count()
        if count == 0:
            return []

        # Search HNSW (get more than needed for filtering), clamped to [1, count].
        search_k = max(1, min(top_k * 3, count))
        # BE-B-002: time the HNSW search separately from Ollama-side latency
        # so dashboards can split slow-HNSW-with-healthy-Ollama from the
        # inverse failure mode.
        knn_start = time.monotonic()
        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1), k=search_k
        )
        knn_latency_ms = (time.monotonic() - knn_start) * 1000.0
        if not hasattr(self, "_hnsw_latency_samples"):
            self._hnsw_latency_samples = deque(maxlen=1000)
        self._hnsw_latency_samples.append(knn_latency_ms)

        # Convert distances to similarities (hnswlib returns 1 - cosine for cosine space)
        similarities = 1 - distances[0]
        # BE-B-008: track score samples so a leftward drift in p50 surfaces
        # degrading recall (e.g. corpus outgrew the HNSW knobs).
        for s in similarities[: min(top_k, len(similarities))]:
            try:
                self._score_samples.append(float(s))
            except Exception:
                pass

        results = []
        for label, similarity in zip(labels[0], similarities):
            tool = self._get_tool_by_id(int(label))
            if tool is None:
                continue

            # Apply filters
            if category_filter and tool.category != category_filter:
                continue
            if server_filter and tool.server != server_filter:
                continue

            results.append(
                SearchResult(tool=tool, score=float(similarity), rank=len(results) + 1)
            )

            if len(results) >= top_k:
                break

        return results

    def search_sync(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Synchronous search wrapper.

        Safe to call from either a normal synchronous context or inside an
        active event loop (e.g., Gradio, FastMCP). Mirrors SyncEmbedder._run.
        """
        coro = self.search(query, top_k, **kwargs)
        try:
            asyncio.get_running_loop()
            # A loop is already running — dispatch to a worker thread with
            # its own loop so we don't deadlock the caller's loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly.
            return asyncio.run(coro)

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.db is None:
            self._init_db()

        stats = {}

        # Tool counts
        cursor = self.db.execute("SELECT COUNT(*) as count FROM tools")
        stats["total_tools"] = cursor.fetchone()["count"]

        cursor = self.db.execute(
            "SELECT COUNT(*) as count FROM tools WHERE is_core = 1"
        )
        stats["core_tools"] = cursor.fetchone()["count"]

        # Category breakdown
        cursor = self.db.execute(
            "SELECT category, COUNT(*) as count FROM tools GROUP BY category"
        )
        stats["by_category"] = {
            row["category"]: row["count"] for row in cursor.fetchall()
        }

        # Server breakdown
        cursor = self.db.execute(
            "SELECT server, COUNT(*) as count FROM tools GROUP BY server"
        )
        stats["by_server"] = {row["server"]: row["count"] for row in cursor.fetchall()}

        # Index metadata
        cursor = self.db.execute("SELECT key, value FROM index_meta")
        stats["index_meta"] = {row["key"]: row["value"] for row in cursor.fetchall()}

        # Index age + orphan counts (IDX-B-008 + BE-A-013).
        # build_time is the duration of the most recent build in seconds.
        # built_at_unix (added in BE-A-013) is the wall-clock timestamp of
        # when that build completed; tool_compass_index_age_seconds reads
        # this. We keep build_time around for backwards compatibility but
        # prefer built_at_unix where present.
        build_time_raw = stats["index_meta"].get("build_time")
        built_at_raw = stats["index_meta"].get("built_at_unix")
        stats["last_build_at"] = built_at_raw or build_time_raw
        stats["index_age_seconds"] = None
        try:
            if built_at_raw is not None:
                stats["index_age_seconds"] = max(
                    0.0, time.time() - float(built_at_raw)
                )
            elif build_time_raw is not None:
                # Legacy: if build_time happens to look like a unix timestamp,
                # treat it as one. Otherwise leave as None.
                bt = float(build_time_raw)
                if bt > 1_000_000_000:
                    stats["index_age_seconds"] = max(0.0, time.time() - bt)
        except (TypeError, ValueError):
            stats["index_age_seconds"] = None

        # HNSW stats
        if self.index:
            hnsw_count = self.index.get_current_count()
            stats["hnsw"] = {
                "current_count": hnsw_count,
                "max_elements": self.index.get_max_elements(),
                "ef": self.index.ef,
                "m": self.hnsw_m,
                "ef_construction": self.hnsw_ef_construction,
                "ef_search": self.hnsw_ef_search,
            }
            # Orphaned vectors = HNSW has entries that aren't in the DB
            # mapping. Clamp at 0 — DB can legitimately have rows not yet
            # loaded into the id mapping, and we don't want a negative count
            # confusing operators.
            stats["orphaned_vector_count"] = max(
                0, hnsw_count - len(self._id_to_name)
            )
        else:
            stats["orphaned_vector_count"] = 0

        # BE-B-002: HNSW search-latency percentiles (separate from Ollama).
        hnsw_samples = list(getattr(self, "_hnsw_latency_samples", []) or [])
        if hnsw_samples:
            sorted_s = sorted(hnsw_samples)
            n = len(sorted_s)
            stats["hnsw_search_latency_ms_p50"] = sorted_s[n // 2]
            stats["hnsw_search_latency_ms_p95"] = sorted_s[min(n - 1, int(n * 0.95))]
        else:
            stats["hnsw_search_latency_ms_p50"] = 0.0
            stats["hnsw_search_latency_ms_p95"] = 0.0

        # BE-B-008: returned similarity score percentiles. A persistent
        # leftward drift in p50 means recall is degrading.
        score_samples = list(self._score_samples or [])
        if score_samples:
            sorted_sc = sorted(score_samples)
            n = len(sorted_sc)
            stats["search_score_p50"] = sorted_sc[n // 2]
            stats["search_score_p95"] = sorted_sc[min(n - 1, int(n * 0.95))]
        else:
            stats["search_score_p50"] = 0.0
            stats["search_score_p95"] = 0.0

        # Embedder metrics (IDX-B-003 + IDX-B-008 surface).
        try:
            stats["embedder_stats"] = self.embedder.get_stats()
        except Exception as e:  # defensive — never let stats crash
            logger.debug(f"embedder.get_stats failed: {e}")
            stats["embedder_stats"] = None

        return stats

    async def add_single_tool(self, tool: ToolDefinition) -> bool:
        """
        Add a single tool to the index without full rebuild.
        HNSW supports dynamic element addition.

        Args:
            tool: The tool definition to add

        Returns:
            True if added successfully, False otherwise
        """
        if self.index is None or self.db is None:
            logger.error("Index not initialized. Call load_index() first.")
            return False

        try:
            # Generate embedding FIRST — if Ollama fails we never touched the DB.
            embedding_text = tool.embedding_text()
            # Consult embedding cache (IDX-FT-003) — skip Ollama on hit.
            text_hash = self._compute_text_hash(embedding_text)
            embedding = self._cache_get(text_hash)
            if embedding is not None:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
                embedding = await self.embedder.embed(embedding_text)
                provider = getattr(self.embedder, "base_url", "unknown")
                self._cache_put(text_hash, embedding, EMBEDDING_DIM, provider)

            # Now do DB write + HNSW add inside a single transaction.
            # Check if tool already exists
            cursor = self.db.execute(
                "SELECT id FROM tools WHERE name = ?", (tool.name,)
            )
            existing = cursor.fetchone()

            with self._db_write_lock:
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    if existing:
                        # Update existing tool
                        tool_id = existing["id"]

                        self.db.execute(
                            """
                            UPDATE tools SET
                                description = ?, category = ?, server = ?,
                                parameters = ?, examples = ?, is_core = ?,
                                embedding_text = ?
                            WHERE id = ?
                        """,
                            (
                                tool.description,
                                tool.category,
                                tool.server,
                                json.dumps(tool.parameters),
                                json.dumps(tool.examples),
                                1 if tool.is_core else 0,
                                embedding_text,
                                tool_id,
                            ),
                        )
                    else:
                        # Insert new tool
                        cursor = self.db.execute(
                            """
                            INSERT INTO tools (name, description, category, server, parameters, examples, is_core, embedding_text)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                tool.name,
                                tool.description,
                                tool.category,
                                tool.server,
                                json.dumps(tool.parameters),
                                json.dumps(tool.examples),
                                1 if tool.is_core else 0,
                                embedding_text,
                            ),
                        )
                        tool_id = cursor.lastrowid

                    # Check if we need to resize the index
                    if self.index.get_current_count() >= self.index.get_max_elements() - 1:
                        # Need to resize - HNSW doesn't support dynamic resize, so we extend
                        new_max = self.index.get_max_elements() * 2
                        self.index.resize_index(new_max)
                        logger.info(f"Resized HNSW index to {new_max} elements")

                    # BE-A2-001: on the UPDATE path, hnswlib raises on a
                    # duplicate label by default. The index is initialized
                    # with allow_replace_deleted=True so we can mark the old
                    # label deleted and re-add with replace_deleted=True. This
                    # is the supported way to overwrite an existing vector.
                    if existing:
                        try:
                            self.index.mark_deleted(tool_id)
                        except RuntimeError as mark_err:
                            # mark_deleted raises if the label is already
                            # marked deleted (idempotent for our purposes) or
                            # not present in the index (HNSW/DB drift — treat
                            # as a fresh add). Log and continue.
                            logger.debug(
                                f"mark_deleted({tool_id}) skipped: {mark_err}"
                            )
                        self.index.add_items(
                            embedding.reshape(1, -1),
                            [tool_id],
                            replace_deleted=True,
                        )
                    else:
                        self.index.add_items(embedding.reshape(1, -1), [tool_id])

                    # Save index before committing SQLite.
                    self.index.save_index(str(self.index_path))

                    self.db.commit()
                except Exception:
                    self.db.rollback()
                    raise

            # Update ID mapping (post-commit, in-memory only)
            self._id_to_name[tool_id] = tool.name

            logger.info(f"Added tool to index: {tool.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add tool {tool.name}: {e}")
            return False

    async def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the database.
        Note: HNSW doesn't support element removal, so the vector remains
        but won't be returned in searches (no matching DB entry).

        For full cleanup, rebuild the index with build_index().

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if removed from DB, False otherwise
        """
        if self.db is None:
            logger.error("Database not initialized")
            return False

        try:
            cursor = self.db.execute(
                "SELECT id FROM tools WHERE name = ?", (tool_name,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Tool not found: {tool_name}")
                return False

            tool_id = row["id"]

            # Remove from database
            with self._db_write_lock:
                self.db.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
                self.db.commit()

            # Remove from ID mapping
            self._id_to_name.pop(tool_id, None)

            logger.info(f"Removed tool from index: {tool_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove tool {tool_name}: {e}")
            return False

    async def close(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
            self.db = None
        await self.embedder.close()


async def build_compass_index():
    """Build the compass index from scratch."""
    logging.basicConfig(level=logging.INFO)

    index = CompassIndex()

    # Check Ollama
    print("Checking Ollama availability...")
    if not await index.embedder.health_check():
        print("ERROR: Ollama not available or nomic-embed-text not loaded")
        print("Run: ollama pull nomic-embed-text")
        return

    # Build index
    print("\nBuilding Tool Compass index...")
    result = await index.build_index()

    print("\n✓ Index built successfully!")
    print(f"  Tools indexed: {result['tools_indexed']}")
    print(f"  Embedding time: {result['embedding_time']:.2f}s")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Index path: {result['index_path']}")
    print(f"  Database path: {result['db_path']}")

    # Test search
    print("\n--- Testing search ---")
    test_queries = [
        "read a file from disk",
        "generate an image with AI",
        "search for text in documents",
        "check git status",
        "analyze code quality",
    ]

    for query in test_queries:
        results = await index.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  {r.rank}. {r.tool.name} (score: {r.score:.3f})")

    await index.close()


if __name__ == "__main__":
    asyncio.run(build_compass_index())
