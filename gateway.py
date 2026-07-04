"""
Tool Compass Gateway - MCP Proxy Server
A semantic routing gateway that aggregates multiple MCP servers.

Architecture (based on 2026 best practices):
- Semantic search via HNSW + nomic-embed-text (MCP-Zero pattern)
- Progressive disclosure: compass() -> describe() -> execute()
- Configurable backend connections (stdio subprocess)
- 98% token reduction vs loading all tool schemas

Usage:
    python gateway.py              # Start gateway server
    python gateway.py --sync       # Sync tools from backends and rebuild index
    python gateway.py --test       # Run test queries
    python gateway.py --config     # Show current configuration

HTTP mode (PORT env var set) exposes three operational endpoints:

    GET /health   — Liveness probe. Always 200 while the process is up.
                    Returns ``{status, server, version}``.
    GET /ready    — Deep readiness probe (GW-FT-003). 200 only when the index
                    is loaded, Ollama is reachable (or the circuit breaker is
                    closed), and at least one configured backend is connected.
                    Otherwise 503 with per-check detail. Result is cached for
                    30s so load-balancer polling cannot DoS Ollama.
    GET /metrics  — Prometheus text format (GW-FT-003). Exposes counters and
                    gauges for search volume, Ollama availability, per-backend
                    health, embedder p95 latency / failures, and index age /
                    orphaned-vector counts. No ``prometheus_client`` dep — the
                    text is emitted manually.
"""

import asyncio
import argparse
import logging
import json
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from typing import Optional, List, Dict, Any

from _version import __version__

# MCP imports
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    import sys as _sys

    # BE-A-016: diagnostics belong on stderr, not stdout — MCP stdio mode
    # treats stdout as JSON-RPC framing, so a print() here corrupts the
    # protocol envelope.
    print("FastMCP not installed. Install with: pip install mcp", file=_sys.stderr)
    raise

from indexer import CompassIndex, SearchResult
from tool_manifest import ToolDefinition
from config import (
    load_config,
    CompassConfig,
    CONFIG_PATH,
    redact_url_credentials,
    _redact_structural,
)
# Use simple backend client to avoid anyio conflicts when nested inside another MCP server
from backend_client_simple import SimpleBackendManager as BackendManager
from analytics import CompassAnalytics, get_analytics
from sync_manager import SyncManager, get_sync_manager
from chain_indexer import ChainIndexer, get_chain_indexer

# Configure logging
# CFGDOC-01: honor the LOG_LEVEL env var that .env.example advertises and
# docker-compose's dev profile sets to DEBUG. Previously hardcoded to INFO,
# making the documented dev-debug workflow a silent no-op. Unknown values fall
# back to INFO via getattr's default.
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("tool-compass-gateway")

# ---------------------------------------------------------------------------
# Runtime assumptions (single-process MCP server):
# - The gateway runs as a single process (FastMCP over stdio).
# - Module-level singletons are initialized lazily with asyncio.Lock
#   (double-checked locking) to handle concurrent coroutine access.
# - NOT thread-safe by design — MCP servers are single-threaded async.
# - For the Gradio UI (multi-threaded), see ui.py which uses threading.Lock.
# ---------------------------------------------------------------------------
_compass_index: Optional[CompassIndex] = None
_backend_manager: Optional[BackendManager] = None
_config: Optional[CompassConfig] = None
_analytics: Optional[CompassAnalytics] = None
_sync_manager: Optional[SyncManager] = None
_chain_indexer: Optional[ChainIndexer] = None
_startup_sync_done: bool = False

# GW-B-001 / GW-B-009: runtime health state surfaced via compass_status().
# The gateway degrades gracefully when a dependency is down — compass() falls
# back to lexical search, describe() falls back to the backend path — and
# every response envelope tells the user *why* results look different.
_health_state: Dict[str, Any] = {
    "ollama_available": True,
    "last_ollama_check": 0.0,
    "last_ollama_error": None,
    "index_available": True,
    "last_index_error": None,
}


# BE-B-011: registered by build_http_app() so health-state changes invalidate
# the /ready cache immediately. Module-level so tests don't import HTTP code.
_ready_cache_invalidators: List[Any] = []


def _invalidate_ready_cache() -> None:
    """Call every registered ready-cache invalidator (BE-B-011)."""
    for fn in _ready_cache_invalidators:
        try:
            fn()
        except Exception as e:
            logger.debug(f"ready cache invalidator failed: {e}")


def _mark_ollama_down(err: BaseException) -> None:
    """Record an Ollama failure so status + hints can surface it."""
    _health_state["ollama_available"] = False
    _health_state["last_ollama_check"] = time.time()
    _health_state["last_ollama_error"] = f"{type(err).__name__}: {err}"
    _invalidate_ready_cache()


def _mark_ollama_up() -> None:
    _health_state["ollama_available"] = True
    _health_state["last_ollama_check"] = time.time()
    _health_state["last_ollama_error"] = None
    _invalidate_ready_cache()


# ---------------------------------------------------------------------------
# BE-B-001: structured error envelope.
#
# All @mcp.tool() entry points that previously returned a bare
# `{error: "..."}` string now route through _error_envelope(). The shape mirrors
# RFC 9457 (Problem Details), MCP `isError`, and Stripe's 3-level taxonomy:
#
#   {type, title, code, category, detail, retryable, instance: trace_id,
#    retry_after_seconds?, suggestions?, nearest_tools?}
#
# LLM consumers can switch strategies on `code` (e.g. "tool_not_found" ->
# present nearest_tools), `category` ("validation" vs "service_unavailable"),
# and `retryable` (bool — don't re-issue the same call on validation errors).
# ---------------------------------------------------------------------------

# Closed enum of error categories. Keep this list small — LLM consumers
# pattern-match on it.
_ERROR_CATEGORIES = {
    "validation",
    "not_found",
    "backend_error",
    "service_unavailable",
    "configuration",
}

# Closed enum of error codes (extend deliberately, not opportunistically).
_ERROR_CODES = {
    "tool_not_found",
    "backend_unreachable",
    "backend_timeout",
    "backend_connect_failed",
    "ollama_unavailable",
    "index_unhealthy",
    "invalid_argument",
    "invalid_action",
    "sync_disabled",
    "analytics_disabled",
    "chain_indexing_disabled",
    "analytics_unavailable",
    "chain_indexer_unavailable",
    "sync_manager_unavailable",
    "execute_unhandled_exception",
}


def _error_envelope(
    code: str,
    title: str,
    detail: str,
    *,
    category: str = "backend_error",
    retryable: bool = False,
    trace_id: Optional[str] = None,
    retry_after_seconds: Optional[float] = None,
    nearest_tools: Optional[List[Dict[str, Any]]] = None,
    suggestions: Optional[List[str]] = None,
    **extras: Any,
) -> Dict[str, Any]:
    """Build a structured error envelope (BE-B-001).

    Args:
        code: Short machine-readable code from _ERROR_CODES. Unknown codes log
            a warning but pass through — never raise from an error helper.
        title: One-line human-readable summary (Nielsen #9: name the failure
            mode discretely).
        detail: Longer human-readable explanation. Safe to interpolate
            user-supplied values; never include secrets.
        category: One of _ERROR_CATEGORIES.
        retryable: True iff the caller may retry the same call and reasonably
            expect a different outcome (e.g. transient network error). False
            for validation / not-found.
        trace_id: Correlation id (used as the `instance` field per RFC 9457).
        retry_after_seconds: Hint for transient failures.
        nearest_tools: For tool_not_found, an array of {tool, score, server,
            category} entries the user might have meant. Critical LLM-recovery
            signal — let the agent retry against the suggested name without
            hallucinating.
        suggestions: Generic free-form next-action hints (legacy "hint" field
            ports cleanly into this).

    Returns:
        Dict with stable keys. Always carries `error: True` for fast detection
        AND `error: <legacy string>` shape compatibility via the "error" key
        being a dict with `code`, etc. Existing callers checking
        ``if "error" in resp`` keep working; new callers can branch on
        ``resp["error"]["code"]``.
    """
    if code not in _ERROR_CODES:
        logger.warning(f"_error_envelope: unknown code {code!r}")
    if category not in _ERROR_CATEGORIES:
        logger.warning(f"_error_envelope: unknown category {category!r}")

    payload: Dict[str, Any] = {
        "type": f"compass.error.{code}",
        "title": title,
        "code": code,
        "category": category,
        "detail": detail,
        "retryable": bool(retryable),
    }
    if trace_id is not None:
        payload["instance"] = trace_id
    if retry_after_seconds is not None:
        payload["retry_after_seconds"] = float(retry_after_seconds)
    if nearest_tools:
        payload["nearest_tools"] = nearest_tools
    if suggestions:
        payload["suggestions"] = suggestions
    for k, v in extras.items():
        payload[k] = v

    return {
        # Legacy callers (and tests) check `"error" in resp` — keep that True
        # by surfacing the string form here in addition to the structured
        # envelope. The structured form is canonical for LLM consumers.
        "error": detail,
        "error_envelope": payload,
        "trace_id": trace_id,
    }


def _augment_with_health(response: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp `degraded` + `degraded_reasons` onto a response (BE-B-003).

    Idempotent — re-applying never doubles up reasons. Safe to call on
    error envelopes (they're dicts).

    Hystrix / Nygard pattern: every response that was served from a fallback
    path (or whose surrounding subsystem is in degraded state) MUST carry a
    structured boolean, not just a human-readable warning. LLM agents skip
    prose warnings but act on structured flags.
    """
    if not isinstance(response, dict):
        return response

    reasons: List[str] = []
    if not _health_state.get("ollama_available", True):
        reasons.append("ollama_unavailable")
    if not _health_state.get("index_available", True):
        reasons.append("index_unhealthy")

    # If the response itself already declared degraded=True, preserve that.
    already_degraded = bool(response.get("degraded"))
    is_degraded = already_degraded or bool(reasons)
    response["degraded"] = is_degraded
    if reasons:
        existing = response.get("degraded_reasons") or []
        merged: List[str] = []
        for r in list(existing) + reasons:
            if r not in merged:
                merged.append(r)
        response["degraded_reasons"] = merged
    elif "degraded_reasons" not in response and is_degraded:
        response["degraded_reasons"] = []
    return response


def _cold_start_envelope(
    error: BaseException, *, trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """Structured envelope for the get_index() cold-start RuntimeError.

    GW-A-001: get_index() raises a RuntimeError when there is no baked index
    on disk AND Ollama is unreachable — it cannot build the index without
    embeddings. The handlers that call get_index() at the top of their body
    (compass / describe / compass_categories) must surface that as the same
    structured `service_unavailable` envelope that compass_status /
    compass_audit already degrade into, never as a raw stack to the caller.

    The code distinguishes the two halves of the precondition: if Ollama is
    known-down we report `ollama_unavailable`; otherwise the index itself is
    the blocker (`index_unhealthy`). Both are retryable — the operator can
    bring Ollama up or run a sync and retry the same call.
    """
    ollama_down = not _health_state.get("ollama_available", True)
    code = "ollama_unavailable" if ollama_down else "index_unhealthy"
    title = "Ollama unavailable" if ollama_down else "Index unhealthy"
    return _augment_with_health(
        _error_envelope(
            code=code,
            title=title,
            detail=f"Tool index unavailable on cold start: {error}",
            category="service_unavailable",
            retryable=True,
            trace_id=trace_id,
            suggestions=[
                "Start Ollama: ollama serve",
                "Run python gateway.py --sync to build the index",
            ],
        )
    )


# Process-wide counters for /metrics (BE-B-002).
# Kept here so all four signals live in one place. Locks unnecessary in MCP
# stdio mode (single async event loop); in HTTP mode the GIL + atomic int
# increments are sufficient for the precision Prometheus expects.
_metric_counters: Dict[str, Any] = {
    "lexical_fallback_total": 0,
    "degraded_responses_total": defaultdict(int),  # keyed by reason
    "circuit_breaker_transitions_total": defaultdict(int),  # keyed by from->to
    "fallback_invocations_total": defaultdict(int),  # keyed by type
}


def _record_lexical_fallback() -> None:
    _metric_counters["lexical_fallback_total"] = (
        _metric_counters["lexical_fallback_total"] + 1
    )
    _metric_counters["fallback_invocations_total"]["lexical"] += 1


def _record_degraded_response(reason: str) -> None:
    _metric_counters["degraded_responses_total"][reason] += 1


def _record_breaker_transition(from_state: str, to_state: str) -> None:
    key = f"{from_state}->{to_state}"
    _metric_counters["circuit_breaker_transitions_total"][key] += 1


# SC-001 sibling: per-event-loop singleton-init locks. A module-global
# asyncio.Lock binds its internal waiter Future to the first loop that awaits
# it under contention, then raises "bound to a different event loop" when the
# CLI (fresh asyncio.run per subcommand) or the Gradio UI (worker-thread
# asyncio.run) awaits the same lock from another loop. Key locks on the running
# loop id instead — mirrors embedder._get_global_embed_semaphore.
_loop_init_locks: Dict[int, Dict[str, "asyncio.Lock"]] = {}


def _loop_lock(name: str) -> "asyncio.Lock":
    """Return the named singleton-init lock for the running event loop."""
    loop = asyncio.get_running_loop()
    locks = _loop_init_locks.get(id(loop))
    if locks is None:
        locks = {}
        _loop_init_locks[id(loop)] = locks
    lock = locks.get(name)
    if lock is None:
        lock = asyncio.Lock()
        locks[name] = lock
    return lock


def get_config() -> CompassConfig:
    """Get or load configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


async def get_index() -> CompassIndex:
    """Get or initialize the compass index.

    Uses double-checked locking pattern with asyncio.Lock to prevent
    race conditions when multiple coroutines call this concurrently.

    GW-B-001: If Ollama is unreachable, we still return a usable CompassIndex
    provided a prior ``db/compass.hnsw`` + ``db/tools.db`` exist on disk. The
    index won't be able to service semantic ``search()`` calls, but callers
    (compass(), describe()) can fall back to a lexical LIKE query against
    ``index.db`` and keep serving users instead of raising opaquely.
    """
    global _compass_index

    # Fast path: already initialized
    if _compass_index is not None:
        return _compass_index

    # Slow path: acquire lock and check again
    async with _loop_lock("index"):
        # Double-check after acquiring lock (another coroutine may have initialized)
        if _compass_index is not None:
            return _compass_index

        # BE-B-014 + BE-B-008: pass tunables from CompassConfig so operators
        # can re-shape the index + breaker without code edits. BE-B-002:
        # on_breaker_transition fires the breaker_transitions_total counter.
        cfg = get_config()
        from embedder import Embedder
        embedder = Embedder(
            base_url=cfg.ollama_url,
            model=cfg.embedding_model,
            breaker_failure_threshold=cfg.ollama_breaker_failure_threshold,
            breaker_open_seconds=cfg.ollama_breaker_open_seconds,
            retry_attempts=cfg.ollama_retry_attempts,
            retry_backoffs=tuple(cfg.ollama_retry_backoffs),
            on_breaker_transition=_record_breaker_transition,
        )
        index = CompassIndex(
            embedder=embedder,
            hnsw_m=cfg.hnsw_m,
            hnsw_ef_construction=cfg.hnsw_ef_construction,
            hnsw_ef_search=cfg.hnsw_ef_search,
        )

        # Try to load existing index
        if index.load_index():
            _compass_index = index
            return _compass_index

        logger.warning("No existing index found. Building from manifest...")

        # Check Ollama — building needs embeddings, so this is non-negotiable here.
        try:
            ollama_ok = await index.embedder.health_check()
        except Exception as e:
            _mark_ollama_down(e)
            raise RuntimeError(
                "Ollama not available and no cached index found at "
                f"{index.index_path}. Start Ollama (ollama serve) and run: "
                "ollama pull nomic-embed-text"
            ) from e

        if not ollama_ok:
            _mark_ollama_down(RuntimeError("health_check returned False"))
            raise RuntimeError(
                "Ollama not available and no cached index found at "
                f"{index.index_path}. Start Ollama (ollama serve) and run: "
                "ollama pull nomic-embed-text"
            )

        _mark_ollama_up()

        # Build index from static manifest
        await index.build_index()

        _compass_index = index

    return _compass_index


async def get_backends() -> BackendManager:
    """Get or initialize the backend manager.

    Uses double-checked locking pattern with asyncio.Lock.
    """
    global _backend_manager

    # Fast path
    if _backend_manager is not None:
        return _backend_manager

    # Slow path with lock
    async with _loop_lock("backend"):
        if _backend_manager is not None:
            return _backend_manager

        _backend_manager = BackendManager(get_config())

    return _backend_manager


async def get_analytics_instance() -> Optional[CompassAnalytics]:
    """Get or initialize the analytics engine.

    Uses double-checked locking pattern with asyncio.Lock.
    Returns None if analytics is disabled in config.
    """
    global _analytics
    config = get_config()

    if not config.analytics_enabled:
        return None

    # Fast path
    if _analytics is not None:
        return _analytics

    # Slow path with lock
    async with _loop_lock("analytics"):
        if _analytics is not None:
            return _analytics

        _analytics = get_analytics()
        await _analytics.load_hot_cache_from_db()

    return _analytics


async def get_sync_manager_instance() -> Optional[SyncManager]:
    """Get or initialize the sync manager.

    Uses double-checked locking pattern with asyncio.Lock.
    Returns None if auto_sync is disabled in config.
    """
    global _sync_manager
    config = get_config()

    if not config.auto_sync:
        return None

    # Fast path
    if _sync_manager is not None:
        return _sync_manager

    # Slow path with lock
    async with _loop_lock("sync_manager"):
        if _sync_manager is not None:
            return _sync_manager

        # GW-A-001 sibling: get_index() raises RuntimeError on cold start (no
        # baked index + Ollama down). Returning None here ("sync features
        # unavailable until an index exists") keeps that from escaping callers
        # like compass()'s maybe_startup_sync() as a raw stack.
        try:
            index = await get_index()
        except RuntimeError as e:
            logger.warning(f"sync manager unavailable on cold start: {e}")
            return None
        backends = await get_backends()
        _sync_manager = get_sync_manager(config, index, backends)

    return _sync_manager


async def get_chain_indexer_instance() -> Optional[ChainIndexer]:
    """Get or initialize the chain indexer.

    Uses double-checked locking pattern with asyncio.Lock.
    Returns None if chain_indexing is disabled in config.
    """
    global _chain_indexer
    config = get_config()

    if not config.chain_indexing_enabled:
        return None

    # Fast path
    if _chain_indexer is not None:
        return _chain_indexer

    # Slow path with lock
    async with _loop_lock("chain_indexer"):
        if _chain_indexer is not None:
            return _chain_indexer

        # GW-A-001 sibling: same cold-start guard as get_sync_manager_instance.
        try:
            index = await get_index()
        except RuntimeError as e:
            logger.warning(f"chain indexer unavailable on cold start: {e}")
            return None
        analytics = await get_analytics_instance()
        chain_indexer = get_chain_indexer(index.embedder, analytics)

        # Load existing chain index or build it
        if not await chain_indexer.load_chain_index():
            # Seed default chains and build index
            await chain_indexer.seed_default_chains()
            await chain_indexer.build_chain_index()

        _chain_indexer = chain_indexer

    return _chain_indexer


async def maybe_startup_sync():
    """Run startup sync if enabled and not yet done.

    Uses double-checked locking pattern with asyncio.Lock to ensure
    sync only runs once even with concurrent requests.
    """
    global _startup_sync_done
    config = get_config()

    if not config.sync_check_on_startup:
        return

    # Fast path: already done
    if _startup_sync_done:
        return

    # Slow path with lock
    async with _loop_lock("startup_sync"):
        # Double-check after acquiring lock
        if _startup_sync_done:
            return

        # GW-A-001 sibling: do NOT latch _startup_sync_done before the work.
        # get_sync_manager_instance() returns None on cold start (index not
        # ready); leave the flag unset there so the next call retries once the
        # index/Ollama is available. Only latch once a sync actually ran (or
        # sync is disabled), so a cold-start deferral is not permanent.
        sync_manager = await get_sync_manager_instance()
        if sync_manager is None:
            if not get_config().auto_sync:
                _startup_sync_done = True  # sync disabled: nothing to do, ever
            # else cold start — leave unset to retry on a later call
            return
        try:
            await sync_manager.sync_if_needed()
        except Exception as e:
            logger.warning(f"Startup sync failed: {e}")
        _startup_sync_done = True


# =============================================================================
# GW-B-001: lexical fallback for when semantic search is unavailable.
# Simple LIKE query against the tools table — matches the same metadata fields
# the semantic index covers (name, description, category, server). No FTS5
# setup required.
# =============================================================================


# BE-B-007: cap user-supplied query length at the boundary so semantic and
# lexical paths don't waste cycles on accidental 10MB pastes. 512 chars is
# generous for natural-language intent.
_MAX_QUERY_LEN = 512


def _clamp_query(query: Optional[str]) -> str:
    """Strip + length-clamp a user-supplied query (BE-B-007)."""
    if not query:
        return ""
    q = query.strip()
    if len(q) > _MAX_QUERY_LEN:
        logger.warning(
            f"query length {len(q)} exceeds {_MAX_QUERY_LEN}; truncating"
        )
        q = q[:_MAX_QUERY_LEN]
    return q


def _escape_like(s: str) -> str:
    """Escape SQL LIKE wildcards in user input (BE-A-007).

    SQLite LIKE treats % and _ as wildcards. The query is bound via a
    parameter so there's no SQL injection risk, but a user query containing
    'foo%bar' would otherwise match 'foobar', 'fooXbar', etc. We escape
    with a backslash and pair the LIKE clause with ``ESCAPE '\\'``.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _lexical_search_fallback(
    index: CompassIndex,
    query: str,
    top_k: int,
    category: Optional[str],
    server: Optional[str],
) -> List[Dict[str, Any]]:
    """Return up to ``top_k`` tool matches via a case-insensitive LIKE scan.

    Each match is already in the response-envelope shape (``tool``,
    ``description``, ``server``, ``category``, ``confidence``, ``degraded``).
    Confidence is a coarse heuristic: 0.6 for a whole-query name substring,
    0.4 for a whole-query description substring, 0.3 for a row that only
    matched on an individual token — enough to produce a sensible ordering
    without pretending it's a real similarity score.

    GW-A-003: the query is tokenized on whitespace and each token contributes
    its own ``(name LIKE ? OR description LIKE ?)`` clause OR'd together, so a
    multi-word degraded-mode intent like "read file" still matches a tool
    named ``read_file`` whose name never contains the literal "read file"
    substring. A row that matched only via a token (not the whole query) takes
    the 0.3 confidence tier — previously unreachable dead code because the SQL
    used a single whole-query needle.

    BE-B-007: empty queries return [] immediately rather than '%%' matching
    the whole catalog. BE-A-007: % and _ in each token are escaped so they
    don't act as wildcards (the ESCAPE clause stays paired with every LIKE).
    """
    if not index or not getattr(index, "db", None):
        return []

    q = _clamp_query(query)
    if not q:
        return []

    # Per-token LIKE: split on whitespace, drop empties, and fall back to the
    # whole query when tokenization yields nothing (e.g. punctuation-only).
    tokens = [t for t in q.split() if t]
    if not tokens:
        tokens = [q]

    token_clauses: List[str] = []
    params: List[Any] = []
    for tok in tokens:
        needle = f"%{_escape_like(tok)}%"
        token_clauses.append(
            "lower(name) LIKE lower(?) ESCAPE '\\' "
            "OR lower(description) LIKE lower(?) ESCAPE '\\'"
        )
        params.extend([needle, needle])

    where = ["(" + " OR ".join(token_clauses) + ")"]
    if category:
        where.append("category = ?")
        params.append(category)
    if server:
        where.append("server = ?")
        params.append(server)

    sql = (
        "SELECT name, description, category, server "
        "FROM tools WHERE " + " AND ".join(where) + " LIMIT ?"
    )
    params.append(max(1, top_k * 3))

    rows = index.db.execute(sql, params).fetchall()
    q_lower = q.lower()

    scored: List[Dict[str, Any]] = []
    for row in rows:
        name_lower = (row["name"] or "").lower()
        desc_lower = (row["description"] or "").lower()
        if q_lower in name_lower:
            confidence = 0.6
        elif q_lower in desc_lower:
            confidence = 0.4
        else:
            # Matched via per-token LIKE but not a whole-query substring.
            confidence = 0.3
        scored.append({
            "tool": row["name"],
            "description": row["description"],
            "server": row["server"],
            "category": row["category"],
            "confidence": confidence,
            "degraded": True,
        })

    scored.sort(key=lambda m: m["confidence"], reverse=True)
    return scored[:top_k]


# =============================================================================
# MCP TOOLS - The Gateway Interface
# =============================================================================


@mcp.tool()
async def compass(
    intent: str,
    top_k: int = 5,
    category: Optional[str] = None,
    server: Optional[str] = None,
    min_confidence: float = 0.3,
    include_chains: bool = True,
) -> Dict[str, Any]:
    """
    Find tools by describing what you want to accomplish.

    This is your starting point for tool discovery. Describe your task in natural
    language, and compass will return the most relevant tools using semantic search.
    Also searches for tool chains (workflows) that match your intent.

    WORKFLOW:
    1. compass("your task") -> get tool names, summaries, and matching workflows
    2. describe("tool_name") -> get full schema for chosen tool
    3. execute("tool_name", {...}) -> run the tool

    Args:
        intent: Natural language description of what you want to do.
                Examples: "read a file", "generate an image", "search documents"
        top_k: Maximum number of tools to return (1-10, default 5)
        category: Filter by category (file, git, database, ai, search, analysis, etc.)
        server: Filter by server (bridge, doc, comfy, video, chat)
        min_confidence: Minimum similarity score (0-1, default 0.3)
        include_chains: Also search for matching tool chains/workflows (default True)

    Returns:
        Tool matches with names, descriptions, and confidence scores.
        Also includes matching chains (workflows) if found.
        Use describe() to get full schemas, execute() to run tools.
    """
    start_time = time.time()
    trace_id = uuid.uuid4().hex[:8]
    config = get_config()
    top_k = max(1, min(10, top_k))
    min_confidence = max(0.0, min(1.0, min_confidence))

    # BE-B-007: clamp the user-supplied intent at the boundary so a 10MB
    # paste doesn't become a 10MB Ollama call or a 10MB LIKE parameter.
    intent = _clamp_query(intent)

    warnings: List[str] = []
    degraded = False

    logger.info(
        f"[compass] [{trace_id}] intent={intent!r} top_k={top_k} "
        f"category={category} server={server} min_conf={min_confidence}"
    )

    # Check for sync on first call
    await maybe_startup_sync()

    # GW-A-001: get_index() raises RuntimeError on cold start (no baked index
    # AND Ollama unreachable). Surface that as the structured
    # service_unavailable envelope instead of letting it bubble as a raw
    # stack — mirrors compass_status / compass_audit.
    try:
        index = await get_index()
    except RuntimeError as e:
        logger.error(f"[compass] [{trace_id}] index unavailable on cold start: {e}")
        return _cold_start_envelope(e, trace_id=trace_id)

    # Search tools — on embedder/Ollama failure fall back to lexical LIKE
    # over the existing tools table so users keep getting results.
    results: List[SearchResult] = []
    fallback_matches: List[Dict[str, Any]] = []
    try:
        results = await index.search(
            query=intent, top_k=top_k, category_filter=category, server_filter=server
        )
        _mark_ollama_up()
        # Filter by confidence
        results = [r for r in results if r.score >= min_confidence]
    except Exception as e:
        _mark_ollama_down(e)
        degraded = True
        # BE-B-002: count the fallback invocation so /metrics can alert on
        # "served 100 lexical responses in 60s" before users notice.
        _record_lexical_fallback()
        logger.warning(
            f"[compass] [{trace_id}] semantic search failed ({type(e).__name__}: {e}); "
            "falling back to lexical search"
        )
        warnings.append(
            "Semantic search unavailable: Ollama is unreachable at "
            f"{config.ollama_url}. Try: ollama serve. "
            "Showing keyword-based results instead."
        )
        fallback_matches = _lexical_search_fallback(
            index, intent, top_k, category, server
        )
        # BE-A-004: honour the user-supplied min_confidence on the fallback
        # path. Lexical matches assign coarse heuristic confidences
        # (0.6/0.4/0.3); without this filter, a caller passing
        # min_confidence=0.9 would still receive 0.3-tier results when
        # Ollama is down, violating the documented contract.
        fallback_matches = [
            m for m in fallback_matches if m["confidence"] >= min_confidence
        ]

    # Search chains if enabled — chain search also relies on embeddings,
    # so a semantic outage will usually take this path down too. Don't let
    # that kill the whole response.
    chain_matches = []
    if include_chains and config.chain_indexing_enabled and not degraded:
        chain_indexer = await get_chain_indexer_instance()
        if chain_indexer:
            try:
                chain_results = await chain_indexer.search_chains(
                    intent, top_k=3, min_confidence=min_confidence
                )
                for cr in chain_results:
                    chain_matches.append({
                        "name": cr.chain.name,
                        "tools": cr.chain.tools,
                        "description": cr.chain.description,
                        "confidence": float(round(cr.score, 3)),
                        "use_count": cr.chain.use_count,
                    })
            except Exception as e:
                logger.warning(
                    f"[compass] [{trace_id}] chain search failed "
                    f"({type(e).__name__}: {e}); skipping chain matches"
                )
                warnings.append(
                    "Chain search skipped: embedding service unavailable."
                )

    # Build response - progressive disclosure means we only return summaries
    matches: List[Dict[str, Any]] = []
    if fallback_matches:
        # Lexical fallback path — fallback_matches is already shaped correctly.
        matches = fallback_matches
    else:
        for r in results:
            match_data = {
                "tool": r.tool.name,
                "description": r.tool.description,
                "server": r.tool.server,
                "category": r.tool.category,
                "confidence": float(round(r.score, 3)),
            }

            # Only include full schema if progressive disclosure is disabled
            if not config.progressive_disclosure:
                match_data["parameters"] = r.tool.parameters
                match_data["examples"] = r.tool.examples

            matches.append(match_data)

    # Stats
    stats = index.get_stats()
    total_tools = stats.get("total_tools", 0)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Record analytics
    analytics = await get_analytics_instance()
    if analytics:
        await analytics.record_search(intent, results, latency_ms, category, server)

    # Hint for next steps
    if not matches and not chain_matches:
        hint = f"No tools found for '{intent}'. Try broader terms or use compass_categories() to see available categories."
    elif chain_matches and chain_matches[0]["confidence"] > (
        matches[0]["confidence"] if matches else 0
    ):
        # Chain is the best match
        chain_name = chain_matches[0]["name"]
        hint = f"Found workflow '{chain_name}' ({chain_matches[0]['confidence']:.0%}). Tools: {' → '.join(chain_matches[0]['tools'])}"
    elif not matches:
        # GW-COMPOSED-001: matches is empty but chains exist and none of them
        # scored above the (0) tool threshold — e.g. a single chain matched at
        # 0.000 with min_confidence=0.0. The prior chain of elifs fell through
        # to the final `else`, which indexed matches[0] and raised IndexError
        # (violating BE-B-004: handlers return a structured payload, never
        # raise). Produce a chains-only hint instead of touching matches[0].
        chain_name = chain_matches[0]["name"]
        hint = f"Found workflow '{chain_name}' ({chain_matches[0]['confidence']:.0%}). Tools: {' → '.join(chain_matches[0]['tools'])}"
    elif len(matches) == 1:
        tool_name = matches[0]["tool"]
        if config.progressive_disclosure:
            hint = f"Found: {tool_name}. Use describe('{tool_name}') for full schema, then execute() to run."
        else:
            hint = f"Found: {tool_name}. Use execute('{tool_name}', {{...}}) to run."
    else:
        top_name = matches[0]["tool"]
        if config.progressive_disclosure:
            hint = f"Found {len(matches)} tools. Top: {top_name} ({matches[0]['confidence']:.0%}). Use describe() for schemas."
        else:
            hint = f"Found {len(matches)} tools. Top: {top_name} ({matches[0]['confidence']:.0%}). Use execute() to run."

    response = {
        "trace_id": trace_id,
        "matches": matches,
        "total_indexed": total_tools,
        "tokens_saved": max(0, (total_tools - len(matches)) * 500),
        "hint": hint,
        "workflow": "compass() -> describe() -> execute()"
        if config.progressive_disclosure
        else "compass() -> execute()",
        "degraded": degraded,
    }

    # Include chains if any found
    if chain_matches:
        response["chains"] = chain_matches

    if warnings:
        response["warnings"] = warnings

    # BE-B-003: standardize degraded + degraded_reasons on every response.
    response = _augment_with_health(response)
    # BE-B-002: record degraded-response counter for /metrics.
    if response.get("degraded"):
        for reason in response.get("degraded_reasons", []) or ["unknown"]:
            _record_degraded_response(reason)
    return response


@mcp.tool()
async def describe(tool_name: str) -> Dict[str, Any]:
    """
    Get the full schema for a specific tool.

    Use this after compass() to get complete parameter information before calling execute().
    This progressive disclosure pattern saves tokens by only loading schemas when needed.

    Args:
        tool_name: The tool name from compass results (e.g., "bridge:read_file")

    Returns:
        Full tool schema including all parameters, types, and descriptions.
    """
    trace_id = uuid.uuid4().hex[:8]
    logger.info(f"[describe] [{trace_id}] tool_name={tool_name!r}")

    # GW-A-001: surface the cold-start RuntimeError as a structured envelope
    # rather than a raw stack (mirrors compass_status / compass_audit).
    try:
        index = await get_index()
    except RuntimeError as e:
        logger.error(f"[describe] [{trace_id}] index unavailable on cold start: {e}")
        return _cold_start_envelope(e, trace_id=trace_id)

    # Try to find in index first (from manifest).
    # GW-B-009: trap sqlite errors so the user sees "index unhealthy" + a
    # fallthrough to the backend lookup path instead of an opaque stack.
    if index.db:
        try:
            cursor = index.db.execute(
                "SELECT name, description, category, server, parameters, examples FROM tools WHERE name = ?",
                (tool_name,),
            )
            row = cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(
                f"[describe] [{trace_id}] index DB error: {type(e).__name__}: {e}"
            )
            _health_state["index_available"] = False
            _health_state["last_index_error"] = f"{type(e).__name__}: {e}"
            row = None
        else:
            _health_state["index_available"] = True

        if row:
            # GW-A-002: a malformed JSON blob in the index row must NOT raise
            # an uncaught JSONDecodeError. Degrade the same way a sqlite error
            # does — flag the index unhealthy and fall back to {}/[] so the
            # caller still gets a usable (if partial) schema.
            try:
                params = json.loads(row["parameters"]) if row["parameters"] else {}
                examples = json.loads(row["examples"]) if row["examples"] else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(
                    f"[describe] [{trace_id}] malformed index JSON for "
                    f"{tool_name!r}: {type(e).__name__}: {e}"
                )
                _health_state["index_available"] = False
                _health_state["last_index_error"] = f"{type(e).__name__}: {e}"
                params = {}
                examples = []

            response = {
                "trace_id": trace_id,
                "tool": row["name"],
                "description": row["description"],
                "server": row["server"],
                "category": row["category"],
                "parameters": params,
                "examples": examples,
                "hint": f"Use execute('{tool_name}', {{...}}) to run this tool.",
            }
            return _augment_with_health(response)

    # Try backends if connected
    manager = await get_backends()
    schema = manager.get_tool_schema(tool_name)
    if schema:
        response = {
            "trace_id": trace_id,
            **schema,
            "hint": f"Use execute('{tool_name}', {{...}}) to run this tool.",
        }
        if not _health_state.get("index_available", True):
            response["warnings"] = [
                "Index database unhealthy — served schema from backend directly. "
                "Try compass_sync(force=True) to rebuild the index."
            ]
        return _augment_with_health(response)

    # BE-B-001: emit a structured error envelope with nearest_tools so the
    # caller (especially an LLM agent) can recover from a typo without a
    # second round-trip. nearest_tools[] is the single most actionable
    # signal — "tool_not_found: 'brige:redafile'" gets the agent nowhere,
    # but "nearest_tools=[{'tool': 'bridge:read_file', score: 0.6}]" lets
    # it switch strategies deterministically.
    nearest = _lexical_search_fallback(index, tool_name, top_k=3, category=None, server=None)
    nearest_envelope: Optional[List[Dict[str, Any]]] = None
    if nearest:
        nearest_envelope = [
            {
                "tool": m["tool"],
                "score": m["confidence"],
                "server": m.get("server"),
                "category": m.get("category"),
            }
            for m in nearest
        ]

    response = _error_envelope(
        code="tool_not_found",
        title="Tool not found",
        detail=f"Tool not found: {tool_name}",
        category="not_found",
        retryable=False,
        trace_id=trace_id,
        nearest_tools=nearest_envelope,
        suggestions=[
            "Use compass() to search for available tools.",
            "If you have a typo, try the nearest_tools[] suggestions.",
        ],
    )
    if not _health_state.get("index_available", True):
        response["warnings"] = [
            "Index database unhealthy — lookup may be incomplete. "
            "Try compass_sync(force=True) to rebuild the index."
        ]
    response["hint"] = "Use compass() to search for available tools."
    return _augment_with_health(response)


@mcp.tool()
async def execute(
    tool_name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a tool on its backend server.

    This proxies the call to the appropriate MCP backend server.
    Use compass() to find tools and describe() to get parameter schemas.

    Args:
        tool_name: The tool to execute (e.g., "bridge:read_file" or "comfy:comfy_generate")
        arguments: Tool arguments as a dictionary. Check describe() for required parameters.

    Returns:
        The tool's response or an error message.
    """
    start_time = time.time()
    trace_id = uuid.uuid4().hex[:8]
    logger.info(f"[execute] [{trace_id}] tool_name={tool_name!r}")

    if arguments is None:
        arguments = {}

    manager = await get_backends()
    analytics = await get_analytics_instance()

    # Check hot cache for faster schema lookup (optional optimization)
    if analytics:
        hot_tool = analytics.get_hot_tool(tool_name)
        if hot_tool:
            logger.debug(f"Using hot cache for {tool_name}")

    # Connect to backend if needed
    if ":" in tool_name:
        server_name = tool_name.split(":")[0]
        if not manager.is_backend_connected(server_name):
            logger.info(f"Connecting to backend: {server_name}")
            success = await manager.connect_backend(server_name)
            if not success:
                # Record failed call
                latency_ms = (time.time() - start_time) * 1000
                if analytics:
                    await analytics.record_tool_call(
                        tool_name,
                        success=False,
                        latency_ms=latency_ms,
                        error_message=f"Failed to connect to backend: {server_name}",
                    )
                logger.warning(
                    f"[execute] [{trace_id}] backend connect failed: {server_name}"
                )
                envelope = _error_envelope(
                    code="backend_connect_failed",
                    title="Backend connect failed",
                    detail=f"Failed to connect to backend: {server_name}",
                    category="service_unavailable",
                    retryable=True,
                    retry_after_seconds=5.0,
                    trace_id=trace_id,
                    suggestions=[
                        "Check that the backend server is configured correctly.",
                        "Inspect compass_status() for backend health.",
                    ],
                )
                envelope["success"] = False
                envelope["hint"] = (
                    "Check that the backend server is configured correctly."
                )
                return _augment_with_health(envelope)

    # BE-B-004: wrap manager.execute_tool() so an unhandled raise from the
    # backend client doesn't propagate a Python traceback through the MCP
    # JSON-RPC envelope. The MCP spec requires tool handlers to return a
    # structured payload, never raise unhandled.
    try:
        result = await manager.execute_tool(tool_name, arguments)
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error_text = f"{type(e).__name__}: {e}"
        logger.error(
            f"[execute] [{trace_id}] tool={tool_name!r} unhandled exception: {error_text}"
        )
        if analytics:
            try:
                await analytics.record_tool_call(
                    tool_name,
                    success=False,
                    latency_ms=latency_ms,
                    error_message=error_text,
                )
            except Exception as rec_err:
                logger.debug(f"analytics record failed: {rec_err}")
        envelope = _error_envelope(
            code="execute_unhandled_exception",
            title="Tool execution failed (unhandled)",
            detail=error_text,
            category="backend_error",
            retryable=False,
            trace_id=trace_id,
            suggestions=[
                "Inspect server logs (search for the trace_id).",
                "Verify the backend is healthy via compass_status().",
            ],
        )
        envelope["success"] = False
        return _augment_with_health(envelope)

    # Record analytics
    latency_ms = (time.time() - start_time) * 1000
    if isinstance(result, dict):
        if "success" in result:
            success = result["success"]
        else:
            # Missing 'success' key: treat as failure to avoid masking backend errors
            logger.warning(
                f"Backend result for {tool_name} lacks 'success' key; defaulting to False"
            )
            success = False
    else:
        success = False
    error_msg = (
        result.get("error") if isinstance(result, dict) and not success else None
    )

    if analytics:
        await analytics.record_tool_call(
            tool_name,
            success=success,
            latency_ms=latency_ms,
            error_message=error_msg,
            arguments=arguments,
        )

    # GW-B-003: stamp trace_id into both success and failure envelopes so the
    # user can paste it into a bug report.
    if isinstance(result, dict):
        result.setdefault("trace_id", trace_id)
        if not success:
            logger.warning(
                f"[execute] [{trace_id}] tool={tool_name!r} failed: {error_msg}"
            )
        else:
            logger.info(
                f"[execute] [{trace_id}] tool={tool_name!r} ok in {latency_ms:.1f}ms"
            )
        return _augment_with_health(result)
    return result


@mcp.tool()
async def compass_categories() -> Dict[str, Any]:
    """
    List available tool categories and servers.

    Use this to understand what kinds of tools are available before searching.
    """
    trace_id = uuid.uuid4().hex[:8]
    # GW-A-001: surface the cold-start RuntimeError as a structured envelope
    # rather than a raw stack (mirrors compass_status / compass_audit).
    try:
        index = await get_index()
    except RuntimeError as e:
        logger.error(
            f"[compass_categories] [{trace_id}] index unavailable on cold start: {e}"
        )
        return _cold_start_envelope(e, trace_id=trace_id)
    stats = index.get_stats()

    response = {
        "categories": stats.get("by_category", {}),
        "servers": stats.get("by_server", {}),
        "total_tools": stats.get("total_tools", 0),
        "hint": "Use compass(intent, category='file') to filter searches.",
    }
    return _augment_with_health(response)


@mcp.tool()
async def compass_status() -> Dict[str, Any]:
    """
    Get Tool Compass gateway status and health information.

    Returns index stats, backend connection status, configuration, analytics summary,
    hot cache status, and sync status.
    """
    config = get_config()
    trace_id = uuid.uuid4().hex[:8]
    logger.info(f"[compass_status] [{trace_id}]")

    # BE-B-004: each subsystem block is wrapped independently so a single
    # failure degrades that section rather than aborting the whole status.
    response: Dict[str, Any] = {"trace_id": trace_id}

    index_stats: Dict[str, Any] = {}
    try:
        index = await get_index()
        index_stats = index.get_stats()
        response["index"] = {
            "total_tools": index_stats.get("total_tools", 0),
            "by_category": index_stats.get("by_category", {}),
            "by_server": index_stats.get("by_server", {}),
        }
    except Exception as e:
        logger.error(f"[compass_status] [{trace_id}] index stats failed: {e}")
        response["index"] = {"error": f"{type(e).__name__}: {e}", "trace_id": trace_id}

    try:
        manager = await get_backends()
        response["backends"] = manager.get_stats()
    except Exception as e:
        logger.error(f"[compass_status] [{trace_id}] backend stats failed: {e}")
        response["backends"] = {"error": f"{type(e).__name__}: {e}", "trace_id": trace_id}

    response["config"] = {
        "progressive_disclosure": config.progressive_disclosure,
        "auto_sync": config.auto_sync,
        "embedding_model": config.embedding_model,
        "analytics_enabled": config.analytics_enabled,
        "chain_indexing_enabled": config.chain_indexing_enabled,
    }
    # GW-B-001 / GW-B-009: expose degraded-mode flags so users / operators
    # can tell at a glance when compass() is serving lexical fallbacks or
    # when the index DB has gone sour.
    response["health"] = {
        "ollama_available": _health_state["ollama_available"],
        "last_ollama_error": _health_state["last_ollama_error"],
        "index_available": _health_state["index_available"],
        "last_index_error": _health_state["last_index_error"],
        "degraded_mode": (
            not _health_state["ollama_available"]
            or not _health_state["index_available"]
        ),
    }

    # Add analytics info if enabled
    if config.analytics_enabled:
        try:
            analytics = await get_analytics_instance()
            if analytics:
                response["hot_cache"] = {
                    "size": len(analytics._hot_cache),
                    "tools": list(analytics._hot_cache.keys()),
                }
                # PC-B-003: surface the analytics _degraded flag on the primary
                # runtime status surface. get_health() already computes it (so the
                # UI/status tab can distinguish "analytics fine" from "sqlite
                # broke; tool calls still work but metrics aren't recorded"), but
                # until now it was only readable via config.doctor(), a separate
                # out-of-process command. Fold it into the health block so the
                # degraded flag the code already tracks is visible here too.
                health = analytics.get_health()
                response["health"]["analytics_degraded"] = health.get("degraded", False)
                response["health"]["analytics_degraded_reason"] = health.get("reason")
        except Exception as e:
            # Status must never itself raise if analytics is unavailable
            # (BE-B-004): degrade this section, and surface the analytics-health
            # subsystem as unknown rather than asserting it is healthy.
            logger.error(f"[compass_status] [{trace_id}] analytics failed: {e}")
            response["hot_cache"] = {"error": f"{type(e).__name__}: {e}"}
            response["health"]["analytics_degraded"] = None
            response["health"]["analytics_degraded_reason"] = f"{type(e).__name__}: {e}"

    # Add sync status if enabled
    if config.auto_sync:
        try:
            sync_manager = await get_sync_manager_instance()
            if sync_manager:
                response["sync"] = await sync_manager.get_sync_status()
        except Exception as e:
            logger.error(f"[compass_status] [{trace_id}] sync status failed: {e}")
            response["sync"] = {"error": f"{type(e).__name__}: {e}"}

    # Add chain info if enabled
    if config.chain_indexing_enabled:
        try:
            chain_indexer = await get_chain_indexer_instance()
            if chain_indexer:
                chains = await chain_indexer.load_chains_from_db()
                response["chains"] = {
                    "total": len(chains),
                    "cached": len(chain_indexer._chain_cache),
                }
        except Exception as e:
            logger.error(f"[compass_status] [{trace_id}] chain info failed: {e}")
            response["chains"] = {"error": f"{type(e).__name__}: {e}"}

    return _augment_with_health(response)


@mcp.tool()
async def compass_analytics(
    timeframe: str = "24h", include_failures: bool = True
) -> Dict[str, Any]:
    """
    Get detailed usage analytics and tool health metrics.

    Tracks search patterns, tool usage, success/failure rates, and latencies.
    Use this to understand how tools are being used and identify issues.

    Args:
        timeframe: Time window for stats ("1h", "24h", "7d", "30d")
        include_failures: Include details about failed tool calls

    Returns:
        Comprehensive analytics including top tools, failure rates, chains, etc.
    """
    config = get_config()
    trace_id = uuid.uuid4().hex[:8]

    if not config.analytics_enabled:
        return _augment_with_health(_error_envelope(
            code="analytics_disabled",
            title="Analytics is disabled",
            detail="Analytics is disabled.",
            category="configuration",
            retryable=False,
            trace_id=trace_id,
            suggestions=["Enable analytics_enabled in config to track usage."],
        ))

    analytics = await get_analytics_instance()
    if not analytics:
        return _augment_with_health(_error_envelope(
            code="analytics_unavailable",
            title="Analytics not initialized",
            detail="Analytics not initialized.",
            category="service_unavailable",
            retryable=True,
            trace_id=trace_id,
        ))

    try:
        summary = await analytics.get_analytics_summary(timeframe)
    except Exception as e:
        logger.error(f"[compass_analytics] [{trace_id}] failed: {e}")
        return _augment_with_health(_error_envelope(
            code="analytics_unavailable",
            title="Analytics query failed",
            detail=f"{type(e).__name__}: {e}",
            category="backend_error",
            retryable=True,
            trace_id=trace_id,
        ))

    if not include_failures:
        summary.pop("failures", None)

    return _augment_with_health(summary)


@mcp.tool()
async def compass_chains(
    action: str = "list",
    chain_name: Optional[str] = None,
    tools: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List and manage tool chains (workflows).

    Tool chains are sequences of tools that commonly go together.
    They're auto-detected from usage patterns or can be manually defined.

    Args:
        action: "list" to see all chains, "create" to add a new chain, "detect" to find patterns
        chain_name: Name for new chain (required for "create")
        tools: List of tool names for new chain (required for "create")
        description: Description for new chain (optional for "create")

    Returns:
        Chain information based on action
    """
    config = get_config()
    trace_id = uuid.uuid4().hex[:8]

    if not config.chain_indexing_enabled:
        return _augment_with_health(_error_envelope(
            code="chain_indexing_disabled",
            title="Chain indexing is disabled",
            detail="Chain indexing is disabled.",
            category="configuration",
            retryable=False,
            trace_id=trace_id,
            suggestions=["Enable chain_indexing_enabled in config."],
        ))

    chain_indexer = await get_chain_indexer_instance()
    if not chain_indexer:
        return _augment_with_health(_error_envelope(
            code="chain_indexer_unavailable",
            title="Chain indexer not initialized",
            detail="Chain indexer not initialized.",
            category="service_unavailable",
            retryable=True,
            trace_id=trace_id,
        ))

    if action == "list":
        chains = await chain_indexer.load_chains_from_db()
        return _augment_with_health({
            "chains": [
                {
                    "name": c.name,
                    "tools": c.tools,
                    "description": c.description,
                    "use_count": c.use_count,
                    "is_auto_detected": c.is_auto_detected,
                }
                for c in chains
            ],
            "total": len(chains),
            "cached": len(chain_indexer._chain_cache),
        })

    elif action == "create":
        if not chain_name or not tools:
            return _augment_with_health(_error_envelope(
                code="invalid_argument",
                title="Missing required arguments",
                detail="chain_name and tools are required for create.",
                category="validation",
                retryable=False,
                trace_id=trace_id,
                suggestions=[
                    "compass_chains(action='create', chain_name='my_workflow', "
                    "tools=['tool1', 'tool2'])",
                ],
            ))

        # GW-COMPOSED-002: add_chain() calls embedder.embed(), which raises
        # when Ollama is unreachable (breaker open -> RuntimeError; connect/
        # timeout -> httpx.TransportError/TimeoutException). Mirror the rest of
        # the surface (compass/execute) and return a structured
        # service_unavailable envelope instead of leaking a raw exception.
        try:
            chain = await chain_indexer.add_chain(
                name=chain_name,
                tools=tools,
                description=description,
                is_auto_detected=False,
            )
        except Exception as e:
            _mark_ollama_down(e)
            logger.warning(
                f"[compass_chains] [{trace_id}] create failed "
                f"({type(e).__name__}: {e}); embedding service unavailable"
            )
            return _augment_with_health(_error_envelope(
                code="ollama_unavailable",
                title="Embedding service unavailable",
                detail=f"Could not embed chain '{chain_name}': {e}",
                category="service_unavailable",
                retryable=True,
                trace_id=trace_id,
                suggestions=["Start Ollama: ollama serve", "Retry once Ollama is reachable."],
            ))

        return _augment_with_health({
            "created": {
                "name": chain.name,
                "tools": chain.tools,
                "description": chain.description,
            },
            "hint": f"Chain '{chain_name}' created. It will now appear in compass() search results.",
        })

    elif action == "detect":
        analytics = await get_analytics_instance()
        if analytics:
            # GW-COMPOSED-002 + IDX-COMPOSED-003: detect_chains() raw-INSERTs
            # promoted chains into tool_chains but does NOT embed them into the
            # chain HNSW index, so they were unsearchable until process restart
            # even though the hint claimed "indexed and searchable". Add each
            # detected chain into the live index via chain_indexer.add_chain()
            # (embeds + dual-writes DB+HNSW). Both the detect call and the
            # per-chain embedding can fail when Ollama is down, so guard the
            # whole block and degrade into the same envelope as create.
            try:
                detected = await analytics.detect_chains()
                indexed = 0
                for c in detected:
                    await chain_indexer.add_chain(
                        name=c["name"],
                        tools=c["tools"],
                        description=c.get("description"),
                        is_auto_detected=True,
                    )
                    indexed += 1
            except Exception as e:
                _mark_ollama_down(e)
                logger.warning(
                    f"[compass_chains] [{trace_id}] detect failed "
                    f"({type(e).__name__}: {e}); embedding service unavailable"
                )
                return _augment_with_health(_error_envelope(
                    code="ollama_unavailable",
                    title="Embedding service unavailable",
                    detail=f"Chain detection/indexing failed: {e}",
                    category="service_unavailable",
                    retryable=True,
                    trace_id=trace_id,
                    suggestions=["Start Ollama: ollama serve", "Retry once Ollama is reachable."],
                ))

            return _augment_with_health({
                "detected": detected,
                "count": len(detected),
                "indexed": indexed,
                "hint": (
                    f"Detected {len(detected)} chain(s); {indexed} now indexed "
                    "and searchable via compass()."
                    if detected
                    else "No new chains detected."
                ),
            })
        return _augment_with_health(_error_envelope(
            code="analytics_unavailable",
            title="Analytics required for chain detection",
            detail="Analytics required for chain detection.",
            category="service_unavailable",
            retryable=True,
            trace_id=trace_id,
        ))

    else:
        return _augment_with_health(_error_envelope(
            code="invalid_action",
            title="Unknown action",
            detail=f"Unknown action: {action}",
            category="validation",
            retryable=False,
            trace_id=trace_id,
            valid_actions=["list", "create", "detect"],
        ))


@mcp.tool()
async def compass_sync(force: bool = False) -> Dict[str, Any]:
    """
    Check for backend changes and sync the index.

    Normally, sync happens automatically on startup. Use this to manually
    trigger a sync check or force a full rebuild.

    Args:
        force: If True, force a full sync regardless of detected changes

    Returns:
        Sync status for each backend
    """
    config = get_config()
    trace_id = uuid.uuid4().hex[:8]

    if not config.auto_sync:
        return _augment_with_health(_error_envelope(
            code="sync_disabled",
            title="Auto-sync is disabled",
            detail="Auto-sync is disabled.",
            category="configuration",
            retryable=False,
            trace_id=trace_id,
            suggestions=["Enable auto_sync in config for automatic synchronization."],
        ))

    sync_manager = await get_sync_manager_instance()
    if not sync_manager:
        return _augment_with_health(_error_envelope(
            code="sync_manager_unavailable",
            title="Sync manager not initialized",
            detail="Sync manager not initialized.",
            category="service_unavailable",
            retryable=True,
            trace_id=trace_id,
        ))

    if force:
        result = await sync_manager.full_sync()
        return _augment_with_health({"action": "full_sync", "result": result})
    else:
        results = await sync_manager.sync_if_needed()
        return _augment_with_health({
            "action": "sync_if_needed",
            "backends": results,
            "hint": "Use force=True to rebuild the entire index",
        })


@mcp.tool()
async def compass_audit(
    include_tools: bool = False, timeframe: str = "24h"
) -> Dict[str, Any]:
    """
    Comprehensive audit of the Tool Compass system.

    Returns a complete overview including:
    - Index health and tool counts by category/server
    - Backend connection status
    - Hot cache status (top 10 most-used tools)
    - Tool chain definitions
    - Usage analytics summary
    - Configuration status

    Args:
        include_tools: If True, include full list of all indexed tools
        timeframe: Timeframe for analytics ("1h", "24h", "7d", "30d")

    Returns:
        Complete system audit with all subsystems
    """
    config = get_config()
    trace_id = uuid.uuid4().hex[:8]

    # BE-B-004: wrap each subsystem block independently so one failure
    # degrades that section to {error, trace_id} rather than aborting the
    # whole audit.
    audit: Dict[str, Any] = {"trace_id": trace_id}
    index: Optional[CompassIndex] = None
    index_stats: Dict[str, Any] = {}

    try:
        index = await get_index()
        index_stats = index.get_stats()
        audit["system"] = {
            "version": __version__,
            "total_tools": index_stats.get("total_tools", 0),
            "index_path": str(index.index_path),
            "db_path": str(index.db_path),
        }
        audit["categories"] = index_stats.get("by_category", {})
        audit["servers"] = index_stats.get("by_server", {})
    except Exception as e:
        logger.error(f"[compass_audit] [{trace_id}] index/system failed: {e}")
        audit["system"] = {"error": f"{type(e).__name__}: {e}"}
        audit["categories"] = {}
        audit["servers"] = {}

    try:
        manager = await get_backends()
        audit["backends"] = manager.get_stats()
    except Exception as e:
        logger.error(f"[compass_audit] [{trace_id}] backend stats failed: {e}")
        audit["backends"] = {"error": f"{type(e).__name__}: {e}"}

    audit["config"] = {
        "progressive_disclosure": config.progressive_disclosure,
        "auto_sync": config.auto_sync,
        "analytics_enabled": config.analytics_enabled,
        "chain_indexing_enabled": config.chain_indexing_enabled,
        "hot_cache_size": config.hot_cache_size,
        "embedding_model": config.embedding_model,
    }

    # Hot cache
    analytics = None
    if config.analytics_enabled:
        try:
            analytics = await get_analytics_instance()
            if analytics:
                hot_tools = list(analytics._hot_cache.keys())
                audit["hot_cache"] = {
                    "size": len(hot_tools),
                    "tools": hot_tools,
                    "status": "active" if hot_tools else "empty (populates with usage)",
                }

                # Analytics summary
                summary = await analytics.get_analytics_summary(timeframe)
                audit["analytics"] = {
                    "timeframe": timeframe,
                    "total_searches": summary["searches"]["total"],
                    "avg_search_latency_ms": summary["searches"]["avg_latency_ms"],
                    "total_tool_calls": summary["tool_calls"]["total"],
                    "success_rate": summary["tool_calls"]["success_rate"],
                    "top_tools": [
                        t["tool"] for t in summary["tool_calls"]["top_tools"][:5]
                    ],
                    "top_queries": [
                        q["query"] for q in summary["searches"]["top_queries"][:5]
                    ],
                }
        except Exception as e:
            logger.error(f"[compass_audit] [{trace_id}] analytics failed: {e}")
            audit["analytics"] = {"error": f"{type(e).__name__}: {e}"}

    # Chains
    if config.chain_indexing_enabled:
        try:
            chain_indexer = await get_chain_indexer_instance()
            if chain_indexer:
                chains = await chain_indexer.load_chains_from_db()
                audit["chains"] = {
                    "total": len(chains),
                    "cached": len(chain_indexer._chain_cache),
                    "workflows": [
                        {
                            "name": c.name,
                            "tools": [t.split(":")[-1] for t in c.tools],
                            "use_count": c.use_count,
                            "auto_detected": c.is_auto_detected,
                        }
                        for c in chains
                    ],
                }
        except Exception as e:
            logger.error(f"[compass_audit] [{trace_id}] chains failed: {e}")
            audit["chains"] = {"error": f"{type(e).__name__}: {e}"}

    # Sync status
    if config.auto_sync:
        try:
            sync_manager = await get_sync_manager_instance()
            if sync_manager:
                sync_status = await sync_manager.get_sync_status()
                audit["sync"] = sync_status
        except Exception as e:
            logger.error(f"[compass_audit] [{trace_id}] sync failed: {e}")
            audit["sync"] = {"error": f"{type(e).__name__}: {e}"}

    # Optionally include all tools
    if include_tools:
        try:
            if index is not None and index.db:
                cursor = index.db.execute(
                    "SELECT name, description, category, server FROM tools ORDER BY server, category, name"
                )
                audit["tools"] = [
                    {
                        "name": row["name"],
                        "description": row["description"][:80] + "..."
                        if len(row["description"]) > 80
                        else row["description"],
                        "category": row["category"],
                        "server": row["server"],
                    }
                    for row in cursor.fetchall()
                ]
            else:
                audit["tools"] = []
                audit["tools_note"] = "Index database not available; tool list empty."
        except Exception as e:
            logger.error(f"[compass_audit] [{trace_id}] tools list failed: {e}")
            audit["tools"] = []
            audit["tools_note"] = f"{type(e).__name__}: {e}"

    # Health check
    issues: List[str] = []
    if index_stats.get("total_tools", 0) == 0:
        issues.append("No tools indexed - run compass_sync(force=True)")
    if config.analytics_enabled and analytics and not analytics._hot_cache:
        issues.append("Hot cache empty - will populate as tools are used")
    if not config.chain_indexing_enabled:
        issues.append("Chain indexing disabled - enable for workflow detection")

    audit["health"] = {
        "status": "healthy" if not issues else "needs_attention",
        "issues": issues if issues else ["All systems operational"],
    }

    return _augment_with_health(audit)


# =============================================================================
# CLI COMMANDS
# =============================================================================


async def sync_from_backends() -> bool:
    """Sync tool definitions from live backend servers and rebuild index.

    Returns True when the index was rebuilt, False when the sync indexed
    nothing (no backends connected, no tools discovered, or Ollama
    unavailable). GW-SB-001: the caller turns a False into a non-zero exit
    so CI/deploy wrappers can't mistake an empty sync for success.
    """

    print("\n" + "=" * 60)
    print("  TOOL COMPASS - INDEX SYNC")
    print("=" * 60)

    # Step 1: Load config
    print("\n[1/4] Loading configuration...", end=" ", flush=True)
    config = load_config()
    print(f"OK ({len(config.backends)} backends configured)")

    # Step 2: Connect to backends
    print("\n[2/4] Connecting to backends...")
    manager = BackendManager(config)

    results = await manager.connect_all()

    connected = 0
    for name, success in results.items():
        if success:
            print(f"      ✓ {name}")
            connected += 1
        else:
            print(f"      ✗ {name} (FAILED)")

    if connected == 0:
        print("\n❌ No backends connected. Check that servers are running.")
        print("   Hint: Start MCP servers or check compass_config.json")
        return False

    # Step 3: Discover tools
    print("\n[3/4] Discovering tools...")
    tools = manager.get_all_tools()
    print(f"      Found {len(tools)} tools from {connected} backend(s)")

    if not tools:
        print("\n❌ No tools discovered. Backends may be misconfigured.")
        await manager.disconnect_all()
        return False

    # Convert to ToolDefinition format for indexing
    print("      Converting to index format...", end=" ", flush=True)
    tool_defs = []
    for tool in tools:
        # Parse server and name from qualified name
        if ":" in tool.qualified_name:
            server, name = tool.qualified_name.split(":", 1)
        else:
            server = tool.server
            name = tool.name

        # Extract parameter names from schema
        params = {}
        if tool.input_schema and "properties" in tool.input_schema:
            for param_name, param_info in tool.input_schema["properties"].items():
                param_type = param_info.get("type", "any")
                if isinstance(param_type, list):
                    param_type = "/".join(param_type)
                params[param_name] = param_type

        tool_defs.append(
            ToolDefinition(
                name=tool.qualified_name,
                description=tool.description,
                category=categorize_tool(tool.name, tool.description),
                server=server,
                parameters=params,
                examples=[],
                is_core=False,
            )
        )

    print(f"OK ({len(tool_defs)} definitions)")

    # Step 4: Build index
    print("\n[4/4] Building HNSW search index...")
    index = CompassIndex()

    # Check Ollama first
    print("      Checking Ollama embeddings service...", end=" ", flush=True)
    if not await index.embedder.health_check():
        print("FAILED")
        print("\n❌ Ollama not available. Please start Ollama and pull the embedding model:")
        print("   1. ollama serve")
        print("   2. ollama pull nomic-embed-text")
        await manager.disconnect_all()
        return False
    print("OK")

    print("      Generating embeddings and building index...")
    result = await index.build_index(tool_defs)

    # Cleanup
    await index.close()
    await manager.disconnect_all()

    # Summary
    print("\n" + "-" * 60)
    print("  SYNC COMPLETE")
    print("-" * 60)
    print(f"  Tools indexed: {result['tools_indexed']}")
    print(f"  Build time: {result['total_time']:.2f}s")
    print("  Index ready for queries")
    print("-" * 60 + "\n")

    return True


def categorize_tool(name: str, description: str) -> str:
    """Infer category from tool name and description."""
    name_lower = name.lower()
    description_lower = (description or "").lower()

    # Category keywords checked against both name and description
    categories = [
        ("file", ["file", "read", "write", "directory", "path"]),
        ("git", ["git", "commit", "branch", "repo"]),
        ("database", ["db_", "sql", "database", "query"]),
        ("search", ["search", "find", "lookup"]),
        ("ai", ["comfy", "image", "generate", "video"]),
        ("analysis", ["scan", "analyze", "health", "report"]),
        ("project", ["project", "session", "content"]),
        ("system", ["status", "health", "service"]),
    ]

    for category, keywords in categories:
        if any(kw in name_lower for kw in keywords):
            return category

    # Fallback: check description when name yields no match
    for category, keywords in categories:
        if any(kw in description_lower for kw in keywords):
            return category

    return "other"


async def run_tests():
    """Run test queries to verify gateway functionality."""
    print("\n" + "=" * 60)
    print("TOOL COMPASS GATEWAY - TEST SUITE")
    print("=" * 60)

    # GW-SB-002: get_index() raises RuntimeError on cold start (no baked index
    # AND Ollama unreachable) — the same error the MCP tools convert to a
    # structured envelope. Surface the human remediation to stderr instead of a
    # raw traceback, then exit non-zero so CLI callers see the failure.
    try:
        index = await get_index()
    except RuntimeError as e:
        import sys

        print(
            "❌ No index yet — run 'python gateway.py --sync' first, "
            "ensure 'ollama serve' is running.",
            file=sys.stderr,
        )
        print(f"   ({e})", file=sys.stderr)
        sys.exit(1)
    stats = index.get_stats()

    print(f"\nIndex: {stats['total_tools']} tools")
    print(f"Categories: {list(stats['by_category'].keys())}")

    test_cases = [
        ("read a file from disk", "read_file"),
        ("write content to a file", "write_file"),
        ("show git commit history", "git_log"),
        ("generate an AI image from text", "comfy_generate"),
        ("search for documents", "search"),
        ("check database schema", "db_inspect"),
        ("analyze code quality", "scan"),
        ("create a video from prompt", "video_generate"),
        ("list all projects", "list_projects"),
        ("execute SQL query", "db_execute"),
    ]

    print("\n" + "-" * 60)
    print("Semantic Search Tests")
    print("-" * 60)

    passed = 0
    for query, expected in test_cases:
        results = await index.search(query, top_k=3)
        top_match = results[0] if results else None

        if top_match and expected.lower() in top_match.tool.name.lower():
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        actual = top_match.tool.name if top_match else "None"
        score = f"{top_match.score:.3f}" if top_match else "N/A"
        print(f"[{status}] '{query}' -> {actual} ({score})")

    print(
        f"\nResults: {passed}/{len(test_cases)} passed ({100 * passed / len(test_cases):.0f}%)"
    )

    # Latency test
    print("\n" + "-" * 60)
    print("Latency Test")
    print("-" * 60)

    import time

    times = []
    for query, _ in test_cases:
        start = time.time()
        await index.search(query, top_k=5)
        times.append(time.time() - start)

    avg_ms = 1000 * sum(times) / len(times)
    print(f"Average search latency: {avg_ms:.1f}ms")

    await index.close()


def show_config():
    """Display current configuration."""
    config = load_config()

    print("\n" + "=" * 60)
    print("TOOL COMPASS GATEWAY - CONFIGURATION")
    print("=" * 60)

    print(f"\nConfig file: {CONFIG_PATH}")
    print(f"Config exists: {CONFIG_PATH.exists()}")

    print("\n--- Settings ---")
    print(f"Progressive disclosure: {config.progressive_disclosure}")
    print(f"Auto sync: {config.auto_sync}")
    print(f"Embedding model: {config.embedding_model}")
    # CFG-A-001 sibling: scrub credentials embedded in the URL before printing.
    print(f"Ollama URL: {redact_url_credentials(config.ollama_url)}")
    print(f"Default top_k: {config.default_top_k}")
    print(f"Min confidence: {config.min_confidence}")

    print("\n--- Backends ---")
    for name, backend in config.backends.items():
        print(f"\n{name}:")
        print(f"  Type: {backend.type}")
        if hasattr(backend, "command"):
            print(f"  Command: {backend.command}")
            # backend.args carries ${VAR}-resolved secrets (e.g.
            # --password=...) exactly like env/headers, so redact each entry
            # rather than printing it raw. (CFG-A-001 sibling.)
            redacted_args = _redact_structural(backend.args)
            print(
                f"  Args: {redacted_args[:2]}..."
                if len(redacted_args) > 2
                else f"  Args: {redacted_args}"
            )


async def async_main(args) -> bool:
    """Handle async CLI operations.

    Returns True on success, False when --sync indexed nothing (GW-SB-001).
    """
    if args.sync:
        return await sync_from_backends()
    elif args.test:
        await run_tests()
    return True


def build_http_app():
    """Construct the gateway's HTTP ASGI app with the ops endpoints attached.

    Registers /health, /ready and /metrics on FastMCP's custom Starlette route
    list (idempotently) and returns ``mcp.streamable_http_app()`` — the very
    app ``_run_http`` serves. Exposed at module scope (GW-FT-003) so tests and
    operators can mount the app, e.g. via ``starlette.testclient.TestClient``,
    without binding a socket. The route handlers below are byte-for-byte the
    ones used in production; nothing about their runtime behavior changes.
    """
    from starlette.routing import Route
    from starlette.responses import JSONResponse, PlainTextResponse

    # GW-FT-003 + BE-B-011: cache /ready result so LB polling can't hammer
    # Ollama or the backend pool, but with asymmetric TTLs — 'ready' caches
    # generously, 'not_ready' caches briefly. Without asymmetry a transition
    # ready -> not_ready can route up to 30s of traffic into a degraded
    # instance after the failure.
    _ready_cache: Dict[str, Any] = {"at": 0.0, "status_code": 0, "body": None}
    _READY_CACHE_TTL_OK = 30.0   # generous when everything is fine
    _READY_CACHE_TTL_FAIL = 2.0  # tight when something is broken

    def _invalidate_local_ready_cache() -> None:
        _ready_cache["body"] = None
        _ready_cache["at"] = 0.0
        _ready_cache["status_code"] = 0

    # BE-B-011: a hook so _mark_ollama_down() can drop the cache immediately
    # rather than waiting for the next 2s TTL window — registered once, in the
    # guarded block below alongside the routes.

    async def health(_request):
        return JSONResponse({
            "status": "ok",
            "server": "tool-compass-gateway",
            "version": __version__,
        })

    async def ready(_request):
        """Deep readiness probe — 200 iff all dependencies are usable.

        BE-B-011: cache TTL is asymmetric. Failures cache for 2s only so a
        k8s liveness probe sees a state change within seconds; successes
        cache for 30s to keep dependency load low.
        """
        now = time.time()
        if _ready_cache["body"] is not None:
            ttl = (
                _READY_CACHE_TTL_OK
                if _ready_cache["status_code"] == 200
                else _READY_CACHE_TTL_FAIL
            )
            if (now - _ready_cache["at"]) < ttl:
                return JSONResponse(
                    _ready_cache["body"], status_code=_ready_cache["status_code"]
                )

        checks: Dict[str, Any] = {}

        # Index check — don't force a load here, just inspect current state.
        index_ok = False
        try:
            idx = _compass_index
            index_ok = idx is not None and getattr(idx, "db", None) is not None
            checks["index"] = {"ok": index_ok}
            if not index_ok:
                checks["index"]["reason"] = "not loaded"
        except Exception as e:  # defensive
            checks["index"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}

        # Ollama check — honour breaker state too. Breaker "closed" means
        # healthy; "open" means known-down; "half_open" means probing.
        ollama_ok = bool(_health_state.get("ollama_available"))
        breaker_state = None
        try:
            idx = _compass_index
            if idx is not None and getattr(idx, "embedder", None) is not None:
                breaker_state = idx.embedder.circuit_breaker_state()
                if breaker_state == "closed":
                    ollama_ok = True
        except Exception:
            pass
        checks["ollama"] = {
            "ok": ollama_ok,
            "breaker": breaker_state,
            "last_error": _health_state.get("last_ollama_error"),
        }

        # Backend check — at least one connected backend (and the manager exists).
        backend_ok = False
        connected_backends: List[str] = []
        configured_backends: List[str] = []
        try:
            mgr = _backend_manager
            if mgr is not None:
                configured_backends = list(mgr.config.backends.keys())
                for name in configured_backends:
                    if mgr.is_backend_connected(name):
                        connected_backends.append(name)
                backend_ok = bool(connected_backends) or not configured_backends
        except Exception as e:
            checks["backends"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        else:
            checks["backends"] = {
                "ok": backend_ok,
                "connected": connected_backends,
                "configured": configured_backends,
            }
            if not backend_ok:
                checks["backends"]["reason"] = (
                    "no backends connected" if configured_backends else "manager not initialized"
                )

        all_ok = index_ok and ollama_ok and backend_ok
        body = {
            "status": "ready" if all_ok else "not_ready",
            "checks": checks,
        }
        status_code = 200 if all_ok else 503
        _ready_cache["at"] = now
        _ready_cache["status_code"] = status_code
        _ready_cache["body"] = body
        return JSONResponse(body, status_code=status_code)

    # BE-B-002 + BE-B-015: Prometheus / OpenMetrics text format.
    #
    # Metrics added in BE-B-002:
    #   tool_compass_circuit_breaker_transitions_total{from,to,breaker}
    #   tool_compass_fallback_invocations_total{type}
    #   tool_compass_lexical_fallback_total
    #   tool_compass_degraded_responses_total{reason}
    #   tool_compass_hnsw_search_duration_seconds (gauge of p95)
    #   tool_compass_embedder_inflight (gauge)
    #   tool_compass_embedder_queue_wait_seconds (gauge of p95)
    #
    # Naming follows OpenMetrics rules (BE-B-015): _total suffix for
    # counters, _seconds suffix for time gauges. Emission terminates with
    # `# EOF\n` and the media type is application/openmetrics-text.

    async def metrics(_request):
        """Prometheus / OpenMetrics text format — no prometheus_client dep."""

        def _escape_label(v: str) -> str:
            # Prometheus label-value escaping: \, ", newline.
            return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        def _int_fmt(v: Any) -> str:
            try:
                return str(int(v))
            except (TypeError, ValueError):
                return "0"

        def _float_fmt(v: Any) -> str:
            try:
                return f"{float(v):.6g}"
            except (TypeError, ValueError):
                return "0"

        lines: List[str] = []

        # Search counter from analytics.
        search_total: Optional[int] = None
        try:
            analytics = _analytics
            if analytics is not None:
                summary = await analytics.get_analytics_summary("24h")
                # get_analytics_summary returns a dict; tolerate shape drift.
                if isinstance(summary, dict):
                    stats = summary.get("search_stats") or summary
                    # Try common keys.
                    for k in ("total_searches", "total", "count"):
                        v = stats.get(k) if isinstance(stats, dict) else None
                        if isinstance(v, (int, float)):
                            search_total = int(v)
                            break
        except Exception as e:
            logger.debug(f"metrics: analytics read failed: {e}")

        lines.append("# HELP tool_compass_search_total Total number of compass() searches (24h window).")
        lines.append("# TYPE tool_compass_search_total counter")
        lines.append(f"tool_compass_search_total {_int_fmt(search_total or 0)}")

        # Ollama availability gauge — 1 closed, 0 otherwise.
        ollama_val = 1 if _health_state.get("ollama_available") else 0
        try:
            idx = _compass_index
            if idx is not None and getattr(idx, "embedder", None) is not None:
                ollama_val = 1 if idx.embedder.circuit_breaker_state() == "closed" else 0
        except Exception:
            pass
        lines.append("# HELP tool_compass_ollama_available 1 if the Ollama circuit breaker is closed, else 0.")
        lines.append("# TYPE tool_compass_ollama_available gauge")
        lines.append(f"tool_compass_ollama_available {_int_fmt(ollama_val)}")

        # Per-backend gauges + call counters.
        lines.append("# HELP tool_compass_backend_up 1 if the named backend is connected, else 0.")
        lines.append("# TYPE tool_compass_backend_up gauge")
        lines.append("# HELP tool_compass_backend_call_total Backend tool-call counter, labelled by status.")
        lines.append("# TYPE tool_compass_backend_call_total counter")
        try:
            mgr = _backend_manager
            if mgr is not None:
                stats = mgr.get_stats()
                configured = stats.get("configured_backends") or list(mgr.config.backends.keys())
                connected = set(stats.get("connected_backends") or [])
                per_backend = stats.get("stats") or {}
                for name in configured:
                    label = _escape_label(name)
                    up = 1 if name in connected else 0
                    lines.append(f'tool_compass_backend_up{{name="{label}"}} {_int_fmt(up)}')
                    entry = per_backend.get(name, {}) if isinstance(per_backend, dict) else {}
                    total = int(entry.get("total_calls", 0) or 0)
                    failed = int(entry.get("failed_calls", 0) or 0)
                    success = max(0, total - failed)
                    lines.append(
                        f'tool_compass_backend_call_total{{name="{label}",status="success"}} {_int_fmt(success)}'
                    )
                    lines.append(
                        f'tool_compass_backend_call_total{{name="{label}",status="error"}} {_int_fmt(failed)}'
                    )
        except Exception as e:
            logger.debug(f"metrics: backend stats failed: {e}")

        # Embedder p95 + failures + new saturation gauges (BE-B-005 + BE-B-013).
        es: Dict[str, Any] = {}
        try:
            idx = _compass_index
            if idx is not None and getattr(idx, "embedder", None) is not None:
                es = idx.embedder.get_stats() or {}
        except Exception as e:
            logger.debug(f"metrics: embedder stats failed: {e}")

        embed_p95 = es.get("p95_latency_ms", 0.0)
        embed_failures = es.get("total_failures", 0)
        embed_inflight = es.get("inflight", 0)
        embed_consec = es.get("consecutive_failures", 0)
        embed_last_success_ms = es.get("time_since_last_success_ms")
        embed_qw_p95_ms = es.get("queue_wait_ms_p95", 0.0)

        lines.append("# HELP tool_compass_embed_latency_p95_ms p95 embed latency in ms from the bounded sample window.")
        lines.append("# TYPE tool_compass_embed_latency_p95_ms gauge")
        lines.append(f"tool_compass_embed_latency_p95_ms {_float_fmt(embed_p95)}")

        lines.append("# HELP tool_compass_embed_failures_total Total embed failures across the process lifetime.")
        lines.append("# TYPE tool_compass_embed_failures_total counter")
        lines.append(f"tool_compass_embed_failures_total {_int_fmt(embed_failures)}")

        # BE-B-005: embedder inflight gauge — saturation signal that surfaces
        # contention BEFORE Ollama latency p95 spikes.
        lines.append("# HELP tool_compass_embedder_inflight In-flight Ollama embed calls.")
        lines.append("# TYPE tool_compass_embedder_inflight gauge")
        lines.append(f"tool_compass_embedder_inflight {_int_fmt(embed_inflight)}")

        # BE-B-005: queue-wait time. Use _seconds suffix per OpenMetrics
        # (BE-B-015) but report a gauge of p95 (full histogram would inflate
        # the /metrics body — operators paginate to the structured event log
        # for higher resolution).
        lines.append("# HELP tool_compass_embedder_queue_wait_seconds p95 wait time for embedder concurrency slot.")
        lines.append("# TYPE tool_compass_embedder_queue_wait_seconds gauge")
        lines.append("# UNIT tool_compass_embedder_queue_wait_seconds seconds")
        lines.append(
            f"tool_compass_embedder_queue_wait_seconds {_float_fmt((embed_qw_p95_ms or 0.0) / 1000.0)}"
        )

        # BE-B-013: consecutive failures + time since last success — surfaces
        # "breaker about to trip" vs. "breaker just tripped" vs. "breaker
        # down all day".
        lines.append("# HELP tool_compass_embed_consecutive_failures Live consecutive Ollama failure count.")
        lines.append("# TYPE tool_compass_embed_consecutive_failures gauge")
        lines.append(f"tool_compass_embed_consecutive_failures {_int_fmt(embed_consec)}")

        if embed_last_success_ms is not None:
            lines.append(
                "# HELP tool_compass_embed_time_since_last_success_seconds "
                "Seconds since the last successful embed call."
            )
            lines.append(
                "# TYPE tool_compass_embed_time_since_last_success_seconds gauge"
            )
            lines.append(
                f"tool_compass_embed_time_since_last_success_seconds "
                f"{_float_fmt(float(embed_last_success_ms) / 1000.0)}"
            )

        # BE-B-002: circuit-breaker transitions counter. Distinguishes a
        # flapping breaker from a steadily-open breaker (Nygard 'Release It!').
        lines.append(
            "# HELP tool_compass_circuit_breaker_transitions_total "
            "Count of circuit-breaker state transitions."
        )
        lines.append(
            "# TYPE tool_compass_circuit_breaker_transitions_total counter"
        )
        transitions = _metric_counters["circuit_breaker_transitions_total"]
        emitted_any_transition = False
        for key, count in transitions.items():
            try:
                from_state, to_state = key.split("->", 1)
            except ValueError:
                continue
            emitted_any_transition = True
            lines.append(
                f'tool_compass_circuit_breaker_transitions_total'
                f'{{from="{_escape_label(from_state)}",to="{_escape_label(to_state)}",'
                f'breaker="ollama"}} {_int_fmt(count)}'
            )
        if not emitted_any_transition:
            # Always emit at least one zero-valued line so dashboards don't
            # break with "no such metric" errors on a fresh process.
            lines.append(
                'tool_compass_circuit_breaker_transitions_total'
                '{from="closed",to="closed",breaker="ollama"} 0'
            )

        # BE-B-002: lexical fallback counter — every time semantic search
        # falls back to LIKE. Alert on "served N lexical responses in 60s".
        lines.append(
            "# HELP tool_compass_lexical_fallback_total Lexical fallback invocations on compass()."
        )
        lines.append("# TYPE tool_compass_lexical_fallback_total counter")
        lines.append(
            f"tool_compass_lexical_fallback_total "
            f"{_int_fmt(_metric_counters['lexical_fallback_total'])}"
        )

        # BE-B-002: generic fallback-by-type counter (lexical | chain | ...)
        lines.append(
            "# HELP tool_compass_fallback_invocations_total "
            "Generic fallback invocation counter by type."
        )
        lines.append("# TYPE tool_compass_fallback_invocations_total counter")
        fb_invocations = _metric_counters["fallback_invocations_total"]
        if fb_invocations:
            for fb_type, count in fb_invocations.items():
                lines.append(
                    f'tool_compass_fallback_invocations_total{{type="{_escape_label(fb_type)}"}} '
                    f"{_int_fmt(count)}"
                )
        else:
            lines.append('tool_compass_fallback_invocations_total{type="lexical"} 0')

        # BE-B-002: degraded responses counter by reason.
        lines.append(
            "# HELP tool_compass_degraded_responses_total "
            "Responses served while a degraded reason was active."
        )
        lines.append("# TYPE tool_compass_degraded_responses_total counter")
        deg_by_reason = _metric_counters["degraded_responses_total"]
        if deg_by_reason:
            for reason, count in deg_by_reason.items():
                lines.append(
                    f'tool_compass_degraded_responses_total{{reason="{_escape_label(reason)}"}} '
                    f"{_int_fmt(count)}"
                )
        else:
            lines.append(
                'tool_compass_degraded_responses_total{reason="ollama_unavailable"} 0'
            )

        # BE-B-002: HNSW search latency p95 gauge with ef_search label so a
        # config change is visible in the time series.
        hnsw_p95_ms = 0.0
        hnsw_ef = 0
        index_age = 0.0
        orphaned = 0
        try:
            idx = _compass_index
            if idx is not None:
                istats = idx.get_stats()
                age = istats.get("index_age_seconds")
                if isinstance(age, (int, float)):
                    index_age = float(age)
                orphaned = int(istats.get("orphaned_vector_count", 0) or 0)
                hnsw_p95_ms = float(istats.get("hnsw_search_latency_ms_p95", 0.0) or 0.0)
                hnsw = istats.get("hnsw") or {}
                hnsw_ef = int(hnsw.get("ef_search", 0) or 0)
        except Exception as e:
            logger.debug(f"metrics: index stats failed: {e}")

        lines.append(
            "# HELP tool_compass_hnsw_search_duration_seconds "
            "p95 HNSW knn_query latency (seconds), labelled by ef_search."
        )
        lines.append("# TYPE tool_compass_hnsw_search_duration_seconds gauge")
        lines.append("# UNIT tool_compass_hnsw_search_duration_seconds seconds")
        lines.append(
            f'tool_compass_hnsw_search_duration_seconds{{ef_search="{hnsw_ef}"}} '
            f"{_float_fmt(hnsw_p95_ms / 1000.0)}"
        )

        lines.append("# HELP tool_compass_index_age_seconds Seconds since the index was last built (0 if unknown).")
        lines.append("# TYPE tool_compass_index_age_seconds gauge")
        lines.append("# UNIT tool_compass_index_age_seconds seconds")
        lines.append(f"tool_compass_index_age_seconds {_float_fmt(index_age)}")

        lines.append("# HELP tool_compass_orphaned_vectors HNSW entries with no DB mapping.")
        lines.append("# TYPE tool_compass_orphaned_vectors gauge")
        lines.append(f"tool_compass_orphaned_vectors {_int_fmt(orphaned)}")

        # OpenMetrics 1.0.0 terminator (BE-B-015).
        lines.append("# EOF")
        body = "\n".join(lines) + "\n"
        return PlainTextResponse(
            body,
            media_type="application/openmetrics-text; version=1.0.0; charset=utf-8",
        )

    # Add health / ready / metrics routes to FastMCP's custom routes
    # (included in streamable_http_app). Guarded against the *current* route
    # list so repeated build_http_app() calls don't stack duplicate routes —
    # keyed on path rather than a module flag so a test that swaps in a fresh
    # _custom_starlette_routes still gets its handlers registered.
    handlers = {"/health": health, "/ready": ready, "/metrics": metrics}
    existing = {getattr(r, "path", None) for r in mcp._custom_starlette_routes}
    if "/ready" not in existing:
        # First registration onto this list also wires the cache invalidator
        # so _mark_ollama_down() can drop the cached /ready result immediately.
        _ready_cache_invalidators.append(_invalidate_local_ready_cache)
    for path, handler in handlers.items():
        if path not in existing:
            mcp._custom_starlette_routes.append(Route(path, handler, methods=["GET"]))

    return mcp.streamable_http_app()


def _run_http(port: int) -> None:
    """Run the MCP gateway in HTTP mode with /health, /ready, /metrics.

    SECURITY: The gateway proxies arbitrary MCP tool calls to backend servers.
    Binding to a non-loopback interface exposes RCE-class surface. The HOST env
    var defaults to 127.0.0.1 (loopback). Only bind to public interfaces when
    running behind an authenticated reverse proxy (Fly.io edge, etc.).
    """
    import os
    from mcp.server.transport_security import TransportSecuritySettings

    # Build + register the ops routes (idempotent) before the server starts.
    build_http_app()

    host = os.environ.get("HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            f"HTTP mode binding to non-loopback host {host!r}. "
            "The gateway proxies arbitrary MCP tool calls — ensure an authenticated "
            "reverse proxy is in front. Set HOST=127.0.0.1 for loopback-only."
        )

    mcp.settings.host = host
    mcp.settings.port = port
    # Allow Fly.io and Smithery proxy hosts (0.0.0.0 intentionally omitted — never a valid Host header)
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "tool-compass-gateway.fly.dev",
            "tool-compass-gateway--mcp-tool-shop.run.tools",
            "localhost",
            "127.0.0.1",
        ],
    )
    mcp.run(transport="streamable-http")


def main():
    parser = argparse.ArgumentParser(
        prog="gateway",
        description="Tool Compass Gateway - Semantic MCP Proxy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gateway.py              Start the MCP gateway server (stdio mode)
  python gateway.py --sync       Sync tools from backends and rebuild index
  python gateway.py --test       Run test queries against the index
  python gateway.py --config     Show current configuration

Prerequisites:
  - Ollama must be running: ollama serve
  - Embedding model required: ollama pull nomic-embed-text

Workflow:
  1. First run --sync to build the tool index from backend servers
  2. Then start the gateway for MCP clients to connect
  3. Use compass() -> describe() -> execute() pattern

For more info, see: https://github.com/mcp-tool-shop-org/tool-compass
        """
    )
    parser.add_argument("--sync", action="store_true",
                        help="Sync tools from backend MCP servers and rebuild the HNSW index")
    parser.add_argument("--test", action="store_true",
                        help="Run semantic search tests to verify index quality")
    parser.add_argument("--config", action="store_true",
                        help="Display current configuration including backends and settings")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output with detailed progress")

    args = parser.parse_args()

    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if args.config:
        show_config()
    elif args.sync or args.test:
        # GW-SB-001: a sync that connected no backends / discovered no tools /
        # found Ollama down returns False — exit non-zero with a one-line hint
        # so CI/deploy wrappers don't mistake an empty sync for success.
        ok = asyncio.run(async_main(args))
        if not ok:
            import sys

            print(
                "Index NOT rebuilt: 0 backends connected (or no tools / Ollama down).",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # NOTE: Never print() to stdout in MCP mode - it corrupts JSON-RPC!
        # Use stderr for diagnostics if needed
        import sys

        # FE-W11-008: read __version__ from _version.py rather than embedding
        # a hardcoded literal that drifts on every release. The Wave-10 audit
        # called this out as a stale banner.
        print(f"Starting Tool Compass Gateway v{__version__}...", file=sys.stderr)
        print(
            "Tools: compass, describe, execute, compass_categories, compass_status",
            file=sys.stderr,
        )
        print(
            "       compass_analytics, compass_chains, compass_sync, compass_audit",
            file=sys.stderr,
        )
        print(
            "Features: auto-sync, hot cache, usage analytics, tool chains",
            file=sys.stderr,
        )
        print("Workflow: compass() -> describe() -> execute()", file=sys.stderr)

        # Select transport: PORT env var → HTTP (Fly.io), else stdio (local)
        import os
        port = os.environ.get("PORT")
        if port:
            # GW-SB-003: a malformed PORT used to raise a bare ValueError with
            # no hint that PORT was the culprit. Catch it, name the variable and
            # the expected range, and exit 2 (usage error).
            try:
                port_num = int(port)
            except ValueError:
                print(
                    f"Invalid PORT={port!r}: must be an integer 1-65535",
                    file=sys.stderr,
                )
                sys.exit(2)
            if not 1 <= port_num <= 65535:
                print(
                    f"Invalid PORT={port!r}: must be an integer 1-65535",
                    file=sys.stderr,
                )
                sys.exit(2)
            host = os.environ.get("HOST", "127.0.0.1")
            print(f"Transport: streamable-http on {host}:{port}", file=sys.stderr)
            _run_http(port_num)
        else:
            print("Transport: stdio", file=sys.stderr)
            mcp.run()


if __name__ == "__main__":
    main()
