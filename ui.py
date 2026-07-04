"""
Tool Compass - Gradio UI
Interactive web interface for semantic tool discovery, browsing, and analytics.

Usage:
    python ui.py              # Launch standalone UI on port 7860
    python ui.py --port 7861  # Custom port
    python ui.py --share      # Create public Gradio link
"""

import asyncio
import html
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, List, Dict
import argparse

import gradio as gr

from _version import __version__

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from indexer import CompassIndex
from analytics import CompassAnalytics, get_analytics
from chain_indexer import ChainIndexer, get_chain_indexer
from config import load_config


# =============================================================================
# ASYNC HELPERS
# =============================================================================


def run_async(coro):
    """Run an async coroutine from synchronous Gradio callbacks.

    Safe in two scenarios:
    - No running loop → uses asyncio.run() directly.
    - Inside a running loop (Gradio's event loop) → dispatches to a
      worker thread with its own asyncio.run() to avoid nested-loop crashes.
    """
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # Loop is already running — use a worker thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop — safe to call directly.
        return asyncio.run(coro)


# =============================================================================
# GLOBAL STATE
# Gradio serves requests from a thread pool, so singletons need a
# threading.Lock (not asyncio.Lock) for safe double-checked init.
# =============================================================================

_index: Optional[CompassIndex] = None
_analytics: Optional[CompassAnalytics] = None
_chain_indexer: Optional[ChainIndexer] = None
_config = None
_init_lock = threading.RLock()


def _lazy_singleton(global_name: str, factory):
    """Thread-safe lazy init with publish-after-success guard (FE-B-011).

    Centralises the FE-A2-003/004/005 partial-init pattern previously
    copy-pasted across get_index, get_analytics_instance, and
    get_chain_indexer_instance:

    1. Fast path — return cached without taking the lock if already built.
    2. Acquire lock, re-check (double-checked locking).
    3. Run factory; bind to a local first. If factory raises, the global
       stays at None so the next caller retries from scratch — no leaked
       half-initialized singletons.
    4. Publish to the module global AFTER the factory returned cleanly.

    factory() is called inside the lock and may itself run async setup via
    run_async; it must return the fully-initialized instance.

    Returns the singleton (or whatever the factory returns; None is a
    legitimate value and will be re-attempted on the next call).
    """
    current = globals().get(global_name)
    if current is not None:
        return current
    with _init_lock:
        current = globals().get(global_name)
        if current is not None:
            return current
        instance = factory()
        if instance is not None:
            globals()[global_name] = instance
        return instance


def get_index() -> CompassIndex:
    """Get or initialize compass index (thread-safe).

    FE-A2-003 + FE-B-011: factored through `_lazy_singleton` so the
    partial-init guard is centralised. The factory raises RuntimeError
    when load_index() fails — that propagates to the caller and the
    global stays at None so the next call retries cleanly.
    """

    def _build():
        idx = CompassIndex()
        if not idx.load_index():
            raise RuntimeError("Failed to load index. Run: tool-compass sync")
        return idx

    return _lazy_singleton("_index", _build)


def get_analytics_instance() -> CompassAnalytics:
    """Get or initialize analytics (thread-safe).

    FE-A2-005 + FE-B-011: factored through `_lazy_singleton`. hot_cache
    load can raise (sqlite3.OperationalError, schema mismatch, DB locked
    from concurrent sync_manager/chain_indexer writes); on raise the
    global stays at None and the next caller re-attempts.
    """

    def _build():
        analytics = get_analytics()
        run_async(analytics.load_hot_cache_from_db())
        return analytics

    return _lazy_singleton("_analytics", _build)


def get_chain_indexer_instance() -> Optional[ChainIndexer]:
    """Get or initialize chain indexer (thread-safe).

    FE-A2-004 + FE-B-011: factored through `_lazy_singleton` for the load
    path. The config-gated short-circuit (chain_indexing_enabled=False)
    legitimately returns None, which `_lazy_singleton` will treat as
    "retry next call" — matches the current observable behavior since a
    disabled chain indexer should not be cached.
    """
    global _config
    with _init_lock:
        if _config is None:
            _config = load_config()
    if not _config.chain_indexing_enabled:
        return None

    def _build():
        index = get_index()
        analytics = get_analytics_instance()
        ci = get_chain_indexer(index.embedder, analytics)
        run_async(ci.load_chain_index())
        return ci

    return _lazy_singleton("_chain_indexer", _build)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def sanitize_query(query: str) -> str:
    """Sanitize search query - remove potentially problematic characters."""
    if not query:
        return ""
    # Allow alphanumeric, spaces, basic punctuation for natural language queries
    # Strip control characters and excessive whitespace
    sanitized = "".join(c for c in query if c.isprintable())
    return " ".join(sanitized.split())[:500]  # Limit length


def truncate_text(text: str, max_length: int = 120) -> str:
    """Truncate text gracefully with ellipsis."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[: max_length - 3].rsplit(" ", 1)[0] + "..."


def confidence_label(score: float) -> str:
    """Return human-readable confidence label."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Low"


def _check_ollama_banner() -> str:
    """Return an Ollama-down banner (markdown) or empty string (MCC-B-007).

    Runs a fast 2s health_check. The banner is only emitted as a separate
    chrome element when Ollama is unreachable AND no active search is showing
    its own inline fallback notice (FE-B-008 — degraded-mode signals belong
    adjacent to the affected results, not in chrome).

    Returned text does NOT promise "keyword-based fallback" — that claim is
    only made by `search_tools()` when the fallback actually runs (FE-B-001).
    """
    try:
        from embedder import Embedder

        cfg = _config if _config is not None else load_config()
        ollama_url = getattr(cfg, "ollama_url", "http://localhost:11434")
        embedder = Embedder(base_url=ollama_url, timeout=2.0)
        # FE-SB-003: close() must run even when health_check() raises (Ollama
        # down is the common path) or the aiohttp session leaks.
        try:
            is_healthy = run_async(embedder.health_check())
        finally:
            run_async(embedder.close())
        if is_healthy:
            return ""
        return (
            f"⚠️ **Ollama unavailable** at {ollama_url}. "
            "Searches will fall back to keyword matching until you start it "
            "with `ollama serve`."
        )
    except Exception as e:
        # Keep the banner check itself non-fatal — if probing fails, assume
        # down and warn rather than letting the UI crash.
        logger.debug(f"Ollama banner check failed: {e}")
        return (
            "⚠️ **Ollama unavailable**. Searches will fall back to keyword "
            "matching until you start it with `ollama serve`."
        )


def _lexical_fallback_for_ui(
    index, query: str, top_k: int, category: Optional[str], server: Optional[str]
) -> List[Dict]:
    """UI-side lexical fallback over the indexed tools (FE-B-001).

    Re-uses `gateway._lexical_search_fallback` so the UI keeps the same
    fallback contract the MCP `compass` tool already documents. Cross-domain
    coupling is intentional — the function is the single source of truth for
    the "graceful Ollama-down" marketing claim on site-config.ts and
    operations.md. If the import fails (gateway not on path, circular guard),
    we fall through to an inlined LIKE scan that matches the gateway shape.
    """
    try:
        from gateway import _lexical_search_fallback

        return _lexical_search_fallback(index, query, top_k, category, server)
    except Exception:
        # Inline fallback — defensive, matches gateway shape.
        if not index or not getattr(index, "db", None):
            return []
        # FE-SA-002: escape LIKE wildcards (% and _) and pair with ESCAPE so a
        # query containing those characters is matched literally, not as a
        # wildcard — mirroring gateway._escape_like / _lexical_search_fallback.
        escaped = (
            query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        needle = f"%{escaped}%"
        where = [
            "(lower(name) LIKE lower(?) ESCAPE '\\' "
            "OR lower(description) LIKE lower(?) ESCAPE '\\')"
        ]
        params: list = [needle, needle]
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
        try:
            rows = index.db.execute(sql, params).fetchall()
        except Exception:
            return []
        q_lower = query.strip().lower()
        scored = []
        for row in rows:
            name_lower = (row["name"] or "").lower()
            desc_lower = (row["description"] or "").lower()
            if q_lower in name_lower:
                confidence = 0.6
            elif q_lower in desc_lower:
                confidence = 0.4
            else:
                confidence = 0.3
            scored.append({
                "tool": row["name"],
                "description": row["description"] or "",
                "server": row["server"],
                "category": row["category"],
                "confidence": confidence,
                "degraded": True,
            })
        scored.sort(key=lambda m: m["confidence"], reverse=True)
        return scored[:top_k]


def format_error(error: Exception, context: str = "") -> str:
    """Format error message for user display.

    FE-A2-001/002: All caller-supplied context and exception strings are
    HTML-escaped before interpolation. Both ``context`` (often contains the
    user query / tool_name) and ``str(error)`` (can carry arbitrary backend
    payloads from Ollama, sqlite, hnswlib, MCP servers) are untrusted from
    the renderer's perspective — escape at the boundary, not at every
    caller, so future callers can't reintroduce the gap.
    """
    error_type = type(error).__name__
    safe_error_type = html.escape(error_type, quote=True)
    safe_error_str = html.escape(str(error)[:200], quote=True)
    safe_context = html.escape(context, quote=True) if context else ""

    # MCC-B-008: error banners get role="alert" so screen readers announce
    # them immediately; warning emoji is aria-hidden (text twin "Warning:"
    # carries the semantic — Léonie Watson 2023). FE-B-012: contrast-fixed
    # body greys (#e8e8f0 / #a0a0b0) replace #ccc / #888 which fail 4.5:1.
    # SD-V-001: body text bumped to #e8e8f0 (APCA Lc target 75-90 on dark);
    # mono font stack used for the inline `ollama serve` / `tool-compass
    # sync` commands so the eye picks them up as runnable code.
    warn_icon = '<span aria-hidden="true">⚠️</span>'
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, monospace"
    if "Connection" in error_type or "refused" in str(error).lower():
        return f"""
        <div role="alert" style="border: 1px solid #ef5350; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a1a1a;">
            <div style="color: #ef5350; font-weight: bold;">{warn_icon} Service unavailable</div>
            <p style="color: #e8e8f0; margin: 8px 0; line-height: 1.5; max-width: 75ch;">
                Cannot connect to Ollama embeddings service. Start it with the command below, then retry.
            </p>
            <code style="color: #d4d4dc; font-size: 0.9em; font-family: {mono};">ollama serve</code>
        </div>
        """
    elif "index" in str(error).lower() or "not loaded" in str(error).lower():
        # FE-A-008: replace the legacy `cd tool_compass && python gateway.py
        # --sync` snippet with the v2.2 canonical CLI form.
        return f"""
        <div role="alert" style="border: 1px solid #ffb74d; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a2a1a;">
            <div style="color: #ffb74d; font-weight: bold;">{warn_icon} Index not ready</div>
            <p style="color: #e8e8f0; margin: 8px 0; line-height: 1.5; max-width: 75ch;">
                The tool index has not been built yet. Run the sync command below to build it.
            </p>
            <code style="color: #d4d4dc; font-size: 0.9em; font-family: {mono};">tool-compass sync</code>
        </div>
        """
    else:
        return f"""
        <div role="alert" style="border: 1px solid #ef5350; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a1a1a;">
            <div style="color: #ef5350; font-weight: bold;">{warn_icon} Something went wrong</div>
            <p style="color: #e8e8f0; margin: 8px 0; line-height: 1.5; max-width: 75ch;">{safe_context or "An error occurred while running this action."}</p>
            <details style="color: #a0a0b0; font-size: 0.9em;">
                <summary>Technical details</summary>
                <code style="font-family: {mono};">{safe_error_type}: {safe_error_str}</code>
            </details>
        </div>
        """


# =============================================================================
# SEARCH FUNCTIONS
# =============================================================================


def search_tools(
    query: str,
    top_k: int = 5,
    category: str = "All",
    server: str = "All",
    min_confidence: float = 0.3,
) -> tuple:
    """
    Search for tools using semantic search.

    Returns (results_html, results_json).

    FE-B-001: when semantic search raises (Ollama unreachable, embedder
    failure), we fall back to the same lexical LIKE scan the MCP `compass`
    tool uses (`gateway._lexical_search_fallback`). The Ollama-down banner
    promises keyword fallback — this is where that promise gets kept. The
    fallback path emits a *single* inline banner above the results so users
    don't see contradicting "fallback active" + "service unavailable" cards
    (the FE-B-001 contradiction).

    FE-B-002: when results are empty after confidence filtering, surface
    up-to-3 nearest-neighbor "did you mean" matches via a LIKE scan instead
    of leaving the user on a dead-end zero-results page (NN/g zero-results
    pattern — show closest matches as actionable suggestions).
    """
    # Empty query
    if not query.strip():
        return (
            """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔍</div>
            <p style="color: #e8e8f0; line-height: 1.5;">Enter a search query above to find tools.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Try: "generate an image", "read a file", "search documents"</p>
        </div>
        """,
            "{}",
        )

    # Sanitize input
    query = sanitize_query(query)
    if not query:
        return "<p style='color: orange;'>Please enter a valid search query.</p>", "{}"

    try:
        index = get_index()
    except Exception as e:
        return format_error(e, "Could not load the tool index"), "{}"

    # FE-SB-002: a synced-but-empty index (load_index returns True with 0 rows)
    # must show the same "no tools indexed yet → sync" card the Browser tab
    # uses — NOT the generic no-match page ("reword / lower confidence /
    # remove filters"), whose advice can never surface a tool from an empty
    # index. Short-circuit before running search.
    try:
        if index.get_stats().get("total_tools", 0) == 0:
            return _render_empty_index(), "{}"
    except Exception:
        # Stats unavailable — fall through and let search run / fail normally.
        pass

    # Handle filter values
    cat_filter = None if category == "All" else category
    srv_filter = None if server == "All" else server

    # Run semantic search; on failure fall back to lexical LIKE.
    degraded = False
    fallback_used = False
    try:

        async def do_search():
            return await index.search(
                query=query,
                top_k=int(top_k),
                category_filter=cat_filter,
                server_filter=srv_filter,
            )

        results = run_async(do_search())
        # Filter by confidence
        results = [r for r in results if r.score >= min_confidence]
    except Exception as e:
        # FE-B-001: keep the search alive by falling back to lexical instead
        # of erroring out. The Ollama-down banner already explains WHY this
        # mode is active; we render a single inline notice above the cards
        # and DO NOT also surface a red "service unavailable" card.
        logger.warning(
            "Semantic search failed (%s: %s); using lexical fallback.",
            type(e).__name__,
            truncate_text(str(e), 80),
        )
        degraded = True
        fallback_used = True
        fb_matches = _lexical_fallback_for_ui(
            index, query, int(top_k), cat_filter, srv_filter
        )
        fb_matches = [m for m in fb_matches if m["confidence"] >= min_confidence]
        # Convert lexical matches to the SearchResult-like duck shape the
        # downstream rendering loop expects (.tool.name, .tool.description,
        # .tool.server, .tool.category, .score, .tool.parameters).
        from types import SimpleNamespace

        results = [
            SimpleNamespace(
                tool=SimpleNamespace(
                    name=m["tool"],
                    description=m.get("description") or "",
                    server=m["server"],
                    category=m["category"],
                    parameters={},
                ),
                score=m["confidence"],
            )
            for m in fb_matches
        ]

    # No results — try a relaxed nearest-match pass (FE-B-002) before
    # surrendering. Lower the floor by 0.1 and re-run; if still empty, do a
    # LIKE substring scan against the indexed tools and surface up to 3
    # "did you mean" suggestions.
    if not results:
        suggestions = _nearest_matches(
            index, query, max_results=3, cat_filter=cat_filter, srv_filter=srv_filter
        )
        return _render_no_results(query, suggestions), "{}"

    # Build HTML output
    html_parts = []
    if fallback_used:
        # Single source of truth for the degraded state (FE-B-001): inline
        # banner adjacent to the results region, NOT in chrome.
        html_parts.append(_inline_fallback_banner())
    # FE-B-004: result-count heading is a focusable h2 with aria-live so SR
    # users hear the count change after async load. SD-V-001 contrast: body
    # grey lifted to #a0a0b0 (APCA Lc ~70 on #1a1a2e).
    safe_q = html.escape(truncate_text(query, 60), quote=True)
    count_text = (
        f"Found {len(results)} tool" + ("s" if len(results) != 1 else "")
        + f' for "{safe_q}"'
    )
    html_parts.append(
        f'<h2 id="search-results-count" tabindex="-1" '
        f'aria-live="polite" aria-atomic="true" '
        f'style="color: #a0a0b0; font-size: 1em; font-weight: normal; '
        f'margin: 0 0 12px 0;">{count_text}</h2>'
    )
    # FE-B-006: announce the results container as a listbox so SR users know
    # the result cards are options associated with the search input.
    html_parts.append(
        '<ul id="search-results-list" role="listbox" '
        'aria-label="Tool search results" '
        'style="list-style: none; padding: 0; margin: 0;">'
    )
    json_results = []

    for idx_i, r in enumerate(results):
        confidence_pct = int(r.score * 100)
        conf_label = confidence_label(r.score)
        # MCC-B-008: brighter palette for higher contrast on the dark theme.
        confidence_color = (
            "#a5d6a7" if r.score > 0.7 else "#ffcc80" if r.score > 0.5 else "#e0e0e0"
        )

        # Stars based on confidence
        stars = "★" * min(5, int(r.score * 5 + 0.5)) + "☆" * (
            5 - min(5, int(r.score * 5 + 0.5))
        )

        # Truncate long descriptions — escape every untrusted string that lands
        # in HTML to block <script>/style/attr injection from tool metadata.
        desc_display = truncate_text(r.tool.description, 150)
        safe_name = html.escape(r.tool.name, quote=True)
        safe_name_short = html.escape(truncate_text(r.tool.name, 40), quote=True)
        safe_desc = html.escape(r.tool.description or "", quote=True)
        safe_desc_short = html.escape(desc_display, quote=True)
        safe_server = html.escape(r.tool.server, quote=True)
        safe_category = html.escape(r.tool.category, quote=True)

        # MCC-FT-002: show a deprecation badge when the tool has
        # `deprecated_since` set. Defensive getattr — dynamically-discovered
        # tools that pre-date this field should render unchanged.
        deprecated_since = getattr(r.tool, "deprecated_since", None)
        deprecated_badge = ""
        if deprecated_since:
            safe_dep_ver = html.escape(str(deprecated_since), quote=True)
            deprecated_badge = (
                f'<span class="deprecated" role="status" aria-label="deprecated" '
                f'style="background: #5c2c2c; color: #ffb4b4; padding: 2px 8px; '
                f'border-radius: 4px; font-size: 0.8em;">'
                f'Deprecated since v{safe_dep_ver}</span>'
            )

        # FE-B-005 + FE-B-006: each result is a listbox option with a stable
        # id so future combobox wiring can set aria-activedescendant. The
        # emoji + text-twin pattern (server name spoken, 📦 hidden from SR)
        # — Léonie Watson 2023 recommendation.
        # SD-V-002: tonal surface ladder — card base #22223e (raised one
        # step from #1a1a2e page background) so cards lift in dark mode
        # without box-shadow. SD-V-004: tool name uses the JetBrains-Mono
        # fallback stack so identifiers like `comfy:comfy_generate` read as
        # runnable code. Hierarchy (name → description → metadata) works
        # in grayscale alone via weight + size + colour-step.
        html_parts.append(f"""
        <li role="option" id="search-result-{idx_i}" aria-label="{safe_name_short}, {conf_label} match {confidence_pct} percent" style="border: 1px solid #3a3a52; border-radius: 8px; padding: 16px; margin: 12px 0; background: #22223e; list-style: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.05em; font-weight: 600; color: #4fc3f7; font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;" title="{safe_name}">{safe_name_short}</span>
                <span style="color: {confidence_color}; font-size: 0.95em;" aria-label="{conf_label} match {confidence_pct} percent" title="{conf_label} match ({confidence_pct}%)">{stars} {conf_label} ({confidence_pct}%)</span>
            </div>
            <p style="margin: 12px 0 8px 0; color: #e8e8f0; line-height: 1.5; max-width: 75ch;" title="{safe_desc}">{safe_desc_short}</p>
            <div style="display: flex; gap: 16px; font-size: 0.875em; color: #a0a0b0; flex-wrap: wrap; align-items: center;">
                <span><span aria-hidden="true">📦</span> Server: {safe_server}</span>
                <span><span aria-hidden="true">🏷️</span> Category: {safe_category}</span>
                {deprecated_badge}
            </div>
        </li>
        """)

        json_results.append(
            {
                "tool": r.tool.name,
                "description": r.tool.description,
                "server": r.tool.server,
                "category": r.tool.category,
                "confidence": round(r.score, 3),
                "parameters": getattr(r.tool, "parameters", {}) or {},
                "degraded": degraded,
            }
        )

    html_parts.append("</ul>")
    return "".join(html_parts), json.dumps(json_results, indent=2)


def _inline_fallback_banner() -> str:
    """Render the single source-of-truth degraded-mode banner.

    FE-B-001 + FE-B-008 (Nielsen Heuristic #1): degraded-mode signals belong
    adjacent to the affected data, not in chrome. The Search tab top-level
    Ollama banner explains the global state; this banner explains "what you
    are looking at right now."
    """
    return (
        '<div role="status" aria-live="polite" '
        'style="border: 1px solid #ffb74d; border-radius: 8px; '
        'padding: 12px; margin: 8px 0; background: #2a2a1a;">'
        '<span aria-hidden="true">⚠️</span> '
        '<strong style="color: #ffb74d;">Showing keyword-based results.</strong> '
        '<span style="color: #d4d4d4;">'
        'Semantic search is unavailable (Ollama unreachable). '
        'Start Ollama and re-run the search for better matches.'
        '</span>'
        '</div>'
    )


def _nearest_matches(
    index,
    query: str,
    max_results: int = 3,
    cat_filter: Optional[str] = None,
    srv_filter: Optional[str] = None,
) -> List[Dict]:
    """Substring nearest-neighbor lookup for the zero-results state (FE-B-002).

    Runs against the same tools table the indexer materializes. The query is
    matched as a substring against name and description. Used when semantic
    search returns nothing — the NN/g zero-results pattern is "never a dead
    end" (Whitenton 2018). Same shape as `_lexical_fallback_for_ui` so the
    rendering layer can treat both uniformly.
    """
    try:
        return _lexical_fallback_for_ui(
            index, query, max_results, cat_filter, srv_filter
        )
    except Exception:
        return []


def _render_empty_index() -> str:
    """Build the 'no tools indexed yet → sync' card (FE-SB-002 + FE-A-008).

    Single source of truth for the empty-index state shared by the Browser
    tab (`filter_tools`) and Search (`search_tools`). A synced-but-empty
    index must surface the actionable `tool-compass sync` next-step, NOT the
    generic no-match advice ("reword / lower confidence / remove filters"),
    which can never produce a tool from a zero-row index.
    """
    return """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">📦</div>
            <p style="color: #ffb74d;">No tools indexed yet.</p>
            <p style="font-size: 0.9em; color: #e8e8f0;">Build the index first:</p>
            <code style="color: #d4d4dc; font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;">tool-compass sync</code>
        </div>
        """


def _render_no_results(query: str, suggestions: List[Dict]) -> str:
    """Build the zero-results HTML, with optional 'did you mean' chips.

    FE-B-002 + Nielsen #9: always offer next-step actions. Empty-results is
    an error-adjacent state from the user's perspective — show closest
    matches as actionable links instead of just suggesting they try harder.
    """
    # SD-V-001/004: bumped body grey to #e8e8f0; mono font for suggested
    # tool names so they read as runnable identifiers.
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, monospace"
    safe_q = html.escape(truncate_text(query, 50), quote=True)
    parts = [
        '<div style="text-align: center; padding: 48px 24px; color: #a0a0b0; max-width: 75ch; margin: 0 auto;">',
        '<div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔎</div>',
        f'<p style="color: #ffb74d;">No tools found matching "{safe_q}"</p>',
    ]
    if suggestions:
        parts.append(
            '<p style="font-size: 0.95em; color: #e8e8f0; margin-top: 16px;">'
            'Did you mean one of these?</p>'
        )
        parts.append(
            '<ul role="list" style="text-align: left; display: inline-block; '
            'list-style: none; padding: 0; margin: 8px 0;">'
        )
        for s in suggestions:
            safe_name = html.escape(s["tool"], quote=True)
            safe_desc = html.escape(
                truncate_text(s.get("description") or "", 80), quote=True
            )
            parts.append(
                f'<li style="margin: 8px 0; color: #e8e8f0; line-height: 1.5;">'
                f'<code style="color: #4fc3f7; font-family: {mono};">{safe_name}</code>'
                f' — <span style="color: #a0a0b0;">{safe_desc}</span>'
                f'</li>'
            )
        parts.append('</ul>')
    parts.extend([
        '<p style="font-size: 0.9em; color: #a0a0b0; margin-top: 16px;">'
        'Or try:</p>',
        '<ul style="text-align: left; display: inline-block; color: #a0a0b0; line-height: 1.5;">',
        '<li>Broader or simpler terms</li>',
        '<li>Lowering the confidence threshold</li>',
        '<li>Removing the server or category filter</li>',
        '</ul>',
        '</div>',
    ])
    return "".join(parts)


def search_chains(query: str, top_k: int = 5, min_confidence: float = 0.3) -> str:
    """Search for tool chains/workflows."""
    # Empty query
    if not query.strip():
        return """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔗</div>
            <p style="color: #e8e8f0; line-height: 1.5;">Enter a query to search for workflows.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Try: "modify a file", "commit changes", "generate and save image"</p>
        </div>
        """

    # Sanitize input
    query = sanitize_query(query)
    if not query:
        return "<p style='color: orange;'>Please enter a valid search query.</p>"

    chain_indexer = get_chain_indexer_instance()
    if not chain_indexer:
        return """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">⚙️</div>
            <p style="color: #ffb74d;">Chain indexing is disabled in configuration.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Enable it in compass_config.json to use workflow search.</p>
        </div>
        """

    try:

        async def do_search():
            return await chain_indexer.search_chains(
                query, top_k=int(top_k), min_confidence=min_confidence
            )

        results = run_async(do_search())
    except Exception as e:
        # FE-A2-001: escape query at the call boundary in addition to
        # format_error's own escape, so the raw query is never carried as
        # live HTML through any intermediate string.
        return format_error(e, f"Workflow search failed for: {html.escape(query, quote=True)}")

    # No results
    if not results:
        return f"""
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔎</div>
            <p style="color: #ffb74d;">No workflows found matching "{html.escape(truncate_text(query, 50), quote=True)}"</p>
            <p style="font-size: 0.9em; color: #e8e8f0;">Workflows are auto-detected from usage patterns.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Use tools together to create workflows.</p>
        </div>
        """

    html_parts = [
        f'<p style="color: #a0a0b0; margin-bottom: 12px;">Found {len(results)} workflow{"s" if len(results) != 1 else ""}</p>'
    ]

    for cr in results:
        confidence_pct = int(cr.score * 100)
        conf_label = confidence_label(cr.score)
        # MCC-B-008: higher-contrast badge palette.
        confidence_color = (
            "#a5d6a7" if cr.score > 0.7 else "#ffcc80" if cr.score > 0.5 else "#e0e0e0"
        )
        tool_flow = " → ".join([t.split(":")[-1] for t in cr.chain.tools])
        safe_chain_name = html.escape(cr.chain.name, quote=True)
        safe_chain_name_short = html.escape(truncate_text(cr.chain.name, 40), quote=True)
        safe_flow = html.escape(tool_flow, quote=True)
        safe_flow_short = html.escape(truncate_text(tool_flow, 80), quote=True)
        safe_desc = html.escape(truncate_text(cr.chain.description or "", 100), quote=True)

        # FE-B-005: text-twin emoji.
        # SD-V-002: workflow card on the raised tonal step. SD-V-004:
        # the chain flow (tool → tool → tool) uses the JetBrains-Mono
        # fallback stack — it's the chain stage diagram so it should read
        # as code identifiers, not body prose.
        if cr.chain.is_auto_detected:
            badge_html = '<span aria-hidden="true">🤖</span> Auto-detected'
        else:
            badge_html = '<span aria-hidden="true">👤</span> Manual'
        html_parts.append(f"""
        <div role="article" aria-label="{safe_chain_name_short} workflow result" style="border: 1px solid #3a3a52; border-radius: 8px; padding: 16px; margin: 12px 0; background: #22223e;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.05em; font-weight: 600; color: #81c784; font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;" title="{safe_chain_name}">{safe_chain_name_short}</span>
                <span style="color: {confidence_color}; font-size: 0.95em;" aria-label="{conf_label} match {confidence_pct} percent" title="{conf_label} match ({confidence_pct}%)">{conf_label} ({confidence_pct}%)</span>
            </div>
            <p style="margin: 12px 0 8px 0; color: #e8e8f0; font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace; line-height: 1.5;" title="{safe_flow}">{safe_flow_short}</p>
            <p style="margin: 4px 0; color: #a0a0b0; font-size: 0.9em; line-height: 1.5; max-width: 75ch;">{safe_desc}</p>
            <div style="font-size: 0.875em; color: #a0a0b0; margin-top: 8px;">
                Used {cr.chain.use_count} times &middot; {badge_html}
            </div>
        </div>
        """)

    return "".join(html_parts)


# =============================================================================
# BROWSER FUNCTIONS
# =============================================================================


def get_all_tools() -> List[Dict]:
    """Get all indexed tools."""
    try:
        index = get_index()
        if not index.db:
            return []

        cursor = index.db.execute("""
            SELECT name, description, category, server, parameters, examples
            FROM tools ORDER BY server, category, name
        """)

        tools = []
        for row in cursor.fetchall():
            tools.append(
                {
                    "name": row["name"],
                    "description": row["description"],
                    "category": row["category"],
                    "server": row["server"],
                    "parameters": json.loads(row["parameters"])
                    if row["parameters"]
                    else {},
                    "examples": json.loads(row["examples"]) if row["examples"] else [],
                }
            )

        return tools
    except Exception as e:
        logger.error(f"Failed to get tools: {e}")
        return []


def filter_tools(server: str, category: str, search_text: str) -> str:
    """Filter and display tools in browser."""
    try:
        tools = get_all_tools()
    except Exception as e:
        return format_error(e, "Could not load tools from index")

    # Empty index — FE-A-008 + SD-V-001: v2.2 canonical CLI + contrast-safe
    # greys + mono command snippet. Shared with search_tools via the
    # _render_empty_index() single source of truth (FE-SB-002).
    if not tools:
        return _render_empty_index()

    # Sanitize search text
    if search_text:
        search_text = sanitize_query(search_text)

    # Apply filters
    if server != "All":
        tools = [t for t in tools if t["server"] == server]
    if category != "All":
        tools = [t for t in tools if t["category"] == category]
    if search_text.strip():
        search_lower = search_text.lower()
        tools = [
            t
            for t in tools
            if search_lower in t["name"].lower()
            or search_lower in t["description"].lower()
        ]

    # No matches after filtering
    if not tools:
        return """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔎</div>
            <p style="color: #ffb74d;">No tools match the current filters.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Try removing filters or using different search terms.</p>
        </div>
        """

    # Group by server
    by_server = {}
    for t in tools:
        by_server.setdefault(t["server"], []).append(t)

    html_parts = [
        f'<p style="color: #a0a0b0; margin-bottom: 12px;">Showing {len(tools)} tool{"s" if len(tools) != 1 else ""}</p>'
    ]

    # SD-V-002 + SD-V-004: browser rows use the raised tonal step
    # (#22223e) and the JetBrains-Mono fallback stack on tool names so the
    # eye locks onto the identifier first.
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, monospace"
    for server_name, server_tools in sorted(by_server.items()):
        html_parts.append(f"""
        <details open style="margin: 12px 0;">
            <summary style="cursor: pointer; font-size: 1.05em; font-weight: 600; color: #64b5f6; padding: 8px 0;">
                <span aria-hidden="true">📦</span> {html.escape(server_name, quote=True)} ({len(server_tools)} tool{"s" if len(server_tools) != 1 else ""})
            </summary>
            <div style="padding-left: 16px;">
        """)

        for t in server_tools:
            param_count = len(t["parameters"])
            desc_truncated = truncate_text(t["description"] or "", 120)
            safe_name = html.escape(t["name"], quote=True)
            safe_name_short = html.escape(truncate_text(t["name"], 45), quote=True)
            safe_desc = html.escape(t["description"] or "", quote=True)
            safe_desc_short = html.escape(desc_truncated, quote=True)
            safe_category = html.escape(t["category"], quote=True)
            html_parts.append(f"""
            <div style="border-left: 3px solid #3a3a52; padding: 12px 16px; margin: 8px 0; background: #22223e; border-radius: 4px;">
                <div style="font-weight: 600; color: #4fc3f7; font-family: {mono};" title="{safe_name}">{safe_name_short}</div>
                <div style="color: #e8e8f0; font-size: 0.9em; margin: 8px 0; line-height: 1.5; max-width: 75ch;" title="{safe_desc}">{safe_desc_short}</div>
                <div style="color: #a0a0b0; font-size: 0.85em;">
                    <span aria-hidden="true">🏷️</span> {safe_category} | <span aria-hidden="true">📝</span> {param_count} param{"s" if param_count != 1 else ""}
                </div>
            </div>
            """)

        html_parts.append("</div></details>")

    return "".join(html_parts)


def get_tool_details(tool_name: str) -> str:
    """Get detailed view of a single tool."""
    # Empty input
    if not tool_name.strip():
        return """
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔎</div>
            <p style="color: #e8e8f0; line-height: 1.5;">Enter a tool name to view details.</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Or click on a tool from the browser above.</p>
        </div>
        """

    # Sanitize input
    tool_name = sanitize_query(tool_name)
    if not tool_name:
        return "<p style='color: orange;'>Please enter a valid tool name.</p>"

    try:
        index = get_index()
        if not index.db:
            return format_error(
                RuntimeError("Index not loaded"), "Could not access tool index"
            )
    except Exception as e:
        return format_error(e, "Could not load tool index")

    try:
        cursor = index.db.execute(
            """
            SELECT name, description, category, server, parameters, examples
            FROM tools WHERE name = ?
        """,
            (tool_name,),
        )

        row = cursor.fetchone()
        if not row:
            # Try partial match
            cursor = index.db.execute(
                """
                SELECT name, description, category, server, parameters, examples
                FROM tools WHERE name LIKE ?
                LIMIT 1
            """,
                (f"%{tool_name}%",),
            )
            row = cursor.fetchone()
    except Exception as e:
        # FE-A2-001: tool_name is user-controlled (input box). Escape at the
        # boundary even though format_error escapes context too.
        return format_error(e, f"Could not search for tool: {html.escape(tool_name, quote=True)}")

    # Tool not found
    if not row:
        return f"""
        <div style="text-align: center; padding: 48px 24px; color: #a0a0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">❓</div>
            <p style="color: #ffb74d;">Tool not found: "{html.escape(truncate_text(tool_name, 40), quote=True)}"</p>
            <p style="font-size: 0.9em; color: #a0a0b0;">Check the tool name and try again.</p>
        </div>
        """

    # FE-SA-001: mirror get_all_tools — a malformed JSON blob from the index
    # must degrade to empty defaults, not raise an uncaught JSONDecodeError out
    # of the Gradio callback (which would dump a raw traceback under
    # show_error=True instead of the styled format_error card).
    try:
        params = json.loads(row["parameters"]) if row["parameters"] else {}
    except json.JSONDecodeError:
        params = {}
    try:
        examples = json.loads(row["examples"]) if row["examples"] else []
    except json.JSONDecodeError:
        examples = []

    # Build parameters table — all untrusted strings run through html.escape to
    # block HTML/script injection from malicious tool metadata. SD-V-004:
    # parameter names use the JetBrains-Mono fallback stack — they are
    # identifiers, not prose. SD-V-001: body grey lifted to #e8e8f0.
    mono = "'JetBrains Mono', 'SF Mono', ui-monospace, monospace"
    params_html = ""
    if params:
        params_html = f"""
        <h4 style="color: #81c784; margin-top: 24px;">Parameters ({len(params)})</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #2a2a48;">
                <th style="padding: 12px; text-align: left; border: 1px solid #3a3a52;">Name</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #3a3a52;">Type</th>
            </tr>
        """
        for name, ptype in params.items():
            params_html += f"""
            <tr>
                <td style="padding: 12px; border: 1px solid #3a3a52; font-family: {mono}; color: #4fc3f7;">{html.escape(truncate_text(name, 30), quote=True)}</td>
                <td style="padding: 12px; border: 1px solid #3a3a52; color: #e8e8f0; font-family: {mono};">{html.escape(truncate_text(str(ptype), 50), quote=True)}</td>
            </tr>
            """
        params_html += "</table>"
    else:
        params_html = """
        <h4 style="color: #81c784; margin-top: 24px;">Parameters</h4>
        <p style="color: #a0a0b0; font-style: italic;">No parameters required</p>
        """

    # Build examples
    examples_html = ""
    if examples:
        examples_html = f"<h4 style='color: #81c784; margin-top: 24px;'>Examples ({len(examples)})</h4>"
        for ex in examples:
            examples_html += f"<pre style='background: #22223e; color: #e8e8f0; padding: 16px; border-radius: 4px; overflow-x: auto; font-family: {mono}; line-height: 1.5;'>{html.escape(truncate_text(ex, 200), quote=True)}</pre>"

    return f"""
    <div style="padding: 16px;">
        <h2 style="color: #4fc3f7; margin: 0; word-break: break-all; font-family: {mono};">{html.escape(row["name"], quote=True)}</h2>
        <div style="color: #a0a0b0; margin: 8px 0; font-size: 0.9em;">
            <span aria-hidden="true">📦</span> {html.escape(row["server"], quote=True)} | <span aria-hidden="true">🏷️</span> {html.escape(row["category"], quote=True)}
        </div>
        <p style="color: #e8e8f0; font-size: 1.05em; margin: 16px 0; line-height: 1.5; max-width: 75ch;">{html.escape(row["description"] or "", quote=True)}</p>
        {params_html}
        {examples_html}
    </div>
    """


# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================


def get_analytics_dashboard(timeframe: str = "24h") -> str:
    """Get analytics summary as HTML and chart data."""
    try:
        analytics = get_analytics_instance()

        async def get_summary():
            return await analytics.get_analytics_summary(timeframe)

        summary = run_async(get_summary())
    except Exception as e:
        return format_error(e, "Could not load analytics data")

    # Build HTML dashboard
    searches = summary["searches"]
    calls = summary["tool_calls"]

    # UX-C-002: first-run empty state. On a fresh install both totals are 0,
    # so the four "0 / 0 / 0% / 0ms" stat cards read as broken. Replace them
    # with a centered empty-state card (same style used by Search/Chains
    # empty states) that tells the user how to populate the dashboard.
    if searches["total"] == 0 and calls["total"] == 0:
        return """
    <div style="text-align: center; padding: 40px; color: #a0a0b0;">
        <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">📊</div>
        <p style="color: #e8e8f0;">No activity recorded yet — run some searches and tool calls, then refresh to see usage analytics.</p>
    </div>
    """

    # Local var named `out` (not `html`) to avoid shadowing the html module.
    out = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
        <div style="background: #1a2e3a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #4fc3f7;">{searches["total"]}</div>
            <div style="color: #a0a0b0;">Searches ({timeframe})</div>
        </div>
        <div style="background: #1a3a2a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #81c784;">{calls["total"]}</div>
            <div style="color: #a0a0b0;">Tool Calls</div>
        </div>
        <div style="background: #3a2a1a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #ffb74d;">{calls["success_rate"]}%</div>
            <div style="color: #a0a0b0;">Success Rate</div>
        </div>
        <div style="background: #2a2a3a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #ba68c8;">{searches["avg_latency_ms"]}ms</div>
            <div style="color: #a0a0b0;">Avg Search Latency</div>
        </div>
    </div>
    """

    # Top tools
    if calls["top_tools"]:
        out += """
        <h3 style="color: #4fc3f7;">Top Tools</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px;">
            <tr style="background: #2a2a4a;">
                <th style="padding: 8px; text-align: left; border: 1px solid #3a3a52;">Tool</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #3a3a52;">Calls</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #3a3a52;">Success</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #3a3a52;">Latency</th>
            </tr>
        """
        for t in calls["top_tools"][:10]:
            out += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #3a3a52; color: #4fc3f7;">{html.escape(t["tool"], quote=True)}</td>
                <td style="padding: 8px; border: 1px solid #3a3a52; text-align: right;">{t["calls"]}</td>
                <td style="padding: 8px; border: 1px solid #3a3a52; text-align: right; color: {"#81c784" if t["success_rate"] > 90 else "#ffb74d"};">{t["success_rate"]}%</td>
                <td style="padding: 8px; border: 1px solid #3a3a52; text-align: right; color: #a0a0b0;">{t["avg_latency_ms"]}ms</td>
            </tr>
            """
        out += "</table>"

    # Top queries
    if searches["top_queries"]:
        out += """
        <h3 style="color: #81c784;">Top Queries</h3>
        <ul style="color: #e8e8f0;">
        """
        for q in searches["top_queries"][:10]:
            out += f'<li>"{html.escape(q["query"], quote=True)}" <span style="color: #a0a0b0;">({q["count"]} times)</span></li>'
        out += "</ul>"

    # Failures
    if summary.get("failures"):
        out += """
        <h3 style="color: #ef5350;">Recent Failures</h3>
        <ul style="color: #e8e8f0;">
        """
        for f in summary["failures"][:5]:
            out += f'<li style="color: #ef5350;">{html.escape(f["tool"], quote=True)}: {html.escape(f["error"] or "Unknown error", quote=True)} ({f["count"]}x)</li>'
        out += "</ul>"

    # Hot cache
    hot_cache = summary.get("hot_cache", {})
    if hot_cache.get("tools"):
        out += f"""
        <h3 style="color: #ba68c8;">Hot Cache ({hot_cache["size"]} tools)</h3>
        <p style="color: #a0a0b0; font-family: monospace;">{html.escape(", ".join(hot_cache["tools"]), quote=True)}</p>
        """

    return out


# =============================================================================
# CHAIN VIEWER FUNCTIONS
# =============================================================================


def get_chains_view() -> str:
    """Display all tool chains."""
    chain_indexer = get_chain_indexer_instance()
    if not chain_indexer:
        return """
        <div style="text-align: center; padding: 40px; color: #b0b0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">⚙️</div>
            <p style="color: #ffb74d;">Chain indexing is disabled in configuration.</p>
            <p style="font-size: 0.9em;">Enable <code>chain_indexing_enabled</code> in compass_config.json to use workflows.</p>
        </div>
        """

    try:

        async def load_chains():
            return await chain_indexer.load_chains_from_db()

        chains = run_async(load_chains())
    except Exception as e:
        return format_error(e, "Could not load workflows")

    # No workflows
    if not chains:
        return """
        <div style="text-align: center; padding: 40px; color: #b0b0b0;">
            <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔗</div>
            <p>No workflows defined yet.</p>
            <p style="font-size: 0.9em; color: #b0b0b0;">Workflows are auto-detected from usage patterns.</p>
            <p style="font-size: 0.9em; color: #b0b0b0;">Use tools together to create workflows.</p>
        </div>
        """

    html_parts = [
        f'<p style="color: #b0b0b0; margin-bottom: 12px;">{len(chains)} workflow{"s" if len(chains) != 1 else ""} available</p>'
    ]

    for chain in sorted(chains, key=lambda c: c.use_count, reverse=True):
        tool_flow = " → ".join([t.split(":")[-1] for t in chain.tools])
        # FE-B-005: text-twin emoji pattern — sighted readers see the glyph,
        # SR readers hear the meaningful word "Auto-detected" / "Manual" and
        # don't have to decode "robot face" / "bust in silhouette".
        if chain.is_auto_detected:
            badge = '<span aria-hidden="true">🤖</span> Auto-detected'
        else:
            badge = '<span aria-hidden="true">👤</span> Manual'

        safe_chain_name = html.escape(chain.name, quote=True)
        safe_chain_name_short = html.escape(truncate_text(chain.name, 35), quote=True)
        safe_flow = html.escape(tool_flow, quote=True)
        safe_flow_short = html.escape(truncate_text(tool_flow, 80), quote=True)
        safe_desc = html.escape(truncate_text(chain.description or "", 120), quote=True)

        # UX-D-004: card palette aligned to the refreshed tonal ladder
        # (SD-V-002/004) — border #3a3a52, background #22223e, primary body
        # #e8e8f0, secondary grey #a0a0b0 — so Chains cards match Search/Browser.
        html_parts.append(f"""
        <div style="border: 1px solid #3a3a52; border-radius: 8px; padding: 16px; margin: 12px 0; background: #22223e;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.2em; font-weight: bold; color: #81c784;" title="{safe_chain_name}">{safe_chain_name_short}</span>
                <span style="color: #a0a0b0; font-size: 0.9em;">{badge}</span>
            </div>
            <div style="font-family: monospace; color: #4fc3f7; margin: 12px 0; font-size: 1.1em;" title="{safe_flow}">
                {safe_flow_short}
            </div>
            <p style="color: #e8e8f0; margin: 8px 0;">{safe_desc}</p>
            <div style="color: #a0a0b0; font-size: 0.9em;">
                Used {chain.use_count} time{"s" if chain.use_count != 1 else ""}
            </div>
        </div>
        """)

    return "".join(html_parts)


# =============================================================================
# SYSTEM STATUS
# =============================================================================


def get_system_status() -> str:
    """Get system status overview.

    FE-A-014 + FE-B-005: status indicators use the text-twin pattern — emoji
    is aria-hidden, the status word is the semantic carrier. All exception
    strings are html.escape'd before interpolation (truncate_text alone does
    not escape).
    """
    # Load config first (doesn't require index)
    global _config
    if _config is None:
        try:
            _config = load_config()
        except Exception as e:
            return format_error(e, "Could not load configuration")

    # SD-V-005: status indicators wrap emoji + label so SR users hear the
    # text-twin (Léonie Watson 2023) AND sighted users get the colour cue.
    # Label is the carrier; emoji is aria-hidden chrome.
    index_status = '<span aria-hidden="true">✅</span> <span>Passed</span> &mdash; Loaded'
    stats = {}
    index_path = "Unknown"
    try:
        index = get_index()
        stats = index.get_stats()
        index_path = str(index.index_path)
    except Exception as e:
        safe_err = html.escape(truncate_text(str(e), 50), quote=True)
        index_status = f'<span aria-hidden="true">⚠️</span> <span>Warning</span> &mdash; Not loaded: {safe_err}'

    # Check analytics — FE-A-013 carry-forward: private attr access remains
    # (analytics._hot_cache); flagged in skipped[] for backend domain.
    analytics_status = '<span aria-hidden="true">✅</span> <span>Passed</span> &mdash; Available'
    hot_cache_size = 0
    try:
        analytics = get_analytics_instance()
        hot_cache_size = len(analytics._hot_cache)
    except Exception as e:
        safe_err = html.escape(truncate_text(str(e), 50), quote=True)
        analytics_status = f'<span aria-hidden="true">⚠️</span> <span>Warning</span> &mdash; {safe_err}'

    # Check Ollama
    ollama_status = '<span aria-hidden="true">❓</span> <span>Unknown</span> &mdash; Not checked'
    try:
        from embedder import Embedder

        # FE-SB-001: probe the configured endpoint + model so the Status tab
        # agrees with _check_ollama_banner() (which reads cfg). A bare
        # Embedder() uses the hardcoded module defaults and would report the
        # WRONG endpoint's health for any non-default config.
        cfg = _config if _config is not None else load_config()
        embedder = Embedder(
            base_url=getattr(cfg, "ollama_url", "http://localhost:11434"),
            model=getattr(cfg, "embedding_model", "nomic-embed-text"),
        )
        # FE-SB-003: close() must run even when health_check() raises (Ollama
        # down is the common path) or the aiohttp session leaks.
        try:
            is_healthy = run_async(embedder.health_check())
            if is_healthy:
                ollama_status = '<span aria-hidden="true">✅</span> <span>Passed</span> &mdash; Connected'
            else:
                ollama_status = '<span aria-hidden="true">⚠️</span> <span>Warning</span> &mdash; Model not loaded'
        finally:
            run_async(embedder.close())
    except Exception as e:
        safe_err = html.escape(truncate_text(str(e), 40), quote=True)
        ollama_status = f'<span aria-hidden="true">❌</span> <span>Failed</span> &mdash; Unavailable: {safe_err}'

    safe_embedding_model = html.escape(str(_config.embedding_model), quote=True)

    out = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;">
        <div>
            <h3 style="color: #4fc3f7;">System Health</h3>
            <ul style="color: #e8e8f0; list-style: none; padding-left: 0;">
                <li style="margin: 8px 0;">Index: {index_status}</li>
                <li style="margin: 8px 0;">Analytics: {analytics_status}</li>
                <li style="margin: 8px 0;">Ollama: {ollama_status}</li>
            </ul>

            <h3 style="color: #4fc3f7;">Index Status</h3>
            <ul style="color: #e8e8f0;">
                <li>Total tools: <strong>{stats.get("total_tools", 0)}</strong></li>
                <li>Core tools: {stats.get("core_tools", 0)}</li>
                <li>Index path: <code style="font-size: 0.85em;">{truncate_text(index_path, 40)}</code></li>
            </ul>

            <h4 style="color: #81c784;">By Server</h4>
    """

    if stats.get("by_server"):
        out += "<ul style='color: #e8e8f0;'>"
        for server, count in sorted(stats.get("by_server", {}).items()):
            safe_server = html.escape(str(server), quote=True)
            out += f"<li>{safe_server}: {count}</li>"
        out += "</ul>"
    else:
        out += "<p style='color: #a0a0b0; font-style: italic;'>No data</p>"

    out += "<h4 style='color: #81c784;'>By Category</h4>"

    if stats.get("by_category"):
        out += "<ul style='color: #e8e8f0;'>"
        for category, count in sorted(stats.get("by_category", {}).items()):
            safe_category = html.escape(str(category), quote=True)
            out += f"<li>{safe_category}: {count}</li>"
        out += "</ul>"
    else:
        out += "<p style='color: #a0a0b0; font-style: italic;'>No data</p>"

    out += f"""
        </div>

        <div>
            <h3 style="color: #4fc3f7;">Configuration</h3>
            <ul style="color: #e8e8f0;">
                <li>Progressive disclosure: {"✅" if _config.progressive_disclosure else "❌"}</li>
                <li>Auto sync: {"✅" if _config.auto_sync else "❌"}</li>
                <li>Analytics: {"✅" if _config.analytics_enabled else "❌"}</li>
                <li>Chain indexing: {"✅" if _config.chain_indexing_enabled else "❌"}</li>
                <li>Embedding model: <code>{safe_embedding_model}</code></li>
                <li>Hot cache: {hot_cache_size}/{_config.hot_cache_size}</li>
            </ul>

            <h3 style="color: #4fc3f7;">Backends ({len(_config.backends)})</h3>
            <ul style="color: #e8e8f0;">
    """

    for name in _config.backends.keys():
        safe_name = html.escape(str(name), quote=True)
        out += f"<li>{safe_name}</li>"

    out += """
            </ul>

            <h3 style="color: #4fc3f7;">Quick Commands</h3>
            <div style="font-size: 0.9em; color: #a0a0b0;">
                <p style="margin: 4px 0;"><code>tool-compass sync</code> &mdash; Rebuild index</p>
                <p style="margin: 4px 0;"><code>tool-compass doctor</code> &mdash; Print diagnostic info</p>
                <p style="margin: 4px 0;"><code>ollama serve</code> &mdash; Start Ollama</p>
            </div>
        </div>
    </div>
    """

    return out


# =============================================================================
# BUILD UI
# =============================================================================


def get_filter_choices():
    """Get choices for filter dropdowns."""
    try:
        index = get_index()
        stats = index.get_stats()

        servers = ["All"] + sorted(stats.get("by_server", {}).keys())
        categories = ["All"] + sorted(stats.get("by_category", {}).keys())

        return servers, categories
    except:
        return ["All"], ["All"]


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""

    servers, categories = get_filter_choices()

    with gr.Blocks(
        title="Tool Compass",
        # VIS-D-001/002: the entire UI is hand-coded for a dark page (#1a1a2e
        # surface, #22223e cards, light #e8e8f0 text). gr.themes.Soft() defaults
        # to LIGHT, so backgroundless state blocks (empty/no-result/status views)
        # were near-white text on a white panel (~1.2:1, invisible) out of the
        # box. Pin the dark surface the palette assumes — on BOTH the light and
        # _dark token variants — so the page, blocks, inputs and the gr.Code box
        # all track the same dark chrome as the custom cards (no half-light seam).
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ).set(
            body_background_fill="#1a1a2e",
            body_background_fill_dark="#1a1a2e",
            body_text_color="#e8e8f0",
            body_text_color_dark="#e8e8f0",
            background_fill_primary="#22223e",
            background_fill_primary_dark="#22223e",
            background_fill_secondary="#1a1a2e",
            background_fill_secondary_dark="#1a1a2e",
            block_background_fill="#22223e",
            block_background_fill_dark="#22223e",
            input_background_fill="#2a2a44",
            input_background_fill_dark="#2a2a44",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .tool-result { border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; }
        """,
    ) as demo:
        # FE-B-016: pull the count from the indexer's O(1) COUNT(*) via
        # get_stats() instead of materializing every row + N json.loads.
        # ~1000× cheaper at the 1000-tool scale (same correctness).
        try:
            _tool_count = get_index().get_stats().get("total_tools", 0)
        except Exception:
            _tool_count = 0
        gr.Markdown(f"""
        # 🧭 Tool Compass
        **Semantic search across {_tool_count} MCP tools** | Progressive discovery: Search → Describe → Execute
        """)

        with gr.Tabs():
            # =================================================================
            # SEARCH TAB
            # =================================================================
            with gr.Tab("🔍 Search", id="search"):
                gr.Markdown(
                    "Search tools using natural language. Describe what you want to do."
                )

                # MCC-B-007 + FE-B-014: Ollama-down banner. The button is the
                # explicit recovery affordance — verb-noun form ("Re-check
                # Ollama") makes the action unambiguous (Krug 'Don't Make Me
                # Think' 2014). search_tools() also surfaces an inline
                # fallback banner adjacent to result cards when this banner
                # is non-empty (FE-B-008) — the chrome banner explains the
                # global state, the inline banner explains the current
                # results region.
                ollama_banner = gr.Markdown(value=_check_ollama_banner())
                refresh_status_btn = gr.Button(
                    "Re-check Ollama", size="sm", variant="secondary"
                )
                refresh_status_btn.click(
                    fn=_check_ollama_banner,
                    inputs=[],
                    outputs=[ollama_banner],
                    show_progress="minimal",
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        # FE-B-006: elem_id makes the underlying <input>
                        # addressable from the page-load JS that wires
                        # role="combobox" + aria-controls="search-results-list"
                        # at runtime. Default gr.Textbox is a plain input;
                        # the JS upgrade lands the W3C APG combobox role
                        # without forking the Gradio widget.
                        search_input = gr.Textbox(
                            label="What do you want to do?",
                            placeholder="e.g., 'generate an image with AI', 'read a file', 'search documents'",
                            lines=1,
                            elem_id="tc-search-input",
                        )
                    with gr.Column(scale=1):
                        search_btn = gr.Button(
                            "Search", variant="primary", elem_id="tc-search-btn"
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        server_filter = gr.Dropdown(
                            choices=servers, value="All", label="Server"
                        )
                    with gr.Column(scale=1):
                        category_filter = gr.Dropdown(
                            choices=categories, value="All", label="Category"
                        )
                    with gr.Column(scale=1):
                        top_k = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1, label="Results"
                        )
                    with gr.Column(scale=1):
                        min_conf = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Min Confidence",
                        )

                with gr.Row():
                    with gr.Column(scale=2):
                        # FE-B-004 + FE-B-006: the results region exists in
                        # the DOM at first render with role="region" +
                        # aria-live="polite" so SR users see the live
                        # region BEFORE content arrives (MDN aria-live).
                        # The combobox/listbox wiring is finalized at
                        # render time by the bottom-of-page JS via
                        # elem_id="search-input" / id="search-results-list".
                        search_results = gr.HTML(
                            value="""
                            <div role="region" aria-live="polite"
                                 aria-label="Search results"
                                 id="search-results-region"
                                 style="text-align: center; padding: 40px; color: #b0b0b0;">
                                <div style="font-size: 2em; margin-bottom: 12px;"
                                     aria-hidden="true">🔍</div>
                                <p>Enter a search query above to find tools.</p>
                                <p style="font-size: 0.9em;">Try: "generate an image", "read a file", "search documents"</p>
                            </div>
                            """,
                            label="Results",
                            elem_id="search-results-region-wrap",
                        )
                    with gr.Column(scale=1):
                        results_json = gr.Code(label="JSON", language="json", lines=15)

                # Search for chains
                gr.Markdown("---")
                gr.Markdown("### 🔗 Workflow Search")

                with gr.Row():
                    chain_query = gr.Textbox(
                        label="Search workflows",
                        placeholder="e.g., 'modify a file', 'commit changes', 'generate and save image'",
                        lines=1,
                    )
                    chain_btn = gr.Button("Search Workflows")

                chain_results = gr.HTML(
                    value="""
                    <div style="text-align: center; padding: 40px; color: #b0b0b0;">
                        <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔗</div>
                        <p>Enter a query to search for workflows.</p>
                        <p style="font-size: 0.9em;">Try: "modify a file", "commit changes", "generate and save image"</p>
                    </div>
                    """
                )

                # Wire up search — FE-B-007: `show_progress="minimal"` so
                # users see a visible loading indicator during the embedder
                # call (cold-start can exceed 1s; NN/g "Response Times"
                # 1.0s = limit of user flow).
                search_btn.click(
                    fn=search_tools,
                    inputs=[
                        search_input,
                        top_k,
                        category_filter,
                        server_filter,
                        min_conf,
                    ],
                    outputs=[search_results, results_json],
                    show_progress="minimal",
                )
                search_input.submit(
                    fn=search_tools,
                    inputs=[
                        search_input,
                        top_k,
                        category_filter,
                        server_filter,
                        min_conf,
                    ],
                    outputs=[search_results, results_json],
                    show_progress="minimal",
                )
                chain_btn.click(
                    fn=search_chains,
                    inputs=[chain_query, top_k, min_conf],
                    outputs=[chain_results],
                    show_progress="minimal",
                )

            # =================================================================
            # BROWSER TAB
            # =================================================================
            with gr.Tab("📦 Browser", id="browser"):
                gr.Markdown("Browse all indexed tools by server and category.")

                with gr.Row():
                    with gr.Column(scale=1):
                        browser_server = gr.Dropdown(
                            choices=servers, value="All", label="Filter by Server"
                        )
                    with gr.Column(scale=1):
                        browser_category = gr.Dropdown(
                            choices=categories, value="All", label="Filter by Category"
                        )
                    with gr.Column(scale=2):
                        browser_search = gr.Textbox(
                            label="Search",
                            placeholder="Filter by name or description...",
                        )
                    with gr.Column(scale=1):
                        browser_btn = gr.Button("Filter", variant="primary")

                browser_results = gr.HTML(value=filter_tools("All", "All", ""))

                gr.Markdown("---")
                gr.Markdown("### 🔎 Tool Details")

                with gr.Row():
                    tool_name_input = gr.Textbox(
                        label="Tool Name",
                        placeholder="Enter tool name (e.g., bridge:read_file)",
                    )
                    detail_btn = gr.Button("View Details")

                tool_details = gr.HTML(
                    value="""
                    <div style="text-align: center; padding: 40px; color: #b0b0b0;">
                        <div style="font-size: 2em; margin-bottom: 12px;" aria-hidden="true">🔎</div>
                        <p>Enter a tool name to view details.</p>
                        <p style="font-size: 0.9em;">Or click on a tool from the browser above.</p>
                    </div>
                    """
                )

                # Wire up browser
                browser_btn.click(
                    fn=filter_tools,
                    inputs=[browser_server, browser_category, browser_search],
                    outputs=[browser_results],
                )
                detail_btn.click(
                    fn=get_tool_details,
                    inputs=[tool_name_input],
                    outputs=[tool_details],
                )

            # =================================================================
            # ANALYTICS TAB
            # =================================================================
            with gr.Tab("📊 Analytics", id="analytics"):
                gr.Markdown("Usage analytics and tool health metrics.")

                with gr.Row():
                    timeframe = gr.Dropdown(
                        choices=["1h", "24h", "7d", "30d"],
                        value="24h",
                        label="Timeframe",
                    )
                    refresh_btn = gr.Button("Refresh", variant="primary")

                analytics_html = gr.HTML(value=get_analytics_dashboard("24h"))

                refresh_btn.click(
                    fn=get_analytics_dashboard,
                    inputs=[timeframe],
                    outputs=[analytics_html],
                    show_progress="minimal",
                )

            # =================================================================
            # CHAINS TAB
            # =================================================================
            with gr.Tab("🔗 Workflows", id="chains"):
                gr.Markdown(
                    "Tool chains are multi-step workflows that combine tools. Auto-detected from usage patterns."
                )

                chains_btn = gr.Button("Refresh Workflows", variant="primary")
                chains_html = gr.HTML(value=get_chains_view())

                chains_btn.click(
                    fn=get_chains_view,
                    inputs=[],
                    outputs=[chains_html],
                    show_progress="minimal",
                )

            # =================================================================
            # STATUS TAB
            # =================================================================
            with gr.Tab("⚙️ Status", id="status"):
                gr.Markdown("System status and configuration.")

                status_btn = gr.Button("Refresh Status", variant="primary")
                status_html = gr.HTML(value=get_system_status())

                status_btn.click(
                    fn=get_system_status,
                    inputs=[],
                    outputs=[status_html],
                    show_progress="minimal",
                )

        # VIS-D-003: footer grey lifted to #b4b4c4 — #a0a0a0 on the #1a1a2e
        # surface is only ~4.0:1, which clears AA for large text (3:1) but NOT
        # for this body-scale copy (needs 4.5:1). #b4b4c4 on #1a1a2e is ~5.4:1,
        # a real AA pass. (Corrects the prior comment's overstated 4.05:1.)
        gr.Markdown(f"""
        ---
        <div style="text-align: center; color: #b4b4c4;">
            Tool Compass v{__version__} | Semantic tool discovery for MCP
        </div>
        """)

        # FE-B-003 + FE-B-004: enhance the Gradio-rendered tabs with the
        # W3C APG tablist roving-tabindex pattern AND focus the
        # search-results-count heading after async results land.
        # Gradio's default Tabs render as flex-of-buttons with no arrow-key
        # navigation and no aria-controls/tabpanel linkage — the JS below
        # patches both at runtime without forking gr.Tabs.
        gr.HTML(
            """
<script>
(function() {
    'use strict';

    // ----- FE-B-003: roving tabindex for the tab bar -----
    function enhanceTabs() {
        const tabBars = document.querySelectorAll('.tab-nav, [role="tablist"]');
        tabBars.forEach((bar) => {
            if (bar.dataset.tcEnhanced) return;
            const tabs = bar.querySelectorAll('button');
            if (!tabs.length) return;
            bar.setAttribute('role', 'tablist');
            tabs.forEach((tab, i) => {
                tab.setAttribute('role', 'tab');
                tab.setAttribute('tabindex', i === 0 ? '0' : '-1');
                if (!tab.getAttribute('aria-selected')) {
                    tab.setAttribute(
                        'aria-selected', i === 0 ? 'true' : 'false'
                    );
                }
                tab.addEventListener('keydown', (e) => {
                    let next = null;
                    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                        next = tabs[(i + 1) % tabs.length];
                    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                        next = tabs[(i - 1 + tabs.length) % tabs.length];
                    } else if (e.key === 'Home') {
                        next = tabs[0];
                    } else if (e.key === 'End') {
                        next = tabs[tabs.length - 1];
                    }
                    if (next) {
                        e.preventDefault();
                        next.focus();
                        next.click();
                    }
                });
                tab.addEventListener('click', () => {
                    tabs.forEach((t) => {
                        t.setAttribute('tabindex', '-1');
                        t.setAttribute('aria-selected', 'false');
                    });
                    tab.setAttribute('tabindex', '0');
                    tab.setAttribute('aria-selected', 'true');
                });
            });
            bar.dataset.tcEnhanced = '1';
        });
    }

    // ----- FE-B-004: focus search-results-count after async results land
    // The heading is rendered with id="search-results-count" tabindex="-1"
    // so it can receive programmatic focus without entering the tab order.
    function focusResultsCount() {
        const h = document.getElementById('search-results-count');
        if (h && document.activeElement !== h
              && !h.dataset.tcFocused) {
            // Mark so the MutationObserver doesn't keep stealing focus on
            // every subsequent unrelated DOM tick.
            h.dataset.tcFocused = '1';
            h.focus({ preventScroll: false });
        }
        // Unmark when the heading is replaced (next search). Gradio swaps
        // the whole innerHTML, so the dataset is gone with the old node.
    }

    // ----- FE-B-006: combobox + listbox wiring on the search input -----
    // FE-SB-004: aria-controls/aria-expanded must reflect whether the listbox
    // actually exists. The #search-results-list element is only rendered on
    // the populated-results path; before any search (and in the empty/
    // no-match states) the reference would dangle. syncComboboxState() runs
    // on every MutationObserver tick (Gradio swaps the results HTML) so the
    // attributes stay honest as the listbox appears/disappears.
    function syncComboboxState() {
        const wrapper = document.getElementById('tc-search-input');
        if (!wrapper) return;
        const input = wrapper.querySelector('input, textarea');
        if (!input) return;
        const hasList = !!document.getElementById('search-results-list');
        if (hasList) {
            input.setAttribute('aria-controls', 'search-results-list');
            input.setAttribute('aria-expanded', 'true');
        } else {
            input.removeAttribute('aria-controls');
            input.setAttribute('aria-expanded', 'false');
        }
    }

    function enhanceCombobox() {
        const wrapper = document.getElementById('tc-search-input');
        if (!wrapper) return;
        const input = wrapper.querySelector('input, textarea');
        if (!input) return;
        // Keep aria-controls/aria-expanded in sync on every tick even after
        // the one-time wiring below has run.
        syncComboboxState();
        if (input.dataset.tcCombobox) return;
        input.setAttribute('role', 'combobox');
        input.setAttribute('aria-autocomplete', 'list');
        input.dataset.tcCombobox = '1';
        // ArrowDown into the listbox focuses the first option, ArrowUp
        // focuses the last — matches W3C APG editable combobox.
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                const list = document.getElementById('search-results-list');
                if (!list) return;
                const opts = list.querySelectorAll('[role="option"]');
                if (!opts.length) return;
                e.preventDefault();
                const target = e.key === 'ArrowDown'
                    ? opts[0]
                    : opts[opts.length - 1];
                target.setAttribute('tabindex', '0');
                target.focus();
            }
        });
    }

    // MutationObserver wakes all three enhancements as Gradio swaps HTML.
    const target = document.body;
    const obs = new MutationObserver(() => {
        enhanceTabs();
        enhanceCombobox();
        focusResultsCount();
    });
    obs.observe(target, { childList: true, subtree: true });

    // Run once on load.
    enhanceTabs();
    enhanceCombobox();
})();
</script>
            """,
            visible=True,
        )

    return demo


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Tool Compass Gradio UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    print("🧭 Starting Tool Compass UI...")
    print(f"   Port: {args.port}")
    print(f"   Host: {args.host}")

    # Pre-load index
    try:
        index = get_index()
        stats = index.get_stats()
        print(f"   Tools indexed: {stats.get('total_tools', 0)}")
    except Exception as e:
        print(f"   Warning: Could not load index: {e}")
        print("   Run 'tool-compass sync' to build the index first.")

    # Gradio `share=True` publishes a public tunnel. Require basic auth via
    # GRADIO_AUTH=user:pass so the public URL is not wide open.
    auth = None
    if args.share:
        auth_env = os.environ.get("GRADIO_AUTH", "").strip()
        if not auth_env or ":" not in auth_env:
            print(
                "   ERROR: --share refused. Set GRADIO_AUTH='user:pass' to "
                "enable basic auth on the public tunnel.",
                file=sys.stderr,
            )
            sys.exit(2)
        user, _, pwd = auth_env.partition(":")
        if not user or not pwd:
            print(
                "   ERROR: GRADIO_AUTH must be 'user:pass' (non-empty on both sides).",
                file=sys.stderr,
            )
            sys.exit(2)
        auth = (user, pwd)
        print(
            "   WARNING: --share creates a PUBLIC tunnel. Basic auth is enabled "
            f"for user '{user}'. Anyone with the URL and credentials can access."
        )

    demo = create_ui()
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            auth=auth,
            show_error=True,
        )
    except OSError as e:
        # MCC-B-006: a port collision used to dump a bare traceback. Surface
        # an actionable one-liner and exit with the argparse-style usage code
        # so shell scripts can distinguish "user error" from "crash."
        msg = str(e).lower()
        if "address already in use" in msg or "eaddrinuse" in msg or e.errno in (
            48,  # macOS
            98,  # Linux
            10048,  # Windows
        ):
            print(
                f"Port {args.port} is already in use. Try:\n"
                f"  tool-compass ui --port {args.port + 1}\n"
                f"Or free the port and retry.",
                file=sys.stderr,
            )
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
