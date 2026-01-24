"""
Tool Compass - Gradio UI
Interactive web interface for semantic tool discovery, browsing, and analytics.

Usage:
    python ui.py              # Launch standalone UI on port 7860
    python ui.py --port 7861  # Custom port
    python ui.py --share      # Create public Gradio link
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict
import argparse

import gradio as gr

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
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new loop for nested async
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_index: Optional[CompassIndex] = None
_analytics: Optional[CompassAnalytics] = None
_chain_indexer: Optional[ChainIndexer] = None
_config = None


def get_index() -> CompassIndex:
    """Get or initialize compass index."""
    global _index
    if _index is None:
        _index = CompassIndex()
        if not _index.load_index():
            raise RuntimeError("Failed to load index. Run: python gateway.py --sync")
    return _index


def get_analytics_instance() -> CompassAnalytics:
    """Get or initialize analytics."""
    global _analytics
    if _analytics is None:
        _analytics = get_analytics()
        run_async(_analytics.load_hot_cache_from_db())
    return _analytics


def get_chain_indexer_instance() -> Optional[ChainIndexer]:
    """Get or initialize chain indexer."""
    global _chain_indexer, _config
    if _config is None:
        _config = load_config()

    if _chain_indexer is None and _config.chain_indexing_enabled:
        index = get_index()
        analytics = get_analytics_instance()
        _chain_indexer = get_chain_indexer(index.embedder, analytics)
        run_async(_chain_indexer.load_chain_index())

    return _chain_indexer


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


def format_error(error: Exception, context: str = "") -> str:
    """Format error message for user display."""
    error_type = type(error).__name__

    if "Connection" in error_type or "refused" in str(error).lower():
        return """
        <div style="border: 1px solid #ef5350; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a1a1a;">
            <div style="color: #ef5350; font-weight: bold;">⚠️ Service Unavailable</div>
            <p style="color: #ccc; margin: 8px 0;">
                Cannot connect to Ollama embeddings service. Please ensure Ollama is running.
            </p>
            <code style="color: #888; font-size: 0.85em;">ollama serve</code>
        </div>
        """
    elif "index" in str(error).lower() or "not loaded" in str(error).lower():
        return """
        <div style="border: 1px solid #ffb74d; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a2a1a;">
            <div style="color: #ffb74d; font-weight: bold;">⚠️ Index Not Ready</div>
            <p style="color: #ccc; margin: 8px 0;">
                Tool index not found. Please build the index first.
            </p>
            <code style="color: #888; font-size: 0.85em;">cd tool_compass && python gateway.py --sync</code>
        </div>
        """
    else:
        return f"""
        <div style="border: 1px solid #ef5350; border-radius: 8px; padding: 16px; margin: 8px 0; background: #2a1a1a;">
            <div style="color: #ef5350; font-weight: bold;">⚠️ Error</div>
            <p style="color: #ccc; margin: 8px 0;">{context or "An error occurred"}</p>
            <details style="color: #888; font-size: 0.85em;">
                <summary>Technical details</summary>
                <code>{error_type}: {str(error)[:200]}</code>
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
    """
    # Empty query
    if not query.strip():
        return (
            """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔍</div>
            <p>Enter a search query above to find tools.</p>
            <p style="font-size: 0.9em;">Try: "generate an image", "read a file", "search documents"</p>
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

    # Handle filter values
    cat_filter = None if category == "All" else category
    srv_filter = None if server == "All" else server

    # Run search with error handling
    try:

        async def do_search():
            return await index.search(
                query=query,
                top_k=int(top_k),
                category_filter=cat_filter,
                server_filter=srv_filter,
            )

        results = run_async(do_search())
    except Exception as e:
        return format_error(e, f"Search failed for: {query}"), "{}"

    # Filter by confidence
    results = [r for r in results if r.score >= min_confidence]

    # No results
    if not results:
        return (
            f"""
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔎</div>
            <p style="color: #ffb74d;">No tools found matching "{truncate_text(query, 50)}"</p>
            <p style="font-size: 0.9em;">Suggestions:</p>
            <ul style="text-align: left; display: inline-block; color: #aaa;">
                <li>Try broader or simpler terms</li>
                <li>Lower the confidence threshold</li>
                <li>Remove filters</li>
            </ul>
        </div>
        """,
            "{}",
        )

    # Build HTML output
    html_parts = [
        f'<p style="color: #888; margin-bottom: 12px;">Found {len(results)} tool{"s" if len(results) != 1 else ""}</p>'
    ]
    json_results = []

    for r in results:
        confidence_pct = int(r.score * 100)
        conf_label = confidence_label(r.score)
        confidence_color = (
            "#81c784" if r.score > 0.7 else "#ffb74d" if r.score > 0.5 else "#9e9e9e"
        )

        # Stars based on confidence
        stars = "★" * min(5, int(r.score * 5 + 0.5)) + "☆" * (
            5 - min(5, int(r.score * 5 + 0.5))
        )

        # Truncate long descriptions
        desc_display = truncate_text(r.tool.description, 150)

        html_parts.append(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; background: #1a1a2e;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.1em; font-weight: bold; color: #4fc3f7;" title="{r.tool.name}">{truncate_text(r.tool.name, 40)}</span>
                <span style="color: {confidence_color};" title="{conf_label} match ({confidence_pct}%)">{stars} {conf_label} ({confidence_pct}%)</span>
            </div>
            <p style="margin: 8px 0; color: #ccc;" title="{r.tool.description}">{desc_display}</p>
            <div style="display: flex; gap: 12px; font-size: 0.9em; color: #888; flex-wrap: wrap;">
                <span>📦 {r.tool.server}</span>
                <span>🏷️ {r.tool.category}</span>
            </div>
        </div>
        """)

        json_results.append(
            {
                "tool": r.tool.name,
                "description": r.tool.description,
                "server": r.tool.server,
                "category": r.tool.category,
                "confidence": round(r.score, 3),
                "parameters": r.tool.parameters,
            }
        )

    return "".join(html_parts), json.dumps(json_results, indent=2)


def search_chains(query: str, top_k: int = 5, min_confidence: float = 0.3) -> str:
    """Search for tool chains/workflows."""
    # Empty query
    if not query.strip():
        return """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔗</div>
            <p>Enter a query to search for workflows.</p>
            <p style="font-size: 0.9em;">Try: "modify a file", "commit changes", "generate and save image"</p>
        </div>
        """

    # Sanitize input
    query = sanitize_query(query)
    if not query:
        return "<p style='color: orange;'>Please enter a valid search query.</p>"

    chain_indexer = get_chain_indexer_instance()
    if not chain_indexer:
        return """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">⚙️</div>
            <p style="color: #ffb74d;">Chain indexing is disabled in configuration.</p>
            <p style="font-size: 0.9em;">Enable it in config.yaml to use workflow search.</p>
        </div>
        """

    try:

        async def do_search():
            return await chain_indexer.search_chains(
                query, top_k=int(top_k), min_confidence=min_confidence
            )

        results = run_async(do_search())
    except Exception as e:
        return format_error(e, f"Workflow search failed for: {query}")

    # No results
    if not results:
        return f"""
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔎</div>
            <p style="color: #ffb74d;">No workflows found matching "{truncate_text(query, 50)}"</p>
            <p style="font-size: 0.9em;">Workflows are auto-detected from usage patterns.</p>
            <p style="font-size: 0.9em; color: #aaa;">Use tools together to create workflows.</p>
        </div>
        """

    html_parts = [
        f'<p style="color: #888; margin-bottom: 12px;">Found {len(results)} workflow{"s" if len(results) != 1 else ""}</p>'
    ]

    for cr in results:
        confidence_pct = int(cr.score * 100)
        conf_label = confidence_label(cr.score)
        confidence_color = (
            "#81c784" if cr.score > 0.7 else "#ffb74d" if cr.score > 0.5 else "#9e9e9e"
        )
        tool_flow = " → ".join([t.split(":")[-1] for t in cr.chain.tools])

        html_parts.append(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; background: #1a2e1a;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.1em; font-weight: bold; color: #81c784;" title="{cr.chain.name}">{truncate_text(cr.chain.name, 40)}</span>
                <span style="color: {confidence_color};" title="{conf_label} match ({confidence_pct}%)">{conf_label} ({confidence_pct}%)</span>
            </div>
            <p style="margin: 8px 0; color: #ccc; font-family: monospace;" title="{tool_flow}">{truncate_text(tool_flow, 80)}</p>
            <p style="margin: 4px 0; color: #888; font-size: 0.9em;">{truncate_text(cr.chain.description, 100)}</p>
            <div style="font-size: 0.85em; color: #666;">
                Used {cr.chain.use_count} times | {"🤖 Auto-detected" if cr.chain.is_auto_detected else "👤 Manual"}
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

    # Empty index
    if not tools:
        return """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">📦</div>
            <p style="color: #ffb74d;">No tools indexed yet.</p>
            <p style="font-size: 0.9em;">Build the index first:</p>
            <code style="color: #888;">cd tool_compass && python gateway.py --sync</code>
        </div>
        """

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
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔎</div>
            <p style="color: #ffb74d;">No tools match the current filters.</p>
            <p style="font-size: 0.9em; color: #aaa;">Try removing filters or using different search terms.</p>
        </div>
        """

    # Group by server
    by_server = {}
    for t in tools:
        by_server.setdefault(t["server"], []).append(t)

    html_parts = [
        f'<p style="color: #888; margin-bottom: 12px;">Showing {len(tools)} tool{"s" if len(tools) != 1 else ""}</p>'
    ]

    for server_name, server_tools in sorted(by_server.items()):
        html_parts.append(f"""
        <details open style="margin: 12px 0;">
            <summary style="cursor: pointer; font-size: 1.1em; font-weight: bold; color: #64b5f6; padding: 8px 0;">
                📦 {server_name} ({len(server_tools)} tool{"s" if len(server_tools) != 1 else ""})
            </summary>
            <div style="padding-left: 16px;">
        """)

        for t in server_tools:
            param_count = len(t["parameters"])
            desc_truncated = truncate_text(t["description"], 120)
            html_parts.append(f"""
            <div style="border-left: 3px solid #444; padding: 8px 12px; margin: 8px 0; background: #1a1a2e;">
                <div style="font-weight: bold; color: #4fc3f7;" title="{t["name"]}">{truncate_text(t["name"], 45)}</div>
                <div style="color: #aaa; font-size: 0.9em; margin: 4px 0;" title="{t["description"]}">{desc_truncated}</div>
                <div style="color: #666; font-size: 0.85em;">
                    🏷️ {t["category"]} | 📝 {param_count} param{"s" if param_count != 1 else ""}
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
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔎</div>
            <p>Enter a tool name to view details.</p>
            <p style="font-size: 0.9em;">Or click on a tool from the browser above.</p>
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
        return format_error(e, f"Could not search for tool: {tool_name}")

    # Tool not found
    if not row:
        return f"""
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">❓</div>
            <p style="color: #ffb74d;">Tool not found: "{truncate_text(tool_name, 40)}"</p>
            <p style="font-size: 0.9em; color: #aaa;">Check the tool name and try again.</p>
        </div>
        """

    params = json.loads(row["parameters"]) if row["parameters"] else {}
    examples = json.loads(row["examples"]) if row["examples"] else []

    # Build parameters table
    params_html = ""
    if params:
        params_html = f"""
        <h4 style="color: #81c784; margin-top: 16px;">Parameters ({len(params)})</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #2a2a4a;">
                <th style="padding: 8px; text-align: left; border: 1px solid #444;">Name</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #444;">Type</th>
            </tr>
        """
        for name, ptype in params.items():
            params_html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #444; font-family: monospace; color: #4fc3f7;">{truncate_text(name, 30)}</td>
                <td style="padding: 8px; border: 1px solid #444; color: #888;">{truncate_text(str(ptype), 50)}</td>
            </tr>
            """
        params_html += "</table>"
    else:
        params_html = """
        <h4 style="color: #81c784; margin-top: 16px;">Parameters</h4>
        <p style="color: #888; font-style: italic;">No parameters required</p>
        """

    # Build examples
    examples_html = ""
    if examples:
        examples_html = f"<h4 style='color: #81c784; margin-top: 16px;'>Examples ({len(examples)})</h4>"
        for ex in examples:
            examples_html += f"<pre style='background: #1a1a2e; padding: 8px; border-radius: 4px; overflow-x: auto;'>{truncate_text(ex, 200)}</pre>"

    return f"""
    <div style="padding: 16px;">
        <h2 style="color: #4fc3f7; margin: 0; word-break: break-all;">{row["name"]}</h2>
        <div style="color: #888; margin: 8px 0;">
            📦 {row["server"]} | 🏷️ {row["category"]}
        </div>
        <p style="color: #ccc; font-size: 1.1em; margin: 16px 0;">{row["description"]}</p>
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

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;">
        <div style="background: #1a2e3a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #4fc3f7;">{searches["total"]}</div>
            <div style="color: #888;">Searches ({timeframe})</div>
        </div>
        <div style="background: #1a3a2a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #81c784;">{calls["total"]}</div>
            <div style="color: #888;">Tool Calls</div>
        </div>
        <div style="background: #3a2a1a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #ffb74d;">{calls["success_rate"]}%</div>
            <div style="color: #888;">Success Rate</div>
        </div>
        <div style="background: #2a2a3a; padding: 16px; border-radius: 8px; text-align: center;">
            <div style="font-size: 2em; font-weight: bold; color: #ba68c8;">{searches["avg_latency_ms"]}ms</div>
            <div style="color: #888;">Avg Search Latency</div>
        </div>
    </div>
    """

    # Top tools
    if calls["top_tools"]:
        html += """
        <h3 style="color: #4fc3f7;">Top Tools</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 24px;">
            <tr style="background: #2a2a4a;">
                <th style="padding: 8px; text-align: left; border: 1px solid #444;">Tool</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #444;">Calls</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #444;">Success</th>
                <th style="padding: 8px; text-align: right; border: 1px solid #444;">Latency</th>
            </tr>
        """
        for t in calls["top_tools"][:10]:
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #444; color: #4fc3f7;">{t["tool"]}</td>
                <td style="padding: 8px; border: 1px solid #444; text-align: right;">{t["calls"]}</td>
                <td style="padding: 8px; border: 1px solid #444; text-align: right; color: {"#81c784" if t["success_rate"] > 90 else "#ffb74d"};">{t["success_rate"]}%</td>
                <td style="padding: 8px; border: 1px solid #444; text-align: right; color: #888;">{t["avg_latency_ms"]}ms</td>
            </tr>
            """
        html += "</table>"

    # Top queries
    if searches["top_queries"]:
        html += """
        <h3 style="color: #81c784;">Top Queries</h3>
        <ul style="color: #ccc;">
        """
        for q in searches["top_queries"][:10]:
            html += f'<li>"{q["query"]}" <span style="color: #888;">({q["count"]} times)</span></li>'
        html += "</ul>"

    # Failures
    if summary.get("failures"):
        html += """
        <h3 style="color: #ef5350;">Recent Failures</h3>
        <ul style="color: #ccc;">
        """
        for f in summary["failures"][:5]:
            html += f'<li style="color: #ef5350;">{f["tool"]}: {f["error"] or "Unknown error"} ({f["count"]}x)</li>'
        html += "</ul>"

    # Hot cache
    hot_cache = summary.get("hot_cache", {})
    if hot_cache.get("tools"):
        html += f"""
        <h3 style="color: #ba68c8;">Hot Cache ({hot_cache["size"]} tools)</h3>
        <p style="color: #888; font-family: monospace;">{", ".join(hot_cache["tools"])}</p>
        """

    return html


# =============================================================================
# CHAIN VIEWER FUNCTIONS
# =============================================================================


def get_chains_view() -> str:
    """Display all tool chains."""
    chain_indexer = get_chain_indexer_instance()
    if not chain_indexer:
        return """
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">⚙️</div>
            <p style="color: #ffb74d;">Chain indexing is disabled in configuration.</p>
            <p style="font-size: 0.9em;">Enable <code>chain_indexing_enabled</code> in config.yaml to use workflows.</p>
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
        <div style="text-align: center; padding: 40px; color: #888;">
            <div style="font-size: 2em; margin-bottom: 12px;">🔗</div>
            <p>No workflows defined yet.</p>
            <p style="font-size: 0.9em; color: #aaa;">Workflows are auto-detected from usage patterns.</p>
            <p style="font-size: 0.9em; color: #aaa;">Use tools together to create workflows.</p>
        </div>
        """

    html_parts = [
        f'<p style="color: #888; margin-bottom: 12px;">{len(chains)} workflow{"s" if len(chains) != 1 else ""} available</p>'
    ]

    for chain in sorted(chains, key=lambda c: c.use_count, reverse=True):
        tool_flow = " → ".join([t.split(":")[-1] for t in chain.tools])
        badge = "🤖 Auto-detected" if chain.is_auto_detected else "👤 Manual"

        html_parts.append(f"""
        <div style="border: 1px solid #444; border-radius: 8px; padding: 16px; margin: 12px 0; background: #1a2e1a;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <span style="font-size: 1.2em; font-weight: bold; color: #81c784;" title="{chain.name}">{truncate_text(chain.name, 35)}</span>
                <span style="color: #888; font-size: 0.9em;">{badge}</span>
            </div>
            <div style="font-family: monospace; color: #4fc3f7; margin: 12px 0; font-size: 1.1em;" title="{tool_flow}">
                {truncate_text(tool_flow, 80)}
            </div>
            <p style="color: #aaa; margin: 8px 0;">{truncate_text(chain.description, 120)}</p>
            <div style="color: #666; font-size: 0.9em;">
                Used {chain.use_count} time{"s" if chain.use_count != 1 else ""}
            </div>
        </div>
        """)

    return "".join(html_parts)


# =============================================================================
# SYSTEM STATUS
# =============================================================================


def get_system_status() -> str:
    """Get system status overview."""
    # Load config first (doesn't require index)
    global _config
    if _config is None:
        try:
            _config = load_config()
        except Exception as e:
            return format_error(e, "Could not load configuration")

    # Check index status
    index_status = "✅ Loaded"
    stats = {}
    index_path = "Unknown"
    try:
        index = get_index()
        stats = index.get_stats()
        index_path = str(index.index_path)
    except Exception as e:
        index_status = f"⚠️ Not loaded: {truncate_text(str(e), 50)}"

    # Check analytics
    analytics_status = "✅ Available"
    hot_cache_size = 0
    try:
        analytics = get_analytics_instance()
        hot_cache_size = len(analytics._hot_cache)
    except Exception as e:
        analytics_status = f"⚠️ Error: {truncate_text(str(e), 50)}"

    # Check Ollama
    ollama_status = "❓ Not checked"
    try:
        from embedder import Embedder

        embedder = Embedder()
        is_healthy = run_async(embedder.health_check())
        ollama_status = "✅ Connected" if is_healthy else "⚠️ Model not loaded"
        run_async(embedder.close())
    except Exception as e:
        ollama_status = f"❌ Unavailable: {truncate_text(str(e), 40)}"

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;">
        <div>
            <h3 style="color: #4fc3f7;">System Health</h3>
            <ul style="color: #ccc; list-style: none; padding-left: 0;">
                <li style="margin: 8px 0;">Index: {index_status}</li>
                <li style="margin: 8px 0;">Analytics: {analytics_status}</li>
                <li style="margin: 8px 0;">Ollama: {ollama_status}</li>
            </ul>

            <h3 style="color: #4fc3f7;">Index Status</h3>
            <ul style="color: #ccc;">
                <li>Total tools: <strong>{stats.get("total_tools", 0)}</strong></li>
                <li>Core tools: {stats.get("core_tools", 0)}</li>
                <li>Index path: <code style="font-size: 0.85em;">{truncate_text(index_path, 40)}</code></li>
            </ul>

            <h4 style="color: #81c784;">By Server</h4>
    """

    if stats.get("by_server"):
        html += "<ul style='color: #ccc;'>"
        for server, count in sorted(stats.get("by_server", {}).items()):
            html += f"<li>{server}: {count}</li>"
        html += "</ul>"
    else:
        html += "<p style='color: #888; font-style: italic;'>No data</p>"

    html += "<h4 style='color: #81c784;'>By Category</h4>"

    if stats.get("by_category"):
        html += "<ul style='color: #ccc;'>"
        for category, count in sorted(stats.get("by_category", {}).items()):
            html += f"<li>{category}: {count}</li>"
        html += "</ul>"
    else:
        html += "<p style='color: #888; font-style: italic;'>No data</p>"

    html += f"""
        </div>

        <div>
            <h3 style="color: #4fc3f7;">Configuration</h3>
            <ul style="color: #ccc;">
                <li>Progressive disclosure: {"✅" if _config.progressive_disclosure else "❌"}</li>
                <li>Auto sync: {"✅" if _config.auto_sync else "❌"}</li>
                <li>Analytics: {"✅" if _config.analytics_enabled else "❌"}</li>
                <li>Chain indexing: {"✅" if _config.chain_indexing_enabled else "❌"}</li>
                <li>Embedding model: <code>{_config.embedding_model}</code></li>
                <li>Hot cache: {hot_cache_size}/{_config.hot_cache_size}</li>
            </ul>

            <h3 style="color: #4fc3f7;">Backends ({len(_config.backends)})</h3>
            <ul style="color: #ccc;">
    """

    for name in _config.backends.keys():
        html += f"<li>{name}</li>"

    html += """
            </ul>

            <h3 style="color: #4fc3f7;">Quick Commands</h3>
            <div style="font-size: 0.85em; color: #888;">
                <p style="margin: 4px 0;"><code>python gateway.py --sync</code> - Rebuild index</p>
                <p style="margin: 4px 0;"><code>python gateway.py --test</code> - Run tests</p>
                <p style="margin: 4px 0;"><code>ollama serve</code> - Start Ollama</p>
            </div>
        </div>
    </div>
    """

    return html


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
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .tool-result { border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; }
        """,
    ) as demo:
        gr.Markdown("""
        # 🧭 Tool Compass
        **Semantic search across 44 MCP tools** | Progressive discovery: Search → Describe → Execute
        """)

        with gr.Tabs():
            # =================================================================
            # SEARCH TAB
            # =================================================================
            with gr.Tab("🔍 Search", id="search"):
                gr.Markdown(
                    "Search tools using natural language. Describe what you want to do."
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        search_input = gr.Textbox(
                            label="What do you want to do?",
                            placeholder="e.g., 'generate an image with AI', 'read a file', 'search documents'",
                            lines=1,
                        )
                    with gr.Column(scale=1):
                        search_btn = gr.Button("Search", variant="primary")

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
                        search_results = gr.HTML(
                            value="""
                            <div style="text-align: center; padding: 40px; color: #888;">
                                <div style="font-size: 2em; margin-bottom: 12px;">🔍</div>
                                <p>Enter a search query above to find tools.</p>
                                <p style="font-size: 0.9em;">Try: "generate an image", "read a file", "search documents"</p>
                            </div>
                            """,
                            label="Results",
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
                    <div style="text-align: center; padding: 40px; color: #888;">
                        <div style="font-size: 2em; margin-bottom: 12px;">🔗</div>
                        <p>Enter a query to search for workflows.</p>
                        <p style="font-size: 0.9em;">Try: "modify a file", "commit changes", "generate and save image"</p>
                    </div>
                    """
                )

                # Wire up search
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
                )
                chain_btn.click(
                    fn=search_chains,
                    inputs=[chain_query, top_k, min_conf],
                    outputs=[chain_results],
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
                    <div style="text-align: center; padding: 40px; color: #888;">
                        <div style="font-size: 2em; margin-bottom: 12px;">🔎</div>
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

                chains_btn.click(fn=get_chains_view, inputs=[], outputs=[chains_html])

            # =================================================================
            # STATUS TAB
            # =================================================================
            with gr.Tab("⚙️ Status", id="status"):
                gr.Markdown("System status and configuration.")

                status_btn = gr.Button("Refresh Status", variant="primary")
                status_html = gr.HTML(value=get_system_status())

                status_btn.click(fn=get_system_status, inputs=[], outputs=[status_html])

        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
            Tool Compass v2.0 | Semantic tool discovery for MCP
        </div>
        """)

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
        print("   Run 'python gateway.py --sync' to build the index first.")

    demo = create_ui()
    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share, show_error=True
    )


if __name__ == "__main__":
    main()
