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
"""

import asyncio
import argparse
import logging
import json
import time
from typing import Optional, List, Dict, Any

# MCP imports
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("FastMCP not installed. Install with: pip install mcp")
    raise

from indexer import CompassIndex, SearchResult
from tool_manifest import ToolDefinition
from config import load_config, CompassConfig, CONFIG_PATH
from backend_client import BackendManager
from analytics import CompassAnalytics, get_analytics
from sync_manager import SyncManager, get_sync_manager
from chain_indexer import ChainIndexer, get_chain_indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("tool-compass-gateway")

# Global state with async locks for thread-safe singleton initialization
# Pattern: Double-checked locking with asyncio.Lock
# See: https://www.hevalhazalkurt.com/blog/implementing-singleton-with-asyncawait-in-python/
_compass_index: Optional[CompassIndex] = None
_backend_manager: Optional[BackendManager] = None
_config: Optional[CompassConfig] = None
_analytics: Optional[CompassAnalytics] = None
_sync_manager: Optional[SyncManager] = None
_chain_indexer: Optional[ChainIndexer] = None
_startup_sync_done: bool = False

# Async locks to prevent race conditions during singleton initialization
_index_lock = asyncio.Lock()
_backend_lock = asyncio.Lock()
_analytics_lock = asyncio.Lock()
_sync_manager_lock = asyncio.Lock()
_chain_indexer_lock = asyncio.Lock()
_startup_sync_lock = asyncio.Lock()


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
    """
    global _compass_index

    # Fast path: already initialized
    if _compass_index is not None:
        return _compass_index

    # Slow path: acquire lock and check again
    async with _index_lock:
        # Double-check after acquiring lock (another coroutine may have initialized)
        if _compass_index is not None:
            return _compass_index

        index = CompassIndex()

        # Try to load existing index
        if not index.load_index():
            logger.warning("No existing index found. Building from manifest...")

            # Check Ollama
            if not await index.embedder.health_check():
                raise RuntimeError(
                    "Ollama not available. Start Ollama and run: ollama pull nomic-embed-text"
                )

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
    async with _backend_lock:
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
    async with _analytics_lock:
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
    async with _sync_manager_lock:
        if _sync_manager is not None:
            return _sync_manager

        index = await get_index()
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
    async with _chain_indexer_lock:
        if _chain_indexer is not None:
            return _chain_indexer

        index = await get_index()
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
    async with _startup_sync_lock:
        # Double-check after acquiring lock
        if _startup_sync_done:
            return

        _startup_sync_done = True
        sync_manager = await get_sync_manager_instance()
        if sync_manager:
            try:
                await sync_manager.sync_if_needed()
            except Exception as e:
                logger.warning(f"Startup sync failed: {e}")


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
    config = get_config()
    top_k = max(1, min(10, top_k))
    min_confidence = max(0.0, min(1.0, min_confidence))

    # Check for sync on first call
    await maybe_startup_sync()

    index = await get_index()

    # Search tools
    results: List[SearchResult] = await index.search(
        query=intent, top_k=top_k, category_filter=category, server_filter=server
    )

    # Filter by confidence
    results = [r for r in results if r.score >= min_confidence]

    # Search chains if enabled
    chain_matches = []
    if include_chains and config.chain_indexing_enabled:
        chain_indexer = await get_chain_indexer_instance()
        if chain_indexer:
            chain_results = await chain_indexer.search_chains(
                intent, top_k=3, min_confidence=min_confidence
            )
            for cr in chain_results:
                chain_matches.append(
                    {
                        "name": cr.chain.name,
                        "tools": cr.chain.tools,
                        "description": cr.chain.description,
                        "confidence": round(cr.score, 3),
                        "use_count": cr.chain.use_count,
                    }
                )

    # Build response - progressive disclosure means we only return summaries
    matches = []
    for r in results:
        match_data = {
            "tool": r.tool.name,
            "description": r.tool.description,
            "server": r.tool.server,
            "category": r.tool.category,
            "confidence": round(r.score, 3),
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
        "matches": matches,
        "total_indexed": total_tools,
        "tokens_saved": (total_tools - len(matches)) * 500,
        "hint": hint,
        "workflow": "compass() -> describe() -> execute()"
        if config.progressive_disclosure
        else "compass() -> execute()",
    }

    # Include chains if any found
    if chain_matches:
        response["chains"] = chain_matches

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
    index = await get_index()

    # Try to find in index first (from manifest)
    if index.db:
        cursor = index.db.execute(
            "SELECT name, description, category, server, parameters, examples FROM tools WHERE name = ?",
            (tool_name,),
        )
        row = cursor.fetchone()

        if row:
            params = json.loads(row["parameters"]) if row["parameters"] else {}
            examples = json.loads(row["examples"]) if row["examples"] else []

            return {
                "tool": row["name"],
                "description": row["description"],
                "server": row["server"],
                "category": row["category"],
                "parameters": params,
                "examples": examples,
                "hint": f"Use execute('{tool_name}', {{...}}) to run this tool.",
            }

    # Try backends if connected
    manager = await get_backends()
    schema = manager.get_tool_schema(tool_name)
    if schema:
        return {
            **schema,
            "hint": f"Use execute('{tool_name}', {{...}}) to run this tool.",
        }

    return {
        "error": f"Tool not found: {tool_name}",
        "hint": "Use compass() to search for available tools.",
    }


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
        if server_name not in [
            name for name, conn in manager._backends.items() if conn.is_connected
        ]:
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
                return {
                    "success": False,
                    "error": f"Failed to connect to backend: {server_name}",
                    "hint": "Check that the backend server is configured correctly.",
                }

    # Execute
    result = await manager.execute_tool(tool_name, arguments)

    # Record analytics
    latency_ms = (time.time() - start_time) * 1000
    success = result.get("success", True) if isinstance(result, dict) else True
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

    return result


@mcp.tool()
async def compass_categories() -> Dict[str, Any]:
    """
    List available tool categories and servers.

    Use this to understand what kinds of tools are available before searching.
    """
    index = await get_index()
    stats = index.get_stats()

    return {
        "categories": stats.get("by_category", {}),
        "servers": stats.get("by_server", {}),
        "total_tools": stats.get("total_tools", 0),
        "hint": "Use compass(intent, category='file') to filter searches.",
    }


@mcp.tool()
async def compass_status() -> Dict[str, Any]:
    """
    Get Tool Compass gateway status and health information.

    Returns index stats, backend connection status, configuration, analytics summary,
    hot cache status, and sync status.
    """
    config = get_config()
    index = await get_index()
    manager = await get_backends()

    index_stats = index.get_stats()
    backend_stats = manager.get_stats()

    response = {
        "index": {
            "total_tools": index_stats.get("total_tools", 0),
            "by_category": index_stats.get("by_category", {}),
            "by_server": index_stats.get("by_server", {}),
        },
        "backends": backend_stats,
        "config": {
            "progressive_disclosure": config.progressive_disclosure,
            "auto_sync": config.auto_sync,
            "embedding_model": config.embedding_model,
            "analytics_enabled": config.analytics_enabled,
            "chain_indexing_enabled": config.chain_indexing_enabled,
        },
    }

    # Add analytics info if enabled
    if config.analytics_enabled:
        analytics = await get_analytics_instance()
        if analytics:
            response["hot_cache"] = {
                "size": len(analytics._hot_cache),
                "tools": list(analytics._hot_cache.keys()),
            }

    # Add sync status if enabled
    if config.auto_sync:
        sync_manager = await get_sync_manager_instance()
        if sync_manager:
            response["sync"] = await sync_manager.get_sync_status()

    # Add chain info if enabled
    if config.chain_indexing_enabled:
        chain_indexer = await get_chain_indexer_instance()
        if chain_indexer:
            chains = await chain_indexer.load_chains_from_db()
            response["chains"] = {
                "total": len(chains),
                "cached": len(chain_indexer._chain_cache),
            }

    return response


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

    if not config.analytics_enabled:
        return {
            "error": "Analytics is disabled",
            "hint": "Enable analytics_enabled in config to track usage",
        }

    analytics = await get_analytics_instance()
    if not analytics:
        return {"error": "Analytics not initialized"}

    summary = await analytics.get_analytics_summary(timeframe)

    if not include_failures:
        summary.pop("failures", None)

    return summary


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

    if not config.chain_indexing_enabled:
        return {
            "error": "Chain indexing is disabled",
            "hint": "Enable chain_indexing_enabled in config",
        }

    chain_indexer = await get_chain_indexer_instance()
    if not chain_indexer:
        return {"error": "Chain indexer not initialized"}

    if action == "list":
        chains = await chain_indexer.load_chains_from_db()
        return {
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
        }

    elif action == "create":
        if not chain_name or not tools:
            return {
                "error": "chain_name and tools are required for create",
                "hint": "compass_chains(action='create', chain_name='my_workflow', tools=['tool1', 'tool2'])",
            }

        chain = await chain_indexer.add_chain(
            name=chain_name,
            tools=tools,
            description=description,
            is_auto_detected=False,
        )

        return {
            "created": {
                "name": chain.name,
                "tools": chain.tools,
                "description": chain.description,
            },
            "hint": f"Chain '{chain_name}' created. It will now appear in compass() search results.",
        }

    elif action == "detect":
        analytics = await get_analytics_instance()
        if analytics:
            detected = await analytics.detect_chains()
            return {
                "detected": detected,
                "count": len(detected),
                "hint": "Detected chains are now indexed and searchable",
            }
        return {"error": "Analytics required for chain detection"}

    else:
        return {
            "error": f"Unknown action: {action}",
            "valid_actions": ["list", "create", "detect"],
        }


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

    if not config.auto_sync:
        return {
            "error": "Auto-sync is disabled",
            "hint": "Enable auto_sync in config for automatic synchronization",
        }

    sync_manager = await get_sync_manager_instance()
    if not sync_manager:
        return {"error": "Sync manager not initialized"}

    if force:
        result = await sync_manager.full_sync()
        return {"action": "full_sync", "result": result}
    else:
        results = await sync_manager.sync_if_needed()
        return {
            "action": "sync_if_needed",
            "backends": results,
            "hint": "Use force=True to rebuild the entire index",
        }


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
    index = await get_index()
    manager = await get_backends()

    index_stats = index.get_stats()
    backend_stats = manager.get_stats()

    audit = {
        "system": {
            "version": "2.0",
            "total_tools": index_stats.get("total_tools", 0),
            "index_path": str(index.index_path),
            "db_path": str(index.db_path),
        },
        "categories": index_stats.get("by_category", {}),
        "servers": index_stats.get("by_server", {}),
        "backends": backend_stats,
        "config": {
            "progressive_disclosure": config.progressive_disclosure,
            "auto_sync": config.auto_sync,
            "analytics_enabled": config.analytics_enabled,
            "chain_indexing_enabled": config.chain_indexing_enabled,
            "hot_cache_size": config.hot_cache_size,
            "embedding_model": config.embedding_model,
        },
    }

    # Hot cache
    if config.analytics_enabled:
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

    # Chains
    if config.chain_indexing_enabled:
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

    # Sync status
    if config.auto_sync:
        sync_manager = await get_sync_manager_instance()
        if sync_manager:
            sync_status = await sync_manager.get_sync_status()
            audit["sync"] = sync_status

    # Optionally include all tools
    if include_tools:
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

    # Health check
    issues = []
    if index_stats.get("total_tools", 0) == 0:
        issues.append("No tools indexed - run compass_sync(force=True)")
    if config.analytics_enabled and not analytics._hot_cache:
        issues.append("Hot cache empty - will populate as tools are used")
    if not config.chain_indexing_enabled:
        issues.append("Chain indexing disabled - enable for workflow detection")

    audit["health"] = {
        "status": "healthy" if not issues else "needs_attention",
        "issues": issues if issues else ["All systems operational"],
    }

    return audit


# =============================================================================
# CLI COMMANDS
# =============================================================================


async def sync_from_backends():
    """Sync tool definitions from live backend servers and rebuild index."""
    print("Syncing tools from backend servers...")

    config = load_config()
    manager = BackendManager(config)

    # Connect to all backends
    print(f"Connecting to {len(config.backends)} backends...")
    results = await manager.connect_all()

    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    # Get all tools
    tools = manager.get_all_tools()
    print(f"\nDiscovered {len(tools)} tools from backends")

    # Convert to ToolDefinition format for indexing
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

    # Build index
    print("\nBuilding search index...")
    index = CompassIndex()

    if not await index.embedder.health_check():
        print("ERROR: Ollama not available. Run: ollama pull nomic-embed-text")
        await manager.disconnect_all()
        return

    result = await index.build_index(tool_defs)
    print(
        f"Index built: {result['tools_indexed']} tools in {result['total_time']:.2f}s"
    )

    # Cleanup
    await index.close()
    await manager.disconnect_all()

    print("\nSync complete!")


def categorize_tool(name: str, description: str) -> str:
    """Infer category from tool name and description."""
    name_lower = name.lower()
    description.lower()

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


async def run_tests():
    """Run test queries to verify gateway functionality."""
    print("\n" + "=" * 60)
    print("TOOL COMPASS GATEWAY - TEST SUITE")
    print("=" * 60)

    index = await get_index()
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
    print(f"Ollama URL: {config.ollama_url}")
    print(f"Default top_k: {config.default_top_k}")
    print(f"Min confidence: {config.min_confidence}")

    print("\n--- Backends ---")
    for name, backend in config.backends.items():
        print(f"\n{name}:")
        print(f"  Type: {backend.type}")
        if hasattr(backend, "command"):
            print(f"  Command: {backend.command}")
            print(
                f"  Args: {backend.args[:2]}..."
                if len(backend.args) > 2
                else f"  Args: {backend.args}"
            )


async def async_main(args):
    """Handle async CLI operations."""
    if args.sync:
        await sync_from_backends()
    elif args.test:
        await run_tests()


def main():
    parser = argparse.ArgumentParser(
        description="Tool Compass Gateway - MCP Proxy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gateway.py              Start the gateway server
  python gateway.py --sync       Sync tools from backends and rebuild index
  python gateway.py --test       Run test queries
  python gateway.py --config     Show current configuration
        """,
    )
    parser.add_argument(
        "--sync", action="store_true", help="Sync tools from backend servers"
    )
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--config", action="store_true", help="Show configuration")

    args = parser.parse_args()

    if args.config:
        show_config()
    elif args.sync or args.test:
        asyncio.run(async_main(args))
    else:
        # NOTE: Never print() to stdout in MCP mode - it corrupts JSON-RPC!
        # Use stderr for diagnostics if needed
        import sys

        print("Starting Tool Compass Gateway v2.0...", file=sys.stderr)
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
        mcp.run()


if __name__ == "__main__":
    main()
