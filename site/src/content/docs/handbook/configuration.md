---
title: Configuration
description: Environment variables, config file, Docker, and troubleshooting.
sidebar:
  order: 4
---

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | Project root directory | Auto-detected from module location |
| `TOOL_COMPASS_PYTHON` | Python executable path | Current interpreter or venv auto-detect |
| `TOOL_COMPASS_CONFIG` | Path to config JSON file | `./compass_config.json` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `COMFYUI_URL` | ComfyUI server URL (used by the comfy backend) | `http://localhost:8188` |
| `PORT` | Set to enable HTTP (streamable-http) transport instead of stdio | unset (stdio mode) |
| `HOST` | HTTP bind address (only used when `PORT` is set) | `127.0.0.1` (loopback) |
| `GRADIO_AUTH` | `user:pass` required when launching `tool-compass ui --share` (also settable via `--auth user:pass`) | unset (refuses `--share` without it) |
| `HYPOTHESIS_PROFILE` | `dev` / `ci` / `nightly` profile for fuzz tests | `dev` |

:::caution
**`HOST=0.0.0.0` exposes the gateway on every network interface.** Tool
Compass proxies arbitrary MCP tool calls — only set it behind an
authenticated reverse proxy. Default loopback is safe.
:::

## Config file

Tool Compass reads settings from a JSON file at startup. The resolution order for the config path is:

1. `TOOL_COMPASS_CONFIG` environment variable
2. `./compass_config.json` in the tool-compass directory

If no config file is found, built-in defaults are used. See `compass_config.example.json` in the repository for a complete example.

### All config options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backends` | object | (built-in defaults) | Backend server connections keyed by name |
| `embedding_model` | string | `nomic-embed-text` | Ollama embedding model name |
| `ollama_url` | string | `http://localhost:11434` | Ollama API base URL |
| `index_dir` | string | `./db` | Directory for HNSW index and SQLite databases |
| `auto_sync` | bool | `true` | Enable automatic tool discovery from backends |
| `default_top_k` | int | `5` | Default number of results for compass searches |
| `min_confidence` | float | `0.3` | Default minimum similarity threshold |
| `progressive_disclosure` | bool | `true` | Return summaries only from compass (use describe for full schemas) |
| `sync_check_on_startup` | bool | `true` | Check for backend changes on first compass call |
| `sync_polling_interval` | int | `300` | Background polling interval in seconds (0 = disabled) |
| `analytics_enabled` | bool | `true` | Track search queries and tool calls in local SQLite |
| `hot_cache_size` | int | `10` | Number of frequently used tools to keep in memory |
| `chain_indexing_enabled` | bool | `true` | Enable tool chain detection and indexing |
| `chain_detection_min_occurrences` | int | `3` | Minimum pattern occurrences before promoting to a chain |
| `top_chains_cache_size` | int | `5` | Number of top chains to keep in memory cache (clamped to ≥ 0; 0 disables the cache) |
| `ollama_breaker_failure_threshold` | int | `3` | Consecutive Ollama failures before the circuit breaker opens (clamped `1`–`20`) |
| `ollama_breaker_open_seconds` | float | `30.0` | How long the breaker stays open before a trial request, in seconds (clamped `1.0`–`600.0`) |
| `ollama_retry_attempts` | int | `3` | Retry attempts for a failed embedding request (clamped `0`–`10`) |
| `ollama_retry_backoffs` | float[] | `[0.5, 1.0, 2.0]` | Per-retry backoff delays in seconds; resets to the default if not a list of non-negative numbers |
| `hnsw_m` | int | `16` | HNSW graph connectivity (`M`); higher = better recall, more memory (clamped `4`–`64`) |
| `hnsw_ef_construction` | int | `200` | HNSW build-time search width; higher = better index quality, slower build (clamped `40`–`800`) |
| `hnsw_ef_search` | int | `50` | HNSW query-time search width; higher = better recall, slower search (clamped `10`–`400`) |

#### Added in v2.5.0

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hybrid_search` | bool | `true` | Fuse the semantic (HNSW) and lexical signals with Reciprocal Rank Fusion on the healthy path. `false` = pure-semantic ordering. |
| `exact_name_boost` | bool | `true` | An exact tool-name match (full `server:tool` or the bare tool name) is boosted to rank #1. |
| `exact_match_confidence` | float | `1.0` | Confidence assigned to an exact-name match (clamped `0.0`–`1.0`). |
| `analytics_retention_days` | int | `30` | Prune analytics rows older than this on startup sync (`0` = keep forever). |
| `gateway_auth_token` | string \| null | `null` | Bearer token required on the HTTP transport (opt-in). Set here or via `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` (env wins). Redacted in `doctor` output. Never affects stdio. |

**Per-backend options** (set inside a `backends.<name>` object):

| Key | Type | Applies to | Description |
|-----|------|------------|-------------|
| `default_timeout` | float \| null | stdio, http | Per-backend tool-call timeout in seconds, overriding the 15 s default (clamped `1.0`–`600.0`). |
| `tool_timeouts` | object | stdio, http | Per-tool timeout override, keyed by the **bare** tool name (e.g. `{"generate_image": 120}`). |
| `allow_tools` | string[] | stdio, http, import | Glob allowlist — only matching tools are indexed/executed (empty = allow all). |
| `deny_tools` | string[] | stdio, http, import | Glob denylist — matching tools are never indexed/executed (deny wins over allow). |

> Backend **names must not contain `:`** — it is the reserved `backend:tool`
> separator. A colon in a backend name is rejected at config load.

### HTTP (remote) backends

Beyond local stdio subprocess servers, a backend can be a remote MCP server
spoken over streamable-http:

```json
{
  "backends": {
    "remote-fs": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer ${REMOTE_MCP_TOKEN}" },
      "default_timeout": 30,
      "deny_tools": ["*delete*", "*write*"]
    }
  }
}
```

Response headers are redacted in `doctor`/`show_config` output. To require a
token on **your** gateway's HTTP endpoint (as opposed to an upstream backend),
set `gateway_auth_token`.

### Variable substitution

The config file supports `${VAR}` substitution. Values are resolved from environment variables first, then from a `defaults` block in the config file:

```json
{
  "defaults": {
    "BASE": "/home/user/project"
  },
  "backends": {
    "bridge": {
      "type": "stdio",
      "command": "python",
      "args": ["-u", "${BASE}/app/mcp/bridge_mcp_server.py"]
    }
  }
}
```

## Docker

```bash
# Start with Docker Compose
docker-compose up

# Include Ollama in the stack
docker-compose --profile with-ollama up
```

The Gradio UI is available at `http://localhost:7860` when running in Docker.

## Troubleshooting

### MCP server not connecting

If Claude Desktop logs show JSON parse errors like:

```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**Cause:** `print()` statements corrupt the JSON-RPC protocol.

**Fix:** Use logging or write to stderr:

```python
import sys
print("Debug message", file=sys.stderr)
```

### Ollama connection failed

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### Index not found

Rebuild the index using the v2.2 canonical CLI:

```bash
tool-compass sync
```

### Backend not connecting

If a backend fails to connect, verify:

1. The command in your config file is correct and the script exists
2. The backend MCP server starts successfully when run standalone
3. Check gateway logs with `--verbose` for detailed connection errors

## Security and data scope

- **Data touched:** Tool descriptions indexed in local HNSW vector DB, search queries logged to local SQLite, embeddings generated via local Ollama
- **Data NOT touched:** No user code, no file contents, no credentials. Tool call arguments are hashed, not stored in plain text
- **Network:** Connects to local Ollama for embeddings. Optional Gradio UI binds to localhost. No external telemetry
