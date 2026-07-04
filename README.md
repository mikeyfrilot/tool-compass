<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/tool-compass/readme.png" alt="Tool Compass Logo" width="400"></p>

**Semantic navigator for MCP tools - Find the right tool by intent, not memory**

<a href="https://github.com/mcp-tool-shop-org/tool-compass/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mcp-tool-shop-org/tool-compass/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/mcp-tool-shop-org/tool-compass"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/tool-compass?style=flat-square" alt="Codecov"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<a href="LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/tool-compass?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white" alt="Docker">
<a href="https://mcp-tool-shop-org.github.io/tool-compass/"><img src="https://img.shields.io/badge/Landing_Page-live-blue?style=flat-square" alt="Landing Page"></a>


*95% fewer tokens. Find tools by describing what you want to do.*

[Installation](#quick-start) • [Usage](#usage) • [Docker](#option-2-docker) • [Handbook](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) • [Performance](#performance) • [Contributing](#contributing)

</div>

---

## The Problem

MCP servers expose dozens or hundreds of tools. Loading all tool definitions into context wastes tokens and slows down responses.

```
Before: 77 tools × ~500 tokens = 38,500 tokens per request
After:  1 compass tool + 3 results = ~2,000 tokens per request

Savings: 95%
```

## The Solution

Tool Compass uses **semantic search** to find relevant tools from a natural language description. Instead of loading all tools, Claude calls `compass()` with an intent and gets back only the relevant tools.

## Quick Start

📖 **Full documentation:** See the [Tool Compass Handbook](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) for installation, configuration, and architecture deep-dives.

### Option 1: npm (zero-prerequisite, no Python install)

```bash
npx @mcptoolshop/tool-compass --help
npx @mcptoolshop/tool-compass serve                 # MCP gateway
npx @mcptoolshop/tool-compass ui                    # Gradio UI
npx @mcptoolshop/tool-compass doctor                # Diagnose setup
npx @mcptoolshop/tool-compass execute fs:read_file '{"path":"README.md"}'  # Smoke-test a proxied call
```

Downloads a verified platform binary on first run (SHA256-checked against the GitHub Release). Cached locally — subsequent invocations launch instantly. See [@mcptoolshop/tool-compass](https://www.npmjs.com/package/@mcptoolshop/tool-compass) on npm.

### Option 2: PyPI

```bash
pip install tool-compass
tool-compass --help
```

### Option 3: Local clone

```bash
# Prerequisites: Ollama with nomic-embed-text
ollama pull nomic-embed-text

# Clone and setup
git clone https://github.com/mcp-tool-shop-org/tool-compass.git
cd tool-compass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the search index
tool-compass sync

# Run the MCP server
tool-compass serve

# Or launch the Gradio UI
tool-compass ui
```

### Option 4: Docker

```bash
# Clone the repo
git clone https://github.com/mcp-tool-shop-org/tool-compass.git
cd tool-compass

# Start with Docker Compose (requires Ollama running locally)
docker-compose up

# Or include Ollama in the stack
docker-compose --profile with-ollama up

# Access the UI at http://localhost:7860
```

> The GHCR image (`ghcr.io/mcp-tool-shop-org/tool-compass`) supports
> `linux/amd64` and `linux/arm64`, so the same tag runs on x86_64 servers
> and Apple Silicon / ARM workstations.

## Features

- **Hybrid Search** - Semantic (HNSW) + lexical fusion with exact-name boost — describe what you want, or paste a tool name and it ranks #1
- **Full-Schema Progressive Disclosure** - `compass()` → `describe()` → `execute()`; `describe()` returns the complete `inputSchema` (required fields, descriptions, enums, defaults)
- **stdio + HTTP backends** - Front local subprocess MCP servers *and* remote / SaaS servers over streamable-http, with optional bearer-token auth
- **Per-tool timeouts & allow/deny** - Override the default timeout per backend/tool; expose a safe subset of a broad backend
- **Hot Cache & Chain Detection** - Frequently used tools pre-loaded; common tool workflows discovered automatically
- **Analytics** - Track usage patterns and tool performance (with retention/prune)
- **Cross-Platform & Docker Ready** - Windows, macOS, Linux; one-command deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       TOOL COMPASS                          │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Ollama     │    │   hnswlib    │    │   SQLite     │   │
│  │   Embedder   │───▶│    HNSW      │◀───│   Metadata   │   │
│  │  (nomic)     │    │   Index      │    │   Store      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                              │                              │
│                              ▼                              │
│                    ┌───────────────────┐                    │
│                    │ Gateway (9 tools)  │                   │
│                    │ compass, describe  │                   │
│                    │ execute, etc.      │                   │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### The `compass()` Tool

```python
compass(
    intent="I need to generate an AI image from a text description",
    top_k=3,
    category=None,  # Optional: "file", "git", "database", "ai", etc.
    min_confidence=0.3
)
```

Returns:
```json
{
  "matches": [
    {
      "tool": "comfy:comfy_generate",
      "description": "Generate image from text prompt using AI",
      "category": "ai",
      "confidence": 0.912
    }
  ],
  "total_indexed": 44,
  "tokens_saved": 20500,
  "hint": "Found: comfy:comfy_generate. Use describe() for full schema."
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `compass(intent)` | Hybrid semantic + lexical search with exact-name boost |
| `describe(tool_name)` | Get the full `inputSchema` for a tool (required/enums/defaults) |
| `execute(tool_name, args)` | Run a tool on its backend |
| `compass_categories()` | List categories and servers |
| `compass_status(active)` | System health and config; `active=True` runs a live backend liveness probe |
| `compass_analytics(timeframe)` | Usage statistics |
| `compass_chains(action)` | Manage tool workflows |
| `compass_sync(force)` | Rebuild index from backends |
| `compass_audit()` | Full system report |

The same actions are available from the CLI — including `tool-compass execute <tool> '<json>'` to smoke-test a proxied call from the terminal.

### Progressive Disclosure Pattern

Tool Compass uses a three-step progressive disclosure pattern to minimize token usage:

```
1. compass("your intent")     → Get tool name + short description (~100 tokens)
2. describe("tool:name")      → Get full parameter schema (~500 tokens)
3. execute("tool:name", args) → Run the tool
```

**Why this matters:**
- Loading 77 tools upfront = ~38,500 tokens
- Progressive disclosure = ~600 tokens per tool used
- Savings: **95%+ for typical workflows**

**Example workflow:**

```python
# Step 1: Find the right tool
compass("generate an image from text")
# Returns: comfy:comfy_generate (confidence: 0.91)

# Step 2: Get the schema (only if needed)
describe("comfy:comfy_generate")
# Returns: Full parameter definitions, types, examples

# Step 3: Execute
execute("comfy:comfy_generate", {"prompt": "a sunset over mountains"})
```

The `hint` field in compass results guides this flow, suggesting when to use `describe()`.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | Project root | Auto-detected |
| `TOOL_COMPASS_PYTHON` | Python executable | Auto-detected |
| `TOOL_COMPASS_CONFIG` | Config file path | `~/.config/tool-compass/compass_config.json` |
| `TOOL_COMPASS_DATA_DIR` | Data directory | Platform-specific (see below) |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `COMFYUI_URL` | ComfyUI server | `http://localhost:8188` |
| `PORT` | Set to enable HTTP transport (e.g., for Fly.io) | unset (stdio) |
| `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` | Bearer token required on the HTTP transport (opt-in; overrides the `gateway_auth_token` config field) | unset (no auth) |

**Default data directories:**
- **Windows:** `%LOCALAPPDATA%\tool-compass\`
- **macOS:** `~/Library/Application Support/tool-compass/`
- **Linux:** `~/.config/tool-compass/` (or `$XDG_CONFIG_HOME/tool-compass/`)

Config-file settings (in `compass_config.json`) added in v2.5.0 — `hybrid_search`,
`exact_name_boost`, per-backend `default_timeout` / `tool_timeouts`,
`allow_tools` / `deny_tools`, `analytics_retention_days`, and HTTP (`type: "http"`)
backends — are documented in the [Handbook → Configuration](https://mcp-tool-shop-org.github.io/tool-compass/handbook/configuration/).
See [`.env.example`](.env.example) for env-var options.

## Performance

| Metric | Value |
|--------|-------|
| Index build time | ~5s for 44 tools |
| Query latency | ~15ms (including embedding) |
| Token savings | ~95% (38K → 2K) |
| Accuracy@3 | ~95% (correct tool in top 3) |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## Troubleshooting

### MCP Server Not Connecting

If Claude Desktop logs show JSON parse errors:
```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**Cause**: `print()` statements corrupt JSON-RPC protocol.

**Fix**: Use logging or `file=sys.stderr`:
```python
import sys
print("Debug message", file=sys.stderr)
```

### Ollama Connection Failed

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### Index Not Found

```bash
tool-compass sync
```

## Related Projects

Part of the **Compass Suite** for AI-powered development:

- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) - Semantic file search
- [Integradio](https://github.com/mcp-tool-shop-org/integradio) - Vector-embedded Gradio components
- [Backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) - Headless LLM fine-tuning
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) - ComfyUI without the complexity

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security & Data Scope

Tool Compass is a **local-first** development tool. See [SECURITY.md](SECURITY.md) for full policy.

- **Data touched:** tool descriptions indexed in local HNSW vector DB, search queries logged to local SQLite (`compass_analytics.db`), embeddings generated via local Ollama.
- **Data NOT touched:** no user code, no file contents, no credentials. Tool call arguments are hashed, not stored in plain text.
- **Network:** connects to local Ollama for embeddings. Optional Gradio UI binds to localhost. No external telemetry.
- **No telemetry:** collects nothing externally. Analytics are local-only.

## Scorecard

Per-category scores are regenerated post-swarm via
`bash scripts/regenerate-scorecard.sh` (which wraps `npx
@mcptoolshop/shipcheck audit`). See [SCORECARD.md](SCORECARD.md) for the
current authoritative breakdown — the table below mirrors it and is
intentionally not hand-authored. Hand-curated sections (Known Gaps,
Remediation History) live outside the `<!-- SHIPCHECK-AUTO-START/END -->`
markers in SCORECARD.md and survive regenerations.

Latest `shipcheck audit`: **32 checked · 0 unchecked · 5 skipped · 100% pass — all hard gates pass.**

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | ✅ Pass | SHA-pinned actions; digest-pinned base image; SLSA provenance + SBOM on PyPI + GHCR; pre-commit secrets scan; opt-in gateway bearer auth |
| B. Error Handling | ✅ Pass | Structured results, graceful degradation, exit codes |
| C. Operator Docs | ✅ Pass | README, CHANGELOG, LICENSE, Makefile `verify` + `verify-metrics` + `scorecard` |
| D. Shipping Hygiene | ✅ Pass | CI consolidated; timeout-minutes + retention-days on every job; pytest config in pyproject.toml |
| E. Identity (soft) | ✅ Pass | Logo, landing page, GitHub metadata; explicit maintainers in pyproject.toml |
| **Total** | **100%** | All hard gates pass — regenerate via `make scorecard` |

## License

[MIT](LICENSE) - see LICENSE file for details.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>

