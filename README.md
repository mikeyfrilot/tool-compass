<div align="center">

# Tool Compass

**Semantic navigator for MCP tools - Find the right tool by intent, not memory**

[![Tests](https://github.com/mcp-tool-shop/tool-compass/actions/workflows/test.yml/badge.svg)](https://github.com/mcp-tool-shop/tool-compass/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mcp-tool-shop/tool-compass/graph/badge.svg)](https://codecov.io/gh/mcp-tool-shop/tool-compass)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/mcp-tool-shop/tool-compass?style=social)](https://github.com/mcp-tool-shop/tool-compass)

*95% fewer tokens. Find tools by describing what you want to do.*

[Installation](#quick-start) • [Usage](#usage) • [Docker](#option-2-docker) • [Performance](#performance) • [Contributing](#contributing)

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

<!--
## Demo

<p align="center">
  <img src="docs/assets/demo.gif" alt="Tool Compass Demo" width="600">
</p>
-->

## Quick Start

### Option 1: Local Installation

```bash
# Prerequisites: Ollama with nomic-embed-text
ollama pull nomic-embed-text

# Clone and setup
git clone https://github.com/mcp-tool-shop/tool-compass.git
cd tool-compass/tool_compass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the search index
python gateway.py --sync

# Run the MCP server
python gateway.py

# Or launch the Gradio UI
python ui.py
```

### Option 2: Docker

```bash
# Clone the repo
git clone https://github.com/mcp-tool-shop/tool-compass.git
cd tool-compass/tool_compass

# Start with Docker Compose (requires Ollama running locally)
docker-compose up

# Or include Ollama in the stack
docker-compose --profile with-ollama up

# Access the UI at http://localhost:7860
```

## Features

- **Semantic Search** - Find tools by describing what you want to do
- **Progressive Disclosure** - `compass()` → `describe()` → `execute()`
- **Hot Cache** - Frequently used tools are pre-loaded
- **Chain Detection** - Automatically discovers common tool workflows
- **Analytics** - Track usage patterns and tool performance
- **Cross-Platform** - Windows, macOS, Linux
- **Docker Ready** - One-command deployment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TOOL COMPASS                            │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Ollama     │    │   hnswlib    │    │   SQLite     │  │
│  │   Embedder   │───▶│    HNSW      │◀───│   Metadata   │  │
│  │  (nomic)     │    │   Index      │    │   Store      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                              │
│                              ▼                              │
│                    ┌──────────────────┐                    │
│                    │  Gateway (9 tools)│                   │
│                    │  compass, describe│                   │
│                    │  execute, etc.    │                   │
│                    └──────────────────┘                    │
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
| `compass(intent)` | Semantic search for tools |
| `describe(tool_name)` | Get full schema for a tool |
| `execute(tool_name, args)` | Run a tool on its backend |
| `compass_categories()` | List categories and servers |
| `compass_status()` | System health and config |
| `compass_analytics(timeframe)` | Usage statistics |
| `compass_chains(action)` | Manage tool workflows |
| `compass_sync(force)` | Rebuild index from backends |
| `compass_audit()` | Full system report |

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

**Default data directories:**
- **Windows:** `%LOCALAPPDATA%\tool-compass\`
- **macOS:** `~/Library/Application Support/tool-compass/`
- **Linux:** `~/.config/tool-compass/` (or `$XDG_CONFIG_HOME/tool-compass/`)

See [`.env.example`](.env.example) for all options.

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
python gateway.py --sync
```

## Related Projects

Part of the **Compass Suite** for AI-powered development:

- [File Compass](https://github.com/mikeyfrilot/file-compass) - Semantic file search
- [Integradio](https://github.com/mikeyfrilot/integradio) - Vector-embedded Gradio components
- [Backpropagate](https://github.com/mikeyfrilot/backpropagate) - Headless LLM fine-tuning
- [Comfy Headless](https://github.com/mikeyfrilot/comfy-headless) - ComfyUI without the complexity

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md). **Do not open public issues for security bugs.**

## License

[MIT](LICENSE) - see LICENSE file for details.

## Credits

- **HNSW**: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016)
- **nomic-embed-text**: Nomic AI's open embedding model
- **FastMCP**: Anthropic's MCP framework
- **Gradio**: Hugging Face's ML web framework

---

<div align="center">

*"Syntropy above all else."*

Tool Compass reduces entropy in the MCP ecosystem by organizing tools by semantic meaning.

**[Documentation](https://github.com/mcp-tool-shop/tool-compass#readme)** • **[Issues](https://github.com/mcp-tool-shop/tool-compass/issues)** • **[Discussions](https://github.com/mcp-tool-shop/tool-compass/discussions)**

</div>
