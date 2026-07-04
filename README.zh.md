<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/tool-compass/readme.png" alt="Tool Compass Logo" width="400"></p>

**用于 MCP 工具的语义导航器——通过意图而非记忆来查找正确的工具**

<a href="https://github.com/mcp-tool-shop-org/tool-compass/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mcp-tool-shop-org/tool-compass/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/mcp-tool-shop-org/tool-compass"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/tool-compass?style=flat-square" alt="Codecov"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<a href="LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/tool-compass?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white" alt="Docker">
<a href="https://mcp-tool-shop-org.github.io/tool-compass/"><img src="https://img.shields.io/badge/Landing_Page-live-blue?style=flat-square" alt="Landing Page"></a>


*令牌数量减少 95%。 通过描述您想要执行的操作来查找工具。*

[安装](#quick-start) • [用法](#usage) • [Docker](#option-2-docker) • [手册](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) • [性能](#performance) • [贡献](#contributing)

</div>

---

## 问题

MCP 服务器提供数十甚至数百个工具。 将所有工具定义加载到上下文中会浪费令牌并降低响应速度。

```
Before: 77 tools × ~500 tokens = 38,500 tokens per request
After:  1 compass tool + 3 results = ~2,000 tokens per request

Savings: 95%
```

## 解决方案

Tool Compass 使用**语义搜索**从自然语言描述中查找相关工具。 与其加载所有工具，不如让 Claude 调用 `compass()` 并提供意图，然后仅返回相关的工具。

## 快速入门

📖 **完整文档：** 请参阅 [Tool Compass 手册](https://mcp-tool-shop-org.github.io/tool-compass/handbook/)，了解有关安装、配置和架构的详细信息。

### 选项 1：npm（零先决条件，无需安装 Python）

```bash
npx @mcptoolshop/tool-compass --help
npx @mcptoolshop/tool-compass serve                 # MCP gateway
npx @mcptoolshop/tool-compass ui                    # Gradio UI
npx @mcptoolshop/tool-compass doctor                # Diagnose setup
npx @mcptoolshop/tool-compass execute fs:read_file '{"path":"README.md"}'  # Smoke-test a proxied call
```

首次运行时，它会下载经过验证的平台二进制文件（与 GitHub 发布中的 SHA256 值进行检查）。 本地缓存——后续调用可以立即启动。 请参阅 npm 上的 [@mcptoolshop/tool-compass](https://www.npmjs.com/package/@mcptoolshop/tool-compass)。

### 选项 2：PyPI

```bash
pip install tool-compass
tool-compass --help
```

### 选项 3：本地克隆

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

### 选项 4：Docker

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

> GHCR 镜像 (`ghcr.io/mcp-tool-shop-org/tool-compass`) 支持 `linux/amd64` 和 `linux/arm64`，因此相同的标签可以在 x86_64 服务器和 Apple Silicon / ARM 工作站上运行。

## 特性

- **混合搜索**——语义（HNSW）+ 词法融合，并增强了精确名称匹配——描述您想要的内容，或者粘贴工具名称，它将排名第一。
- **完整模式的逐步呈现**——`compass()` → `describe()` → `execute()`；`describe()` 返回完整的 `inputSchema`（必需字段、描述、枚举、默认值）。
- **stdio + HTTP 后端**——支持本地子进程 MCP 服务器*以及*通过可流式传输的 HTTP 访问远程/SaaS 服务器，并可选地使用 bearer 令牌进行身份验证。
- **每个工具的超时时间和允许/拒绝列表**——覆盖每个后端/工具的默认超时时间；公开广泛后端中的安全子集。
- **热缓存和链检测**——常用工具预加载；自动发现常见的工具工作流程。
- **分析**——跟踪使用模式和工具性能（具有保留/清理功能）。
- **跨平台且支持 Docker**——Windows、macOS、Linux；一键部署。

## 架构

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

## 用法

### `compass()` 工具

```python
compass(
    intent="I need to generate an AI image from a text description",
    top_k=3,
    category=None,  # Optional: "file", "git", "database", "ai", etc.
    min_confidence=0.3
)
```

返回值：
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

### 可用工具

| 工具 | 描述 |
|------|-------------|
| `compass(intent)` | 混合语义 + 词法搜索，并增强了精确名称匹配 |
| `describe(tool_name)` | 获取工具的完整 `inputSchema`（必需字段/枚举/默认值） |
| `execute(tool_name, args)` | 在后端上运行工具 |
| `compass_categories()` | 列出类别和服务器 |
| `compass_status(active)` | 系统健康状况和配置；`active=True` 运行实时后端存活探测。 |
| `compass_analytics(timeframe)` | 使用情况统计信息 |
| `compass_chains(action)` | 管理工具工作流程 |
| `compass_sync(force)` | 从后端重新构建索引 |
| `compass_audit()` | 完整的系统报告 |

可以通过 CLI 使用相同的操作——包括 `tool-compass execute <tool> '<json>'`，以测试从终端代理调用的功能。

### 逐步呈现模式

Tool Compass 采用三步式逐步呈现模式来最大限度地减少令牌使用量：

```
1. compass("your intent")     → Get tool name + short description (~100 tokens)
2. describe("tool:name")      → Get full parameter schema (~500 tokens)
3. execute("tool:name", args) → Run the tool
```

**为什么这很重要：**
- 预加载 77 个工具 = ~38,500 个令牌
- 逐步呈现 = 每个使用的工具 ~600 个令牌
- 节省：**对于典型的流程，可节省 95% 以上。**

**示例工作流程：**

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

`compass` 结果中的 `hint` 字段指导此流程，并建议何时使用 `describe()`。

## 配置

| 变量 | 描述 | 默认值 |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | 项目根目录 | 自动检测 |
| `TOOL_COMPASS_PYTHON` | Python 可执行文件 | 自动检测 |
| `TOOL_COMPASS_CONFIG` | 配置文件路径 | `~/.config/tool-compass/compass_config.json` |
| `TOOL_COMPASS_DATA_DIR` | 数据目录 | 特定于平台（如下所示） |
| `OLLAMA_URL` | Ollama 服务器 URL | `http://localhost:11434` |
| `COMFYUI_URL` | ComfyUI 服务器 | `http://localhost:8188` |
| `PORT` | 设置为启用 HTTP 传输（例如，用于 Fly.io）。 | 未设置（stdio） |
| `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` | HTTP 传输上所需的 bearer 令牌（可选；覆盖 `gateway_auth_token` 配置字段）。 | 未设置（无身份验证） |

**默认数据目录：**
- **Windows：** `%LOCALAPPDATA%\tool-compass\`
- **macOS：** `~/Library/Application Support/tool-compass/`
- **Linux：** `~/.config/tool-compass/`（或 `$XDG_CONFIG_HOME/tool-compass/`）

配置文件设置（在 `compass_config.json` 中），于 v2.5.0 中添加——`hybrid_search`、`exact_name_boost`、每个后端的 `default_timeout`/`tool_timeouts`、`allow_tools`/`deny_tools`、`analytics_retention_days` 以及 HTTP（`type: "http"`）后端——在 [手册 → 配置](https://mcp-tool-shop-org.github.io/tool-compass/handbook/configuration/) 中进行了记录。 请参阅 [.env.example](.env.example)，了解环境变量选项。

## 性能

| 指标 | 值 |
|--------|-------|
| 索引构建时间 | ~5 秒（对于 44 个工具） |
| 查询延迟 | ~15 毫秒（包括嵌入） |
| 令牌节省 | ~95%（38K → 2K） |
| 准确率@3 | ~95%（前 3 个工具中有正确的工具） |

## 测试

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## 故障排除

### MCP 服务器未连接

如果 Claude Desktop 日志显示 JSON 解析错误：
```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**原因**：`print()` 语句会破坏 JSON-RPC 协议。

**解决方法**：使用日志记录或 `file=sys.stderr`。
```python
import sys
print("Debug message", file=sys.stderr)
```

### Ollama 连接失败

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### 未找到索引

```bash
tool-compass sync
```

## 相关项目

作为基于 AI 的开发工具包 **Compass Suite** 的一部分：

- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) - 语义文件搜索。
- [Integradio](https://github.com/mcp-tool-shop-org/integradio) - 基于向量嵌入的 Gradio 组件。
- [Backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) - 无头 LLM 微调。
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) - 没有复杂功能的 ComfyUI。

## 贡献

我们欢迎大家的贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)，了解相关指南。

## 安全与数据范围

Tool Compass 是一款**以本地优先**的开发工具。请参阅 [SECURITY.md](SECURITY.md) 以获取完整策略。

- **涉及的数据：**存储在本地 HNSW 向量数据库中的工具描述，搜索查询记录到本地 SQLite (`compass_analytics.db`)，通过本地 Ollama 生成的嵌入数据。
- **不涉及的数据：**没有用户代码、文件内容或凭据。工具调用参数会被哈希处理，而不是以明文形式存储。
- **网络：**连接到本地 Ollama 以进行嵌入操作。可选的 Gradio UI 绑定到 localhost。没有外部遥测数据。
- **无遥测数据：**不收集任何外部数据。分析仅限于本地。

## 评估报告

每个类别的分数会在集群完成后重新生成，方法是使用 `bash scripts/regenerate-scorecard.sh`（该脚本封装了 `npx @mcptoolshop/shipcheck audit`）。请参阅 [SCORECARD.md](SCORECARD.md)，以获取当前的权威分解报告——下表是对其的镜像，并且并非手动编写。经过人工整理的部分（已知差距、修复历史记录）位于 SCORECARD.md 中的 `<!-- SHIPCHECK-AUTO-START/END -->` 标记之外，并在重新生成时保留。

最新的 `shipcheck audit`：**已检查 32 项 · 未检查 0 项 · 跳过 5 项 · 通过率 100%——所有硬性要求均已通过。**

| 类别。 | 分数。 | 备注。 |
|----------|-------|-------|
| A. 安全。 | ✅ 通过。 | SHA 固定操作；摘要固定的基础镜像；SLSA 溯源 + PyPI 和 GHCR 上的 SBOM；预提交的密钥扫描；可选的网关承载令牌身份验证。 |
| B. 错误处理。 | ✅ 通过。 | 结构化结果、优雅降级、退出代码。 |
| C. 操作文档。 | ✅ 通过。 | README、CHANGELOG、LICENSE、Makefile `verify` + `verify-metrics` + `scorecard`。 |
| D. 发布规范。 | ✅ 通过。 | CI 整合；每个作业的 timeout-minutes + retention-days；pytest 配置位于 pyproject.toml 中。 |
| E. 身份（软性要求）。 | ✅ 通过。 | 徽标、登录页面、GitHub 元数据；在 pyproject.toml 中明确指定维护者。 |
| **Total** | **100%** | 所有硬性要求均已通过——通过 `make scorecard` 进行重新生成。 |

## 许可证

[MIT](LICENSE) - 详情请参阅 LICENSE 文件。

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>

