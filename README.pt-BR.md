<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
</p>

<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/tool-compass/readme.png" alt="Tool Compass Logo" width="400"></p>

**Navegador semântico para ferramentas MCP – Encontre a ferramenta certa com base na intenção, não na memória.**

<a href="https://github.com/mcp-tool-shop-org/tool-compass/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mcp-tool-shop-org/tool-compass/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/mcp-tool-shop-org/tool-compass"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/tool-compass?style=flat-square" alt="Codecov"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<a href="LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/tool-compass?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white" alt="Docker">
<a href="https://mcp-tool-shop-org.github.io/tool-compass/"><img src="https://img.shields.io/badge/Landing_Page-live-blue?style=flat-square" alt="Landing Page"></a>


*95% menos tokens. Encontre ferramentas descrevendo o que você deseja fazer.*

[Instalação](#quick-start) • [Uso](#usage) • [Docker](#option-2-docker) • [Manual](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) • [Desempenho](#performance) • [Contribuições](#contributing)

</div>

---

## O Problema

Servidores MCP expõem dezenas ou centenas de ferramentas. Carregar todas as definições de ferramentas no contexto desperdiça tokens e diminui a velocidade das respostas.

```
Before: 77 tools × ~500 tokens = 38,500 tokens per request
After:  1 compass tool + 3 results = ~2,000 tokens per request

Savings: 95%
```

## A Solução

O Tool Compass usa **busca semântica** para encontrar ferramentas relevantes a partir de uma descrição em linguagem natural. Em vez de carregar todas as ferramentas, o Claude chama `compass()` com uma intenção e recebe apenas as ferramentas relevantes.

## Início Rápido

📖 **Documentação completa:** Consulte o [Manual do Tool Compass](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) para obter informações detalhadas sobre instalação, configuração e arquitetura.

### Opção 1: npm (sem pré-requisitos, sem necessidade de instalar o Python)

```bash
npx @mcptoolshop/tool-compass --help
npx @mcptoolshop/tool-compass serve                 # MCP gateway
npx @mcptoolshop/tool-compass ui                    # Gradio UI
npx @mcptoolshop/tool-compass doctor                # Diagnose setup
npx @mcptoolshop/tool-compass execute fs:read_file '{"path":"README.md"}'  # Smoke-test a proxied call
```

Baixa um binário de plataforma verificado na primeira execução (SHA256 verificado em relação ao lançamento do GitHub). Armazenado localmente – invocações subsequentes são iniciadas instantaneamente. Consulte [@mcptoolshop/tool-compass](https://www.npmjs.com/package/@mcptoolshop/tool-compass) no npm.

### Opção 2: PyPI

```bash
pip install tool-compass
tool-compass --help
```

### Opção 3: Clone local

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

### Opção 4: Docker

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

> A imagem GHCR (`ghcr.io/mcp-tool-shop-org/tool-compass`) oferece suporte a
> `linux/amd64` e `linux/arm64`, portanto, a mesma tag funciona em servidores x86_64
> e estações de trabalho Apple Silicon / ARM.

## Recursos

- **Pesquisa Híbrida** - Semântica (HNSW) + fusão léxica com reforço de nome exato — descreva o que deseja ou cole o nome de uma ferramenta e ela será classificada como a primeira.
- **Divulgação Progressiva do Esquema Completo** - `compass()` → `describe()` → `execute()`; `describe()` retorna o `inputSchema` completo (campos obrigatórios, descrições, enumerações, valores padrão).
- **Backends stdio + HTTP** - Servidores MCP locais de subprocesso *e* servidores remotos/SaaS via streamable-http, com autenticação opcional por token.
- **Tempos limite e permissões/restrições por ferramenta** - Substitua o tempo limite padrão por backend/ferramenta; exponha um subconjunto seguro de um backend amplo.
- **Cache Dinâmico e Detecção de Cadeia** - Ferramentas frequentemente usadas pré-carregadas; fluxos de trabalho comuns detectados automaticamente.
- **Análise** - Rastreie padrões de uso e desempenho da ferramenta (com retenção/limpeza).
- **Compatível com várias plataformas e pronto para Docker** - Windows, macOS, Linux; implantação com um único comando.

## Arquitetura

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

## Uso

### A Ferramenta `compass()`

```python
compass(
    intent="I need to generate an AI image from a text description",
    top_k=3,
    category=None,  # Optional: "file", "git", "database", "ai", etc.
    min_confidence=0.3
)
```

Retorna:
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

### Ferramentas Disponíveis

| Ferramenta | Descrição |
|------|-------------|
| `compass(intent)` | Pesquisa híbrida semântica + léxica com reforço de nome exato. |
| `describe(tool_name)` | Obtenha o `inputSchema` completo para uma ferramenta (campos obrigatórios/enumerações/valores padrão). |
| `execute(tool_name, args)` | Execute uma ferramenta em seu backend |
| `compass_categories()` | Liste categorias e servidores |
| `compass_status(active)` | Saúde e configuração do sistema; `active=True` executa um teste de funcionamento em tempo real do backend. |
| `compass_analytics(timeframe)` | Estatísticas de uso |
| `compass_chains(action)` | Gerencie fluxos de trabalho de ferramentas |
| `compass_sync(force)` | Reconstrua o índice a partir dos backends |
| `compass_audit()` | Relatório completo do sistema |

As mesmas ações estão disponíveis na CLI — incluindo `tool-compass execute <ferramenta> '<json>'` para testar uma chamada proxy a partir do terminal.

### Padrão de Divulgação Progressiva

O Tool Compass usa um padrão de divulgação progressiva em três etapas para minimizar o uso de tokens:

```
1. compass("your intent")     → Get tool name + short description (~100 tokens)
2. describe("tool:name")      → Get full parameter schema (~500 tokens)
3. execute("tool:name", args) → Run the tool
```

**Por que isso é importante:**
- Carregar 77 ferramentas inicialmente = ~38.500 tokens
- Divulgação progressiva = ~600 tokens por ferramenta usada
- Economia: **95% ou mais para fluxos de trabalho típicos**

**Exemplo de fluxo de trabalho:**

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

O campo `hint` nos resultados do compass guia esse fluxo, sugerindo quando usar `describe()`.

## Configuração

| Variável | Descrição | Padrão |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | Diretório do projeto | Detectado automaticamente |
| `TOOL_COMPASS_PYTHON` | Executável Python | Detectado automaticamente |
| `TOOL_COMPASS_CONFIG` | Caminho do arquivo de configuração | `~/.config/tool-compass/compass_config.json` |
| `TOOL_COMPASS_DATA_DIR` | Diretório de dados | Específico da plataforma (veja abaixo) |
| `OLLAMA_URL` | URL do servidor Ollama | `http://localhost:11434` |
| `COMFYUI_URL` | Servidor ComfyUI | `http://localhost:8188` |
| `PORT` | Defina para habilitar o transporte HTTP (por exemplo, para Fly.io) | não definido (stdio) |
| `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` | Token necessário no transporte HTTP (opcional; substitui o campo de configuração `gateway_auth_token`). | desativado (sem autenticação). |

**Diretórios de dados padrão:**
- **Windows:** `%LOCALAPPDATA%\tool-compass\`
- **macOS:** `~/Library/Application Support/tool-compass/`
- **Linux:** `~/.config/tool-compass/` (ou `$XDG_CONFIG_HOME/tool-compass/`)

Configurações do arquivo de configuração (em `compass_config.json`) adicionadas na v2.5.0 — `hybrid_search`, `exact_name_boost`, `default_timeout` / `tool_timeouts` por backend, `allow_tools` / `deny_tools`, `analytics_retention_days` e backends HTTP (`type: "http"`) — estão documentados no [Handbook → Configuration](https://mcp-tool-shop-org.github.io/tool-compass/handbook/configuration/). Consulte o arquivo [`env.example`](.env.example) para opções de variáveis de ambiente.

## Desempenho

| Métrica | Valor |
|--------|-------|
| Tempo de construção do índice | ~5 segundos para 44 ferramentas |
| Latência da consulta | ~15 ms (incluindo a incorporação) |
| Economia de tokens | ~95% (38 mil → 2 mil) |
| Precisão@3 | ~95% (ferramenta correta no top 3) |

## Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## Solução de problemas

### Servidor MCP não está se conectando

Se os logs do Claude Desktop mostrarem erros de análise JSON:
```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**Causa:** As instruções `print()` corrompem o protocolo JSON-RPC.

**Correção:** Use registro ou `file=sys.stderr`:
```python
import sys
print("Debug message", file=sys.stderr)
```

### Falha na conexão com o Ollama

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### Índice não encontrado

```bash
tool-compass sync
```

## Projetos relacionados

Faz parte do **Pacote Compass** para desenvolvimento baseado em IA:

- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) – Busca semântica de arquivos
- [Integradio](https://github.com/mcp-tool-shop-org/integradio) – Componentes Gradio incorporados em vetores
- [Backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) – Ajuste fino de LLM sem interface gráfica
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) – ComfyUI sem a complexidade

## Contribuições

Agradecemos as contribuições! Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para obter diretrizes.

## Segurança e Escopo de Dados

O Tool Compass é uma ferramenta de desenvolvimento **local-first**. Consulte [SECURITY.md](SECURITY.md) para obter a política completa.

- **Dados acessados:** descrições de ferramentas indexadas no banco de dados vetorial HNSW local, consultas de pesquisa registradas no SQLite local (`compass_analytics.db`), incorporações geradas por meio do Ollama local.
- **Dados NÃO acessados:** nenhum código de usuário, nenhum conteúdo de arquivo, nenhuma credencial. Os argumentos das chamadas de ferramentas são criptografados, não armazenados em texto simples.
- **Rede:** conecta-se ao Ollama local para gerar incorporações. Interface Gradio opcional vinculada ao localhost. Sem telemetria externa.
- **Sem telemetria:** não coleta nada externamente. A análise é apenas local.

## Tabela de avaliação

As pontuações por categoria são regeneradas após a execução em lote por meio do comando:
`bash scripts/regenerate-scorecard.sh` (que envolve `npx @mcptoolshop/shipcheck audit`). Consulte [SCORECARD.md](SCORECARD.md) para obter a análise detalhada mais recente — a tabela abaixo é um espelho e não foi criada manualmente. As seções selecionadas manualmente (Lacunas Conhecidas, Histórico de Correção) estão fora das tags `<!-- SHIPCHECK-AUTO-START/END -->` no arquivo SCORECARD.md e permanecem após as regenerações.

Última auditoria `shipcheck`: **32 verificados · 0 não verificados · 5 ignorados · 100% aprovados — todas as barreiras rígidas foram superadas.**

| Categoria | Pontuação | Observações |
|----------|-------|-------|
| A. Segurança | ✅ Aprovado. | Ações com SHA fixo; imagem base com hash de digestão fixo; proveniência SLSA + SBOM no PyPI + GHCR; verificação pré-commit de segredos; autenticação opcional por token do gateway. |
| B. Tratamento de erros | ✅ Aprovado. | Resultados estruturados, degradação gradual, códigos de saída |
| C. Documentação para operadores | ✅ Aprovado. | README, CHANGELOG, LICENSE, Makefile `verify` + `verify-metrics` + `scorecard` |
| D. Boas práticas de distribuição | ✅ Aprovado. | CI consolidado; tempo limite em minutos + dias de retenção em cada tarefa; configuração pytest em pyproject.toml |
| E. Identidade (suave) | ✅ Aprovado. | Logotipo, página inicial, metadados do GitHub; mantenedores explícitos em pyproject.toml |
| **Total** | **100%** | Todas as barreiras rígidas foram superadas — gere novamente usando `make scorecard`. |

## Licença

[MIT](LICENSE) - consulte o arquivo LICENSE para obter detalhes.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>

