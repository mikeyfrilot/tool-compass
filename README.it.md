<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/tool-compass/readme.png" alt="Tool Compass Logo" width="400"></p>

**Motore semantico per gli strumenti MCP: trova lo strumento giusto in base all'intento, non alla memoria.**

<a href="https://github.com/mcp-tool-shop-org/tool-compass/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mcp-tool-shop-org/tool-compass/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/mcp-tool-shop-org/tool-compass"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/tool-compass?style=flat-square" alt="Codecov"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<a href="LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/tool-compass?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white" alt="Docker">
<a href="https://mcp-tool-shop-org.github.io/tool-compass/"><img src="https://img.shields.io/badge/Landing_Page-live-blue?style=flat-square" alt="Landing Page"></a>


*Riduzione del 95% dei token. Trova gli strumenti descrivendo cosa vuoi fare.*

[Installazione](#quick-start) • [Utilizzo](#usage) • [Docker](#option-2-docker) • [Manuale](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) • [Prestazioni](#performance) • [Contributi](#contributing)

</div>

---

## Il problema

I server MCP espongono decine o centinaia di strumenti. Caricare tutte le definizioni degli strumenti nel contesto spreca token e rallenta le risposte.

```
Before: 77 tools × ~500 tokens = 38,500 tokens per request
After:  1 compass tool + 3 results = ~2,000 tokens per request

Savings: 95%
```

## La soluzione

Tool Compass utilizza la **ricerca semantica** per trovare gli strumenti pertinenti da una descrizione in linguaggio naturale. Invece di caricare tutti gli strumenti, Claude chiama `compass()` con un intento e riceve solo gli strumenti rilevanti.

## Avvio rapido

📖 **Documentazione completa:** Consulta il [Manuale di Tool Compass](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) per l'installazione, la configurazione e un approfondimento dell'architettura.

### Opzione 1: npm (nessun prerequisito, nessuna installazione di Python)

```bash
npx @mcptoolshop/tool-compass --help
npx @mcptoolshop/tool-compass serve                 # MCP gateway
npx @mcptoolshop/tool-compass ui                    # Gradio UI
npx @mcptoolshop/tool-compass doctor                # Diagnose setup
npx @mcptoolshop/tool-compass execute fs:read_file '{"path":"README.md"}'  # Smoke-test a proxied call
```

Scarica un binario della piattaforma verificato alla prima esecuzione (SHA256 controllato rispetto al rilascio su GitHub). Memorizzato in cache localmente: le invocazioni successive vengono eseguite istantaneamente. Consulta [@mcptoolshop/tool-compass](https://www.npmjs.com/package/@mcptoolshop/tool-compass) su npm.

### Opzione 2: PyPI

```bash
pip install tool-compass
tool-compass --help
```

### Opzione 3: Clonazione locale

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

### Opzione 4: Docker

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

> L'immagine GHCR (`ghcr.io/mcp-tool-shop-org/tool-compass`) supporta
> `linux/amd64` e `linux/arm64`, quindi lo stesso tag funziona su server x86_64
> e workstation Apple Silicon / ARM.

## Funzionalità

- **Ricerca ibrida:** Semantica (HNSW) + fusione lessicale con potenziamento del nome esatto: descrivi cosa vuoi o incolla il nome di uno strumento e questo otterrà il punteggio più alto.
- **Divulgazione progressiva dello schema completo:** `compass()` → `describe()` → `execute()`; `describe()` restituisce lo `inputSchema` completo (campi obbligatori, descrizioni, enumerazioni, valori predefiniti).
- **Backend stdio + HTTP:** Esegue i server MCP locali e i server remoti/SaaS tramite streamable-http, con autenticazione opzionale tramite token bearer.
- **Timeout per strumento e autorizzazione/negazione:** Sovrascrivi il timeout predefinito per ciascun backend/strumento; esponi un sottoinsieme sicuro di un backend ampio.
- **Cache dinamica e rilevamento della catena:** Gli strumenti utilizzati frequentemente vengono caricati in anticipo; i flussi di lavoro comuni degli strumenti vengono scoperti automaticamente.
- **Analisi:** Traccia i modelli di utilizzo e le prestazioni degli strumenti (con conservazione/eliminazione).
- **Compatibilità multipiattaforma e pronto per Docker:** Windows, macOS, Linux; distribuzione con un solo comando.

## Architettura

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

## Utilizzo

### Lo strumento `compass()`

```python
compass(
    intent="I need to generate an AI image from a text description",
    top_k=3,
    category=None,  # Optional: "file", "git", "database", "ai", etc.
    min_confidence=0.3
)
```

Restituisce:
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

### Strumenti disponibili

| Strumento | Descrizione |
|------|-------------|
| `compass(intent)` | Ricerca ibrida semantica + lessicale con potenziamento del nome esatto |
| `describe(tool_name)` | Ottieni lo `inputSchema` completo per uno strumento (campi obbligatori/enumerazioni/valori predefiniti) |
| `execute(tool_name, args)` | Esegui uno strumento sul suo backend |
| `compass_categories()` | Elenca le categorie e i server |
| `compass_status(active)` | Stato del sistema e configurazione; `active=True` esegue un probe di attività del backend in tempo reale. |
| `compass_analytics(timeframe)` | Statistiche sull'utilizzo |
| `compass_chains(action)` | Gestisci i flussi di lavoro degli strumenti |
| `compass_sync(force)` | Ricostruisci l'indice dai backend |
| `compass_audit()` | Report completo del sistema |

Le stesse azioni sono disponibili dalla CLI, incluso `tool-compass execute <strumento> '<json>'` per eseguire un test di uno strumento tramite proxy dal terminale.

### Modello di divulgazione progressiva

Tool Compass utilizza un modello di divulgazione progressiva in tre fasi per ridurre al minimo l'utilizzo dei token:

```
1. compass("your intent")     → Get tool name + short description (~100 tokens)
2. describe("tool:name")      → Get full parameter schema (~500 tokens)
3. execute("tool:name", args) → Run the tool
```

**Perché è importante:**
- Caricamento iniziale di 77 strumenti = ~38.500 token
- Divulgazione progressiva = ~600 token per strumento utilizzato
- Risparmio: **95% o più per i flussi di lavoro tipici**

**Esempio di flusso di lavoro:**

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

Il campo `hint` nei risultati di compass guida questo flusso, suggerendo quando utilizzare `describe()`.

## Configurazione

| Variabile | Descrizione | Predefinito |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | Directory del progetto | Rilevata automaticamente |
| `TOOL_COMPASS_PYTHON` | Eseguibile Python | Rilevata automaticamente |
| `TOOL_COMPASS_CONFIG` | Percorso del file di configurazione | `~/.config/tool-compass/compass_config.json` |
| `TOOL_COMPASS_DATA_DIR` | Directory dei dati | Specifica per la piattaforma (vedi sotto) |
| `OLLAMA_URL` | URL del server Ollama | `http://localhost:11434` |
| `COMFYUI_URL` | Server ComfyUI | `http://localhost:8188` |
| `PORT` | Imposta per abilitare il trasporto HTTP (ad esempio, per Fly.io) | non impostato (stdio) |
| `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` | Token bearer richiesto sul trasporto HTTP (opzionale; sovrascrive il campo di configurazione `gateway_auth_token`) | non impostato (nessuna autenticazione) |

**Directory dei dati predefinite:**
- **Windows:** `%LOCALAPPDATA%\tool-compass\`
- **macOS:** `~/Library/Application Support/tool-compass/`
- **Linux:** `~/.config/tool-compass/` (o `$XDG_CONFIG_HOME/tool-compass/`)

Impostazioni del file di configurazione (in `compass_config.json`) aggiunte nella versione 2.5.0: `hybrid_search`,
`exact_name_boost`, `default_timeout` / `tool_timeouts` per ciascun backend,
`allow_tools` / `deny_tools`, `analytics_retention_days` e backend HTTP (`type: "http"`)
— sono documentate nel [Manuale → Configurazione](https://mcp-tool-shop-org.github.io/tool-compass/handbook/configuration/).
Consulta [`.env.example`](.env.example) per le opzioni delle variabili d'ambiente.

## Prestazioni

| Metrica | Valore |
|--------|-------|
| Tempo di creazione dell'indice | ~5 secondi per 44 strumenti |
| Latenza della query | ~15 ms (inclusa l'incorporazione) |
| Risparmio di token | ~95% (38K → 2K) |
| Precisione@3 | ~95% (strumento corretto tra i primi 3) |

## Test

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## Risoluzione dei problemi

### Il server MCP non si connette

Se i log di Claude Desktop mostrano errori di analisi JSON:
```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**Causa**: le istruzioni `print()` corrompono il protocollo JSON-RPC.

**Soluzione**: utilizzare la registrazione (logging) o `file=sys.stderr`:
```python
import sys
print("Debug message", file=sys.stderr)
```

### Connessione a Ollama non riuscita

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### Indice non trovato

```bash
tool-compass sync
```

## Progetti correlati

Parte della **Compass Suite** per lo sviluppo basato sull'intelligenza artificiale:

- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) - Ricerca semantica di file
- [Integradio](https://github.com/mcp-tool-shop-org/integradio) - Componenti Gradio con incorporamenti vettoriali
- [Backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) - Fine-tuning di LLM senza interfaccia grafica
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) - ComfyUI senza complessità aggiuntive

## Contributi

Accogliamo volentieri i contributi! Consultare [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.

## Sicurezza e ambito dei dati

Tool Compass è uno strumento di sviluppo **local-first**. Consultare [SECURITY.md](SECURITY.md) per la politica completa.

- **Dati elaborati:** descrizioni degli strumenti indicizzate nel database vettoriale HNSW locale, query di ricerca registrate in SQLite locale (`compass_analytics.db`), incorporamenti generati tramite Ollama locale.
- **Dati NON elaborati:** nessun codice utente, nessun contenuto dei file, nessuna credenziale. Gli argomenti delle chiamate agli strumenti vengono sottoposti a hashing e non vengono memorizzati in testo semplice.
- **Rete:** si connette a Ollama locale per gli incorporamenti. L'interfaccia utente Gradio opzionale è collegata a localhost. Nessuna telemetria esterna.
- **Nessuna telemetria:** non raccoglie nulla esternamente. L'analisi dei dati è solo locale.

## Valutazione

I punteggi per categoria vengono rigenerati dopo l'esecuzione del processo tramite
`bash scripts/regenerate-scorecard.sh` (che include `npx @mcptoolshop/shipcheck audit`). Consultare [SCORECARD.md](SCORECARD.md) per la versione corrente e definitiva; la tabella sottostante la rispecchia e non è stata creata manualmente. Le sezioni curate manualmente (Lacune note, Cronologia delle correzioni) si trovano al di fuori dei marcatori `<!-- SHIPCHECK-AUTO-START/END -->` in SCORECARD.md e sopravvivono alle rigenerazioni.

Ultima esecuzione di `shipcheck audit`: **32 elementi controllati · 0 elementi non controllati · 5 elementi saltati · 100% superato — tutti i criteri obbligatori sono soddisfatti.**

| Categoria | Punteggio | Note |
|----------|-------|-------|
| A. Sicurezza | ✅ Superato | Azioni con SHA; immagine di base con digest; provenienza SLSA + SBOM su PyPI + GHCR; scansione dei segreti pre-commit; autenticazione bearer del gateway opzionale. |
| B. Gestione degli errori | ✅ Superato | Risultati strutturati, degradazione controllata, codici di uscita |
| C. Documentazione per l'utente | ✅ Superato | README, CHANGELOG, LICENSE, Makefile `verify` + `verify-metrics` + `scorecard` |
| D. Pratiche di sviluppo | ✅ Superato | CI consolidato; timeout-minutes + retention-days su ogni job; configurazione pytest in pyproject.toml |
| E. Identità (soft) | ✅ Superato | Logo, pagina di destinazione, metadati GitHub; manutentori espliciti in pyproject.toml |
| **Total** | **100%** | Tutti i criteri obbligatori sono soddisfatti — rigenerare tramite `make scorecard` |

## Licenza

[MIT](LICENSE) - consultare il file LICENSE per maggiori dettagli.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>

