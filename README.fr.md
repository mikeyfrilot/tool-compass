<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<div align="center">

<p align="center"><img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/tool-compass/readme.png" alt="Tool Compass Logo" width="400"></p>

**Navigateur sémantique pour les outils MCP – Trouvez l’outil approprié en fonction de l’intention, et non de la mémoire.**

<a href="https://github.com/mcp-tool-shop-org/tool-compass/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mcp-tool-shop-org/tool-compass/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/mcp-tool-shop-org/tool-compass"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/tool-compass?style=flat-square" alt="Codecov"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
<a href="LICENSE"><img src="https://img.shields.io/github/license/mcp-tool-shop-org/tool-compass?style=flat-square" alt="License"></a>
<img src="https://img.shields.io/badge/docker-ready-blue?style=flat-square&logo=docker&logoColor=white" alt="Docker">
<a href="https://mcp-tool-shop-org.github.io/tool-compass/"><img src="https://img.shields.io/badge/Landing_Page-live-blue?style=flat-square" alt="Landing Page"></a>


*95 % moins de jetons. Trouvez des outils en décrivant ce que vous voulez faire.*

[Installation](#quick-start) • [Utilisation](#usage) • [Docker](#option-2-docker) • [Guide d’utilisation](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) • [Performances](#performance) • [Contribution](#contributing)

</div>

---

## Le problème

Les serveurs MCP exposent des dizaines, voire des centaines d’outils. Le chargement de toutes les définitions d’outils dans le contexte gaspille des jetons et ralentit les réponses.

```
Before: 77 tools × ~500 tokens = 38,500 tokens per request
After:  1 compass tool + 3 results = ~2,000 tokens per request

Savings: 95%
```

## La solution

Tool Compass utilise la **recherche sémantique** pour trouver les outils pertinents à partir d’une description en langage naturel. Au lieu de charger tous les outils, Claude appelle `compass()` avec une intention et renvoie uniquement les outils pertinents.

## Démarrage rapide

📖 **Documentation complète :** Consultez le [Guide d’utilisation de Tool Compass](https://mcp-tool-shop-org.github.io/tool-compass/handbook/) pour l’installation, la configuration et une analyse approfondie de l’architecture.

### Option 1 : npm (aucune condition préalable, aucune installation de Python)

```bash
npx @mcptoolshop/tool-compass --help
npx @mcptoolshop/tool-compass serve                 # MCP gateway
npx @mcptoolshop/tool-compass ui                    # Gradio UI
npx @mcptoolshop/tool-compass doctor                # Diagnose setup
npx @mcptoolshop/tool-compass execute fs:read_file '{"path":"README.md"}'  # Smoke-test a proxied call
```

Télécharge un binaire de plateforme vérifié lors du premier lancement (SHA256 vérifié par rapport à la version GitHub). Mis en cache localement – les invocations suivantes se lancent instantanément. Consultez [@mcptoolshop/tool-compass](https://www.npmjs.com/package/@mcptoolshop/tool-compass) sur npm.

### Option 2 : PyPI

```bash
pip install tool-compass
tool-compass --help
```

### Option 3 : Clone local

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

### Option 4 : Docker

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

> L’image GHCR (`ghcr.io/mcp-tool-shop-org/tool-compass`) prend en charge
> `linux/amd64` et `linux/arm64`, de sorte que le même tag fonctionne sur les serveurs x86_64
> et les postes de travail Apple Silicon / ARM.

## Fonctionnalités

- **Recherche hybride** – Sémantique (HNSW) + fusion lexicale avec un renforcement du nom exact : décrivez ce que vous voulez, ou collez le nom d’un outil et il sera classé en première position.
- **Divulgation progressive complète du schéma** – `compass()` → `describe()` → `execute()` ; `describe()` renvoie le `inputSchema` complet (champs obligatoires, descriptions, énumérations, valeurs par défaut).
- **Backends stdio + HTTP** – Prend en charge les serveurs MCP locaux et distants/SaaS via http diffusible, avec une authentification facultative par jeton.
- **Délai d’attente et autorisation/refus par outil** – Remplacez le délai d’attente par défaut pour chaque backend/outil ; exposez un sous-ensemble sûr d’un backend étendu.
- **Cache dynamique et détection de chaîne** – Les outils fréquemment utilisés sont préchargés ; les flux de travail courants sont détectés automatiquement.
- **Analytique** – Suivez les modèles d’utilisation et les performances des outils (avec conservation/suppression).
- **Multiplateforme et prêt pour Docker** – Windows, macOS, Linux ; déploiement en une seule commande.

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

## Utilisation

### L’outil `compass()`

```python
compass(
    intent="I need to generate an AI image from a text description",
    top_k=3,
    category=None,  # Optional: "file", "git", "database", "ai", etc.
    min_confidence=0.3
)
```

Renvoie :
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

### Outils disponibles

| Outil | Description |
|------|-------------|
| `compass(intent)` | Recherche sémantique et lexicale hybride avec un renforcement du nom exact. |
| `describe(tool_name)` | Obtenez le `inputSchema` complet pour un outil (champs obligatoires/énumérations/valeurs par défaut). |
| `execute(tool_name, args)` | Exécutez un outil sur son backend. |
| `compass_categories()` | Affichez les catégories et les serveurs. |
| `compass_status(active)` | État du système et configuration ; `active=True` exécute une sonde de réactivité du backend en direct. |
| `compass_analytics(timeframe)` | Statistiques d’utilisation. |
| `compass_chains(action)` | Gérez les flux de travail des outils. |
| `compass_sync(force)` | Reconstruisez l’index à partir des backends. |
| `compass_audit()` | Rapport complet du système. |

Les mêmes actions sont disponibles depuis la ligne de commande, y compris `tool-compass execute <outil> '<json>'` pour tester un appel proxy depuis le terminal.

### Modèle de divulgation progressive

Tool Compass utilise un modèle de divulgation progressive en trois étapes afin de minimiser l’utilisation des jetons :

```
1. compass("your intent")     → Get tool name + short description (~100 tokens)
2. describe("tool:name")      → Get full parameter schema (~500 tokens)
3. execute("tool:name", args) → Run the tool
```

**Pourquoi c’est important :**
- Chargement initial de 77 outils = ~38 500 jetons.
- Divulgation progressive = ~600 jetons par outil utilisé.
- Économies : **95 % ou plus pour les flux de travail typiques**.

**Exemple de flux de travail :**

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

Le champ `hint` dans les résultats de compass guide ce flux, en suggérant quand utiliser `describe()`.

## Configuration

| Variable | Description | Valeur par défaut |
|----------|-------------|---------|
| `TOOL_COMPASS_BASE_PATH` | Répertoire du projet | Détecté automatiquement |
| `TOOL_COMPASS_PYTHON` | Exécutable Python | Détecté automatiquement |
| `TOOL_COMPASS_CONFIG` | Chemin d’accès au fichier de configuration | `~/.config/tool-compass/compass_config.json` |
| `TOOL_COMPASS_DATA_DIR` | Répertoire des données | Spécifique à la plateforme (voir ci-dessous) |
| `OLLAMA_URL` | URL du serveur Ollama | `http://localhost:11434` |
| `COMFYUI_URL` | Serveur ComfyUI | `http://localhost:8188` |
| `PORT` | Définissez pour activer le transport HTTP (par exemple, pour Fly.io). | non défini (stdio) |
| `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` | Jeton de porteur requis sur le transport HTTP (optionnel ; remplace le champ de configuration `gateway_auth_token`). | non défini (aucune authentification). |

**Répertoires de données par défaut :**
- **Windows :** `%LOCALAPPDATA%\tool-compass\`
- **macOS :** `~/Library/Application Support/tool-compass/`
- **Linux :** `~/.config/tool-compass/` (ou `$XDG_CONFIG_HOME/tool-compass/`)

Paramètres du fichier de configuration (dans `compass_config.json`) ajoutés dans la version 2.5.0 – `hybrid_search`,
`exact_name_boost`, `default_timeout`/`tool_timeouts` par backend,
`allow_tools`/`deny_tools`, `analytics_retention_days` et backends HTTP (`type : "http"`) – sont documentés dans le [Guide d’utilisation → Configuration](https://mcp-tool-shop-org.github.io/tool-compass/handbook/configuration/).
Consultez [`.env.example`](.env.example) pour les options de variables d’environnement.

## Performances

| Métrique | Valeur |
|--------|-------|
| Temps de construction de l’index | ~5 s pour 44 outils. |
| Latence des requêtes | ~15 ms (y compris l’intégration). |
| Économie de jetons | ~95 % (38 000 → 2 000) |
| Précision à 3 | ~95 % (l’outil correct dans les 3 premiers). |

## Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Skip integration tests (no Ollama required)
pytest -m "not integration"
```

## Dépannage

### Le serveur MCP ne se connecte pas

Si les journaux de Claude Desktop affichent des erreurs d’analyse JSON :
```
Unexpected token 'S', "Starting T"... is not valid JSON
```

**Cause :** Les instructions `print()` corrompent le protocole JSON-RPC.

**Solution :** Utilisez la journalisation ou `file=sys.stderr :`.
```python
import sys
print("Debug message", file=sys.stderr)
```

### Échec de la connexion à Ollama

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull the embedding model
ollama pull nomic-embed-text
```

### Index introuvable

```bash
tool-compass sync
```

## Projets connexes

Fait partie de la **suite Compass**, pour un développement basé sur l’IA :

- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) - Recherche sémantique de fichiers
- [Integradio](https://github.com/mcp-tool-shop-org/integradio) - Composants Gradio avec intégration vectorielle
- [Backpropagate](https://github.com/mcp-tool-shop-org/backpropagate) - Ajustement fin d’un LLM sans interface graphique
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) - ComfyUI sans la complexité

## Contributions

Nous acceptons les contributions ! Consultez le fichier [CONTRIBUTING.md](CONTRIBUTING.md) pour connaître les directives.

## Sécurité et portée des données

Tool Compass est un outil de développement **d’abord local**. Consultez le fichier [SECURITY.md](SECURITY.md) pour obtenir la politique complète.

- **Données concernées :** descriptions d’outils indexées dans une base de données vectorielle HNSW locale, requêtes de recherche enregistrées dans une base de données SQLite locale (`compass_analytics.db`), intégrations générées via Ollama local.
- **Données non concernées :** aucun code utilisateur, aucun contenu de fichier, aucune information d’identification. Les arguments des appels d’outils sont hachés et ne sont pas stockés en texte clair.
- **Réseau :** se connecte à Ollama local pour les intégrations. L’interface utilisateur Gradio facultative est liée à localhost. Aucune télémétrie externe.
- **Aucune télémétrie :** ne collecte aucune donnée en externe. Les analyses sont uniquement locales.

## Tableau de bord

Les scores par catégorie sont régénérés après l’exécution du script via
`bash scripts/regenerate-scorecard.sh` (qui encapsule `npx @mcptoolshop/shipcheck audit`). Consultez le fichier [SCORECARD.md](SCORECARD.md) pour obtenir la version actuelle et définitive ; le tableau ci-dessous est un reflet de celle-ci et n’est pas créé manuellement. Les sections créées manuellement (Lacunes connues, Historique des corrections) se trouvent en dehors des marqueurs `<!-- SHIPCHECK-AUTO-START/END -->` dans SCORECARD.md et sont conservées lors des régénérations.

Dernier audit `shipcheck` : **32 éléments vérifiés · 0 éléments non vérifiés · 5 éléments ignorés · 100 % de réussite — tous les seuils obligatoires sont respectés.**

| Catégorie | Score | Notes |
|----------|-------|-------|
| A. Sécurité | ✅ Réussi | Actions avec hachage SHA ; image de base avec hachage de digest ; provenance SLSA + SBOM sur PyPI + GHCR ; analyse des secrets pré-commit ; authentification par jeton d’accès facultative. |
| B. Gestion des erreurs | ✅ Réussi | Résultats structurés, dégradation progressive, codes de sortie |
| C. Documentation pour les opérateurs | ✅ Réussi | README, CHANGELOG, LICENSE, Makefile `verify` + `verify-metrics` + `scorecard` |
| D. Hygiène d’expédition | ✅ Réussi | CI consolidé ; timeout-minutes + retention-days sur chaque tâche ; configuration pytest dans pyproject.toml |
| E. Identité (souple) | ✅ Réussi | Logo, page d’accueil, métadonnées GitHub ; mainteneurs explicites dans pyproject.toml |
| **Total** | **100%** | Tous les seuils obligatoires sont respectés — régénérez via `make scorecard` |

## Licence

[MIT](LICENSE) - consultez le fichier LICENSE pour plus de détails.

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>

