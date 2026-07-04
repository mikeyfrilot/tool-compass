# Changelog

All notable changes to Tool Compass will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.0] - 2026-07-04

Dogfood swarm v4 — a composed re-audit health pass (Stage A bug/security →
B/C/D proactive/humanization/visual) followed by a 12-feature pass, each
feature closed by an adversarial cross-wave re-audit. Tests 1148 → 1337
(88.6% coverage); ruff clean. No breaking API changes — every new config
field defaults to today's behavior and all existing tool/CLI responses keep
their shape. Two behavior changes are called out under **Changed**.

### Added
- **HTTP / streamable-http backend transport.** Backends of type `http` now
  connect at runtime via the MCP SDK's Streamable HTTP client — tool-compass
  can front remote / SaaS MCP servers, not just local stdio subprocesses.
  `url` + `headers` (headers redacted in `doctor` output) as before.
- **Optional bearer-token gateway auth** (`gateway_auth_token`, or the
  `TOOL_COMPASS_GATEWAY_AUTH_TOKEN` env var). When set, the HTTP transport
  requires `Authorization: Bearer <token>` (constant-time compare, `/health`
  and `/ready` stay open). Opt-in and fail-closed only when enabled; the stdio
  transport is never affected.
- **Hybrid search + exact-name boost.** `compass()` now fuses the semantic
  (HNSW) and lexical signals with Reciprocal Rank Fusion on the healthy path
  (not only as an Ollama-outage fallback), and an exact tool-name paste
  (`fs:read_file`, or the bare `read_file`) ranks #1. Toggles: `hybrid_search`,
  `exact_name_boost`, `exact_match_confidence`.
- **Full-schema `describe()`.** The complete JSON `inputSchema`
  (required fields, per-parameter descriptions, enums, defaults) is now
  preserved through sync and returned by `describe()` alongside the collapsed
  `parameters` — so an agent can build a correct `execute()` call in one hop.
- **Per-backend / per-tool timeouts** (`default_timeout`, `tool_timeouts`) —
  override the 15 s default so slow image/video/LLM backends aren't cut off.
- **`tool-compass execute <tool> '<json-args>'`** — run a proxied tool call
  from the terminal to smoke-test a backend. `--timeout`, `--json`.
- **Per-backend tool allow/deny** (`allow_tools` / `deny_tools` glob lists) —
  expose a safe subset of a broad backend. Enforced at index time and at the
  `execute()` boundary (defense-in-depth).
- **Active liveness probe** on `compass_status(active=True)` — surfaces
  hung-but-connected backends a passive connection check would miss.
- **Analytics retention** (`analytics_retention_days`, default 30, `0` =
  forever) with an automatic prune on startup sync, so the analytics DB no
  longer grows without bound.
- **Auto-refresh polling** — `sync_polling_interval` is now honored at runtime
  (the gateway starts a background re-sync loop when it is `> 0`).
- **Schema-aware change detection** — a backend that revises a tool's
  description or parameters (same name) is now re-synced and re-embedded
  (previously only name changes were detected).

### Fixed
- **Multi-backend index data loss (HIGH).** Orphan-vector compaction rebuilt
  the index from only the changed-backend subset, silently deleting every
  other backend's tools after ~20 churn cycles. Compaction now rebuilds from
  the full connected catalog.
- **Analytics time-window skew (HIGH).** Timeframe queries compared naive
  local time against UTC-stored timestamps, shifting every windowed summary by
  the host's UTC offset. All analytics windows are now computed in UTC.
- **Docker UI unreachable.** The published image ran the Gradio UI bound to
  `127.0.0.1` (loopback) despite the port mapping — `docker-compose up` served
  nothing on `localhost:7860`. The image now binds `0.0.0.0`.
- `compass()` no longer raises an unhandled `IndexError` on an empty-match +
  zero-confidence-chain query; `compass_chains(create/detect)` returns the
  structured service-unavailable envelope when Ollama is down instead of a raw
  exception; `compass_chains(detect)` now actually indexes detected chains
  (was reporting success without indexing them).
- The embedding-cache write no longer commits inside `build_index`'s atomic
  rebuild transaction (a failed HNSW save could leave DB/HNSW divergence).
- CLI `search` now catches the real Ollama-down exception types and falls back
  to keyword search (matching the UI/gateway), so `--help`'s promise holds.
- `requirements.txt` now includes `rich` (the Docker image was missing it);
  `.env.example` / `llms.txt` corrected to the env vars the code actually reads.
- Several test-integrity fixes (coroutine leak, tautological/mis-named tests).

### Changed
- **Hybrid search is on by default** (`hybrid_search=True`). This changes the
  default result ranking (for the better); set `hybrid_search=false` to restore
  pure-semantic ordering. The response envelope, `min_confidence`, and `top_k`
  contracts are unchanged.
- **Backend names containing `:` are now rejected at config load** with a clear
  error — a colon is the reserved `backend:tool` separator and made routing and
  the allow/deny policy ambiguous. Rename any such backend (this only rejects
  configs that were already mis-routing).

## [2.4.0] - 2026-06-20

Dogfood swarm v3 — full health pass (bug/security → proactive → humanization
→ visual) + a user-scoped feature pass. Tests 991 → 1151; ruff errors 31 → 0.
No breaking changes; the default config path, the Ollama embedding default,
and all existing tool/CLI behavior are unchanged.

### Added
- **`tool-compass init`** — scaffolds `compass_config.json` at the resolved
  config path and prints a ready-to-paste Claude Desktop `mcpServers` snippet.
  `--force` to overwrite, `--json` for machine output.
- **Pluggable embedding backend** — `embedding_provider` config selects the
  embedder: `ollama` (default, unchanged) or `openai`/`openai-compatible`
  (`/v1/embeddings`; covers OpenAI, LM Studio, and any compatible server),
  with `embedding_base_url`, `embedding_api_key` (redacted;
  `TOOL_COMPASS_EMBEDDING_API_KEY` env override), and prefix overrides.
- MCP-client registration recipes (Claude Desktop, Cursor, Cline) in the handbook.
- `LOG_LEVEL`, `TOOL_COMPASS_ANALYTICS_DISABLED`, and
  `TOOL_COMPASS_HOT_CACHE_SIZE` env overrides are now honored (previously
  advertised in `.env.example` but ignored).

### Fixed
- **`tool-compass serve` / `serve --http`** crashed with "unrecognized
  arguments: serve" — the CLI now neutralizes `sys.argv` before delegating to
  the gateway.
- Oversize backend lines (>1 MiB) no longer wedge a connection — the read loop
  caught the wrong exception type (`LimitOverrunError` vs the `ValueError`
  `readline()` actually raises).
- `config doctor`/`show_config` no longer leak `${VAR}`-resolved secrets in
  backend `headers`/`env`/`args` or credentialed `ollama_url`/backend URLs.
- Cold-start (`get_index()` RuntimeError) now returns the structured
  service-unavailable envelope on every read tool instead of a raw stack.
- `tool-compass sync` reports the honest indexed count + warns on backends
  that failed to connect (was a fabricated "+0 ~0 -0" line); per-backend sync
  status is now durable (a failing backend stops reporting healthy).
- `gateway.py --sync` exits non-zero when it indexes nothing (CI-safe).
- Gradio UI: pinned the dark theme surface (the UI was invisible in light
  mode); Status tab now probes the configured Ollama; empty index shows a
  "run sync" card.
- Cross-event-loop `RuntimeError` in the embedder semaphore + gateway locks
  (per-loop keying); several unguarded `json.loads`/error paths hardened.

### Changed
- Lint gate cleaned (31 ruff errors → 0); 4 vacuous/skip-masking tests
  hardened so `/ready` + `/metrics` now have executing coverage.

## [2.3.0] - 2026-05-15

Dogfood swarm v2 release — comprehensive Stage A bug/security pass +
Stage B proactive hardening + Stage C humanization + Stage D visual/CLI
polish + Wave-10 feature pass. ~250 findings addressed across waves 1-11
of swarm-1778813065-e2dc. New: `@mcptoolshop/tool-compass` npm wrapper
(zero-prerequisite `npx` install), `release-binaries.yml` workflow
(PyInstaller binaries for linux-x64/darwin-arm64/win-x64 with SHA256
checksums), 6 new CLI subcommands for full MCP-surface parity, RFC 9457
error envelope with `retryable` discriminator and `nearest_tools[]`
suggestions, full OTel `gen_ai.*` metrics surface, Hystrix-style circuit
breaker observability, Rich-powered CLI with `--json` flag and color
discipline, golden-set Recall@k regression benchmark. Tests 455 → 498
(+43). No breaking changes; v2.2.x API callers continue to work
(legacy `error: <str>` shape preserved alongside new envelope).

### Migrating from v2.2.x
- Error envelope: tool responses now include a structured `error_envelope`
  object (RFC 9457 + Stripe-style 3-level taxonomy) ALONGSIDE the legacy
  `error: <str>` field. New callers should branch on `error_envelope.code`
  rather than parsing the prose string; legacy callers continue to work.
- `degraded: true` field: compass-family responses now include this flag
  when serving lexical-fallback results (Ollama unavailable). Callers can
  surface this to end users or downgrade confidence.
- `/metrics` additions: 5 new Prometheus series (circuit_breaker_transitions,
  fallback_invocations, hnsw_search_duration histogram, embedder_inflight,
  queue_wait_seconds). Existing series unchanged.
- CLI: 6 new subcommands added (`ui`, `status`, `categories`, `audit`,
  `analytics`, `chains`). `tool-compass serve --http <port>` now actually
  sets the port (was previously parsed but ignored).

Wave-11 CLI feature parity + test hardening on top of the Stage B+C
release-hygiene work. Resolves audit findings flagged in Wave-10. No
breaking changes — the new subcommands are additive, and the existing
default-to-`serve` behavior is preserved.

### Added (Wave-11 — CLI feature parity)
- **`tool-compass ui`** — launch the Gradio web UI inline. Thin wrapper
  that forwards `--port`, `--host`, `--share`, and `--auth user:pass`
  (the last bridges to the `GRADIO_AUTH` env var ui.py reads). Closes
  the gap where the README + handbook advertised `tool-compass ui` but
  only `tool-compass-ui` existed. Requires `pip install
  tool-compass[ui]` extras (FE-W11-002).
- **`tool-compass status`** — delegates to `compass_status`; renders
  index health, backend connection counts, health flags (ollama
  reachable, index available, degraded_mode), and last sync timestamp.
  Honors `--json` for script-friendly output (FE-W11-003).
- **`tool-compass categories`** — delegates to `compass_categories`;
  prints categories sorted by tool count desc, with `--json`
  (FE-W11-004).
- **`tool-compass audit`** — delegates to `compass_audit`. Accepts
  `--timeframe` (1h/24h/7d/30d) and `--include-tools`. Renders system
  version, category/server counts, backend health, hot cache size, and
  chain summary; `--json` emits the raw payload (FE-W11-005).
- **`tool-compass analytics`** — delegates to `compass_analytics`.
  Accepts `--timeframe` and `--no-failures`. Renders top tools and
  total calls; gracefully unwraps the `analytics_disabled` error
  envelope when analytics is off (FE-W11-006a).
- **`tool-compass chains`** — delegates to `compass_chains`. Accepts
  `--action {list,detect}`; renders chain name, use count, auto-detect
  tag, and the tool arrow chain (FE-W11-006b).

### Fixed (Wave-11)
- **`tool-compass serve --http` now actually sets the port.** Previously
  the flag was parsed but ignored — the gateway only read `PORT` from
  env. `--http <port>` now exports `PORT=<port>`; bare `--http` falls
  back to existing `PORT` env or 8080. Validates integer input; rejects
  garbage with exit code 2 and a usage hint (FE-W11-007).
- **Gateway startup banner reads `_version.__version__`** instead of
  the hardcoded `"v2.0..."` literal that drifted on every release
  (gateway.py:2406). Matches the source-of-truth pattern used by `tool-compass
  --version` and the audit / status JSON shapes (FE-W11-008).

### Changed (Wave-11 — tests)
- **GW-FT-001 un-skipped.** The per-backend stdout reader has been
  shipped (`backend_client_simple.py:432`); the test fixture now spins
  up `_read_task` explicitly and feeds responses through an
  `asyncio.Queue` so the head-of-line guarantee is exercised even on
  systems where the previous timeout-then-skip pattern masked a
  genuine routing regression (FE-W11-009).
- **MCC-FT-002 conditional skips removed.** `get_canonical_name` and
  `ToolDefinition.deprecated_aliases` both ship in v2.2.0; the
  conditional `pytest.skip` lines turned silent regressions into
  invisible no-ops. Both `test_deprecated_aliases_resolves_to_canonical`
  and `test_analytics_canonicalizes_deprecated_name` are now hard
  assertions on the canonical path (FE-W11-010).
- 11 new smoke tests covering the 6 new subcommands, `--http` PORT
  export (with and without explicit value), `--auth` -> GRADIO_AUTH
  bridge, argparse choice enforcement, and the banner version-read.

---

CI / supply-chain / release-hygiene hardening pass (Dogfood Stage B+C,
ci-tooling domain, wave-7 of swarm-1778813065-e2dc). No public API or
runtime behavior changes.

### Added
- **`@mcptoolshop/tool-compass` npm wrapper** for zero-prerequisite
  `npx` install. Downloads SHA256-verified platform binaries from the
  GitHub Release via `@mcptoolshop/npm-launcher`. Wrapper lives under
  `npm/` in the repo; tracks the source release version 1:1 via the
  `publish-npm` job in `release-binaries.yml`. Ships with the source
  README + 7 translations bundled at publish time.
- **`release-binaries.yml` workflow** — on release, PyInstaller builds
  `tool-compass-<version>-{linux-x64,darwin-arm64,win-x64}` single-file
  binaries with SHA256 checksums (`checksums-<version>.txt`), attaches
  to the GitHub Release, then publishes the npm wrapper with `--provenance`.
- SLSA build provenance attestations on PyPI publish via
  `attestations: true` on `pypa/gh-action-pypi-publish` (CT-B-009).
- SLSA `provenance: mode=max` plus SBOM attestation on the multi-arch
  GHCR image via `docker/build-push-action` (CT-B-009).
- `.github/dependabot.yml` docker ecosystem entry with the all-docker
  group (minor + patch); pairs with the digest-pinned base image
  (CT-B-005).
- Daily security-only Dependabot overlay for pip + github-actions +
  docker ecosystems with PR cap 10 (CT-B-006).
- `actions/configure-pages` step in `pages-build` so Pages config is no
  longer an implicit dependency on the Settings → Pages UI (CT-B-016).
- `.pre-commit-config.yaml` with `ruff-format`, `ruff --fix`, the
  standard pre-commit-hooks set, and gitleaks for a secrets scan.
  CONTRIBUTING.md quick-start now includes `pre-commit install`
  (CT-B-018).
- `scripts/regenerate-scorecard.sh` — refreshes the auto-generated
  block in SCORECARD.md between SHIPCHECK markers while preserving
  hand-curated sections (Known Gaps, Remediation History). New
  `make scorecard` + `make verify-scorecard` wrappers (CT-B-017).
- `scripts/verify-metrics.sh` + `make verify-metrics` — boots the
  gateway, scrapes `/metrics`, asserts the Four Golden Signals surface
  is present. Includes warn-only checks for the saturation gauges
  expected from BE-B-002 (CT-B-008).
- `maintainers` field in `pyproject.toml [project]` (CT-B-019).

### Changed
- All remaining floating GitHub Actions tags pinned by full commit SHA:
  `actions/github-script@v7` → `60a0d83...` (CT-B-001) and
  `docker/setup-qemu-action@v3` → `c7c5346...` (CT-B-002). Every action
  in `.github/workflows/` is now SHA-pinned.
- `Dockerfile` base image pinned by digest: `python:3.11-slim@sha256:9a7765b367...`
  on both the builder and production stages. Dependabot's docker
  ecosystem (CT-B-005) keeps the digest fresh (CT-B-003).
- Production Dockerfile stage now copies only the builder-assembled
  `/build` tree instead of the full build context. Tests, docs, site,
  `.github`, translation READMEs, and audit docs no longer ship inside
  the production image (CT-B-004).
- `.dockerignore` extended with the same exclude list for
  defense-in-depth and to keep the build context small.
- `LABEL version="..."` removed from the Dockerfile; the OCI
  `opencontainers.image.version` label is emitted at publish time by
  `docker/metadata-action` from the git tag (CT-B-015).
- `Python :: 3.13` classifier removed from `pyproject.toml` until
  hnswlib ships a working `cp313` wheel (`requires-python` still
  permits install on 3.13; this only affects PyPI search filters)
  (CT-B-019).
- Every CI job in `ci.yml` and `publish.yml` now declares
  `timeout-minutes` (lint 5, test 15, integration 25, docker 20,
  nightly-fuzz 45, pages-build 10, pages-deploy 5, build 10,
  publish-pypi 10, docker 30, release-smoke 15) — replaces the 360-min
  GitHub default (CT-B-011).
- All `actions/upload-artifact` invocations now set `retention-days`
  (14 for JUnit + pip-audit diagnostic reports in `ci.yml`; 7 for the
  `dist/` build-to-publish handoff in `publish.yml`) (CT-B-012).
- `release-smoke` PyPI-install retry loop now loud-fails when no
  iteration succeeds, replacing the redundant follow-up install that
  silently masked propagation failures (CT-B-014).
- `SECURITY.md` consolidates the preferred reporting path (GitHub
  Security Advisories) and tightens the Critical resolution SLA to 72h
  acknowledged plan / 7d patch (CT-B-020).
- `SCORECARD.md` Date + Version refreshed; auto-generated block now
  lives between `SHIPCHECK-AUTO-START/END` markers so regenerations
  preserve Known Gaps + Remediation History.

### Notes
- Translations (`README.{ja,zh,es,fr,hi,it,pt-BR}.md`) re-run on
  TranslateGemma 12B as Phase 10 of release prep BEFORE
  `npm publish` / `gh release create` per the global ordering rule
  (CT-B-013). This wave intentionally left them untouched.

## [2.2.2] - 2026-04-23

Patch release. Fixes the Docker image so the `tool-compass` console script
is actually installed inside the container (was missing in v2.2.0/v2.2.1).

### Fixed
- Docker image now runs `pip install --no-deps .` during build, so
  `docker run ghcr.io/mcp-tool-shop-org/tool-compass:2.2.2 tool-compass --version`
  (or any subcommand) works out of the box. v2.2.0/v2.2.1 images only
  shipped the source tree; the console script was never registered on
  PATH.
- Dockerfile `LABEL version` bumped 2.0.7 → 2.2.2.

## [2.2.1] - 2026-04-23

Patch release. Fixes the v2.2.0 release-smoke defect.

### Fixed
- `tool-compass --version` now prints the version and exits. v2.2.0's new
  CLI subcommand shell (MCC-FT-001) forgot to wire `--version` on the root
  parser, so the publish-time release-smoke check failed even though the
  artifacts themselves were valid. Purely a CLI ergonomics fix.

## [2.2.0] - 2026-04-23

Dogfood swarm release: Stage A bug/security health pass + Stage B/C humanization
+ Feature Pass. Shipcheck 17/17 retained.

### Added
- **Per-backend stdout reader** — isolated log streams per MCP backend so a noisy
  child process can't crowd out siblings in the combined view
- **`/ready` + `/metrics` HTTP endpoints** — readiness probe + Prometheus-style
  metrics surface for operators running the gateway behind a load balancer
- **Embedding cache** — LRU cache in `Embedder` so repeated identical queries
  skip the Ollama round-trip
- **Diffing sync** — `SyncManager` now emits a structured diff (added / removed /
  changed) instead of a full rebuild signal; downstream consumers can act on
  partial changes
- **`tool-compass` CLI subcommand shell** (`cli.py`) — top-level entry point now
  dispatches to `serve` (gateway), `ui` (Gradio), `doctor` (config health),
  `sync`, `test`, and `config`. `python gateway.py` still works unchanged.
- **`deprecated_aliases` in tool manifest** — lets backends rename tools without
  breaking old callers; the compass surfaces both names and marks legacy ones
- **`make scorecard` + `make verify-scorecard`** (CDS-FT-001) — regenerate
  SCORECARD.md from shipcheck output; CI soft-fails on drift
- **`make dev` + `make dev-ui`** (CDS-FT-002) — one-shot install + run targets
  for the fast local loop, with a warning if Ollama isn't reachable
- **Nightly Hypothesis fuzz job** (TST-FT-002) — `nightly-fuzz` runs Mon/Wed/Fri
  at 09:00 UTC in `ci.yml` (no new workflow file), auto-opens a tracking issue
  on failure
- **Coverage floor** (TST-FT-001) — `[tool.coverage.report] fail_under = 60`
  and `--cov-fail-under=60` in `make test`; conservative first gate
- **Architecture Mermaid diagrams** (CDS-FT-005) — component graph + request
  sequence diagrams in `site/src/content/docs/handbook/architecture.md`

### Changed
- `tool-compass` console script now targets `cli:main` (was `gateway:main`).
  `python gateway.py` remains fully supported for direct invocation.
- Wheel bundle includes `cli.py`

### Fixed (Stage A — 23 HIGH bug/security)
- Misc security, bug, and error-shape fixes surfaced by the health pass

### Humanized (Stage B/C — 15 HIGH + 14 MED)
- **Ollama-down lexical fallback** — compass returns keyword matches when the
  embedding backend is unreachable, with a clear degraded-mode marker
- **`trace_id` correlation** — every request carries a trace ID end-to-end for
  log stitching
- **Circuit breaker** on the embedder — opens after N consecutive failures,
  half-opens after a cooldown
- **Corrupt-config recovery** — malformed `compass_config.json` no longer
  crashes startup; loader falls back to defaults with a warning
- **`tool-compass doctor`** — diagnoses Ollama reachability, index presence,
  backend health, and prints actionable fixes

## [2.0.7] - 2026-03-25

### Added
- 5 version consistency tests (semver, >= 1.0.0, CHANGELOG, pyproject parse, CLI)

### Security
- SHA-pinned all GitHub Actions across 3 workflows (ci, pages, publish)

## [2.0.6] - 2026-02-27

### Added
- SHIP_GATE.md and SCORECARD.md (Shipcheck compliance)
- Makefile with `verify` target (lint + test + build)
- Security & Data Scope section and scorecard in README
- Standard email in SECURITY.md

### Changed
- Removed redundant h1 heading (logo already contains name)
- Replaced footer with standard MCP Tool Shop link
- Scorecard 46/50 → 50/50
- Bumped to 2.0.6

## [2.0.3] - 2026-02-14

### Fixed
- `categorize_tool()` now falls back to description matching when name has no category keywords
- `analytics.last_success_at` stores real timestamps instead of the literal string "CURRENT_TIMESTAMP"
- `SyncEmbedder` no longer crashes when called from inside a running event loop (Gradio, FastMCP)
- UI `run_async` helper replaced with deterministic loop detection (no more `RuntimeError` roulette)
- Removed discarded `sequence_hash` read in chain detection

### Changed
- Private `_backends` dict access replaced with public `is_backend_connected()` API
- `backend_client.py` renamed to `backend_client_mcp.py` (experimental; not used at runtime)
- Version reporting unified via `_version.py` module (reads from `importlib.metadata` or `pyproject.toml`)
- UI singletons protected with `threading.Lock` (Gradio is multi-threaded)
- Runtime assumptions documented in `gateway.py`

### Infrastructure
- Python 3.13 added to CI test matrix (3.10–3.13, 12 matrix jobs)
- `actions/checkout` normalized to v6 across all workflows
- `pip-audit` dependency vulnerability scan added (warn-only)
- `scripts/**` added to CI path triggers

### Tests
- 23 new tests: `backend_client_simple.py` API, `run_async` loop safety, `SyncEmbedder` loop safety, `_version.py` (387 → 410 tests)

## [2.0.2] - 2026-02-14

### Fixed
- Windows subprocess stability with `SimpleBackendManager` (#11)
- Auto-fix lint issues with ruff (#13)

### Changed
- Migrated all repository URLs to `mcp-tool-shop-org` GitHub organization
- Updated CI actions to latest major versions (checkout v6, setup-python v6, codecov v5)

### Infrastructure
- Added GHCR Docker publish workflow
- Added `llms.txt` for LLM discoverability
- Added social preview assets
- Updated dependency version ranges (black, pytest, pytest-cov, isort, gradio)

## [2.0.1] - 2026-01-18

### Added
- `pyproject.toml` for modern Python packaging (PEP 517/518)
- PyPI publishing workflow (GitHub Actions)
- Optional dependencies: `[ui]`, `[dev]`, `[all]`

### Changed
- Fixed CI workflow paths for standalone repository
- Removed hardcoded paths from documentation

### Infrastructure
- Published to PyPI as `tool-compass`
- Added `PYPI_API_TOKEN` secret for automated releases

## [2.0.0] - 2026-01-17

### Added
- **Gradio Web UI** (`ui.py`) - Interactive browser for tool discovery
  - Semantic search with confidence scores and text labels
  - Tool browser with server/category filtering
  - Workflow search and visualization
  - Analytics dashboard with usage metrics
  - System status with health checks
- **User-friendly error handling** - Graceful degradation when services unavailable
  - Ollama connection errors show helpful recovery steps
  - Missing index errors provide rebuild instructions
  - All errors include collapsible technical details
- **Input sanitization** - Query validation and length limits
- **Empty states** - Helpful guidance when no results/data
- **Text truncation** - Long names/descriptions truncate gracefully with tooltips

### Changed
- Confidence scores now show text labels ("Excellent/Good/Fair/Low") alongside percentages
- Improved responsive layout with flex-wrap for mobile
- System status tab now shows real-time Ollama health check

### Fixed
- Fixed `get_backend_tools()` method that was missing from `CompassIndex`
- Fixed potential SQL injection in tool search (was already safe, added explicit parameterization)

## [1.1.0] - 2026-01-16

### Added
- **Chain Indexer** (`chain_indexer.py`) - Workflow detection from usage patterns
  - Auto-detects common tool sequences
  - HNSW index for semantic workflow search
  - Manual workflow definition support
- **Analytics System** (`analytics.py`) - Usage tracking and hot cache
  - Search query tracking
  - Tool call success/failure rates
  - Latency monitoring
  - Hot cache for frequently used tools
- **Sync Manager** (`sync_manager.py`) - Backend synchronization
  - Multi-backend tool discovery
  - Incremental index updates
  - Connection pooling

### Changed
- Gateway now supports progressive disclosure (core tools first)
- Improved embedding generation with batching

## [1.0.0] - 2026-01-15

### Added
- **Core Gateway** (`gateway.py`) - MCP server with 9 tools
  - `compass(intent)` - Semantic tool search
  - `describe(tool_name)` - Get tool schema
  - `execute(tool_name, args)` - Run tools
  - `compass_categories()` - List categories
  - `compass_analytics()` - Usage stats
  - `compass_chains()` - Workflow management
  - `compass_sync()` - Rebuild index
  - `compass_audit()` - System report
- **HNSW Indexer** (`indexer.py`) - Vector search for tools
  - O(log n) approximate nearest neighbor search
  - SQLite metadata storage
  - Dynamic tool addition/removal
- **Ollama Embedder** (`embedder.py`) - nomic-embed-text integration
  - 768-dimensional embeddings
  - Async batch processing
  - Health checks and auto-recovery
- **Backend Client** (`backend_client.py`) - MCP backend proxy
  - stdio, HTTP, and import modes
  - Connection pooling
  - Timeout handling
- **Configuration** (`config.py`) - Environment-driven settings
  - YAML/JSON config files
  - Environment variable overrides
  - Sensible defaults
- **Tool Manifest** (`tool_manifest.py`) - Tool definitions
  - 44 tools across 5 backends
  - Category and server metadata
  - Example usage strings

### Infrastructure
- Dockerfile with multi-stage build
- docker-compose.yml for development
- GitHub Actions CI/CD pipeline
- pytest test suite with async support
- MIT License

---

[Unreleased]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.2.2...HEAD
[2.2.2]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.7...v2.2.0
[2.0.7]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.6...v2.0.7
[2.0.6]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.3...v2.0.6
[2.0.3]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/mcp-tool-shop-org/tool-compass/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/mcp-tool-shop-org/tool-compass/releases/tag/v1.0.0
