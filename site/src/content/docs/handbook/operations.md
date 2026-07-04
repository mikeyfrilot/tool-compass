---
title: Operations
description: Observability, degraded modes, and deployment posture for Tool Compass in HTTP mode.
sidebar:
  order: 5
---

Tool Compass runs in two transports: **stdio** (the default — for Claude
Desktop, Cursor, Continue.dev, any MCP client that launches it as a
subprocess) and **streamable-http** (set `PORT=N` — for Fly.io,
Kubernetes, or any remote deployment).

HTTP mode adds three HTTP routes alongside the JSON-RPC transport. This
page covers how to use them.

## `/health` — liveness

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

Always returns 200 while the process is up. Wire this to your load
balancer's liveness probe — it answers "is the process alive", nothing
more.

## `/ready` — deep readiness

```bash
curl http://localhost:8080/ready
```

Returns 200 only when ALL of the following are true:

- `CompassIndex` is loaded with a working SQLite connection.
- Ollama is reachable (or the circuit breaker is closed).
- At least one configured backend is connected.

If any check fails, returns 503 with a JSON breakdown:

```json
{
  "status": "not_ready",
  "checks": {
    "index": {"ok": true},
    "ollama": {"ok": false, "reason": "circuit_breaker_open"},
    "backends": {"ok": true, "connected": 3, "configured": 3}
  }
}
```

Wire this to your **readiness** probe (not liveness) so a pod that can't
serve queries gets taken out of rotation but not restarted. Result is
cached for 30 seconds to avoid DoS-ing Ollama from load-balancer polling.

## `/metrics` — Prometheus

```bash
curl http://localhost:8080/metrics
```

Returns `text/plain; version=0.0.4` Prometheus format. Sampled metrics:

| Metric | Type | Notes |
|---|---|---|
| `tool_compass_search_total` | counter | Total `compass()` calls (from analytics, 24h window) |
| `tool_compass_ollama_available` | gauge | 1 if embedder circuit breaker is closed, else 0 |
| `tool_compass_backend_up{name}` | gauge | 1 per connected backend |
| `tool_compass_backend_call_total{name,status}` | counter | Per-backend success/error counts |
| `tool_compass_embed_latency_p95_ms` | gauge | p95 of last 1000 embed calls |
| `tool_compass_embed_failures_total` | counter | Total embed failures |
| `tool_compass_index_age_seconds` | gauge | Seconds since last `compass_sync` |
| `tool_compass_orphaned_vectors` | gauge | HNSW vectors without a matching SQLite row |

No `prometheus_client` dependency — the text is built by the gateway
itself.

## Trace IDs

Every `@mcp.tool()` handler generates an 8-char `trace_id` (`uuid4[:8]`)
on entry. It's threaded through:

- Every `logger.info/warning/error` call in that request.
- The `trace_id` field in the success response envelope.
- The `trace_id` field in the error response envelope.
- Analytics rows where the underlying function signature supports it.

When a user reports "my `compass()` call failed", ask them for the
`trace_id` and `grep` your logs. Example error envelope:

```json
{
  "error": "Backend unreachable: kubernetes",
  "trace_id": "9f3a1c7b",
  "hint": "Run tool-compass doctor to check backend config."
}
```

## Degraded modes

Tool Compass prefers **degraded** over **down** whenever possible.

### Ollama-offline

If `embed_query()` fails (network, timeout, breaker open), `compass()`
falls back to SQLite `LIKE` lexical search over tool names + descriptions.
Results are marked `degraded: true` and the envelope includes a
`warnings[]` array:

```json
{
  "matches": [
    {"tool": "bridge:read_file", "degraded": true, "confidence": 0.0}
  ],
  "warnings": [
    "Semantic search unavailable: Ollama at http://localhost:11434 is unreachable. Try: ollama serve. Showing keyword-based results."
  ],
  "trace_id": "a1b2c3d4"
}
```

The embedder circuit breaker opens after 3 consecutive failures and
stays open for 30 seconds — subsequent calls fast-fail into the fallback
instead of each eating a 30-second Ollama timeout.

### Analytics-degraded

If the SQLite analytics DB errors (locked, disk full, corrupted schema),
analytics writes log a warning **once** and set a `_degraded` flag —
the primary tool-discovery path keeps working. Surface the state:

```python
from analytics import get_analytics
analytics = get_analytics()
health = analytics.get_health()  # {"degraded": bool, "reason": str | None}
```

### Index-corrupt / dim-mismatch

If the HNSW file on disk was built with a different `embedding_dim`
(e.g. you switched models), `load_index()` raises a `RuntimeError` that
names the path and advises deletion:

```
Index file at db/compass.hnsw uses 1024-dim vectors but code expects 768.
The embedding model likely changed. Delete db/compass.hnsw and run
`tool-compass sync` to rebuild.
```

## Docker + GHCR

The published image supports multi-arch:

```bash
docker pull ghcr.io/mcp-tool-shop-org/tool-compass:latest
# linux/amd64 + linux/arm64 from the same tag
# (pin a specific release instead of :latest for reproducible deploys, e.g. :v2.5.0)
```

Same tag runs on x86_64 servers and Apple Silicon / ARM workstations
without emulation.

## Fly.io

See the `fly.toml` at the repo root. Typical deployment:

```bash
fly launch --image ghcr.io/mcp-tool-shop-org/tool-compass:v2.2.0
fly secrets set OLLAMA_URL=https://your-ollama-host
fly deploy
```

Wire `/ready` to Fly's health check so pods that can't actually serve
queries get pulled out of the pool without flapping.
