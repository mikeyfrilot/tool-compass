import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'Tool Compass',
  description: 'Semantic MCP tool discovery gateway — find tools by intent, not memory.',
  logoBadge: 'TC',
  brandName: 'Tool Compass',
  repoUrl: 'https://github.com/mcp-tool-shop-org/tool-compass',
  footerText: 'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>',

  hero: {
    badge: 'MCP Gateway — v2.4',
    headline: 'Find tools by',
    headlineAccent: 'intent, not memory.',
    description: 'Semantic search for MCP tools. 95% fewer tokens, ~15ms latency, graceful Ollama-offline fallback, trace IDs you can paste into a bug report.',
    primaryCta: { href: '#quick-start', label: 'Quick Start' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Install',
        code: 'ollama pull nomic-embed-text\npip install tool-compass\ntool-compass sync',
      },
      {
        label: 'CLI',
        code: 'tool-compass search "generate an AI image"\ntool-compass describe comfy:comfy_generate\ntool-compass doctor',
      },
      {
        label: 'MCP tool',
        code: 'compass(\n  intent="generate an AI image",\n  top_k=3\n)',
      },
      {
        label: 'Result',
        code: '# comfy:comfy_generate  0.91\n# tokens_saved: 20,500\n# trace_id: 9f3a1c7b',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'Why Tool Compass',
      subtitle: 'Stop loading all 77 tools into context. Find the right one — and keep finding it when things go sideways.',
      features: [
        {
          title: 'Semantic Discovery',
          desc: 'compass(intent) finds tools by meaning. Describe what you want to do — get the right tool, not a list of all of them.',
        },
        {
          title: '95% Token Savings',
          desc: '38,500 tokens → 2,000 per request. HNSW vector index, nomic-embed-text, ~15ms query latency, ~95% accuracy@3.',
        },
        {
          title: 'Progressive Disclosure',
          desc: 'compass() → describe() → execute(). Load only the schema you need, only when you actually need it.',
        },
        {
          title: 'Graceful degradation',
          desc: 'Ollama down? Compass falls back to keyword search over the same index, marks results degraded, and tells you exactly what to start.',
        },
        {
          title: 'Trace IDs in every response',
          desc: 'Every MCP call gets an 8-char trace_id plumbed into logs, success envelopes, and error envelopes. Paste it into a bug report and grep the logs.',
        },
        {
          title: 'Production observability',
          desc: '/ready does a deep probe (index + Ollama + backends). /metrics emits Prometheus text with embed p95, orphan vectors, per-backend up/down.',
        },
      ],
    },
    {
      kind: 'features',
      id: 'whats-new',
      title: "What's new in v2.5",
      subtitle: 'Remote HTTP backends, hybrid search, full-schema describe(), per-tool timeouts, allow/deny, and an execute CLI — a 12-feature pass, each closed by an adversarial cross-wave re-audit.',
      features: [
        {
          title: 'HTTP / remote backends',
          desc: 'Front remote and SaaS MCP servers over streamable-http, not just local stdio subprocesses. Optional bearer-token auth on your own gateway endpoint.',
        },
        {
          title: 'Hybrid search + exact-name boost',
          desc: 'Semantic (HNSW) and lexical signals fused with Reciprocal Rank Fusion on the healthy path — describe what you want, or paste a tool name and it ranks #1.',
        },
        {
          title: 'Full-schema describe()',
          desc: 'The complete inputSchema (required fields, descriptions, enums, defaults) is preserved through sync, so an agent can build a correct execute() call in one hop.',
        },
        {
          title: 'Per-tool timeouts & allow/deny',
          desc: 'Override the 15s default per backend or per tool for slow image/video backends; expose a safe subset of a broad backend with allow/deny globs.',
        },
        {
          title: 'execute CLI',
          desc: 'tool-compass execute <tool> \'<json>\' runs a proxied tool call from the terminal — smoke-test a backend without wiring up an agent.',
        },
        {
          title: 'Ops hardening',
          desc: 'Active liveness probe surfaces hung-but-connected backends; analytics retention keeps the local DB bounded; schema-aware change detection re-syncs revised tools.',
        },
      ],
    },
    {
      kind: 'code-cards',
      id: 'quick-start',
      title: 'Quick Start',
      cards: [
        {
          title: 'Install & Index',
          code: '# Prerequisites: Ollama running locally\nollama pull nomic-embed-text\n\n# Install the CLI + gateway\npip install tool-compass\n\n# Build the search index from your configured backends\ntool-compass sync\n\n# Start the MCP gateway (or `tool-compass serve`)\ntool-compass',
        },
        {
          title: 'One-shot CLI',
          code: '# Search by intent without starting a server\ntool-compass search "generate an AI image" --top 3 --json\n\n# Inspect a tool\'s schema\ntool-compass describe comfy:comfy_generate\n\n# Diagnostics dump — version, config, Ollama reachability\ntool-compass doctor',
        },
        {
          title: 'MCP tool call',
          code: 'compass(\n  intent="I need to generate an AI image from text",\n  top_k=3,\n  min_confidence=0.3\n)\n\n# Returns (excerpt):\n# {\n#   "matches": [{\n#     "tool": "comfy:comfy_generate",\n#     "confidence": 0.912\n#   }],\n#   "tokens_saved": 20500,\n#   "trace_id": "9f3a1c7b"\n# }',
        },
        {
          title: 'Graceful Ollama-down',
          code: '# Ollama stopped mid-session? compass still answers:\ncompass(intent="read a file")\n\n# Returns:\n# {\n#   "matches": [{\n#     "tool": "bridge:read_file",\n#     "degraded": true\n#   }],\n#   "warnings": [\n#     "Semantic search unavailable: Ollama at\n#      http://localhost:11434 is unreachable.\n#      Try: ollama serve"\n#   ]\n# }',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'tools',
      title: 'Gateway Tools',
      subtitle: 'Nine MCP tools — one semantic entry point for your entire MCP ecosystem.',
      columns: ['Tool', 'Description'],
      rows: [
        ['compass(intent)', 'Hybrid semantic + lexical search with exact-name boost — describe your intent, or paste a tool name'],
        ['describe(tool)', 'Get the full inputSchema (required fields, descriptions, enums, defaults) before calling'],
        ['execute(tool, args)', 'Run any indexed tool directly through the gateway'],
        ['compass_categories()', 'List all tool categories and connected MCP servers'],
        ['compass_status(active)', 'System health; active=True runs a live backend liveness probe'],
        ['compass_analytics(timeframe)', 'Usage statistics, accuracy metrics, and performance data'],
        ['compass_chains(action)', 'Discover and manage common multi-tool workflows'],
        ['compass_sync(force)', 'Rebuild the HNSW search index from connected backends'],
        ['compass_audit()', 'Full system diagnostic — index integrity, server health'],
      ],
    },
    {
      kind: 'data-table',
      id: 'operations',
      title: 'HTTP Endpoints',
      subtitle: 'Start the gateway in HTTP mode with PORT=8000 tool-compass for operator-grade endpoints.',
      columns: ['Endpoint', 'Purpose'],
      rows: [
        ['/health', 'Liveness probe — always 200 if the process is up.'],
        ['/ready', 'Deep readiness probe; 503 with a JSON breakdown when a check fails.'],
        ['/metrics', 'Prometheus text — search, backend, and embed latency metrics.'],
        ['MCP JSON-RPC', 'Standard MCP streamable-http transport at the root. HOST defaults to 127.0.0.1; expose via a reverse proxy or set gateway_auth_token (TOOL_COMPASS_GATEWAY_AUTH_TOKEN) for built-in bearer-token auth.'],
      ],
    },
  ],
};
