# Tool Compass - Semantic MCP Tool Discovery
# Multi-stage build for minimal production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
# Base image is digest-pinned (CT-B-003) so re-published 3.11-slim tags cannot
# silently change what we ship; Dependabot's docker ecosystem (CT-B-005) bumps
# the digest monthly via .github/dependabot.yml.
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
COPY pyproject.toml README.md LICENSE ./
COPY cli.py gateway.py ui.py indexer.py embedder.py chain_indexer.py \
     analytics.py config.py sync_manager.py tool_manifest.py \
     bootstrap.py backend_client_simple.py backend_client_mcp.py \
     _version.py llms.txt compass_config.example.json ./

# Create virtualenv, install deps, and install the package itself so
# the `tool-compass` console script ends up on PATH (ships with the image).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn && \
    pip install --no-cache-dir --no-deps .

# =============================================================================
# Stage 2: Production
# =============================================================================
# Digest-pinned base (CT-B-003). Dependabot keeps both FROMs in lockstep.
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834 AS production

LABEL maintainer="Tool Compass <github.com/mcp-tool-shop-org/tool-compass>"
LABEL description="Semantic search gateway for MCP tools"
# Version label is set automatically at publish-time by docker/metadata-action
# (publish.yml) via the OCI annotation opencontainers.image.version, computed
# from the git release tag. The hand-maintained "LABEL version" was dropped
# per CT-B-015 to remove the manual sync hazard.

# Security: Run as non-root user
RUN groupadd -r compass && useradd -r -g compass compass

WORKDIR /app/tool_compass

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the runtime artifacts the builder explicitly assembled (CT-B-004).
# This excludes tests/, docs/, site/, .github/, .hypothesis/, archive/,
# translation READMEs, SCORECARD/SHIP_GATE drafts, etc. from the production
# image — both attack-surface reduction and image-size win. Mirrors the
# builder's explicit module list at line 19-22.
COPY --chown=compass:compass --from=builder /build /app/tool_compass

# Create data directory for indexes
RUN mkdir -p /app/tool_compass/db && \
    chown -R compass:compass /app/tool_compass/db

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Tool Compass settings
    TOOL_COMPASS_BASE_PATH=/app \
    OLLAMA_URL=http://host.docker.internal:11434 \
    # Gradio settings
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Expose Gradio UI port
EXPOSE 7860

# Switch to non-root user
USER compass

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from indexer import CompassIndex; idx = CompassIndex(); print('healthy' if idx.load_index() else 'no index')" || exit 1

# Default command: Run Gradio UI
CMD ["python", "ui.py"]

# =============================================================================
# Stage 3: MCP Gateway (alternative entrypoint)
# =============================================================================
FROM production AS mcp-gateway

# Override for HTTP mode (Fly.io / Smithery)
# ci-infra-01: gateway.py's _run_http defaults HOST to 127.0.0.1 (loopback), so
# without this the published image binds loopback and any orchestrator's proxy /
# health check (the HEALTHCHECK below hits localhost, but Fly/Smithery reach the
# container over the network) can't reach the gateway. 0.0.0.0 is the documented
# reverse-proxy path — this image is meant to sit behind an authenticated edge.
ENV PORT=8080 \
    HOST=0.0.0.0

# Expose MCP HTTP port
EXPOSE 8080

# Health check via /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); r.raise_for_status()" || exit 1

CMD ["python", "gateway.py"]
