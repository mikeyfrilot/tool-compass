"""
Tool Compass - Configuration Schema
Defines how backends are configured and connected.

Environment Variables:
    TOOL_COMPASS_BASE_PATH: Base path for the project (default: parent of tool_compass)
    TOOL_COMPASS_PYTHON: Path to Python executable (default: auto-detect from venv)
    TOOL_COMPASS_CONFIG: Path to config file (default: <user_config_dir>/compass_config.json)
    TOOL_COMPASS_DATA_DIR: Override user data directory (default: platform-specific)
    TOOL_COMPASS_GATEWAY_AUTH_TOKEN: Bearer token for the HTTP transport (OPS-1).
        Overrides the config gateway_auth_token field; secret (never printed).
    OLLAMA_URL: Ollama server URL (default: http://localhost:11434)

Default config directories by platform:
    Windows: %LOCALAPPDATA%/tool-compass/
    macOS: ~/Library/Application Support/tool-compass/
    Linux: ~/.config/tool-compass/ (or $XDG_CONFIG_HOME/tool-compass/)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pathlib import Path
import json
import logging
import os
import platform as _platform
import shutil
import sys
import re
import time
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


@dataclass
class StdioBackend:
    """Backend that spawns an MCP server as subprocess."""

    type: Literal["stdio"] = "stdio"
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    # INT-02: per-backend tool-call outer deadline in seconds. None = no
    # per-backend deadline. Clamped to [1.0, 600.0] in validate_and_clamp.
    default_timeout: Optional[float] = None
    # INT-02: per-tool timeout override, keyed by BARE tool name (not the
    # server:tool qualified name). Each value clamped to [1.0, 600.0].
    tool_timeouts: Dict[str, float] = field(default_factory=dict)
    # FEAT-06: glob allowlist; empty list = allow all tools from this backend.
    allow_tools: List[str] = field(default_factory=list)
    # FEAT-06: glob denylist; deny takes precedence over allow.
    deny_tools: List[str] = field(default_factory=list)


@dataclass
class HttpBackend:
    """Backend that connects to an MCP server over HTTP/SSE."""

    type: Literal["http"] = "http"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    # INT-02: per-backend tool-call outer deadline in seconds (distinct from
    # `timeout`, the transport/connect timeout). None = no per-backend
    # deadline. Clamped to [1.0, 600.0] in validate_and_clamp.
    default_timeout: Optional[float] = None
    # INT-02: per-tool timeout override, keyed by BARE tool name. Clamped to
    # [1.0, 600.0].
    tool_timeouts: Dict[str, float] = field(default_factory=dict)
    # FEAT-06: glob allowlist; empty list = allow all tools from this backend.
    allow_tools: List[str] = field(default_factory=list)
    # FEAT-06: glob denylist; deny takes precedence over allow.
    deny_tools: List[str] = field(default_factory=list)


@dataclass
class ImportBackend:
    """Backend that imports an MCP server module directly (same process)."""

    type: Literal["import"] = "import"
    module: str = ""
    server_var: str = "mcp"  # Variable name of the FastMCP instance
    # FEAT-06: glob allowlist; empty list = allow all tools from this backend.
    allow_tools: List[str] = field(default_factory=list)
    # FEAT-06: glob denylist; deny takes precedence over allow.
    deny_tools: List[str] = field(default_factory=list)


BackendConfig = StdioBackend | HttpBackend | ImportBackend


@dataclass
class CompassConfig:
    """Full Tool Compass configuration."""

    # Backend server connections
    backends: Dict[str, BackendConfig] = field(default_factory=dict)

    # Embedding settings
    embedding_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"

    # Pluggable embedding backend (BE-FT-PE-001). Defaults preserve the
    # current Ollama behavior exactly.
    #   embedding_provider: "ollama" (default) | "openai" | "openai-compatible".
    #     Unknown values warn + fall back to "ollama" (validate_and_clamp).
    #   embedding_base_url: override the embed endpoint base. None -> use
    #     ollama_url for the ollama provider; REQUIRED for openai-compatible.
    #   embedding_api_key: secret for OpenAI-compatible auth (Bearer token).
    #     Redacted in doctor() dumps; an env override
    #     (TOOL_COMPASS_EMBEDDING_API_KEY) wins over the file value.
    #   embedding_query_prefix / embedding_document_prefix: override the
    #     retrieval-prefix convention. None -> provider default (nomic
    #     search_* for ollama, empty for openai-compatible).
    embedding_provider: str = "ollama"
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_query_prefix: Optional[str] = None
    embedding_document_prefix: Optional[str] = None

    # Index settings
    index_dir: str = "./db"
    auto_sync: bool = True  # Auto-discover tools from backends on startup

    # Search settings
    default_top_k: int = 5
    min_confidence: float = 0.3

    # Hybrid search (DISC-01): blend semantic + keyword retrieval. Default on.
    hybrid_search: bool = True

    # Exact-name boost (DISC-02): when a query exactly matches a tool name,
    # promote it and assign exact_match_confidence to the hit.
    exact_name_boost: bool = True
    exact_match_confidence: float = 1.0

    # Analytics retention (FEAT-04): prune analytics rows older than N days.
    # 0 = keep forever. Clamped to >= 0 in validate_and_clamp.
    analytics_retention_days: int = 30

    # Gateway auth (OPS-1): opt-in bearer token for the HTTP transport.
    # None = no auth. Treated as a SECRET: ${VAR}-resolved at load like other
    # secret-bearing fields, falls back to the TOOL_COMPASS_GATEWAY_AUTH_TOKEN
    # env var when unset (apply_env_overrides), and is redacted in
    # doctor()/show_config output (name ends in _token -> _SECRET_FIELD_HINTS).
    gateway_auth_token: Optional[str] = None

    # Progressive disclosure (reduces tokens further)
    progressive_disclosure: bool = True

    # Sync settings
    sync_check_on_startup: bool = True
    sync_polling_interval: int = 300  # seconds, 0 = disabled

    # Analytics settings
    analytics_enabled: bool = True
    hot_cache_size: int = 10

    # Chain detection settings
    chain_indexing_enabled: bool = True
    chain_detection_min_occurrences: int = 3
    top_chains_cache_size: int = 5

    # Circuit breaker / retry tuning (BE-B-014). Promoted from module-level
    # constants in embedder.py so operators can tune without code edits.
    # Ranges clamped in validate_and_clamp() to keep the behavior sane.
    ollama_breaker_failure_threshold: int = 3
    ollama_breaker_open_seconds: float = 30.0
    ollama_retry_attempts: int = 3
    ollama_retry_backoffs: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0]
    )

    # HNSW parameters (BE-B-008). Promoted from module-level constants in
    # indexer.py so operators can re-tune at scale without forking.
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50

    @classmethod
    def from_file(cls, path: Path) -> "CompassConfig":
        """Load config from JSON file with variable substitution.

        On corrupt/unreadable config (MCC-B-001 + BE-A-006), MOVES the bad
        file aside (rather than copy) so repeated load_config() calls don't
        spawn a new .bak.<ts> on every restart. The user gets a single,
        durable rescue copy and an actionable log line.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # BE-A-006: use a deterministic sentinel name (single .bak suffix)
            # so the backup count stays at 1 regardless of restart count. We
            # only stamp a timestamp if the .bak slot is already taken (to
            # preserve the FIRST corruption, which is usually the diagnostic
            # one). Move (rename) not copy so the original is gone — that
            # prevents next-restart from re-triggering the same branch.
            base_backup = path.with_suffix(path.suffix + ".bak")
            if base_backup.exists():
                backup_path: Optional[Path] = path.with_suffix(
                    path.suffix + f".bak.{int(time.time())}"
                )
            else:
                backup_path = base_backup
            try:
                # Path.rename atomically replaces / removes the source. Falls
                # back to copy+unlink if rename across filesystems fails.
                try:
                    path.rename(backup_path)
                except OSError:
                    shutil.copy2(path, backup_path)
                    try:
                        path.unlink()
                    except OSError:
                        pass
            except OSError:
                # If even the backup fails (e.g. path vanished), still fall
                # back rather than crash — the user needs a working tool.
                backup_path = None
            logger.error(
                f"Config file at {path} is corrupt: {e}.\n"
                f"Backup saved to {backup_path}.\n"
                f"Falling back to default config. Edit {path} or delete it to reset."
            )
            return get_default_config()

        # Get defaults for variable substitution
        defaults = data.get("defaults", {})

        # BE-A-015: track unresolved ${VAR} references so doctor() can flag
        # them. Substitution silently leaving ${FOO} in args is a debugging
        # trap — the backend then receives the literal string and fails
        # opaquely.
        unresolved: List[str] = []

        def resolve_var(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            default_val = defaults.get(var_name)
            if default_val is not None:
                return default_val
            if var_name not in unresolved:
                unresolved.append(var_name)
            return match.group(0)

        # Recursively substitute ${VAR} patterns
        def substitute(obj):
            if isinstance(obj, str):
                return re.sub(r"\$\{(\w+)\}", resolve_var, obj)
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            return obj

        data = substitute(data)
        cfg = cls.from_dict(data)
        # Stash unresolved names on the instance (non-serialized) for doctor().
        # The field isn't a declared dataclass attribute, so we attach it
        # directly — dataclasses without slots permit dynamic attributes.
        try:
            cfg._unresolved_vars = list(unresolved)
        except Exception:
            pass
        if unresolved:
            logger.warning(
                f"Config at {path} references unresolved variables: "
                f"{', '.join(unresolved)}. Set them in env or defaults block."
            )
        return cfg

    @classmethod
    def from_dict(cls, data: dict) -> "CompassConfig":
        """Create config from dictionary."""
        config = cls()

        # Parse backends
        for name, backend_data in data.get("backends", {}).items():
            # BR-A-018: a backend NAME containing ':' makes the "backend:tool"
            # qualified-name scheme ambiguous. qualified_name.split(":", 1) —
            # used by the router (backend_client_simple), the sync_manager
            # allow/deny policy check, and the gateway policy check — mis-splits
            # e.g. "grp:svc:tool" to server "grp", finds no config, and the
            # FEAT-06 allow/deny policy FAILS OPEN (a denied tool gets indexed
            # AND executed). Reject the ambiguous name here (fail-closed) so the
            # hole is closed at the source rather than papered over downstream.
            if ":" in name:
                raise ValueError(
                    f"Backend name {name!r} contains ':' which is reserved as "
                    f"the backend:tool separator — rename the backend without a "
                    f"colon."
                )
            backend_type = backend_data.get("type", "stdio")
            if backend_type == "stdio":
                config.backends[name] = StdioBackend(
                    command=backend_data.get("command", ""),
                    args=backend_data.get("args", []),
                    env=backend_data.get("env", {}),
                    cwd=backend_data.get("cwd"),
                    # INT-02 / FEAT-06: per-backend timeout + tool filters.
                    default_timeout=backend_data.get("default_timeout"),
                    tool_timeouts=backend_data.get("tool_timeouts", {}),
                    allow_tools=backend_data.get("allow_tools", []),
                    deny_tools=backend_data.get("deny_tools", []),
                )
            elif backend_type == "http":
                config.backends[name] = HttpBackend(
                    url=backend_data.get("url", ""),
                    headers=backend_data.get("headers", {}),
                    timeout=backend_data.get("timeout", 30.0),
                    # INT-02 / FEAT-06: per-backend timeout + tool filters.
                    default_timeout=backend_data.get("default_timeout"),
                    tool_timeouts=backend_data.get("tool_timeouts", {}),
                    allow_tools=backend_data.get("allow_tools", []),
                    deny_tools=backend_data.get("deny_tools", []),
                )
            elif backend_type == "import":
                config.backends[name] = ImportBackend(
                    module=backend_data.get("module", ""),
                    server_var=backend_data.get("server_var", "mcp"),
                    # FEAT-06: tool filters (import backends carry no timeout).
                    allow_tools=backend_data.get("allow_tools", []),
                    deny_tools=backend_data.get("deny_tools", []),
                )

        # Other settings
        config.embedding_model = data.get("embedding_model", config.embedding_model)
        config.ollama_url = data.get("ollama_url", config.ollama_url)

        # Pluggable embedding backend (BE-FT-PE-001).
        config.embedding_provider = data.get(
            "embedding_provider", config.embedding_provider
        )
        config.embedding_base_url = data.get(
            "embedding_base_url", config.embedding_base_url
        )
        config.embedding_api_key = data.get(
            "embedding_api_key", config.embedding_api_key
        )
        config.embedding_query_prefix = data.get(
            "embedding_query_prefix", config.embedding_query_prefix
        )
        config.embedding_document_prefix = data.get(
            "embedding_document_prefix", config.embedding_document_prefix
        )

        config.index_dir = data.get("index_dir", config.index_dir)
        config.auto_sync = data.get("auto_sync", config.auto_sync)
        config.default_top_k = data.get("default_top_k", config.default_top_k)
        config.min_confidence = data.get("min_confidence", config.min_confidence)
        config.progressive_disclosure = data.get(
            "progressive_disclosure", config.progressive_disclosure
        )

        # Hybrid search + exact-name boost (DISC-01/DISC-02) + analytics
        # retention (FEAT-04) + gateway auth token (OPS-1).
        config.hybrid_search = data.get("hybrid_search", config.hybrid_search)
        config.exact_name_boost = data.get(
            "exact_name_boost", config.exact_name_boost
        )
        config.exact_match_confidence = data.get(
            "exact_match_confidence", config.exact_match_confidence
        )
        config.analytics_retention_days = data.get(
            "analytics_retention_days", config.analytics_retention_days
        )
        config.gateway_auth_token = data.get(
            "gateway_auth_token", config.gateway_auth_token
        )

        # Sync settings
        config.sync_check_on_startup = data.get(
            "sync_check_on_startup", config.sync_check_on_startup
        )
        config.sync_polling_interval = data.get(
            "sync_polling_interval", config.sync_polling_interval
        )

        # Analytics settings
        config.analytics_enabled = data.get(
            "analytics_enabled", config.analytics_enabled
        )
        config.hot_cache_size = data.get("hot_cache_size", config.hot_cache_size)

        # Chain settings
        config.chain_indexing_enabled = data.get(
            "chain_indexing_enabled", config.chain_indexing_enabled
        )
        config.chain_detection_min_occurrences = data.get(
            "chain_detection_min_occurrences", config.chain_detection_min_occurrences
        )
        config.top_chains_cache_size = data.get(
            "top_chains_cache_size", config.top_chains_cache_size
        )

        # Circuit breaker / retry tuning (BE-B-014)
        config.ollama_breaker_failure_threshold = data.get(
            "ollama_breaker_failure_threshold",
            config.ollama_breaker_failure_threshold,
        )
        config.ollama_breaker_open_seconds = data.get(
            "ollama_breaker_open_seconds", config.ollama_breaker_open_seconds
        )
        config.ollama_retry_attempts = data.get(
            "ollama_retry_attempts", config.ollama_retry_attempts
        )
        backoffs = data.get("ollama_retry_backoffs")
        if backoffs is not None:
            try:
                config.ollama_retry_backoffs = [float(b) for b in backoffs]
            except (TypeError, ValueError):
                logger.warning(
                    "ollama_retry_backoffs must be a list of numbers; ignoring."
                )

        # HNSW tuning (BE-B-008)
        config.hnsw_m = data.get("hnsw_m", config.hnsw_m)
        config.hnsw_ef_construction = data.get(
            "hnsw_ef_construction", config.hnsw_ef_construction
        )
        config.hnsw_ef_search = data.get("hnsw_ef_search", config.hnsw_ef_search)

        # MCC-B-002: clamp out-of-range values rather than letting them slip
        # through and silently produce weird search/cache behavior downstream.
        config.validate_and_clamp()

        return config

    def validate_and_clamp(self) -> None:
        """Clamp config fields to safe ranges, logging each clamp.

        Silent acceptance of out-of-range values was causing debugging pain
        (e.g. negative polling intervals, hot_cache_size=0). Clamp here so
        the surface is always sane even with a hand-edited config file.

        CFG-A-002: COERCE numeric fields BEFORE the range comparisons. A
        hand-edited config can carry a string/null where a number is expected
        (e.g. ``"min_confidence": "high"``); comparing before coercing raised
        TypeError, and from_file's recovery except only catches JSON/OS errors,
        so startup crashed with a raw traceback. We coerce-or-reset each
        numeric field up front: if the value can't become the right numeric
        type, reset it to the class default and warn.
        """
        # CFG-A-002: coerce-or-reset every numeric field first, so the range
        # checks below always operate on numbers. (field, caster, default)
        _defaults = CompassConfig
        for fname, caster in (
            ("min_confidence", float),
            ("default_top_k", int),
            ("sync_polling_interval", int),
            ("hot_cache_size", int),
            ("top_chains_cache_size", int),
            ("chain_detection_min_occurrences", int),
            ("ollama_breaker_failure_threshold", int),
            ("ollama_breaker_open_seconds", float),
            ("ollama_retry_attempts", int),
            ("hnsw_m", int),
            ("hnsw_ef_construction", int),
            ("hnsw_ef_search", int),
            # DISC-02 / FEAT-04: coerce the new numeric knobs up front too, so
            # a hand-edited string/null resets to default instead of crashing
            # the range checks below.
            ("exact_match_confidence", float),
            ("analytics_retention_days", int),
        ):
            value = getattr(self, fname)
            try:
                # bool is an int subclass; treat it as the numeric it casts to.
                setattr(self, fname, caster(value))
            except (TypeError, ValueError):
                default_value = getattr(_defaults, fname)
                logger.warning(
                    f"Config value {fname}={value!r} is not numeric; "
                    f"resetting to default {default_value!r}"
                )
                setattr(self, fname, default_value)

        # min_confidence: [0.0, 1.0]
        if not 0.0 <= self.min_confidence <= 1.0:
            original = self.min_confidence
            clamped = max(0.0, min(1.0, float(self.min_confidence)))
            logger.warning(
                f"Config value min_confidence clamped from {original} to {clamped}"
            )
            self.min_confidence = clamped

        # default_top_k: [1, 50]
        if not 1 <= self.default_top_k <= 50:
            original = self.default_top_k
            clamped = max(1, min(50, int(self.default_top_k)))
            logger.warning(
                f"Config value default_top_k clamped from {original} to {clamped}"
            )
            self.default_top_k = clamped

        # sync_polling_interval: max(0, value). 0 disables polling by design.
        if self.sync_polling_interval < 0:
            original = self.sync_polling_interval
            clamped = max(0, int(self.sync_polling_interval))
            logger.warning(
                f"Config value sync_polling_interval clamped from {original} to {clamped}"
            )
            self.sync_polling_interval = clamped

        # hot_cache_size: max(1, value). Zero would disable the cache silently.
        if self.hot_cache_size < 1:
            original = self.hot_cache_size
            clamped = max(1, int(self.hot_cache_size))
            logger.warning(
                f"Config value hot_cache_size clamped from {original} to {clamped}"
            )
            self.hot_cache_size = clamped

        # top_chains_cache_size: max(0, value). It is used as a slice bound
        # (chain_indexer.py: chains[:n]); a hand-edited 0/negative would
        # silently empty/truncate the chain cache with no warning. 0 is allowed
        # by design — it disables the in-memory chain cache.
        if self.top_chains_cache_size < 0:
            original = self.top_chains_cache_size
            clamped = max(0, int(self.top_chains_cache_size))
            logger.warning(
                f"Config value top_chains_cache_size clamped from {original} to {clamped}"
            )
            self.top_chains_cache_size = clamped

        # chain_detection_min_occurrences: max(2, value). Below 2, every pair
        # of tools becomes a "chain" and the detector drowns in noise.
        if self.chain_detection_min_occurrences < 2:
            original = self.chain_detection_min_occurrences
            clamped = max(2, int(self.chain_detection_min_occurrences))
            logger.warning(
                f"Config value chain_detection_min_occurrences clamped from {original} to {clamped}"
            )
            self.chain_detection_min_occurrences = clamped

        # BE-B-014: clamp circuit-breaker tuning to safe ranges.
        if not 1 <= self.ollama_breaker_failure_threshold <= 20:
            original = self.ollama_breaker_failure_threshold
            clamped = max(1, min(20, int(self.ollama_breaker_failure_threshold)))
            logger.warning(
                f"Config value ollama_breaker_failure_threshold clamped from "
                f"{original} to {clamped}"
            )
            self.ollama_breaker_failure_threshold = clamped

        if not 1.0 <= self.ollama_breaker_open_seconds <= 600.0:
            original = self.ollama_breaker_open_seconds
            clamped = max(1.0, min(600.0, float(self.ollama_breaker_open_seconds)))
            logger.warning(
                f"Config value ollama_breaker_open_seconds clamped from "
                f"{original} to {clamped}"
            )
            self.ollama_breaker_open_seconds = clamped

        if not 0 <= self.ollama_retry_attempts <= 10:
            original = self.ollama_retry_attempts
            clamped = max(0, min(10, int(self.ollama_retry_attempts)))
            logger.warning(
                f"Config value ollama_retry_attempts clamped from "
                f"{original} to {clamped}"
            )
            self.ollama_retry_attempts = clamped

        if not isinstance(self.ollama_retry_backoffs, list) or not all(
            isinstance(b, (int, float)) and b >= 0 for b in self.ollama_retry_backoffs
        ):
            logger.warning(
                "ollama_retry_backoffs must be a list of non-negative numbers; "
                "resetting to [0.5, 1.0, 2.0]"
            )
            self.ollama_retry_backoffs = [0.5, 1.0, 2.0]

        # BE-B-008: clamp HNSW knobs to safe ranges.
        if not 4 <= self.hnsw_m <= 64:
            original = self.hnsw_m
            self.hnsw_m = max(4, min(64, int(self.hnsw_m)))
            logger.warning(
                f"Config value hnsw_m clamped from {original} to {self.hnsw_m}"
            )
        if not 40 <= self.hnsw_ef_construction <= 800:
            original = self.hnsw_ef_construction
            self.hnsw_ef_construction = max(
                40, min(800, int(self.hnsw_ef_construction))
            )
            logger.warning(
                f"Config value hnsw_ef_construction clamped from "
                f"{original} to {self.hnsw_ef_construction}"
            )
        if not 10 <= self.hnsw_ef_search <= 400:
            original = self.hnsw_ef_search
            self.hnsw_ef_search = max(10, min(400, int(self.hnsw_ef_search)))
            logger.warning(
                f"Config value hnsw_ef_search clamped from "
                f"{original} to {self.hnsw_ef_search}"
            )

        # DISC-02: exact_match_confidence is a confidence score in [0.0, 1.0].
        if not 0.0 <= self.exact_match_confidence <= 1.0:
            original = self.exact_match_confidence
            clamped = max(0.0, min(1.0, float(self.exact_match_confidence)))
            logger.warning(
                f"Config value exact_match_confidence clamped from "
                f"{original} to {clamped}"
            )
            self.exact_match_confidence = clamped

        # FEAT-04: analytics_retention_days >= 0. 0 disables pruning (keep
        # forever) by design; a negative would be meaningless as a cutoff.
        if self.analytics_retention_days < 0:
            original = self.analytics_retention_days
            clamped = max(0, int(self.analytics_retention_days))
            logger.warning(
                f"Config value analytics_retention_days clamped from "
                f"{original} to {clamped}"
            )
            self.analytics_retention_days = clamped

        # INT-02 / FEAT-06: normalize per-backend timeout + tool-filter fields.
        self._validate_and_clamp_backends()

        # BE-FT-PE-001: validate embedding_provider against the known
        # providers; an unknown value warns + falls back to "ollama" so a
        # typo degrades to working behavior rather than breaking the embed
        # path. Import locally to avoid a config<->embedder import cycle.
        try:
            from embedder import known_providers, DEFAULT_PROVIDER
        except Exception:  # pragma: no cover - embedder import should not fail
            known_providers = None
            DEFAULT_PROVIDER = "ollama"
        provider = self.embedding_provider
        normalized = provider.strip().lower() if isinstance(provider, str) else None
        valid = set(known_providers()) if known_providers is not None else None
        if normalized is None or (valid is not None and normalized not in valid):
            logger.warning(
                "Config value embedding_provider=%r is not a known provider; "
                "falling back to %r%s",
                provider,
                DEFAULT_PROVIDER,
                f" (known: {', '.join(sorted(valid))})" if valid else "",
            )
            self.embedding_provider = DEFAULT_PROVIDER
        else:
            # Store the normalized (lower-cased, stripped) name.
            self.embedding_provider = normalized

    # INT-02 / FEAT-06: per-backend timeout + tool-filter bounds. Shared with
    # the CompassConfig-level clamps so hand-edited backend configs are just as
    # safe as top-level ones.
    _TIMEOUT_MIN = 1.0
    _TIMEOUT_MAX = 600.0

    def _validate_and_clamp_backends(self) -> None:
        """Coerce/clamp the per-backend timeout + tool-filter fields.

        Mirrors the "warn on every clamp" convention used for the top-level
        numeric knobs:

        - INT-02 default_timeout: coerce to float or reset to None (with a
          warning) if non-numeric; else clamp to [1.0, 600.0].
        - INT-02 tool_timeouts: drop non-numeric entries (with a warning);
          clamp each surviving value to [1.0, 600.0].
        - FEAT-06 allow_tools / deny_tools: coerce to list-of-str; a non-list
          resets to [] with a warning.
        """
        for name, backend in self.backends.items():
            # default_timeout / tool_timeouts only exist on stdio + http.
            if hasattr(backend, "default_timeout"):
                dt = backend.default_timeout
                if dt is not None:
                    try:
                        dt = float(dt)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Backend {name!r} default_timeout={dt!r} is not "
                            f"numeric; resetting to None"
                        )
                        dt = None
                    else:
                        clamped = max(
                            self._TIMEOUT_MIN, min(self._TIMEOUT_MAX, dt)
                        )
                        if clamped != dt:
                            logger.warning(
                                f"Backend {name!r} default_timeout clamped from "
                                f"{dt} to {clamped}"
                            )
                            dt = clamped
                    backend.default_timeout = dt

            if hasattr(backend, "tool_timeouts"):
                raw = backend.tool_timeouts
                if not isinstance(raw, dict):
                    logger.warning(
                        f"Backend {name!r} tool_timeouts is not a mapping; "
                        f"resetting to {{}}"
                    )
                    backend.tool_timeouts = {}
                else:
                    cleaned: Dict[str, float] = {}
                    for tool_name, value in raw.items():
                        try:
                            fval = float(value)
                        except (TypeError, ValueError):
                            logger.warning(
                                f"Backend {name!r} tool_timeouts[{tool_name!r}]="
                                f"{value!r} is not numeric; dropping entry"
                            )
                            continue
                        clamped = max(
                            self._TIMEOUT_MIN, min(self._TIMEOUT_MAX, fval)
                        )
                        if clamped != fval:
                            logger.warning(
                                f"Backend {name!r} tool_timeouts[{tool_name!r}] "
                                f"clamped from {fval} to {clamped}"
                            )
                        cleaned[str(tool_name)] = clamped
                    backend.tool_timeouts = cleaned

            # FEAT-06: allow_tools / deny_tools exist on all three backends.
            for attr in ("allow_tools", "deny_tools"):
                if hasattr(backend, attr):
                    value = getattr(backend, attr)
                    if isinstance(value, list):
                        setattr(backend, attr, [str(v) for v in value])
                    else:
                        logger.warning(
                            f"Backend {name!r} {attr} is not a list; "
                            f"resetting to []"
                        )
                        setattr(backend, attr, [])

    def resolved_embedding_base_url(self) -> str:
        """Return the base URL the embedder should POST to (BE-FT-PE-001).

        For the ollama provider, ``embedding_base_url`` is an optional
        override of ``ollama_url`` (so the legacy single-URL config keeps
        working). For openai-compatible, ``embedding_base_url`` is the
        configured endpoint base; if an operator left it unset we fall back to
        ``ollama_url`` and log a warning so the misconfiguration is visible
        rather than silently hitting the Ollama port with the wrong contract.
        """
        if self.embedding_base_url:
            return self.embedding_base_url
        if self.embedding_provider != "ollama":
            logger.warning(
                "embedding_provider=%r but embedding_base_url is unset; "
                "falling back to ollama_url=%r. Set embedding_base_url to the "
                "provider endpoint.",
                self.embedding_provider,
                self.ollama_url,
            )
        return self.ollama_url

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        backends = {}
        for name, backend in self.backends.items():
            if isinstance(backend, StdioBackend):
                backends[name] = {
                    "type": "stdio",
                    "command": backend.command,
                    "args": backend.args,
                    "env": backend.env,
                    "cwd": backend.cwd,
                    "default_timeout": backend.default_timeout,
                    "tool_timeouts": backend.tool_timeouts,
                    "allow_tools": backend.allow_tools,
                    "deny_tools": backend.deny_tools,
                }
            elif isinstance(backend, HttpBackend):
                backends[name] = {
                    "type": "http",
                    "url": backend.url,
                    "headers": backend.headers,
                    "timeout": backend.timeout,
                    "default_timeout": backend.default_timeout,
                    "tool_timeouts": backend.tool_timeouts,
                    "allow_tools": backend.allow_tools,
                    "deny_tools": backend.deny_tools,
                }
            elif isinstance(backend, ImportBackend):
                backends[name] = {
                    "type": "import",
                    "module": backend.module,
                    "server_var": backend.server_var,
                    "allow_tools": backend.allow_tools,
                    "deny_tools": backend.deny_tools,
                }

        return {
            "backends": backends,
            "embedding_model": self.embedding_model,
            "ollama_url": self.ollama_url,
            "embedding_provider": self.embedding_provider,
            "embedding_base_url": self.embedding_base_url,
            "embedding_api_key": self.embedding_api_key,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_document_prefix": self.embedding_document_prefix,
            "index_dir": self.index_dir,
            "auto_sync": self.auto_sync,
            "default_top_k": self.default_top_k,
            "min_confidence": self.min_confidence,
            "hybrid_search": self.hybrid_search,
            "exact_name_boost": self.exact_name_boost,
            "exact_match_confidence": self.exact_match_confidence,
            "analytics_retention_days": self.analytics_retention_days,
            "gateway_auth_token": self.gateway_auth_token,
            "progressive_disclosure": self.progressive_disclosure,
            "sync_check_on_startup": self.sync_check_on_startup,
            "sync_polling_interval": self.sync_polling_interval,
            "analytics_enabled": self.analytics_enabled,
            "hot_cache_size": self.hot_cache_size,
            "chain_indexing_enabled": self.chain_indexing_enabled,
            "chain_detection_min_occurrences": self.chain_detection_min_occurrences,
            "top_chains_cache_size": self.top_chains_cache_size,
            "ollama_breaker_failure_threshold": self.ollama_breaker_failure_threshold,
            "ollama_breaker_open_seconds": self.ollama_breaker_open_seconds,
            "ollama_retry_attempts": self.ollama_retry_attempts,
            "ollama_retry_backoffs": list(self.ollama_retry_backoffs),
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_base_path() -> Path:
    """
    Get the base path for the project.

    Resolution order:
    1. TOOL_COMPASS_BASE_PATH environment variable
    2. Parent of tool_compass directory (typical install)
    """
    env_path = os.environ.get("TOOL_COMPASS_BASE_PATH")
    if env_path:
        return Path(env_path).resolve()

    # Default: parent of tool_compass directory
    return Path(__file__).parent.parent.resolve()


def get_python_executable() -> str:
    """
    Get the Python executable path.

    Resolution order:
    1. TOOL_COMPASS_PYTHON environment variable
    2. Current Python interpreter (sys.executable)
    3. Platform-specific venv detection
    """
    env_python = os.environ.get("TOOL_COMPASS_PYTHON")
    if env_python:
        return env_python

    # Use current interpreter if running from venv
    if sys.prefix != sys.base_prefix:
        return sys.executable

    # Try to find venv in base path
    base_path = get_base_path()

    # Platform-specific venv paths
    if sys.platform == "win32":
        venv_python = base_path / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = base_path / "venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    # Fallback to current interpreter
    return sys.executable


def _env_truthy(value: Optional[str]) -> bool:
    """Interpret an env-var string as a boolean.

    CFGDOC-01: ``.env.example`` advertises ``TOOL_COMPASS_ANALYTICS_DISABLED``
    as a flag; users reasonably set it to ``true`` / ``1`` / ``yes``. Treat the
    common truthy spellings as True and everything else (including the empty
    string and ``false``/``0``/``no``) as False, so an exported-but-empty var
    doesn't accidentally disable analytics.
    """
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def apply_env_overrides(config: CompassConfig) -> CompassConfig:
    """Apply env-var overrides that ``.env.example`` documents, in place.

    CFGDOC-01: two vars were advertised in ``.env.example`` but never read —
    the app honored them only via config-JSON keys, so a user who set the
    documented env var saw no effect. Wire them here so the documentation is
    truthful:

    - ``TOOL_COMPASS_ANALYTICS_DISABLED`` (truthy) -> ``analytics_enabled=False``
    - ``TOOL_COMPASS_HOT_CACHE_SIZE`` (int) -> ``hot_cache_size``

    The hot-cache value flows through ``validate_and_clamp()`` so an out-of-range
    or non-numeric env value is clamped/reset with the same warning as a
    hand-edited config file (rather than crashing or silently disabling the
    cache). Env overrides win over file/default values by design — they are the
    most specific signal of operator intent.
    """
    changed = False

    if "TOOL_COMPASS_ANALYTICS_DISABLED" in os.environ:
        if _env_truthy(os.environ.get("TOOL_COMPASS_ANALYTICS_DISABLED")):
            config.analytics_enabled = False
            changed = True

    raw_cache = os.environ.get("TOOL_COMPASS_HOT_CACHE_SIZE")
    if raw_cache is not None and raw_cache.strip() != "":
        try:
            config.hot_cache_size = int(raw_cache)
            changed = True
        except (TypeError, ValueError):
            logger.warning(
                f"TOOL_COMPASS_HOT_CACHE_SIZE={raw_cache!r} is not an integer; "
                "ignoring env override."
            )

    # BE-FT-PE-001: the embedding API key is a secret — prefer the env var
    # over a value baked into the config file so operators can keep the key out
    # of the on-disk config entirely. An empty string is ignored (treated as
    # "not set") so an exported-but-empty var doesn't blank out a file value.
    raw_api_key = os.environ.get("TOOL_COMPASS_EMBEDDING_API_KEY")
    if raw_api_key is not None and raw_api_key.strip() != "":
        config.embedding_api_key = raw_api_key
        changed = True

    # OPS-1: the gateway auth token is a secret — prefer the env var over a
    # value baked into the config file so operators can keep the token off
    # disk. An empty string is ignored (treated as "not set") so an
    # exported-but-empty var doesn't blank out a file value.
    raw_gw_token = os.environ.get("TOOL_COMPASS_GATEWAY_AUTH_TOKEN")
    if raw_gw_token is not None and raw_gw_token.strip() != "":
        config.gateway_auth_token = raw_gw_token
        changed = True

    # Re-clamp so the hot_cache_size env value lands in the same safe range
    # (and emits the same warning) as a config-file value would.
    if changed:
        config.validate_and_clamp()

    return config


def get_default_config() -> CompassConfig:
    """
    Get default config with no backends configured.

    Users should create a compass_config.json with their own backend configurations.
    See get_example_config() for an example configuration.

    Uses environment variables for Ollama URL and other settings.
    """
    config = CompassConfig(
        backends={},  # No backends by default - user must configure
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        auto_sync=True,
        progressive_disclosure=True,
    )
    return apply_env_overrides(config)


def get_example_config() -> CompassConfig:
    """
    Get example config demonstrating backend configuration patterns.

    This is for documentation purposes - actual paths will vary by installation.
    Copy compass_config.example.json to compass_config.json and customize.

    Uses environment variables and auto-detection for cross-platform support.
    Set TOOL_COMPASS_BASE_PATH to override the project root.
    """
    base_path = get_base_path()
    python_exe = get_python_executable()

    # Default environment for all backends
    base_env = {
        "PYTHONPATH": str(base_path),
        "PYTHONIOENCODING": "utf-8",
    }

    return CompassConfig(
        backends={
            "bridge": StdioBackend(
                command=python_exe,
                args=["-u", str(base_path / "app/mcp/bridge_mcp_server.py")],
                env=base_env.copy(),
            ),
            "comfy": StdioBackend(
                command=python_exe,
                args=["-u", str(base_path / "app/mcp/comfy_mcp_server.py")],
                env={
                    **base_env,
                    "COMFYUI_URL": os.environ.get(
                        "COMFYUI_URL", "http://localhost:8188"
                    ),
                },
            ),
            "video": StdioBackend(
                command=python_exe,
                args=["-u", str(base_path / "app/mcp/video_mcp_server.py")],
                env=base_env.copy(),
            ),
            "chat": StdioBackend(
                command=python_exe,
                args=["-u", str(base_path / "app/mcp/chat_mcp_server.py")],
                env=base_env.copy(),
            ),
            "doc": StdioBackend(
                command=python_exe,
                args=["-u", str(base_path / "app/mcp/doc_mcp_server.py")],
                env=base_env.copy(),
            ),
        },
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        auto_sync=True,
        progressive_disclosure=True,
    )


def get_user_config_dir() -> Path:
    """
    Get the user-writable config directory for Tool Compass.

    Resolution order:
    1. TOOL_COMPASS_DATA_DIR environment variable
    2. Platform-specific user config directory:
       - Windows: %LOCALAPPDATA%/tool-compass
       - macOS: ~/Library/Application Support/tool-compass
       - Linux: ~/.config/tool-compass (or $XDG_CONFIG_HOME/tool-compass)
    3. Fallback: .tool-compass in current working directory (if HOME unavailable)
    """
    env_dir = os.environ.get("TOOL_COMPASS_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    def _safe_home() -> Path:
        """Get home directory with fallback for CI environments."""
        try:
            return Path.home()
        except (RuntimeError, KeyError):
            # HOME/USERPROFILE not set (common in CI)
            return Path.cwd()

    if sys.platform == "win32":
        # Windows: use LOCALAPPDATA
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "tool-compass"
        return _safe_home() / "AppData" / "Local" / "tool-compass"
    elif sys.platform == "darwin":
        # macOS: use Application Support
        return _safe_home() / "Library" / "Application Support" / "tool-compass"
    else:
        # Linux/Unix: use XDG_CONFIG_HOME or ~/.config
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "tool-compass"
        return _safe_home() / ".config" / "tool-compass"


def get_config_path() -> Path:
    """
    Get the config file path.

    Resolution order:
    1. TOOL_COMPASS_CONFIG environment variable
    2. User config directory: <user_config_dir>/compass_config.json
    """
    env_config = os.environ.get("TOOL_COMPASS_CONFIG")
    if env_config:
        return Path(env_config).resolve()
    return get_user_config_dir() / "compass_config.json"


# Default config file location (for backward compatibility)
CONFIG_PATH = get_config_path()


def load_config() -> CompassConfig:
    """Load config from file or return defaults.

    CFGDOC-01: env overrides (TOOL_COMPASS_ANALYTICS_DISABLED /
    TOOL_COMPASS_HOT_CACHE_SIZE) are applied on top of the file-loaded config
    too, so the documented env vars work whether or not a config file exists.
    They win over file values — the env var is the more specific operator
    signal. (get_default_config already applies them on the no-file path.)
    """
    config_path = get_config_path()
    if config_path.exists():
        return apply_env_overrides(CompassConfig.from_file(config_path))
    return get_default_config()


# Field-name substrings that indicate a secret — redacted in doctor() dumps.
# No such fields exist today, but the scan is defensive in case someone adds
# a "github_token" or "api_key" field and forgets to redact it.
_SECRET_FIELD_HINTS = ("_token", "_key", "_secret", "_password")


# CFG-A-001: backend sub-fields that carry ${VAR}-resolved secrets. Their KEY
# names ('Authorization', 'GITHUB_TOKEN', '--password=...') don't all match
# _SECRET_FIELD_HINTS, so name-based redaction misses them. Redact these
# STRUCTURALLY — every value (dict) or entry (list) — while keeping the keys
# visible so the doctor() dump stays diagnosable.
_SECRET_STRUCT_FIELDS = ("env", "headers", "args")


def _redact_structural(value):
    """Redact a backend env/headers (dict values) or args (list entries),
    preserving keys/structure so the dump shows e.g. 'Authorization:
    [REDACTED]' rather than dropping the field entirely."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" for k in value}
    if isinstance(value, list):
        return ["[REDACTED]" for _ in value]
    return "[REDACTED]"


def redact_url_credentials(value):
    """CFG-A-001 (post-fix sibling): strip userinfo from an http(s) URL so a
    credentialed endpoint like ``http://user:${TOKEN}@host:11434`` (ollama_url
    or a backend ``url``, both ${VAR}-substituted at load) doesn't leak its
    secret into the doctor()/show_config diagnostic dumps. Host:port is kept
    for diagnosability. Non-URL / credential-free values pass through."""
    if not isinstance(value, str) or "://" not in value:
        return value
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not (parts.username or parts.password):
        return value
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit(
        (parts.scheme, f"[REDACTED]@{host}", parts.path, parts.query, parts.fragment)
    )


def _redact_config(cfg_dict: dict) -> dict:
    """Walk the config dict and redact secrets.

    Two layers:
    1. Name-based — any field whose key hints at a secret (_SECRET_FIELD_HINTS).
    2. Structural (CFG-A-001) — under each backend, the 'env'/'headers'/'args'
       fields carry ${VAR}-resolved secrets whose keys don't match the name
       hints, so their values/entries are redacted structurally with keys kept.
    """

    def walk(obj):
        if isinstance(obj, dict):
            redacted = {}
            for k, v in obj.items():
                if isinstance(k, str) and any(
                    hint in k.lower() for hint in _SECRET_FIELD_HINTS
                ):
                    redacted[k] = "[REDACTED]"
                elif isinstance(k, str) and k in _SECRET_STRUCT_FIELDS:
                    redacted[k] = _redact_structural(v)
                else:
                    redacted[k] = walk(v)
            return redacted
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        # Leaf scalar: scrub embedded URL credentials (ollama_url, backend url).
        return redact_url_credentials(obj)

    return walk(cfg_dict)


def _ollama_reachable(url: str, timeout: float = 2.0) -> bool:
    """Quick reachability probe for the doctor dump. Never blocks > timeout."""
    try:
        import httpx  # local import — keeps module-level imports stable
    except ImportError:
        return False
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def doctor() -> dict:
    """Produce a JSON-serializable diagnostic dump.

    MCC-B-004: one-shot environment snapshot for bug reports. Captures
    version, platform, resolved paths, file sizes, and an Ollama reachability
    probe. Secrets are redacted defensively on field-name match. Ollama probe
    has a hard 2s timeout so `python config.py` never hangs on a dead server.
    """
    # Local imports to avoid polluting the module namespace with rarely-used
    # stdlib paths and to keep import-time cost low on the hot path.
    from _version import __version__

    # MCC-FT-002 bonus: include deprecated tool count in the diagnostic dump
    # so bug reports surface stale tools without a separate introspection
    # step. Defensive — tool_manifest should always import, but if it breaks
    # we still want doctor() to succeed for the user.
    deprecated_tools_count: Optional[int] = None
    try:
        from tool_manifest import get_all_tools

        deprecated_tools_count = sum(
            1 for t in get_all_tools() if t.deprecated_since is not None
        )
    except Exception as e:
        logger.debug(f"doctor(): could not count deprecated tools: {e}")

    cfg_path = get_config_path()
    cfg = load_config()
    cfg_dict = _redact_config(cfg.to_dict())

    base_path = get_base_path()
    data_dir = get_user_config_dir()
    python_exe = get_python_executable()

    index_path = Path(cfg.index_dir)
    if not index_path.is_absolute():
        index_path = (base_path / cfg.index_dir).resolve()
    index_exists = index_path.exists()
    index_size = (
        sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file())
        if index_exists and index_path.is_dir()
        else (index_path.stat().st_size if index_exists else 0)
    )

    # Analytics DB lives at <module_dir>/db/compass_analytics.db — matches
    # analytics.ANALYTICS_DB_PATH. Resolve here without importing analytics
    # to avoid dragging in sqlite3 at doctor-run time.
    analytics_db_path = Path(__file__).parent / "db" / "compass_analytics.db"
    analytics_exists = analytics_db_path.exists()
    analytics_size = analytics_db_path.stat().st_size if analytics_exists else 0
    analytics_schema_version: Optional[int] = None
    if analytics_exists:
        try:
            import sqlite3

            with sqlite3.connect(str(analytics_db_path)) as _db:
                row = _db.execute(
                    "SELECT value FROM schema_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row:
                    analytics_schema_version = int(row[0])
        except Exception:
            # Old DB without schema_meta is fine — just report unknown.
            analytics_schema_version = None

    unresolved_vars = list(getattr(cfg, "_unresolved_vars", []) or [])

    # CFGDOC-04: surface the analytics degraded state ("sqlite broke, metrics
    # silently stopped") in bug reports — but ONLY if a live analytics
    # singleton already exists. We must NOT force-create one here: doctor() is
    # a read-only snapshot, and instantiating CompassAnalytics would open the
    # DB and write the schema as a side effect of running diagnostics. Import
    # the module to read its singleton without constructing the class.
    analytics_health: Optional[dict] = None
    try:
        import analytics as _analytics_mod

        if _analytics_mod._analytics_instance is not None:
            analytics_health = _analytics_mod._analytics_instance.get_health()
    except Exception as e:
        logger.debug(f"doctor(): could not read analytics health: {e}")

    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": _platform.platform(),
        "config_path": str(cfg_path),
        "config": cfg_dict,
        # BE-A-015: surface unresolved ${VAR} references so bug reports
        # capture them. Empty list means the config substituted cleanly.
        "config_unresolved_vars": unresolved_vars,
        "base_path": str(base_path),
        "data_dir": str(data_dir),
        "python_executable": python_exe,
        "index_path": str(index_path),
        "index_exists": index_exists,
        "index_size_bytes": index_size,
        "analytics_db_path": str(analytics_db_path),
        "analytics_exists": analytics_exists,
        "analytics_size_bytes": analytics_size,
        "analytics_schema_version": analytics_schema_version,
        # CFGDOC-04: None when no live analytics singleton exists (e.g. a fresh
        # `python config.py` run); otherwise the {'degraded', 'reason'} health
        # dict so bug reports capture the "metrics silently stopped" mode.
        "analytics_health": analytics_health,
        # CFG-A-001 sibling: scrub any user:pass@ embedded in the URL before
        # it lands in a pasteable bug-report dump. The reachability probe below
        # still uses the raw cfg value (it returns only a bool, not the URL).
        "ollama_url": redact_url_credentials(cfg.ollama_url),
        "ollama_reachable": _ollama_reachable(cfg.ollama_url),
        "deprecated_tools": deprecated_tools_count,
    }


if __name__ == "__main__":
    # Human-runnable diagnostic dump — `python config.py` for bug reports.
    print(json.dumps(doctor(), indent=2, default=str))
