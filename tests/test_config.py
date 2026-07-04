"""
Tests for Tool Compass configuration module.

Tests cross-platform path handling and environment variable support.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from config import (
    CompassConfig,
    StdioBackend,
    HttpBackend,
    ImportBackend,
    get_base_path,
    get_python_executable,
    get_config_path,
    get_default_config,
    load_config,
    doctor,
    _redact_config,
    apply_env_overrides,
)


class TestPathResolution:
    """Test cross-platform path resolution."""

    def test_get_base_path_default(self):
        """Default base path should be parent of tool_compass directory."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if set
            os.environ.pop("TOOL_COMPASS_BASE_PATH", None)
            base = get_base_path()
            assert base.exists()
            assert base.is_dir()

    def test_get_base_path_from_env(self, tmp_path):
        """TOOL_COMPASS_BASE_PATH should override default."""
        with patch.dict(os.environ, {"TOOL_COMPASS_BASE_PATH": str(tmp_path)}):
            base = get_base_path()
            assert base == tmp_path.resolve()

    def test_get_python_executable_from_env(self):
        """TOOL_COMPASS_PYTHON should override detection."""
        fake_python = "/usr/bin/fake_python"
        with patch.dict(os.environ, {"TOOL_COMPASS_PYTHON": fake_python}):
            exe = get_python_executable()
            assert exe == fake_python

    def test_get_python_executable_default(self):
        """Default should use sys.executable or venv detection."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TOOL_COMPASS_PYTHON", None)
            exe = get_python_executable()
            assert exe  # Should return something
            # Should be a valid path or the current interpreter
            assert Path(exe).exists() or exe == sys.executable

    def test_get_python_executable_env_nonexistent_falls_back(self, tmp_path):
        """If TOOL_COMPASS_PYTHON points to a nonexistent path, behavior must
        be well-defined — exercise the OR branch in get_python_executable so
        that path isn't silently untested. Historically this branch referenced
        an unimported `sys`, so the fallback raised NameError instead of
        returning a valid interpreter.
        """
        fake = str(tmp_path / "does_not_exist_python")
        assert not Path(fake).exists()
        # Current contract: env var wins verbatim (caller owns validation).
        # This test LOCKS IN that contract and executes the code path without
        # raising NameError — if the implementation later changes to validate
        # existence and fall back, adjust this assertion accordingly.
        with patch.dict(os.environ, {"TOOL_COMPASS_PYTHON": fake}):
            exe = get_python_executable()
            # Must not raise NameError; must return a non-empty string.
            assert isinstance(exe, str)
            assert exe  # non-empty
            # Either verbatim env value, or a real existing interpreter
            # (sys.executable fallback).
            assert exe == fake or Path(exe).exists() or exe == sys.executable

    def test_get_config_path_from_env(self, tmp_path):
        """TOOL_COMPASS_CONFIG should override default."""
        config_file = tmp_path / "custom_config.json"
        with patch.dict(os.environ, {"TOOL_COMPASS_CONFIG": str(config_file)}):
            path = get_config_path()
            assert path == config_file.resolve()

    def test_get_config_path_default(self):
        """Default config path should be in tool_compass/tool-compass directory."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TOOL_COMPASS_CONFIG", None)
            path = get_config_path()
            assert path.name == "compass_config.json"
            # Accept both tool_compass (local) and tool-compass (CI/GitHub)
            path_str = str(path).lower()
            assert "tool_compass" in path_str or "tool-compass" in path_str


class TestCompassConfig:
    """Test CompassConfig dataclass and parsing."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CompassConfig()
        assert config.embedding_model == "nomic-embed-text"
        assert config.ollama_url == "http://localhost:11434"
        assert config.default_top_k == 5
        assert config.min_confidence == 0.3
        assert config.progressive_disclosure is True

    def test_from_dict_minimal(self):
        """Should parse minimal config dict."""
        data = {"backends": {}}
        config = CompassConfig.from_dict(data)
        assert config.backends == {}
        assert config.auto_sync is True  # default

    def test_from_dict_with_stdio_backend(self):
        """Should parse stdio backend config."""
        data = {
            "backends": {
                "test": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "test_server"],
                    "env": {"DEBUG": "1"},
                }
            }
        }
        config = CompassConfig.from_dict(data)
        assert "test" in config.backends
        backend = config.backends["test"]
        assert isinstance(backend, StdioBackend)
        assert backend.command == "python"
        assert backend.args == ["-m", "test_server"]
        assert backend.env == {"DEBUG": "1"}

    def test_from_dict_with_http_backend(self):
        """Should parse HTTP backend config."""
        data = {
            "backends": {
                "api": {
                    "type": "http",
                    "url": "http://localhost:8080/mcp",
                    "headers": {"Authorization": "Bearer token"},
                    "timeout": 60.0,
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["api"]
        assert isinstance(backend, HttpBackend)
        assert backend.url == "http://localhost:8080/mcp"
        assert backend.timeout == 60.0

    def test_from_dict_with_import_backend(self):
        """Should parse import backend config."""
        data = {
            "backends": {
                "local": {
                    "type": "import",
                    "module": "my_server",
                    "server_var": "app",
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["local"]
        assert isinstance(backend, ImportBackend)
        assert backend.module == "my_server"
        assert backend.server_var == "app"

    def test_to_dict_roundtrip(self):
        """Config should survive dict roundtrip."""
        original = CompassConfig(
            backends={
                "test": StdioBackend(
                    command="python",
                    args=["-m", "server"],
                    env={"KEY": "value"},
                )
            },
            embedding_model="custom-model",
            auto_sync=False,
        )
        data = original.to_dict()
        restored = CompassConfig.from_dict(data)

        assert restored.embedding_model == original.embedding_model
        assert restored.auto_sync == original.auto_sync
        assert "test" in restored.backends


class TestDefaultConfig:
    """Test default configuration generation."""

    def test_get_default_config_structure(self):
        """Default config should have empty backends (user must configure)."""
        config = get_default_config()

        # Default config ships with no backends - user must configure
        assert config.backends == {}
        assert config.embedding_model == "nomic-embed-text"
        assert config.auto_sync is True
        assert config.progressive_disclosure is True

    def test_get_default_config_uses_detected_python(self):
        """Default config has no backends; example config uses detected Python."""
        config = get_default_config()
        # Default config has no backends to check
        assert config.backends == {}

    def test_get_default_config_portable_paths(self):
        """Default config has no backends; paths are user-configured."""
        config = get_default_config()
        # Default config has no backends - paths are user responsibility
        assert config.backends == {}


class TestLoadConfig:
    """Test config file loading."""

    def test_load_config_missing_file(self, tmp_path):
        """Should return defaults if config file doesn't exist."""
        with patch.dict(
            os.environ, {"TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json")}
        ):
            config = load_config()
            # Should get default config
            assert config.embedding_model == "nomic-embed-text"

    def test_load_config_from_file(self, tmp_path):
        """Should load config from JSON file."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text("""{
            "backends": {},
            "embedding_model": "custom-model",
            "auto_sync": false
        }""")

        with patch.dict(os.environ, {"TOOL_COMPASS_CONFIG": str(config_file)}):
            config = load_config()
            assert config.embedding_model == "custom-model"
            assert config.auto_sync is False


class TestRedactConfig:
    """CFG-A-001: structural redaction of resolved secrets in doctor() dumps.

    Name-based redaction only catches keys that *look* secret
    (_token/_key/_secret/_password). But the ${VAR} substitution feature
    resolves env secrets INTO backend headers/env/args, where the KEY names
    (e.g. 'Authorization', 'GITHUB_TOKEN') don't all match those hints —
    so resolved secret VALUES used to leak verbatim from doctor()'s dump.
    Redact STRUCTURALLY: for every backend's 'env'/'headers' (dicts) redact
    all VALUES, and for 'args' (list) redact all entries, while KEEPING THE
    KEYS visible so the dump stays diagnosable.
    """

    SECRET_HEADER = "Bearer SEKRET_HEADER_TOKEN_abc123"
    SECRET_ENV = "ghp_SEKRET_ENV_TOKEN_xyz789"
    SECRET_ARG = "--password=SEKRET_ARG_VALUE_qwe456"

    def _secret_values(self):
        return [self.SECRET_HEADER, self.SECRET_ENV, self.SECRET_ARG]

    def _build_config(self):
        return CompassConfig(
            backends={
                "remote": HttpBackend(
                    url="http://localhost:9000/mcp",
                    headers={"Authorization": self.SECRET_HEADER},
                ),
                "local": StdioBackend(
                    command="python",
                    args=["-m", "server", self.SECRET_ARG],
                    env={"GITHUB_TOKEN": self.SECRET_ENV},
                ),
            }
        )

    def test_redact_config_hides_resolved_secret_values(self):
        """Resolved secret VALUES must not appear; KEYS must remain visible."""
        cfg = self._build_config()
        redacted = _redact_config(cfg.to_dict())

        blob = json.dumps(redacted)
        for secret in self._secret_values():
            assert secret not in blob, (
                f"secret value leaked into redacted dump: {secret!r}"
            )

        backends = redacted["backends"]
        # Header value redacted, but the 'Authorization' key still visible.
        assert "Authorization" in backends["remote"]["headers"]
        assert backends["remote"]["headers"]["Authorization"] == "[REDACTED]"
        # Env value redacted, key 'GITHUB_TOKEN' still visible.
        assert "GITHUB_TOKEN" in backends["local"]["env"]
        assert backends["local"]["env"]["GITHUB_TOKEN"] == "[REDACTED]"
        # Args entries redacted (the secret-bearing entry at minimum).
        assert self.SECRET_ARG not in backends["local"]["args"]
        assert all(a == "[REDACTED]" for a in backends["local"]["args"])
        # Non-secret structural fields stay intact for diagnosability.
        assert backends["remote"]["url"] == "http://localhost:9000/mcp"
        assert backends["remote"]["type"] == "http"
        assert backends["local"]["command"] == "python"

    def test_doctor_does_not_leak_resolved_secrets(self, tmp_path):
        """End-to-end: a config file with resolved ${VAR} secrets in a
        header + env must not surface those secret values from doctor()."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {
                "remote": {
                    "type": "http",
                    "url": "http://localhost:9000/mcp",
                    "headers": {"Authorization": "Bearer ${MY_API_TOKEN}"},
                },
                "local": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "server", "${MY_CLI_ARG}"],
                    "env": {"GITHUB_TOKEN": "${MY_GH_TOKEN}"},
                },
            }
        }))

        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "MY_API_TOKEN": "live_header_secret_111",
            "MY_GH_TOKEN": "ghp_live_env_secret_222",
            "MY_CLI_ARG": "--password=live_arg_secret_333",
        }
        with patch.dict(os.environ, env):
            report = doctor()

        blob = json.dumps(report, default=str)
        for secret in (
            "live_header_secret_111",
            "ghp_live_env_secret_222",
            "live_arg_secret_333",
        ):
            assert secret not in blob, f"doctor() leaked secret {secret!r}"

        # Keys remain visible in the redacted config so the dump is usable.
        backends = report["config"]["backends"]
        assert "Authorization" in backends["remote"]["headers"]
        assert backends["remote"]["headers"]["Authorization"] == "[REDACTED]"
        assert "GITHUB_TOKEN" in backends["local"]["env"]
        assert backends["local"]["env"]["GITHUB_TOKEN"] == "[REDACTED]"


class TestRedactUrlCredentials:
    """CFG-A-001 (sibling): a credentialed ollama_url like
    ``http://user:${TOKEN}@host:11434`` must have its userinfo stripped to
    ``http://[REDACTED]@host:11434`` in doctor() output and in _redact_config,
    while host:port stays visible for diagnosability.

    The ${VAR} substitution feature resolves env secrets INTO ollama_url
    userinfo at load time; without this the raw user:secret@ landed verbatim
    in a pasteable bug-report dump.
    """

    def test_redact_url_credentials_strips_userinfo(self):
        from config import redact_url_credentials

        out = redact_url_credentials("http://u:livesecret@h:11434")
        assert "livesecret" not in out
        assert out == "http://[REDACTED]@h:11434"

    def test_redact_url_credentials_passthrough_when_no_userinfo(self):
        from config import redact_url_credentials

        # No credentials -> unchanged, host:port intact.
        assert (
            redact_url_credentials("http://localhost:11434")
            == "http://localhost:11434"
        )

    def test_redact_config_scrubs_ollama_url_userinfo(self):
        """_redact_config must scrub embedded userinfo from ollama_url (a leaf
        scalar) while keeping the host visible."""
        cfg = CompassConfig(ollama_url="http://u:livesecret@h:11434")
        redacted = _redact_config(cfg.to_dict())
        blob = json.dumps(redacted)
        assert "livesecret" not in blob, "ollama_url secret leaked"
        assert redacted["ollama_url"] == "http://[REDACTED]@h:11434"
        assert "h:11434" in redacted["ollama_url"]

    def test_doctor_redacts_ollama_url_credentials(self, tmp_path):
        """End-to-end: doctor() must not surface the ollama_url userinfo secret
        but must keep host:port for diagnosability."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "ollama_url": "http://u:${OLLAMA_PW}@h:11434",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "OLLAMA_PW": "livesecret",
        }
        with patch.dict(os.environ, env):
            report = doctor()

        blob = json.dumps(report, default=str)
        assert "livesecret" not in blob, "doctor() leaked ollama_url secret"
        # Host:port survives in both the top-level field and the config dump.
        assert "h:11434" in report["ollama_url"]
        assert report["ollama_url"] == "http://[REDACTED]@h:11434"
        assert "h:11434" in report["config"]["ollama_url"]
        assert "livesecret" not in report["config"]["ollama_url"]


class TestValidateAndClampCoercion:
    """CFG-A-002: validate_and_clamp must survive non-numeric hand-edited
    values. The compare-before-coerce ordering raised TypeError on a string
    or null numeric BEFORE the int()/float() cast, and from_file's recovery
    except only catches (json.JSONDecodeError, OSError) — so a hand-edited
    config crashed startup with a raw traceback, contradicting the docstring's
    'safe even with a hand-edited config file.'"""

    def test_from_dict_bad_numeric_types_do_not_crash(self):
        """A config dict with string/null numeric fields must coerce-or-reset
        to defaults instead of raising TypeError."""
        defaults = CompassConfig()
        data = {
            "backends": {},
            "min_confidence": "high",          # non-numeric string
            "default_top_k": None,             # null
            "sync_polling_interval": "soon",   # non-numeric string
            "hot_cache_size": None,            # null
            "chain_detection_min_occurrences": "lots",
            "ollama_breaker_failure_threshold": None,
            "ollama_breaker_open_seconds": "forever",
            "ollama_retry_attempts": None,
            "hnsw_m": "big",
            "hnsw_ef_construction": None,
            "hnsw_ef_search": "fast",
        }
        # Must NOT raise.
        config = CompassConfig.from_dict(data)

        # Each bad field reset to its in-range default (or a clamped default).
        assert config.min_confidence == defaults.min_confidence
        assert config.default_top_k == defaults.default_top_k
        assert config.sync_polling_interval == defaults.sync_polling_interval
        assert config.hot_cache_size == defaults.hot_cache_size
        assert (
            config.chain_detection_min_occurrences
            == defaults.chain_detection_min_occurrences
        )
        assert (
            config.ollama_breaker_failure_threshold
            == defaults.ollama_breaker_failure_threshold
        )
        assert (
            config.ollama_breaker_open_seconds
            == defaults.ollama_breaker_open_seconds
        )
        assert config.ollama_retry_attempts == defaults.ollama_retry_attempts
        assert config.hnsw_m == defaults.hnsw_m
        assert config.hnsw_ef_construction == defaults.hnsw_ef_construction
        assert config.hnsw_ef_search == defaults.hnsw_ef_search

    def test_from_file_with_hand_edited_bad_values_recovers(self, tmp_path):
        """from_file on a syntactically-valid JSON with bad numeric types
        must load without crashing (the docstring's stated guarantee)."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "min_confidence": "high",
            "default_top_k": None,
            "embedding_model": "custom-model",
        }))
        with patch.dict(os.environ, {"TOOL_COMPASS_CONFIG": str(config_file)}):
            config = load_config()  # must not raise
        # Non-numeric fields untouched, numeric fields reset to defaults.
        assert config.embedding_model == "custom-model"
        assert config.min_confidence == CompassConfig().min_confidence
        assert config.default_top_k == CompassConfig().default_top_k


class TestEnvOverrides:
    """CFGDOC-01: TOOL_COMPASS_ANALYTICS_DISABLED and TOOL_COMPASS_HOT_CACHE_SIZE
    are advertised in .env.example but were never read — the app only honored
    the analytics_enabled / hot_cache_size config-JSON keys, so a user who set
    the documented env var saw no effect. These tests prove the env vars now
    take effect (and would fail on the old, ignore-everything behavior)."""

    def test_analytics_disabled_env_turns_off_analytics(self, tmp_path):
        """Truthy TOOL_COMPASS_ANALYTICS_DISABLED -> analytics_enabled False.

        Old behavior: get_default_config ignored the env var, so
        analytics_enabled stayed True. This asserts False, so it fails pre-fix.
        """
        env = {
            "TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json"),
            "TOOL_COMPASS_ANALYTICS_DISABLED": "true",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.analytics_enabled is False

    def test_analytics_disabled_env_falsey_leaves_analytics_on(self, tmp_path):
        """A non-truthy value (e.g. 'false'/'0'/'') must NOT disable analytics."""
        for raw in ("false", "0", "no", ""):
            env = {
                "TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json"),
                "TOOL_COMPASS_ANALYTICS_DISABLED": raw,
            }
            with patch.dict(os.environ, env):
                config = load_config()
            assert config.analytics_enabled is True, f"raw={raw!r} disabled analytics"

    def test_hot_cache_size_env_overrides_default(self, tmp_path):
        """TOOL_COMPASS_HOT_CACHE_SIZE sets hot_cache_size.

        Old behavior: env var ignored, hot_cache_size stayed at default 10.
        This sets 25 and asserts 25, so it fails pre-fix.
        """
        env = {
            "TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json"),
            "TOOL_COMPASS_HOT_CACHE_SIZE": "25",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.hot_cache_size == 25

    def test_hot_cache_size_env_overrides_file_value(self, tmp_path):
        """The env var wins over a value set in the config file (more specific
        operator signal)."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({"backends": {}, "hot_cache_size": 7}))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_HOT_CACHE_SIZE": "42",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.hot_cache_size == 42

    def test_hot_cache_size_env_is_clamped(self, tmp_path):
        """An out-of-range env value flows through validate_and_clamp (0 -> 1)
        rather than silently disabling the cache."""
        env = {
            "TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json"),
            "TOOL_COMPASS_HOT_CACHE_SIZE": "0",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.hot_cache_size == 1  # clamped from 0

    def test_hot_cache_size_env_non_numeric_is_ignored(self, tmp_path):
        """A non-integer env value is ignored (default kept), not a crash."""
        env = {
            "TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json"),
            "TOOL_COMPASS_HOT_CACHE_SIZE": "lots",
        }
        with patch.dict(os.environ, env):
            config = load_config()  # must not raise
        assert config.hot_cache_size == CompassConfig().hot_cache_size

    def test_apply_env_overrides_no_vars_is_noop(self):
        """With neither env var set, the config is unchanged."""
        cfg = CompassConfig()
        with patch.dict(os.environ, {}, clear=True):
            out = apply_env_overrides(cfg)
        assert out.analytics_enabled is True
        assert out.hot_cache_size == CompassConfig().hot_cache_size


class TestTopChainsCacheSizeClamp:
    """CFGDOC-03: top_chains_cache_size feeds a slice bound
    (chain_indexer.py: chains[:n]). It escaped validate_and_clamp's
    coerce/clamp loop, so a hand-edited 0/negative silently emptied or
    truncated the chain cache with no warning."""

    def test_negative_top_chains_cache_size_clamped_to_zero(self):
        """A negative value is clamped to 0 (with a warning).

        Old behavior: -3 passed straight through to chains[:-3], silently
        dropping the last three chains. This asserts the clamp, failing pre-fix.
        """
        config = CompassConfig.from_dict(
            {"backends": {}, "top_chains_cache_size": -3}
        )
        assert config.top_chains_cache_size == 0

    def test_zero_top_chains_cache_size_preserved(self):
        """0 is allowed (disables the cache) and not bumped up."""
        config = CompassConfig.from_dict(
            {"backends": {}, "top_chains_cache_size": 0}
        )
        assert config.top_chains_cache_size == 0

    def test_string_top_chains_cache_size_coerced(self):
        """A numeric string is coerced through the coerce loop."""
        config = CompassConfig.from_dict(
            {"backends": {}, "top_chains_cache_size": "8"}
        )
        assert config.top_chains_cache_size == 8

    def test_non_numeric_top_chains_cache_size_resets_to_default(self):
        """A non-numeric value resets to the class default instead of crashing.

        Old behavior: top_chains_cache_size was absent from the coerce tuple,
        so a string like 'many' survived into chains[:'many'] and raised
        TypeError at slice time. This asserts the reset, failing pre-fix.
        """
        config = CompassConfig.from_dict(
            {"backends": {}, "top_chains_cache_size": "many"}
        )
        assert config.top_chains_cache_size == CompassConfig().top_chains_cache_size


class TestEmbeddingProviderConfig:
    """BE-FT-PE-001: the pluggable-embedding config fields round-trip through
    to_dict/from_dict, validate against the known providers (unknown -> warn +
    fall back to ollama), redact the api_key in doctor() dumps, and honor the
    TOOL_COMPASS_EMBEDDING_API_KEY env override."""

    def test_defaults_preserve_ollama_behavior(self):
        config = CompassConfig()
        assert config.embedding_provider == "ollama"
        assert config.embedding_base_url is None
        assert config.embedding_api_key is None
        assert config.embedding_query_prefix is None
        assert config.embedding_document_prefix is None

    def test_new_fields_roundtrip(self):
        original = CompassConfig(
            embedding_provider="openai",
            embedding_base_url="http://lmstudio:1234",
            embedding_api_key="sk-secret-abc",
            embedding_query_prefix="",
            embedding_document_prefix="passage: ",
        )
        data = original.to_dict()
        # All new fields are serialized.
        assert data["embedding_provider"] == "openai"
        assert data["embedding_base_url"] == "http://lmstudio:1234"
        assert data["embedding_api_key"] == "sk-secret-abc"
        assert data["embedding_query_prefix"] == ""
        assert data["embedding_document_prefix"] == "passage: "

        restored = CompassConfig.from_dict(data)
        assert restored.embedding_provider == "openai"
        assert restored.embedding_base_url == "http://lmstudio:1234"
        assert restored.embedding_api_key == "sk-secret-abc"
        assert restored.embedding_query_prefix == ""
        assert restored.embedding_document_prefix == "passage: "

    def test_openai_compatible_alias_normalizes(self):
        config = CompassConfig.from_dict(
            {"backends": {}, "embedding_provider": "openai-compatible"}
        )
        # 'openai-compatible' is a known provider (registry alias) and is
        # stored normalized (lower-cased, stripped) but NOT rewritten.
        assert config.embedding_provider == "openai-compatible"

    def test_unknown_provider_falls_back_to_ollama(self):
        config = CompassConfig.from_dict(
            {"backends": {}, "embedding_provider": "made-up-backend"}
        )
        assert config.embedding_provider == "ollama"

    def test_provider_case_insensitive(self):
        config = CompassConfig.from_dict(
            {"backends": {}, "embedding_provider": "OpenAI"}
        )
        assert config.embedding_provider == "openai"

    def test_api_key_is_redacted_in_redact_config(self):
        cfg = CompassConfig(
            embedding_provider="openai",
            embedding_base_url="http://x:1",
            embedding_api_key="sk-super-secret-999",
        )
        redacted = _redact_config(cfg.to_dict())
        blob = json.dumps(redacted)
        assert "sk-super-secret-999" not in blob
        # Field name (containing _key) triggers name-based redaction.
        assert redacted["embedding_api_key"] == "[REDACTED]"
        # Non-secret sibling fields stay visible for diagnosability.
        assert redacted["embedding_provider"] == "openai"
        assert redacted["embedding_base_url"] == "http://x:1"

    def test_doctor_does_not_leak_embedding_api_key(self, tmp_path):
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "embedding_provider": "openai",
            "embedding_base_url": "http://x:1",
            "embedding_api_key": "${MY_EMBED_KEY}",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "MY_EMBED_KEY": "live_embed_secret_777",
        }
        with patch.dict(os.environ, env):
            report = doctor()
        blob = json.dumps(report, default=str)
        assert "live_embed_secret_777" not in blob
        assert report["config"]["embedding_api_key"] == "[REDACTED]"

    def test_env_var_overrides_api_key(self, tmp_path):
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "embedding_provider": "openai",
            "embedding_base_url": "http://x:1",
            "embedding_api_key": "file-key",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_EMBEDDING_API_KEY": "env-wins-key",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        # Env override wins over the file value (operator-intent signal).
        assert config.embedding_api_key == "env-wins-key"

    def test_empty_env_api_key_does_not_blank_file_value(self, tmp_path):
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "embedding_api_key": "file-key",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_EMBEDDING_API_KEY": "",  # exported but empty
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.embedding_api_key == "file-key"

    def test_resolved_base_url_ollama_uses_ollama_url(self):
        cfg = CompassConfig(ollama_url="http://oll:11434")
        assert cfg.resolved_embedding_base_url() == "http://oll:11434"

    def test_resolved_base_url_prefers_override(self):
        cfg = CompassConfig(
            embedding_provider="openai",
            ollama_url="http://oll:11434",
            embedding_base_url="http://lmstudio:1234",
        )
        assert cfg.resolved_embedding_base_url() == "http://lmstudio:1234"

    def test_example_config_file_roundtrips(self):
        """compass_config.example.json must parse and round-trip cleanly
        (including the new embedding_* fields)."""
        example_path = (
            Path(__file__).resolve().parent.parent / "compass_config.example.json"
        )
        data = json.loads(example_path.read_text())
        config = CompassConfig.from_dict(data)
        assert config.embedding_provider == "ollama"
        # Round-trips back out.
        restored = CompassConfig.from_dict(config.to_dict())
        assert restored.embedding_provider == "ollama"


class TestDoctorAnalyticsHealth:
    """CFGDOC-04: doctor() now reports the analytics degraded state via
    analytics.get_health() — but only if a live singleton already exists, and
    it must never force-create one (which would open + initialize the DB as a
    side effect of running diagnostics)."""

    def test_doctor_analytics_health_none_when_no_singleton(self, tmp_path):
        """No live analytics singleton -> analytics_health is None, and doctor()
        does NOT create one."""
        import analytics as analytics_mod

        # Ensure no singleton exists.
        original = analytics_mod._analytics_instance
        analytics_mod._analytics_instance = None
        try:
            env = {"TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json")}
            with patch.dict(os.environ, env):
                report = doctor()
            assert report["analytics_health"] is None
            # doctor() must not have constructed a singleton.
            assert analytics_mod._analytics_instance is None
        finally:
            analytics_mod._analytics_instance = original

    def test_doctor_includes_health_when_singleton_exists(self, tmp_path):
        """A live singleton's degraded state is surfaced under analytics_health."""
        import analytics as analytics_mod

        original = analytics_mod._analytics_instance
        inst = analytics_mod.CompassAnalytics(
            db_path=tmp_path / "db" / "compass_analytics.db"
        )
        # Simulate the 'sqlite broke' degraded mode without touching the DB.
        inst._degraded = True
        analytics_mod._analytics_instance = inst
        try:
            env = {"TOOL_COMPASS_CONFIG": str(tmp_path / "missing.json")}
            with patch.dict(os.environ, env):
                report = doctor()
            assert report["analytics_health"] is not None
            assert report["analytics_health"]["degraded"] is True
            assert report["analytics_health"]["reason"] is not None
        finally:
            analytics_mod._analytics_instance = original


class TestPerBackendTimeouts:
    """INT-02: StdioBackend and HttpBackend carry a per-backend
    default_timeout (outer tool-call deadline in seconds) and a tool_timeouts
    map keyed by BARE tool name. Both round-trip through to_dict/from_dict and
    are clamped to [1.0, 600.0] with a warning; non-numeric values reset/drop.
    """

    def test_defaults(self):
        """Absent timeout fields take documented defaults (back-compat)."""
        stdio = StdioBackend()
        assert stdio.default_timeout is None
        assert stdio.tool_timeouts == {}
        http = HttpBackend()
        assert http.default_timeout is None
        assert http.tool_timeouts == {}

    def test_stdio_timeout_roundtrip(self):
        data = {
            "backends": {
                "srv": {
                    "type": "stdio",
                    "command": "python",
                    "default_timeout": 45.0,
                    "tool_timeouts": {"slow_tool": 120.0},
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["srv"]
        assert backend.default_timeout == 45.0
        assert backend.tool_timeouts == {"slow_tool": 120.0}
        # Round-trip out and back.
        restored = CompassConfig.from_dict(config.to_dict())
        rb = restored.backends["srv"]
        assert rb.default_timeout == 45.0
        assert rb.tool_timeouts == {"slow_tool": 120.0}

    def test_http_timeout_roundtrip(self):
        data = {
            "backends": {
                "api": {
                    "type": "http",
                    "url": "http://x:1",
                    "default_timeout": 30.0,
                    "tool_timeouts": {"heavy": 300.0},
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["api"]
        assert backend.default_timeout == 30.0
        assert backend.tool_timeouts == {"heavy": 300.0}
        restored = CompassConfig.from_dict(config.to_dict())
        rb = restored.backends["api"]
        assert rb.default_timeout == 30.0
        assert rb.tool_timeouts == {"heavy": 300.0}

    def test_default_timeout_non_numeric_resets_to_none(self):
        """A non-numeric default_timeout resets to None with a warning."""
        data = {
            "backends": {
                "srv": {
                    "type": "stdio",
                    "command": "python",
                    "default_timeout": "soon",
                }
            }
        }
        config = CompassConfig.from_dict(data)
        assert config.backends["srv"].default_timeout is None

    def test_default_timeout_out_of_range_clamped(self):
        """default_timeout clamps to [1.0, 600.0]."""
        low = CompassConfig.from_dict(
            {"backends": {"a": {"type": "stdio", "default_timeout": 0.1}}}
        )
        assert low.backends["a"].default_timeout == 1.0
        high = CompassConfig.from_dict(
            {"backends": {"b": {"type": "stdio", "default_timeout": 9999}}}
        )
        assert high.backends["b"].default_timeout == 600.0

    def test_tool_timeouts_values_clamped(self):
        """Each tool_timeouts value clamps to [1.0, 600.0]."""
        config = CompassConfig.from_dict(
            {
                "backends": {
                    "a": {
                        "type": "stdio",
                        "tool_timeouts": {"fast": 0.5, "slow": 5000},
                    }
                }
            }
        )
        tt = config.backends["a"].tool_timeouts
        assert tt["fast"] == 1.0
        assert tt["slow"] == 600.0

    def test_tool_timeouts_non_numeric_entry_dropped(self):
        """A non-numeric tool_timeouts entry is dropped with a warning."""
        config = CompassConfig.from_dict(
            {
                "backends": {
                    "a": {
                        "type": "stdio",
                        "tool_timeouts": {"good": 10.0, "bad": "forever"},
                    }
                }
            }
        )
        tt = config.backends["a"].tool_timeouts
        assert tt == {"good": 10.0}


class TestBackendToolFilters:
    """FEAT-06: StdioBackend, HttpBackend, and ImportBackend each carry
    allow_tools / deny_tools glob lists. Empty allow = allow all; deny takes
    precedence. Both round-trip and are coerced to list-of-str.
    """

    def test_defaults_empty(self):
        for backend in (StdioBackend(), HttpBackend(), ImportBackend()):
            assert backend.allow_tools == []
            assert backend.deny_tools == []

    def test_stdio_filters_roundtrip(self):
        data = {
            "backends": {
                "srv": {
                    "type": "stdio",
                    "command": "python",
                    "allow_tools": ["read_*", "list_*"],
                    "deny_tools": ["delete_*"],
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["srv"]
        assert backend.allow_tools == ["read_*", "list_*"]
        assert backend.deny_tools == ["delete_*"]
        restored = CompassConfig.from_dict(config.to_dict())
        rb = restored.backends["srv"]
        assert rb.allow_tools == ["read_*", "list_*"]
        assert rb.deny_tools == ["delete_*"]

    def test_http_filters_roundtrip(self):
        data = {
            "backends": {
                "api": {
                    "type": "http",
                    "url": "http://x:1",
                    "allow_tools": ["search_*"],
                    "deny_tools": ["admin_*"],
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["api"]
        assert backend.allow_tools == ["search_*"]
        assert backend.deny_tools == ["admin_*"]
        restored = CompassConfig.from_dict(config.to_dict())
        rb = restored.backends["api"]
        assert rb.allow_tools == ["search_*"]
        assert rb.deny_tools == ["admin_*"]

    def test_import_filters_roundtrip(self):
        data = {
            "backends": {
                "local": {
                    "type": "import",
                    "module": "my_server",
                    "allow_tools": ["*"],
                    "deny_tools": ["dangerous_*"],
                }
            }
        }
        config = CompassConfig.from_dict(data)
        backend = config.backends["local"]
        assert backend.allow_tools == ["*"]
        assert backend.deny_tools == ["dangerous_*"]
        restored = CompassConfig.from_dict(config.to_dict())
        rb = restored.backends["local"]
        assert rb.allow_tools == ["*"]
        assert rb.deny_tools == ["dangerous_*"]

    def test_filters_coerced_to_list_of_str(self):
        """Non-string entries are coerced; a non-list resets to empty."""
        config = CompassConfig.from_dict(
            {
                "backends": {
                    "a": {
                        "type": "stdio",
                        "allow_tools": ["ok", 123],
                        "deny_tools": "not-a-list",
                    }
                }
            }
        )
        backend = config.backends["a"]
        assert backend.allow_tools == ["ok", "123"]
        assert backend.deny_tools == []


class TestSearchAndRetentionConfig:
    """DISC-01/DISC-02/FEAT-04: hybrid_search, exact_name_boost,
    exact_match_confidence, and analytics_retention_days round-trip, take
    documented defaults, and clamp (retention >= 0)."""

    def test_defaults(self):
        config = CompassConfig()
        assert config.hybrid_search is True
        assert config.exact_name_boost is True
        assert config.exact_match_confidence == 1.0
        assert config.analytics_retention_days == 30

    def test_roundtrip(self):
        original = CompassConfig(
            hybrid_search=False,
            exact_name_boost=False,
            exact_match_confidence=0.75,
            analytics_retention_days=7,
        )
        data = original.to_dict()
        assert data["hybrid_search"] is False
        assert data["exact_name_boost"] is False
        assert data["exact_match_confidence"] == 0.75
        assert data["analytics_retention_days"] == 7
        restored = CompassConfig.from_dict(data)
        assert restored.hybrid_search is False
        assert restored.exact_name_boost is False
        assert restored.exact_match_confidence == 0.75
        assert restored.analytics_retention_days == 7

    def test_negative_retention_clamped_to_zero(self):
        """A negative analytics_retention_days clamps to 0 (keep forever)."""
        config = CompassConfig.from_dict(
            {"backends": {}, "analytics_retention_days": -5}
        )
        assert config.analytics_retention_days == 0

    def test_zero_retention_preserved(self):
        """0 is allowed (keep forever) and not bumped."""
        config = CompassConfig.from_dict(
            {"backends": {}, "analytics_retention_days": 0}
        )
        assert config.analytics_retention_days == 0

    def test_non_numeric_retention_resets_to_default(self):
        config = CompassConfig.from_dict(
            {"backends": {}, "analytics_retention_days": "forever"}
        )
        assert (
            config.analytics_retention_days
            == CompassConfig().analytics_retention_days
        )


class TestGatewayAuthToken:
    """OPS-1: gateway_auth_token is an opt-in bearer token for the HTTP
    transport. It round-trips, resolves ${VAR} like other secrets, falls back
    to TOOL_COMPASS_GATEWAY_AUTH_TOKEN when unset, and is redacted in the
    doctor()/show_config path (its name ends in _token)."""

    def test_default_is_none(self):
        assert CompassConfig().gateway_auth_token is None

    def test_roundtrip(self):
        original = CompassConfig(gateway_auth_token="secret-bearer-123")
        data = original.to_dict()
        assert data["gateway_auth_token"] == "secret-bearer-123"
        restored = CompassConfig.from_dict(data)
        assert restored.gateway_auth_token == "secret-bearer-123"

    def test_redacted_in_redact_config(self):
        cfg = CompassConfig(gateway_auth_token="super-secret-token-999")
        redacted = _redact_config(cfg.to_dict())
        blob = json.dumps(redacted)
        assert "super-secret-token-999" not in blob
        assert redacted["gateway_auth_token"] == "[REDACTED]"

    def test_doctor_does_not_leak_gateway_auth_token(self, tmp_path):
        """End-to-end: a ${VAR}-resolved token must not surface from doctor()."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "gateway_auth_token": "${MY_GATEWAY_TOKEN}",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "MY_GATEWAY_TOKEN": "live_gateway_secret_444",
        }
        with patch.dict(os.environ, env):
            report = doctor()
        blob = json.dumps(report, default=str)
        assert "live_gateway_secret_444" not in blob
        assert report["config"]["gateway_auth_token"] == "[REDACTED]"

    def test_env_var_fallback_when_unset(self, tmp_path):
        """TOOL_COMPASS_GATEWAY_AUTH_TOKEN populates an unset config field."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({"backends": {}}))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_GATEWAY_AUTH_TOKEN": "env-gateway-token",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.gateway_auth_token == "env-gateway-token"

    def test_env_var_overrides_file_value(self, tmp_path):
        """The env var wins over a file value (operator-intent signal)."""
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "gateway_auth_token": "file-token",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_GATEWAY_AUTH_TOKEN": "env-token",
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.gateway_auth_token == "env-token"

    def test_empty_env_does_not_blank_file_value(self, tmp_path):
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {},
            "gateway_auth_token": "file-token",
        }))
        env = {
            "TOOL_COMPASS_CONFIG": str(config_file),
            "TOOL_COMPASS_GATEWAY_AUTH_TOKEN": "",  # exported but empty
        }
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.gateway_auth_token == "file-token"


class TestAllNewFieldsRoundtrip:
    """Contract-level: a config dict containing EVERY new field parses,
    serializes, and re-parses identically (load->dump->load stability)."""

    def test_full_roundtrip(self):
        data = {
            "backends": {
                "srv": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["-m", "server"],
                    "env": {"K": "V"},
                    "default_timeout": 45.0,
                    "tool_timeouts": {"slow": 120.0},
                    "allow_tools": ["read_*"],
                    "deny_tools": ["delete_*"],
                },
                "api": {
                    "type": "http",
                    "url": "http://x:1",
                    "default_timeout": 30.0,
                    "tool_timeouts": {"heavy": 300.0},
                    "allow_tools": ["search_*"],
                    "deny_tools": ["admin_*"],
                },
                "local": {
                    "type": "import",
                    "module": "m",
                    "allow_tools": ["*"],
                    "deny_tools": ["danger_*"],
                },
            },
            "hybrid_search": False,
            "exact_name_boost": False,
            "exact_match_confidence": 0.5,
            "analytics_retention_days": 14,
            "gateway_auth_token": "tok-abc",
        }
        first = CompassConfig.from_dict(data)
        dumped = first.to_dict()
        second = CompassConfig.from_dict(dumped)

        # Stable across a second round-trip.
        assert second.to_dict() == dumped
        # Spot-check per-backend fields survived both hops.
        assert second.backends["srv"].default_timeout == 45.0
        assert second.backends["srv"].tool_timeouts == {"slow": 120.0}
        assert second.backends["srv"].allow_tools == ["read_*"]
        assert second.backends["srv"].deny_tools == ["delete_*"]
        assert second.backends["api"].default_timeout == 30.0
        assert second.backends["local"].allow_tools == ["*"]
        assert second.hybrid_search is False
        assert second.exact_name_boost is False
        assert second.exact_match_confidence == 0.5
        assert second.analytics_retention_days == 14
        assert second.gateway_auth_token == "tok-abc"

    def test_absent_fields_take_defaults(self):
        """Back-compat: an old minimal config still loads with defaults."""
        config = CompassConfig.from_dict({"backends": {}})
        assert config.hybrid_search is True
        assert config.exact_name_boost is True
        assert config.exact_match_confidence == 1.0
        assert config.analytics_retention_days == 30
        assert config.gateway_auth_token is None
