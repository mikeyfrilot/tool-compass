"""
Tests for Tool Compass configuration module.

Tests cross-platform path handling and environment variable support.
"""

import os
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
        """Default config should have expected backends."""
        config = get_default_config()

        # Should have 5 backends
        expected_backends = ["bridge", "comfy", "video", "chat", "doc"]
        for name in expected_backends:
            assert name in config.backends
            assert isinstance(config.backends[name], StdioBackend)

    def test_get_default_config_uses_detected_python(self):
        """Default config should use detected Python executable."""
        config = get_default_config()

        for backend in config.backends.values():
            if isinstance(backend, StdioBackend):
                # Command should be a valid Python path
                assert backend.command
                # Should be an absolute path or the detected executable
                assert (
                    Path(backend.command).is_absolute()
                    or backend.command == sys.executable
                )

    def test_get_default_config_portable_paths(self):
        """Backend paths should be relative to base_path."""
        config = get_default_config()
        base = get_base_path()

        for backend in config.backends.values():
            if isinstance(backend, StdioBackend):
                for arg in backend.args:
                    if arg.endswith(".py"):
                        # Python file paths should be under base_path
                        assert str(base) in arg or not Path(arg).is_absolute()


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
