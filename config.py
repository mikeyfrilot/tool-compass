"""
Tool Compass - Configuration Schema
Defines how backends are configured and connected.

Environment Variables:
    TOOL_COMPASS_BASE_PATH: Base path for the project (default: parent of tool_compass)
    TOOL_COMPASS_PYTHON: Path to Python executable (default: auto-detect from venv)
    TOOL_COMPASS_CONFIG: Path to config file (default: ./compass_config.json)
    OLLAMA_URL: Ollama server URL (default: http://localhost:11434)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from pathlib import Path
import json
import os
import sys
import re


@dataclass
class StdioBackend:
    """Backend that spawns an MCP server as subprocess."""

    type: Literal["stdio"] = "stdio"
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


@dataclass
class HttpBackend:
    """Backend that connects to an MCP server over HTTP/SSE."""

    type: Literal["http"] = "http"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class ImportBackend:
    """Backend that imports an MCP server module directly (same process)."""

    type: Literal["import"] = "import"
    module: str = ""
    server_var: str = "mcp"  # Variable name of the FastMCP instance


BackendConfig = StdioBackend | HttpBackend | ImportBackend


@dataclass
class CompassConfig:
    """Full Tool Compass configuration."""

    # Backend server connections
    backends: Dict[str, BackendConfig] = field(default_factory=dict)

    # Embedding settings
    embedding_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"

    # Index settings
    index_dir: str = "./db"
    auto_sync: bool = True  # Auto-discover tools from backends on startup

    # Search settings
    default_top_k: int = 5
    min_confidence: float = 0.3

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

    @classmethod
    def from_file(cls, path: Path) -> "CompassConfig":
        """Load config from JSON file with variable substitution."""
        with open(path) as f:
            data = json.load(f)

        # Get defaults for variable substitution
        defaults = data.get("defaults", {})

        # Also check environment variables (env vars take precedence)
        def resolve_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, defaults.get(var_name, match.group(0)))

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
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CompassConfig":
        """Create config from dictionary."""
        config = cls()

        # Parse backends
        for name, backend_data in data.get("backends", {}).items():
            backend_type = backend_data.get("type", "stdio")
            if backend_type == "stdio":
                config.backends[name] = StdioBackend(
                    command=backend_data.get("command", ""),
                    args=backend_data.get("args", []),
                    env=backend_data.get("env", {}),
                    cwd=backend_data.get("cwd"),
                )
            elif backend_type == "http":
                config.backends[name] = HttpBackend(
                    url=backend_data.get("url", ""),
                    headers=backend_data.get("headers", {}),
                    timeout=backend_data.get("timeout", 30.0),
                )
            elif backend_type == "import":
                config.backends[name] = ImportBackend(
                    module=backend_data.get("module", ""),
                    server_var=backend_data.get("server_var", "mcp"),
                )

        # Other settings
        config.embedding_model = data.get("embedding_model", config.embedding_model)
        config.ollama_url = data.get("ollama_url", config.ollama_url)
        config.index_dir = data.get("index_dir", config.index_dir)
        config.auto_sync = data.get("auto_sync", config.auto_sync)
        config.default_top_k = data.get("default_top_k", config.default_top_k)
        config.min_confidence = data.get("min_confidence", config.min_confidence)
        config.progressive_disclosure = data.get(
            "progressive_disclosure", config.progressive_disclosure
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

        return config

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
                }
            elif isinstance(backend, HttpBackend):
                backends[name] = {
                    "type": "http",
                    "url": backend.url,
                    "headers": backend.headers,
                    "timeout": backend.timeout,
                }
            elif isinstance(backend, ImportBackend):
                backends[name] = {
                    "type": "import",
                    "module": backend.module,
                    "server_var": backend.server_var,
                }

        return {
            "backends": backends,
            "embedding_model": self.embedding_model,
            "ollama_url": self.ollama_url,
            "index_dir": self.index_dir,
            "auto_sync": self.auto_sync,
            "default_top_k": self.default_top_k,
            "min_confidence": self.min_confidence,
            "progressive_disclosure": self.progressive_disclosure,
            "sync_check_on_startup": self.sync_check_on_startup,
            "sync_polling_interval": self.sync_polling_interval,
            "analytics_enabled": self.analytics_enabled,
            "hot_cache_size": self.hot_cache_size,
            "chain_indexing_enabled": self.chain_indexing_enabled,
            "chain_detection_min_occurrences": self.chain_detection_min_occurrences,
            "top_chains_cache_size": self.top_chains_cache_size,
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


def get_default_config() -> CompassConfig:
    """
    Get default config for the MCP tool shop setup.

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


def get_config_path() -> Path:
    """
    Get the config file path.

    Resolution order:
    1. TOOL_COMPASS_CONFIG environment variable
    2. ./compass_config.json in tool_compass directory
    """
    env_config = os.environ.get("TOOL_COMPASS_CONFIG")
    if env_config:
        return Path(env_config).resolve()
    return Path(__file__).parent / "compass_config.json"


# Default config file location (for backward compatibility)
CONFIG_PATH = get_config_path()


def load_config() -> CompassConfig:
    """Load config from file or return defaults."""
    config_path = get_config_path()
    if config_path.exists():
        return CompassConfig.from_file(config_path)
    return get_default_config()
