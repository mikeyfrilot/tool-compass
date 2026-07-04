"""Tool Compass — CLI entry point.

MCC-FT-001: subcommand shell that subsumes the old `gateway:main` entry so
`tool-compass` can be both a server launcher AND a one-shot CLI without
installing a second command. MCC-FT-004 adds `search` and `describe` so
users don't need an MCP client to poke the index.

Backward compat is preserved — `tool-compass` with no args (or with the
explicit `serve` subcommand) still starts the gateway server, matching the
pre-CLI behavior users may have wired into Claude Desktop etc.

Stage D polish (SD-CLI-001..007) added on top of the v2.2.x argparse spine:

- Color discipline (4 colors max: green/red/yellow/dim) with isatty +
  NO_COLOR + TERM=dumb + --no-color detection (`_should_color`).
- `--json` flag everywhere a script might consume the output (doctor/search/
  describe already had it; sync gains a stable JSON shape).
- Rich `Progress` spinners on doctor + sync TEXT mode only — JSON mode
  prints pure JSON to stdout with zero decorations (script-composability).
- Error rewriting: expected exceptions (FileNotFoundError, ConnectionError,
  JSONDecodeError, sqlite3.OperationalError) get a one-line `_print_error`
  with an actionable hint instead of a raw traceback.
- Exit-code audit: 0 success / 1 expected failure (index missing, backend
  down, tool not found) / 2 usage error (bad flag, missing arg) / 130 SIGINT.
- `--version` lives on the root parser only — argparse propagates it to the
  invocation context so `tool-compass --version` works regardless of the
  selected subcommand (tested by test_version.py via _version import).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Stage D polish helpers — color, output, error
# =============================================================================


# 4-color discipline (clig.dev "Use color with intention"). Mapped to Rich
# style strings; the `_make_console` factory honors NO_COLOR / TTY / dumb-term.
_C_SUCCESS = "green"
_C_ERROR = "red"
_C_WARN = "yellow"
_C_DIM = "dim"


def _should_color(stream, no_color_flag: bool = False) -> bool:
    """Decide whether to emit ANSI color on a given stream.

    Honors the de-facto cross-tool conventions:

    - ``--no-color`` flag (passed in here as ``no_color_flag``)
    - ``NO_COLOR`` env var (any non-empty value disables — https://no-color.org)
    - ``TERM=dumb`` (legacy terminal that does not handle ANSI)
    - Non-TTY stream (piped to a file or another command — never colorize)

    Returns True only when all four checks allow color. Callers funnel this
    through ``_make_console`` so every Rich Console in the CLI uses the same
    rule.
    """
    if no_color_flag:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    # ``isatty`` may be missing on captured streams (pytest's capsys). Treat
    # missing-isatty as "no" — capsys is by definition not a real terminal.
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _make_console(*, stderr: bool = False, no_color_flag: bool = False):
    """Construct a Rich Console honoring the color-discipline rules.

    Returns a Rich ``Console`` if ``rich`` is importable, else a small shim
    that exposes ``.print()`` / ``.status()`` so callers don't need to branch.
    The shim never adds color and never blocks on an animated spinner.
    """
    stream = sys.stderr if stderr else sys.stdout
    use_color = _should_color(stream, no_color_flag=no_color_flag)
    try:
        from rich.console import Console
    except ImportError:  # pragma: no cover — rich is a hard dep, but defensive
        return _PlainConsole(stderr=stderr)
    return Console(
        stderr=stderr,
        force_terminal=use_color or None,  # None => Rich auto-detects
        no_color=not use_color,
        # Wider default than 80 so long tool descriptions don't wrap awkwardly
        # when the user is piping into `less -R`.
        width=None,
        highlight=False,
        # markup=True always — Rich strips markup tags into plain text when
        # no_color=True, but if markup=False it emits the raw "[green]" text
        # literally. Tests + non-TTY callers rely on the markup-stripping path.
        markup=True,
        emoji=False,
        legacy_windows=False,
    )


class _PlainConsole:
    """Fallback Console used if rich isn't importable. Stdout-only, no color.

    Behaves like a thin shim over ``print``. Used in pragma:no-cover paths
    (rich is a declared dependency) but kept so a stripped install does not
    crash the CLI hard.
    """

    def __init__(self, *, stderr: bool = False) -> None:
        self._stream = sys.stderr if stderr else sys.stdout

    def print(self, *objects, **_kwargs) -> None:
        msg = " ".join(str(o) for o in objects)
        # Strip Rich markup like [green]ok[/green] so the user does not see
        # raw tags when running without rich.
        import re

        msg = re.sub(r"\[/?[a-zA-Z0-9_ #]+\]", "", msg)
        print(msg, file=self._stream)

    def status(self, message, **_kwargs):  # pragma: no cover - shim
        return _NullStatus()


class _NullStatus:
    """No-op status used by ``_PlainConsole.status`` to keep call-sites uniform."""

    def __enter__(self):  # pragma: no cover - shim
        return self

    def __exit__(self, *_args):  # pragma: no cover - shim
        return False

    def update(self, *_args, **_kwargs):  # pragma: no cover - shim
        return None


def _print_error(
    console,
    msg: str,
    *,
    hint: Optional[str] = None,
    exit_code: int = 1,
) -> int:
    """Print a uniform error line (red ✗) and optional dim hint, return exit code.

    Implements the clig.dev "Catch errors and rewrite them for humans" pattern:
    a single readable sentence on stderr, optionally followed by a dim
    next-action hint. Callers ``return _print_error(...)`` so exit codes
    bubble up unchanged.
    """
    console.print(f"[{_C_ERROR}]✗[/{_C_ERROR}] {msg}")
    if hint:
        console.print(f"  [{_C_DIM}]› {hint}[/{_C_DIM}]")
    return exit_code


def _print_warn(console, msg: str, *, hint: Optional[str] = None) -> None:
    """Print a yellow ⚠ warning line on the supplied console."""
    console.print(f"[{_C_WARN}]⚠[/{_C_WARN}] {msg}")
    if hint:
        console.print(f"  [{_C_DIM}]› {hint}[/{_C_DIM}]")


def _print_success(console, msg: str) -> None:
    """Print a green ✓ success line on the supplied console."""
    console.print(f"[{_C_SUCCESS}]✓[/{_C_SUCCESS}] {msg}")


def _print_dim(console, msg: str) -> None:
    """Print a dim grey hint line on the supplied console."""
    console.print(f"[{_C_DIM}]{msg}[/{_C_DIM}]")


# =============================================================================
# Argparse parser — unchanged shape, with --no-color + --json on `sync`
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse tree.

    Pulled out of main() so tests can introspect the parser without invoking
    any subcommand side effects.

    FE-A-010: top-level description names every subcommand the user might
    type next, including `ui` (which delegates to `tool-compass-ui` if the
    extras are installed). New users running `tool-compass --help` now have
    a discoverable path to the Gradio surface.

    FE-B-009: every subcommand carries an `epilog` with one curated
    example, per clig.dev (CLI Guidelines) and Heroku CLI style.

    SD-CLI-001: ``--no-color`` lives on the root parser so users can disable
    ANSI globally even when wrapping the CLI from a non-interactive harness
    that exposes ``isatty()`` (rare, but it happens with some CI runners).
    """
    parser = argparse.ArgumentParser(
        prog="tool-compass",
        description=(
            "Tool Compass — semantic MCP tool discovery gateway.\n"
            "\n"
            "With no subcommand, runs the gateway server (default).\n"
            "Subcommands: init, serve, search, describe, execute, sync,\n"
            "             doctor, ui, status, categories, audit, analytics,\n"
            "             chains.\n"
            "First run: `tool-compass init` scaffolds a config + prints setup.\n"
            "Web UI: install `tool-compass[ui]` and run `tool-compass ui`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tool-compass init                  # first-run setup (config + MCP snippet)\n"
            "  tool-compass                       # run the MCP gateway\n"
            "  tool-compass search 'read a file'  # one-shot search\n"
            "  tool-compass describe bridge:read_file\n"
            "  tool-compass execute bridge:read_file '{\"path\": \"README.md\"}'\n"
            "  tool-compass sync                  # rebuild the index\n"
            "  tool-compass doctor                # diagnostics JSON\n"
            "  tool-compass ui                    # launch Gradio web UI\n"
            "  tool-compass status                # backend + index health\n"
            "  tool-compass categories            # list categories + counts\n"
            "  tool-compass audit                 # system-wide audit JSON\n"
            "  tool-compass analytics             # usage stats / hot tools\n"
            "  tool-compass chains                # workflow chains\n"
            "\n"
            "Color is on by default in interactive terminals. Disable via\n"
            "`--no-color`, NO_COLOR=1, or TERM=dumb. Output piped to a file\n"
            "or another command is never colored."
        ),
    )
    # FE-A-017: catch the wider exception set so a bad regex, syntax error,
    # or import-time side-effect in _version.py does not crash the CLI on
    # first run. The fallback is the same "unknown" string we used before.
    try:
        from _version import __version__ as _tc_version
    except Exception:  # pragma: no cover — defensive fallback
        _tc_version = "unknown"
    parser.add_argument(
        "--version",
        action="version",
        version=f"tool-compass {_tc_version}",
    )
    # SD-CLI-001: --no-color flag on the root parser. argparse forwards it to
    # every subcommand via the shared Namespace so subparsers don't need to
    # redeclare it.
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        help=(
            "Disable ANSI color even on a TTY. Same effect as NO_COLOR=1 "
            "or TERM=dumb. Output piped to a file or another command is "
            "never colored regardless of this flag."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # doctor — environment snapshot (delegates to config.doctor()).
    p_doctor = sub.add_parser(
        "doctor",
        help="Print diagnostic info (config path, backends, Ollama reach)",
        epilog=(
            "Examples:\n"
            "  tool-compass doctor              # JSON for jq pipelines (default)\n"
            "  tool-compass doctor --text       # human-readable summary\n"
            "  tool-compass doctor | jq .config_path\n"
            "\n"
            "JSON fields: version, python_version, platform, config_path,\n"
            "config, base_path, data_dir, index_path, index_exists,\n"
            "index_size_bytes, analytics_db_path, ollama_url,\n"
            "ollama_reachable, deprecated_tools."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_doctor.add_argument(
        "--text",
        action="store_true",
        help="Text summary (default emits JSON for jq pipelines).",
    )
    # SD-CLI-002: explicit --json flag is a no-op aliasing the default, but
    # documenting it makes pipelines self-explanatory ("tool-compass doctor
    # --json" reads better than depending on an implicit default).
    p_doctor.add_argument(
        "--json",
        action="store_true",
        dest="json",
        help=(
            "Emit JSON (default). Explicit form for self-documenting pipelines."
        ),
    )

    # search — one-shot semantic search against the built index.
    p_search = sub.add_parser(
        "search",
        help="One-shot semantic search (free-text intent)",
        epilog=(
            "Examples:\n"
            "  tool-compass search 'generate an AI image'\n"
            "  tool-compass search 'read a file' --top 3 --json | jq '.[0].tool'\n"
            "\n"
            "If Ollama is unreachable, search falls back to keyword matching."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_search.add_argument(
        "intent",
        type=str,
        help="Free-text description of what you want to do.",
    )
    p_search.add_argument(
        "--top", type=int, default=5,
        help="Maximum number of results to return (default 5; 1-10 valid).",
    )
    p_search.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # describe — print the schema / metadata for a specific tool.
    p_describe = sub.add_parser(
        "describe",
        help="Print tool schema (parameters, examples, server, category)",
        epilog=(
            "Examples:\n"
            "  tool-compass describe bridge:read_file\n"
            "  tool-compass describe comfy:comfy_generate --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_describe.add_argument(
        "tool_name",
        type=str,
        help="Qualified tool name (e.g., bridge:read_file).",
    )
    p_describe.add_argument(
        "--json", action="store_true",
        help="JSON output (default emits Markdown).",
    )

    # execute — proxy a single tool call to its backend and print the result.
    # The terminal counterpart to the MCP `execute` tool: search/describe let
    # you FIND a tool from the shell; execute lets you SMOKE-TEST it without
    # wiring up an MCP client. Arguments are passed as one positional JSON
    # object (kept simple + copy-pasteable from `describe`'s example blocks).
    p_execute = sub.add_parser(
        "execute",
        help="Run a tool on its backend and print the result",
        epilog=(
            "Examples:\n"
            "  tool-compass execute bridge:read_file '{\"path\": \"README.md\"}'\n"
            "  tool-compass execute bridge:list_dir            # no-arg tool\n"
            "  tool-compass execute comfy:comfy_generate '{\"prompt\": \"a cat\"}' \\\n"
            "      --timeout 120 --json | jq .result\n"
            "\n"
            "TOOL is a qualified name (server:tool_name) — see `tool-compass\n"
            "describe <tool>` for its parameters. ARGS is a single JSON object;\n"
            "omit it for tools that take no arguments. On a tool/transport/\n"
            "timeout failure the error is printed and the exit code is nonzero."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_execute.add_argument(
        "tool",
        type=str,
        help="Qualified tool name to run (e.g., bridge:read_file).",
    )
    p_execute.add_argument(
        "args",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Tool arguments as a single JSON object "
            "(e.g. '{\"path\": \"README.md\"}'). Omit for a no-arg tool."
        ),
    )
    p_execute.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Per-call timeout in seconds (float). Defaults to the backend "
            "manager's built-in tool-call timeout when omitted."
        ),
    )
    p_execute.add_argument(
        "--json", action="store_true",
        help="Dump the raw result envelope (success/result/error) as JSON.",
    )

    # sync — rebuild the index from configured backends.
    p_sync = sub.add_parser(
        "sync",
        help="Rebuild the index from backends (run after config changes)",
        epilog=(
            "Examples:\n"
            "  tool-compass sync                # rebuild and print summary\n"
            "  tool-compass sync --json | jq .  # script-friendly output\n"
            "\n"
            "Idempotent. Reads compass_config.json. JSON shape mirrors the\n"
            "internal sync result (tools_added/updated/removed, duration, errors)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sync.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force a full rebuild even if no backend changes are detected. "
            "Currently a no-op (full_sync always rebuilds) — reserved for "
            "the incremental-sync path."
        ),
    )
    # SD-CLI-002: surface --json for sync. Default mode prints a Rich-colored
    # human summary with a spinner; --json strips all decoration and prints
    # the raw sync result for scripts.
    p_sync.add_argument(
        "--json",
        action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # serve — explicit form of the default (server launch). Kept separate
    # so `tool-compass --http` still works on the root parser in future.
    p_serve = sub.add_parser(
        "serve",
        help="Run MCP gateway server (default when no subcommand given)",
        epilog=(
            "Examples:\n"
            "  tool-compass serve              # stdio transport (Claude Desktop)\n"
            "  tool-compass serve --http       # HTTP transport on PORT env\n"
            "  PORT=8080 tool-compass serve    # explicit port via env\n"
            "\n"
            "FE-W11-007: --http now exports PORT (default 8080 if unset) so the\n"
            "gateway selects streamable-http transport on startup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_serve.add_argument(
        "--http",
        nargs="?",
        const="",
        default=None,
        help=(
            "Run with HTTP transport. Without a value: uses $PORT (or 8080). "
            "With a value (e.g. --http 9090): sets PORT to that value."
        ),
    )

    # init — scaffold a compass_config.json at the resolved user config path
    # and print a ready-to-paste Claude Desktop MCP snippet. The onboarding
    # entry point: a first-run user types `tool-compass init`, gets a config
    # file plus the next three commands, and can paste the mcpServers block
    # straight into their client.
    p_init = sub.add_parser(
        "init",
        help="Scaffold compass_config.json + print MCP client setup",
        epilog=(
            "Examples:\n"
            "  tool-compass init                # write config + print next steps\n"
            "  tool-compass init --force        # overwrite an existing config\n"
            "  tool-compass init --json | jq .created\n"
            "\n"
            "Writes to the resolved user config path (see `tool-compass doctor`).\n"
            "Refuses to clobber an existing config unless --force is passed.\n"
            "Prints a Claude Desktop mcpServers snippet you can paste verbatim."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing compass_config.json (default refuses).",
    )
    p_init.add_argument(
        "--json",
        action="store_true",
        help="JSON output ({created, force, overwrote}) for script pipelines.",
    )

    # ui — launch the Gradio web UI. Thin wrapper around `tool-compass-ui`.
    p_ui = sub.add_parser(
        "ui",
        help="Launch the Gradio web UI (requires tool-compass[ui])",
        epilog=(
            "Examples:\n"
            "  tool-compass ui                       # default port 7860\n"
            "  tool-compass ui --port 7861           # custom port\n"
            "  tool-compass ui --share               # public Gradio tunnel\n"
            "  tool-compass ui --auth user:pass      # basic auth (sets GRADIO_AUTH)\n"
            "\n"
            "Equivalent to `tool-compass-ui`. Install the extras with:\n"
            "  pip install tool-compass[ui]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ui.add_argument("--port", type=int, default=7860, help="Port to run on (default 7860)")
    p_ui.add_argument("--host", default="127.0.0.1", help="Host to bind to (default 127.0.0.1)")
    p_ui.add_argument(
        "--share", action="store_true",
        help="Create public Gradio tunnel (requires GRADIO_AUTH or --auth)",
    )
    p_ui.add_argument(
        "--auth", default=None,
        help="Basic auth for --share, format `user:pass`. Sets GRADIO_AUTH env.",
    )

    # status — backend health, breaker state, embedder status, index size,
    # sync state. Delegates to gateway.compass_status().
    p_status = sub.add_parser(
        "status",
        help="Backend + index health, breaker state, sync status",
        epilog=(
            "Examples:\n"
            "  tool-compass status               # human-readable summary\n"
            "  tool-compass status --json | jq .\n"
            "\n"
            "Fields include: index (total_tools, by_category, by_server),\n"
            "backends, health (ollama_available, index_available, degraded_mode),\n"
            "config, hot_cache, sync, chains."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_status.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # categories — list available tool categories + counts. Cheap call.
    p_categories = sub.add_parser(
        "categories",
        help="List available tool categories with counts",
        epilog=(
            "Examples:\n"
            "  tool-compass categories\n"
            "  tool-compass categories --json | jq '.categories'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_categories.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # audit — comprehensive system audit. Delegates to gateway.compass_audit().
    p_audit = sub.add_parser(
        "audit",
        help="Comprehensive system audit (index + backends + chains + analytics)",
        epilog=(
            "Examples:\n"
            "  tool-compass audit                              # 24h timeframe\n"
            "  tool-compass audit --timeframe 7d --json | jq .\n"
            "  tool-compass audit --include-tools              # full tool list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_audit.add_argument(
        "--timeframe", default="24h",
        help="Time window for analytics (1h, 24h, 7d, 30d). Default 24h.",
    )
    p_audit.add_argument(
        "--include-tools", action="store_true",
        help="Include the full list of indexed tools in the audit (large).",
    )
    p_audit.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # analytics — usage stats. Delegates to gateway.compass_analytics().
    p_analytics = sub.add_parser(
        "analytics",
        help="Usage statistics, hot tools, top chains",
        epilog=(
            "Examples:\n"
            "  tool-compass analytics                          # 24h, with failures\n"
            "  tool-compass analytics --timeframe 7d --json\n"
            "  tool-compass analytics --no-failures            # success-only view"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_analytics.add_argument(
        "--timeframe", default="24h",
        help="Time window for stats (1h, 24h, 7d, 30d). Default 24h.",
    )
    p_analytics.add_argument(
        "--no-failures", action="store_true",
        help="Omit failure details from the report (defaults to including them).",
    )
    p_analytics.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    # chains — detected workflow chains. Delegates to gateway.compass_chains().
    p_chains = sub.add_parser(
        "chains",
        help="List or detect tool chains (workflow patterns)",
        epilog=(
            "Examples:\n"
            "  tool-compass chains                  # list all known chains\n"
            "  tool-compass chains --action detect  # detect from analytics\n"
            "  tool-compass chains --json | jq '.chains[].name'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_chains.add_argument(
        "--action", default="list", choices=["list", "detect"],
        help="`list` (default) shows all known chains; `detect` runs detection.",
    )
    p_chains.add_argument(
        "--json", action="store_true",
        help="JSON output suitable for jq/script pipelines.",
    )

    return parser


def _load_index():
    """Open CompassIndex against the on-disk HNSW + SQLite.

    Returns the loaded index or None if loading failed. Callers print a
    user-facing hint when None so we don't leak RuntimeError tracebacks.
    """
    from indexer import CompassIndex

    index = CompassIndex()
    if not index.load_index():
        return None
    return index


def _no_color(args: argparse.Namespace) -> bool:
    """True if the user passed --no-color (or argparse never set the attribute)."""
    return bool(getattr(args, "no_color", False))


# =============================================================================
# Subcommand handlers — text-mode polish layered on; JSON mode unchanged
# =============================================================================


def _cmd_search(args: argparse.Namespace) -> int:
    """Run a one-shot semantic search and print results.

    Text mode is a Rich-colored compact table; --json emits a stable shape
    suited to piping into `jq`. Load-failure is a hint, not a crash — most
    users hitting this path just haven't run `tool-compass sync` yet.

    SD-CLI-005: index-missing now produces a one-line ``_print_error`` with
    an actionable hint, not a flat "Index not available." sentence.
    SD-CLI-006: exit code 1 reserved for expected failures (index missing,
    empty results). Crashes still surface as exit 2 via the top-level
    ``main`` wrapper.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))

    # cli-ux-01: --top help promises "1-10 valid" but the value was passed
    # straight to index.search(top_k=...) / _lexical_search_fallback with no
    # guard. `--top 0` silently rendered a misleading "No tools matched"; huge
    # or negative values slipped through too. Reject out-of-range early (before
    # the index is even loaded) so --help stays truthful. Exit code 2 mirrors
    # argparse's own usage-error convention for bad CLI input.
    if not 1 <= args.top <= 10:
        return _print_error(
            err_console,
            f"--top must be between 1 and 10, got {args.top}.",
            hint="Try --top 5.",
            exit_code=2,
        )

    index = _load_index()
    if index is None:
        return _print_error(
            err_console,
            "Index not available.",
            hint="Run `tool-compass sync` to build it.",
            exit_code=1,
        )

    # cli-ui-001: on ANY embedder failure, degrade to the lexical keyword
    # fallback rather than erroring out — mirroring gateway.compass and
    # ui.search_tools, which both fall back to `gateway._lexical_search_fallback`
    # on any semantic-search exception. This makes the subcommand help ("If
    # Ollama is unreachable, search falls back to keyword matching") true.
    #
    # The old guard was `except (ConnectionError, OSError)`, which the real
    # embedder outages never raise: `embedder._embed_with_concurrency_cap`
    # raises `RuntimeError("Ollama circuit breaker open")` when the breaker is
    # open and re-raises `httpx.TransportError` / `httpx.TimeoutException` on
    # connect/timeout — none subclass ConnectionError/OSError, so they fell
    # through to main()'s catch-all (wrong exit code 2, no actionable hint, and
    # no fallback despite the help promising one).
    degraded = False
    try:
        results = asyncio.run(index.search(args.intent, top_k=args.top))
    except Exception as e:  # noqa: BLE001 — degrade broadly, like gateway/ui
        from gateway import _lexical_search_fallback

        matches = _lexical_search_fallback(
            index, args.intent, args.top, None, None
        )
        # Reshape the lexical dicts into the SearchResult-like duck shape the
        # renderer below expects (.rank / .score / .tool.{name,description,
        # category,server}) — same pattern ui.search_tools uses.
        from types import SimpleNamespace

        results = [
            SimpleNamespace(
                rank=i + 1,
                score=m["confidence"],
                tool=SimpleNamespace(
                    name=m["tool"],
                    description=m.get("description") or "",
                    category=m["category"],
                    server=m["server"],
                ),
            )
            for i, m in enumerate(matches)
        ]
        degraded = True
        # Log the reason to stderr (dim) so the exit-0 degradation is not
        # silent, without derailing the results on stdout.
        _print_dim(
            err_console,
            f"› Semantic search unavailable ({type(e).__name__}); showing "
            "keyword results. Run `tool-compass doctor` to confirm Ollama.",
        )

    if args.json:
        payload = [
            {
                "rank": r.rank,
                "tool": r.tool.name,
                "score": round(r.score, 4),
                "category": r.tool.category,
                "server": r.tool.server,
                "description": r.tool.description,
                # cli-ui-001: flag keyword-fallback results so scripts can tell
                # a degraded response from a semantic one (mirrors ui.py's JSON).
                "degraded": degraded,
            }
            for r in results
        ]
        # SD-CLI-002: --json output goes to stdout with zero decoration so
        # `tool-compass search ... --json | jq` works without `2>/dev/null`.
        print(json.dumps(payload, indent=2))
        return 0

    if not results:
        # FE-B-010 + Nielsen #9: empty results is an error-adjacent state.
        # State the problem AND suggest the next action; keep exit code 0
        # (an intentional empty result is not a failure).
        out_console.print(f"No tools matched intent: {args.intent!r}")
        _print_dim(
            err_console,
            "› Try a broader intent, lower --top to widen, or run "
            "`tool-compass describe <name>` if you know the tool name.",
        )
        return 0

    # cli-ui-001: single inline notice when the results came from the keyword
    # fallback (Ollama/embedder down). Adjacent to the results, like ui.py's
    # _inline_fallback_banner — the user sees WHY the ordering looks coarse.
    if degraded:
        _print_warn(
            out_console,
            "Showing keyword-based results (semantic search unavailable).",
            hint="Start Ollama and re-run for better matches.",
        )

    # Plain-text table — Rich-styled. Header dim, score column green where
    # it's a high-confidence match (>= 0.7) so users can eyeball the spread.
    header = f"{'rank':<4} {'score':<7} {'tool':<40} description"
    out_console.print(f"[{_C_DIM}]{header}[/{_C_DIM}]")
    out_console.print(f"[{_C_DIM}]{'-' * 80}[/{_C_DIM}]")
    for r in results:
        name = r.tool.name if len(r.tool.name) <= 40 else r.tool.name[:37] + "..."
        desc = r.tool.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        score_color = _C_SUCCESS if r.score >= 0.7 else _C_DIM
        out_console.print(
            f"{r.rank:<4} "
            f"[{score_color}]{r.score:<7.3f}[/{score_color}] "
            f"{name:<40} {desc}"
        )
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    """Print schema + description + examples for a named tool.

    Reads straight from the index's SQLite rather than going through
    CompassIndex.search, because describe is a by-name lookup that should
    not depend on the HNSW being loaded.

    SD-CLI-005: missing DB and unknown-tool both rewritten as `_print_error`
    with hints. Exit codes: 1 for "expected" (DB missing, tool not found),
    2 for usage (a DB query that fails to parse the schema).
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))

    from indexer import SQLITE_DB_PATH

    db_path = Path(SQLITE_DB_PATH)
    if not db_path.exists():
        return _print_error(
            err_console,
            f"No tool DB at {db_path}.",
            hint="Run `tool-compass sync` first to build the index.",
            exit_code=1,
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    suggestions: List[str] = []
    try:
        try:
            row = conn.execute(
                "SELECT name, description, category, server, parameters, examples, is_core "
                "FROM tools WHERE name = ?",
                (args.tool_name,),
            ).fetchone()
            # FE-A-018: if no exact match, gather up to 3 substring suggestions
            # so the user is not left on a dead-end "not found". Matches the UI
            # surface's partial-match LIKE path so CLI and UI behave the same.
            #
            # cli-ui-002: escape LIKE wildcards (% and _) in the tool name and
            # pair every LIKE with `ESCAPE '\'` — otherwise a name containing
            # those chars is interpreted as a wildcard (e.g. `a_c` matching
            # `axc`), diverging from gateway._lexical_search_fallback and the UI,
            # which both escape + append ESCAPE. Reuse gateway._escape_like so
            # there is one source of truth for the escape rule.
            if row is None:
                from gateway import _escape_like

                needle = f"%{_escape_like(args.tool_name)}%"
                rows = conn.execute(
                    "SELECT name FROM tools "
                    "WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' "
                    "ORDER BY (CASE WHEN name LIKE ? ESCAPE '\\' THEN 0 ELSE 1 END), name "
                    "LIMIT 3",
                    (needle, needle, needle),
                ).fetchall()
                suggestions = [r["name"] for r in rows]
        # SD-CLI-005: a corrupt DB file raises sqlite3.DatabaseError (the parent
        # of OperationalError) — e.g. "file is not a database". Catching only
        # OperationalError let that escape the rebuild-hint path with a raw
        # traceback. Broaden to DatabaseError so any malformed/corrupt DB routes
        # through the same actionable hint.
        except sqlite3.DatabaseError as e:
            return _print_error(
                err_console,
                f"DB query failed: {e}",
                hint=(
                    "The tool index may be corrupted. Try `tool-compass sync` "
                    "to rebuild it."
                ),
                exit_code=2,
            )
    finally:
        # Always close the connection — the previous early-return-on-error path
        # leaked the open conn on some branches.
        conn.close()

    if row is None:
        # Build a hint that includes suggestions when we have them.
        if suggestions:
            hint = "Did you mean: " + ", ".join(suggestions)
        else:
            hint = (
                "Run `tool-compass search` with the intent you have in mind "
                "to discover available tools."
            )
        return _print_error(
            err_console,
            f"Tool not found: {args.tool_name}",
            hint=hint,
            exit_code=1,
        )

    try:
        parameters = json.loads(row["parameters"]) if row["parameters"] else {}
        examples = json.loads(row["examples"]) if row["examples"] else []
    except json.JSONDecodeError as e:
        # SD-CLI-005: a malformed schema in the DB used to crash with a raw
        # traceback. Rewrite it so the user sees the path they should
        # rebuild from.
        return _print_error(
            err_console,
            f"Tool schema malformed for {args.tool_name}: {e}",
            hint="Run `tool-compass sync` to rebuild the index.",
            exit_code=2,
        )

    if args.json:
        payload = {
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "server": row["server"],
            "parameters": parameters,
            "examples": examples,
            "is_core": bool(row["is_core"]),
        }
        print(json.dumps(payload, indent=2))
        return 0

    # Markdown — human-friendly, pipeable into a viewer. Headings get the
    # green ✓ accent so colored terminals can scan structure at a glance;
    # uncolored output stays plain Markdown.
    out_console.print(f"[bold]# {row['name']}[/bold]")
    out_console.print()
    out_console.print(f"[{_C_DIM}]**Category:**[/{_C_DIM}] {row['category']}")
    out_console.print(f"[{_C_DIM}]**Server:**[/{_C_DIM}] {row['server']}")
    if row["is_core"]:
        out_console.print(f"[{_C_SUCCESS}]**Core:** yes[/{_C_SUCCESS}]")
    out_console.print()
    out_console.print("[bold]## Description[/bold]")
    out_console.print(row["description"] or "(no description)")
    if parameters:
        out_console.print()
        out_console.print("[bold]## Parameters[/bold]")
        for pname, ptype in parameters.items():
            out_console.print(f"- `{pname}`: {ptype}")
    if examples:
        out_console.print()
        out_console.print("[bold]## Examples[/bold]")
        for ex in examples:
            out_console.print(f"- {ex}")
    return 0


def _cmd_execute(args: argparse.Namespace) -> int:
    """Proxy a single tool call to its backend and print the result.

    FEAT-05: the terminal counterpart to the MCP ``execute`` tool. Adopters
    can already ``search`` / ``describe`` from the shell; this lets them
    smoke-test a proxied call without standing up an MCP client.

    Design — argument passing:
        A single positional JSON object (``args``) rather than repeatable
        ``--arg key=value``. Rationale: (1) MCP tool arguments are already
        JSON with nested/typed values (numbers, booleans, arrays, objects);
        a flat ``key=value`` surface can't express those without a bespoke
        mini-DSL, whereas one JSON blob is lossless. (2) ``describe`` already
        prints example JSON the user can paste verbatim. (3) It mirrors the
        MCP ``execute(tool_name, arguments)`` contract 1:1, so behavior is
        identical whether the caller is the LLM or the terminal — no second
        code path to drift. Malformed JSON is a usage error (exit 2), not a
        crash.

    Execution path:
        Obtains the backend-manager SINGLETON the same way the gateway does
        (``gateway.get_backends()`` — no reimplementation) and calls the
        manager's ``execute_tool(tool, arguments, timeout=...)``. That method
        handles connect/reconnect internally (``ensure_connected``) and returns
        the structured envelope (BR-B-001/012): success is
        ``{success: True, result, content}``; failure is
        ``{success: False, error_kind, error, ...}``. We render success to
        stdout and route any failure through ``_print_error`` with the
        ``error_kind`` as the hint, exiting nonzero. ``--json`` dumps the raw
        envelope verbatim (still nonzero on failure so scripts can branch on
        the exit code without parsing).

    Exit codes (consistent with the SD-CLI-006 audit):
        0   tool ran and returned success
        1   tool/transport/timeout/backend failure (an expected runtime outcome)
        2   usage error (malformed / non-object args JSON)
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    # Parse the args JSON up front — a usage error (exit 2) before we touch a
    # backend, mirroring _cmd_search's early --top validation. ``None`` (the
    # positional was omitted) means "no arguments" -> empty dict.
    raw_args = getattr(args, "args", None)
    if raw_args is None:
        arguments: Dict[str, Any] = {}
    else:
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return _print_error(
                err_console,
                f"Could not parse ARGS as JSON: {e}",
                hint=(
                    "Pass a single JSON object, e.g. "
                    "'{\"path\": \"README.md\"}'. Quote it so the shell "
                    "keeps it as one argument."
                ),
                exit_code=2,
            )
        # MCP tool arguments are always an object. A list/scalar would be
        # rejected by the backend anyway — catch it here with a clearer message.
        if not isinstance(arguments, dict):
            return _print_error(
                err_console,
                f"ARGS must be a JSON object, got {type(arguments).__name__}.",
                hint="Wrap the arguments in braces: '{\"key\": \"value\"}'.",
                exit_code=2,
            )

    try:
        from gateway import get_backends
    except Exception as e:  # pragma: no cover — gateway import is exercised elsewhere
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    async def _run() -> Dict[str, Any]:
        # Reuse the gateway's backend-manager singleton (do NOT construct a
        # second one — that would spawn a duplicate set of child processes).
        manager = await get_backends()
        try:
            return await manager.execute_tool(
                args.tool, arguments, timeout=args.timeout
            )
        finally:
            # One-shot CLI: shut the backend children down cleanly so we don't
            # leak processes on exit. Same disconnect discipline as _cmd_sync.
            try:
                await manager.disconnect_all()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass

    try:
        envelope = asyncio.run(_run())
    except Exception as e:
        # An unhandled raise from the manager (rare — execute_tool is designed
        # to return envelopes, not raise) degrades to one line, never a stack.
        return _print_error(
            err_console,
            f"Tool execution failed: {type(e).__name__}: {e}",
            hint="Run `tool-compass status` to inspect backend health.",
            exit_code=1,
        )

    success = bool(envelope.get("success")) if isinstance(envelope, dict) else False

    if json_mode:
        # Raw envelope verbatim on stdout. Exit code still reflects success so
        # `... --json && ...` composition works without parsing the payload.
        print(json.dumps(envelope, indent=2, default=str))
        return 0 if success else 1

    if success:
        # Print the tool result. ``result`` is the concatenated text the
        # backend emitted; fall back to the full envelope if it's absent.
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if result is None:
            result = json.dumps(envelope, indent=2, default=str)
        _print_success(out_console, f"{args.tool} executed.")
        # Print the payload via a plain print so Rich never reflows/styles it —
        # the user may be piping the tool's output somewhere.
        print(result)
        return 0

    # Failure envelope — surface the message + the error_kind (so the user can
    # tell a tool_error from a transport/timeout failure) and exit nonzero.
    error_msg = (
        envelope.get("error") if isinstance(envelope, dict) else None
    ) or "Tool execution failed."
    error_kind = envelope.get("error_kind") if isinstance(envelope, dict) else None
    hint = f"error_kind: {error_kind}" if error_kind else None
    rc = _print_error(err_console, error_msg, hint=hint, exit_code=1)
    # On a cold install a not-found routes here; nudge toward sync like the
    # other index-aware commands.
    _maybe_suggest_sync(err_console)
    return rc


def _cmd_sync(args: argparse.Namespace) -> int:
    """Rebuild the index from configured backends.

    `--force` is accepted for future compatibility; today SyncManager.full_sync
    always does a full rebuild, so the flag is a no-op but is retained so
    scripts can be written against the documented surface.

    SD-CLI-003: text mode shows a Rich spinner while the rebuild runs (the
    underlying sync is opaque to us — there's no per-backend progress event
    surface yet). JSON mode (``--json``) prints the raw sync result with
    zero decoration. We keep the legacy "print full JSON to stdout" behavior
    behind the flag so existing scripts pipe through unchanged.
    SD-CLI-005: a config without backends now emits ``_print_error`` with a
    hint pointing at the config path. ``ConnectionError`` and
    ``FileNotFoundError`` from the sync path are caught and rewritten.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    from backend_client_simple import SimpleBackendManager as BackendManager
    from config import load_config
    from indexer import CompassIndex
    from sync_manager import SyncManager

    async def _run() -> tuple[int, Optional[dict[str, Any]]]:
        config = load_config()
        if not config.backends:
            _print_error(
                err_console,
                "No backends configured.",
                hint=(
                    "Edit your compass_config.json "
                    "(see `tool-compass doctor` for the resolved path)."
                ),
                exit_code=1,
            )
            return 1, None

        backends = BackendManager(config)
        index = CompassIndex()
        # Best-effort load; full_sync will rebuild regardless.
        index.load_index()

        sync = SyncManager(config, index, backends)
        try:
            result = await sync.full_sync()
        finally:
            await backends.disconnect_all()

        return 0, result

    # `args.force` is deliberately unused today — see docstring. Reference it
    # so linters don't flag the attribute as dead.
    _ = args.force

    try:
        # JSON mode: no spinner (decorations on stdout would corrupt the
        # script-composable shape). Text mode: spinner around the work, then
        # a colored summary on stdout.
        if json_mode:
            rc, result = asyncio.run(_run())
            if result is not None:
                print(json.dumps(result, indent=2, default=str))
            return rc

        # Text mode — Rich Progress with a Spinner column. ``transient=True``
        # means the spinner disappears once work finishes, leaving the
        # terminal clean for the summary lines below.
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=err_console,  # spinner on stderr so stdout stays clean
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Rebuilding tool index from backends...",
                    total=None,
                )
                rc, result = asyncio.run(_run())
                progress.update(task, completed=True)
        except ImportError:  # pragma: no cover — rich is a hard dep
            rc, result = asyncio.run(_run())

        if rc != 0 or result is None:
            return rc

        # Summary line — honest count straight from the real full_sync
        # contract (status / tools_indexed / backends_synced /
        # connected_backends / failed_backends / build_result). The prior
        # version read tools_added/updated/removed/duration_seconds, none of
        # which full_sync emits, so it always printed "+0 ~0 -0". We report
        # the indexed count instead — the one number full_sync guarantees.
        tools_indexed = result.get("tools_indexed", 0)
        failed_backends = result.get("failed_backends") or []

        _print_success(
            out_console, f"Sync complete — {tools_indexed} tools indexed."
        )
        if failed_backends:
            # cli-ux-002: configured backends that failed to connect this pass.
            # The old `if errors:` branch read a key full_sync never returns,
            # so partial failures were reported as unqualified success.
            _print_warn(
                err_console,
                f"{len(failed_backends)} backend(s) failed to connect: "
                f"{', '.join(failed_backends)}",
                hint="Run `tool-compass doctor` to inspect backend health.",
            )
        return rc
    except FileNotFoundError as e:
        return _print_error(
            err_console,
            f"Required file missing: {e}",
            hint="Run `tool-compass doctor` to confirm config + index paths.",
            exit_code=1,
        )
    except ConnectionError as e:
        return _print_error(
            err_console,
            f"Backend connection failed: {e}",
            hint=(
                "Check `tool-compass doctor` for backend health and confirm "
                "the configured server commands are runnable."
            ),
            exit_code=1,
        )


# =============================================================================
# init — onboarding scaffold + MCP-client registration snippet
# =============================================================================


# Server key + npx package name used in the pasteable Claude Desktop snippet.
# Kept as module constants so the handler and tests reference one source.
_MCP_SERVER_KEY = "tool-compass"
_NPX_PACKAGE = "@mcptoolshop/tool-compass"


def _claude_desktop_snippet() -> str:
    """Return a ready-to-paste Claude Desktop `mcpServers` JSON block.

    Uses the zero-prerequisite `npx -y @mcptoolshop/tool-compass serve` form
    (matches npm/README.md + the handbook recipes). Deliberately carries NO
    secrets — backends and any tokens live in compass_config.json, never in
    the client config — so a user can paste this verbatim without leaking
    anything. Built via ``json.dumps`` so the indentation is always valid
    JSON the user can drop straight into ``claude_desktop_config.json``.
    """
    block = {
        "mcpServers": {
            _MCP_SERVER_KEY: {
                "command": "npx",
                "args": ["-y", _NPX_PACKAGE, "serve"],
            }
        }
    }
    return json.dumps(block, indent=2)


def _minimal_config_json() -> str:
    """Serialize a minimal-but-valid compass_config.json.

    Fallback used only when the repo's ``compass_config.example.json`` cannot
    be located (e.g. a wheel install that doesn't ship the example). Rather
    than duplicate the full schema, we round-trip the in-code defaults through
    ``CompassConfig.to_dict()`` so the written file always tracks the live
    schema (including any new fields a sibling agent adds to the dataclass).
    The empty ``backends: {}`` skeleton signals the user to fill it in.
    """
    from config import get_default_config

    return json.dumps(get_default_config().to_dict(), indent=2)


def _locate_example_config() -> Optional[Path]:
    """Find the repo's compass_config.example.json, or None if absent.

    Checks next to this module first (source/editable installs ship the
    example alongside cli.py), then the resolved base path. Copying the
    example at runtime means a field another agent adds to the example
    (e.g. ``embedding_provider``) is picked up automatically — we never
    parse or rewrite it, just copy the bytes.
    """
    candidates = [Path(__file__).parent / "compass_config.example.json"]
    try:
        from config import get_base_path

        candidates.append(get_base_path() / "compass_config.example.json")
    except Exception:  # pragma: no cover — defensive
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _cmd_init(args: argparse.Namespace) -> int:
    """Scaffold compass_config.json + print MCP-client setup.

    The onboarding entry point. Resolves the user config path via
    ``config.get_config_path()`` (honors TOOL_COMPASS_CONFIG /
    TOOL_COMPASS_DATA_DIR exactly like every other command), creates parent
    dirs, and writes a config — sourced from the repo's
    ``compass_config.example.json`` if locatable, else a minimal valid config
    round-tripped from the live dataclass defaults.

    Refuses to overwrite an existing config unless ``--force`` is passed
    (exit 1 + hint). On success prints the written path, the next three
    commands, and a Claude Desktop ``mcpServers`` snippet. ``--json`` emits a
    ``{created, force, overwrote}`` shape with zero decoration for scripts.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))
    force = bool(getattr(args, "force", False))

    from config import get_config_path

    config_path = get_config_path()

    # Refuse to clobber an existing config unless --force. Exit 1 (expected
    # failure) with an actionable hint — never a silent overwrite.
    existed = config_path.exists()
    if existed and not force:
        return _print_error(
            err_console,
            f"Config already exists at {config_path}.",
            hint="Pass --force to overwrite it, or edit it directly.",
            exit_code=1,
        )

    # Resolve the content: prefer copying the example (so a field a sibling
    # agent adds to the example is picked up for free), else a minimal config.
    example = _locate_example_config()
    try:
        if example is not None:
            content = example.read_text(encoding="utf-8")
            source = str(example)
        else:
            content = _minimal_config_json()
            source = "built-in defaults"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(content, encoding="utf-8")
    except OSError as e:
        return _print_error(
            err_console,
            f"Could not write config to {config_path}: {e}",
            hint="Check the directory is writable, or set TOOL_COMPASS_CONFIG.",
            exit_code=1,
        )

    snippet = _claude_desktop_snippet()

    if json_mode:
        # Pure JSON on stdout — no decoration so `... --json | jq` works.
        payload = {
            "created": str(config_path),
            "source": source,
            "force": force,
            "overwrote": existed,
            "claude_desktop_config": _claude_desktop_snippet_obj(),
        }
        print(json.dumps(payload, indent=2))
        return 0

    verb = "Overwrote" if existed else "Created"
    # Print the path on its own plain line so Rich never reflows it mid-path —
    # a wrapped config path is un-copyable. The ✓ line stays styled.
    _print_success(out_console, f"{verb} config:")
    print(str(config_path))
    out_console.print()
    out_console.print("[bold]Next steps[/bold]")
    out_console.print(
        f"  [{_C_DIM}]1.[/{_C_DIM}] Edit [bold]backends[/bold] in "
        f"{config_path} to point at your MCP servers."
    )
    out_console.print(
        f"  [{_C_DIM}]2.[/{_C_DIM}] Run [bold]tool-compass sync[/bold] to "
        "build the search index."
    )
    out_console.print(
        f"  [{_C_DIM}]3.[/{_C_DIM}] Run [bold]tool-compass serve[/bold] to "
        "start the MCP gateway."
    )
    out_console.print()
    out_console.print(
        "[bold]Register with Claude Desktop[/bold] "
        f"[{_C_DIM}](paste into claude_desktop_config.json):[/{_C_DIM}]"
    )
    # Print the snippet via a plain print so Rich never reflows / styles the
    # JSON — the user must be able to copy it byte-for-byte.
    print(snippet)
    return 0


def _claude_desktop_snippet_obj() -> dict:
    """The Claude Desktop snippet as a dict (for the --json payload)."""
    return {
        "mcpServers": {
            _MCP_SERVER_KEY: {
                "command": "npx",
                "args": ["-y", _NPX_PACKAGE, "serve"],
            }
        }
    }


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Print diagnostic info.

    FE-A-011: wrap config.doctor() in try/except so a corrupted config or
    transient permission error returns a non-zero exit code with a single
    user-facing line instead of a raw traceback. The error goes to stderr;
    JSON/text payload goes to stdout, keeping the contract pipeline-safe.

    FE-B-018: default is JSON (this is the canonical machine-readable
    diagnostic surface — release-smoke + CI gates consume the JSON form);
    `--text` switches to a compact human-readable summary so terminal
    users aren't forced to install jq.

    SD-CLI-003: text mode now shows a Rich spinner per logical check
    (config load, version probe, Ollama reach). JSON mode preserves the
    pre-Stage-D behavior exactly — pure JSON to stdout, no decorations
    (tests parse the JSON directly, so any spinner output would break them).
    SD-CLI-004: text mode summary uses the 4-color discipline: green ✓ for
    "ollama reachable" and "config loaded", yellow ⚠ for "ollama down",
    dim grey for paths.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    text_mode = bool(getattr(args, "text", False))

    try:
        if text_mode:
            # Spinner only in text mode — JSON callers must see pure JSON.
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=err_console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(
                        "Collecting diagnostics...",
                        total=None,
                    )
                    from config import doctor

                    payload = doctor()
                    progress.update(task, completed=True)
            except ImportError:  # pragma: no cover — rich is a hard dep
                from config import doctor

                payload = doctor()
        else:
            from config import doctor

            payload = doctor()
    except FileNotFoundError as e:
        return _print_error(
            err_console,
            f"Diagnostic check failed — config file missing: {e}",
            hint=(
                "Edit compass_config.json (see CONTRIBUTING.md for the schema)."
            ),
            exit_code=2,
        )
    except json.JSONDecodeError as e:
        return _print_error(
            err_console,
            f"Diagnostic check failed — config JSON is malformed: {e}",
            hint=(
                "Validate compass_config.json with `python -m json.tool` and "
                "re-run `tool-compass doctor`."
            ),
            exit_code=2,
        )
    except Exception as e:
        return _print_error(
            err_console,
            f"Diagnostic check failed: {type(e).__name__}: {e}",
            exit_code=2,
        )

    if text_mode:
        # Compact text summary — version, config path, backend count, ollama
        # health. Each line colored per the 4-color discipline so the user
        # can eyeball "what's wrong" without re-reading the JSON.
        version = payload.get("version", "unknown")
        config_path = payload.get("config_path", "(unknown)")
        backends = payload.get("backends") or payload.get("config", {}).get(
            "backends"
        ) or {}
        ollama_url = payload.get("ollama_url", "(unset)")
        ollama_ok = payload.get("ollama_reachable", False)
        index_exists = payload.get("index_exists", False)

        out_console.print(f"[bold]tool-compass {version}[/bold]")
        out_console.print(f"  [{_C_DIM}]config:[/{_C_DIM}] {config_path}")
        # Backend count works for both dict-of-backends and list shapes.
        if isinstance(backends, (list, tuple)):
            backend_count = len(backends)
        elif isinstance(backends, dict):
            backend_count = len(backends)
        else:
            backend_count = 0
        out_console.print(
            f"  [{_C_DIM}]backends:[/{_C_DIM}] {backend_count} configured"
        )
        # cli-ux-005: surface unresolved ${VAR} references in text mode too
        # (JSON mode already carries them via config_unresolved_vars). An
        # unresolved var means a backend command/url still contains a literal
        # ${NAME} that never got an env value — a common silent misconfig.
        unresolved_vars = payload.get("config_unresolved_vars") or []
        if unresolved_vars:
            _print_warn(
                out_console,
                f"unresolved config vars: {', '.join(unresolved_vars)}",
                hint=(
                    "Set these env vars or hard-code values in "
                    "compass_config.json."
                ),
            )
        if ollama_ok:
            _print_success(out_console, f"ollama reachable at {ollama_url}")
        else:
            _print_warn(
                out_console,
                f"ollama unreachable at {ollama_url}",
                hint="Run `ollama serve` or set OLLAMA_URL.",
            )
        if index_exists:
            _print_success(out_console, "tool index present")
        else:
            _print_warn(
                out_console,
                "tool index missing",
                hint="Run `tool-compass sync` to build the index.",
            )
        return 0

    # SD-CLI-002: JSON mode — pure JSON on stdout. Tests parse this directly.
    print(json.dumps(payload, indent=2, default=str))
    return 0


# =============================================================================
# FE-W11 — gateway-handler-backed subcommands (status, categories, audit,
# analytics, chains, ui). Each is a thin wrapper around the matching @mcp.tool
# function in gateway.py so the CLI surface stays in lockstep with the MCP
# surface (no parallel implementation to drift).
# =============================================================================


def _check_index_initialized() -> bool:
    """Return True if the on-disk index DB exists.

    Used by gateway-backed CLI commands to print a "run sync first" hint when
    callers run a status/categories/audit on a cold install. The gateway
    handlers themselves can build the index lazily, but emitting the hint
    keeps the CLI conversational rather than mysterious.
    """
    try:
        from indexer import SQLITE_DB_PATH

        return Path(SQLITE_DB_PATH).exists()
    except Exception:  # pragma: no cover — defensive
        return False


def _maybe_suggest_sync(err_console) -> None:
    """Emit a one-line "run sync first" hint if the index DB is missing."""
    if not _check_index_initialized():
        _print_dim(
            err_console,
            "› No tool index found yet. Run `tool-compass sync` first.",
        )


def _dump_json(payload: Any) -> int:
    """Print JSON to stdout with the same shape `doctor --json` uses."""
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _is_error_envelope(payload: Any) -> bool:
    """True when ``payload`` looks like a compass error envelope.

    The gateway returns ``{"error": {"code": "...", "title": "...", ...}}``
    when a feature is disabled or a subsystem is down. We unwrap that into
    a `_print_error` line so the CLI surface looks consistent across calls.
    """
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and "code" in payload["error"]
    )


def _print_envelope_error(err_console, payload: dict) -> int:
    """Render a gateway error envelope as a single colored error line."""
    err = payload["error"]
    title = err.get("title") or err.get("detail") or "Operation failed"
    suggestions = err.get("suggestions") or []
    hint = suggestions[0] if suggestions else None
    return _print_error(err_console, title, hint=hint, exit_code=1)


def _cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Gradio web UI.

    Thin wrapper that delegates straight to ``ui:main`` (the existing console
    script). Forwards ``--port`` / ``--host`` / ``--share`` and propagates
    ``--auth user:pass`` through the GRADIO_AUTH env var that ui.py already
    reads for the public-tunnel auth check.

    FE-W11-002: the README + docs advertise `tool-compass ui`; before this
    subcommand it 404'd. Tests pass when `tool-compass ui --help` works and
    the dispatch reaches ``ui.main``.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))

    # The UI is in an optional extras install (`tool-compass[ui]`). Probing the
    # import lets us emit a friendly "pip install tool-compass[ui]" hint
    # instead of leaking ModuleNotFoundError("gradio") into the user's face.
    try:
        import ui as _ui_module
    except ImportError as e:
        return _print_error(
            err_console,
            f"UI extras not installed ({e}).",
            hint="Install with: pip install tool-compass[ui]",
            exit_code=1,
        )

    # Bridge --auth to GRADIO_AUTH so ui.main's existing share-auth gate sees it.
    if args.auth:
        os.environ["GRADIO_AUTH"] = args.auth

    # Rebuild argv for ui.main so it can re-parse with its own argparse. We
    # forward only the flags ui.main knows about — extra root-parser flags
    # like --no-color would crash ui's strict parser.
    forwarded: List[str] = []
    if args.port is not None:
        forwarded += ["--port", str(args.port)]
    if args.host:
        forwarded += ["--host", args.host]
    if args.share:
        forwarded.append("--share")

    # ui.main reads sys.argv directly, so swap it for the forwarded slice
    # for the duration of the call and restore on return.
    saved_argv = sys.argv
    try:
        sys.argv = ["tool-compass-ui"] + forwarded
        return _ui_module.main() or 0
    finally:
        sys.argv = saved_argv


def _cmd_status(args: argparse.Namespace) -> int:
    """Print backend + index health (calls gateway.compass_status).

    FE-W11-003: text mode renders a human-readable summary; --json emits the
    raw gateway shape so dashboards / scripts can consume it unchanged. We
    don't re-implement the gateway's logic — single source of truth.
    """
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    try:
        from gateway import compass_status
    except Exception as e:
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    try:
        payload = asyncio.run(compass_status())
    except Exception as e:
        return _print_error(
            err_console,
            f"compass_status failed: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if json_mode:
        return _dump_json(payload)

    # Text mode — render the key facts. Each block is wrapped so a missing
    # subsystem (e.g. analytics disabled) just dims instead of erroring.
    if _is_error_envelope(payload):
        return _print_envelope_error(err_console, payload)

    index_info = payload.get("index") or {}
    backends_info = payload.get("backends") or {}
    health = payload.get("health") or {}
    sync_info = payload.get("sync") or {}

    total_tools = index_info.get("total_tools", 0)
    out_console.print("[bold]Tool Compass status[/bold]")
    out_console.print(f"  [{_C_DIM}]index:[/{_C_DIM}] {total_tools} tools indexed")
    by_server = index_info.get("by_server") or {}
    if by_server:
        out_console.print(
            f"  [{_C_DIM}]servers:[/{_C_DIM}] "
            + ", ".join(f"{k}={v}" for k, v in sorted(by_server.items()))
        )

    # Backend health — degraded_mode flag is the load-bearing signal.
    connected = backends_info.get("connected_backends") or []
    configured = backends_info.get("configured_backends") or []
    if connected:
        _print_success(
            out_console,
            f"backends connected: {len(connected)}/{len(configured) or len(connected)}",
        )
    elif configured:
        _print_warn(
            out_console,
            f"no backends connected ({len(configured)} configured)",
            hint="Check `tool-compass doctor` for details.",
        )

    if health.get("ollama_available"):
        _print_success(out_console, "ollama reachable")
    else:
        _print_warn(out_console, "ollama unreachable",
                    hint="Run `ollama serve` or set OLLAMA_URL.")
    if not health.get("index_available", True):
        _print_warn(out_console, "index in degraded mode",
                    hint="Run `tool-compass sync` to rebuild.")
    if health.get("degraded_mode"):
        _print_warn(out_console, "compass is serving DEGRADED responses")

    if sync_info and "last_sync_at" in sync_info:
        out_console.print(
            f"  [{_C_DIM}]last sync:[/{_C_DIM}] {sync_info.get('last_sync_at')}"
        )

    _maybe_suggest_sync(err_console)
    return 0


def _cmd_categories(args: argparse.Namespace) -> int:
    """List categories + counts (calls gateway.compass_categories)."""
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    try:
        from gateway import compass_categories
    except Exception as e:
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    try:
        payload = asyncio.run(compass_categories())
    except Exception as e:
        return _print_error(
            err_console,
            f"compass_categories failed: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if json_mode:
        return _dump_json(payload)

    if _is_error_envelope(payload):
        return _print_envelope_error(err_console, payload)

    cats = payload.get("categories") or {}
    total = payload.get("total_tools", 0)
    out_console.print(f"[bold]Categories[/bold] ({total} tools indexed)")
    out_console.print(f"[{_C_DIM}]{'-' * 40}[/{_C_DIM}]")
    if not cats:
        out_console.print(f"  [{_C_DIM}](no categories — index empty)[/{_C_DIM}]")
        _maybe_suggest_sync(err_console)
        return 0
    # Sort by count desc so the heavyweights show first.
    for name, count in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0])):
        out_console.print(f"  [{_C_SUCCESS}]{count:>5}[/{_C_SUCCESS}]  {name}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Comprehensive system audit (calls gateway.compass_audit)."""
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    try:
        from gateway import compass_audit
    except Exception as e:
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    try:
        payload = asyncio.run(
            compass_audit(
                include_tools=bool(args.include_tools),
                timeframe=args.timeframe,
            )
        )
    except Exception as e:
        return _print_error(
            err_console,
            f"compass_audit failed: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if json_mode:
        return _dump_json(payload)

    if _is_error_envelope(payload):
        return _print_envelope_error(err_console, payload)

    system = payload.get("system") or {}
    categories = payload.get("categories") or {}
    servers = payload.get("servers") or {}
    out_console.print(f"[bold]Tool Compass audit[/bold] ({args.timeframe})")
    out_console.print(
        f"  [{_C_DIM}]version:[/{_C_DIM}] {system.get('version', 'unknown')}  "
        f"[{_C_DIM}]tools:[/{_C_DIM}] {system.get('total_tools', 0)}"
    )
    if categories:
        out_console.print(
            f"  [{_C_DIM}]categories:[/{_C_DIM}] "
            + ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))
        )
    if servers:
        out_console.print(
            f"  [{_C_DIM}]servers:[/{_C_DIM}] "
            + ", ".join(f"{k}={v}" for k, v in sorted(servers.items()))
        )

    # Backends + hot cache + chain summary if present.
    if "backends" in payload:
        be = payload["backends"]
        if isinstance(be, dict) and not be.get("error"):
            connected = be.get("connected_backends") or []
            configured = be.get("configured_backends") or []
            _print_success(
                out_console,
                f"backends: {len(connected)}/{len(configured) or len(connected)} connected",
            )
        elif isinstance(be, dict) and be.get("error"):
            _print_warn(out_console, f"backends error: {be['error']}")
    if "hot_cache" in payload and isinstance(payload["hot_cache"], dict):
        hc = payload["hot_cache"]
        size = hc.get("size", 0)
        out_console.print(f"  [{_C_DIM}]hot cache:[/{_C_DIM}] {size} tools")
    if "chains" in payload and isinstance(payload["chains"], dict):
        ch = payload["chains"]
        total_chains = ch.get("total", 0)
        out_console.print(f"  [{_C_DIM}]chains:[/{_C_DIM}] {total_chains} detected")

    if args.include_tools:
        tools = payload.get("tools") or []
        out_console.print(f"\n[bold]Tools ({len(tools)})[/bold]")
        for t in tools:
            name = t.get("name") if isinstance(t, dict) else str(t)
            out_console.print(f"  {name}")
    _maybe_suggest_sync(err_console)
    return 0


def _cmd_analytics(args: argparse.Namespace) -> int:
    """Usage statistics (calls gateway.compass_analytics)."""
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    try:
        from gateway import compass_analytics
    except Exception as e:
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    include_failures = not bool(args.no_failures)
    try:
        payload = asyncio.run(
            compass_analytics(
                timeframe=args.timeframe,
                include_failures=include_failures,
            )
        )
    except Exception as e:
        return _print_error(
            err_console,
            f"compass_analytics failed: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if json_mode:
        return _dump_json(payload)

    if _is_error_envelope(payload):
        return _print_envelope_error(err_console, payload)

    out_console.print(f"[bold]Tool Compass analytics[/bold] ({args.timeframe})")
    # Top tools — list the first ~10 by call count if the gateway returned a
    # sorted shape. The gateway's exact shape can vary by analytics build, so
    # we fall back to `_dump_json` rendering if the expected keys are absent.
    top_tools = payload.get("top_tools") or payload.get("hot_tools") or []
    rendered_top = isinstance(top_tools, list) and bool(top_tools)
    if rendered_top:
        out_console.print(f"  [{_C_DIM}]top tools:[/{_C_DIM}]")
        for entry in top_tools[:10]:
            if isinstance(entry, dict):
                name = entry.get("tool_name") or entry.get("name") or "?"
                count = entry.get("call_count") or entry.get("count") or 0
                out_console.print(f"    {count:>5}  {name}")
            else:
                out_console.print(f"    {entry}")
    summary = payload.get("summary") or {}
    if summary:
        total = summary.get("total_calls", 0)
        out_console.print(f"  [{_C_DIM}]total calls:[/{_C_DIM}] {total}")
    # cli-ux-03: with analytics enabled but nothing logged yet (empty top_tools
    # AND empty summary — the normal state right after `sync`), the command
    # otherwise prints only the header, which reads as broken. Emit a dim
    # empty-state line mirroring the search/chains empty-state guidance.
    if not rendered_top and not summary:
        _print_dim(
            out_console,
            f"  (no usage recorded in the last {args.timeframe} — analytics is "
            "enabled and waiting for tool calls)",
        )
    if include_failures:
        failures = payload.get("failures") or []
        if failures:
            _print_warn(err_console, f"{len(failures)} failure(s) recorded in window")
    # cli-ux-02: emit the cold-install "run sync first" hint when the index DB
    # is absent, uniform with status/categories/audit/chains.
    _maybe_suggest_sync(err_console)
    return 0


def _cmd_chains(args: argparse.Namespace) -> int:
    """List or detect workflow chains (calls gateway.compass_chains)."""
    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))
    out_console = _make_console(no_color_flag=_no_color(args))
    json_mode = bool(getattr(args, "json", False))

    try:
        from gateway import compass_chains
    except Exception as e:
        return _print_error(
            err_console,
            f"Could not import gateway: {type(e).__name__}: {e}",
            exit_code=2,
        )

    try:
        payload = asyncio.run(compass_chains(action=args.action))
    except Exception as e:
        return _print_error(
            err_console,
            f"compass_chains failed: {type(e).__name__}: {e}",
            exit_code=1,
        )

    if json_mode:
        return _dump_json(payload)

    if _is_error_envelope(payload):
        return _print_envelope_error(err_console, payload)

    if args.action == "list":
        chains = payload.get("chains") or []
        out_console.print(
            f"[bold]Tool chains[/bold] ({len(chains)} total)"
        )
        if not chains:
            out_console.print(f"  [{_C_DIM}](no chains detected yet)[/{_C_DIM}]")
            _maybe_suggest_sync(err_console)
            return 0
        for c in chains:
            name = c.get("name", "?")
            tools = c.get("tools") or []
            use_count = c.get("use_count", 0)
            tag = " (auto)" if c.get("is_auto_detected") else ""
            out_console.print(
                f"  [{_C_SUCCESS}]{name}[/{_C_SUCCESS}]{tag}  "
                f"[{_C_DIM}]used {use_count}×[/{_C_DIM}]"
            )
            if tools:
                out_console.print(f"    [{_C_DIM}]→ {' -> '.join(tools)}[/{_C_DIM}]")
    elif args.action == "detect":
        detected = payload.get("detected") or []
        count = payload.get("count", len(detected))
        out_console.print(f"[bold]Detection complete[/bold]: {count} chain(s) detected")
        for c in detected:
            if isinstance(c, dict):
                name = c.get("name", "?")
                tools = c.get("tools") or []
                out_console.print(
                    f"  [{_C_SUCCESS}]{name}[/{_C_SUCCESS}]  "
                    f"[{_C_DIM}]{' -> '.join(tools)}[/{_C_DIM}]"
                )
            else:
                out_console.print(f"  {c}")
    # cli-ux-02: emit the cold-install "run sync first" hint when the index DB
    # is absent, uniform with status/categories/audit/analytics.
    _maybe_suggest_sync(err_console)
    return 0


# =============================================================================
# Top-level dispatch
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    """Top-level CLI dispatch.

    Default behavior (no args, or explicit `serve`) delegates to the gateway
    server so existing `tool-compass` integrations keep working without
    edits. New subcommands short-circuit before touching gateway.

    FE-A-011 / SD-CLI-005: every subcommand path is wrapped — unexpected
    exceptions produce a single-line ``_print_error`` and exit code 2 (system
    error / usage) instead of leaking a traceback to terminals or log scrapers.

    SD-CLI-006 exit-code audit:
        0   success
        1   expected failure (index missing, tool not found, backend down)
        2   usage error / unexpected internal exception
        130 SIGINT (Unix convention: 128 + signal number)
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    err_console = _make_console(stderr=True, no_color_flag=_no_color(args))

    try:
        if args.command is None or args.command == "serve":
            # FE-W11-007: --http used to be parsed but ignored (the gateway
            # reads PORT from env). Wire it so a value triggers HTTP transport
            # at the requested port; a bare flag falls back to the existing
            # PORT env or 8080. This keeps backward compatibility (existing
            # PORT-only deployments still work) while fixing the documented
            # surface.
            http_val = getattr(args, "http", None) if args.command == "serve" else None
            if http_val is not None:
                # http_val is "" when --http was passed with no value
                # (argparse const). Otherwise it's the string the user gave us.
                port = http_val.strip() if http_val else os.environ.get("PORT", "8080")
                if not port:
                    port = "8080"
                # Validate that we got an integer; reject anything else with a
                # usage error so we don't hand a garbage value to the gateway.
                try:
                    port_int = int(port)
                except ValueError:
                    return _print_error(
                        err_console,
                        f"--http expects an integer port, got: {http_val!r}",
                        hint="Try `tool-compass serve --http 8080`.",
                        exit_code=2,
                    )
                # An int alone isn't enough: '-1', '0', and '99999999' all parse
                # but aren't bindable ports. Enforce the TCP port range so the
                # gateway never gets handed an unbindable value.
                if not (1 <= port_int <= 65535):
                    return _print_error(
                        err_console,
                        f"--http port out of range: {port_int}",
                        hint="Port must be in the valid range 1-65535.",
                        exit_code=2,
                    )
                os.environ["PORT"] = str(port_int)
            # Backward-compat path: legacy behavior was to launch the server.
            from gateway import main as gateway_main

            # gateway.main() calls argparse.parse_args() with no args, so it
            # reads sys.argv[1:]. When we're invoked as `tool-compass serve
            # [--http ...]`, sys.argv still carries those tokens and gateway's
            # strict parser dies with "unrecognized arguments: serve". The port
            # is already handed off via os.environ['PORT'] (gateway reads PORT
            # from env, never from argv), so neutralizing argv loses nothing.
            # Same pattern _cmd_ui uses for ui.main.
            saved_argv = sys.argv
            try:
                sys.argv = ["tool-compass"]
                return gateway_main() or 0
            finally:
                sys.argv = saved_argv
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "search":
            return _cmd_search(args)
        if args.command == "describe":
            return _cmd_describe(args)
        if args.command == "execute":
            return _cmd_execute(args)
        if args.command == "sync":
            return _cmd_sync(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "categories":
            return _cmd_categories(args)
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "analytics":
            return _cmd_analytics(args)
        if args.command == "chains":
            return _cmd_chains(args)
    except KeyboardInterrupt:
        # Standard Unix convention: 128 + SIGINT (2) = 130.
        err_console.print(
            f"\n[{_C_WARN}]Interrupted.[/{_C_WARN}]"
        )
        return 130
    except Exception as e:
        return _print_error(
            err_console,
            f"tool-compass {args.command} failed: {type(e).__name__}: {e}",
            exit_code=2,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
