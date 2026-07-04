"""Focused coverage tests for cli.py (raise cli.py from 45% → ≥85%).

Wave-12 of the v2.3.0 dogfood swarm. cli.py grew +1000 LOC in Wave-11
when six new subcommands (ui, status, categories, audit, analytics, chains)
plus the serve --http port export shipped. The existing
``tests/test_features_v2_2_0.py`` covers the smoke surface; this file
fills the gaps:

- ``_should_color`` matrix (TTY + NO_COLOR + TERM=dumb + --no-color)
- ``_print_error`` / ``_print_warn`` / ``_print_success`` / ``_print_dim``
  including the ``hint=...`` keyword path (Stage C humanization)
- ``_PlainConsole`` fallback and ``_NullStatus`` shim
- ``_make_console`` for stdout + stderr pair
- ``_load_index`` failure path
- ``_no_color`` reflection of the parsed Namespace
- ``_check_index_initialized`` / ``_maybe_suggest_sync``
- ``_dump_json`` / ``_is_error_envelope`` / ``_print_envelope_error``
- The six gateway-handler-backed subcommands in BOTH text and JSON modes
- Error path (envelope, import failure, handler raises)
- ``cmd_search`` (text mode, empty results, JSON mode, load-failure,
  ConnectionError) — only JSON path was previously covered
- ``cmd_describe`` (text mode, malformed JSON, suggestions, DB missing)
- ``cmd_sync`` (success text, error envelope, FileNotFoundError, no-backends)
- ``cmd_doctor`` (text mode, FileNotFoundError, JSONDecodeError, generic exc)
- ``cmd_ui`` (import error path)
- ``main`` (KeyboardInterrupt → 130, unknown command → help+2,
  serve --http malformed, exception handler → 2)

The tests mock at the MCP handler boundary
(``gateway.compass_*``) and at ``cli._load_index`` for index-backed paths;
no real Ollama / HNSW / SQLite is touched. ``capsys`` captures stdout +
stderr for content assertions, with markup stripping verified.

Style notes:
- Each subcommand has at least 4 tests (text, JSON, error, NO_COLOR)
- JSON output is parsed with ``json.loads`` to verify stable shape
- Exit codes asserted explicitly per the SD-CLI-006 audit
- Rich markup ([green]✓[/green] etc) is stripped when no_color=True,
  but rich still emits the ✓ glyph itself — assertions match the glyph
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli


REPO_ROOT = Path(__file__).parent.parent


# =============================================================================
# _should_color matrix
# =============================================================================


class _FakeTTY:
    """Stream stub that pretends to be a TTY."""

    def __init__(self, is_tty: bool = True):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class _RaisingTTY:
    """isatty() raises — _should_color must swallow this defensively."""

    def isatty(self) -> bool:
        raise OSError("simulated terminal check failure")


class TestShouldColor:
    """Exercise every branch of cli._should_color."""

    def test_no_color_flag_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        # Even with a TTY the explicit flag wins.
        assert cli._should_color(_FakeTTY(True), no_color_flag=True) is False

    def test_no_color_env_disables(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.delenv("TERM", raising=False)
        assert cli._should_color(_FakeTTY(True)) is False

    def test_term_dumb_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert cli._should_color(_FakeTTY(True)) is False

    def test_non_tty_disables(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        assert cli._should_color(_FakeTTY(False)) is False

    def test_stream_missing_isatty(self, monkeypatch):
        """A capsys-style buffered stream lacks .isatty entirely."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        class _NoIsatty:
            pass

        assert cli._should_color(_NoIsatty()) is False

    def test_isatty_raises_returns_false(self, monkeypatch):
        """Defensive: a stream whose .isatty raises should not crash us."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        assert cli._should_color(_RaisingTTY()) is False

    def test_real_tty_allows_color(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        assert cli._should_color(_FakeTTY(True)) is True

    def test_empty_no_color_env_still_disables(self, monkeypatch):
        """no-color.org spec: any non-empty value disables. Empty falls through."""
        monkeypatch.setenv("NO_COLOR", "")
        monkeypatch.delenv("TERM", raising=False)
        # Empty string is falsy — color is allowed on a TTY.
        assert cli._should_color(_FakeTTY(True)) is True

    def test_isatty_non_callable(self, monkeypatch):
        """If isatty is an attribute but not callable, treat as non-TTY."""
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)

        class _BadStream:
            isatty = "not a function"

        assert cli._should_color(_BadStream()) is False


# =============================================================================
# _make_console + _PlainConsole + _NullStatus
# =============================================================================


class TestMakeConsole:
    """Cover console factory and the rich-missing fallback shim."""

    def test_make_console_stdout(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        c = cli._make_console(stderr=False)
        # Either a Rich Console or _PlainConsole — both have .print
        assert hasattr(c, "print")

    def test_make_console_stderr(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        c = cli._make_console(stderr=True)
        assert hasattr(c, "print")

    def test_make_console_no_color(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        # Explicit no_color_flag should still produce a usable console.
        c = cli._make_console(stderr=False, no_color_flag=True)
        assert hasattr(c, "print")


class TestPlainConsole:
    """_PlainConsole + _NullStatus exercise the rich-missing fallback path."""

    def test_plain_console_prints_to_stdout(self, capsys):
        pc = cli._PlainConsole(stderr=False)
        pc.print("hello world")
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_plain_console_prints_to_stderr(self, capsys):
        pc = cli._PlainConsole(stderr=True)
        pc.print("oops")
        err = capsys.readouterr().err
        assert "oops" in err

    def test_plain_console_strips_rich_markup(self, capsys):
        pc = cli._PlainConsole()
        pc.print("[green]ok[/green] tag stripped")
        out = capsys.readouterr().out
        assert "[green]" not in out
        assert "ok" in out
        assert "tag stripped" in out


# =============================================================================
# _print_error / _print_warn / _print_success / _print_dim
# =============================================================================


class TestPrintHelpers:
    """The four colored-line helpers + the hint keyword."""

    def test_print_error_returns_exit_code(self, capsys):
        # _PlainConsole strips markup so we can assert on plain text.
        console = cli._PlainConsole(stderr=True)
        rc = cli._print_error(console, "boom")
        assert rc == 1
        err = capsys.readouterr().err
        assert "boom" in err
        # The ✗ glyph should always be present.
        assert "✗" in err  # ✗

    def test_print_error_with_hint(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_error(console, "missing config", hint="try again")
        err = capsys.readouterr().err
        assert "missing config" in err
        assert "try again" in err

    def test_print_error_custom_exit_code(self):
        console = cli._PlainConsole(stderr=True)
        rc = cli._print_error(console, "usage", exit_code=2)
        assert rc == 2

    def test_print_warn(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_warn(console, "watch out", hint="be careful")
        err = capsys.readouterr().err
        assert "watch out" in err
        assert "be careful" in err
        assert "⚠" in err  # ⚠

    def test_print_warn_no_hint(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_warn(console, "watch out")
        err = capsys.readouterr().err
        assert "watch out" in err

    def test_print_success(self, capsys):
        console = cli._PlainConsole()
        cli._print_success(console, "all good")
        out = capsys.readouterr().out
        assert "all good" in out
        assert "✓" in out  # ✓

    def test_print_dim(self, capsys):
        console = cli._PlainConsole()
        cli._print_dim(console, "subtle hint")
        out = capsys.readouterr().out
        assert "subtle hint" in out


# =============================================================================
# Parser construction
# =============================================================================


class TestBuildParser:
    """Smoke tests for the argparse tree shape."""

    def test_parser_has_version(self):
        parser = cli._build_parser()
        # --version action raises SystemExit on parse.
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_has_no_color(self):
        parser = cli._build_parser()
        args = parser.parse_args(["--no-color", "doctor"])
        assert args.no_color is True
        assert args.command == "doctor"

    def test_parser_no_color_default_false(self):
        parser = cli._build_parser()
        args = parser.parse_args(["doctor"])
        assert args.no_color is False

    def test_parser_search_top_default(self):
        parser = cli._build_parser()
        args = parser.parse_args(["search", "intent"])
        assert args.top == 5
        assert args.intent == "intent"

    def test_parser_doctor_text_flag(self):
        parser = cli._build_parser()
        args = parser.parse_args(["doctor", "--text"])
        assert args.text is True

    def test_parser_chains_action_choices(self):
        parser = cli._build_parser()
        args = parser.parse_args(["chains", "--action", "detect"])
        assert args.action == "detect"

    def test_parser_chains_invalid_action_exits_2(self, capsys):
        parser = cli._build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["chains", "--action", "bogus"])
        assert exc.value.code == 2

    def test_parser_audit_timeframe_default(self):
        parser = cli._build_parser()
        args = parser.parse_args(["audit"])
        assert args.timeframe == "24h"
        assert args.include_tools is False

    def test_parser_analytics_no_failures(self):
        parser = cli._build_parser()
        args = parser.parse_args(["analytics", "--no-failures"])
        assert args.no_failures is True

    def test_parser_serve_http_with_value(self):
        parser = cli._build_parser()
        args = parser.parse_args(["serve", "--http", "9999"])
        assert args.http == "9999"

    def test_parser_serve_http_no_value(self):
        parser = cli._build_parser()
        args = parser.parse_args(["serve", "--http"])
        # argparse const for nargs="?" with no value is "" per the source.
        assert args.http == ""

    def test_parser_serve_default_http_none(self):
        parser = cli._build_parser()
        args = parser.parse_args(["serve"])
        assert args.http is None

    def test_parser_ui_defaults(self):
        parser = cli._build_parser()
        args = parser.parse_args(["ui"])
        assert args.port == 7860
        assert args.host == "127.0.0.1"
        assert args.share is False
        assert args.auth is None


# =============================================================================
# _no_color helper + _load_index failure path
# =============================================================================


class TestNoColorHelper:
    def test_no_color_true(self):
        ns = SimpleNamespace(no_color=True)
        assert cli._no_color(ns) is True

    def test_no_color_false(self):
        ns = SimpleNamespace(no_color=False)
        assert cli._no_color(ns) is False

    def test_no_color_missing_attr(self):
        """argparse Namespace without no_color (defensive default)."""
        ns = SimpleNamespace()
        assert cli._no_color(ns) is False


class TestLoadIndex:
    """_load_index returns None when CompassIndex.load_index() is False."""

    def test_load_index_returns_none_when_load_fails(self, monkeypatch):
        import indexer

        class _StubIndex:
            def load_index(self):
                return False

        monkeypatch.setattr(indexer, "CompassIndex", lambda: _StubIndex())
        assert cli._load_index() is None

    def test_load_index_returns_index_on_success(self, monkeypatch):
        import indexer

        object()

        class _StubIndex:
            def load_index(self):
                return True

        # Patch the constructor so it returns our stub; CompassIndex() in
        # _load_index returns the stub.
        stub = _StubIndex()
        monkeypatch.setattr(indexer, "CompassIndex", lambda: stub)
        result = cli._load_index()
        assert result is stub


# =============================================================================
# Internal gateway-backed helpers
# =============================================================================


class TestEnvelopeHelpers:
    """_is_error_envelope + _print_envelope_error + _dump_json shape checks."""

    def test_is_error_envelope_true(self):
        assert cli._is_error_envelope(
            {"error": {"code": "x", "title": "Y"}}
        ) is True

    def test_is_error_envelope_no_code(self):
        assert cli._is_error_envelope({"error": {"title": "missing code"}}) is False

    def test_is_error_envelope_non_dict_error(self):
        assert cli._is_error_envelope({"error": "string not dict"}) is False

    def test_is_error_envelope_non_dict_payload(self):
        assert cli._is_error_envelope("not even a dict") is False
        assert cli._is_error_envelope(None) is False
        assert cli._is_error_envelope([1, 2, 3]) is False

    def test_is_error_envelope_no_error_key(self):
        assert cli._is_error_envelope({"ok": True}) is False

    def test_print_envelope_error_uses_title(self, capsys):
        console = cli._PlainConsole(stderr=True)
        rc = cli._print_envelope_error(
            console,
            {"error": {"code": "x", "title": "Boom title"}},
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "Boom title" in err

    def test_print_envelope_error_falls_back_to_detail(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_envelope_error(
            console,
            {"error": {"code": "x", "detail": "Detail fallback"}},
        )
        err = capsys.readouterr().err
        assert "Detail fallback" in err

    def test_print_envelope_error_default_message(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_envelope_error(console, {"error": {"code": "x"}})
        err = capsys.readouterr().err
        assert "Operation failed" in err

    def test_print_envelope_error_uses_first_suggestion(self, capsys):
        console = cli._PlainConsole(stderr=True)
        cli._print_envelope_error(
            console,
            {
                "error": {
                    "code": "x",
                    "title": "Boom",
                    "suggestions": ["do this first", "or this"],
                }
            },
        )
        err = capsys.readouterr().err
        assert "do this first" in err

    def test_dump_json_returns_zero(self, capsys):
        rc = cli._dump_json({"k": "v", "n": 1})
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed == {"k": "v", "n": 1}


class TestCheckIndexInitialized:
    """The on-disk DB existence check + the suggestion hint."""

    def test_check_index_true_when_db_exists(self, tmp_path, monkeypatch):
        fake_db = tmp_path / "tools.db"
        fake_db.write_text("not really sqlite but exists")
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", fake_db)
        assert cli._check_index_initialized() is True

    def test_check_index_false_when_db_missing(self, tmp_path, monkeypatch):
        fake_db = tmp_path / "nope.db"
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", fake_db)
        assert cli._check_index_initialized() is False

    def test_maybe_suggest_sync_emits_when_missing(self, tmp_path, monkeypatch, capsys):
        fake_db = tmp_path / "nope.db"
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", fake_db)
        console = cli._PlainConsole(stderr=True)
        cli._maybe_suggest_sync(console)
        err = capsys.readouterr().err
        assert "sync" in err.lower()

    def test_maybe_suggest_sync_silent_when_present(self, tmp_path, monkeypatch, capsys):
        fake_db = tmp_path / "tools.db"
        fake_db.write_text("ok")
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", fake_db)
        console = cli._PlainConsole(stderr=True)
        cli._maybe_suggest_sync(console)
        err = capsys.readouterr().err
        # No output expected when index is present.
        assert err == ""


# =============================================================================
# Patch helper — replace gateway.compass_* with an async stub returning payload
# =============================================================================


def _patch_gateway(monkeypatch, name: str, payload):
    """Replace gateway.<name> with an async function returning ``payload``."""
    import gateway

    async def fake(*args, **kwargs):
        return payload

    monkeypatch.setattr(gateway, name, fake)
    return fake


def _patch_gateway_raises(monkeypatch, name: str, exc):
    """Replace gateway.<name> with an async function that raises ``exc``."""
    import gateway

    async def fake(*args, **kwargs):
        raise exc

    monkeypatch.setattr(gateway, name, fake)


# =============================================================================
# cmd_status — text + JSON + error envelope + handler-raises + NO_COLOR
# =============================================================================


_STATUS_PAYLOAD = {
    "index": {
        "total_tools": 12,
        "by_category": {"file": 5, "ai": 3},
        "by_server": {"bridge": 9, "comfy": 3},
    },
    "backends": {
        "connected_backends": ["bridge"],
        "configured_backends": ["bridge", "comfy"],
    },
    "health": {
        "ollama_available": True,
        "index_available": True,
        "degraded_mode": False,
    },
    "sync": {"last_sync_at": "2026-05-15T01:00:00Z"},
}


class TestCmdStatus:
    def test_status_text(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_status", _STATUS_PAYLOAD)
        rc = cli.main(["status"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "12" in out  # total_tools
        assert "bridge" in out  # by_server

    def test_status_json(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_status", _STATUS_PAYLOAD)
        rc = cli.main(["status", "--json"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        parsed = json.loads(out)
        # Stable key order check — the test asserts presence rather than
        # rely on dict iteration order across Python versions.
        assert set(parsed.keys()) >= {"index", "backends", "health", "sync"}
        assert parsed["index"]["total_tools"] == 12

    def test_status_no_color_flag(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_status", _STATUS_PAYLOAD)
        rc = cli.main(["--no-color", "status"])
        out = capsys.readouterr().out
        # No-color path should not contain ANSI escape codes (rich strips them).
        assert "\x1b[" not in out
        assert rc == 0

    def test_status_error_envelope(self, monkeypatch, capsys):
        payload = {
            "error": {
                "code": "unavailable",
                "title": "Status unavailable",
                "suggestions": ["Try later"],
            }
        }
        _patch_gateway(monkeypatch, "compass_status", payload)
        rc = cli.main(["status"])
        err = capsys.readouterr().err
        assert "Status unavailable" in err
        assert rc == 1

    def test_status_handler_raises_returns_1(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_status", RuntimeError("boom"))
        rc = cli.main(["status"])
        err = capsys.readouterr().err
        assert "compass_status failed" in err
        assert rc == 1

    def test_status_no_backends_field(self, monkeypatch, capsys):
        """When no backends key present, the connected/configured branch is skipped."""
        payload = {
            "index": {"total_tools": 0, "by_server": {}},
            "health": {"ollama_available": False, "index_available": True},
        }
        _patch_gateway(monkeypatch, "compass_status", payload)
        rc = cli.main(["status"])
        out = capsys.readouterr().out
        # "ollama unreachable" warning should render
        assert "ollama" in out.lower()
        assert rc == 0

    def test_status_degraded_mode_warned(self, monkeypatch, capsys):
        payload = {
            "index": {"total_tools": 0, "by_server": {}},
            "health": {
                "ollama_available": True,
                "index_available": False,
                "degraded_mode": True,
            },
            "backends": {"connected_backends": [], "configured_backends": ["a"]},
        }
        _patch_gateway(monkeypatch, "compass_status", payload)
        rc = cli.main(["status"])
        out = capsys.readouterr().out
        assert "degraded" in out.lower() or "DEGRADED" in out
        assert rc == 0


# =============================================================================
# cmd_categories — text + JSON + envelope + empty
# =============================================================================


class TestCmdCategories:
    def test_categories_text_sorted_by_count_desc(self, monkeypatch, capsys):
        payload = {
            "categories": {"file": 10, "ai": 5, "git": 7},
            "total_tools": 22,
        }
        _patch_gateway(monkeypatch, "compass_categories", payload)
        rc = cli.main(["categories"])
        out = capsys.readouterr().out
        assert rc == 0
        # All present
        assert "file" in out and "ai" in out and "git" in out
        # "file" (count=10) should appear before "git" (count=7) — sort order desc.
        idx_file = out.find("file")
        idx_git = out.find("git")
        idx_ai = out.find("ai")
        assert idx_file < idx_git < idx_ai

    def test_categories_json(self, monkeypatch, capsys):
        payload = {"categories": {"x": 1}, "total_tools": 1}
        _patch_gateway(monkeypatch, "compass_categories", payload)
        rc = cli.main(["categories", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["categories"] == {"x": 1}
        assert rc == 0

    def test_categories_empty(self, monkeypatch, capsys, tmp_path):
        """Empty categories triggers the maybe-suggest-sync hint."""
        # Point the indexer DB path at a non-existent file so the hint emits.
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", tmp_path / "missing.db")

        payload = {"categories": {}, "total_tools": 0}
        _patch_gateway(monkeypatch, "compass_categories", payload)
        rc = cli.main(["categories"])
        out = capsys.readouterr().out
        capsys.readouterr().err
        assert rc == 0
        assert "no categories" in out.lower() or "empty" in out.lower()

    def test_categories_error_envelope(self, monkeypatch, capsys):
        payload = {"error": {"code": "fail", "title": "Categories unavailable"}}
        _patch_gateway(monkeypatch, "compass_categories", payload)
        rc = cli.main(["categories"])
        err = capsys.readouterr().err
        assert "Categories unavailable" in err
        assert rc == 1

    def test_categories_handler_raises(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_categories", RuntimeError("kapow"))
        rc = cli.main(["categories"])
        err = capsys.readouterr().err
        assert "compass_categories failed" in err
        assert rc == 1

    def test_categories_no_color(self, monkeypatch, capsys):
        payload = {"categories": {"file": 3}, "total_tools": 3}
        _patch_gateway(monkeypatch, "compass_categories", payload)
        rc = cli.main(["--no-color", "categories"])
        out = capsys.readouterr().out
        # No ANSI escape codes
        assert "\x1b[" not in out
        assert "file" in out
        assert rc == 0


# =============================================================================
# cmd_audit — text + JSON + envelope + include_tools + handler-raises
# =============================================================================


_AUDIT_PAYLOAD = {
    "system": {"version": "2.3.0", "total_tools": 5},
    "categories": {"file": 3, "ai": 2},
    "servers": {"bridge": 5},
    "backends": {
        "connected_backends": ["bridge"],
        "configured_backends": ["bridge"],
    },
    "hot_cache": {"size": 2, "tools": ["a", "b"]},
    "chains": {"total": 4, "cached": 1},
}


class TestCmdAudit:
    def test_audit_text(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_audit", _AUDIT_PAYLOAD)
        rc = cli.main(["audit"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "2.3.0" in out
        assert "bridge" in out

    def test_audit_json(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_audit", _AUDIT_PAYLOAD)
        rc = cli.main(["audit", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["system"]["version"] == "2.3.0"
        assert rc == 0

    def test_audit_timeframe_arg(self, monkeypatch, capsys):
        seen = {}

        async def fake(*args, **kwargs):
            seen.update(kwargs)
            return _AUDIT_PAYLOAD

        import gateway

        monkeypatch.setattr(gateway, "compass_audit", fake)
        rc = cli.main(["audit", "--timeframe", "7d"])
        assert rc == 0
        assert seen.get("timeframe") == "7d"

    def test_audit_include_tools(self, monkeypatch, capsys):
        payload = dict(_AUDIT_PAYLOAD)
        payload["tools"] = [
            {"name": "tool1"},
            {"name": "tool2"},
            {"name": "tool3"},
        ]

        seen = {}

        async def fake(*args, **kwargs):
            seen.update(kwargs)
            return payload

        import gateway

        monkeypatch.setattr(gateway, "compass_audit", fake)
        rc = cli.main(["audit", "--include-tools"])
        out = capsys.readouterr().out
        assert rc == 0
        assert seen.get("include_tools") is True
        assert "tool1" in out and "tool2" in out

    def test_audit_envelope(self, monkeypatch, capsys):
        _patch_gateway(
            monkeypatch,
            "compass_audit",
            {"error": {"code": "x", "title": "Audit blew up"}},
        )
        rc = cli.main(["audit"])
        err = capsys.readouterr().err
        assert "Audit blew up" in err
        assert rc == 1

    def test_audit_handler_raises(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_audit", ValueError("nope"))
        rc = cli.main(["audit"])
        err = capsys.readouterr().err
        assert "compass_audit failed" in err
        assert rc == 1

    def test_audit_backends_error_dict(self, monkeypatch, capsys):
        """If backends comes back as a dict with 'error' key, we emit a warn."""
        payload = dict(_AUDIT_PAYLOAD)
        payload["backends"] = {"error": "backend reader unavailable"}
        _patch_gateway(monkeypatch, "compass_audit", payload)
        rc = cli.main(["audit"])
        out = capsys.readouterr().out
        assert "backends error" in out
        assert rc == 0

    def test_audit_no_color(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_audit", _AUDIT_PAYLOAD)
        rc = cli.main(["--no-color", "audit"])
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert rc == 0


# =============================================================================
# cmd_analytics — text + JSON + envelope + no_failures + top_tools render
# =============================================================================


class TestCmdAnalytics:
    def test_analytics_text_renders_top_tools(self, monkeypatch, capsys):
        payload = {
            "top_tools": [
                {"tool_name": "bridge:read_file", "call_count": 42},
                {"tool_name": "comfy:gen", "call_count": 10},
            ],
            "summary": {"total_calls": 52},
            "failures": [{"tool_name": "x"}],
        }
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics"])
        out = capsys.readouterr().out
        capsys.readouterr().err
        assert rc == 0
        assert "bridge:read_file" in out
        assert "42" in out
        assert "52" in out  # total

    def test_analytics_top_tools_hot_tools_alias(self, monkeypatch, capsys):
        """gateway may return `hot_tools` instead of `top_tools`."""
        payload = {
            "hot_tools": [{"name": "x", "count": 5}],
            "summary": {"total_calls": 5},
        }
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics"])
        out = capsys.readouterr().out
        assert "x" in out and "5" in out
        assert rc == 0

    def test_analytics_json(self, monkeypatch, capsys):
        payload = {"summary": {"total_calls": 0}}
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["summary"]["total_calls"] == 0
        assert rc == 0

    def test_analytics_no_failures_flag_forwarded(self, monkeypatch, capsys):
        seen = {}

        async def fake(*args, **kwargs):
            seen.update(kwargs)
            return {"summary": {"total_calls": 0}}

        import gateway

        monkeypatch.setattr(gateway, "compass_analytics", fake)
        rc = cli.main(["analytics", "--no-failures"])
        assert rc == 0
        assert seen.get("include_failures") is False

    def test_analytics_handler_raises(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_analytics", OSError("disk"))
        rc = cli.main(["analytics"])
        err = capsys.readouterr().err
        assert "compass_analytics failed" in err
        assert rc == 1

    def test_analytics_envelope(self, monkeypatch, capsys):
        payload = {"error": {"code": "disabled", "title": "Analytics is disabled"}}
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics"])
        err = capsys.readouterr().err
        assert "Analytics is disabled" in err
        assert rc == 1

    def test_analytics_top_tools_non_dict_entries(self, monkeypatch, capsys):
        """If top_tools list has non-dict entries, render as string."""
        payload = {"top_tools": ["raw-string-tool", "another"]}
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics"])
        out = capsys.readouterr().out
        assert "raw-string-tool" in out
        assert rc == 0

    def test_analytics_failures_warn(self, monkeypatch, capsys):
        """include_failures=True surfaces failure count as warn."""
        payload = {
            "summary": {"total_calls": 1},
            "failures": [{"tool_name": "x"}, {"tool_name": "y"}],
        }
        _patch_gateway(monkeypatch, "compass_analytics", payload)
        rc = cli.main(["analytics"])
        err = capsys.readouterr().err
        assert "2 failure" in err
        assert rc == 0

    def test_analytics_no_color(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_analytics", {"summary": {"total_calls": 0}})
        rc = cli.main(["--no-color", "analytics"])
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert rc == 0


# =============================================================================
# cmd_chains — list + detect + envelope + empty + handler-raises
# =============================================================================


class TestCmdChains:
    def test_chains_list_text(self, monkeypatch, capsys):
        payload = {
            "chains": [
                {
                    "name": "read-then-write",
                    "tools": ["fs:read_file", "fs:write_file"],
                    "use_count": 3,
                    "is_auto_detected": True,
                }
            ]
        }
        _patch_gateway(monkeypatch, "compass_chains", payload)
        rc = cli.main(["chains"])
        out = capsys.readouterr().out
        assert "read-then-write" in out
        assert "fs:read_file" in out
        # is_auto_detected flag should produce " (auto)" tag.
        assert "auto" in out
        assert rc == 0

    def test_chains_list_empty(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_chains", {"chains": []})
        rc = cli.main(["chains"])
        out = capsys.readouterr().out
        assert "no chains" in out.lower() or "0 total" in out
        assert rc == 0

    def test_chains_detect_text(self, monkeypatch, capsys):
        payload = {
            "detected": [
                {"name": "auto-1", "tools": ["a", "b"]},
                {"name": "auto-2", "tools": ["c"]},
            ],
            "count": 2,
        }
        _patch_gateway(monkeypatch, "compass_chains", payload)
        rc = cli.main(["chains", "--action", "detect"])
        out = capsys.readouterr().out
        assert "auto-1" in out and "auto-2" in out
        assert rc == 0

    def test_chains_detect_string_entries(self, monkeypatch, capsys):
        """If detected returns plain strings (not dicts), render them straight."""
        payload = {"detected": ["chain-string-1", "chain-string-2"], "count": 2}
        _patch_gateway(monkeypatch, "compass_chains", payload)
        rc = cli.main(["chains", "--action", "detect"])
        out = capsys.readouterr().out
        assert "chain-string-1" in out
        assert rc == 0

    def test_chains_json(self, monkeypatch, capsys):
        payload = {"chains": [{"name": "x", "tools": ["a"]}]}
        _patch_gateway(monkeypatch, "compass_chains", payload)
        rc = cli.main(["chains", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["chains"][0]["name"] == "x"
        assert rc == 0

    def test_chains_envelope(self, monkeypatch, capsys):
        _patch_gateway(
            monkeypatch,
            "compass_chains",
            {"error": {"code": "fail", "title": "Chains broken"}},
        )
        rc = cli.main(["chains"])
        err = capsys.readouterr().err
        assert "Chains broken" in err
        assert rc == 1

    def test_chains_handler_raises(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_chains", RuntimeError("crash"))
        rc = cli.main(["chains"])
        err = capsys.readouterr().err
        assert "compass_chains failed" in err
        assert rc == 1

    def test_chains_no_color(self, monkeypatch, capsys):
        _patch_gateway(monkeypatch, "compass_chains", {"chains": []})
        rc = cli.main(["--no-color", "chains"])
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert rc == 0


# =============================================================================
# cmd_ui — import error + auth env + dispatch
# =============================================================================


class TestCmdUI:
    def test_ui_import_error_returns_1(self, monkeypatch, capsys):
        """If ui module raises ImportError, CLI prints hint + exits 1."""
        # Remove cached ui module so the import inside _cmd_ui fails.
        monkeypatch.delitem(sys.modules, "ui", raising=False)

        # Force ImportError on `import ui`.
        original_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict
        ) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ui":
                raise ImportError("simulated: gradio not installed")
            return original_import(name, *args, **kwargs)

        # Patch the builtins.__import__ used inside _cmd_ui.
        monkeypatch.setattr("builtins.__import__", fake_import)

        try:
            rc = cli.main(["ui"])
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        err = capsys.readouterr().err
        assert "UI extras not installed" in err
        assert "pip install" in err
        assert rc == 1

    def test_ui_dispatch_no_share(self, monkeypatch, capsys):
        """Without --share, the forwarded argv must NOT contain --share."""
        captured_argv: list[str] = []

        def fake_ui_main():
            captured_argv.extend(sys.argv)
            return 0

        fake_ui = type(sys)("ui")
        fake_ui.main = fake_ui_main
        monkeypatch.setitem(sys.modules, "ui", fake_ui)

        rc = cli.main(["ui", "--port", "8888"])
        assert rc == 0
        assert "--share" not in captured_argv


# =============================================================================
# cmd_search — text mode + empty results + ConnectionError + load-failure
# =============================================================================


def _stub_index(results):
    """Build a fake CompassIndex that returns ``results`` from .search()."""

    class _Index:
        async def search(self, query, top_k=5):
            return results

    return _Index()


def _stub_index_with_db(rows, *, search_raises=None):
    """Fake CompassIndex whose ``.search`` raises and whose ``.db`` is a real
    in-memory SQLite ``tools`` table.

    cli-ui-001: exercises the Option-A lexical fallback path in ``_cmd_search``.
    When ``.search`` raises the given embedder-style exception, ``_cmd_search``
    is expected to fall back to ``gateway._lexical_search_fallback`` scanning
    this ``.db``. ``rows`` is a list of (name, description, category, server).
    """
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE tools (name TEXT, description TEXT, category TEXT, server TEXT)"
    )
    for name, desc, cat, srv in rows:
        db.execute(
            "INSERT INTO tools (name, description, category, server) VALUES (?, ?, ?, ?)",
            (name, desc, cat, srv),
        )
    db.commit()

    class _Index:
        def __init__(self):
            self.db = db

        async def search(self, query, top_k=5, **kwargs):
            raise search_raises or RuntimeError("Ollama circuit breaker open")

    return _Index()


def _result(rank: int, name: str, score: float, desc: str = "desc"):
    """Build a result object matching cli._cmd_search's expected attrs."""
    return SimpleNamespace(
        tool=SimpleNamespace(
            name=name,
            category="cat",
            server="srv",
            description=desc,
        ),
        score=score,
        rank=rank,
    )


class TestCmdSearchExtended:
    """Extend the existing search smoke test with text + edge paths."""

    def test_search_text_mode(self, monkeypatch, capsys):
        results = [
            _result(1, "bridge:read_file", 0.92, "read a file"),
            _result(2, "bridge:write_file", 0.51, "write a file"),
        ]
        monkeypatch.setattr(cli, "_load_index", lambda: _stub_index(results))
        rc = cli.main(["search", "read"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "bridge:read_file" in out
        assert "0.920" in out or "0.92" in out

    def test_search_long_name_truncated(self, monkeypatch, capsys):
        """Tool names over 40 chars are truncated with ellipsis."""
        long_name = "a" * 50 + ":tool"
        results = [_result(1, long_name, 0.5, "desc")]
        monkeypatch.setattr(cli, "_load_index", lambda: _stub_index(results))
        rc = cli.main(["search", "x"])
        out = capsys.readouterr().out
        assert "..." in out
        assert rc == 0

    def test_search_long_description_truncated(self, monkeypatch, capsys):
        """Descriptions over 60 chars are truncated with ellipsis."""
        long_desc = "x" * 80
        results = [_result(1, "n", 0.5, long_desc)]
        monkeypatch.setattr(cli, "_load_index", lambda: _stub_index(results))
        rc = cli.main(["search", "q"])
        out = capsys.readouterr().out
        assert "..." in out
        assert rc == 0

    def test_search_load_index_failure_emits_hint(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_load_index", lambda: None)
        rc = cli.main(["search", "foo"])
        err = capsys.readouterr().err
        assert "Index not available" in err
        assert "sync" in err.lower()
        assert rc == 1

    def test_search_connection_error_falls_back_to_keyword(self, monkeypatch, capsys):
        """cli-ui-001: ConnectionError from the embedder now degrades to the
        lexical keyword fallback (Option A) rather than exiting 1. The help text
        (cli.py) promises "falls back to keyword matching" — this keeps it true.
        """
        idx = _stub_index_with_db(
            [("bridge:read_file", "read a file", "file", "bridge")],
            search_raises=ConnectionError("ollama unreachable"),
        )
        monkeypatch.setattr(cli, "_load_index", lambda: idx)
        rc = cli.main(["search", "read"])
        captured = capsys.readouterr()
        # Keyword result surfaced, exit 0 (a degraded-but-served response).
        assert rc == 0
        assert "bridge:read_file" in captured.out
        # A keyword/degraded notice is emitted (adjacent to results).
        assert "keyword" in (captured.out + captured.err).lower()

    def test_search_os_error_falls_back_to_keyword(self, monkeypatch, capsys):
        """cli-ui-001: OSError from the embedder also degrades to keyword
        results + exit 0 (was exit 1 with a dead hint before the fix)."""
        idx = _stub_index_with_db(
            [("bridge:read_file", "read a file", "file", "bridge")],
            search_raises=OSError("disk gone"),
        )
        monkeypatch.setattr(cli, "_load_index", lambda: idx)
        rc = cli.main(["search", "read"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "bridge:read_file" in captured.out

    def test_search_breaker_open_falls_back_to_keyword(self, monkeypatch, capsys):
        """cli-ui-001 core regression: the embedder raises
        ``RuntimeError("Ollama circuit breaker open")`` when the breaker is open
        (embedder.py). RuntimeError does NOT subclass ConnectionError/OSError, so
        before the fix it fell straight through the ``except (ConnectionError,
        OSError)`` guard, hit main()'s catch-all, and exited 2 with a generic
        message and no actionable hint — while the help text promised a keyword
        fallback that never happened. Option A wires the real fallback.
        """
        idx = _stub_index_with_db(
            [
                ("bridge:read_file", "read a file", "file", "bridge"),
                ("comfy:generate", "make an image", "ai", "comfy"),
            ],
            search_raises=RuntimeError("Ollama circuit breaker open"),
        )
        monkeypatch.setattr(cli, "_load_index", lambda: idx)
        rc = cli.main(["search", "read"])
        captured = capsys.readouterr()
        # Degraded-but-served: exit 0, keyword result present, notice shown.
        assert rc == 0, (
            "breaker-open must degrade to keyword fallback (exit 0), not "
            "exit 2 via the catch-all"
        )
        assert "bridge:read_file" in captured.out
        assert "keyword" in (captured.out + captured.err).lower()

    def test_search_transport_error_falls_back_to_keyword(self, monkeypatch, capsys):
        """cli-ui-001: httpx.TransportError (connect/timeout re-raised by the
        embedder) also degrades to the keyword fallback + exit 0. Like
        RuntimeError, httpx.TransportError is not a ConnectionError/OSError
        subclass, so the old narrow guard missed it."""
        import httpx

        idx = _stub_index_with_db(
            [("bridge:read_file", "read a file", "file", "bridge")],
            search_raises=httpx.TransportError("connect failed"),
        )
        monkeypatch.setattr(cli, "_load_index", lambda: idx)
        rc = cli.main(["search", "read"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "bridge:read_file" in captured.out

    def test_search_fallback_no_matches_is_empty_not_error(self, monkeypatch, capsys):
        """cli-ui-001 corollary: when the embedder is down AND the lexical scan
        finds nothing, the CLI shows the standard no-match message at exit 0 —
        it does not resurrect the old "Search failed" exit-1 path."""
        idx = _stub_index_with_db(
            [("bridge:read_file", "read a file", "file", "bridge")],
            search_raises=RuntimeError("Ollama circuit breaker open"),
        )
        monkeypatch.setattr(cli, "_load_index", lambda: idx)
        rc = cli.main(["search", "zzz_no_such_tool_xyz"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "Search failed" not in captured.err
        assert (
            "No tools matched" in captured.out
            or "no tools" in captured.out.lower()
        )

    def test_search_help_text_promises_keyword_fallback(self):
        """cli-ui-001: the search subcommand help must still promise the keyword
        fallback — and now the behavior actually delivers it (Option A). This
        guards against the help/behavior drift the finding flagged."""
        parser = cli._build_parser()
        # Locate the `search` subparser's epilog via the subparsers action.
        search_epilog = None
        for action in parser._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict) and "search" in choices:
                search_epilog = choices["search"].epilog or ""
                break
        assert search_epilog is not None
        assert "keyword" in search_epilog.lower()

    def test_search_empty_results(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_load_index", lambda: _stub_index([]))
        rc = cli.main(["search", "no match"])
        out = capsys.readouterr().out
        capsys.readouterr().err
        # Empty results = exit 0 with a "no tools matched" message.
        assert rc == 0
        assert "No tools matched" in out or "no match" in out.lower()


# =============================================================================
# cmd_describe — text + JSON + DB missing + malformed JSON + suggestions
# =============================================================================


def _setup_describe_db(tmp_path, monkeypatch, rows):
    """Create a fresh tools.db with the given rows."""
    import sqlite3

    db_path = tmp_path / "tools.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE tools (
            name TEXT PRIMARY KEY,
            description TEXT,
            category TEXT,
            server TEXT,
            parameters TEXT,
            examples TEXT,
            is_core INTEGER
        )
        """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO tools VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["name"],
                row.get("description", ""),
                row.get("category", "cat"),
                row.get("server", "srv"),
                row.get("parameters", "{}"),
                row.get("examples", "[]"),
                int(row.get("is_core", 0)),
            ),
        )
    conn.commit()
    conn.close()

    import indexer

    monkeypatch.setattr(indexer, "SQLITE_DB_PATH", db_path)
    return db_path


class TestCmdDescribeExtended:
    def test_describe_text_mode(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {
                    "name": "bridge:read_file",
                    "description": "Read a file",
                    "category": "file",
                    "server": "bridge",
                    "parameters": json.dumps({"filepath": "str"}),
                    "examples": json.dumps(["read file"]),
                    "is_core": 1,
                }
            ],
        )
        rc = cli.main(["describe", "bridge:read_file"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "bridge:read_file" in out
        assert "Read a file" in out
        assert "filepath" in out
        assert "read file" in out  # examples
        # is_core=1 emits a "**Core:** yes" line.
        assert "Core" in out

    def test_describe_json_mode(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {
                    "name": "bridge:read_file",
                    "description": "Read a file",
                    "parameters": json.dumps({"p": "int"}),
                    "examples": json.dumps(["e1", "e2"]),
                    "is_core": 0,
                }
            ],
        )
        rc = cli.main(["describe", "bridge:read_file", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["name"] == "bridge:read_file"
        assert parsed["parameters"] == {"p": "int"}
        assert parsed["examples"] == ["e1", "e2"]
        assert parsed["is_core"] is False
        assert rc == 0

    def test_describe_db_missing(self, tmp_path, monkeypatch, capsys):
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", tmp_path / "nope.db")
        rc = cli.main(["describe", "anything"])
        err = capsys.readouterr().err
        assert "No tool DB" in err
        assert "sync" in err.lower()
        assert rc == 1

    def test_describe_unknown_tool_with_suggestions(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {"name": "bridge:read_file", "description": "x"},
                {"name": "bridge:read_dir", "description": "y"},
            ],
        )
        rc = cli.main(["describe", "read"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "not found" in err.lower() or "Tool not found" in err
        assert "Did you mean" in err
        assert "bridge:read_file" in err or "bridge:read_dir" in err

    def test_describe_unknown_tool_no_suggestions(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [{"name": "bridge:read_file", "description": "x"}],
        )
        rc = cli.main(["describe", "zzzz_no_match_xyz"])
        err = capsys.readouterr().err
        assert rc == 1
        # No suggestions matched, so the alternate hint is shown.
        assert "search" in err.lower() or "discover" in err.lower()

    def test_describe_malformed_parameters_json(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {
                    "name": "bad",
                    "parameters": "{not-json{",
                    "examples": "[]",
                }
            ],
        )
        rc = cli.main(["describe", "bad"])
        err = capsys.readouterr().err
        assert "malformed" in err.lower()
        assert rc == 2

    def test_describe_text_no_description(self, tmp_path, monkeypatch, capsys):
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {
                    "name": "blank",
                    "description": None,  # null in db -> "(no description)"
                    "parameters": "",  # falsy -> empty dict
                    "examples": "",
                }
            ],
        )
        rc = cli.main(["describe", "blank"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no description" in out.lower()

    def test_describe_suggestion_underscore_is_literal_not_wildcard(
        self, tmp_path, monkeypatch, capsys
    ):
        """cli-ui-002: the not-found suggestion LIKE must treat ``_`` as a
        literal, not a single-char wildcard.

        We look up a tool name ``a_c`` that does not exist. With ``_`` unescaped
        (the bug), ``%a_c%`` wildcard-matches the unrelated tool ``axc`` and
        offers it as a bogus "Did you mean". With the fix (``_escape_like`` +
        ``ESCAPE``), ``_`` is literal, so ``axc`` must NOT be suggested — only a
        tool literally containing ``a_c`` would match, and none does here.
        """
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {"name": "axc", "description": "unrelated wildcard bait"},
                {"name": "ayc", "description": "more bait"},
            ],
        )
        rc = cli.main(["describe", "a_c"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "not found" in err.lower() or "Tool not found" in err
        # The wildcard-interpretation would have surfaced axc/ayc as suggestions.
        # Escaped, they must not appear — no bogus "Did you mean".
        assert "axc" not in err
        assert "ayc" not in err

    def test_describe_suggestion_literal_underscore_matches(
        self, tmp_path, monkeypatch, capsys
    ):
        """cli-ui-002 corollary: a tool whose name literally contains ``_`` is
        still surfaced as a suggestion when the (non-exact) query shares that
        literal ``_`` substring — proving the ESCAPE makes ``_`` match itself,
        not that escaping breaks legitimate matches."""
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {"name": "read_file", "description": "reads a file"},
                {"name": "write_file", "description": "writes a file"},
            ],
        )
        # Query 'read_' is not an exact tool name, so the suggestion path runs.
        # The literal '_' must match 'read_file' (which contains 'read_').
        rc = cli.main(["describe", "read_"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "read_file" in err
        # 'read_' does not appear in 'write_file', so it must NOT be suggested
        # (it would only via a wildcard reading of '_').
        assert "write_file" not in err

    def test_describe_suggestion_percent_is_literal_not_wildcard(
        self, tmp_path, monkeypatch, capsys
    ):
        """cli-ui-002: ``%`` in the queried name must be a literal too. A query
        of ``a%c`` (no exact match) must not wildcard-match ``abc``."""
        _setup_describe_db(
            tmp_path,
            monkeypatch,
            [
                {"name": "abc", "description": "percent wildcard bait"},
            ],
        )
        rc = cli.main(["describe", "a%c"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "not found" in err.lower() or "Tool not found" in err
        assert "abc" not in err


# =============================================================================
# cmd_doctor — text mode, FileNotFoundError, JSONDecodeError, generic Exception
# =============================================================================


class TestCmdDoctorExtended:
    def test_doctor_text_mode_renders_summary(self, monkeypatch, capsys):
        payload = {
            "version": "2.3.0",
            "config_path": "/tmp/compass_config.json",
            "backends": {"a": {}, "b": {}},
            "ollama_url": "http://localhost:11434",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "2.3.0" in out
        assert "compass_config" in out
        # ollama reachable → success line
        assert "ollama" in out.lower()

    def test_doctor_text_mode_ollama_unreachable(self, monkeypatch, capsys):
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "backends": [],
            "ollama_url": "http://localhost:11434",
            "ollama_reachable": False,
            "index_exists": False,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        # Warnings for ollama-unreachable + index-missing
        assert "unreachable" in out.lower()
        assert "missing" in out.lower()

    def test_doctor_text_backends_unknown_shape(self, monkeypatch, capsys):
        """backend_count falls back to 0 when backends is neither list/dict."""
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "backends": "this-is-a-string-not-a-list",
            "ollama_url": "u",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "0 configured" in out

    def test_doctor_filenotfound(self, monkeypatch, capsys):
        import config

        def bad():
            raise FileNotFoundError("config gone")

        monkeypatch.setattr(config, "doctor", bad)
        rc = cli.main(["doctor"])
        err = capsys.readouterr().err
        assert "config file missing" in err
        assert rc == 2

    def test_doctor_jsondecodeerror(self, monkeypatch, capsys):
        import config

        def bad():
            raise json.JSONDecodeError("not json", "doc", 0)

        monkeypatch.setattr(config, "doctor", bad)
        rc = cli.main(["doctor"])
        err = capsys.readouterr().err
        assert "malformed" in err
        assert rc == 2

    def test_doctor_generic_exception(self, monkeypatch, capsys):
        import config

        def bad():
            raise RuntimeError("unexpected")

        monkeypatch.setattr(config, "doctor", bad)
        rc = cli.main(["doctor"])
        err = capsys.readouterr().err
        assert "RuntimeError" in err
        assert rc == 2

    def test_doctor_text_backends_from_nested_config(self, monkeypatch, capsys):
        """When top-level 'backends' missing but config.backends present."""
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "config": {"backends": {"a": {}}},
            "ollama_url": "u",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 configured" in out

    def test_doctor_text_shows_unresolved_vars(self, monkeypatch, capsys):
        """cli-ux-005: text mode warns on config_unresolved_vars.

        Before the fix, text mode silently dropped this field even though
        doctor() returns it (only JSON mode surfaced it).
        """
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "backends": {"a": {}},
            "config_unresolved_vars": ["API_TOKEN", "OLLAMA_HOST"],
            "ollama_url": "u",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "unresolved config vars" in out
        assert "API_TOKEN" in out
        assert "OLLAMA_HOST" in out

    def test_doctor_text_no_unresolved_vars_no_warning(self, monkeypatch, capsys):
        """Empty/absent config_unresolved_vars prints no unresolved-var line."""
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "backends": {"a": {}},
            "config_unresolved_vars": [],
            "ollama_url": "u",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "unresolved config vars" not in out


# =============================================================================
# cmd_sync — text-mode success, error envelope, no-backends, FileNotFoundError
# =============================================================================


def _patch_sync(monkeypatch, *, config_obj=None, sync_result=None, sync_raises=None):
    """Patch out the load_config, BackendManager, CompassIndex, SyncManager imports."""
    from unittest.mock import AsyncMock, Mock

    import config

    if config_obj is None:
        config_obj = SimpleNamespace(backends={"x": object()})

    monkeypatch.setattr(config, "load_config", lambda: config_obj)

    # Fake BackendManager
    bm = Mock()
    bm.disconnect_all = AsyncMock(return_value=None)
    import backend_client_simple

    monkeypatch.setattr(
        backend_client_simple,
        "SimpleBackendManager",
        lambda config: bm,
    )

    # Fake CompassIndex
    fake_index = Mock()
    fake_index.load_index = Mock(return_value=True)
    import indexer

    monkeypatch.setattr(indexer, "CompassIndex", lambda: fake_index)

    # Fake SyncManager
    sm = Mock()
    if sync_raises is not None:
        sm.full_sync = AsyncMock(side_effect=sync_raises)
    else:
        sm.full_sync = AsyncMock(return_value=sync_result or {})
    import sync_manager

    monkeypatch.setattr(sync_manager, "SyncManager", lambda *a, **k: sm)


class TestCmdSyncExtended:
    # full_sync's real return shape (the shared contract): status,
    # tools_indexed, backends_synced, connected_backends, failed_backends,
    # build_result. The old mocks here fabricated tools_added/updated/removed/
    # duration_seconds/errors — keys full_sync never emits — which masked the
    # fact that _cmd_sync always printed "+0 ~0 -0" and never warned on a
    # partial failure (cli-ux-001 / cli-ux-002).
    def test_sync_text_success(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            sync_result={
                "status": "complete",
                "tools_indexed": 6,
                "backends_synced": ["a", "b"],
                "connected_backends": ["a", "b"],
                "failed_backends": [],
                "build_result": {},
            },
        )
        rc = cli.main(["sync"])
        out = capsys.readouterr().out
        assert rc == 0
        # Honest count from tools_indexed — no more fabricated "+0 ~0 -0".
        assert "6 tools indexed" in out

    def test_sync_text_with_failed_backends(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            sync_result={
                "status": "complete",
                "tools_indexed": 4,
                "backends_synced": ["foo", "bar"],
                "connected_backends": ["bar"],
                "failed_backends": ["foo"],
                "build_result": {},
            },
        )
        rc = cli.main(["sync"])
        captured = capsys.readouterr()
        # Warning rides on stderr; sync still succeeds (partial, not fatal).
        assert rc == 0
        assert "failed to connect" in captured.err
        assert "foo" in captured.err
        assert "4 tools indexed" in captured.out

    def test_sync_json_mode(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            sync_result={
                "status": "complete",
                "tools_indexed": 1,
                "backends_synced": ["a"],
                "connected_backends": ["a"],
                "failed_backends": [],
                "build_result": {},
            },
        )
        rc = cli.main(["sync", "--json"])
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["tools_indexed"] == 1
        assert rc == 0

    def test_sync_no_backends_configured(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            config_obj=SimpleNamespace(backends={}),
        )
        rc = cli.main(["sync"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "No backends" in err

    def test_sync_force_flag_accepted(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            sync_result={
                "status": "complete",
                "tools_indexed": 0,
                "backends_synced": ["a"],
                "connected_backends": ["a"],
                "failed_backends": [],
                "build_result": {},
            },
        )
        rc = cli.main(["sync", "--force"])
        assert rc == 0

    def test_sync_filenotfound(self, monkeypatch, capsys):
        _patch_sync(monkeypatch, sync_raises=FileNotFoundError("config.json"))
        rc = cli.main(["sync"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "missing" in err

    def test_sync_connection_error(self, monkeypatch, capsys):
        _patch_sync(monkeypatch, sync_raises=ConnectionError("backend dead"))
        rc = cli.main(["sync"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "connection" in err.lower() or "backend" in err.lower()

    def test_sync_no_color(self, monkeypatch, capsys):
        _patch_sync(
            monkeypatch,
            sync_result={
                "status": "complete",
                "tools_indexed": 0,
                "backends_synced": ["a"],
                "connected_backends": ["a"],
                "failed_backends": [],
                "build_result": {},
            },
        )
        rc = cli.main(["--no-color", "sync"])
        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert rc == 0


# =============================================================================
# Top-level main dispatch — KeyboardInterrupt, unknown, exception path
# =============================================================================


class TestMainDispatch:
    def test_main_keyboard_interrupt_returns_130(self, monkeypatch, capsys):
        _patch_gateway_raises(monkeypatch, "compass_status", KeyboardInterrupt())
        rc = cli.main(["status"])
        assert rc == 130
        err = capsys.readouterr().err
        assert "Interrupted" in err

    def test_main_unhandled_exception_returns_2(self, monkeypatch, capsys):
        """An exception that bubbles past _cmd_* handlers reaches main's catch."""
        # _build_parser is the easiest target — replace it with a raising stub.

        class _BadParser:
            def parse_args(self, argv=None):
                ns = SimpleNamespace(command="ui", no_color=False, port=7860,
                                     host="x", share=False, auth=None)
                return ns

        # We can't easily inject an exception in main without controlling args.
        # Instead, force _cmd_ui to raise via the ui module stub.
        def fake_ui_main():
            raise ValueError("boom in ui")

        fake_ui = type(sys)("ui")
        fake_ui.main = fake_ui_main
        monkeypatch.setitem(sys.modules, "ui", fake_ui)

        rc = cli.main(["ui"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "ui failed" in err.lower() or "ValueError" in err

    def test_main_serve_http_invalid_port_returns_2(self, monkeypatch, capsys):
        rc = cli.main(["serve", "--http", "not-a-number"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "integer" in err.lower()

    @pytest.mark.parametrize("bad_port", ["-1", "0", "99999999", "65536"])
    def test_main_serve_http_out_of_range_port_returns_2(
        self, bad_port, monkeypatch, capsys
    ):
        """cli-003 regression: ports that parse as int() but fall outside the
        valid TCP range (1-65535) must be rejected with a usage error and a
        range hint — never handed to the gateway. Before the fix, int() alone
        let '-1'/'0'/'99999999' through.

        gateway.main is monkeypatched so that IF a bad port leaked through and
        the server actually launched, we'd notice (it must not be called).
        """
        import gateway

        launched = {"n": 0}

        def fake_gateway_main():
            launched["n"] += 1
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)

        rc = cli.main(["serve", "--http", bad_port])
        err = capsys.readouterr().err
        assert rc == 2, f"port {bad_port!r} should be rejected"
        assert "range" in err.lower()
        assert "1-65535" in err
        # The gateway must NOT have been launched with an out-of-range port.
        assert launched["n"] == 0

    @pytest.mark.parametrize("good_port", ["1", "8080", "65535"])
    def test_main_serve_http_in_range_port_accepted(
        self, good_port, monkeypatch, capsys
    ):
        """cli-003 corollary: valid boundary ports (1, 65535) and a normal port
        are accepted, exported to PORT, and reach the gateway."""
        import gateway

        seen = {"port": None}

        def fake_gateway_main():
            seen["port"] = os.environ.get("PORT")
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)

        rc = cli.main(["serve", "--http", good_port])
        assert rc == 0
        assert seen["port"] == good_port

    def test_main_serve_http_default_8080(self, monkeypatch, capsys):
        """--http with no value AND no PORT env var defaults to 8080."""
        import gateway

        seen = {}

        def fake_gateway_main():
            seen["port"] = os.environ.get("PORT")
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)
        monkeypatch.delenv("PORT", raising=False)

        rc = cli.main(["serve", "--http"])
        assert rc == 0
        assert seen["port"] == "8080"

    def test_main_dispatch_to_doctor(self, monkeypatch, capsys):
        """Round-trip through main → _cmd_doctor."""
        import config

        monkeypatch.setattr(config, "doctor", lambda: {"version": "2.3.0"})
        rc = cli.main(["doctor"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "2.3.0" in out

    def test_main_default_serve_invokes_gateway(self, monkeypatch):
        """Bare `tool-compass` (no subcommand) falls through to gateway.main."""
        import gateway

        called = {"n": 0}

        def fake_gateway_main():
            called["n"] += 1
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)
        rc = cli.main([])
        assert rc == 0
        assert called["n"] == 1

    def test_main_explicit_serve_invokes_gateway(self, monkeypatch):
        import gateway

        called = {"n": 0, "argv": None}

        def fake_gateway_main():
            called["n"] += 1
            # cli-001: capture argv as gateway.main sees it. The fix neutralizes
            # sys.argv to ['tool-compass'] so the real gateway's argparse (which
            # reads sys.argv[1:]) never sees the 'serve' token. Asserting the
            # captured argv keeps this test honest even though we monkeypatch
            # main away here — if the neutralization regresses, argv would carry
            # 'serve' and this assert fails.
            called["argv"] = list(sys.argv)
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)
        rc = cli.main(["serve"])
        assert rc == 0
        assert called["n"] == 1
        # gateway.main must NOT see the 'serve' token in argv.
        assert "serve" not in called["argv"]
        assert called["argv"] == ["tool-compass"]

    def test_main_serve_real_gateway_no_argparse_crash(self, monkeypatch):
        """cli-001 regression: `tool-compass serve` runs the REAL gateway.main
        (not a monkeypatched fake) and must NOT crash with SystemExit /
        "unrecognized arguments: serve".

        gateway.main() calls argparse.parse_args() with no args, so it reads
        sys.argv[1:]. Before the fix, sys.argv was ['tool-compass', 'serve'],
        and gateway's strict parser raised SystemExit(2). We patch only the
        blocking server-run primitives (_run_http / mcp.run) so the function
        returns instead of binding a socket — the argparse path is fully real.
        """
        import gateway

        ran = {"http": None, "stdio": 0}

        # Neutralize the blocking transports so main() returns.
        monkeypatch.setattr(gateway, "_run_http", lambda port: ran.__setitem__("http", port))
        monkeypatch.setattr(gateway.mcp, "run", lambda *a, **k: ran.__setitem__("stdio", ran["stdio"] + 1))
        # Ensure stdio path (no PORT) for the bare-serve case.
        monkeypatch.delenv("PORT", raising=False)

        # Bare `serve` — if the bug is present, gateway argparse sees 'serve'
        # and raises SystemExit. The fix neutralizes argv so this is clean.
        try:
            rc = cli.main(["serve"])
        except SystemExit as e:  # pragma: no cover - asserts the bug is gone
            pytest.fail(
                f"`tool-compass serve` leaked SystemExit({e.code}) from gateway "
                "argparse — argv was not neutralized (cli-001 regression)."
            )
        assert rc == 0
        # The stdio transport path was reached (PORT unset).
        assert ran["stdio"] == 1

    def test_main_serve_http_real_gateway_no_argparse_crash(self, monkeypatch):
        """cli-001 regression for the `serve --http <port>` shape against the
        REAL gateway.main. The port must arrive via os.environ['PORT'] and the
        gateway must take the _run_http branch — never choke on argv tokens.
        """
        import gateway

        ran = {"http": None}
        monkeypatch.setattr(gateway, "_run_http", lambda port: ran.__setitem__("http", port))
        monkeypatch.setattr(gateway.mcp, "run", lambda *a, **k: pytest.fail(
            "stdio transport reached despite --http 9001 (PORT not exported)"
        ))
        monkeypatch.delenv("PORT", raising=False)

        try:
            rc = cli.main(["serve", "--http", "9001"])
        except SystemExit as e:  # pragma: no cover - asserts the bug is gone
            pytest.fail(
                f"`tool-compass serve --http 9001` leaked SystemExit({e.code}) "
                "from gateway argparse (cli-001 regression)."
            )
        assert rc == 0
        # Port flowed through PORT env into gateway's HTTP transport.
        assert ran["http"] == 9001

    def test_main_serve_http_empty_port_env_falls_back(self, monkeypatch):
        """--http with no value AND PORT='' in env defaults to 8080."""
        import gateway

        seen = {}

        def fake_gateway_main():
            seen["port"] = os.environ.get("PORT")
            return 0

        monkeypatch.setattr(gateway, "main", fake_gateway_main, raising=False)
        if hasattr(cli, "gateway"):
            monkeypatch.setattr(cli.gateway, "main", fake_gateway_main, raising=False)
        # An empty-string PORT triggers the `if not port: port = "8080"` branch.
        monkeypatch.setenv("PORT", "")
        rc = cli.main(["serve", "--http"])
        assert rc == 0
        assert seen["port"] == "8080"


# =============================================================================
# Hard-to-reach defensive branches
# =============================================================================


class TestDoctorListBackends:
    """Cover the list/tuple branch of the backend_count detection in doctor."""

    def test_doctor_text_backends_as_list(self, monkeypatch, capsys):
        """When backends comes back as a list, len(list) feeds backend_count."""
        payload = {
            "version": "2.3.0",
            "config_path": "x",
            "backends": ["a", "b", "c"],  # list shape, not dict
            "ollama_url": "u",
            "ollama_reachable": True,
            "index_exists": True,
        }
        import config

        monkeypatch.setattr(config, "doctor", lambda: payload)
        rc = cli.main(["doctor", "--text"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "3 configured" in out


class TestDescribeOperationalError:
    """Cover the sqlite3.OperationalError branch in cmd_describe."""

    def test_describe_operational_error_returns_2(self, tmp_path, monkeypatch, capsys):
        """If sqlite3.execute() raises OperationalError, _print_error fires with rc=2."""
        # Build a valid DB so the path-exists check passes; then patch
        # sqlite3.connect so the .execute call inside _cmd_describe raises.
        import sqlite3 as sqlite3_module

        db_path = tmp_path / "tools.db"
        # Create the file so Path.exists() returns True.
        db_path.write_text("dummy")
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", db_path)

        class _BadCursor:
            def __init__(self):
                pass

            def execute(self, *args, **kwargs):
                raise sqlite3_module.OperationalError("simulated corruption")

            def fetchone(self):
                return None

            def fetchall(self):
                return []

        class _BadConn:
            row_factory = None

            def execute(self, *args, **kwargs):
                raise sqlite3_module.OperationalError("simulated corruption")

            def close(self):
                pass

        # Patch sqlite3.connect inside cli (the module imports sqlite3 top-level).
        monkeypatch.setattr(cli.sqlite3, "connect", lambda *a, **k: _BadConn())

        rc = cli.main(["describe", "anything"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "DB query failed" in err
        assert "corrupted" in err.lower() or "sync" in err.lower()

    def test_describe_corrupt_db_raises_database_error_not_operational(
        self, tmp_path, monkeypatch, capsys
    ):
        """cli-002 regression: a physically corrupt DB raises sqlite3.DatabaseError
        (the PARENT of OperationalError, NOT a subclass), which the old narrow
        `except sqlite3.OperationalError` did not catch — leaking a raw traceback
        past the SD-CLI-005 rebuild-hint path. The fix broadens to DatabaseError.

        We inject sqlite3.DatabaseError directly (NOT OperationalError) so this
        test FAILS if the catch is narrowed back to OperationalError.
        """
        import sqlite3 as sqlite3_module

        # DatabaseError is the base; OperationalError subclasses it. Confirm the
        # injected type is genuinely NOT an OperationalError so the test probes
        # the real gap rather than the already-covered subclass.
        assert not issubclass(sqlite3_module.DatabaseError, sqlite3_module.OperationalError)

        db_path = tmp_path / "tools.db"
        db_path.write_text("dummy")  # exists -> path check passes
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", db_path)

        closed = {"n": 0}

        class _CorruptConn:
            row_factory = None

            def execute(self, *args, **kwargs):
                # "file is not a database" surfaces as DatabaseError, not
                # OperationalError.
                raise sqlite3_module.DatabaseError("file is not a database")

            def close(self):
                closed["n"] += 1

        monkeypatch.setattr(cli.sqlite3, "connect", lambda *a, **k: _CorruptConn())

        rc = cli.main(["describe", "anything"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "DB query failed" in err
        assert "corrupted" in err.lower() or "sync" in err.lower()
        # cli-002: the conn must be closed even on the error path (no leak).
        assert closed["n"] == 1

    def test_describe_closes_conn_on_success(self, tmp_path, monkeypatch, capsys):
        """cli-002 corollary: the happy path also closes the connection exactly
        once via the finally block (no leak on success either)."""
        import sqlite3 as sqlite3_module

        db_path = tmp_path / "tools.db"
        db_path.write_text("dummy")
        import indexer

        monkeypatch.setattr(indexer, "SQLITE_DB_PATH", db_path)

        closed = {"n": 0}

        class _Cursor:
            def fetchone(self):
                # Minimal row supporting __getitem__ access used by _cmd_describe.
                return {
                    "name": "x",
                    "description": "d",
                    "category": "c",
                    "server": "s",
                    "parameters": "{}",
                    "examples": "[]",
                    "is_core": 0,
                }

            def fetchall(self):
                return []

        class _OkConn:
            row_factory = None

            def execute(self, *args, **kwargs):
                return _Cursor()

            def close(self):
                closed["n"] += 1

        monkeypatch.setattr(cli.sqlite3, "connect", lambda *a, **k: _OkConn())
        rc = cli.main(["describe", "x"])
        assert rc == 0
        assert closed["n"] == 1
        _ = sqlite3_module  # silence unused in some linters


class TestGatewayImportFailure:
    """Cover the `from gateway import X` failure branches for each subcommand."""

    def test_status_gateway_import_fail(self, monkeypatch, capsys):
        """If `from gateway import compass_status` raises, return rc=2."""
        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gateway" and fromlist and "compass_status" in fromlist:
                raise ImportError("gateway broken")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = cli.main(["status"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Could not import gateway" in err

    def test_categories_gateway_import_fail(self, monkeypatch, capsys):
        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gateway" and fromlist and "compass_categories" in fromlist:
                raise ImportError("gateway broken")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = cli.main(["categories"])
        capsys.readouterr().err
        assert rc == 2

    def test_audit_gateway_import_fail(self, monkeypatch, capsys):
        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gateway" and fromlist and "compass_audit" in fromlist:
                raise ImportError("gateway broken")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = cli.main(["audit"])
        capsys.readouterr().err
        assert rc == 2

    def test_analytics_gateway_import_fail(self, monkeypatch, capsys):
        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gateway" and fromlist and "compass_analytics" in fromlist:
                raise ImportError("gateway broken")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = cli.main(["analytics"])
        capsys.readouterr().err
        assert rc == 2

    def test_chains_gateway_import_fail(self, monkeypatch, capsys):
        import builtins
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "gateway" and fromlist and "compass_chains" in fromlist:
                raise ImportError("gateway broken")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rc = cli.main(["chains"])
        capsys.readouterr().err
        assert rc == 2


class TestMainHelpFallback:
    """Hit the parser.print_help() + return 2 fallback at the bottom of main."""

    def test_main_unrecognized_command_prints_help(self, monkeypatch, capsys):
        """If args.command somehow falls through the dispatch chain, return 2.

        Practically unreachable via argparse (it constrains COMMAND to the
        registered set), but we test it by patching parse_args to return a
        Namespace with a command argparse would never set.
        """
        original_build = cli._build_parser

        def fake_build():
            parser = original_build()
            real_parse = parser.parse_args

            def fake_parse(argv=None):
                # Real parse first, then mutate command to a bogus value so
                # the dispatch chain never matches anything.
                ns = real_parse(argv)
                ns.command = "does_not_exist"
                return ns

            parser.parse_args = fake_parse
            return parser

        monkeypatch.setattr(cli, "_build_parser", fake_build)
        # `doctor` so the parser actually accepts the argv; the mutation
        # happens after parsing.
        rc = cli.main(["doctor"])
        assert rc == 2


# =============================================================================
# bootstrap._ollama_has_model — cli-ux-003: respects OLLAMA_URL + no curl
# dependency (reuses config._ollama_reachable, probes /api/tags via httpx).
# =============================================================================


class TestBootstrapOllamaProbe:
    def test_returns_false_when_unreachable(self, monkeypatch):
        import bootstrap
        import config

        # Reachability gate fails -> short-circuit False, never touches httpx.
        monkeypatch.setattr(config, "_ollama_reachable", lambda u, t=2.0: False)
        assert bootstrap._ollama_has_model("http://x:11434", "nomic-embed-text") is False

    def test_true_when_reachable_and_model_present(self, monkeypatch):
        import bootstrap
        import config
        import httpx

        monkeypatch.setattr(config, "_ollama_reachable", lambda u, t=2.0: True)

        class _Resp:
            status_code = 200
            text = '{"models":[{"name":"nomic-embed-text:latest"}]}'

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url):
                # cli-ux-003: must hit the OLLAMA_URL-derived endpoint.
                assert url == "http://x:11434/api/tags"
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)
        assert bootstrap._ollama_has_model("http://x:11434", "nomic-embed-text") is True

    def test_false_when_model_missing(self, monkeypatch):
        import bootstrap
        import config
        import httpx

        monkeypatch.setattr(config, "_ollama_reachable", lambda u, t=2.0: True)

        class _Resp:
            status_code = 200
            text = '{"models":[{"name":"llama3:latest"}]}'

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url):
                return _Resp()

        monkeypatch.setattr(httpx, "Client", _Client)
        assert (
            bootstrap._ollama_has_model("http://x:11434", "nomic-embed-text") is False
        )


# =============================================================================
# cmd_init — onboarding scaffold + MCP-client registration snippet
# =============================================================================
#
# Feature: `tool-compass init`. Resolves the user config path via
# config.get_config_path() (which honors TOOL_COMPASS_CONFIG), writes a
# config there (parent dirs created as needed), refuses to clobber an
# existing config without --force, and prints a Claude Desktop mcpServers
# snippet. --json emits {created, source, force, overwrote,
# claude_desktop_config}. We isolate every test by pointing
# TOOL_COMPASS_CONFIG at a fresh tmp path so the developer's real config is
# never touched.


@pytest.fixture
def init_config_path(tmp_path, monkeypatch):
    """Point get_config_path() at a fresh tmp file via TOOL_COMPASS_CONFIG.

    Uses a NESTED dir that does not exist yet so the parent-dir-creation
    path in _cmd_init is exercised on the happy path.
    """
    target = tmp_path / "nested" / "cfg" / "compass_config.json"
    monkeypatch.setenv("TOOL_COMPASS_CONFIG", str(target))
    # get_config_path resolves the env var, so the returned path is .resolve()d.
    return Path(str(target)).resolve()


class TestCmdInit:
    def test_init_creates_file_at_resolved_path(self, init_config_path, capsys):
        """init writes a config at the TOOL_COMPASS_CONFIG path, creating
        parent dirs, and exits 0."""
        assert not init_config_path.exists()
        rc = cli.main(["init"])
        out = capsys.readouterr().out
        assert rc == 0
        assert init_config_path.exists()
        # The written file is valid JSON with a backends key (example or
        # minimal fallback both carry it).
        written = json.loads(init_config_path.read_text(encoding="utf-8"))
        assert "backends" in written
        # The resolved path is echoed to the user.
        assert str(init_config_path) in out

    def test_init_prints_claude_desktop_snippet(self, init_config_path, capsys):
        """The pasteable Claude Desktop mcpServers block appears in output."""
        rc = cli.main(["init"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mcpServers" in out
        assert "tool-compass" in out
        # The npx serve form must be present and copyable.
        assert "@mcptoolshop/tool-compass" in out
        assert "serve" in out

    def test_init_prints_next_steps(self, init_config_path, capsys):
        """Next-steps guidance references sync + serve."""
        rc = cli.main(["init"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "sync" in out
        assert "serve" in out
        assert "backends" in out.lower()

    def test_init_refuses_overwrite_without_force(self, init_config_path, capsys):
        """An existing config is NOT clobbered without --force; exit 1 + hint."""
        init_config_path.parent.mkdir(parents=True, exist_ok=True)
        init_config_path.write_text('{"backends": {"keep": "me"}}', encoding="utf-8")
        rc = cli.main(["init"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "already exists" in err.lower()
        assert "--force" in err
        # The original content must be untouched.
        preserved = json.loads(init_config_path.read_text(encoding="utf-8"))
        assert preserved == {"backends": {"keep": "me"}}

    def test_init_force_overwrites(self, init_config_path, capsys):
        """--force replaces an existing config and reports the overwrite."""
        init_config_path.parent.mkdir(parents=True, exist_ok=True)
        init_config_path.write_text('{"backends": {"old": "value"}}', encoding="utf-8")
        rc = cli.main(["init", "--force"])
        out = capsys.readouterr().out
        assert rc == 0
        # The file was rewritten — the sentinel "old" key is gone (the scaffold
        # writes the example/minimal config, neither of which has an "old" key).
        written = json.loads(init_config_path.read_text(encoding="utf-8"))
        assert "old" not in written.get("backends", {})
        assert "verwr" in out.lower() or "overwrote" in out.lower()  # "Overwrote"

    def test_init_json_shape(self, init_config_path, capsys):
        """--json emits a stable {created, source, force, overwrote, ...} shape."""
        rc = cli.main(["init", "--json"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["created"] == str(init_config_path)
        assert parsed["force"] is False
        assert parsed["overwrote"] is False
        # The Claude Desktop config is embedded as a structured object.
        cd = parsed["claude_desktop_config"]
        assert cd["mcpServers"]["tool-compass"]["command"] == "npx"
        assert "serve" in cd["mcpServers"]["tool-compass"]["args"]

    def test_init_json_overwrote_flag(self, init_config_path, capsys):
        """--json with --force over an existing file reports overwrote=True."""
        init_config_path.parent.mkdir(parents=True, exist_ok=True)
        init_config_path.write_text('{"backends": {}}', encoding="utf-8")
        rc = cli.main(["init", "--force", "--json"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["overwrote"] is True
        assert parsed["force"] is True

    def test_init_json_refuse_no_stdout_json(self, init_config_path, capsys):
        """Refuse-without-force in --json mode still exits 1 (no created JSON)."""
        init_config_path.parent.mkdir(parents=True, exist_ok=True)
        init_config_path.write_text('{"backends": {}}', encoding="utf-8")
        rc = cli.main(["init", "--json"])
        captured = capsys.readouterr()
        assert rc == 1
        # Error rides on stderr; stdout must not carry a success JSON payload.
        assert "already exists" in captured.err.lower()
        assert captured.out.strip() == ""

    def test_init_no_secrets_in_snippet(self, init_config_path, capsys):
        """The pasteable snippet must NOT embed any token/secret material.

        Asserts against the snippet helper directly (not full stdout) so a
        tmp-dir name that happens to contain a marker word can't false-positive.
        """
        rc = cli.main(["init"])
        assert rc == 0
        snippet = cli._claude_desktop_snippet().lower()
        for marker in ("token", "password", "secret", "api_key", "apikey"):
            assert marker not in snippet

    def test_init_no_color(self, init_config_path, capsys):
        """--no-color path emits no ANSI escapes."""
        rc = cli.main(["--no-color", "init"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "\x1b[" not in out

    def test_init_copies_example_when_present(self, init_config_path, monkeypatch, capsys):
        """When the repo example is locatable, its bytes are copied verbatim —
        so a field a sibling agent adds to the example is picked up for free."""
        # Force _locate_example_config to return a temp example carrying a
        # sentinel field (mimics a future embedding_provider addition).
        sentinel_example = init_config_path.parent.parent / "example.json"
        sentinel_example.parent.mkdir(parents=True, exist_ok=True)
        sentinel_example.write_text(
            json.dumps({"backends": {}, "embedding_provider": "future_field"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli, "_locate_example_config", lambda: sentinel_example)
        rc = cli.main(["init"])
        assert rc == 0
        written = json.loads(init_config_path.read_text(encoding="utf-8"))
        # The sentinel field flowed through untouched (we copy, never reparse).
        assert written["embedding_provider"] == "future_field"

    def test_init_minimal_fallback_when_no_example(self, init_config_path, monkeypatch, capsys):
        """When no example is locatable, a minimal valid config is written from
        the live dataclass defaults (round-trips through to_dict)."""
        monkeypatch.setattr(cli, "_locate_example_config", lambda: None)
        rc = cli.main(["init"])
        assert rc == 0
        written = json.loads(init_config_path.read_text(encoding="utf-8"))
        # Minimal config: empty backends skeleton + documented defaults present.
        assert written["backends"] == {}
        assert "ollama_url" in written
        assert "default_top_k" in written

    def test_init_write_failure_returns_1(self, init_config_path, monkeypatch, capsys):
        """An OSError while writing the config surfaces a clean error + exit 1."""
        import pathlib

        original_write = pathlib.Path.write_text

        def boom(self, *args, **kwargs):
            if self == init_config_path:
                raise OSError("disk full")
            return original_write(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "write_text", boom)
        rc = cli.main(["init"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "could not write" in err.lower()


class TestInitHelpers:
    """Direct unit coverage for the init helper functions."""

    def test_claude_desktop_snippet_is_valid_json(self):
        snippet = cli._claude_desktop_snippet()
        parsed = json.loads(snippet)
        server = parsed["mcpServers"]["tool-compass"]
        assert server["command"] == "npx"
        assert server["args"] == ["-y", "@mcptoolshop/tool-compass", "serve"]

    def test_claude_desktop_snippet_obj_matches_text(self):
        obj = cli._claude_desktop_snippet_obj()
        assert obj == json.loads(cli._claude_desktop_snippet())

    def test_minimal_config_json_round_trips(self):
        """The minimal-config fallback is valid JSON that load-parses into a
        CompassConfig via from_dict (defaults round-trip)."""
        import config

        raw = cli._minimal_config_json()
        data = json.loads(raw)
        # from_dict must accept it without raising — it's a real config shape.
        cfg = config.CompassConfig.from_dict(data)
        assert cfg.backends == {}
        # to_dict/from_dict round-trip stability: the parsed config re-serializes
        # to the same documented-default values.
        assert cfg.to_dict()["default_top_k"] == data["default_top_k"]

    def test_locate_example_config_finds_repo_example(self):
        """In a source checkout the repo example sits next to cli.py."""
        located = cli._locate_example_config()
        # The repo ships compass_config.example.json next to cli.py, so this
        # resolves in the test environment.
        assert located is not None
        assert located.name == "compass_config.example.json"
        assert located.is_file()
