"""
Tool Compass - UI module unit tests

Covers ``ui.py`` without launching a real Gradio server. Strategy:

* Pure helpers (`sanitize_query`, `truncate_text`, `confidence_label`,
  `format_error`, `_inline_fallback_banner`, `_render_no_results`) are
  exercised with literal inputs and asserted on output substrings.
* Module-level lazy singletons (`_index`, `_analytics`, `_chain_indexer`,
  `_config`) are reset by the autouse `_reset_ui_globals` fixture and
  replaced with `unittest.mock.Mock` objects per test.
* Async dependencies (`embedder.health_check`, `index.search`,
  `analytics.get_analytics_summary`, `chain_indexer.search_chains`,
  `chain_indexer.load_chains_from_db`) are mocked with `AsyncMock`.
* The Gradio Blocks construction in `create_ui` is NOT exercised here —
  the handler functions it wires up are tested directly so coverage
  measures the actual business logic, not the layout calls.

Each test has at least one meaningful assert; no test launches a
network call, real Ollama, or a real Gradio server.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import ui


# =============================================================================
# UI module global reset fixture
# =============================================================================


_UI_STATE_NAMES = ("_index", "_analytics", "_chain_indexer", "_config")


@pytest.fixture(autouse=True)
def _reset_ui_globals():
    """Snapshot + restore ui module globals around every test.

    Mirrors the autouse `_reset_gateway_globals` pattern from conftest.py:
    ui.py keeps lazy singletons on the module namespace, so mutating them
    in one test would leak into the next. This fixture resets the four
    state names back to None after each test so every test starts from a
    clean slate.
    """
    snapshot = {name: getattr(ui, name, None) for name in _UI_STATE_NAMES}
    try:
        yield
    finally:
        for name, value in snapshot.items():
            setattr(ui, name, value)


# =============================================================================
# Helpers / factories
# =============================================================================


def _fake_result(name: str, description: str, score: float,
                 server: str = "test", category: str = "file",
                 parameters: dict | None = None,
                 deprecated_since: str | None = None) -> SimpleNamespace:
    """Build a SearchResult duck so search_tools' render loop works."""
    tool_attrs = dict(
        name=name,
        description=description,
        server=server,
        category=category,
        parameters=parameters or {},
    )
    if deprecated_since is not None:
        tool_attrs["deprecated_since"] = deprecated_since
    return SimpleNamespace(tool=SimpleNamespace(**tool_attrs), score=score)


def _fake_chain_result(name: str, tools: list[str], description: str,
                      score: float, use_count: int = 1,
                      is_auto_detected: bool = True) -> SimpleNamespace:
    """Build a ChainSearchResult duck."""
    chain = SimpleNamespace(
        name=name,
        tools=tools,
        description=description,
        use_count=use_count,
        is_auto_detected=is_auto_detected,
    )
    return SimpleNamespace(chain=chain, score=score)


def _make_mock_index(rows: list[dict] | None = None) -> MagicMock:
    """Mock CompassIndex with a sqlite-like `.db.execute` chain."""
    mock = MagicMock()
    mock.index_path = "/tmp/test_index.hnsw"

    rows = rows or []

    # Build a real in-memory sqlite db so the LIKE-scan fallback in
    # _lexical_fallback_for_ui has somewhere real to read from.
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE tools (
            name TEXT PRIMARY KEY,
            description TEXT,
            category TEXT,
            server TEXT,
            parameters TEXT,
            examples TEXT
        )
    """)
    for r in rows:
        db.execute(
            "INSERT INTO tools (name, description, category, server, parameters, examples) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                r["name"],
                r.get("description", ""),
                r.get("category", "file"),
                r.get("server", "test"),
                json.dumps(r.get("parameters", {})),
                json.dumps(r.get("examples", [])),
            ),
        )
    db.commit()
    mock.db = db
    mock.get_stats = MagicMock(return_value={
        "total_tools": len(rows),
        "core_tools": 0,
        "by_server": {"test": len(rows)} if rows else {},
        "by_category": {"file": len(rows)} if rows else {},
    })
    return mock


# =============================================================================
# Pure helpers
# =============================================================================


class TestSanitizeQuery:
    def test_empty_string_returns_empty(self):
        assert ui.sanitize_query("") == ""

    def test_none_returns_empty(self):
        assert ui.sanitize_query(None) == ""  # type: ignore[arg-type]

    def test_strips_control_characters(self):
        # \x00, \x07, \x1f are non-printable
        out = ui.sanitize_query("hello\x00world\x07")
        assert "\x00" not in out
        assert "\x07" not in out
        assert "hello" in out and "world" in out

    def test_collapses_whitespace(self):
        # \t is non-printable per str.isprintable() so it's stripped before
        # the whitespace collapse — multi-space runs collapse to single.
        out = ui.sanitize_query("  multiple   spaces   here  ")
        assert out == "multiple spaces here"

    def test_length_cap_500(self):
        out = ui.sanitize_query("a" * 1000)
        assert len(out) == 500

    def test_preserves_printable_unicode(self):
        out = ui.sanitize_query("café naïve")
        assert "café" in out


class TestTruncateText:
    def test_empty_string(self):
        assert ui.truncate_text("") == ""

    def test_none_returns_empty(self):
        assert ui.truncate_text(None) == ""  # type: ignore[arg-type]

    def test_short_passthrough(self):
        assert ui.truncate_text("hi", max_length=120) == "hi"

    def test_exactly_max_length(self):
        s = "a" * 120
        assert ui.truncate_text(s, max_length=120) == s

    def test_truncates_with_ellipsis(self):
        s = "word " * 100
        out = ui.truncate_text(s, max_length=30)
        assert out.endswith("...")
        assert len(out) <= 30

    def test_truncates_on_word_boundary(self):
        s = "one two three four five six seven eight"
        out = ui.truncate_text(s, max_length=20)
        # truncate_text uses rsplit(" ", 1) so it ends on a word boundary
        assert out.endswith("...")
        assert " " in out or out.endswith("...")


class TestConfidenceLabel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (1.0, "Excellent"),
            (0.85, "Excellent"),
            (0.8, "Excellent"),
            (0.79, "Good"),
            (0.6, "Good"),
            (0.59, "Fair"),
            (0.4, "Fair"),
            (0.39, "Low"),
            (0.0, "Low"),
        ],
    )
    def test_band_thresholds(self, score, expected):
        assert ui.confidence_label(score) == expected


# =============================================================================
# format_error — escaping is load-bearing for FE-A2-001/002
# =============================================================================


class TestFormatError:
    def test_connection_error_renders_ollama_serve_card(self):
        err = ConnectionError("connection refused at localhost:11434")
        out = ui.format_error(err)
        assert "Service unavailable" in out
        assert "ollama serve" in out
        # role="alert" so screen readers announce immediately
        assert 'role="alert"' in out

    def test_index_error_renders_sync_card(self):
        err = RuntimeError("index not loaded")
        out = ui.format_error(err)
        assert "Index not ready" in out
        assert "tool-compass sync" in out

    def test_generic_error_renders_generic_card(self):
        err = ValueError("something exploded")
        out = ui.format_error(err)
        assert "Something went wrong" in out
        # technical-details disclosure carries the exception type + message
        assert "ValueError" in out
        assert "something exploded" in out

    def test_html_in_context_is_escaped(self):
        """FE-A2-001: context may carry user-supplied content — must escape."""
        err = ValueError("oops")
        ctx = "<script>alert('xss')</script>"
        out = ui.format_error(err, context=ctx)
        # raw <script> must NOT appear; escaped form must
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_html_in_exception_string_is_escaped(self):
        """FE-A2-002: exception payloads can carry hostile bytes."""
        err = ValueError("<img src=x onerror=alert(1)>")
        out = ui.format_error(err)
        assert "<img src=x" not in out
        assert "&lt;img" in out

    def test_quote_chars_in_context_are_escaped(self):
        """quote=True so " and ' are escaped, since context lands inside attrs/text."""
        err = ValueError("boom")
        ctx = 'attr="value" with quotes'
        out = ui.format_error(err, context=ctx)
        # html.escape(quote=True) → " becomes &quot;
        assert '"value"' not in out
        assert "&quot;" in out

    def test_long_exception_string_truncated_to_200(self):
        err = ValueError("x" * 500)
        out = ui.format_error(err)
        # body of <code>...</code> only carries first 200 chars (escaped)
        assert "x" * 500 not in out

    def test_role_alert_present_in_all_branches(self):
        for err in (ConnectionError("conn"), RuntimeError("index missing"),
                    ValueError("misc")):
            assert 'role="alert"' in ui.format_error(err)


# =============================================================================
# _inline_fallback_banner / _render_no_results
# =============================================================================


class TestInlineFallbackBanner:
    def test_returns_role_status(self):
        out = ui._inline_fallback_banner()
        assert 'role="status"' in out
        assert "Ollama unreachable" in out
        assert "keyword-based results" in out.lower()


class TestRenderNoResults:
    def test_empty_suggestions(self):
        out = ui._render_no_results("missing query", [])
        assert "No tools found" in out
        assert "missing query" in out
        assert "Or try" in out

    def test_with_suggestions(self):
        suggestions = [
            {"tool": "test:read_file", "description": "Read a file"},
            {"tool": "test:write_file", "description": "Write a file"},
        ]
        out = ui._render_no_results("read", suggestions)
        assert "Did you mean" in out
        assert "test:read_file" in out
        assert "test:write_file" in out
        assert "Read a file" in out

    def test_query_is_escaped_in_no_match_header(self):
        out = ui._render_no_results("<script>x</script>", [])
        assert "<script>x</script>" not in out
        assert "&lt;script&gt;" in out

    def test_suggestion_description_is_escaped(self):
        suggestions = [{"tool": "x:y", "description": "<b>bold</b>"}]
        out = ui._render_no_results("q", suggestions)
        assert "<b>bold</b>" not in out
        assert "&lt;b&gt;" in out


# =============================================================================
# _lexical_fallback_for_ui — exercises the inline branch when gateway is absent
# =============================================================================


class TestLexicalFallback:
    def test_none_index_returns_empty(self):
        assert ui._lexical_fallback_for_ui(None, "anything", 5, None, None) == []

    def test_index_without_db_returns_empty(self):
        mock_idx = MagicMock(spec=["db"])
        mock_idx.db = None
        assert ui._lexical_fallback_for_ui(mock_idx, "q", 5, None, None) == []

    def test_match_via_gateway_path(self):
        """Happy path — gateway._lexical_search_fallback handles it."""
        idx = _make_mock_index(rows=[
            {"name": "test:read_file", "description": "Read contents of a file",
             "category": "file", "server": "test"},
            {"name": "test:write_file", "description": "Write content to a file",
             "category": "file", "server": "test"},
        ])
        results = ui._lexical_fallback_for_ui(idx, "read", 5, None, None)
        assert len(results) >= 1
        names = [r["tool"] for r in results]
        assert "test:read_file" in names
        # gateway path tags degraded=True
        assert results[0]["degraded"] is True

    def test_inline_fallback_when_gateway_import_fails(self):
        """Force the except branch (lines ~256-298) — inline LIKE scan."""
        idx = _make_mock_index(rows=[
            {"name": "tool:alpha", "description": "alpha read tool",
             "category": "file", "server": "srv1"},
            {"name": "tool:beta", "description": "beta tool",
             "category": "web", "server": "srv2"},
        ])
        # Patch the gateway import inside ui._lexical_fallback_for_ui so the
        # try-block raises and we fall through to the inline implementation.
        with patch.dict("sys.modules", {"gateway": None}):
            # `from gateway import _lexical_search_fallback` will raise
            # ImportError when sys.modules['gateway'] is None.
            results = ui._lexical_fallback_for_ui(idx, "alpha", 5, None, None)
        assert len(results) == 1
        assert results[0]["tool"] == "tool:alpha"
        assert results[0]["degraded"] is True
        # name match → higher confidence than description match
        assert results[0]["confidence"] >= 0.4

    def test_inline_fallback_with_category_filter(self):
        idx = _make_mock_index(rows=[
            {"name": "a:tool", "description": "alpha matching",
             "category": "file", "server": "s1"},
            {"name": "b:tool", "description": "alpha matching",
             "category": "web", "server": "s1"},
        ])
        with patch.dict("sys.modules", {"gateway": None}):
            results = ui._lexical_fallback_for_ui(idx, "alpha", 5, "file", None)
        assert len(results) == 1
        assert results[0]["category"] == "file"

    def test_inline_fallback_with_server_filter(self):
        idx = _make_mock_index(rows=[
            {"name": "a:tool", "description": "match",
             "category": "file", "server": "s1"},
            {"name": "b:tool", "description": "match",
             "category": "file", "server": "s2"},
        ])
        with patch.dict("sys.modules", {"gateway": None}):
            results = ui._lexical_fallback_for_ui(idx, "match", 5, None, "s2")
        assert len(results) == 1
        assert results[0]["server"] == "s2"

    def test_inline_fallback_sqlite_error_returns_empty(self):
        """SQL execution raising should not blow up — return []."""
        idx = MagicMock()
        idx.db = MagicMock()
        idx.db.execute.side_effect = sqlite3.OperationalError("no such table")
        with patch.dict("sys.modules", {"gateway": None}):
            assert ui._lexical_fallback_for_ui(idx, "q", 5, None, None) == []

    def test_inline_fallback_underscore_is_not_a_wildcard(self):
        """FE-SA-002: '_' in the query must be treated literally, not as a
        single-char LIKE wildcard. The tool 'alpha' must NOT be returned for
        the query 'a_pha' once % and _ are escaped with an ESCAPE clause.
        Without the fix, '%a_pha%' matches 'alpha' (the '_' wildcard eats the
        'l'), producing over-broad results.
        """
        idx = _make_mock_index(rows=[
            {"name": "alpha", "description": "a tool",
             "category": "file", "server": "s1"},
        ])
        with patch.dict("sys.modules", {"gateway": None}):
            results = ui._lexical_fallback_for_ui(idx, "a_pha", 5, None, None)
        assert results == []

    def test_inline_fallback_percent_is_not_a_wildcard(self):
        """FE-SA-002: '%' must be literal, not a multi-char LIKE wildcard.
        Querying 'a%a' must NOT match 'alpha' once escaping + ESCAPE land.
        Without the fix, '%a%a%' matches 'alpha'.
        """
        idx = _make_mock_index(rows=[
            {"name": "alpha", "description": "a tool",
             "category": "file", "server": "s1"},
        ])
        with patch.dict("sys.modules", {"gateway": None}):
            results = ui._lexical_fallback_for_ui(idx, "a%a", 5, None, None)
        assert results == []

    def test_inline_fallback_literal_underscore_still_matches(self):
        """FE-SA-002: escaping must not break legitimate literal matches —
        a query whose '_' really appears in the tool name still matches.
        """
        idx = _make_mock_index(rows=[
            {"name": "read_file", "description": "reads a file",
             "category": "file", "server": "s1"},
        ])
        with patch.dict("sys.modules", {"gateway": None}):
            results = ui._lexical_fallback_for_ui(idx, "read_file", 5, None, None)
        assert len(results) == 1
        assert results[0]["tool"] == "read_file"


class TestNearestMatches:
    def test_returns_empty_on_exception(self):
        # _nearest_matches catches any exception from _lexical_fallback_for_ui.
        with patch.object(ui, "_lexical_fallback_for_ui",
                          side_effect=RuntimeError("boom")):
            assert ui._nearest_matches(None, "q") == []

    def test_passes_through_to_fallback(self):
        with patch.object(ui, "_lexical_fallback_for_ui",
                          return_value=[{"tool": "a", "description": "d",
                                         "server": "s", "category": "c",
                                         "confidence": 0.5, "degraded": True}]):
            out = ui._nearest_matches(MagicMock(), "q", max_results=3)
        assert len(out) == 1
        assert out[0]["tool"] == "a"


# =============================================================================
# _lazy_singleton — partial-init recovery (FE-A2-003/004/005 + FE-B-011)
# =============================================================================


class TestLazySingleton:
    def test_factory_called_once_on_success(self):
        factory = Mock(return_value="instance")
        ui._index = None
        out1 = ui._lazy_singleton("_index", factory)
        out2 = ui._lazy_singleton("_index", factory)
        assert out1 == "instance"
        assert out2 == "instance"
        assert factory.call_count == 1

    def test_factory_failure_keeps_global_none(self):
        """Raise → global stays None so next call retries."""
        ui._index = None
        factory = Mock(side_effect=RuntimeError("init failed"))
        with pytest.raises(RuntimeError):
            ui._lazy_singleton("_index", factory)
        # Global must NOT have been published.
        assert ui._index is None

    def test_factory_returning_none_does_not_cache(self):
        """None is a legitimate value — chain_indexer can be disabled."""
        ui._chain_indexer = None
        factory = Mock(return_value=None)
        assert ui._lazy_singleton("_chain_indexer", factory) is None
        # Should retry next call (cache wasn't populated).
        assert ui._chain_indexer is None
        ui._lazy_singleton("_chain_indexer", factory)
        assert factory.call_count == 2

    def test_returns_cached_without_calling_factory(self):
        ui._analytics = "already_built"
        factory = Mock()
        out = ui._lazy_singleton("_analytics", factory)
        assert out == "already_built"
        factory.assert_not_called()


# =============================================================================
# get_index / get_analytics_instance / get_chain_indexer_instance
# =============================================================================


class TestGetIndex:
    def test_calls_load_index_and_returns(self):
        ui._index = None
        with patch.object(ui, "CompassIndex") as MockIdx:
            instance = MagicMock()
            instance.load_index.return_value = True
            MockIdx.return_value = instance
            out = ui.get_index()
        assert out is instance
        # singleton was published
        assert ui._index is instance

    def test_load_index_failure_raises_runtime_error(self):
        ui._index = None
        with patch.object(ui, "CompassIndex") as MockIdx:
            instance = MagicMock()
            instance.load_index.return_value = False
            MockIdx.return_value = instance
            with pytest.raises(RuntimeError, match="Failed to load index"):
                ui.get_index()
        # global stays None so the next caller retries
        assert ui._index is None


class TestGetAnalyticsInstance:
    def test_builds_and_caches(self):
        ui._analytics = None
        fake = MagicMock()
        fake.load_hot_cache_from_db = AsyncMock(return_value=None)
        with patch.object(ui, "get_analytics", return_value=fake):
            out = ui.get_analytics_instance()
        assert out is fake
        assert ui._analytics is fake

    def test_hot_cache_load_raises_keeps_global_none(self):
        ui._analytics = None
        fake = MagicMock()

        async def boom():
            raise sqlite3.OperationalError("db locked")

        fake.load_hot_cache_from_db = boom
        with patch.object(ui, "get_analytics", return_value=fake):
            with pytest.raises(sqlite3.OperationalError):
                ui.get_analytics_instance()
        assert ui._analytics is None


class TestGetChainIndexerInstance:
    def test_returns_none_when_disabled(self):
        ui._chain_indexer = None
        ui._config = MagicMock(chain_indexing_enabled=False)
        assert ui.get_chain_indexer_instance() is None

    def test_builds_when_enabled(self):
        ui._chain_indexer = None
        ui._config = MagicMock(chain_indexing_enabled=True)

        fake_index = MagicMock()
        fake_index.embedder = "emb"
        fake_analytics = MagicMock()
        fake_ci = MagicMock()
        fake_ci.load_chain_index = AsyncMock(return_value=None)

        with patch.object(ui, "get_index", return_value=fake_index), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch.object(ui, "get_chain_indexer", return_value=fake_ci):
            out = ui.get_chain_indexer_instance()
        assert out is fake_ci
        assert ui._chain_indexer is fake_ci


# =============================================================================
# _check_ollama_banner
# =============================================================================


class TestCheckOllamaBanner:
    def test_healthy_returns_empty(self):
        ui._config = MagicMock(ollama_url="http://localhost:11434")
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=True)
        fake_embedder.close = AsyncMock()
        with patch("embedder.Embedder", return_value=fake_embedder):
            out = ui._check_ollama_banner()
        assert out == ""

    def test_unhealthy_returns_banner(self):
        ui._config = MagicMock(ollama_url="http://localhost:11434")
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=False)
        fake_embedder.close = AsyncMock()
        with patch("embedder.Embedder", return_value=fake_embedder):
            out = ui._check_ollama_banner()
        assert "Ollama unavailable" in out
        assert "ollama serve" in out

    def test_exception_returns_generic_banner(self):
        ui._config = MagicMock(ollama_url="http://localhost:11434")
        with patch("embedder.Embedder", side_effect=RuntimeError("boom")):
            out = ui._check_ollama_banner()
        # Defensive default banner is emitted on any probe failure
        assert "Ollama unavailable" in out

    def test_closes_embedder_when_health_check_raises(self):
        """FE-SB-003: close() must run even when health_check() raises so the
        aiohttp session does not leak on the common Ollama-down path.
        """
        ui._config = MagicMock(ollama_url="http://localhost:11434")
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(
            side_effect=ConnectionError("ollama down")
        )
        fake_embedder.close = AsyncMock()
        with patch("embedder.Embedder", return_value=fake_embedder):
            out = ui._check_ollama_banner()
        # Probe failed → defensive banner; close() still ran (try/finally).
        assert "Ollama unavailable" in out
        fake_embedder.close.assert_awaited()


# =============================================================================
# search_tools — the main handler
# =============================================================================


class TestSearchTools:
    def test_empty_query_returns_placeholder(self):
        html_out, json_out = ui.search_tools("", top_k=5)
        assert "Enter a search query" in html_out
        assert json_out == "{}"

    def test_whitespace_only_query_returns_placeholder(self):
        html_out, json_out = ui.search_tools("   \t\n", top_k=5)
        assert "Enter a search query" in html_out

    def test_query_collapsed_to_empty_after_sanitize(self):
        # Pure control chars sanitize to "" → fall into the "Please enter a
        # valid search query" branch.
        html_out, json_out = ui.search_tools("\x00\x07", top_k=5)
        assert "valid search query" in html_out or "Please enter" in html_out

    def test_index_failure_renders_error_card(self):
        ui._index = None
        # Note: format_error routes errors containing "index" into the
        # "Index not ready" branch; use a generic exception message so we
        # land in the generic "Something went wrong" card and can match the
        # caller-supplied context line.
        with patch.object(ui, "get_index", side_effect=RuntimeError("boom")):
            html_out, json_out = ui.search_tools("read file", top_k=5)
        assert "Could not load the tool index" in html_out
        assert json_out == "{}"

    def test_happy_path_renders_results(self):
        # Index is populated (total_tools > 0) — the precondition for the
        # search path. An empty index now short-circuits to the sync card
        # (FE-SB-002), so search-path tests must seed a non-empty index.
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])

        async def fake_search(query, top_k, category_filter, server_filter):
            return [_fake_result("test:read_file", "Read a file", 0.85)]

        idx.search = fake_search
        with patch.object(ui, "get_index", return_value=idx):
            html_out, json_out = ui.search_tools("read", top_k=5,
                                                  min_confidence=0.3)
        assert "test:read_file" in html_out
        assert "Read a file" in html_out
        # confidence label visible
        assert "Excellent" in html_out
        data = json.loads(json_out)
        assert data[0]["tool"] == "test:read_file"
        assert data[0]["confidence"] == 0.85

    def test_results_filtered_by_min_confidence(self):
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])

        async def fake_search(**kwargs):
            return [
                _fake_result("good:tool", "good", 0.9),
                _fake_result("bad:tool", "bad", 0.1),
            ]

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, _ = ui.search_tools("q", min_confidence=0.5)
        assert "good:tool" in html_out
        assert "bad:tool" not in html_out

    def test_xss_in_tool_metadata_is_escaped(self):
        """FE-A2-001: tool names from MCP servers are untrusted."""
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])

        async def fake_search(**kwargs):
            return [_fake_result(
                "<script>alert(1)</script>",
                "<img src=x onerror=alert(1)>",
                0.9, server="<svg/>", category="<b>", parameters={})]

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, _ = ui.search_tools("q")
        # Raw script/HTML must NOT survive escaping
        assert "<script>alert(1)</script>" not in html_out
        assert "<img src=x" not in html_out
        # Escaped form is present
        assert "&lt;script&gt;" in html_out

    def test_query_is_escaped_in_count_heading(self):
        """The result count heading interpolates the query — escape it."""
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])

        async def fake_search(**kwargs):
            return [_fake_result("a:b", "desc", 0.9)]

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, _ = ui.search_tools("<script>", top_k=5)
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_deprecated_badge_rendered(self):
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])

        async def fake_search(**kwargs):
            return [_fake_result("old:tool", "deprecated thing", 0.8,
                                 deprecated_since="2.0.0")]

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, _ = ui.search_tools("q")
        assert "Deprecated since v2.0.0" in html_out

    def test_zero_results_renders_did_you_mean(self):
        idx = _make_mock_index(rows=[
            {"name": "test:read_file", "description": "Read a file",
             "category": "file", "server": "test"},
        ])

        async def fake_search(**kwargs):
            return []

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, json_out = ui.search_tools("read", top_k=5)
        assert "No tools found" in html_out
        # Did-you-mean uses the lexical fallback; with our seeded rows it
        # should find test:read_file.
        assert "test:read_file" in html_out
        assert json_out == "{}"

    def test_empty_index_short_circuits_to_sync_card(self):
        """FE-SB-002: a synced-but-empty index (0 rows) must surface the
        actionable 'No tools indexed yet → tool-compass sync' card, NOT the
        generic no-match page whose advice (reword / lower confidence /
        remove filters) can never produce a tool from an empty index.
        """
        idx = _make_mock_index(rows=[])  # get_stats → total_tools == 0
        search_called = {"n": 0}

        async def fake_search(**kwargs):
            search_called["n"] += 1
            return []

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, json_out = ui.search_tools("read", top_k=5)
        # Shows the empty-index next-step, matching the Browser tab.
        assert "No tools indexed yet" in html_out
        assert "tool-compass sync" in html_out
        # Must NOT fall through to the generic no-match advice.
        assert "No tools found" not in html_out
        assert json_out == "{}"
        # Short-circuited before ever running the search.
        assert search_called["n"] == 0

    def test_semantic_failure_triggers_lexical_fallback_banner(self):
        idx = _make_mock_index(rows=[
            {"name": "test:read_file", "description": "Read a file",
             "category": "file", "server": "test"},
        ])

        async def fake_search(**kwargs):
            raise ConnectionError("ollama down")

        idx.search = lambda **kw: fake_search(**kw)
        with patch.object(ui, "get_index", return_value=idx):
            html_out, json_out = ui.search_tools("read", top_k=5,
                                                  min_confidence=0.3)
        # The inline fallback banner should be present.
        assert "keyword-based results" in html_out.lower()
        assert "test:read_file" in html_out
        data = json.loads(json_out)
        assert data[0]["degraded"] is True

    def test_category_and_server_filters_passed_to_search(self):
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])
        captured = {}

        async def fake_search(query, top_k, category_filter, server_filter):
            captured["category"] = category_filter
            captured["server"] = server_filter
            return [_fake_result("x:y", "desc", 0.9)]

        idx.search = fake_search
        with patch.object(ui, "get_index", return_value=idx):
            ui.search_tools("q", top_k=3, category="file", server="test")
        assert captured["category"] == "file"
        assert captured["server"] == "test"

    def test_all_filter_value_becomes_none(self):
        idx = _make_mock_index(rows=[
            {"name": "seed:tool", "description": "seed", "category": "file",
             "server": "seed"},
        ])
        captured = {}

        async def fake_search(query, top_k, category_filter, server_filter):
            captured["category"] = category_filter
            captured["server"] = server_filter
            return [_fake_result("x:y", "d", 0.9)]

        idx.search = fake_search
        with patch.object(ui, "get_index", return_value=idx):
            ui.search_tools("q", category="All", server="All")
        assert captured["category"] is None
        assert captured["server"] is None


# =============================================================================
# search_chains
# =============================================================================


class TestSearchChains:
    def test_empty_query_returns_placeholder(self):
        out = ui.search_chains("")
        assert "Enter a query" in out

    def test_query_collapsed_to_empty_after_sanitize(self):
        out = ui.search_chains("\x00\x07")
        assert "valid search query" in out

    def test_chain_indexer_disabled_returns_message(self):
        with patch.object(ui, "get_chain_indexer_instance", return_value=None):
            out = ui.search_chains("modify a file")
        assert "Chain indexing is disabled" in out

    def test_happy_path_renders_chains(self):
        fake_ci = MagicMock()

        async def fake_search(query, top_k, min_confidence):
            return [_fake_chain_result(
                "modify-flow",
                tools=["bridge:read_file", "bridge:write_file"],
                description="Read then write",
                score=0.85,
                use_count=3,
                is_auto_detected=True,
            )]

        fake_ci.search_chains = fake_search
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.search_chains("modify")
        assert "modify-flow" in out
        assert "read_file" in out
        assert "Auto-detected" in out

    def test_no_results_renders_empty_state(self):
        fake_ci = MagicMock()

        async def fake_search(query, top_k, min_confidence):
            return []

        fake_ci.search_chains = fake_search
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.search_chains("missing")
        assert "No workflows found" in out

    def test_search_failure_renders_format_error(self):
        fake_ci = MagicMock()

        async def fake_search(query, top_k, min_confidence):
            raise RuntimeError("boom")

        fake_ci.search_chains = fake_search
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.search_chains("q")
        assert "Workflow search failed" in out
        assert "RuntimeError" in out

    def test_query_escaped_in_error_context(self):
        """Query is passed through html.escape in the failure path."""
        fake_ci = MagicMock()

        async def fake_search(query, top_k, min_confidence):
            raise RuntimeError("boom")

        fake_ci.search_chains = fake_search
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            # search_chains escapes the query at the call boundary
            # (FE-A2-001) before passing to format_error, which then escapes
            # context again — net result is double-escaped &amp;lt;scriptz&amp;gt;.
            # Either way the raw "<scriptz>" never survives.
            out = ui.search_chains("<scriptz>")
        assert "<scriptz>" not in out
        # The double-escape produces &amp;lt; in the output.
        assert "&amp;lt;scriptz&amp;gt;" in out or "&lt;scriptz&gt;" in out

    def test_manual_chain_badge(self):
        fake_ci = MagicMock()

        async def fake_search(query, top_k, min_confidence):
            return [_fake_chain_result(
                "manual-flow", ["a:b", "c:d"], "desc", 0.7,
                is_auto_detected=False)]

        fake_ci.search_chains = fake_search
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.search_chains("q")
        assert "Manual" in out


# =============================================================================
# get_all_tools / filter_tools / get_tool_details
# =============================================================================


class TestGetAllTools:
    def test_returns_rows(self):
        idx = _make_mock_index(rows=[
            {"name": "a:tool", "description": "desc", "category": "file",
             "server": "a", "parameters": {"x": "str"}, "examples": ["ex"]},
        ])
        with patch.object(ui, "get_index", return_value=idx):
            tools = ui.get_all_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "a:tool"
        assert tools[0]["parameters"] == {"x": "str"}
        assert tools[0]["examples"] == ["ex"]

    def test_no_db_returns_empty(self):
        idx = MagicMock()
        idx.db = None
        with patch.object(ui, "get_index", return_value=idx):
            assert ui.get_all_tools() == []

    def test_exception_returns_empty(self):
        with patch.object(ui, "get_index", side_effect=RuntimeError("boom")):
            assert ui.get_all_tools() == []


class TestFilterTools:
    def test_empty_index_renders_sync_prompt(self):
        with patch.object(ui, "get_all_tools", return_value=[]):
            out = ui.filter_tools("All", "All", "")
        assert "No tools indexed" in out
        assert "tool-compass sync" in out

    def test_filter_by_server(self):
        tools = [
            {"name": "a:t1", "description": "d1", "category": "file",
             "server": "a", "parameters": {}, "examples": []},
            {"name": "b:t2", "description": "d2", "category": "file",
             "server": "b", "parameters": {}, "examples": []},
        ]
        with patch.object(ui, "get_all_tools", return_value=tools):
            out = ui.filter_tools("a", "All", "")
        assert "a:t1" in out
        assert "b:t2" not in out

    def test_filter_by_category(self):
        tools = [
            {"name": "a:t1", "description": "d", "category": "file",
             "server": "a", "parameters": {}, "examples": []},
            {"name": "a:t2", "description": "d", "category": "web",
             "server": "a", "parameters": {}, "examples": []},
        ]
        with patch.object(ui, "get_all_tools", return_value=tools):
            out = ui.filter_tools("All", "web", "")
        assert "a:t2" in out
        assert "a:t1" not in out

    def test_filter_by_search_text(self):
        tools = [
            {"name": "alpha:read", "description": "alpha", "category": "file",
             "server": "alpha", "parameters": {}, "examples": []},
            {"name": "beta:write", "description": "beta", "category": "file",
             "server": "beta", "parameters": {}, "examples": []},
        ]
        with patch.object(ui, "get_all_tools", return_value=tools):
            out = ui.filter_tools("All", "All", "alpha")
        assert "alpha:read" in out
        assert "beta:write" not in out

    def test_no_matches_after_filter_renders_message(self):
        tools = [
            {"name": "a:t", "description": "d", "category": "file",
             "server": "a", "parameters": {}, "examples": []},
        ]
        with patch.object(ui, "get_all_tools", return_value=tools):
            out = ui.filter_tools("nonexistent", "All", "")
        assert "No tools match the current filters" in out

    def test_get_all_tools_failure_renders_format_error(self):
        with patch.object(ui, "get_all_tools", side_effect=RuntimeError("fail")):
            out = ui.filter_tools("All", "All", "")
        assert "Could not load tools" in out

    def test_tool_name_is_escaped_in_browser(self):
        tools = [
            {"name": "<script>", "description": "<b>x</b>",
             "category": "<i>", "server": "<svg/>",
             "parameters": {}, "examples": []},
        ]
        with patch.object(ui, "get_all_tools", return_value=tools):
            out = ui.filter_tools("All", "All", "")
        # Raw HTML must not survive
        assert "<script>" not in out.replace("&lt;script&gt;", "")
        assert "&lt;script&gt;" in out


class TestGetToolDetails:
    def test_empty_input_returns_placeholder(self):
        out = ui.get_tool_details("")
        assert "Enter a tool name" in out

    def test_invalid_input_returns_validation_message(self):
        out = ui.get_tool_details("\x00\x07")
        assert "valid tool name" in out

    def test_index_unavailable_returns_error_card(self):
        # Use a generic exception message — "index" or "not loaded" routes
        # into the "Index not ready" branch which uses a different copy.
        with patch.object(ui, "get_index", side_effect=RuntimeError("boom")):
            out = ui.get_tool_details("a:tool")
        assert "Could not load tool index" in out

    def test_index_db_none_returns_error_card(self):
        idx = MagicMock()
        idx.db = None
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("a:tool")
        # format_error routes "Index not loaded" RuntimeError into the
        # "Index not ready" card — assert on the rendered card copy.
        assert "Index not ready" in out

    def test_exact_match_renders_details(self):
        idx = _make_mock_index(rows=[
            {"name": "test:read_file",
             "description": "Read file from disk",
             "category": "file", "server": "test",
             "parameters": {"path": "str"},
             "examples": ["read file"]},
        ])
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("test:read_file")
        assert "test:read_file" in out
        assert "Read file from disk" in out
        # parameter table rendered
        assert "Parameters (1)" in out
        assert "path" in out
        # examples rendered
        assert "Examples (1)" in out
        assert "read file" in out

    def test_tool_with_no_params_renders_no_params_text(self):
        idx = _make_mock_index(rows=[
            {"name": "noparams:tool", "description": "d",
             "category": "c", "server": "s",
             "parameters": {}, "examples": []},
        ])
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("noparams:tool")
        assert "No parameters required" in out

    def test_partial_match_works(self):
        idx = _make_mock_index(rows=[
            {"name": "fully:qualified:tool", "description": "d",
             "category": "c", "server": "s",
             "parameters": {}, "examples": []},
        ])
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("qualified")
        assert "fully:qualified:tool" in out

    def test_not_found_renders_helpful_message(self):
        idx = _make_mock_index(rows=[])
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("nonexistent")
        assert "Tool not found" in out
        assert "nonexistent" in out

    def test_sqlite_error_rendered(self):
        idx = MagicMock()
        idx.db = MagicMock()
        idx.db.execute.side_effect = sqlite3.OperationalError("locked")
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("a:b")
        assert "Could not search for tool" in out

    def test_xss_in_tool_name_input_is_escaped_in_not_found(self):
        idx = _make_mock_index(rows=[])
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("<script>x</script>")
        assert "<script>x</script>" not in out
        assert "&lt;script&gt;" in out

    def test_corrupt_parameters_json_renders_gracefully(self):
        """FE-SA-001: a malformed parameters blob must not raise out of the
        callback. The json.loads on row['parameters'] sits outside the
        try/except that closes before it; without a guard a JSONDecodeError
        propagates and show_error=True dumps a raw traceback instead of the
        styled card. The fix mirrors get_all_tools (defaults to {}/[]).
        """
        idx = _make_mock_index(rows=[
            {"name": "corrupt:tool", "description": "d",
             "category": "file", "server": "s",
             "parameters": {}, "examples": []},
        ])
        # Overwrite the row with a non-JSON parameters blob the helper can't
        # produce on its own (it always json.dumps). This is the real on-disk
        # corruption the finding describes.
        idx.db.execute(
            "UPDATE tools SET parameters = ? WHERE name = ?",
            ("{not valid json", "corrupt:tool"),
        )
        idx.db.commit()
        with patch.object(ui, "get_index", return_value=idx):
            # Must not raise JSONDecodeError; renders the tool card.
            out = ui.get_tool_details("corrupt:tool")
        assert "corrupt:tool" in out
        # Falls back to no-params view rather than crashing.
        assert "No parameters required" in out

    def test_corrupt_examples_json_renders_gracefully(self):
        """FE-SA-001: the sibling json.loads on row['examples'] is equally
        unguarded — a malformed examples blob must also degrade gracefully.
        """
        idx = _make_mock_index(rows=[
            {"name": "badex:tool", "description": "d",
             "category": "file", "server": "s",
             "parameters": {"path": "str"}, "examples": []},
        ])
        idx.db.execute(
            "UPDATE tools SET examples = ? WHERE name = ?",
            ("[broken json", "badex:tool"),
        )
        idx.db.commit()
        with patch.object(ui, "get_index", return_value=idx):
            out = ui.get_tool_details("badex:tool")
        assert "badex:tool" in out
        # Parameters still render (only examples were corrupt).
        assert "Parameters (1)" in out
        # Corrupt examples default to [] → no Examples section.
        assert "Examples (" not in out


# =============================================================================
# get_analytics_dashboard
# =============================================================================


class TestGetAnalyticsDashboard:
    def test_happy_path_renders_metrics(self):
        fake = MagicMock()

        async def fake_summary(timeframe):
            return {
                "searches": {
                    "total": 12,
                    "avg_latency_ms": 84,
                    "top_queries": [{"query": "read file", "count": 7}],
                },
                "tool_calls": {
                    "total": 5,
                    "success_rate": 80,
                    "top_tools": [{
                        "tool": "test:read",
                        "calls": 3,
                        "success_rate": 100,
                        "avg_latency_ms": 25,
                    }],
                },
                "failures": [
                    {"tool": "test:bad", "error": "boom", "count": 2},
                ],
                "hot_cache": {"size": 2, "tools": ["test:read", "test:write"]},
            }

        fake.get_analytics_summary = fake_summary
        with patch.object(ui, "get_analytics_instance", return_value=fake):
            out = ui.get_analytics_dashboard("24h")
        # Top metrics tiles
        assert ">12<" in out  # searches total
        assert ">5<" in out   # tool calls total
        assert "80%" in out   # success rate
        # Section headers
        assert "Top Tools" in out
        assert "Top Queries" in out
        assert "Recent Failures" in out
        assert "Hot Cache" in out
        # Specific data points
        assert "test:read" in out
        assert "read file" in out
        assert "test:bad" in out

    def test_empty_summary_skips_optional_sections(self):
        # UX-C-002: use nonzero totals so this exercises the "has activity but
        # no optional sections" path, distinct from the all-zero empty state.
        fake = MagicMock()

        async def fake_summary(timeframe):
            return {
                "searches": {"total": 3, "avg_latency_ms": 0, "top_queries": []},
                "tool_calls": {"total": 0, "success_rate": 0, "top_tools": []},
            }

        fake.get_analytics_summary = fake_summary
        with patch.object(ui, "get_analytics_instance", return_value=fake):
            out = ui.get_analytics_dashboard("1h")
        assert "Top Tools" not in out
        assert "Top Queries" not in out
        assert "Recent Failures" not in out
        # Normal stat cards still render when there is activity.
        assert "Searches (1h)" in out

    def test_all_zero_shows_first_run_empty_state(self):
        # UX-C-002: fresh install (no searches AND no tool calls) shows a
        # guidance card instead of "0 / 0 / 0% / 0ms" stat tiles.
        fake = MagicMock()

        async def fake_summary(timeframe):
            return {
                "searches": {"total": 0, "avg_latency_ms": 0, "top_queries": []},
                "tool_calls": {"total": 0, "success_rate": 0, "top_tools": []},
            }

        fake.get_analytics_summary = fake_summary
        with patch.object(ui, "get_analytics_instance", return_value=fake):
            out = ui.get_analytics_dashboard("24h")
        assert "No activity recorded yet" in out
        # The zero-value stat tiles must NOT render in the empty state.
        assert "Searches (24h)" not in out
        assert "Success Rate" not in out
        # Decorative emoji is aria-hidden (UX-C-003 pattern).
        assert 'aria-hidden="true"' in out

    def test_only_searches_renders_normal_cards(self):
        # UX-C-002: the empty state gates on BOTH totals being zero. If there
        # is any search activity, the normal dashboard renders.
        fake = MagicMock()

        async def fake_summary(timeframe):
            return {
                "searches": {"total": 1, "avg_latency_ms": 5, "top_queries": []},
                "tool_calls": {"total": 0, "success_rate": 0, "top_tools": []},
            }

        fake.get_analytics_summary = fake_summary
        with patch.object(ui, "get_analytics_instance", return_value=fake):
            out = ui.get_analytics_dashboard("24h")
        assert "No activity recorded yet" not in out
        assert "Searches (24h)" in out

    def test_query_in_top_queries_is_escaped(self):
        fake = MagicMock()

        async def fake_summary(timeframe):
            return {
                "searches": {
                    "total": 1, "avg_latency_ms": 0,
                    "top_queries": [{"query": "<img src=x>", "count": 1}],
                },
                "tool_calls": {"total": 0, "success_rate": 0, "top_tools": []},
            }

        fake.get_analytics_summary = fake_summary
        with patch.object(ui, "get_analytics_instance", return_value=fake):
            out = ui.get_analytics_dashboard("24h")
        assert "<img src=x>" not in out
        assert "&lt;img" in out

    def test_failure_renders_error_card(self):
        with patch.object(ui, "get_analytics_instance",
                          side_effect=RuntimeError("db gone")):
            out = ui.get_analytics_dashboard("24h")
        assert "Could not load analytics data" in out


# =============================================================================
# get_chains_view
# =============================================================================


class TestGetChainsView:
    def test_disabled_returns_message(self):
        with patch.object(ui, "get_chain_indexer_instance", return_value=None):
            out = ui.get_chains_view()
        assert "Chain indexing is disabled" in out
        # UX-C-003: decorative emoji in the disabled empty state is aria-hidden.
        assert 'aria-hidden="true"' in out

    def test_empty_chains_renders_help_text(self):
        fake_ci = MagicMock()

        async def load():
            return []

        fake_ci.load_chains_from_db = load
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.get_chains_view()
        assert "No workflows defined yet" in out
        # UX-C-003: decorative emoji in the no-workflows empty state is aria-hidden.
        assert 'aria-hidden="true"' in out

    def test_chains_rendered(self):
        fake_ci = MagicMock()

        async def load():
            return [
                SimpleNamespace(
                    name="flow-1",
                    tools=["a:read", "a:write"],
                    description="read then write",
                    use_count=4,
                    is_auto_detected=True,
                ),
                SimpleNamespace(
                    name="flow-2",
                    tools=["b:get", "b:put"],
                    description="manual flow",
                    use_count=1,
                    is_auto_detected=False,
                ),
            ]

        fake_ci.load_chains_from_db = load
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.get_chains_view()
        assert "flow-1" in out
        assert "flow-2" in out
        assert "Auto-detected" in out
        assert "Manual" in out
        # UX-D-004: chain cards use the refreshed tonal ladder — card base
        # #22223e + border #3a3a52 — matching Search/Browser (no stale #444).
        assert "background: #22223e" in out
        assert "border: 1px solid #3a3a52" in out
        assert "#444" not in out

    def test_load_failure_renders_error(self):
        fake_ci = MagicMock()

        async def load():
            raise RuntimeError("db down")

        fake_ci.load_chains_from_db = load
        with patch.object(ui, "get_chain_indexer_instance", return_value=fake_ci):
            out = ui.get_chains_view()
        assert "Could not load workflows" in out


# =============================================================================
# get_system_status
# =============================================================================


class TestGetSystemStatus:
    def _make_config(self, **overrides):
        cfg = MagicMock()
        cfg.embedding_model = "nomic-embed-text"
        cfg.progressive_disclosure = True
        cfg.auto_sync = True
        cfg.analytics_enabled = True
        cfg.chain_indexing_enabled = True
        cfg.hot_cache_size = 10
        cfg.backends = {"a": MagicMock(), "b": MagicMock()}
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_all_healthy(self):
        ui._config = self._make_config()
        idx = _make_mock_index(rows=[
            {"name": "a:t", "description": "d", "category": "c", "server": "s",
             "parameters": {}, "examples": []},
        ])
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {"a:t": None}
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=True)
        fake_embedder.close = AsyncMock()
        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", return_value=fake_embedder):
            out = ui.get_system_status()
        assert "Passed" in out
        assert "Loaded" in out
        assert "Available" in out
        assert "Connected" in out
        assert "Total tools" in out
        assert "Chain indexing" in out

    def test_index_load_failure_shows_warning(self):
        ui._config = self._make_config()
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {}
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=False)
        fake_embedder.close = AsyncMock()
        with patch.object(ui, "get_index", side_effect=RuntimeError("boom")), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", return_value=fake_embedder):
            out = ui.get_system_status()
        assert "Not loaded" in out
        # Ollama unhealthy → Model not loaded
        assert "Model not loaded" in out

    def test_analytics_failure_shows_warning(self):
        ui._config = self._make_config()
        idx = _make_mock_index(rows=[])
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=True)
        fake_embedder.close = AsyncMock()
        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          side_effect=RuntimeError("analytics dead")), \
             patch("embedder.Embedder", return_value=fake_embedder):
            out = ui.get_system_status()
        # analytics status falls into the Warning branch with the truncated err
        assert "analytics dead" in out

    def test_ollama_exception_shows_failed(self):
        ui._config = self._make_config()
        idx = _make_mock_index(rows=[])
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {}
        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", side_effect=RuntimeError("conn refused")):
            out = ui.get_system_status()
        assert "Failed" in out
        assert "Unavailable" in out

    def test_ollama_probe_uses_configured_url_and_model(self):
        """FE-SB-001: the Status-tab probe must build the Embedder from the
        loaded config (ollama_url + embedding_model), NOT a bare Embedder()
        that uses the hardcoded module defaults — otherwise it reports the
        wrong endpoint's health and contradicts _check_ollama_banner().
        """
        ui._config = self._make_config(
            ollama_url="http://remote-host:9999",
            embedding_model="custom-embed",
        )
        idx = _make_mock_index(rows=[])
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {}
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=True)
        fake_embedder.close = AsyncMock()
        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return fake_embedder

        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", side_effect=_capture):
            out = ui.get_system_status()
        # Probe constructed against the CONFIGURED endpoint + model.
        assert captured.get("base_url") == "http://remote-host:9999"
        assert captured.get("model") == "custom-embed"
        assert "Connected" in out

    def test_ollama_probe_closes_embedder_when_health_check_raises(self):
        """FE-SB-003: close() must run even when health_check() raises (the
        common Ollama-down path) so the aiohttp session never leaks.
        """
        ui._config = self._make_config()
        idx = _make_mock_index(rows=[])
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {}
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(
            side_effect=ConnectionError("ollama down")
        )
        fake_embedder.close = AsyncMock()
        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", return_value=fake_embedder):
            out = ui.get_system_status()
        # The raised health_check still routes to the Failed branch...
        assert "Failed" in out
        assert "Unavailable" in out
        # ...AND close() ran despite the raise (try/finally), so no leak.
        fake_embedder.close.assert_awaited()

    def test_config_load_failure_returns_error(self):
        ui._config = None
        with patch.object(ui, "load_config", side_effect=RuntimeError("no cfg")):
            out = ui.get_system_status()
        assert "Could not load configuration" in out

    def test_xss_in_server_name_escaped(self):
        ui._config = self._make_config()
        idx = MagicMock()
        idx.db = MagicMock()
        idx.index_path = "/p"
        idx.get_stats.return_value = {
            "total_tools": 1,
            "core_tools": 0,
            "by_server": {"<script>evil</script>": 1},
            "by_category": {"<b>cat</b>": 1},
        }
        fake_analytics = MagicMock()
        fake_analytics._hot_cache = {}
        fake_embedder = MagicMock()
        fake_embedder.health_check = AsyncMock(return_value=True)
        fake_embedder.close = AsyncMock()
        with patch.object(ui, "get_index", return_value=idx), \
             patch.object(ui, "get_analytics_instance",
                          return_value=fake_analytics), \
             patch("embedder.Embedder", return_value=fake_embedder):
            out = ui.get_system_status()
        assert "<script>evil</script>" not in out
        assert "&lt;script&gt;" in out


# =============================================================================
# get_filter_choices
# =============================================================================


class TestGetFilterChoices:
    def test_happy_path(self):
        idx = MagicMock()
        idx.get_stats.return_value = {
            "by_server": {"a": 1, "b": 2},
            "by_category": {"file": 3, "web": 1},
        }
        with patch.object(ui, "get_index", return_value=idx):
            servers, categories = ui.get_filter_choices()
        assert servers == ["All", "a", "b"]
        assert categories == ["All", "file", "web"]

    def test_index_failure_returns_defaults(self):
        with patch.object(ui, "get_index", side_effect=RuntimeError("no index")):
            servers, categories = ui.get_filter_choices()
        assert servers == ["All"]
        assert categories == ["All"]


# =============================================================================
# run_async — both branches
# =============================================================================


class TestRunAsync:
    def test_no_running_loop_uses_asyncio_run(self):
        """Top-level call (no running loop) takes the asyncio.run path."""

        async def coro():
            return 42

        assert ui.run_async(coro()) == 42

    def test_inside_running_loop_dispatches_to_thread(self):
        """Gradio runs handlers inside an event loop — must not crash."""

        async def driver():
            async def inner():
                return "ok"
            # Calling run_async inside a running loop must succeed via the
            # worker-thread fallback.
            return ui.run_async(inner())

        assert asyncio.run(driver()) == "ok"
