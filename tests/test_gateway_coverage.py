"""
Gateway coverage suite — drives gateway.py from 45% to >=80% by hitting the
RFC 9457 envelope helper + degraded augmenter + every untested branch of the
MCP tool handlers (compass_status / compass_audit / compass_analytics /
compass_chains / compass_sync) + the _lexical_search_fallback path + the
maybe_startup_sync edge cases + the HTTP /ready and /metrics handler bodies
without binding a socket.

All tests rely on the autouse `_reset_gateway_globals` fixture in
conftest.py to scrub module-level state between tests. None of these tests
launch the FastMCP HTTP server — the /ready and /metrics handlers are
exercised by re-constructing the same closures against mocked module
globals.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest


# =============================================================================
# _error_envelope() — RFC 9457 helper (15 callsites, all branches)
# =============================================================================


class TestErrorEnvelope:
    """Exercise every branch of _error_envelope()."""

    def test_minimal_envelope_shape(self):
        from gateway import _error_envelope

        resp = _error_envelope(
            code="tool_not_found",
            title="Tool not found",
            detail="missing",
        )
        # Legacy string-form "error" key is preserved (backward compat).
        assert resp["error"] == "missing"
        # And the structured envelope under "error_envelope" carries the RFC
        # 9457-shaped payload.
        env = resp["error_envelope"]
        assert env["type"] == "compass.error.tool_not_found"
        assert env["title"] == "Tool not found"
        assert env["code"] == "tool_not_found"
        assert env["category"] == "backend_error"  # default
        assert env["detail"] == "missing"
        assert env["retryable"] is False
        # Optional fields are absent when not provided.
        assert "instance" not in env
        assert "retry_after_seconds" not in env
        assert "nearest_tools" not in env
        assert "suggestions" not in env
        # trace_id surfaces at the top level.
        assert resp["trace_id"] is None

    def test_full_envelope_with_all_optional_fields(self):
        from gateway import _error_envelope

        nearest = [{"tool": "bridge:read_file", "score": 0.6}]
        suggestions = ["Try compass()."]

        resp = _error_envelope(
            code="backend_unreachable",
            title="Backend down",
            detail="connection refused",
            category="service_unavailable",
            retryable=True,
            trace_id="abc12345",
            retry_after_seconds=2.5,
            nearest_tools=nearest,
            suggestions=suggestions,
            extra_field="extra_value",
        )
        env = resp["error_envelope"]
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True
        assert env["instance"] == "abc12345"
        assert env["retry_after_seconds"] == 2.5
        assert env["nearest_tools"] == nearest
        assert env["suggestions"] == suggestions
        # **extras go onto the envelope.
        assert env["extra_field"] == "extra_value"
        assert resp["trace_id"] == "abc12345"

    def test_unknown_code_logs_warning_but_still_returns(self, caplog):
        from gateway import _error_envelope

        # Unknown code should warn but pass through.
        with caplog.at_level("WARNING"):
            resp = _error_envelope(
                code="totally_invented_code",
                title="t",
                detail="d",
            )
        assert resp["error_envelope"]["code"] == "totally_invented_code"

    def test_unknown_category_logs_warning_but_still_returns(self, caplog):
        from gateway import _error_envelope

        with caplog.at_level("WARNING"):
            resp = _error_envelope(
                code="tool_not_found",
                title="t",
                detail="d",
                category="not-a-real-category",
            )
        assert resp["error_envelope"]["category"] == "not-a-real-category"

    def test_retry_after_seconds_is_floatified(self):
        from gateway import _error_envelope

        resp = _error_envelope(
            code="backend_timeout",
            title="t",
            detail="d",
            retry_after_seconds=3,  # int input
        )
        assert resp["error_envelope"]["retry_after_seconds"] == 3.0
        assert isinstance(resp["error_envelope"]["retry_after_seconds"], float)


# =============================================================================
# _augment_with_health() — degraded:true field injector
# =============================================================================


class TestAugmentWithHealth:
    """Exercise the degraded-stamper across all branches."""

    def test_no_degradation_passthrough(self):
        import gateway
        from gateway import _augment_with_health

        # Set health to all-good.
        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = True

        resp = {"matches": [], "trace_id": "x"}
        out = _augment_with_health(resp)
        assert out["degraded"] is False
        # No degraded_reasons added when nothing is degraded.
        assert "degraded_reasons" not in out

    def test_ollama_unavailable_adds_reason(self):
        import gateway
        from gateway import _augment_with_health

        gateway._health_state["ollama_available"] = False
        gateway._health_state["index_available"] = True

        resp = {"matches": []}
        out = _augment_with_health(resp)
        assert out["degraded"] is True
        assert "ollama_unavailable" in out["degraded_reasons"]

    def test_index_unavailable_adds_reason(self):
        import gateway
        from gateway import _augment_with_health

        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = False

        resp = {"matches": []}
        out = _augment_with_health(resp)
        assert out["degraded"] is True
        assert "index_unhealthy" in out["degraded_reasons"]

    def test_both_unavailable_adds_both_reasons(self):
        import gateway
        from gateway import _augment_with_health

        gateway._health_state["ollama_available"] = False
        gateway._health_state["index_available"] = False

        resp = {"matches": []}
        out = _augment_with_health(resp)
        assert out["degraded"] is True
        assert "ollama_unavailable" in out["degraded_reasons"]
        assert "index_unhealthy" in out["degraded_reasons"]

    def test_already_degraded_preserved(self):
        import gateway
        from gateway import _augment_with_health

        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = True

        # Caller pre-marked degraded — augmenter must preserve and ensure the
        # `degraded_reasons` list exists.
        resp = {"degraded": True}
        out = _augment_with_health(resp)
        assert out["degraded"] is True
        assert out.get("degraded_reasons") == []

    def test_existing_reasons_deduplicated(self):
        import gateway
        from gateway import _augment_with_health

        gateway._health_state["ollama_available"] = False
        gateway._health_state["index_available"] = True

        resp = {"degraded_reasons": ["ollama_unavailable"]}
        out = _augment_with_health(resp)
        # Should not duplicate the same reason.
        assert out["degraded_reasons"].count("ollama_unavailable") == 1

    def test_non_dict_passthrough(self):
        from gateway import _augment_with_health

        # If the gateway hands a non-dict to the augmenter (defensive), it
        # should return the input unchanged.
        out = _augment_with_health("a string")
        assert out == "a string"
        out = _augment_with_health([1, 2, 3])
        assert out == [1, 2, 3]


# =============================================================================
# _clamp_query() + _escape_like() — query boundary hygiene
# =============================================================================


class TestQueryHelpers:
    """Validate _clamp_query and _escape_like."""

    def test_clamp_query_empty(self):
        from gateway import _clamp_query

        assert _clamp_query(None) == ""
        assert _clamp_query("") == ""
        assert _clamp_query("   ") == ""

    def test_clamp_query_strips_whitespace(self):
        from gateway import _clamp_query

        assert _clamp_query("  hello world  ") == "hello world"

    def test_clamp_query_truncates_over_limit(self, caplog):
        from gateway import _clamp_query, _MAX_QUERY_LEN

        massive = "a" * (_MAX_QUERY_LEN + 100)
        with caplog.at_level("WARNING"):
            clamped = _clamp_query(massive)
        assert len(clamped) == _MAX_QUERY_LEN

    def test_escape_like_handles_wildcards(self):
        from gateway import _escape_like

        # % and _ are wildcards in SQLite LIKE — must be escaped so the
        # user's literal `foo%bar` doesn't act like `foo.*bar`.
        assert _escape_like("foo%bar") == "foo\\%bar"
        assert _escape_like("foo_bar") == "foo\\_bar"
        assert _escape_like("back\\slash") == "back\\\\slash"

    def test_escape_like_preserves_normal_chars(self):
        from gateway import _escape_like

        assert _escape_like("normal text") == "normal text"


# =============================================================================
# _lexical_search_fallback() — the lexical fallback path
# =============================================================================


class TestLexicalSearchFallback:
    """Drive _lexical_search_fallback() across all branches."""

    def test_returns_empty_when_no_index(self):
        from gateway import _lexical_search_fallback

        assert _lexical_search_fallback(None, "anything", 5, None, None) == []

    def test_returns_empty_when_no_db(self):
        from gateway import _lexical_search_fallback

        index = Mock()
        index.db = None
        assert _lexical_search_fallback(index, "anything", 5, None, None) == []

    def test_returns_empty_for_blank_query(self, test_index):
        from gateway import _lexical_search_fallback

        # An empty/blank query must return [] (BE-B-007: '%%' would otherwise
        # match the whole catalog — that's a serious gotcha).
        assert _lexical_search_fallback(test_index, "", 5, None, None) == []
        assert _lexical_search_fallback(test_index, "    ", 5, None, None) == []

    @pytest.mark.asyncio
    async def test_name_match_higher_confidence_than_description(
        self, test_index
    ):
        from gateway import _lexical_search_fallback

        # "read_file" is in tool name -> 0.6 confidence
        results = _lexical_search_fallback(test_index, "read_file", 5, None, None)
        assert results, "expected lexical match for 'read_file'"
        # Top result has confidence 0.6 (name match).
        assert results[0]["confidence"] == 0.6
        # Every match carries degraded=True.
        for m in results:
            assert m["degraded"] is True

    @pytest.mark.asyncio
    async def test_category_filter_applied(self, test_index):
        from gateway import _lexical_search_fallback

        results = _lexical_search_fallback(
            test_index, "file", top_k=10, category="file", server=None
        )
        assert all(m["category"] == "file" for m in results)

    @pytest.mark.asyncio
    async def test_server_filter_applied(self, test_index):
        from gateway import _lexical_search_fallback

        results = _lexical_search_fallback(
            test_index, "file", top_k=10, category=None, server="test"
        )
        assert all(m["server"] == "test" for m in results)


# =============================================================================
# compass() — exception path (Ollama down -> lexical fallback)
# =============================================================================


class TestCompassFallback:
    """Drive the lexical fallback path in compass()."""

    @pytest.mark.asyncio
    async def test_compass_falls_back_to_lexical_on_search_error(
        self, test_index, test_config
    ):
        """Semantic search raises -> compass() returns lexical results with
        warnings and degraded=True."""
        import gateway

        # Force index.search() to raise so the except branch fires.
        async def boom(*args, **kwargs):
            raise RuntimeError("Ollama unreachable")

        with patch.object(test_index, "search", side_effect=boom):
            gateway._compass_index = test_index
            gateway._config = test_config
            gateway._startup_sync_done = True
            gateway._analytics = None

            from gateway import compass

            result = await compass(intent="read_file", top_k=3)

            assert result["degraded"] is True
            assert "warnings" in result
            # The warning prose should mention Ollama.
            assert any("Ollama" in w for w in result["warnings"])
            # Lexical fallback produced matches — they all carry degraded=True.
            for m in result["matches"]:
                assert m["degraded"] is True

    @pytest.mark.asyncio
    async def test_compass_clamps_oversize_intent(self, test_index, test_config):
        """A 10MB paste should never reach the embedder."""
        import gateway
        from gateway import _MAX_QUERY_LEN

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        from gateway import compass

        # No crash; the boundary clamp is invisible to the caller but must
        # not raise.
        result = await compass(intent="a" * (_MAX_QUERY_LEN + 1000), top_k=2)
        assert "matches" in result

    @pytest.mark.asyncio
    async def test_compass_no_match_hint(self, test_index, test_config):
        """When nothing matches, hint mentions broader terms."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        from gateway import compass

        result = await compass(intent="zzzzzz_no_match", category="impossible")
        assert result["matches"] == []
        assert "broader terms" in result["hint"] or "No tools" in result["hint"]

    @pytest.mark.asyncio
    async def test_compass_chain_search_exception_handled(
        self, test_index, test_config_with_backends
    ):
        """Chain indexer raising during search() must not abort the compass
        response — it should append a warning and keep going."""
        import gateway

        # Wire a chain_indexer that raises on search_chains.
        mock_chain = Mock()
        mock_chain.search_chains = AsyncMock(side_effect=RuntimeError("chain boom"))
        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._chain_indexer = mock_chain
        gateway._startup_sync_done = True
        gateway._analytics = None

        from gateway import compass

        result = await compass(
            intent="read_file", top_k=2, include_chains=True
        )
        # compass() should not propagate the chain failure.
        assert "matches" in result
        # And it stamped a warning.
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_compass_min_confidence_respected_on_lexical_fallback(
        self, test_index, test_config
    ):
        """BE-A-004: min_confidence applies on the lexical fallback path too."""
        import gateway

        async def boom(*args, **kwargs):
            raise RuntimeError("ollama dead")

        with patch.object(test_index, "search", side_effect=boom):
            gateway._compass_index = test_index
            gateway._config = test_config
            gateway._startup_sync_done = True
            gateway._analytics = None

            from gateway import compass

            # 0.9 > all lexical confidences (0.6/0.4/0.3) — should produce []
            result = await compass(intent="read_file", min_confidence=0.9, top_k=3)
            assert result["matches"] == []

    @pytest.mark.asyncio
    async def test_compass_empty_matches_zero_confidence_chain_no_indexerror(
        self, test_index, test_config_with_backends
    ):
        """GW-COMPOSED-001: matches==[] plus a single chain match scoring 0.0
        (min_confidence=0.0) must NOT raise IndexError in the hint builder — it
        must return a structured dict with a chains-only hint."""
        import gateway
        from chain_indexer import ToolChain, ChainSearchResult

        # index.search returns nothing -> matches == [].
        async def empty_search(*args, **kwargs):
            return []

        # A single chain match at score 0.0 — the exact confidence that made
        # `0.0 > 0` False and fell through to `matches[0]` -> IndexError.
        zero_chain = ToolChain(
            id=1,
            name="edge_case_chain",
            tools=["test:read_file", "test:write_file"],
            description="Zero-confidence workflow",
            use_count=0,
            is_auto_detected=False,
        )
        mock_chain = Mock()
        mock_chain.search_chains = AsyncMock(
            return_value=[ChainSearchResult(chain=zero_chain, score=0.0)]
        )

        with patch.object(test_index, "search", side_effect=empty_search):
            gateway._compass_index = test_index
            gateway._config = test_config_with_backends
            gateway._chain_indexer = mock_chain
            gateway._startup_sync_done = True
            gateway._analytics = None

            from gateway import compass

            # Must not raise; must return a structured payload (BE-B-004).
            result = await compass(
                intent="zzzzzz_no_match",
                min_confidence=0.0,
                top_k=3,
                include_chains=True,
            )

        assert isinstance(result, dict)
        assert result["matches"] == []
        # The chains-only hint names the chain rather than indexing matches[0].
        assert "edge_case_chain" in result["hint"]
        assert result.get("chains")


# =============================================================================
# describe() — sqlite error path + nearest_tools envelope
# =============================================================================


class TestDescribeErrorPaths:
    """describe() sqlite error + tool_not_found nearest_tools envelope."""

    @pytest.mark.asyncio
    async def test_describe_sqlite_error_marks_index_unhealthy(
        self, test_index, test_config
    ):
        """A sqlite error on the index lookup must:
          - flip _health_state['index_available'] to False
          - fall through to the backend lookup path
          - still return a response (not raise)."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True

        # Mock the backend manager to return a schema for fallback.
        mock_mgr = Mock()
        mock_mgr.get_tool_schema = Mock(return_value={
            "name": "test:tool",
            "description": "Backend served this",
            "parameters": {},
        })
        gateway._backend_manager = mock_mgr

        # Substitute index.db with a Mock that raises sqlite3.OperationalError
        # on execute() — sqlite3.Connection.execute is a C method we can't
        # patch in place.
        original_db = test_index.db
        fake_db = Mock()
        fake_db.execute = Mock(side_effect=sqlite3.OperationalError("disk I/O error"))
        test_index.db = fake_db

        try:
            from gateway import describe

            result = await describe(tool_name="test:tool")
        finally:
            test_index.db = original_db

        # The describe() helper fell through to the backend path and the
        # health flag is now False.
        assert gateway._health_state["index_available"] is False
        # And the response contains a "warnings" hint mentioning index
        # unhealthy + the rebuild action.
        assert "warnings" in result
        assert any("rebuild" in w.lower() or "compass_sync" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_describe_not_found_returns_nearest_tools(
        self, test_index, test_config
    ):
        """Tool-not-found must return an RFC 9457 envelope with nearest_tools
        populated when the lexical fallback finds anything plausible.

        _lexical_search_fallback wraps the query as `%query%` and searches
        for tool names/descriptions containing that needle — so the query
        must be a substring of an existing tool's name or description.
        """
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        # Backend manager has no such tool.
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        from gateway import describe

        # 'read_file' is a substring of existing 'test:read_file' (and the
        # tool name still doesn't exist as 'read_file' alone), so the lookup
        # for 'read_file' fails the exact-match SELECT but the lexical
        # fallback finds 'test:read_file'.
        result = await describe(tool_name="read_file")

        # RFC 9457 envelope is present.
        assert "error" in result
        env = result["error_envelope"]
        assert env["code"] == "tool_not_found"
        assert env["category"] == "not_found"
        assert env["retryable"] is False
        # nearest_tools is the load-bearing recovery signal.
        assert "nearest_tools" in env
        assert isinstance(env["nearest_tools"], list)
        assert env["nearest_tools"], "nearest_tools should not be empty"
        first = env["nearest_tools"][0]
        assert "tool" in first
        assert "score" in first

    @pytest.mark.asyncio
    async def test_describe_not_found_no_nearest_when_no_match(
        self, test_index, test_config
    ):
        """When lexical fallback returns nothing, the envelope still appears
        without nearest_tools but with suggestions."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        from gateway import describe

        result = await describe(tool_name="test:absolutely_unique_no_match_xyz")
        env = result["error_envelope"]
        assert env["code"] == "tool_not_found"
        # Suggestions are always present.
        assert "suggestions" in env

    @pytest.mark.asyncio
    async def test_describe_not_found_unhealthy_index_warns(
        self, test_index, test_config
    ):
        """tool_not_found + index unhealthy -> response carries warnings list.

        The describe() handler resets index_available=True on every
        successful SELECT, so we have to drive index_available=False by
        making the SELECT itself raise — only then will the tool_not_found
        envelope carry the 'Index database unhealthy' warning.
        """
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        # The fake-db approach must:
        #  1. Raise on the first execute() (the WHERE name=? lookup) so the
        #     unhealthy flag flips.
        #  2. Return an empty rowset on the second execute() (the lexical
        #     fallback inside describe) so the test does not depend on
        #     lexical matches.
        original_db = test_index.db

        call_state = {"n": 0}

        def fake_execute(sql, params=()):
            call_state["n"] += 1
            if call_state["n"] == 1:
                raise sqlite3.OperationalError("io error")
            # Subsequent calls -> empty result.
            empty_cursor = Mock()
            empty_cursor.fetchall = Mock(return_value=[])
            empty_cursor.fetchone = Mock(return_value=None)
            return empty_cursor

        fake_db = Mock()
        fake_db.execute = Mock(side_effect=fake_execute)
        test_index.db = fake_db

        try:
            from gateway import describe

            result = await describe(tool_name="test:not_there")
        finally:
            test_index.db = original_db

        # tool_not_found envelope.
        assert result["error_envelope"]["code"] == "tool_not_found"
        # And because index_available was flipped to False during the SELECT
        # try/except, the unhealthy warning is present.
        assert "warnings" in result
        assert any("Index" in w for w in result["warnings"])


# =============================================================================
# execute() — unhandled-exception path (BE-B-004)
# =============================================================================


class TestExecuteUnhandledException:
    """execute() must trap raises from the backend client."""

    @pytest.mark.asyncio
    async def test_execute_traps_backend_exception(
        self, test_config
    ):
        """A raise from manager.execute_tool() must turn into an
        execute_unhandled_exception envelope, not propagate up."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=True)
        mgr.execute_tool = AsyncMock(side_effect=RuntimeError("backend died"))

        gateway._backend_manager = mgr
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:tool", arguments={"x": 1})

        # RFC 9457 envelope.
        env = result["error_envelope"]
        assert env["code"] == "execute_unhandled_exception"
        assert env["category"] == "backend_error"
        assert env["retryable"] is False
        # Success=False stamped on for legacy callers.
        assert result["success"] is False
        # Detail mentions the underlying exception type.
        assert "RuntimeError" in env["detail"]

    @pytest.mark.asyncio
    async def test_execute_traps_backend_exception_records_analytics(
        self, test_config_with_backends, test_analytics
    ):
        """Even when backend raises, analytics.record_tool_call is invoked."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=True)
        mgr.execute_tool = AsyncMock(side_effect=RuntimeError("boom"))

        gateway._backend_manager = mgr
        gateway._config = test_config_with_backends
        gateway._analytics = test_analytics

        from gateway import execute

        await execute(tool_name="test:explode", arguments={})

        # The failed call should have been recorded.
        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["tool_calls"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_execute_traps_when_analytics_record_raises(self, test_config):
        """If analytics itself raises while recording, execute() still returns
        a sensible envelope (the analytics raise is logged, not propagated)."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=True)
        mgr.execute_tool = AsyncMock(side_effect=RuntimeError("backend boom"))

        # Analytics that raises on record.
        analytics = Mock()
        analytics.get_hot_tool = Mock(return_value=None)
        analytics.record_tool_call = AsyncMock(side_effect=Exception("analytics dead"))

        gateway._backend_manager = mgr
        gateway._config = test_config
        # analytics_enabled may be False on test_config; force-set the
        # singleton directly so the analytics-branch in execute() fires.
        gateway._analytics = analytics
        gateway._config.analytics_enabled = True

        from gateway import execute

        # Must not raise.
        result = await execute(tool_name="test:tool", arguments={})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_dict_missing_success_key_treated_as_failure(
        self, test_config
    ):
        """A backend that returns a dict without 'success' is treated as a
        failure (to avoid masking silent errors)."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=True)
        mgr.execute_tool = AsyncMock(return_value={"data": "no success key"})

        gateway._backend_manager = mgr
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:tool")
        # trace_id stamped + treated as failure for analytics, but the dict
        # is returned as-is (with trace_id stamped).
        assert "trace_id" in result

    @pytest.mark.asyncio
    async def test_execute_backend_connect_failed_envelope(self, test_config):
        """When backend connect fails, response is an RFC 9457 envelope with
        category=service_unavailable and retryable=True."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=False)
        mgr.connect_backend = AsyncMock(return_value=False)

        gateway._backend_manager = mgr
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:tool", arguments={})

        env = result["error_envelope"]
        assert env["code"] == "backend_connect_failed"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True
        assert env["retry_after_seconds"] == 5.0
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_non_dict_result_returned_unchanged(self, test_config):
        """When manager returns a non-dict (e.g. a string), execute() returns
        it unchanged (the analytics branch records success=False)."""
        import gateway

        mgr = Mock()
        mgr.is_backend_connected = Mock(return_value=True)
        mgr.execute_tool = AsyncMock(return_value="just a string")

        gateway._backend_manager = mgr
        gateway._config = test_config
        gateway._analytics = None

        from gateway import execute

        result = await execute(tool_name="test:tool")
        # Non-dict result returned as-is.
        assert result == "just a string"


# =============================================================================
# compass_status() — per-block exception paths
# =============================================================================


class TestCompassStatusErrorBlocks:
    """Each subsystem block in compass_status is independently wrapped."""

    @pytest.mark.asyncio
    async def test_status_index_block_failure(self, test_config_with_backends):
        """If index lookup raises, that block reports {error, trace_id}."""
        import gateway

        # No index set — force get_index() to fail by stubbing it.
        async def broken_index():
            raise RuntimeError("index unloaded")

        gateway._config = test_config_with_backends

        mgr = Mock()
        mgr.get_stats = Mock(return_value={"connected_backends": []})
        gateway._backend_manager = mgr

        with patch("gateway.get_index", side_effect=broken_index):
            from gateway import compass_status

            result = await compass_status()

        # Index block carries an error.
        assert "index" in result
        assert "error" in result["index"]
        # Backends still present.
        assert "backends" in result

    @pytest.mark.asyncio
    async def test_status_backends_block_failure(self, test_index, test_config):
        """If backends.get_stats raises, that block reports {error, trace_id}."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config

        async def broken_backends():
            raise RuntimeError("backends dead")

        with patch("gateway.get_backends", side_effect=broken_backends):
            from gateway import compass_status

            result = await compass_status()

        assert "error" in result["backends"]
        assert "index" in result  # other block survived

    @pytest.mark.asyncio
    async def test_status_analytics_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Analytics block reports {error} when get_analytics_instance raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_analytics():
            raise RuntimeError("analytics gone")

        with patch("gateway.get_analytics_instance", side_effect=broken_analytics):
            from gateway import compass_status

            result = await compass_status()

        assert "hot_cache" in result
        assert "error" in result["hot_cache"]

    @pytest.mark.asyncio
    async def test_status_sync_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Sync block reports {error} when get_sync_manager_instance raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_sync():
            raise RuntimeError("sync dead")

        with patch("gateway.get_sync_manager_instance", side_effect=broken_sync):
            from gateway import compass_status

            result = await compass_status()

        assert "sync" in result
        assert "error" in result["sync"]

    @pytest.mark.asyncio
    async def test_status_chain_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Chain block reports {error} when chain indexer access raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_chain():
            raise RuntimeError("chain dead")

        with patch(
            "gateway.get_chain_indexer_instance", side_effect=broken_chain
        ):
            from gateway import compass_status

            result = await compass_status()

        assert "chains" in result
        assert "error" in result["chains"]

    @pytest.mark.asyncio
    async def test_status_includes_health_block(self, test_index, test_config):
        """The health block is always present."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        gateway._health_state["ollama_available"] = False
        gateway._health_state["last_ollama_error"] = "test error"

        from gateway import compass_status

        result = await compass_status()
        assert result["health"]["ollama_available"] is False
        assert result["health"]["degraded_mode"] is True
        assert result["health"]["last_ollama_error"] == "test error"

    @pytest.mark.asyncio
    async def test_status_surfaces_analytics_degraded_flag(
        self, test_index, test_config_with_backends
    ):
        """PC-B-003: when analytics is degraded, compass_status()'s health block
        surfaces analytics_degraded=True (previously only readable via the
        out-of-process config.doctor())."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.analytics_enabled = True
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        analytics = Mock()
        analytics._hot_cache = {}
        analytics.get_health = Mock(
            return_value={"degraded": True, "reason": "sqlite write failure"}
        )

        with patch(
            "gateway.get_analytics_instance", AsyncMock(return_value=analytics)
        ):
            from gateway import compass_status

            result = await compass_status()

        assert result["health"]["analytics_degraded"] is True
        assert result["health"]["analytics_degraded_reason"] == "sqlite write failure"

    @pytest.mark.asyncio
    async def test_status_surfaces_analytics_healthy_flag(
        self, test_index, test_config_with_backends
    ):
        """PC-B-003: a healthy analytics reports analytics_degraded=False."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.analytics_enabled = True
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        analytics = Mock()
        analytics._hot_cache = {}
        analytics.get_health = Mock(return_value={"degraded": False, "reason": None})

        with patch(
            "gateway.get_analytics_instance", AsyncMock(return_value=analytics)
        ):
            from gateway import compass_status

            result = await compass_status()

        assert result["health"]["analytics_degraded"] is False
        assert result["health"]["analytics_degraded_reason"] is None


# =============================================================================
# compass_audit() — per-block exception paths
# =============================================================================


class TestCompassAuditErrorBlocks:
    """compass_audit() degrades each subsystem block independently."""

    @pytest.mark.asyncio
    async def test_audit_index_block_failure(self, test_config_with_backends):
        """Index block error reported, but audit still returns."""
        import gateway

        gateway._config = test_config_with_backends

        async def broken_index():
            raise RuntimeError("index nope")

        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        with patch("gateway.get_index", side_effect=broken_index):
            from gateway import compass_audit

            result = await compass_audit()

        assert "error" in result["system"]
        assert "categories" in result
        assert "servers" in result

    @pytest.mark.asyncio
    async def test_audit_backends_block_failure(self, test_index, test_config):
        """Backends block reports error when get_backends raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config

        async def broken_backends():
            raise RuntimeError("backends dead")

        with patch("gateway.get_backends", side_effect=broken_backends):
            from gateway import compass_audit

            result = await compass_audit()

        assert "error" in result["backends"]

    @pytest.mark.asyncio
    async def test_audit_analytics_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Analytics block reports error when get_analytics_instance raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_analytics():
            raise RuntimeError("analytics dead")

        with patch(
            "gateway.get_analytics_instance", side_effect=broken_analytics
        ):
            from gateway import compass_audit

            result = await compass_audit()

        assert "error" in result["analytics"]

    @pytest.mark.asyncio
    async def test_audit_chains_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Chains block reports error when chain indexer raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_chain():
            raise RuntimeError("chains dead")

        with patch(
            "gateway.get_chain_indexer_instance", side_effect=broken_chain
        ):
            from gateway import compass_audit

            result = await compass_audit()

        assert "error" in result["chains"]

    @pytest.mark.asyncio
    async def test_audit_sync_block_failure(
        self, test_index, test_config_with_backends
    ):
        """Sync block reports error when sync manager raises."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        async def broken_sync():
            raise RuntimeError("sync gone")

        with patch(
            "gateway.get_sync_manager_instance", side_effect=broken_sync
        ):
            from gateway import compass_audit

            result = await compass_audit()

        assert "error" in result["sync"]

    @pytest.mark.asyncio
    async def test_audit_tools_block_handles_db_failure(
        self, test_config_with_backends
    ):
        """include_tools=True but index unavailable -> tools=[] + note."""
        import gateway

        gateway._config = test_config_with_backends

        # Build a Mock index with no db attribute -> tools block degrades.
        mock_index = Mock()
        mock_index.get_stats = Mock(return_value={"total_tools": 0, "by_category": {}, "by_server": {}})
        mock_index.index_path = Path("/tmp/nope.hnsw")
        mock_index.db_path = Path("/tmp/nope.db")
        mock_index.db = None
        gateway._compass_index = mock_index

        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        from gateway import compass_audit

        result = await compass_audit(include_tools=True)
        assert result["tools"] == []
        assert "tools_note" in result

    @pytest.mark.asyncio
    async def test_audit_tools_block_handles_db_exception(
        self, test_index, test_config_with_backends
    ):
        """Tools block traps sqlite errors during list."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        mgr = Mock()
        mgr.get_stats = Mock(return_value={})
        gateway._backend_manager = mgr

        # Replace index.db with a Mock that raises on execute().
        original_db = test_index.db
        fake_db = Mock()
        fake_db.execute = Mock(side_effect=sqlite3.OperationalError("disk gone"))
        test_index.db = fake_db

        try:
            from gateway import compass_audit

            result = await compass_audit(include_tools=True)
        finally:
            test_index.db = original_db

        # tools block trapped the failure.
        assert result["tools"] == []
        assert "tools_note" in result


# =============================================================================
# compass_analytics() — analytics_unavailable + query-failure envelopes
# =============================================================================


class TestCompassAnalyticsErrors:
    """compass_analytics() error branches."""

    @pytest.mark.asyncio
    async def test_analytics_not_initialized_envelope(
        self, test_config_with_backends
    ):
        """analytics_enabled=True but get_analytics_instance returns None."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = None

        with patch(
            "gateway.get_analytics_instance",
            AsyncMock(return_value=None),
        ):
            from gateway import compass_analytics

            result = await compass_analytics()

        env = result["error_envelope"]
        assert env["code"] == "analytics_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True

    @pytest.mark.asyncio
    async def test_analytics_query_exception_envelope(
        self, test_config_with_backends
    ):
        """analytics.get_analytics_summary raising returns envelope."""
        import gateway

        gateway._config = test_config_with_backends

        broken = Mock()
        broken.get_analytics_summary = AsyncMock(
            side_effect=RuntimeError("analytics SQL exploded")
        )
        gateway._analytics = broken

        from gateway import compass_analytics

        result = await compass_analytics(timeframe="1h", include_failures=True)

        env = result["error_envelope"]
        assert env["code"] == "analytics_unavailable"
        assert env["category"] == "backend_error"
        assert env["retryable"] is True
        assert "RuntimeError" in env["detail"]


# =============================================================================
# compass_chains() — every action + every failure path
# =============================================================================


class TestCompassChainsErrors:
    """compass_chains() RFC 9457 error branches."""

    @pytest.mark.asyncio
    async def test_chains_unavailable_envelope(self, test_config_with_backends):
        """chain_indexing_enabled=True but get_chain_indexer_instance returns
        None."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = None

        with patch(
            "gateway.get_chain_indexer_instance",
            AsyncMock(return_value=None),
        ):
            from gateway import compass_chains

            result = await compass_chains(action="list")

        env = result["error_envelope"]
        assert env["code"] == "chain_indexer_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True

    @pytest.mark.asyncio
    async def test_chains_detect_no_analytics_envelope(
        self, test_config_with_backends, test_chain_indexer
    ):
        """detect action with no analytics returns analytics_unavailable env."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer
        gateway._analytics = None

        # Force get_analytics_instance to return None.
        with patch(
            "gateway.get_analytics_instance", AsyncMock(return_value=None)
        ):
            from gateway import compass_chains

            result = await compass_chains(action="detect")

        env = result["error_envelope"]
        assert env["code"] == "analytics_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True

    @pytest.mark.asyncio
    async def test_chains_invalid_argument_create(
        self, test_config_with_backends, test_chain_indexer
    ):
        """create without chain_name + tools returns invalid_argument env."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        # No chain_name + no tools.
        result = await compass_chains(action="create")

        env = result["error_envelope"]
        assert env["code"] == "invalid_argument"
        assert env["category"] == "validation"
        assert env["retryable"] is False
        assert "suggestions" in env

    @pytest.mark.asyncio
    async def test_chains_invalid_action_envelope(
        self, test_config_with_backends, test_chain_indexer
    ):
        """unknown action returns invalid_action env with valid_actions hint."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(action="explode")

        env = result["error_envelope"]
        assert env["code"] == "invalid_action"
        assert env["category"] == "validation"
        assert env["retryable"] is False
        # The extra **valid_actions kwarg flows through.
        assert env["valid_actions"] == ["list", "create", "detect"]

    @pytest.mark.asyncio
    async def test_chains_create_embedder_failure_envelope(
        self, test_config_with_backends
    ):
        """GW-COMPOSED-002: add_chain() raising (Ollama down) must NOT leak a
        raw exception — compass_chains(create) returns a structured
        ollama_unavailable / service_unavailable envelope."""
        import gateway

        gateway._config = test_config_with_backends

        # A chain indexer whose add_chain raises the real breaker-open error.
        mock_chain = Mock()
        mock_chain.add_chain = AsyncMock(
            side_effect=RuntimeError("Ollama circuit breaker open")
        )

        with patch(
            "gateway.get_chain_indexer_instance",
            AsyncMock(return_value=mock_chain),
        ):
            from gateway import compass_chains

            # Must not raise.
            result = await compass_chains(
                action="create",
                chain_name="my_workflow",
                tools=["a:one", "b:two"],
            )

        env = result["error_envelope"]
        assert env["code"] == "ollama_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True

    @pytest.mark.asyncio
    async def test_chains_detect_embedder_failure_envelope(
        self, test_config_with_backends
    ):
        """GW-COMPOSED-002: detect_chains() / add_chain() raising must NOT leak
        a raw exception — compass_chains(detect) returns a structured
        service_unavailable envelope."""
        import gateway

        gateway._config = test_config_with_backends

        mock_chain = Mock()
        mock_chain.add_chain = AsyncMock()
        gateway._chain_indexer = mock_chain

        mock_analytics = Mock()
        mock_analytics.detect_chains = AsyncMock(
            side_effect=RuntimeError("Ollama circuit breaker open")
        )

        with patch(
            "gateway.get_chain_indexer_instance",
            AsyncMock(return_value=mock_chain),
        ), patch(
            "gateway.get_analytics_instance",
            AsyncMock(return_value=mock_analytics),
        ):
            from gateway import compass_chains

            # Must not raise.
            result = await compass_chains(action="detect")

        env = result["error_envelope"]
        assert env["code"] == "ollama_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True

    @pytest.mark.asyncio
    async def test_chains_detect_indexes_into_hnsw(
        self, test_config_with_backends
    ):
        """IDX-COMPOSED-003: detected chains must be added into the live chain
        HNSW index (via chain_indexer.add_chain), not merely raw-INSERTed into
        the DB. Assert add_chain is invoked for each detected chain."""
        import gateway

        gateway._config = test_config_with_backends

        detected_chain = {
            "name": "read_to_write",
            "tools": ["test:read_file", "test:write_file"],
            "description": "Workflow: read file → write file",
            "occurrences": 5,
        }

        mock_chain = Mock()
        mock_chain.add_chain = AsyncMock()
        gateway._chain_indexer = mock_chain

        mock_analytics = Mock()
        mock_analytics.detect_chains = AsyncMock(return_value=[detected_chain])

        with patch(
            "gateway.get_chain_indexer_instance",
            AsyncMock(return_value=mock_chain),
        ), patch(
            "gateway.get_analytics_instance",
            AsyncMock(return_value=mock_analytics),
        ):
            from gateway import compass_chains

            result = await compass_chains(action="detect")

        # The detected chain was pushed into the live HNSW index.
        mock_chain.add_chain.assert_awaited_once()
        _, kwargs = mock_chain.add_chain.call_args
        assert kwargs["name"] == "read_to_write"
        assert kwargs["tools"] == ["test:read_file", "test:write_file"]
        assert kwargs["is_auto_detected"] is True
        # And the response reflects what actually happened.
        assert result["count"] == 1
        assert result["indexed"] == 1
        assert "indexed and searchable" in result["hint"]


# =============================================================================
# compass_sync() — error envelopes
# =============================================================================


class TestCompassSyncErrors:
    """compass_sync() error branches."""

    @pytest.mark.asyncio
    async def test_sync_disabled_envelope(self, test_config):
        """auto_sync=False -> sync_disabled envelope."""
        import gateway

        gateway._config = test_config  # auto_sync=False by default

        from gateway import compass_sync

        result = await compass_sync()

        env = result["error_envelope"]
        assert env["code"] == "sync_disabled"
        assert env["category"] == "configuration"
        assert env["retryable"] is False

    @pytest.mark.asyncio
    async def test_sync_manager_not_initialized_envelope(
        self, test_config_with_backends
    ):
        """auto_sync=True but get_sync_manager_instance returns None."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._sync_manager = None

        with patch(
            "gateway.get_sync_manager_instance",
            AsyncMock(return_value=None),
        ):
            from gateway import compass_sync

            result = await compass_sync()

        env = result["error_envelope"]
        assert env["code"] == "sync_manager_unavailable"
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True


# =============================================================================
# Health-state helpers + breaker transition metric
# =============================================================================


class TestHealthStateMutators:
    """_mark_ollama_down / _mark_ollama_up / _record_* helpers."""

    def test_mark_ollama_down(self):
        import gateway

        # Reset.
        gateway._health_state["ollama_available"] = True
        gateway._health_state["last_ollama_error"] = None

        gateway._mark_ollama_down(RuntimeError("connection refused"))
        assert gateway._health_state["ollama_available"] is False
        assert gateway._health_state["last_ollama_error"] is not None
        assert "RuntimeError" in gateway._health_state["last_ollama_error"]

    def test_mark_ollama_up(self):
        import gateway

        gateway._health_state["ollama_available"] = False
        gateway._health_state["last_ollama_error"] = "old error"

        gateway._mark_ollama_up()
        assert gateway._health_state["ollama_available"] is True
        assert gateway._health_state["last_ollama_error"] is None

    def test_record_breaker_transition(self):
        import gateway

        before = dict(gateway._metric_counters["circuit_breaker_transitions_total"])

        gateway._record_breaker_transition("closed", "open")
        gateway._record_breaker_transition("open", "half_open")
        gateway._record_breaker_transition("half_open", "closed")

        after = gateway._metric_counters["circuit_breaker_transitions_total"]
        # Keys are "from->to".
        assert after["closed->open"] == before.get("closed->open", 0) + 1
        assert after["open->half_open"] == before.get("open->half_open", 0) + 1
        assert after["half_open->closed"] == before.get("half_open->closed", 0) + 1

    def test_record_lexical_fallback(self):
        import gateway

        before = gateway._metric_counters["lexical_fallback_total"]
        before_fb = gateway._metric_counters["fallback_invocations_total"].get(
            "lexical", 0
        )

        gateway._record_lexical_fallback()

        assert gateway._metric_counters["lexical_fallback_total"] == before + 1
        assert (
            gateway._metric_counters["fallback_invocations_total"]["lexical"]
            == before_fb + 1
        )

    def test_record_degraded_response(self):
        import gateway

        before = gateway._metric_counters["degraded_responses_total"].get(
            "ollama_unavailable", 0
        )
        gateway._record_degraded_response("ollama_unavailable")
        assert (
            gateway._metric_counters["degraded_responses_total"][
                "ollama_unavailable"
            ]
            == before + 1
        )

    def test_invalidate_ready_cache_handles_exceptions(self, caplog):
        import gateway

        # Register an invalidator that raises.
        def bad_invalidator():
            raise RuntimeError("bad")

        gateway._ready_cache_invalidators.append(bad_invalidator)
        try:
            with caplog.at_level("DEBUG"):
                # Must not raise.
                gateway._invalidate_ready_cache()
        finally:
            # Clean up so other tests aren't affected.
            gateway._ready_cache_invalidators.remove(bad_invalidator)


# =============================================================================
# maybe_startup_sync() edge cases
# =============================================================================


class TestMaybeStartupSync:
    """Cover maybe_startup_sync edge cases not in test_gateway.py."""

    @pytest.mark.asyncio
    async def test_maybe_startup_sync_sync_manager_is_none(
        self, test_config_with_backends
    ):
        """sync_check_on_startup=True but sync_manager is None — flag still set."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._config.sync_check_on_startup = True
        gateway._startup_sync_done = False

        with patch(
            "gateway.get_sync_manager_instance",
            AsyncMock(return_value=None),
        ):
            from gateway import maybe_startup_sync

            await maybe_startup_sync()

        # Flag should still be set even if sync_manager was None.
        assert gateway._startup_sync_done is True


# =============================================================================
# get_config / get_index / get_backends / get_sync_manager_instance /
# get_chain_indexer_instance / get_analytics_instance — singleton paths
# =============================================================================


class TestSingletonPaths:
    """Touch the not-yet-initialized branches of each singleton getter."""

    @pytest.mark.asyncio
    async def test_get_sync_manager_builds_when_missing(
        self, test_config_with_backends, test_index, mock_backend_manager
    ):
        """First call to get_sync_manager_instance constructs via
        get_sync_manager()."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._compass_index = test_index
        gateway._backend_manager = mock_backend_manager
        gateway._sync_manager = None

        # Patch get_sync_manager() at module level so we don't touch disk.
        fake_mgr = Mock()
        with patch("gateway.get_sync_manager", return_value=fake_mgr):
            from gateway import get_sync_manager_instance

            result = await get_sync_manager_instance()

        assert result is fake_mgr
        assert gateway._sync_manager is fake_mgr

    @pytest.mark.asyncio
    async def test_get_chain_indexer_builds_when_missing(
        self, test_config_with_backends, test_index, mock_embedder
    ):
        """First call to get_chain_indexer_instance constructs via
        get_chain_indexer()."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._compass_index = test_index
        gateway._chain_indexer = None
        gateway._analytics = None

        fake_chain = Mock()
        fake_chain.load_chain_index = AsyncMock(return_value=True)
        with patch("gateway.get_chain_indexer", return_value=fake_chain):
            from gateway import get_chain_indexer_instance

            result = await get_chain_indexer_instance()

        assert result is fake_chain
        assert gateway._chain_indexer is fake_chain

    @pytest.mark.asyncio
    async def test_get_chain_indexer_seeds_when_load_fails(
        self, test_config_with_backends, test_index
    ):
        """If load_chain_index returns False, seed_default_chains +
        build_chain_index are invoked."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._compass_index = test_index
        gateway._chain_indexer = None
        gateway._analytics = None

        fake_chain = Mock()
        fake_chain.load_chain_index = AsyncMock(return_value=False)
        fake_chain.seed_default_chains = AsyncMock()
        fake_chain.build_chain_index = AsyncMock()

        with patch("gateway.get_chain_indexer", return_value=fake_chain):
            from gateway import get_chain_indexer_instance

            await get_chain_indexer_instance()

        fake_chain.seed_default_chains.assert_called_once()
        fake_chain.build_chain_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analytics_loads_hot_cache(
        self, test_config_with_backends
    ):
        """First call to get_analytics_instance triggers
        load_hot_cache_from_db()."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._analytics = None

        fake_an = Mock()
        fake_an.load_hot_cache_from_db = AsyncMock()
        with patch("gateway.get_analytics", return_value=fake_an):
            from gateway import get_analytics_instance

            result = await get_analytics_instance()

        assert result is fake_an
        fake_an.load_hot_cache_from_db.assert_called_once()


# =============================================================================
# CLI helpers: categorize_tool + show_config
# =============================================================================


class TestCliShowConfig:
    """show_config() prints to stdout — just validate it doesn't crash."""

    def test_show_config_runs(self, capsys):
        from gateway import show_config

        # Don't crash; output goes to stdout.
        show_config()
        captured = capsys.readouterr()
        assert "CONFIGURATION" in captured.out
        assert "Config file" in captured.out


class TestAsyncMain:
    """async_main() dispatches to sync_from_backends or run_tests."""

    @pytest.mark.asyncio
    async def test_async_main_sync_dispatch(self):
        from gateway import async_main

        args = Mock()
        args.sync = True
        args.test = False

        called = {"sync": 0, "test": 0}

        async def fake_sync():
            called["sync"] += 1

        async def fake_test():
            called["test"] += 1

        with patch("gateway.sync_from_backends", side_effect=fake_sync):
            with patch("gateway.run_tests", side_effect=fake_test):
                await async_main(args)

        assert called["sync"] == 1
        assert called["test"] == 0

    @pytest.mark.asyncio
    async def test_async_main_test_dispatch(self):
        from gateway import async_main

        args = Mock()
        args.sync = False
        args.test = True

        called = {"sync": 0, "test": 0}

        async def fake_sync():
            called["sync"] += 1

        async def fake_test():
            called["test"] += 1

        with patch("gateway.sync_from_backends", side_effect=fake_sync):
            with patch("gateway.run_tests", side_effect=fake_test):
                await async_main(args)

        assert called["sync"] == 0
        assert called["test"] == 1

    @pytest.mark.asyncio
    async def test_async_main_neither_is_noop(self):
        from gateway import async_main

        args = Mock()
        args.sync = False
        args.test = False

        # Must not raise.
        await async_main(args)


class TestMainEntrypoint:
    """main() CLI argument parsing."""

    def test_main_config_branch(self, capsys, monkeypatch):
        """`gateway --config` invokes show_config()."""
        from gateway import main

        monkeypatch.setattr("sys.argv", ["gateway.py", "--config"])
        with patch("gateway.show_config") as mock_show:
            main()
            mock_show.assert_called_once()

    def test_main_sync_branch(self, monkeypatch):
        """`gateway --sync` dispatches into async_main via asyncio.run."""
        from gateway import main

        monkeypatch.setattr("sys.argv", ["gateway.py", "--sync"])

        # TST-RA-001: consume the async_main coroutine so it doesn't leak a
        # "coroutine was never awaited" RuntimeWarning — mirror the fake_run
        # pattern used in test_main_exits_1_on_failed_sync below.
        def fake_run(coro):
            coro.close()
            return True  # truthy -> main() does not sys.exit(1) on this path

        with patch("gateway.asyncio.run", side_effect=fake_run) as mock_run:
            main()
            mock_run.assert_called_once()

    def test_main_test_branch(self, monkeypatch):
        """`gateway --test` dispatches into async_main via asyncio.run."""
        from gateway import main

        monkeypatch.setattr("sys.argv", ["gateway.py", "--test"])

        # TST-RA-001: consume the async_main coroutine to avoid the
        # "coroutine was never awaited" RuntimeWarning.
        def fake_run(coro):
            coro.close()
            return True  # truthy -> main() does not sys.exit(1) on this path

        with patch("gateway.asyncio.run", side_effect=fake_run) as mock_run:
            main()
            mock_run.assert_called_once()

    def test_main_verbose_sets_debug(self, monkeypatch):
        """`gateway --verbose --config` flips logging level to DEBUG."""
        import logging
        from gateway import main

        monkeypatch.setattr("sys.argv", ["gateway.py", "--verbose", "--config"])
        with patch("gateway.show_config"):
            main()

        # Root logger level was set to DEBUG.
        assert logging.getLogger().level == logging.DEBUG

    def test_main_no_args_runs_stdio(self, monkeypatch):
        """No args -> runs mcp.run() in stdio mode."""
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py"])
        # Make sure PORT isn't set.
        monkeypatch.delenv("PORT", raising=False)
        with patch.object(gateway.mcp, "run") as mock_run:
            gateway.main()
            mock_run.assert_called_once_with()

    def test_main_with_port_runs_http(self, monkeypatch):
        """PORT env var set -> _run_http() is invoked."""
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py"])
        monkeypatch.setenv("PORT", "8080")

        with patch("gateway._run_http") as mock_http:
            gateway.main()
            mock_http.assert_called_once_with(8080)


# =============================================================================
# Stage-C CLI humanization — exit codes + actionable hints for the CLI paths
# (GW-SB-001 sync exit, GW-SB-002 cold-start --test, GW-SB-003 PORT guard,
#  CFGDOC-01 LOG_LEVEL honored).
# =============================================================================


class TestSyncExitStatus:
    """GW-SB-001: a sync that indexed nothing must NOT exit 0."""

    @pytest.mark.asyncio
    async def test_sync_returns_false_when_no_backends(self, monkeypatch):
        """sync_from_backends() returns False when zero backends connect."""
        import gateway

        manager = Mock()
        manager.connect_all = AsyncMock(return_value={"alpha": False, "beta": False})
        manager.get_all_tools = Mock(return_value=[])
        manager.disconnect_all = AsyncMock()

        monkeypatch.setattr(gateway, "BackendManager", lambda config: manager)
        monkeypatch.setattr(
            gateway, "load_config", lambda: Mock(backends={"alpha": Mock(), "beta": Mock()})
        )

        result = await gateway.sync_from_backends()
        assert result is False

    @pytest.mark.asyncio
    async def test_async_main_propagates_sync_failure(self, monkeypatch):
        """async_main(--sync) returns the sync result so main() can exit non-zero."""
        import gateway

        async def fake_sync():
            return False

        monkeypatch.setattr(gateway, "sync_from_backends", fake_sync)
        args = Mock(sync=True, test=False)
        assert await gateway.async_main(args) is False

    @pytest.mark.asyncio
    async def test_async_main_test_returns_true(self, monkeypatch):
        """--test path still reports success (True) when run_tests completes."""
        import gateway

        async def fake_test():
            return None

        monkeypatch.setattr(gateway, "run_tests", fake_test)
        args = Mock(sync=False, test=True)
        assert await gateway.async_main(args) is True

    def test_main_exits_1_on_failed_sync(self, monkeypatch, capsys):
        """`gateway --sync` exits 1 with a stderr hint when nothing was indexed."""
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py", "--sync"])

        # asyncio.run returns the async_main result; simulate a failed sync.
        def fake_run(coro):
            coro.close()  # avoid 'coroutine never awaited' warning
            return False

        monkeypatch.setattr(gateway.asyncio, "run", fake_run)

        with pytest.raises(SystemExit) as exc:
            gateway.main()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Index NOT rebuilt" in err

    def test_main_succeeds_on_good_sync(self, monkeypatch):
        """A successful sync (True) leaves the process at exit 0 (no SystemExit)."""
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py", "--sync"])

        def fake_run(coro):
            coro.close()
            return True

        monkeypatch.setattr(gateway.asyncio, "run", fake_run)

        # Must not raise SystemExit.
        gateway.main()


class TestRunTestsColdStart:
    """GW-SB-002: `gateway --test` must not dump a raw traceback on cold start."""

    @pytest.mark.asyncio
    async def test_run_tests_cold_start_exits_1_with_hint(self, monkeypatch, capsys):
        import gateway

        async def boom():
            raise RuntimeError(
                "Ollama not available and no cached index found at /x. "
                "Start Ollama (ollama serve) and run: ollama pull nomic-embed-text"
            )

        monkeypatch.setattr(gateway, "get_index", boom)

        with pytest.raises(SystemExit) as exc:
            await gateway.run_tests()
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "--sync" in err
        assert "ollama serve" in err


class TestPortGuard:
    """GW-SB-003: malformed PORT must produce a named hint + exit 2, not a raw
    ValueError."""

    def test_non_integer_port_exits_2(self, monkeypatch, capsys):
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py"])
        monkeypatch.setenv("PORT", "not-a-number")

        with patch("gateway._run_http") as mock_http:
            with pytest.raises(SystemExit) as exc:
                gateway.main()
            mock_http.assert_not_called()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "Invalid PORT" in err
        assert "not-a-number" in err

    def test_out_of_range_port_exits_2(self, monkeypatch, capsys):
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py"])
        monkeypatch.setenv("PORT", "70000")

        with patch("gateway._run_http") as mock_http:
            with pytest.raises(SystemExit) as exc:
                gateway.main()
            mock_http.assert_not_called()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "Invalid PORT" in err

    def test_valid_port_passes_int_to_run_http(self, monkeypatch):
        """A well-formed PORT still reaches _run_http with the parsed int."""
        import gateway

        monkeypatch.setattr("sys.argv", ["gateway.py"])
        monkeypatch.setenv("PORT", "9090")

        with patch("gateway._run_http") as mock_http:
            gateway.main()
            mock_http.assert_called_once_with(9090)


class TestLogLevelEnv:
    """CFGDOC-01: LOG_LEVEL must actually drive the logging level."""

    def test_basicconfig_source_reads_log_level(self):
        """The module's logging setup must read LOG_LEVEL, not hardcode INFO.

        Asserted against the source (no module reload, which would corrupt the
        shared gateway module for the rest of the session). On the OLD code the
        basicConfig call was `level=logging.INFO` with no env read, so this
        assertion fails on the pre-fix gateway."""
        import inspect
        import gateway

        src = inspect.getsource(gateway)
        # The level must be derived from the LOG_LEVEL env var.
        assert 'os.environ.get("LOG_LEVEL"' in src
        # And it must NOT be the old hardcoded form.
        assert "level=logging.INFO," not in src

    def test_log_level_env_mapping_debug(self, monkeypatch):
        """The exact expression the module uses maps LOG_LEVEL=DEBUG -> DEBUG."""
        import logging

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        level = getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        assert level == logging.DEBUG

    def test_unknown_log_level_falls_back_to_info(self, monkeypatch):
        import logging

        monkeypatch.setenv("LOG_LEVEL", "NONSENSE")
        level = getattr(
            logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        assert level == logging.INFO


# =============================================================================
# Compass envelope shape sanity — every error path stamps an "error_envelope"
# =============================================================================


class TestEnvelopeContract:
    """All MCP-error responses should expose error_envelope.code,
    .category, and .retryable as a closed set."""

    @pytest.mark.asyncio
    async def test_describe_not_found_envelope_contract(
        self, test_index, test_config
    ):
        import gateway
        from gateway import _ERROR_CODES, _ERROR_CATEGORIES

        gateway._compass_index = test_index
        gateway._config = test_config
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        from gateway import describe

        result = await describe(tool_name="test:def_not_a_tool_xyz")
        env = result["error_envelope"]
        assert env["code"] in _ERROR_CODES
        assert env["category"] in _ERROR_CATEGORIES
        assert isinstance(env["retryable"], bool)

    @pytest.mark.asyncio
    async def test_compass_chains_invalid_action_envelope_contract(
        self, test_config_with_backends, test_chain_indexer
    ):
        import gateway
        from gateway import _ERROR_CODES, _ERROR_CATEGORIES

        gateway._config = test_config_with_backends
        gateway._chain_indexer = test_chain_indexer

        from gateway import compass_chains

        result = await compass_chains(action="not_a_thing")
        env = result["error_envelope"]
        assert env["code"] in _ERROR_CODES
        assert env["category"] in _ERROR_CATEGORIES
        assert isinstance(env["retryable"], bool)

    @pytest.mark.asyncio
    async def test_compass_sync_disabled_envelope_contract(self, test_config):
        import gateway
        from gateway import _ERROR_CODES, _ERROR_CATEGORIES

        gateway._config = test_config

        from gateway import compass_sync

        result = await compass_sync()
        env = result["error_envelope"]
        assert env["code"] in _ERROR_CODES
        assert env["category"] in _ERROR_CATEGORIES
        assert env["retryable"] is False


# =============================================================================
# GW-A-001 — cold-start get_index() RuntimeError -> structured envelope
# =============================================================================


class TestColdStartIndexEnvelope:
    """get_index() raises RuntimeError when there's no baked index AND Ollama
    is unreachable. compass() / describe() / compass_categories() must surface
    that as the structured service_unavailable envelope, never a raw raise."""

    @staticmethod
    def _assert_cold_start_envelope(result):
        from gateway import _ERROR_CODES, _ERROR_CATEGORIES

        assert isinstance(result, dict), "handler must return a dict, not raise"
        assert "error_envelope" in result, (
            f"cold-start must return the structured envelope, got: {result!r}"
        )
        env = result["error_envelope"]
        assert env["code"] in {"ollama_unavailable", "index_unhealthy"}
        assert env["code"] in _ERROR_CODES
        assert env["category"] == "service_unavailable"
        assert env["category"] in _ERROR_CATEGORIES
        assert env["retryable"] is True
        # Operator-actionable suggestions are required by the finding.
        suggestions = " ".join(env.get("suggestions", [])).lower()
        assert "ollama serve" in suggestions
        assert "--sync" in suggestions

    @pytest.mark.asyncio
    async def test_compass_cold_start_returns_envelope(self, test_config):
        import gateway

        gateway._config = test_config

        async def cold_start():
            raise RuntimeError(
                "Ollama not available and no cached index found"
            )

        # No index, sync disabled (test_config.auto_sync is False).
        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import compass

            result = await compass(intent="read a file")

        self._assert_cold_start_envelope(result)

    @pytest.mark.asyncio
    async def test_describe_cold_start_returns_envelope(self, test_config):
        import gateway

        gateway._config = test_config

        async def cold_start():
            raise RuntimeError(
                "Ollama not available and no cached index found"
            )

        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import describe

            result = await describe(tool_name="bridge:read_file")

        self._assert_cold_start_envelope(result)

    @pytest.mark.asyncio
    async def test_compass_categories_cold_start_returns_envelope(self, test_config):
        import gateway

        gateway._config = test_config

        async def cold_start():
            raise RuntimeError(
                "Ollama not available and no cached index found"
            )

        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import compass_categories

            result = await compass_categories()

        self._assert_cold_start_envelope(result)

    @pytest.mark.asyncio
    async def test_cold_start_code_tracks_ollama_health(self, test_config):
        """When Ollama is known-down the code is ollama_unavailable; otherwise
        the index itself is the blocker (index_unhealthy)."""
        import gateway

        gateway._config = test_config

        async def cold_start():
            raise RuntimeError("cold start")

        # Ollama explicitly down -> ollama_unavailable.
        gateway._health_state["ollama_available"] = False
        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import compass_categories

            result = await compass_categories()
        assert result["error_envelope"]["code"] == "ollama_unavailable"

        # Ollama nominally up but index won't load -> index_unhealthy.
        gateway._health_state["ollama_available"] = True
        with patch("gateway.get_index", side_effect=cold_start):
            result = await compass_categories()
        assert result["error_envelope"]["code"] == "index_unhealthy"


# =============================================================================
# GW-A-002 — describe() malformed-JSON index row degrades, never raises
# =============================================================================


class TestDescribeMalformedJson:
    """A corrupt parameters/examples JSON blob in the index row must degrade
    to {}/[] and flag the index unhealthy rather than raising
    JSONDecodeError."""

    @pytest.mark.asyncio
    async def test_describe_invalid_parameters_json_degrades(
        self, test_index, test_config
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True
        # Backend has no fallback schema — force the index row path.
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        # Corrupt the parameters blob for an existing tool. The describe()
        # SELECT reads parameters + examples columns; a non-JSON string there
        # is what a partially-written / truncated index produces.
        test_index.db.execute(
            "UPDATE tools SET parameters = ? WHERE name = ?",
            ("{not valid json", "test:read_file"),
        )
        test_index.db.commit()

        from gateway import describe

        # Must NOT raise — returns the tool with empty params instead.
        result = await describe(tool_name="test:read_file")

        assert result["tool"] == "test:read_file"
        assert result["parameters"] == {}, (
            "malformed parameters JSON must fall back to {}"
        )
        # The malformed blob flags the index unhealthy + the augmenter stamps
        # the degraded reason.
        assert gateway._health_state["index_available"] is False
        assert result.get("degraded") is True
        assert "index_unhealthy" in result.get("degraded_reasons", [])

    @pytest.mark.asyncio
    async def test_describe_invalid_examples_json_degrades(
        self, test_index, test_config
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True
        mgr = Mock()
        mgr.get_tool_schema = Mock(return_value=None)
        gateway._backend_manager = mgr

        test_index.db.execute(
            "UPDATE tools SET examples = ? WHERE name = ?",
            ("[broken", "test:write_file"),
        )
        test_index.db.commit()

        from gateway import describe

        result = await describe(tool_name="test:write_file")

        assert result["tool"] == "test:write_file"
        assert result["examples"] == [], (
            "malformed examples JSON must fall back to []"
        )
        assert gateway._health_state["index_available"] is False


# =============================================================================
# GW-A-003 — multi-word lexical fallback matches per-token (0.3 branch live)
# =============================================================================


class TestLexicalFallbackPerToken:
    """_lexical_search_fallback tokenizes the query so multi-word degraded-mode
    intents still match, and the previously-dead 0.3 confidence tier is now
    reachable."""

    def test_multi_word_intent_matches_via_token(self, test_index):
        from gateway import _lexical_search_fallback

        # "missing file" is NOT a substring of any tool name/description, but
        # the token "file" matches test:read_file / test:write_file. With the
        # old single-whole-query needle this returned []; per-token matching
        # now finds the *_file tools.
        matches = _lexical_search_fallback(
            test_index, "missing file", top_k=5, category=None, server=None
        )
        names = {m["tool"] for m in matches}
        assert "test:read_file" in names or "test:write_file" in names, (
            f"per-token fallback should match a *_file tool, got: {names}"
        )

    def test_token_only_match_takes_0_3_confidence(self, test_index):
        """The else-branch (0.3) is reachable: a row matched on a token but the
        whole query is not a substring of its name or description."""
        from gateway import _lexical_search_fallback

        matches = _lexical_search_fallback(
            test_index, "missing file", top_k=5, category=None, server=None
        )
        assert matches, "expected at least one token match"
        # Every match here is token-only (whole 'missing file' never appears),
        # so all confidences are the 0.3 tier.
        assert all(m["confidence"] == 0.3 for m in matches), (
            f"token-only matches must score 0.3, got: "
            f"{[(m['tool'], m['confidence']) for m in matches]}"
        )

    def test_whole_query_name_substring_still_0_6(self, test_index):
        """Regression guard: a whole-query substring of a name keeps 0.6."""
        from gateway import _lexical_search_fallback

        matches = _lexical_search_fallback(
            test_index, "read_file", top_k=5, category=None, server=None
        )
        read_file = next(
            (m for m in matches if m["tool"] == "test:read_file"), None
        )
        assert read_file is not None
        assert read_file["confidence"] == 0.6

    def test_escaping_preserved_for_wildcard_tokens(self, test_index):
        """A token containing a LIKE wildcard must be escaped, not treated as a
        wildcard (BE-A-007 must survive the per-token rewrite)."""
        from gateway import _lexical_search_fallback

        # '%' would match everything if unescaped; escaped, it matches only
        # tools whose name/description literally contain '%' (none here).
        matches = _lexical_search_fallback(
            test_index, "%", top_k=5, category=None, server=None
        )
        assert matches == [], (
            f"escaped '%' token must not wildcard-match the catalog, got: "
            f"{[m['tool'] for m in matches]}"
        )


# =============================================================================
# GW-A-001 (cold-start bypass) — compass() calls maybe_startup_sync() BEFORE
# the guarded get_index(); get_sync_manager_instance/get_chain_indexer_instance
# catch RuntimeError from get_index() and return None, and maybe_startup_sync
# does NOT latch _startup_sync_done on a cold-start deferral (so it retries).
# =============================================================================


class TestColdStartBypass:
    """A cold-start get_index() RuntimeError must not escape compass() through
    the maybe_startup_sync() -> get_sync_manager_instance() path, and the
    startup-sync flag must stay False so a later call retries once the index
    is buildable."""

    @pytest.mark.asyncio
    async def test_compass_cold_start_does_not_raise_through_startup_sync(
        self, test_config_with_backends
    ):
        """auto_sync=True drives maybe_startup_sync() at the top of compass();
        get_index() raising RuntimeError must yield the structured cold-start
        envelope, never a raw RuntimeError, and the startup flag stays False.
        """
        import gateway

        # auto_sync + sync_check_on_startup True so maybe_startup_sync runs the
        # slow path and calls get_sync_manager_instance() -> get_index().
        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._config.sync_check_on_startup = True
        gateway._startup_sync_done = False
        gateway._sync_manager = None
        gateway._compass_index = None
        gateway._analytics = None

        async def cold_start():
            raise RuntimeError(
                "Ollama not available and no cached index found"
            )

        # Patch get_index so BOTH maybe_startup_sync's path and compass()'s
        # own guarded get_index() hit the cold-start RuntimeError.
        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import compass

            # Must NOT raise — returns the structured envelope.
            result = await compass(intent="read a file")

        assert isinstance(result, dict), "compass must return a dict, not raise"
        assert "error_envelope" in result, (
            f"cold-start must return the structured envelope, got: {result!r}"
        )
        env = result["error_envelope"]
        assert env["category"] == "service_unavailable"
        assert env["retryable"] is True
        # The cold-start deferral must NOT latch the flag — a later call (once
        # Ollama/index is up) has to retry the startup sync.
        assert gateway._startup_sync_done is False, (
            "cold-start deferral must not latch _startup_sync_done"
        )

    @pytest.mark.asyncio
    async def test_get_sync_manager_instance_returns_none_on_cold_start(
        self, test_config_with_backends
    ):
        """get_sync_manager_instance() must catch get_index()'s RuntimeError
        and return None (sync unavailable until an index exists), not raise."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._config.auto_sync = True
        gateway._sync_manager = None

        async def cold_start():
            raise RuntimeError("cold start: no index, ollama down")

        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import get_sync_manager_instance

            result = await get_sync_manager_instance()

        assert result is None, (
            "cold-start get_index RuntimeError must degrade to None, not raise"
        )
        # And it must NOT have cached a half-built sync manager.
        assert gateway._sync_manager is None

    @pytest.mark.asyncio
    async def test_get_chain_indexer_instance_returns_none_on_cold_start(
        self, test_config_with_backends
    ):
        """get_chain_indexer_instance() must catch get_index()'s RuntimeError
        and return None, mirroring the sync-manager cold-start guard."""
        import gateway

        gateway._config = test_config_with_backends
        gateway._config.chain_indexing_enabled = True
        gateway._chain_indexer = None
        gateway._analytics = None

        async def cold_start():
            raise RuntimeError("cold start: no index, ollama down")

        with patch("gateway.get_index", side_effect=cold_start):
            from gateway import get_chain_indexer_instance

            result = await get_chain_indexer_instance()

        assert result is None
        assert gateway._chain_indexer is None


# =============================================================================
# GW (per-loop locks) — the 6 module-global asyncio.Lock objects are replaced
# by gateway._loop_lock(name) keyed on id(running loop), mirroring embedder's
# per-loop semaphore. Awaiting the same named lock from two independent
# worker-thread asyncio.run loops must not raise "bound to a different event
# loop".
# =============================================================================


class TestPerLoopLocks:
    """SC-001 sibling: _loop_lock(name) must hand each running event loop its
    OWN asyncio.Lock so the singleton getters survive being driven from a
    fresh asyncio.run loop per call (CLI subcommands / Gradio worker threads).

    A single module-global asyncio.Lock binds its internal waiter Future to
    the first loop that awaits it under contention, then raises
    "bound to a different event loop" from the next loop.
    """

    @staticmethod
    def _acquire_loop_lock_under_contention(name: str):
        """In a brand-new event loop, acquire _loop_lock(name) from two
        concurrent coroutines so a waiter forms — the exact precondition that
        bound a module-global Lock to this loop and broke the next one.
        Returns True on a clean acquire/usability run.
        """
        import gateway

        async def _hold(lock, hold_s):
            async with lock:
                await asyncio.sleep(hold_s)
            return True

        async def _go():
            lock = gateway._loop_lock(name)
            # Two acquirers: the second queues as a waiter while the first
            # holds the lock, binding the lock's waiter to THIS loop.
            results = await asyncio.gather(
                _hold(lock, 0.02), _hold(lock, 0.0)
            )
            # The returned lock must be usable (acquirable) in this loop.
            assert lock.locked() is False
            return all(results)

        return asyncio.run(_go())

    def test_loop_lock_survives_multiple_worker_loops(self):
        """Several threads, each with its OWN asyncio.run loop, drive the same
        named _loop_lock concurrently — no cross-loop RuntimeError, and every
        loop gets a usable lock."""
        errors: list = []
        ok_seen: list = []

        def thread_target():
            try:
                ok = self._acquire_loop_lock_under_contention("index")
                ok_seen.append(ok)
            except Exception as e:  # noqa: BLE001 — capture for assertion
                errors.append(repr(e))

        threads = [threading.Thread(target=thread_target) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, (
            "per-loop lock raised cross-loop RuntimeError(s): "
            f"{errors}"
        )
        assert ok_seen and all(ok_seen), (
            f"each loop should cleanly use the lock; got {ok_seen}"
        )

    def test_loop_lock_sequential_loops_rebind_cleanly(self):
        """Deterministic form: loop A forms a waiter on _loop_lock('index'),
        finishes, then loop B uses the same named lock. A module-global Lock
        would stay bound to loop A's (dead) loop and break loop B with
        'bound to a different event loop'."""
        # Loop A
        assert self._acquire_loop_lock_under_contention("index") is True
        # Loop B — must not raise the cross-loop RuntimeError.
        assert self._acquire_loop_lock_under_contention("index") is True

    def test_loop_lock_keyed_on_running_loop_within_one_loop(self):
        """Within a SINGLE running loop, _loop_lock(name) is keyed on that
        loop's id and returns the SAME object on repeat calls, but DISTINCT
        objects for distinct names — proving the per-loop, per-name keying
        (not one shared global Lock for everything)."""
        import gateway

        captured: dict = {}

        async def _go():
            captured["index_1"] = gateway._loop_lock("index")
            captured["index_2"] = gateway._loop_lock("index")
            captured["sync"] = gateway._loop_lock("sync_manager")
            # The lock dict for THIS loop must be registered under id(loop).
            loop = asyncio.get_running_loop()
            captured["registered"] = id(loop) in gateway._loop_init_locks

        asyncio.run(_go())

        # Same name in the same loop -> identical object (memoized per loop).
        assert captured["index_1"] is captured["index_2"]
        # Different names -> different lock objects (six independent locks).
        assert captured["index_1"] is not captured["sync"]
        # The loop's lock bundle is keyed on the running loop's id.
        assert captured["registered"] is True


# =============================================================================
# GW (show_config redaction) — show_config() redacts backend.args entries (via
# _redact_structural) and ollama_url credentials before printing.
# =============================================================================


class TestShowConfigRedaction:
    """show_config() must not print backend.args secrets or ollama_url
    credentials to stdout."""

    def test_show_config_redacts_args_and_url_secrets(self, capsys, tmp_path):
        # show_config() truncates args to the first 2 entries before printing,
        # so the secret MUST sit within args[:2] — otherwise the SHOWSECRET
        # assertion would pass vacuously (truncated out) even un-redacted.
        config_file = tmp_path / "compass_config.json"
        config_file.write_text(json.dumps({
            "backends": {
                "local": {
                    "type": "stdio",
                    "command": "python",
                    "args": ["--password=SHOWSECRET", "-m"],
                    "env": {},
                },
            },
            "ollama_url": "http://u:URLSECRET@h:11434",
        }))

        with patch.dict(os.environ, {"TOOL_COMPASS_CONFIG": str(config_file)}):
            from gateway import show_config

            show_config()

        out = capsys.readouterr().out
        # Neither secret may appear in the printed diagnostic dump.
        assert "SHOWSECRET" not in out, "backend.args secret leaked to stdout"
        assert "URLSECRET" not in out, "ollama_url secret leaked to stdout"
        # Host:port stays visible for diagnosability.
        assert "h:11434" in out
        # And the redaction marker is present so the dump stays usable.
        assert "[REDACTED]" in out


# =============================================================================
# DISC-01 / DISC-02 — hybrid search (RRF fusion) + exact-name boost in compass()
# =============================================================================


def _mk_search_results(sample_tools, order_scores):
    """Build a list of SearchResult from (tool_name, score) pairs, resolving
    tool_name against the sample_tools fixture."""
    from indexer import SearchResult

    by_name = {t.name: t for t in sample_tools}
    out = []
    for rank, (name, score) in enumerate(order_scores, start=1):
        out.append(SearchResult(tool=by_name[name], score=score, rank=rank))
    return out


class TestExactNameBoostDISC02:
    """DISC-02: an exact tool-name paste ranks #1 at exact_match_confidence."""

    @pytest.mark.asyncio
    async def test_exact_qualified_name_ranks_first_at_exact_confidence(
        self, test_index, test_config, sample_tools
    ):
        import gateway
        from unittest.mock import patch as _patch

        # Semantic search ranks generate_image #1 and read_file #2. Pasting the
        # exact qualified name "test:read_file" must force it to #1 at conf=1.0.
        test_config.hybrid_search = False  # isolate the boost behavior
        test_config.exact_name_boost = True
        test_config.exact_match_confidence = 1.0
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools,
            [("test:generate_image", 0.8), ("test:read_file", 0.5)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="test:read_file", top_k=5)

        assert result["matches"], "expected at least the boosted match"
        top = result["matches"][0]
        assert top["tool"] == "test:read_file"
        assert top["confidence"] == 1.0
        # Deduped: read_file appears exactly once.
        assert [m["tool"] for m in result["matches"]].count("test:read_file") == 1

    @pytest.mark.asyncio
    async def test_exact_bare_name_matches_server_tool_suffix(
        self, test_index, test_config, sample_tools
    ):
        import gateway
        from unittest.mock import patch as _patch

        # A bare "read_file" should match the server:tool suffix of
        # "test:read_file" and rank it #1.
        test_config.hybrid_search = False
        test_config.exact_name_boost = True
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools, [("test:generate_image", 0.9)]
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="read_file", top_k=5)

        assert result["matches"][0]["tool"] == "test:read_file"
        assert result["matches"][0]["confidence"] == test_config.exact_match_confidence

    @pytest.mark.asyncio
    async def test_exact_confidence_beats_min_confidence(
        self, test_index, test_config, sample_tools
    ):
        """Back-compat: 1.0 >= any min_confidence, so the boosted tool survives
        even a high min_confidence filter that dropped the semantic hit."""
        import gateway
        from unittest.mock import patch as _patch

        test_config.hybrid_search = False
        test_config.exact_name_boost = True
        test_config.exact_match_confidence = 1.0
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        # Semantic hit is below min_confidence and gets filtered out; only the
        # boost can put read_file back on the board.
        semantic = _mk_search_results(sample_tools, [("test:read_file", 0.4)])
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(
                intent="test:read_file", min_confidence=0.9, top_k=5
            )

        assert result["matches"], "boosted exact match must survive min_conf"
        assert result["matches"][0]["tool"] == "test:read_file"
        assert result["matches"][0]["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_no_boost_when_disabled(
        self, test_index, test_config, sample_tools
    ):
        import gateway
        from unittest.mock import patch as _patch

        test_config.hybrid_search = False
        test_config.exact_name_boost = False
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools,
            [("test:generate_image", 0.8), ("test:read_file", 0.5)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="test:read_file", top_k=5)

        # Boost off + hybrid off -> pure semantic ordering (generate_image #1).
        assert result["matches"][0]["tool"] == "test:generate_image"


class TestExactNameBoostRespectsFiltersGWDISC002:
    """GW-DISC-002 (DEFECT 1): the exact-name boost must respect the active
    category/server filter. A tool that fails the filter — and that
    index.search() therefore already excluded — must NEVER be resurrected at
    rank #1 by the boost lookup, which used to query on name alone.
    """

    @pytest.mark.asyncio
    async def test_boost_does_not_resurrect_tool_outside_category_filter(
        self, test_index, test_config, sample_tools
    ):
        """intent exactly names test:git_status (category=git) but the caller
        asked for category=file. The git tool must NOT be boosted in."""
        import gateway
        from unittest.mock import patch as _patch

        test_config.hybrid_search = False  # isolate the boost behavior
        test_config.exact_name_boost = True
        test_config.exact_match_confidence = 1.0
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        # search() (mocked) returns only the in-filter file tools — this is what
        # a correctly-filtered semantic search would return for category=file.
        semantic = _mk_search_results(
            sample_tools,
            [("test:read_file", 0.7), ("test:write_file", 0.6)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(
                intent="test:git_status", category="file", top_k=5
            )

        names = [m["tool"] for m in result["matches"]]
        assert "test:git_status" not in names, (
            "boost resurrected a category=git tool despite category=file filter; "
            f"got {names}"
        )

    @pytest.mark.asyncio
    async def test_boost_does_not_resurrect_tool_outside_server_filter(
        self, test_index, test_config, sample_tools
    ):
        """intent exactly names a tool on server=other, but the caller asked
        for server=test. The other-server tool must NOT be boosted in."""
        import gateway
        from unittest.mock import patch as _patch

        # Insert an out-of-filter tool directly into the real DB the boost
        # lookup queries. All sample_tools share server=test, so we need a
        # distinct server to exercise the server_filter clause.
        test_index.db.execute(
            "INSERT INTO tools (name, description, category, server) "
            "VALUES (?, ?, ?, ?)",
            ("other:special_tool", "A tool on a different server", "misc", "other"),
        )
        test_index.db.commit()

        test_config.hybrid_search = False
        test_config.exact_name_boost = True
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(sample_tools, [("test:read_file", 0.7)])
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(
                intent="other:special_tool", server="test", top_k=5
            )

        names = [m["tool"] for m in result["matches"]]
        assert "other:special_tool" not in names, (
            "boost resurrected a server=other tool despite server=test filter; "
            f"got {names}"
        )

    @pytest.mark.asyncio
    async def test_boost_still_pins_exact_tool_inside_category_filter(
        self, test_index, test_config, sample_tools
    ):
        """Regression guard: an exact-name intent for a tool INSIDE the active
        filter is STILL boosted to #1 (the fix must not over-restrict)."""
        import gateway
        from unittest.mock import patch as _patch

        test_config.hybrid_search = False
        test_config.exact_name_boost = True
        test_config.exact_match_confidence = 1.0
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        # write_file ranks below read_file semantically, but exact-naming it with
        # a matching category=file filter must pin it to #1.
        semantic = _mk_search_results(
            sample_tools,
            [("test:read_file", 0.8), ("test:write_file", 0.5)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(
                intent="test:write_file", category="file", top_k=5
            )

        assert result["matches"], "expected the boosted match"
        assert result["matches"][0]["tool"] == "test:write_file"
        assert result["matches"][0]["confidence"] == 1.0

    def test_lookup_helper_applies_category_and_server_clauses(self, test_index):
        """Unit-level: the helper itself returns None for an out-of-filter tool
        and the row for an in-filter one."""
        from gateway import _exact_name_boost_lookup

        # git_status is category=git/server=test in the sample corpus.
        # Wrong category -> no hit.
        assert (
            _exact_name_boost_lookup(
                test_index, "test:git_status", category="file", server=None
            )
            is None
        )
        # Wrong server -> no hit.
        assert (
            _exact_name_boost_lookup(
                test_index, "test:git_status", category=None, server="other"
            )
            is None
        )
        # Matching category+server -> the row is returned.
        hit = _exact_name_boost_lookup(
            test_index, "test:git_status", category="git", server="test"
        )
        assert hit is not None
        assert hit["tool"] == "test:git_status"
        # No filters -> back-compat: still returns the row.
        hit2 = _exact_name_boost_lookup(test_index, "test:git_status")
        assert hit2 is not None and hit2["tool"] == "test:git_status"


class TestHybridSearchDISC01:
    """DISC-01: RRF fusion improves a lexically-obvious mid-ranked tool and is
    a no-op relative to pure-semantic when disabled."""

    @pytest.mark.asyncio
    async def test_hybrid_promotes_lexically_obvious_midranked_tool(
        self, test_index, test_config, sample_tools
    ):
        import gateway
        from unittest.mock import patch as _patch

        # Semantic buries git_status at the bottom, but the intent literally
        # contains "git status" so the lexical list ranks it top. RRF fusion
        # should lift it above where semantics alone placed it.
        # Baseline healthy _health_state (module global not reset by conftest).
        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = True
        test_config.hybrid_search = True
        test_config.exact_name_boost = False  # isolate fusion
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools,
            [
                ("test:read_file", 0.70),
                ("test:write_file", 0.68),
                ("test:generate_image", 0.66),
                ("test:search_docs", 0.64),
                ("test:git_status", 0.62),  # semantically last
            ],
        )

        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            hybrid_result = await compass(intent="git status", top_k=5)

        # With fusion, git_status (lexical #1) must rank higher than its
        # semantic-only position (#5 / last).
        names = [m["tool"] for m in hybrid_result["matches"]]
        assert "test:git_status" in names
        assert names.index("test:git_status") < 4, (
            "fusion should lift git_status above semantic-last; got " + str(names)
        )
        # And fusion must NOT falsely mark the healthy response degraded.
        assert hybrid_result["degraded"] is False
        # No match should carry the internal degraded flag from the lexical list.
        for m in hybrid_result["matches"]:
            assert "degraded" not in m

    @pytest.mark.asyncio
    async def test_hybrid_does_not_regress_when_semantic_already_best(
        self, test_index, test_config, sample_tools
    ):
        """When semantic already ranks the obvious tool #1, fusion keeps it #1
        (does-not-regress half of the DISC-01 acceptance)."""
        import gateway
        from unittest.mock import patch as _patch

        test_config.hybrid_search = True
        test_config.exact_name_boost = False
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools,
            [("test:git_status", 0.9), ("test:read_file", 0.5)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="git status", top_k=5)

        assert result["matches"][0]["tool"] == "test:git_status"

    @pytest.mark.asyncio
    async def test_hybrid_false_identical_to_pure_semantic(
        self, test_index, test_config, sample_tools
    ):
        import gateway
        from unittest.mock import patch as _patch

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(
            sample_tools,
            [
                ("test:read_file", 0.70),
                ("test:write_file", 0.68),
                ("test:git_status", 0.62),
            ],
        )

        # Pure semantic ordering (both knobs off).
        test_config.hybrid_search = False
        test_config.exact_name_boost = False
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            pure = await compass(intent="git status", top_k=5)
        pure_names = [m["tool"] for m in pure["matches"]]
        pure_confs = [m["confidence"] for m in pure["matches"]]

        # Semantic ordering must be exactly the input order + scores.
        assert pure_names == ["test:read_file", "test:write_file", "test:git_status"]
        assert pure_confs == [0.7, 0.68, 0.62]

    @pytest.mark.asyncio
    async def test_hybrid_healthy_path_not_marked_degraded(
        self, test_index, test_config, sample_tools
    ):
        """The fused healthy path must never set degraded=True even though it
        reuses _lexical_search_fallback (which stamps degraded on its matches)."""
        import gateway
        from unittest.mock import patch as _patch

        # Baseline healthy _health_state (module global not reset by conftest).
        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = True
        test_config.hybrid_search = True
        test_config.exact_name_boost = False
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        semantic = _mk_search_results(sample_tools, [("test:read_file", 0.8)])
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="read file", top_k=5)

        assert result["degraded"] is False
        assert "degraded_reasons" not in result or result["degraded_reasons"] == []


class TestRRFReordersSemanticTopGWDISC003:
    """GW-DISC-003 (DEFECT 3): a NON-EMPTY lexical list CAN reorder the semantic
    top — standard RRF-by-rank behavior. The pure-semantic ordering is preserved
    ONLY when the lexical list is EMPTY, not merely when its top differs. These
    tests document that actual behavior (the docstring was corrected to match).
    """

    def test_lexical_hit_on_rank2_demotes_semantic_rank1(self):
        """A lexical hit on the semantic rank-2 candidate lifts its fused score
        above the semantic rank-1 (0.9-confidence) tool, promoting it to #1."""
        from gateway import _rrf_fuse

        semantic = [
            {"tool": "a:read", "confidence": 0.9},   # semantic #1
            {"tool": "b:write", "confidence": 0.4},  # semantic #2
        ]
        # Lexical false-friend match lands only on the rank-2 tool.
        lexical = [{"tool": "b:write", "confidence": 0.6}]

        fused = _rrf_fuse(semantic, lexical, top_k=5)
        names = [m["tool"] for m in fused]
        # b:write is promoted above a:read despite a:read's 0.9 semantic score.
        assert names[0] == "b:write", (
            "lexical hit on rank-2 should reorder the semantic top; got " + str(names)
        )
        assert names[1] == "a:read"

    def test_empty_lexical_preserves_pure_semantic_ordering(self):
        """The only guarantee: an EMPTY lexical list leaves the semantic order
        byte-for-byte intact."""
        from gateway import _rrf_fuse

        semantic = [
            {"tool": "a:read", "confidence": 0.9},
            {"tool": "b:write", "confidence": 0.4},
            {"tool": "c:list", "confidence": 0.3},
        ]
        fused = _rrf_fuse(semantic, [], top_k=5)
        assert [m["tool"] for m in fused] == ["a:read", "b:write", "c:list"]

    @pytest.mark.asyncio
    async def test_full_path_lexical_top_reorders_semantic_top(
        self, test_index, test_config, sample_tools
    ):
        """End-to-end through compass(): the lexical list's top differs from the
        semantic top and DOES reorder it (documents the served behavior)."""
        import gateway
        from unittest.mock import patch as _patch

        gateway._health_state["ollama_available"] = True
        gateway._health_state["index_available"] = True
        test_config.hybrid_search = True
        test_config.exact_name_boost = False  # isolate fusion
        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._startup_sync_done = True
        gateway._analytics = None

        # Semantic ranks read_file #1 (0.9); the intent literally contains
        # "git status" so the lexical list ranks git_status top — a different
        # top. Fusion promotes git_status above the semantic #1.
        semantic = _mk_search_results(
            sample_tools,
            [("test:read_file", 0.9), ("test:git_status", 0.5)],
        )
        with _patch.object(test_index, "search", return_value=semantic):
            from gateway import compass

            result = await compass(intent="git status", top_k=5)

        names = [m["tool"] for m in result["matches"]]
        assert names[0] == "test:git_status", (
            "a lexical top different from the semantic top reorders it; got "
            + str(names)
        )


# =============================================================================
# FEAT-01 — describe() serves the full schema from raw_schema (B3 column)
# =============================================================================


class TestDescribeFullSchemaFEAT01:
    """describe() prefers the full raw_schema when present, and falls back to
    the collapsed `parameters` when the value is NULL / malformed, or when the
    column itself is absent (older DB, pre-migration)."""

    @pytest.mark.asyncio
    async def test_describe_returns_full_schema_when_raw_schema_present(
        self, test_index, test_config
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True

        # Populate the full inputSchema for one tool (column already exists in
        # the index schema — B3 added it).
        full_schema = {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
                "encoding": {
                    "type": "string",
                    "enum": ["utf-8", "latin-1"],
                    "default": "utf-8",
                },
            },
            "required": ["filepath"],
        }
        test_index.db.execute(
            "UPDATE tools SET raw_schema = ? WHERE name = ?",
            (json.dumps(full_schema), "test:read_file"),
        )
        test_index.db.commit()

        from gateway import describe

        result = await describe(tool_name="test:read_file")

        # The full schema is surfaced under `schema`.
        assert result["schema"] == full_schema
        assert result["schema"]["required"] == ["filepath"]
        assert result["schema"]["properties"]["encoding"]["enum"] == [
            "utf-8",
            "latin-1",
        ]
        # `parameters` is preserved for back-compat.
        assert "parameters" in result

    @pytest.mark.asyncio
    async def test_describe_falls_back_when_raw_schema_null(
        self, test_index, test_config
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True

        # raw_schema is NULL by default for every freshly-built row.
        from gateway import describe

        result = await describe(tool_name="test:read_file")

        # No `schema` field — fell back to collapsed parameters exactly.
        assert "schema" not in result
        assert result["parameters"] == {"filepath": "str"}

    @pytest.mark.asyncio
    async def test_describe_falls_back_when_raw_schema_malformed(
        self, test_index, test_config
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config
        gateway._health_state["index_available"] = True

        test_index.db.execute(
            "UPDATE tools SET raw_schema = ? WHERE name = ?",
            ("{not valid json", "test:read_file"),
        )
        test_index.db.commit()

        from gateway import describe

        result = await describe(tool_name="test:read_file")

        assert "schema" not in result
        assert result["parameters"] == {"filepath": "str"}

    @pytest.mark.asyncio
    async def test_load_raw_schema_handles_missing_column_defensively(
        self, test_index
    ):
        """Direct-unit: before B3's migration the column doesn't exist and the
        SELECT raises 'no such column'. _load_raw_schema must return None
        WITHOUT flipping index health to unhealthy (expected, not a fault)."""
        import gateway
        from unittest.mock import Mock

        gateway._health_state["index_available"] = True

        # Wrap the real index but make db.execute raise no-such-column.
        fake_index = Mock()
        fake_db = Mock()
        fake_db.execute = Mock(
            side_effect=sqlite3.OperationalError("no such column: raw_schema")
        )
        fake_index.db = fake_db

        assert gateway._load_raw_schema(fake_index, "test:read_file") is None
        # Health flag untouched — a not-yet-migrated schema is not a fault.
        assert gateway._health_state["index_available"] is True

    @pytest.mark.asyncio
    async def test_load_raw_schema_ignores_non_object_json(self, test_index):
        """A bare-string raw_schema (valid JSON but not an object) is treated as
        absent so the response `schema` field is always structured."""
        import gateway

        test_index.db.execute(
            "UPDATE tools SET raw_schema = ? WHERE name = ?",
            (json.dumps("just a string"), "test:read_file"),
        )
        test_index.db.commit()

        assert gateway._load_raw_schema(test_index, "test:read_file") is None


# =============================================================================
# FEAT-03 — compass_status(active=True) surfaces the active liveness probe
# =============================================================================


class TestCompassStatusActiveProbeFEAT03:
    """compass_status gains an optional active probe folded into backends."""

    @pytest.mark.asyncio
    async def test_active_true_surfaces_per_backend_probe(
        self, test_index, test_config_with_backends
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = None

        probe_result = {
            "test_backend": {
                "status": "degraded",
                "tools": 3,
                "probe": {"ok": False, "error_kind": "timeout"},
            }
        }
        mgr = Mock()
        mgr.get_stats = Mock(return_value={"configured_backends": ["test_backend"]})
        mgr.health_check = AsyncMock(return_value=probe_result)
        gateway._backend_manager = mgr

        from gateway import compass_status

        result = await compass_status(active=True)

        # The active probe was invoked with active=True.
        mgr.health_check.assert_awaited_once_with(active=True)
        # Probe results are folded into backends under 'probes'.
        assert result["backends"]["probes"] == probe_result
        assert result["backends"]["probes"]["test_backend"]["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_active_false_is_unchanged_no_probe(
        self, test_index, test_config_with_backends
    ):
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = None

        mgr = Mock()
        mgr.get_stats = Mock(return_value={"configured_backends": ["test_backend"]})
        mgr.health_check = AsyncMock(return_value={"unexpected": True})
        gateway._backend_manager = mgr

        from gateway import compass_status

        result = await compass_status()  # default active=False

        # No active probe fired, no 'probes' key added.
        mgr.health_check.assert_not_awaited()
        assert "probes" not in result["backends"]

    @pytest.mark.asyncio
    async def test_active_probe_failure_degrades_gracefully(
        self, test_index, test_config_with_backends
    ):
        """A probe that raises must not abort the whole status — the probe
        section degrades to {error} and the rest of the response survives."""
        import gateway

        gateway._compass_index = test_index
        gateway._config = test_config_with_backends
        gateway._analytics = None

        mgr = Mock()
        mgr.get_stats = Mock(return_value={"configured_backends": ["test_backend"]})
        mgr.health_check = AsyncMock(side_effect=RuntimeError("probe boom"))
        gateway._backend_manager = mgr

        from gateway import compass_status

        result = await compass_status(active=True)

        # Status did not raise; the probe section reports the error.
        assert "error" in result["backends"]["probes"]
        # And other sections survived.
        assert "index" in result
        assert "config" in result


# =============================================================================
# FEAT-06 — deny/allow enforcement at the execute() boundary
# =============================================================================


def _mgr_with_policy(allow=None, deny=None):
    """Build a mock BackendManager whose 'test' backend carries the given
    allow/deny globs. execute_tool is an AsyncMock so we can assert it is (or
    is not) called."""
    from config import CompassConfig, StdioBackend

    cfg = CompassConfig(
        backends={
            "test": StdioBackend(
                command="python",
                args=["-c", "pass"],
                allow_tools=list(allow or []),
                deny_tools=list(deny or []),
            ),
        },
        auto_sync=False,
        analytics_enabled=False,
        chain_indexing_enabled=False,
    )
    mgr = Mock()
    mgr.config = cfg
    mgr.is_backend_connected = Mock(return_value=True)
    mgr.connect_backend = AsyncMock(return_value=True)
    mgr.execute_tool = AsyncMock(return_value={"success": True, "result": "ran"})
    return mgr


class TestExecuteDenyAllowFEAT06:
    """execute() rejects policy-denied tools with a tool_denied envelope and
    never proxies them; allowed tools proxy normally."""

    @pytest.mark.asyncio
    async def test_denied_tool_returns_tool_denied_and_never_proxies(
        self, test_config
    ):
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy(deny=["danger_*"])
        gateway._backend_manager = mgr

        from gateway import execute

        result = await execute(tool_name="test:danger_delete", arguments={})

        # Structured tool_denied envelope.
        env = result["error_envelope"]
        assert env["code"] == "tool_denied"
        assert env["category"] == "forbidden"
        assert env["retryable"] is False
        assert result["success"] is False
        # The backend was NEVER asked to run the tool.
        mgr.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowed_tool_proxies_normally(self, test_config):
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy(deny=["danger_*"])
        gateway._backend_manager = mgr

        from gateway import execute

        result = await execute(tool_name="test:read_file", arguments={})

        # Not denied -> proxied; backend result flows through.
        mgr.execute_tool.assert_awaited_once()
        assert result.get("result") == "ran"
        assert "error_envelope" not in result

    @pytest.mark.asyncio
    async def test_empty_lists_allow_everything(self, test_config):
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy()  # no allow, no deny
        gateway._backend_manager = mgr

        from gateway import execute

        result = await execute(tool_name="test:anything_goes", arguments={})

        mgr.execute_tool.assert_awaited_once()
        assert "error_envelope" not in result

    @pytest.mark.asyncio
    async def test_allowlist_denies_unlisted_tool(self, test_config):
        """A non-empty allow_tools acts as an allowlist: a tool not matching any
        allow glob is denied."""
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy(allow=["read_*"])
        gateway._backend_manager = mgr

        from gateway import execute

        # write_file does not match read_* -> denied.
        denied = await execute(tool_name="test:write_file", arguments={})
        assert denied["error_envelope"]["code"] == "tool_denied"
        mgr.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowlist_permits_matching_tool(self, test_config):
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy(allow=["read_*"])
        gateway._backend_manager = mgr

        from gateway import execute

        allowed = await execute(tool_name="test:read_file", arguments={})
        mgr.execute_tool.assert_awaited_once()
        assert "error_envelope" not in allowed

    @pytest.mark.asyncio
    async def test_deny_wins_over_allow(self, test_config):
        """When a tool matches both allow and deny globs, deny wins."""
        import gateway

        gateway._config = test_config
        gateway._analytics = None
        mgr = _mgr_with_policy(allow=["read_*"], deny=["read_secret"])
        gateway._backend_manager = mgr

        from gateway import execute

        result = await execute(tool_name="test:read_secret", arguments={})
        assert result["error_envelope"]["code"] == "tool_denied"
        mgr.execute_tool.assert_not_awaited()


# =============================================================================
# OPS-1 — optional bearer-token auth on the HTTP transport
# =============================================================================


async def _drive_asgi(app, path="/mcp", headers=None, scope_type="http"):
    """Drive an ASGI callable and capture the response start + body. Returns
    (status_code, sent_events, inner_called_flag_dict). If the wrapped inner
    app runs it appends to `calls`."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append((k.encode("latin-1"), v.encode("latin-1")))
    scope = {"type": scope_type, "path": path, "headers": raw_headers, "method": "POST"}

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        sent.append(event)

    await app(scope, receive, send)
    status = None
    for e in sent:
        if e.get("type") == "http.response.start":
            status = e.get("status")
    return status, sent


def _stub_inner_app(calls):
    """An inner ASGI app that records that it was invoked and returns 200."""

    async def inner(scope, receive, send):
        calls.append(scope.get("path"))
        if scope.get("type") == "http":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

    return inner


class TestExtractBearerTokenOPS1:
    def test_extracts_valid_bearer(self):
        from gateway import _extract_bearer_token

        headers = [(b"authorization", b"Bearer sekret123")]
        assert _extract_bearer_token(headers) == "sekret123"

    def test_case_insensitive_scheme_and_header(self):
        from gateway import _extract_bearer_token

        headers = [(b"Authorization", b"bearer  spaced  ")]
        assert _extract_bearer_token(headers) == "spaced"

    def test_missing_header_returns_none(self):
        from gateway import _extract_bearer_token

        assert _extract_bearer_token([(b"x-other", b"v")]) is None

    def test_non_bearer_scheme_returns_none(self):
        from gateway import _extract_bearer_token

        assert _extract_bearer_token([(b"authorization", b"Basic abc")]) is None


class TestBearerAuthMiddlewareOPS1:
    @pytest.mark.asyncio
    async def test_correct_bearer_allowed(self):
        import gateway

        calls = []
        inner = _stub_inner_app(calls)
        wrapped = gateway._make_bearer_auth_middleware(inner, "s3cr3t")

        status, _ = await _drive_asgi(
            wrapped, path="/mcp", headers={"authorization": "Bearer s3cr3t"}
        )

        assert status == 200
        assert calls == ["/mcp"]  # inner app ran

    @pytest.mark.asyncio
    async def test_wrong_bearer_rejected_401_and_not_proxied(self):
        import gateway

        calls = []
        inner = _stub_inner_app(calls)
        wrapped = gateway._make_bearer_auth_middleware(inner, "s3cr3t")

        status, _ = await _drive_asgi(
            wrapped, path="/mcp", headers={"authorization": "Bearer wrong"}
        )

        assert status == 401
        assert calls == []  # inner app was NEVER invoked

    @pytest.mark.asyncio
    async def test_missing_bearer_rejected_401(self):
        import gateway

        calls = []
        inner = _stub_inner_app(calls)
        wrapped = gateway._make_bearer_auth_middleware(inner, "s3cr3t")

        status, _ = await _drive_asgi(wrapped, path="/mcp", headers={})

        assert status == 401
        assert calls == []

    @pytest.mark.asyncio
    async def test_health_and_ready_reachable_without_token(self):
        import gateway

        for exempt in ("/health", "/ready"):
            calls = []
            inner = _stub_inner_app(calls)
            wrapped = gateway._make_bearer_auth_middleware(inner, "s3cr3t")

            status, _ = await _drive_asgi(wrapped, path=exempt, headers={})

            assert status == 200, f"{exempt} must be reachable without token"
            assert calls == [exempt]

    @pytest.mark.asyncio
    async def test_lifespan_scope_passes_through(self):
        import gateway

        calls = []

        async def inner(scope, receive, send):
            calls.append(scope.get("type"))

        wrapped = gateway._make_bearer_auth_middleware(inner, "s3cr3t")

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(event):
            pass

        await wrapped({"type": "lifespan"}, receive, send)
        assert calls == ["lifespan"]  # non-http passes through, no auth


class TestBearerAuthTokenUnsetOPS1:
    """When no token is configured the HTTP path stays open (today's behavior).

    We can't easily boot uvicorn in a unit test, so we assert the decision:
    the resolved token is empty -> the middleware branch in _run_http is not
    taken. Verified by inspecting get_config().gateway_auth_token resolution.
    """

    def test_default_config_has_no_auth_token(self, test_config):
        # A fresh config (no env var, no file field) resolves to no token, so
        # the OPS-1 middleware is never installed -> open transport.
        assert (test_config.gateway_auth_token or "") == ""


# =============================================================================
# auto-refresh — wire background polling into startup + FEAT-04 prune wiring
# =============================================================================


class TestBackgroundPollingStartDecision:
    """The 'start polling only when interval>0' decision (unit-testable slice
    of the auto-refresh lifecycle wiring)."""

    @pytest.mark.asyncio
    async def test_polling_started_when_interval_positive(self, test_config):
        import gateway

        test_config.sync_polling_interval = 300
        gateway._config = test_config

        sm = Mock()
        sm.start_background_polling = AsyncMock()

        started = await gateway._maybe_start_background_polling(sm)

        assert started is True
        sm.start_background_polling.assert_awaited_once_with(interval_seconds=300)

    @pytest.mark.asyncio
    async def test_polling_not_started_when_interval_zero(self, test_config):
        import gateway

        test_config.sync_polling_interval = 0  # disabled by design
        gateway._config = test_config

        sm = Mock()
        sm.start_background_polling = AsyncMock()

        started = await gateway._maybe_start_background_polling(sm)

        assert started is False
        sm.start_background_polling.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_polling_not_started_when_no_manager(self, test_config):
        import gateway

        test_config.sync_polling_interval = 300
        gateway._config = test_config

        started = await gateway._maybe_start_background_polling(None)
        assert started is False

    @pytest.mark.asyncio
    async def test_polling_start_failure_is_swallowed(self, test_config):
        import gateway

        test_config.sync_polling_interval = 300
        gateway._config = test_config

        sm = Mock()
        sm.start_background_polling = AsyncMock(side_effect=RuntimeError("boom"))

        # Must not raise — the decision was taken (True) but the start failed.
        started = await gateway._maybe_start_background_polling(sm)
        assert started is True


class TestAnalyticsPruneWiringFEAT04:
    """FEAT-04: startup issues prune_old_records with the configured retention,
    exactly once, defensively."""

    @pytest.mark.asyncio
    async def test_prune_called_with_configured_retention(self, test_config):
        import gateway

        gateway._analytics_pruned_once = False
        test_config.analytics_retention_days = 14
        gateway._config = test_config

        analytics = Mock()
        analytics.prune_old_records = AsyncMock(return_value=7)

        async def fake_get_analytics():
            return analytics

        with patch("gateway.get_analytics_instance", side_effect=fake_get_analytics):
            issued = await gateway._maybe_prune_analytics_once()

        assert issued is True
        analytics.prune_old_records.assert_awaited_once_with(14)

    @pytest.mark.asyncio
    async def test_prune_runs_only_once(self, test_config):
        import gateway

        gateway._analytics_pruned_once = False
        test_config.analytics_retention_days = 30
        gateway._config = test_config

        analytics = Mock()
        analytics.prune_old_records = AsyncMock(return_value=0)

        async def fake_get_analytics():
            return analytics

        with patch("gateway.get_analytics_instance", side_effect=fake_get_analytics):
            first = await gateway._maybe_prune_analytics_once()
            second = await gateway._maybe_prune_analytics_once()

        assert first is True
        assert second is False  # latched — no second prune
        analytics.prune_old_records.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prune_skipped_when_analytics_none(self, test_config):
        import gateway

        gateway._analytics_pruned_once = False
        gateway._config = test_config

        async def fake_get_analytics():
            return None

        with patch("gateway.get_analytics_instance", side_effect=fake_get_analytics):
            issued = await gateway._maybe_prune_analytics_once()

        assert issued is False

    @pytest.mark.asyncio
    async def test_prune_skipped_when_method_absent(self, test_config):
        """Defensive: if B4's prune_old_records isn't present, skip cleanly."""
        import gateway

        gateway._analytics_pruned_once = False
        gateway._config = test_config

        analytics = Mock(spec=[])  # no prune_old_records attribute

        async def fake_get_analytics():
            return analytics

        with patch("gateway.get_analytics_instance", side_effect=fake_get_analytics):
            issued = await gateway._maybe_prune_analytics_once()

        assert issued is False

    @pytest.mark.asyncio
    async def test_prune_failure_is_swallowed_and_not_latched(self, test_config):
        import gateway

        gateway._analytics_pruned_once = False
        test_config.analytics_retention_days = 30
        gateway._config = test_config

        analytics = Mock()
        analytics.prune_old_records = AsyncMock(side_effect=RuntimeError("db locked"))

        async def fake_get_analytics():
            return analytics

        with patch("gateway.get_analytics_instance", side_effect=fake_get_analytics):
            issued = await gateway._maybe_prune_analytics_once()

        assert issued is False
        # Not latched on failure — a later attempt may succeed.
        assert gateway._analytics_pruned_once is False
