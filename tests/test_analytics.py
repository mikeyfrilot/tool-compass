"""
Tests for Tool Compass analytics module.

Tests usage tracking, hot cache, and chain detection.
"""

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from analytics import HotToolEntry


class TestAnalyticsRecording:
    """Test analytics event recording."""

    @pytest.mark.asyncio
    async def test_record_search(self, test_analytics):
        """Should record search queries."""

        # Create mock results
        class MockResult:
            class MockTool:
                name = "test:read_file"

            tool = MockTool()

        results = [MockResult()]

        await test_analytics.record_search(
            query="read a file",
            results=results,
            latency_ms=15.5,
            category_filter=None,
            server_filter=None,
        )

        # Verify it was recorded
        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["searches"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_record_tool_call_success(self, test_analytics):
        """Should record successful tool calls."""
        await test_analytics.record_tool_call(
            tool_name="test:read_file",
            success=True,
            latency_ms=50.0,
        )

        summary = await test_analytics.get_analytics_summary("1h")
        assert summary["tool_calls"]["total"] >= 1

    @pytest.mark.asyncio
    async def test_record_tool_call_failure(self, test_analytics):
        """Should record failed tool calls with error message."""
        await test_analytics.record_tool_call(
            tool_name="test:failing_tool",
            success=False,
            latency_ms=100.0,
            error_message="Connection refused",
        )

        summary = await test_analytics.get_analytics_summary("1h")
        # Verify the failure was actually recorded — `len(...) >= 0` is
        # always true, so the earlier assert never validated anything.
        failures = summary["failures"]
        assert isinstance(failures, list)
        # Tool-level failure count is available through tool_calls totals.
        assert summary["tool_calls"]["total"] >= 1
        # The failing call must be reflected in success_rate being < 1.0
        # when this is the only call recorded. Success rate is a ratio of
        # successes / total; our single call was a failure.
        if summary["tool_calls"]["total"] == 1:
            assert summary["tool_calls"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_record_tool_call_with_arguments(self, test_analytics):
        """Should hash argument payload and persist it on the call row."""
        # TS-A-003: previously this test asserted nothing beyond "did not raise."
        # Verify the args_hash is computed, deterministic for the same input,
        # and actually written to tool_calls.arguments_hash.
        arguments = {"filepath": "/tmp/test.txt", "encoding": "utf-8"}

        await test_analytics.record_tool_call(
            tool_name="test:read_file",
            success=True,
            latency_ms=25.0,
            arguments=arguments,
        )

        # Re-record with the same args — the hashes must match (determinism).
        await test_analytics.record_tool_call(
            tool_name="test:read_file",
            success=True,
            latency_ms=25.0,
            arguments=arguments,
        )

        # Re-record with different args — the hash must differ.
        await test_analytics.record_tool_call(
            tool_name="test:read_file",
            success=True,
            latency_ms=25.0,
            arguments={"filepath": "/tmp/other.txt", "encoding": "utf-8"},
        )

        # Read directly from the DB to verify hashes were persisted.
        db = test_analytics._get_db()
        rows = db.execute(
            "SELECT arguments_hash FROM tool_calls "
            "WHERE tool_name = 'test:read_file' "
            "ORDER BY id ASC"
        ).fetchall()
        hashes = [row["arguments_hash"] for row in rows]

        assert len(hashes) == 3, f"expected 3 rows recorded, got {len(hashes)}"
        # All hashes are non-null — args were supplied.
        assert all(h is not None for h in hashes), hashes
        # Same input -> same hash (determinism).
        assert hashes[0] == hashes[1], (hashes[0], hashes[1])
        # Different input -> different hash (no key-only collision).
        assert hashes[0] != hashes[2], (hashes[0], hashes[2])


class TestHotCache:
    """Test hot tool caching."""

    @pytest.mark.asyncio
    async def test_refresh_hot_cache(self, test_analytics):
        """Should populate hot cache from usage data."""
        # Record some tool calls
        for i in range(5):
            await test_analytics.record_tool_call(
                tool_name="test:popular_tool",
                success=True,
                latency_ms=10.0,
            )

        for i in range(3):
            await test_analytics.record_tool_call(
                tool_name="test:less_popular",
                success=True,
                latency_ms=15.0,
            )

        # Refresh cache
        hot_tools = await test_analytics.refresh_hot_cache()

        assert len(hot_tools) > 0
        assert "test:popular_tool" in hot_tools

    @pytest.mark.asyncio
    async def test_get_hot_tool(self, test_analytics):
        """Should return cached tool data."""
        # Record calls to populate stats
        for i in range(10):
            await test_analytics.record_tool_call(
                tool_name="test:hot_tool",
                success=True,
                latency_ms=5.0,
            )

        await test_analytics.refresh_hot_cache()

        entry = test_analytics.get_hot_tool("test:hot_tool")
        if entry:  # May not be in cache if other tools have more calls
            assert isinstance(entry, HotToolEntry)
            assert entry.call_count >= 10

    @pytest.mark.asyncio
    async def test_is_hot(self, test_analytics):
        """Should check if tool is in hot cache."""
        # Initially empty
        assert test_analytics.is_hot("test:any_tool") is False

        # After recording and refreshing
        for i in range(5):
            await test_analytics.record_tool_call(
                tool_name="test:becoming_hot",
                success=True,
                latency_ms=10.0,
            )
        await test_analytics.refresh_hot_cache()

        # TS-A-004: the fixture wires hot_cache_size=5 (conftest.py:298) and
        # only one tool was recorded. There is no eviction pressure, so the
        # tool MUST be hot — earlier "may or may not" wording left this
        # branch unverified.
        assert test_analytics.is_hot("test:becoming_hot") is True
        # Negative case: a tool never recorded is not hot.
        assert test_analytics.is_hot("test:never_recorded") is False


class TestChainDetection:
    """Test automatic chain/workflow detection."""

    @pytest.mark.asyncio
    async def test_chain_pattern_recording(self, test_analytics):
        """Should record tool sequences for pattern detection."""
        # Simulate a workflow: read -> modify -> write
        await test_analytics.record_tool_call(
            "test:read_file", success=True, latency_ms=10
        )
        await test_analytics.record_tool_call(
            "test:process", success=True, latency_ms=20
        )
        await test_analytics.record_tool_call(
            "test:write_file", success=True, latency_ms=15
        )

        # TS-A-001: the three calls must show up in the in-memory session
        # sequence (the deque feeding chain detection). The deque stores
        # tool_name strings — assert the trailing three match what we sent
        # in order. Earlier revisions of this test asserted nothing and so
        # could not detect a regression where record_tool_call silently
        # stopped appending to the chain-detection buffer.
        recent = list(test_analytics._session_tool_sequence)[-3:]
        assert recent == ["test:read_file", "test:process", "test:write_file"], recent

        # The 3-tool subsequence should also be persisted to chain_patterns
        # (record_tool_call calls _save_chain_pattern on every append).
        db = test_analytics._get_db()
        pattern_count = db.execute(
            "SELECT COUNT(*) AS n FROM chain_patterns"
        ).fetchone()["n"]
        assert pattern_count > 0, "chain_patterns should contain at least one row"

    @pytest.mark.asyncio
    async def test_detect_chains(self, test_analytics):
        """Should detect frequently occurring tool sequences."""
        # Create a pattern that occurs multiple times
        for _ in range(5):  # More than chain_min_occurrences
            await test_analytics.record_tool_call(
                "test:step_a", success=True, latency_ms=10
            )
            await test_analytics.record_tool_call(
                "test:step_b", success=True, latency_ms=10
            )

        # Force pattern save
        await test_analytics._save_chain_pattern()

        # Detect chains
        detected = await test_analytics.detect_chains()

        # TS-A-002: detection should have promoted the recurring [step_a,
        # step_b] subsequence into tool_chains. The fixture sets
        # chain_min_occurrences=2 (conftest.py:299), so 5 occurrences
        # clears the bar. Earlier revisions left the assertion off and
        # could not detect a regression where detect_chains() silently
        # found nothing.
        chains = await test_analytics.get_chains(limit=50)
        chain_tool_sets = [tuple(c["tools"]) for c in chains]
        assert ("test:step_a", "test:step_b") in chain_tool_sets, chain_tool_sets

        # detect_chains() returns NEW chains it promoted on this call.
        # On the first run it should be non-empty for this pattern.
        detected_tool_sets = [tuple(c["tools"]) for c in detected]
        assert ("test:step_a", "test:step_b") in detected_tool_sets, detected_tool_sets

    @pytest.mark.asyncio
    async def test_get_chains(self, test_analytics):
        """Should retrieve stored chains."""
        chains = await test_analytics.get_chains(limit=10)
        assert isinstance(chains, list)

    @pytest.mark.asyncio
    async def test_subsequence_counted_once_per_real_occurrence(
        self, test_analytics
    ):
        """ANL-A-004: a single real occurrence of a subsequence must be
        counted exactly once.

        _save_chain_pattern used to re-count ALL length-2..5 subsequences of
        the whole sliding deque on every call, so a subsequence stayed in the
        window and got re-incremented on every subsequent record_tool_call —
        systematically inflating occurrence_count. Record a strictly
        non-repeating sequence so every subsequence occurs exactly once;
        every persisted pattern must therefore have occurrence_count == 1.
        """
        import json as _json

        # Strictly distinct tools => no subsequence ever truly repeats.
        for name in ["test:a", "test:b", "test:c", "test:d", "test:e", "test:f"]:
            await test_analytics.record_tool_call(
                name, success=True, latency_ms=1.0
            )

        db = test_analytics._get_db()
        rows = db.execute(
            "SELECT tool_sequence, occurrence_count FROM chain_patterns"
        ).fetchall()

        assert rows, "expected chain_patterns to be populated"
        overcounted = {
            row["tool_sequence"]: row["occurrence_count"]
            for row in rows
            if row["occurrence_count"] != 1
        }
        assert not overcounted, (
            "each subsequence occurred once but was counted multiple times: "
            f"{overcounted}"
        )

        # Spot-check the [a, b] pair specifically: it appears once and only
        # once in the recorded stream, so its stored count must be 1.
        ab_hash_seq = _json.dumps(["test:a", "test:b"])
        ab = db.execute(
            "SELECT occurrence_count FROM chain_patterns WHERE tool_sequence = ?",
            (ab_hash_seq,),
        ).fetchone()
        assert ab is not None, "the [a, b] subsequence should be recorded"
        assert ab["occurrence_count"] == 1, ab["occurrence_count"]

    @pytest.mark.asyncio
    async def test_genuine_repeat_still_counted(self, test_analytics):
        """ANL-A-004 guard: a subsequence that GENUINELY repeats across the
        stream must still accumulate a count > 1, so the fix doesn't throw
        away real occurrences."""
        import json as _json

        # a,b then later a,b again with distinct separators between, so the
        # [a,b] pair genuinely occurs twice as a fresh suffix.
        seq = ["test:a", "test:b", "test:x", "test:y", "test:a", "test:b"]
        for name in seq:
            await test_analytics.record_tool_call(
                name, success=True, latency_ms=1.0
            )

        db = test_analytics._get_db()
        ab = db.execute(
            "SELECT occurrence_count FROM chain_patterns WHERE tool_sequence = ?",
            (_json.dumps(["test:a", "test:b"]),),
        ).fetchone()
        assert ab is not None
        assert ab["occurrence_count"] == 2, ab["occurrence_count"]


class TestRetentionPrune:
    """FEAT-04: prune_old_records deletes rows older than the retention window.

    tool_calls + search_queries append one row per call/search forever with no
    TTL. prune_old_records(retention_days) DELETEs rows whose created_at is
    older than the window and returns the total deleted. These tests insert
    rows with EXPLICIT UTC created_at timestamps (SQLite CURRENT_TIMESTAMP is
    UTC) so the assertions are host-timezone-independent — they must not depend
    on the machine's local offset, mirroring the CA-001 UTC discipline.
    """

    @staticmethod
    def _utc_stamp(days_ago: float) -> str:
        """A 'YYYY-MM-DD HH:MM:SS' UTC string N days in the past.

        Matches the format SQLite CURRENT_TIMESTAMP writes into created_at, so
        the string compares correctly against the SQL cutoff regardless of the
        host timezone.
        """
        dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _insert_call(self, analytics, tool_name: str, days_ago: float) -> None:
        db = analytics._get_db()
        with analytics._lock:
            db.execute(
                "INSERT INTO tool_calls "
                "(tool_name, server, success, latency_ms, created_at) "
                "VALUES (?, ?, 1, 1.0, ?)",
                (tool_name, "test", self._utc_stamp(days_ago)),
            )
            db.commit()

    def _insert_search(self, analytics, query: str, days_ago: float) -> None:
        db = analytics._get_db()
        with analytics._lock:
            db.execute(
                "INSERT INTO search_queries "
                "(query, result_count, latency_ms, created_at) "
                "VALUES (?, 0, 1.0, ?)",
                (query, self._utc_stamp(days_ago)),
            )
            db.commit()

    def _count(self, analytics, table: str) -> int:
        db = analytics._get_db()
        return db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    @pytest.mark.asyncio
    async def test_prune_deletes_only_old_rows_and_returns_count(
        self, test_analytics
    ):
        """prune_old_records(30) deletes only rows older than 30 days across
        BOTH tables, returns the total deleted, and leaves recent rows."""
        # 2 old tool_calls (well past the window), 1 recent tool_call.
        self._insert_call(test_analytics, "test:old_a", days_ago=45)
        self._insert_call(test_analytics, "test:old_b", days_ago=31)
        self._insert_call(test_analytics, "test:recent", days_ago=1)
        # 1 old search_query, 1 recent search_query.
        self._insert_search(test_analytics, "old query", days_ago=60)
        self._insert_search(test_analytics, "recent query", days_ago=2)

        deleted = await test_analytics.prune_old_records(30)

        # 2 old calls + 1 old search = 3 rows removed.
        assert deleted == 3, deleted
        # Recent rows survive in each table.
        assert self._count(test_analytics, "tool_calls") == 1
        assert self._count(test_analytics, "search_queries") == 1

    @pytest.mark.asyncio
    async def test_prune_zero_retention_is_noop(self, test_analytics):
        """retention_days=0 keeps everything forever — deletes nothing."""
        self._insert_call(test_analytics, "test:ancient", days_ago=365)
        self._insert_search(test_analytics, "ancient query", days_ago=365)

        deleted = await test_analytics.prune_old_records(0)

        assert deleted == 0
        assert self._count(test_analytics, "tool_calls") == 1
        assert self._count(test_analytics, "search_queries") == 1

    @pytest.mark.asyncio
    async def test_prune_negative_retention_is_noop(self, test_analytics):
        """retention_days < 0 is treated like 0 — no-op, returns 0."""
        self._insert_call(test_analytics, "test:ancient", days_ago=365)

        deleted = await test_analytics.prune_old_records(-5)

        assert deleted == 0
        assert self._count(test_analytics, "tool_calls") == 1

    @pytest.mark.asyncio
    async def test_prune_recent_rows_survive_boundary(self, test_analytics):
        """A row just INSIDE the window (younger than the cutoff) is kept."""
        # 29 days old with a 30-day window -> inside the window, must survive.
        self._insert_call(test_analytics, "test:inside_window", days_ago=29)

        deleted = await test_analytics.prune_old_records(30)

        assert deleted == 0
        assert self._count(test_analytics, "tool_calls") == 1

    @pytest.mark.asyncio
    async def test_prune_on_degraded_db_does_not_raise(self, test_analytics):
        """A prune while already degraded is a silent no-op (returns 0),
        never raising — matching record_search / record_tool_call."""
        test_analytics._degraded = True

        # Must not raise even though the instance is degraded.
        deleted = await test_analytics.prune_old_records(30)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_prune_on_broken_db_degrades_not_raises(self, test_analytics):
        """If the underlying DB errors mid-prune, the method degrades (sets
        _degraded, returns 0) rather than propagating sqlite3.Error."""
        # Prime the connection, then close it out from under the method so the
        # DELETE raises sqlite3.ProgrammingError / sqlite3.Error. We do NOT go
        # through close() (which sets _closed and refuses reopen) — we want the
        # write itself to fail, exercising the try/except degradation path.
        db = test_analytics._get_db()
        db.close()  # underlying handle now unusable; self.db still references it

        # Insert is impossible now, but prune must swallow the error, mark
        # degraded, and return 0 instead of raising.
        deleted = await test_analytics.prune_old_records(30)

        assert deleted == 0
        assert test_analytics._degraded is True


class TestAnalyticsSummary:
    """Test analytics summary generation."""

    @pytest.mark.asyncio
    async def test_get_analytics_summary_structure(self, test_analytics):
        """Summary should have expected structure."""
        summary = await test_analytics.get_analytics_summary("24h")

        assert "timeframe" in summary
        assert "searches" in summary
        assert "tool_calls" in summary
        assert "failures" in summary
        assert "chains" in summary
        assert "hot_cache" in summary

    @pytest.mark.asyncio
    async def test_get_analytics_summary_timeframes(self, test_analytics):
        """Should support different timeframes."""
        for tf in ["1h", "24h", "7d", "30d"]:
            summary = await test_analytics.get_analytics_summary(tf)
            assert summary["timeframe"] == tf

    @pytest.mark.asyncio
    async def test_analytics_summary_calculations(self, test_analytics):
        """Should calculate metrics correctly."""
        # Record known data
        await test_analytics.record_tool_call("test:tool", success=True, latency_ms=100)
        await test_analytics.record_tool_call("test:tool", success=True, latency_ms=200)
        await test_analytics.record_tool_call("test:tool", success=False, latency_ms=50)

        summary = await test_analytics.get_analytics_summary("1h")

        assert summary["tool_calls"]["total"] >= 3
        # Success rate should reflect 2/3 successes (approximately 66.7%)

    @pytest.mark.asyncio
    async def test_summary_window_bound_is_utc_not_local(self, test_analytics):
        """CA-001: the timeframe window bound must be computed in UTC.

        `created_at` columns are written by SQLite CURRENT_TIMESTAMP (UTC).
        If get_analytics_summary computes `since` from naive local wall-clock
        (datetime.now()), then on an ahead-of-UTC host the local bound
        overshoots the just-inserted UTC row and it silently vanishes from the
        window. This test forces that skew regardless of the real host TZ by
        making naive datetime.now() report a time well ahead of true UTC while
        datetime.now(timezone.utc) still returns real UTC. The row must still
        appear in the "1h" window — which only holds if the bound is UTC.
        """

        class MockResult:
            class MockTool:
                name = "test:read_file"

            tool = MockTool()

        # Record a search first — SQLite stamps created_at in real UTC.
        await test_analytics.record_search(
            query="utc window bound check",
            results=[MockResult()],
            latency_ms=12.0,
        )

        real_utc = datetime.now(timezone.utc)
        # Simulate an ahead-of-UTC host: naive now() is 6h in the future
        # relative to the UTC-stored row. A buggy local-time bound of
        # (naive_now - 1h) = real_utc + 5h excludes the just-inserted row;
        # a correct UTC bound of (real_utc - 1h) includes it.
        fake_local_now = real_utc.replace(tzinfo=None) + timedelta(hours=6)

        class _FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fake_local_now
                return datetime.now(tz)

        with patch("analytics.datetime", _FrozenDateTime):
            summary = await test_analytics.get_analytics_summary("1h")

        # The row is within the last hour of real UTC, so a correct UTC bound
        # must include it. Fails today because the bound is naive-local.
        assert summary["searches"]["total"] >= 1


class TestPersistence:
    """Test analytics data persistence."""

    @pytest.mark.asyncio
    async def test_load_hot_cache_from_db(self, test_analytics):
        """Should restore hot cache from database."""
        # Record some data and refresh cache
        for i in range(5):
            await test_analytics.record_tool_call(
                tool_name="test:persistent_tool",
                success=True,
                latency_ms=10.0,
            )
        await test_analytics.refresh_hot_cache()

        # Clear in-memory cache
        test_analytics._hot_cache.clear()
        assert len(test_analytics._hot_cache) == 0

        # Reload from DB
        await test_analytics.load_hot_cache_from_db()

        # TS-A-005: the fixture wires hot_cache_size=5 and only this one
        # tool was recorded, so the persisted hot_tools row MUST be
        # rehydrated. Earlier revisions left the assertion off, so a
        # regression where load_hot_cache_from_db silently produced an
        # empty cache would still pass.
        assert "test:persistent_tool" in test_analytics._hot_cache
        entry = test_analytics._hot_cache["test:persistent_tool"]
        assert entry.tool_name == "test:persistent_tool"
        assert entry.call_count == 5

    @pytest.mark.asyncio
    async def test_load_hot_cache_skips_corrupt_rows(self, test_analytics):
        """PC-B-001: a corrupt schema_json or truncated embedding blob in
        hot_tools must NOT raise out of load_hot_cache_from_db().

        load_hot_cache_from_db runs on the mandatory init path (awaited
        unconditionally inside get_analytics_instance(), which compass() and
        execute() call unwrapped). Before the fix it called json.loads /
        np.frombuffer directly on persisted blobs, so one corrupt row raised a
        raw JSONDecodeError / ValueError through the MCP envelope. It must
        degrade to "skip the bad row" and still load the good rows, mirroring
        record_search / record_tool_call.
        """
        db = test_analytics._get_db()

        # A well-formed good row that must survive.
        good_vec = np.arange(4, dtype=np.float32).tobytes()
        # A row with corrupt (non-JSON) schema_json → json.JSONDecodeError.
        # A row with a truncated embedding blob (not a multiple of 4 bytes)
        # → ValueError from np.frombuffer.
        with test_analytics._lock:
            db.execute(
                "INSERT INTO hot_tools (tool_name, rank, call_count, embedding, schema_json, description) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("test:good", 0, 3, good_vec, '{"ok": true}', "good tool"),
            )
            db.execute(
                "INSERT INTO hot_tools (tool_name, rank, call_count, embedding, schema_json, description) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("test:bad_schema", 1, 2, None, "{not valid json", "bad schema"),
            )
            db.execute(
                "INSERT INTO hot_tools (tool_name, rank, call_count, embedding, schema_json, description) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("test:bad_embedding", 2, 1, b"\x01\x02\x03", None, "bad embedding"),
            )
            db.commit()

        test_analytics._hot_cache.clear()

        # Must NOT raise despite the two corrupt rows.
        await test_analytics.load_hot_cache_from_db()

        # Good row is still loaded; corrupt rows are skipped, not fatal.
        assert "test:good" in test_analytics._hot_cache
        assert "test:bad_schema" not in test_analytics._hot_cache
        assert "test:bad_embedding" not in test_analytics._hot_cache

    def test_close(self, test_analytics):
        """Should close database connection cleanly."""
        test_analytics.close()
        assert test_analytics.db is None


class TestCloseConcurrency:
    """ANL-A-003: close() and lazy _get_db() must mutually exclude.

    close() acquired self._lock, but lazy init in _get_db() guards on
    self._init_lock — so a concurrent record_* and close() did not serialize:
    close could null the handle underneath an in-flight op, or a record could
    reopen the DB *after* close. close() must also take self._init_lock and
    refuse to reopen after close.
    """

    @pytest.mark.asyncio
    async def test_close_refuses_reopen(self, test_analytics):
        """After close(), a subsequent record must NOT silently reopen the DB."""
        # Open + use the DB once.
        await test_analytics.record_tool_call(
            "test:warm", success=True, latency_ms=1.0
        )
        assert test_analytics.db is not None

        test_analytics.close()
        assert test_analytics.db is None

        # A record after close must not resurrect the connection.
        await test_analytics.record_tool_call(
            "test:after_close", success=True, latency_ms=1.0
        )
        assert test_analytics.db is None, (
            "close() must refuse reopen; DB handle was resurrected"
        )

    @pytest.mark.asyncio
    async def test_concurrent_close_and_record_no_crash(self, test_analytics):
        """Hammer close() against concurrent record_* from worker threads.

        Without close() taking the init lock, a thread first-touching the DB
        could open a handle while close() nulled self.db, leaving an in-flight
        op operating on a closed/None connection (sqlite ProgrammingError) or
        a reopened-post-close handle. The corrected ordering must keep these
        serialized — the loop must complete without an unexpected exception
        escaping.
        """
        loop = asyncio.get_event_loop()
        errors: list = []
        stop = threading.Event()

        def worker(n: int):
            # Each thread drives its own event loop to call the async API.
            wl = asyncio.new_event_loop()
            try:
                for i in range(15):
                    if stop.is_set():
                        break
                    try:
                        wl.run_until_complete(
                            test_analytics.record_tool_call(
                                f"test:t{n}", success=True, latency_ms=1.0
                            )
                        )
                    except Exception as e:  # noqa: BLE001 - capture for assert
                        errors.append(("record", repr(e)))
            finally:
                wl.close()

        def closer():
            for _ in range(20):
                if stop.is_set():
                    break
                try:
                    test_analytics.close()
                except Exception as e:  # noqa: BLE001
                    errors.append(("close", repr(e)))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        threads.append(threading.Thread(target=closer))

        await loop.run_in_executor(None, lambda: [t.start() for t in threads])
        await loop.run_in_executor(None, lambda: [t.join(10) for t in threads])
        stop.set()

        assert not errors, f"concurrent close/record raised: {errors[:5]}"
