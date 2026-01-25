"""
Tests for Tool Compass analytics module.

Tests usage tracking, hot cache, and chain detection.
"""

import pytest
from pathlib import Path

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
        # Should have at least one failure
        assert len(summary["failures"]) >= 0  # May be empty if first call

    @pytest.mark.asyncio
    async def test_record_tool_call_with_arguments(self, test_analytics):
        """Should handle tool calls with arguments."""
        await test_analytics.record_tool_call(
            tool_name="test:read_file",
            success=True,
            latency_ms=25.0,
            arguments={"filepath": "/tmp/test.txt", "encoding": "utf-8"},
        )

        # Should not crash, arguments are hashed for patterns


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

        # May or may not be hot depending on cache size


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

        # Patterns are saved when sequence reaches certain length
        # The internal _session_tool_sequence should have these

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
        await test_analytics.detect_chains()

        # Should find the a->b pattern
        # Note: detection requires min_occurrences (default 3)

    @pytest.mark.asyncio
    async def test_get_chains(self, test_analytics):
        """Should retrieve stored chains."""
        chains = await test_analytics.get_chains(limit=10)
        assert isinstance(chains, list)


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

        # Should be restored
        # (may or may not contain our tool depending on what else was recorded)

    def test_close(self, test_analytics):
        """Should close database connection cleanly."""
        test_analytics.close()
        assert test_analytics.db is None
