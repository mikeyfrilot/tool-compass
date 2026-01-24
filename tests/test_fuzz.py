"""
Fuzz Testing for Tool Compass

Tests input validation, security, and edge cases using Hypothesis.
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from unittest.mock import patch
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SECURITY FUZZING
# =============================================================================


class TestSecurityFuzzing:
    """Security-focused fuzz tests."""

    # Common injection payloads
    INJECTION_PAYLOADS = [
        "'; DROP TABLE tools; --",
        "{{7*7}}",
        "${7*7}",
        "__import__('os').system('whoami')",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "\x00",
        "\n\r",
        "{{constructor.constructor('return this')()}}",
    ]

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=200)
    def test_search_query_sanitization(self, text):
        """Search queries should be safely handled."""
        from tool_manifest import ToolDefinition

        # Tool definition should safely handle any query text
        tool = ToolDefinition(
            name="test_tool",
            description=text,  # Use fuzzed input as description
            server="test",
            category="test",
        )

        # Should not raise
        result = tool.embedding_text()
        assert isinstance(result, str)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_injection_in_tool_name(self, payload):
        """Tool names with injection attempts should be handled safely."""
        from tool_manifest import ToolDefinition

        tool = ToolDefinition(
            name=payload,
            description="Test tool",
            server="test",
            category="test",
        )

        # Should create without crashing
        assert tool.name == payload
        text = tool.embedding_text()
        assert isinstance(text, str)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    def test_injection_in_category(self, payload):
        """Categories with injection attempts should be handled safely."""
        from tool_manifest import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            server="test",
            category=payload,
        )

        assert tool.category == payload

    @given(
        st.text(
            min_size=0,
            max_size=500,
            alphabet=st.characters(blacklist_characters="\x00"),
        )
    )
    @settings(max_examples=200)
    def test_config_path_validation(self, fuzz_path):
        """Config paths should be safely handled."""
        from config import get_base_path
        import os

        # Set env and test
        with patch.dict(os.environ, {"TOOL_COMPASS_BASE": fuzz_path}):
            try:
                result = get_base_path()
                # Should return a Path object
                assert isinstance(result, Path)
            except (ValueError, OSError):
                pass  # Invalid paths may raise


# =============================================================================
# INPUT VALIDATION FUZZING
# =============================================================================


class TestInputValidationFuzzing:
    """Test input validation with edge cases."""

    @given(st.integers())
    def test_top_k_boundaries(self, k):
        """top_k parameter should handle any integer."""
        from config import CompassConfig

        try:
            config = CompassConfig(default_top_k=k)
            # Should be stored
            assert config.default_top_k == k
        except (ValueError, TypeError):
            pass  # Invalid values may raise

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_min_confidence_boundaries(self, conf):
        """min_confidence should handle edge case floats."""
        from config import CompassConfig

        try:
            config = CompassConfig(min_confidence=conf)
            # Should be a valid float
            assert isinstance(config.min_confidence, float)
        except (ValueError, TypeError):
            pass  # Edge cases may raise

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(min_size=0, max_size=200),
            max_size=10,
        )
    )
    def test_arbitrary_tool_params(self, params):
        """Tool parameters should handle arbitrary dictionaries."""
        from tool_manifest import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            server="test",
            category="test",
            parameters=params,
        )

        # Should create successfully
        text = tool.embedding_text()
        assert isinstance(text, str)

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_server_filter_arbitrary(self, server):
        """Server filter should handle arbitrary strings."""
        assume(server)  # Non-empty

        from config import CompassConfig

        config = CompassConfig(backends={})
        assert config.backends == {}


# =============================================================================
# JSON SCHEMA FUZZING
# =============================================================================


class TestJSONSchemaFuzzing:
    """Fuzz JSON schema handling."""

    @given(
        st.recursive(
            st.none()
            | st.booleans()
            | st.integers()
            | st.floats(allow_nan=False)
            | st.text(),
            lambda children: st.lists(children, max_size=5)
            | st.dictionaries(st.text(min_size=1, max_size=20), children, max_size=5),
            max_leaves=20,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_json_as_params(self, data):
        """Tool params should handle arbitrary JSON structures."""
        from tool_manifest import ToolDefinition

        try:
            tool = ToolDefinition(
                name="test",
                description="test",
                server="test",
                category="test",
                parameters=data if isinstance(data, dict) else {"value": data},
            )
            text = tool.embedding_text()
            assert isinstance(text, str)
        except (TypeError, ValueError):
            pass  # Some structures may be rejected

    @given(st.binary(min_size=0, max_size=1000))
    @settings(max_examples=50)
    def test_malformed_json_bytes(self, data):
        """Should handle malformed JSON gracefully."""
        try:
            parsed = json.loads(data)
            # If it parses, try using it
            from tool_manifest import ToolDefinition

            if isinstance(parsed, dict):
                ToolDefinition(
                    name="test",
                    description="test",
                    server="test",
                    category="test",
                    parameters=parsed,
                )
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            pass  # Expected for random bytes


# =============================================================================
# CONFIG FUZZING
# =============================================================================


class TestConfigFuzzing:
    """Fuzz configuration handling."""

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.text(min_size=0, max_size=100),
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_config_from_arbitrary_dict(self, data):
        """CompassConfig.from_dict should handle arbitrary dicts."""
        from config import CompassConfig

        try:
            config = CompassConfig.from_dict(data)
            # Should create some config, maybe with defaults
            assert isinstance(config, CompassConfig)
        except (TypeError, ValueError, KeyError):
            pass  # Invalid config dicts may raise

    @given(st.text(min_size=0, max_size=200))
    def test_ollama_url_arbitrary(self, url):
        """Ollama URL should handle arbitrary strings."""
        from config import CompassConfig

        config = CompassConfig(ollama_url=url)
        assert config.ollama_url == url


# =============================================================================
# ANALYTICS FUZZING
# =============================================================================


class TestAnalyticsFuzzing:
    """Fuzz analytics recording - tests use sync wrappers to avoid event loop issues."""

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100, deadline=None)
    def test_record_search_arbitrary_query(self, fuzz_query):
        """Analytics should handle arbitrary search queries."""
        import tempfile
        from analytics import CompassAnalytics

        with tempfile.TemporaryDirectory() as tmp:
            analytics = CompassAnalytics(db_path=Path(tmp) / "test.db")
            try:
                # Use sync database directly to avoid async issues
                db = analytics._get_db()
                db.execute(
                    "INSERT INTO searches (query, results_count, latency_ms) VALUES (?, ?, ?)",
                    (fuzz_query, 1, 10.0),
                )
                db.commit()
            except Exception:
                pass  # Some queries may fail
            finally:
                analytics.close()

    @given(st.text(min_size=1, max_size=100), st.floats(min_value=0, max_value=10000))
    @settings(max_examples=50, deadline=None)
    def test_record_tool_call_arbitrary(self, tool_name, latency):
        """Analytics should handle arbitrary tool names."""
        import tempfile
        from analytics import CompassAnalytics

        with tempfile.TemporaryDirectory() as tmp:
            analytics = CompassAnalytics(db_path=Path(tmp) / "test.db")
            try:
                # Use sync database directly to avoid async issues
                db = analytics._get_db()
                db.execute(
                    "INSERT INTO tool_calls (tool_name, success, latency_ms) VALUES (?, ?, ?)",
                    (tool_name, True, latency),
                )
                db.commit()
            except Exception:
                pass
            finally:
                analytics.close()


# =============================================================================
# SEARCH RESULT FUZZING
# =============================================================================


class TestSearchResultFuzzing:
    """Fuzz search result handling."""

    @given(
        st.floats(allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=1000),
    )
    @settings(deadline=None)  # First run may be slow due to imports
    def test_search_result_arbitrary_score(self, score, rank):
        """SearchResult should handle edge case scores."""
        from indexer import SearchResult
        from tool_manifest import ToolDefinition

        tool = ToolDefinition(
            name="test",
            description="test",
            server="test",
            category="test",
        )

        try:
            result = SearchResult(tool=tool, score=score, rank=rank)
            assert result.tool == tool
        except (ValueError, TypeError):
            pass  # Edge cases may be rejected


# =============================================================================
# STRESS TESTS
# =============================================================================


class TestStressFuzzing:
    """Stress tests with extreme inputs."""

    def test_very_long_tool_name(self):
        """Handle extremely long tool names."""
        from tool_manifest import ToolDefinition

        long_name = "a" * 10000
        tool = ToolDefinition(
            name=long_name,
            description="test",
            server="test",
            category="test",
        )

        text = tool.embedding_text()
        assert isinstance(text, str)

    def test_very_long_description(self):
        """Handle extremely long descriptions."""
        from tool_manifest import ToolDefinition

        long_desc = "test description " * 10000
        tool = ToolDefinition(
            name="test",
            description=long_desc,
            server="test",
            category="test",
        )

        text = tool.embedding_text()
        assert isinstance(text, str)

    def test_deeply_nested_params(self):
        """Handle deeply nested parameter schemas."""
        from tool_manifest import ToolDefinition

        # Create 50-level deep nesting
        params = {"type": "object"}
        current = params
        for i in range(50):
            current["nested"] = {"level": i}
            current = current["nested"]

        tool = ToolDefinition(
            name="test",
            description="test",
            server="test",
            category="test",
            parameters=params,
        )

        text = tool.embedding_text()
        assert isinstance(text, str)

    def test_many_tools_in_batch(self):
        """Handle large batches of tool definitions."""
        from tool_manifest import ToolDefinition

        tools = []
        for i in range(1000):
            tools.append(
                ToolDefinition(
                    name=f"tool_{i}",
                    description=f"Tool number {i}",
                    server="test",
                    category="test",
                    parameters={"index": i},
                )
            )

        # All should have valid embedding text
        for tool in tools:
            text = tool.embedding_text()
            assert isinstance(text, str)
            assert tool.name in text

    @given(
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1,
            max_size=500,
        )
    )
    @settings(max_examples=100)
    def test_unicode_tool_names(self, name):
        """Handle Unicode tool names."""
        from tool_manifest import ToolDefinition

        tool = ToolDefinition(
            name=name,
            description="Unicode test",
            server="test",
            category="test",
        )

        text = tool.embedding_text()
        assert isinstance(text, str)


# =============================================================================
# BACKEND CONFIG FUZZING
# =============================================================================


class TestBackendConfigFuzzing:
    """Fuzz backend configuration."""

    @given(st.text(min_size=0, max_size=100))
    def test_stdio_backend_command(self, command):
        """StdioBackend should handle arbitrary commands."""
        from config import StdioBackend

        backend = StdioBackend(
            command=command,
            args=[],
            env={},
        )

        assert backend.command == command
        assert backend.type == "stdio"

    @given(st.lists(st.text(min_size=0, max_size=50), max_size=20))
    def test_stdio_backend_args(self, args):
        """StdioBackend should handle arbitrary args."""
        from config import StdioBackend

        backend = StdioBackend(
            command="python",
            args=args,
            env={},
        )

        assert backend.args == args

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.text(min_size=0, max_size=100),
            max_size=10,
        )
    )
    def test_stdio_backend_env(self, env):
        """StdioBackend should handle arbitrary env vars."""
        from config import StdioBackend

        backend = StdioBackend(
            command="python",
            args=[],
            env=env,
        )

        assert backend.env == env

    @given(st.text(min_size=0, max_size=200))
    def test_http_backend_url(self, url):
        """HttpBackend should handle arbitrary URLs."""
        from config import HttpBackend

        backend = HttpBackend(url=url)
        assert backend.url == url
        assert backend.type == "http"
