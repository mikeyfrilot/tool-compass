"""
Tests for Tool Compass tool manifest module.

Tests tool definitions, filtering, and export functionality.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_manifest import (
    ToolDefinition,
    TOOLS,
    get_all_tools,
    get_tools_by_category,
    get_tools_by_server,
    get_core_tools,
    get_categories,
    get_servers,
    export_manifest,
)


# =============================================================================
# ToolDefinition Tests
# =============================================================================


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_creation_with_all_fields(self):
        """Should create tool with all fields."""
        tool = ToolDefinition(
            name="test:read_file",
            description="Read file contents",
            category="file",
            server="test",
            parameters={"filepath": "str", "encoding": "str?"},
            examples=["read file", "get file contents"],
            is_core=True,
        )

        assert tool.name == "test:read_file"
        assert tool.description == "Read file contents"
        assert tool.category == "file"
        assert tool.server == "test"
        assert tool.parameters == {"filepath": "str", "encoding": "str?"}
        assert tool.examples == ["read file", "get file contents"]
        assert tool.is_core is True

    def test_creation_with_defaults(self):
        """Should use defaults for optional fields."""
        tool = ToolDefinition(
            name="test:simple",
            description="Simple tool",
            category="test",
            server="test",
        )

        assert tool.parameters == {}
        assert tool.examples == []
        assert tool.is_core is False

    def test_embedding_text_includes_name(self):
        """Embedding text should include tool name."""
        tool = ToolDefinition(
            name="bridge:read_file",
            description="Read file",
            category="file",
            server="bridge",
        )

        text = tool.embedding_text()

        assert "bridge:read_file" in text

    def test_embedding_text_includes_description(self):
        """Embedding text should include description."""
        tool = ToolDefinition(
            name="test:tool",
            description="A very specific description",
            category="test",
            server="test",
        )

        text = tool.embedding_text()

        assert "A very specific description" in text

    def test_embedding_text_includes_category(self):
        """Embedding text should include category."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test",
            category="file",
            server="test",
        )

        text = tool.embedding_text()

        assert "file" in text.lower()

    def test_embedding_text_includes_examples(self):
        """Embedding text should include examples."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test",
            category="test",
            server="test",
            examples=["example one", "example two"],
        )

        text = tool.embedding_text()

        assert "example one" in text
        assert "example two" in text

    def test_embedding_text_includes_parameters(self):
        """Embedding text should include parameter names."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test",
            category="test",
            server="test",
            parameters={"filepath": "str", "content": "str"},
        )

        text = tool.embedding_text()

        assert "filepath" in text
        assert "content" in text

    def test_embedding_text_no_examples(self):
        """Should handle empty examples."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test",
            category="test",
            server="test",
            examples=[],
        )

        text = tool.embedding_text()

        assert "Use cases" not in text

    def test_embedding_text_no_parameters(self):
        """Should handle empty parameters."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test",
            category="test",
            server="test",
            parameters={},
        )

        text = tool.embedding_text()

        assert "Parameters" not in text


# =============================================================================
# Serialization Tests
# =============================================================================


class TestToolSerialization:
    """Test ToolDefinition serialization."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        tool = ToolDefinition(
            name="test:tool",
            description="Test tool",
            category="test",
            server="test",
            parameters={"arg": "str"},
            examples=["example"],
            is_core=True,
        )

        data = tool.to_dict()

        assert data["name"] == "test:tool"
        assert data["description"] == "Test tool"
        assert data["category"] == "test"
        assert data["server"] == "test"
        assert data["parameters"] == {"arg": "str"}
        assert data["examples"] == ["example"]
        assert data["is_core"] is True

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "name": "test:tool",
            "description": "Test tool",
            "category": "test",
            "server": "test",
            "parameters": {"arg": "str"},
            "examples": ["example"],
            "is_core": True,
        }

        tool = ToolDefinition.from_dict(data)

        assert tool.name == "test:tool"
        assert tool.description == "Test tool"
        assert tool.category == "test"
        assert tool.server == "test"
        assert tool.parameters == {"arg": "str"}
        assert tool.examples == ["example"]
        assert tool.is_core is True

    def test_roundtrip(self):
        """Should survive dict roundtrip."""
        original = ToolDefinition(
            name="test:roundtrip",
            description="Roundtrip test",
            category="test",
            server="test",
            parameters={"a": "str", "b": "int?"},
            examples=["ex1", "ex2"],
            is_core=False,
        )

        data = original.to_dict()
        restored = ToolDefinition.from_dict(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.category == original.category
        assert restored.server == original.server
        assert restored.parameters == original.parameters
        assert restored.examples == original.examples
        assert restored.is_core == original.is_core


# =============================================================================
# TOOLS Constant Tests
# =============================================================================


class TestToolsConstant:
    """Test the TOOLS constant."""

    def test_tools_not_empty(self):
        """Should have tool definitions."""
        assert len(TOOLS) > 0

    def test_all_tools_have_required_fields(self):
        """All tools should have required fields."""
        for tool in TOOLS:
            assert tool.name, "Tool missing name"
            assert tool.description, f"Tool {tool.name} missing description"
            assert tool.category, f"Tool {tool.name} missing category"
            assert tool.server, f"Tool {tool.name} missing server"

    def test_all_tool_names_unique(self):
        """All tool names should be unique."""
        names = [t.name for t in TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_tool_names_follow_format(self):
        """Tool names should follow server:name format."""
        for tool in TOOLS:
            assert ":" in tool.name, f"Tool name {tool.name} should contain ':'"
            parts = tool.name.split(":")
            assert len(parts) == 2, f"Tool name {tool.name} should have exactly one ':'"
            assert parts[0], f"Tool name {tool.name} has empty server part"
            assert parts[1], f"Tool name {tool.name} has empty name part"

    def test_server_matches_name_prefix(self):
        """Server field should match name prefix."""
        for tool in TOOLS:
            prefix = tool.name.split(":")[0]
            assert tool.server == prefix, (
                f"Tool {tool.name} server mismatch: {tool.server} != {prefix}"
            )


# =============================================================================
# Query Functions Tests
# =============================================================================


class TestGetAllTools:
    """Test get_all_tools function."""

    def test_returns_all_tools(self):
        """Should return all tools."""
        tools = get_all_tools()

        assert len(tools) == len(TOOLS)

    def test_returns_tool_definitions(self):
        """Should return ToolDefinition instances."""
        tools = get_all_tools()

        assert all(isinstance(t, ToolDefinition) for t in tools)


class TestGetToolsByCategory:
    """Test get_tools_by_category function."""

    def test_filter_by_category(self):
        """Should filter tools by category."""
        file_tools = get_tools_by_category("file")

        assert len(file_tools) > 0
        assert all(t.category == "file" for t in file_tools)

    def test_empty_for_nonexistent_category(self):
        """Should return empty list for unknown category."""
        tools = get_tools_by_category("nonexistent_category")

        assert tools == []

    def test_case_sensitive(self):
        """Should be case sensitive."""
        tools_lower = get_tools_by_category("file")
        tools_upper = get_tools_by_category("FILE")

        # Assuming categories are lowercase, FILE should return empty
        assert len(tools_lower) != len(tools_upper) or tools_upper == []


class TestGetToolsByServer:
    """Test get_tools_by_server function."""

    def test_filter_by_server(self):
        """Should filter tools by server."""
        bridge_tools = get_tools_by_server("bridge")

        assert len(bridge_tools) > 0
        assert all(t.server == "bridge" for t in bridge_tools)

    def test_empty_for_nonexistent_server(self):
        """Should return empty list for unknown server."""
        tools = get_tools_by_server("nonexistent_server")

        assert tools == []

    def test_all_known_servers(self):
        """Should find tools for all known servers."""
        servers = get_servers()

        for server in servers:
            tools = get_tools_by_server(server)
            assert len(tools) > 0, f"No tools found for server: {server}"


class TestGetCoreTools:
    """Test get_core_tools function."""

    def test_returns_only_core_tools(self):
        """Should return only core tools."""
        core_tools = get_core_tools()

        assert all(t.is_core for t in core_tools)

    def test_subset_of_all_tools(self):
        """Core tools should be subset of all tools."""
        all_tools = get_all_tools()
        core_tools = get_core_tools()

        assert len(core_tools) <= len(all_tools)

        core_names = {t.name for t in core_tools}
        all_names = {t.name for t in all_tools}

        assert core_names.issubset(all_names)


class TestGetCategories:
    """Test get_categories function."""

    def test_returns_categories(self):
        """Should return list of categories."""
        categories = get_categories()

        assert len(categories) > 0
        assert all(isinstance(c, str) for c in categories)

    def test_categories_unique(self):
        """Should return unique categories."""
        categories = get_categories()

        assert len(categories) == len(set(categories))

    def test_includes_known_categories(self):
        """Should include known categories."""
        categories = get_categories()

        expected = ["file", "git", "search", "ai"]
        for cat in expected:
            assert cat in categories, f"Expected category '{cat}' not found"


class TestGetServers:
    """Test get_servers function."""

    def test_returns_servers(self):
        """Should return list of servers."""
        servers = get_servers()

        assert len(servers) > 0
        assert all(isinstance(s, str) for s in servers)

    def test_servers_unique(self):
        """Should return unique servers."""
        servers = get_servers()

        assert len(servers) == len(set(servers))

    def test_includes_known_servers(self):
        """Should include known servers."""
        servers = get_servers()

        expected = ["bridge", "doc", "comfy", "video", "chat"]
        for srv in expected:
            assert srv in servers, f"Expected server '{srv}' not found"


# =============================================================================
# Export Tests
# =============================================================================


class TestExportManifest:
    """Test export_manifest function."""

    def test_export_creates_file(self, tmp_path):
        """Should create JSON file."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        assert filepath.exists()

    def test_export_valid_json(self, tmp_path):
        """Should write valid JSON."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_export_includes_metadata(self, tmp_path):
        """Should include version and counts."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert "version" in data
        assert "tool_count" in data
        assert "categories" in data
        assert "servers" in data
        assert "tools" in data

    def test_export_tool_count_matches(self, tmp_path):
        """Tool count should match exported tools."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert data["tool_count"] == len(data["tools"])

    def test_export_tools_serialized(self, tmp_path):
        """Tools should be properly serialized."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        for tool_data in data["tools"]:
            assert "name" in tool_data
            assert "description" in tool_data
            assert "category" in tool_data
            assert "server" in tool_data

    def test_export_categories_match_tools(self, tmp_path):
        """Exported categories should match tools."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        tool_categories = {t["category"] for t in data["tools"]}
        exported_categories = set(data["categories"])

        assert tool_categories == exported_categories

    def test_export_servers_match_tools(self, tmp_path):
        """Exported servers should match tools."""
        filepath = tmp_path / "manifest.json"

        export_manifest(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        tool_servers = {t["server"] for t in data["tools"]}
        exported_servers = set(data["servers"])

        assert tool_servers == exported_servers


# =============================================================================
# Tool Integrity Tests
# =============================================================================


class TestToolIntegrity:
    """Test tool definitions for consistency and quality."""

    def test_descriptions_not_empty(self):
        """All descriptions should be non-empty."""
        for tool in TOOLS:
            assert len(tool.description.strip()) > 10, (
                f"Tool {tool.name} has very short description"
            )

    def test_descriptions_not_duplicated(self):
        """Descriptions should be unique."""
        descriptions = [t.description for t in TOOLS]
        unique_descriptions = set(descriptions)

        # Allow some duplication but warn if significant
        duplication_ratio = 1 - (len(unique_descriptions) / len(descriptions))
        assert duplication_ratio < 0.1, (
            f"Too many duplicated descriptions: {duplication_ratio:.0%}"
        )

    def test_categories_consistent(self):
        """Categories should be from known set."""
        known_categories = {
            "file",
            "git",
            "database",
            "search",
            "ai",
            "analysis",
            "project",
            "content",
            "system",
            "meta",
        }

        for tool in TOOLS:
            assert tool.category in known_categories, (
                f"Tool {tool.name} has unknown category: {tool.category}"
            )

    def test_core_tools_have_examples(self):
        """Core tools should have examples."""
        for tool in TOOLS:
            if tool.is_core:
                assert len(tool.examples) > 0, (
                    f"Core tool {tool.name} should have examples"
                )

    def test_embedding_text_reasonable_length(self):
        """Embedding text should be reasonable length."""
        for tool in TOOLS:
            text = tool.embedding_text()
            assert 50 < len(text) < 1000, (
                f"Tool {tool.name} embedding text length: {len(text)}"
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_tool_with_special_characters_in_name(self):
        """Should handle special characters in name."""
        tool = ToolDefinition(
            name="test:file-read_v2.0",
            description="Test",
            category="test",
            server="test",
        )

        assert tool.name == "test:file-read_v2.0"
        text = tool.embedding_text()
        assert "file-read_v2.0" in text

    def test_tool_with_unicode_description(self):
        """Should handle Unicode in description."""
        tool = ToolDefinition(
            name="test:unicode",
            description="Unicode test: \u4e2d\u6587 \u65e5\u672c\u8a9e",
            category="test",
            server="test",
        )

        text = tool.embedding_text()
        assert "\u4e2d\u6587" in text

    def test_tool_with_complex_parameters(self):
        """Should handle complex parameter schemas."""
        tool = ToolDefinition(
            name="test:complex",
            description="Complex tool",
            category="test",
            server="test",
            parameters={
                "simple": "str",
                "optional": "str?",
                "typed_list": "list[str]",
                "nested/path": "object",
            },
        )

        data = tool.to_dict()
        assert data["parameters"]["nested/path"] == "object"

    def test_tool_with_many_examples(self):
        """Should handle many examples."""
        examples = [f"example {i}" for i in range(20)]
        tool = ToolDefinition(
            name="test:many_examples",
            description="Tool with many examples",
            category="test",
            server="test",
            examples=examples,
        )

        text = tool.embedding_text()
        # Should include examples joined
        assert "example 0" in text
        assert "example 19" in text
