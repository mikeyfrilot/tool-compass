"""
Tool Compass - Tool Manifest
Contains all tool definitions for indexing.

Each tool has:
- name: Unique identifier (server:tool_name format)
- description: What the tool does (used for embedding)
- category: Functional category
- parameters: Input schema summary
- examples: Example use cases (improves embedding quality)
"""

from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class ToolDefinition:
    """A tool that can be discovered via compass."""

    name: str
    description: str
    category: str
    server: str
    parameters: Dict = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    is_core: bool = False

    def embedding_text(self) -> str:
        """
        Generate rich text for embedding.
        Combines description, category, and examples for better semantic matching.
        """
        parts = [
            f"Tool: {self.name}",
            f"Category: {self.category}",
            f"Description: {self.description}",
        ]

        if self.examples:
            parts.append(f"Use cases: {'; '.join(self.examples)}")

        if self.parameters:
            param_str = ", ".join(self.parameters.keys())
            parts.append(f"Parameters: {param_str}")

        return " | ".join(parts)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "server": self.server,
            "parameters": self.parameters,
            "examples": self.examples,
            "is_core": self.is_core,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS: List[ToolDefinition] = [
    # =========================================================================
    # BRIDGE SERVER - Content Management
    # =========================================================================
    ToolDefinition(
        name="bridge:search_bridge",
        description="Text search across Bridge content using full-text search",
        category="search",
        server="bridge",
        parameters={"query": "str", "project_id": "str?", "limit": "int?"},
        examples=["find documents about AI", "search notes for meeting"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:list_projects",
        description="List all Bridge projects with optional archive filter",
        category="project",
        server="bridge",
        parameters={"include_archived": "bool?"},
        examples=["show my projects", "list all workspaces"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:get_project_summary",
        description="Get project statistics and recent items overview",
        category="project",
        server="bridge",
        parameters={"project_id": "str"},
        examples=["project overview", "what's in this project"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:add_bridge_content",
        description="Add new content to a Bridge session for storage",
        category="content",
        server="bridge",
        parameters={
            "content": "str",
            "session_id": "str",
            "title": "str?",
            "content_type": "str?",
        },
        examples=["save this note", "store document", "add to session"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:get_content_full",
        description="Retrieve full content text by ID",
        category="content",
        server="bridge",
        parameters={"content_id": "str"},
        examples=["get document content", "read stored note"],
    ),
    ToolDefinition(
        name="bridge:update_bridge_content",
        description="Update existing Bridge content",
        category="content",
        server="bridge",
        parameters={"content_id": "str", "content": "str?", "title": "str?"},
        examples=["edit document", "update note"],
    ),
    ToolDefinition(
        name="bridge:delete_bridge_content",
        description="Delete Bridge content by ID",
        category="content",
        server="bridge",
        parameters={"content_id": "str"},
        examples=["remove document", "delete note"],
    ),
    ToolDefinition(
        name="bridge:semantic_search",
        description="AI-powered semantic search using FAISS vector similarity",
        category="search",
        server="bridge",
        parameters={"query": "str", "project_id": "str?", "limit": "int?"},
        examples=["find similar documents", "semantic lookup", "meaning-based search"],
    ),
    ToolDefinition(
        name="bridge:create_project",
        description="Create new Bridge project workspace",
        category="project",
        server="bridge",
        parameters={"name": "str", "description": "str?"},
        examples=["new project", "create workspace"],
    ),
    ToolDefinition(
        name="bridge:create_session",
        description="Create new session within a project",
        category="project",
        server="bridge",
        parameters={"project_id": "str", "name": "str?"},
        examples=["new session", "start conversation"],
    ),
    ToolDefinition(
        name="bridge:list_sessions",
        description="List all sessions in a project",
        category="project",
        server="bridge",
        parameters={"project_id": "str"},
        examples=["show sessions", "list conversations"],
    ),
    ToolDefinition(
        name="bridge:get_session",
        description="Get session details with all content",
        category="project",
        server="bridge",
        parameters={"session_id": "str"},
        examples=["session details", "get conversation"],
    ),
    ToolDefinition(
        name="bridge:archive_project",
        description="Archive a project (soft delete)",
        category="project",
        server="bridge",
        parameters={"project_id": "str"},
        examples=["archive workspace", "hide project"],
    ),
    ToolDefinition(
        name="bridge:restore_project",
        description="Restore an archived project",
        category="project",
        server="bridge",
        parameters={"project_id": "str"},
        examples=["unarchive project", "restore workspace"],
    ),
    ToolDefinition(
        name="bridge:prepare_bridge_context",
        description="Generate context summary for a project",
        category="content",
        server="bridge",
        parameters={"project_id": "str"},
        examples=["summarize project", "get context"],
    ),
    ToolDefinition(
        name="bridge:create_checkpoint",
        description="Create project state snapshot for backup",
        category="project",
        server="bridge",
        parameters={"project_id": "str", "name": "str?"},
        examples=["save state", "create backup"],
    ),
    ToolDefinition(
        name="bridge:reindex_embeddings",
        description="Rebuild FAISS search index for semantic search",
        category="system",
        server="bridge",
        parameters={"project_id": "str?"},
        examples=["rebuild index", "refresh embeddings"],
    ),
    ToolDefinition(
        name="bridge:generate_title",
        description="Generate title for content using Ollama AI",
        category="ai",
        server="bridge",
        parameters={"content": "str"},
        examples=["auto-title", "generate name"],
    ),
    # =========================================================================
    # BRIDGE SERVER - System
    # =========================================================================
    ToolDefinition(
        name="bridge:service_health",
        description="Check service health status and latency metrics",
        category="system",
        server="bridge",
        parameters={},
        examples=["health check", "is system running", "check status"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:system_status",
        description="Get detailed system diagnostics including CPU, memory, and GPU usage",
        category="system",
        server="bridge",
        parameters={},
        examples=["system info", "resource usage", "memory check"],
    ),
    # =========================================================================
    # BRIDGE SERVER - File Operations
    # =========================================================================
    ToolDefinition(
        name="bridge:read_file",
        description="Read file contents from disk",
        category="file",
        server="bridge",
        parameters={"filepath": "str"},
        examples=["read file", "get file contents", "open document"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:list_files",
        description="List files matching glob pattern in directory",
        category="file",
        server="bridge",
        parameters={"pattern": "str", "directory": "str?", "limit": "int?"},
        examples=["find files", "list *.py", "show directory contents"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:search_files",
        description="Search for text within files using grep-like functionality",
        category="file",
        server="bridge",
        parameters={"pattern": "str", "directory": "str?", "file_pattern": "str?"},
        examples=["search in files", "find text", "grep for pattern"],
    ),
    ToolDefinition(
        name="bridge:file_info",
        description="Get file metadata including size, dates, and permissions",
        category="file",
        server="bridge",
        parameters={"filepath": "str"},
        examples=["file details", "when was file modified", "file size"],
    ),
    ToolDefinition(
        name="bridge:write_file",
        description="Write content to a file, creating or overwriting",
        category="file",
        server="bridge",
        parameters={"filepath": "str", "content": "str"},
        examples=["save file", "write to disk", "create file"],
    ),
    ToolDefinition(
        name="bridge:append_file",
        description="Append content to end of existing file",
        category="file",
        server="bridge",
        parameters={"filepath": "str", "content": "str"},
        examples=["add to file", "append text", "extend file"],
    ),
    ToolDefinition(
        name="bridge:create_directory",
        description="Create directory with parent directories as needed",
        category="file",
        server="bridge",
        parameters={"path": "str"},
        examples=["make folder", "create directory", "mkdir"],
    ),
    # =========================================================================
    # BRIDGE SERVER - Git Operations
    # =========================================================================
    ToolDefinition(
        name="bridge:git_status",
        description="Get git repository status showing changed files",
        category="git",
        server="bridge",
        parameters={"repo": "str?"},
        examples=["git status", "what files changed", "repo state"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:git_log",
        description="Get recent git commit history",
        category="git",
        server="bridge",
        parameters={"repo": "str?", "limit": "int?"},
        examples=["commit history", "recent changes", "git log"],
    ),
    ToolDefinition(
        name="bridge:git_diff",
        description="Show git diff for changed files",
        category="git",
        server="bridge",
        parameters={"repo": "str?", "file": "str?"},
        examples=["show changes", "what was modified", "git diff"],
    ),
    ToolDefinition(
        name="bridge:git_add",
        description="Stage files for git commit",
        category="git",
        server="bridge",
        parameters={"repo": "str?", "files": "list[str]?"},
        examples=["stage files", "git add", "prepare commit"],
    ),
    ToolDefinition(
        name="bridge:git_commit",
        description="Commit staged changes with message",
        category="git",
        server="bridge",
        parameters={"repo": "str?", "message": "str"},
        examples=["commit changes", "save to git", "git commit"],
    ),
    ToolDefinition(
        name="bridge:git_branch",
        description="List or manage git branches",
        category="git",
        server="bridge",
        parameters={"repo": "str?", "action": "str?", "name": "str?"},
        examples=["list branches", "create branch", "switch branch"],
    ),
    ToolDefinition(
        name="bridge:git_init",
        description="Initialize new git repository",
        category="git",
        server="bridge",
        parameters={"path": "str"},
        examples=["init repo", "start git", "new repository"],
    ),
    # =========================================================================
    # BRIDGE SERVER - Database Operations
    # =========================================================================
    ToolDefinition(
        name="bridge:db_inspect_schema",
        description="Get database schema including tables, columns, and types",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["show tables", "database structure", "schema info"],
    ),
    ToolDefinition(
        name="bridge:db_get_stats",
        description="Get database statistics and metrics",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["database stats", "table sizes", "row counts"],
    ),
    ToolDefinition(
        name="bridge:db_analyze_indexes",
        description="Analyze database index efficiency and usage",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["check indexes", "index performance", "optimize suggestions"],
    ),
    ToolDefinition(
        name="bridge:db_execute_sql",
        description="Execute SQL query against database",
        category="database",
        server="bridge",
        parameters={"database": "str?", "query": "str"},
        examples=["run SQL", "query database", "SELECT data"],
    ),
    ToolDefinition(
        name="bridge:db_explain_query",
        description="Analyze SQL query execution plan",
        category="database",
        server="bridge",
        parameters={"database": "str?", "query": "str"},
        examples=["explain query", "query plan", "optimize SQL"],
    ),
    ToolDefinition(
        name="bridge:db_benchmark_query",
        description="Benchmark query performance with timing",
        category="database",
        server="bridge",
        parameters={"database": "str?", "query": "str", "iterations": "int?"},
        examples=["test query speed", "benchmark SQL", "performance test"],
    ),
    ToolDefinition(
        name="bridge:db_vacuum",
        description="Run VACUUM to rebuild and compact database",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["compact database", "vacuum", "reclaim space"],
    ),
    ToolDefinition(
        name="bridge:db_reindex",
        description="Rebuild all database indexes",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["rebuild indexes", "reindex", "fix indexes"],
    ),
    ToolDefinition(
        name="bridge:db_optimize",
        description="Run lightweight database optimization",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["optimize database", "tune performance"],
    ),
    ToolDefinition(
        name="bridge:db_integrity_check",
        description="Check database integrity for corruption",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["check integrity", "verify database", "find corruption"],
    ),
    ToolDefinition(
        name="bridge:db_analyze",
        description="Run ANALYZE to update query planner statistics",
        category="database",
        server="bridge",
        parameters={"database": "str?"},
        examples=["update statistics", "analyze tables"],
    ),
    # =========================================================================
    # BRIDGE SERVER - Meta Tools
    # =========================================================================
    ToolDefinition(
        name="bridge:discover_tools",
        description="Find relevant tools using semantic search by task description",
        category="meta",
        server="bridge",
        parameters={"query": "str", "category": "str?", "top_k": "int?"},
        examples=["find tools", "what tool for X", "discover capabilities"],
        is_core=True,
    ),
    ToolDefinition(
        name="bridge:list_all_tools",
        description="List all available tools by category",
        category="meta",
        server="bridge",
        parameters={"category": "str?"},
        examples=["list tools", "available commands", "tool catalog"],
        is_core=True,
    ),
    # =========================================================================
    # DOC SERVER - Code Analysis
    # =========================================================================
    ToolDefinition(
        name="doc:scan",
        description="Scan file or directory for code health issues and antipatterns",
        category="analysis",
        server="doc",
        parameters={"target": "str", "pattern": "str?", "min_severity": "str?"},
        examples=["check code quality", "find bugs", "scan for issues"],
    ),
    ToolDefinition(
        name="doc:analyze",
        description="Deep analysis of file with detailed pattern explanations",
        category="analysis",
        server="doc",
        parameters={"filepath": "str", "pattern_id": "str?"},
        examples=["explain issue", "deep dive", "understand pattern"],
    ),
    ToolDefinition(
        name="doc:health_score",
        description="Get health score grade (A+ to F) for file or project",
        category="analysis",
        server="doc",
        parameters={"target": "str", "pattern": "str?"},
        examples=["code quality score", "project health", "grade code"],
    ),
    ToolDefinition(
        name="doc:report",
        description="Generate comprehensive code health report",
        category="analysis",
        server="doc",
        parameters={"target": "str", "pattern": "str?", "format": "str?"},
        examples=["generate report", "full analysis", "code review"],
    ),
    ToolDefinition(
        name="doc:fix",
        description="Apply auto-fixes for detected code issues",
        category="analysis",
        server="doc",
        parameters={"filepath": "str", "pattern_id": "str?", "dry_run": "bool?"},
        examples=["fix issues", "auto-repair", "apply fixes"],
    ),
    ToolDefinition(
        name="doc:patterns",
        description="List available detection patterns by category",
        category="analysis",
        server="doc",
        parameters={"category": "str?", "severity": "str?"},
        examples=["list patterns", "what can be detected", "pattern info"],
    ),
    ToolDefinition(
        name="doc:add_pattern",
        description="Add new detection pattern to scanner library",
        category="analysis",
        server="doc",
        parameters={
            "pattern_id": "str",
            "name": "str",
            "description": "str",
            "detection_patterns": "str",
        },
        examples=["create pattern", "add detection rule"],
    ),
    ToolDefinition(
        name="doc:categories",
        description="List valid pattern categories",
        category="analysis",
        server="doc",
        parameters={},
        examples=["pattern categories", "issue types"],
    ),
    # =========================================================================
    # COMFY SERVER - Image Generation
    # =========================================================================
    ToolDefinition(
        name="comfy:comfy_start",
        description="Start ComfyUI server for image generation",
        category="ai",
        server="comfy",
        parameters={"wait": "bool?", "timeout": "int?"},
        examples=["start image generation", "launch ComfyUI"],
    ),
    ToolDefinition(
        name="comfy:comfy_stop",
        description="Stop ComfyUI server to free GPU resources",
        category="ai",
        server="comfy",
        parameters={},
        examples=["stop image generation", "free GPU"],
    ),
    ToolDefinition(
        name="comfy:comfy_status",
        description="Check ComfyUI status and GPU/VRAM info",
        category="ai",
        server="comfy",
        parameters={},
        examples=["ComfyUI status", "GPU usage", "is ComfyUI running"],
    ),
    ToolDefinition(
        name="comfy:comfy_generate",
        description="Generate image from text prompt using AI",
        category="ai",
        server="comfy",
        parameters={
            "prompt": "str",
            "negative": "str?",
            "preset": "str?",
            "width": "int?",
            "height": "int?",
        },
        examples=["generate image", "create artwork", "text to image", "AI art"],
    ),
    ToolDefinition(
        name="comfy:comfy_models",
        description="List available image generation models and LoRAs",
        category="ai",
        server="comfy",
        parameters={},
        examples=["list models", "available checkpoints"],
    ),
    ToolDefinition(
        name="comfy:comfy_presets",
        description="Get available generation presets with settings",
        category="ai",
        server="comfy",
        parameters={},
        examples=["generation presets", "quality settings"],
    ),
    ToolDefinition(
        name="comfy:comfy_queue",
        description="Get current image generation queue status",
        category="ai",
        server="comfy",
        parameters={},
        examples=["generation queue", "pending jobs"],
    ),
    ToolDefinition(
        name="comfy:comfy_history",
        description="Get image generation history and results",
        category="ai",
        server="comfy",
        parameters={"prompt_id": "str?", "limit": "int?"},
        examples=["generation history", "past images"],
    ),
    ToolDefinition(
        name="comfy:comfy_cancel",
        description="Cancel current image generation",
        category="ai",
        server="comfy",
        parameters={},
        examples=["cancel generation", "stop image"],
    ),
    # =========================================================================
    # VIDEO SERVER - Video Generation
    # =========================================================================
    ToolDefinition(
        name="video:video_status",
        description="Check video generation service status and GPU info",
        category="ai",
        server="video",
        parameters={},
        examples=["video service status", "is video ready"],
    ),
    ToolDefinition(
        name="video:video_models",
        description="List available video generation models",
        category="ai",
        server="video",
        parameters={},
        examples=["video models", "available video AI"],
    ),
    ToolDefinition(
        name="video:video_presets",
        description="Get video generation presets with frame/fps settings",
        category="ai",
        server="video",
        parameters={},
        examples=["video presets", "quality options"],
    ),
    ToolDefinition(
        name="video:video_generate",
        description="Generate video from text prompt using AI",
        category="ai",
        server="video",
        parameters={
            "prompt": "str",
            "negative": "str?",
            "model": "str?",
            "preset": "str?",
        },
        examples=["generate video", "create animation", "text to video", "AI video"],
    ),
    ToolDefinition(
        name="video:video_queue",
        description="Get current video generation queue",
        category="ai",
        server="video",
        parameters={},
        examples=["video queue", "pending videos"],
    ),
    ToolDefinition(
        name="video:video_history",
        description="Get video generation history",
        category="ai",
        server="video",
        parameters={"job_id": "str?", "limit": "int?"},
        examples=["video history", "past generations"],
    ),
    ToolDefinition(
        name="video:video_cancel",
        description="Cancel current video generation",
        category="ai",
        server="video",
        parameters={},
        examples=["cancel video", "stop generation"],
    ),
    # =========================================================================
    # CHAT SERVER - AI Orchestration
    # =========================================================================
    ToolDefinition(
        name="chat:create_conversation",
        description="Create new AI conversation with Nexus orchestrator",
        category="ai",
        server="chat",
        parameters={"title": "str?", "model": "str?"},
        examples=["new conversation", "start chat"],
    ),
    ToolDefinition(
        name="chat:send_message",
        description="Send message with intelligent AI orchestration",
        category="ai",
        server="chat",
        parameters={
            "conversation_id": "str",
            "message": "str",
            "use_orchestration": "bool?",
        },
        examples=["send message", "chat with AI"],
    ),
    ToolDefinition(
        name="chat:thinking_cap",
        description="Direct TinyTool access for simple tasks like calculations",
        category="ai",
        server="chat",
        parameters={"task": "str", "task_type": "str?"},
        examples=["quick calculation", "simple task", "extract data"],
    ),
    ToolDefinition(
        name="chat:get_conversation",
        description="Get conversation with all messages",
        category="ai",
        server="chat",
        parameters={"conversation_id": "str"},
        examples=["get chat", "conversation history"],
    ),
    ToolDefinition(
        name="chat:list_conversations",
        description="List all AI conversations",
        category="ai",
        server="chat",
        parameters={"include_archived": "bool?", "limit": "int?"},
        examples=["list chats", "all conversations"],
    ),
    ToolDefinition(
        name="chat:search_conversations",
        description="Search conversations using full-text search",
        category="ai",
        server="chat",
        parameters={"query": "str", "limit": "int?"},
        examples=["search chats", "find conversation"],
    ),
    ToolDefinition(
        name="chat:get_orchestration_stats",
        description="Get Three-Tier Intelligence Architecture statistics",
        category="ai",
        server="chat",
        parameters={},
        examples=["AI stats", "orchestration metrics"],
    ),
]


def get_all_tools() -> List[ToolDefinition]:
    """Get all tool definitions."""
    return TOOLS


def get_tools_by_category(category: str) -> List[ToolDefinition]:
    """Get tools filtered by category."""
    return [t for t in TOOLS if t.category == category]


def get_tools_by_server(server: str) -> List[ToolDefinition]:
    """Get tools filtered by server."""
    return [t for t in TOOLS if t.server == server]


def get_core_tools() -> List[ToolDefinition]:
    """Get only core tools."""
    return [t for t in TOOLS if t.is_core]


def get_categories() -> List[str]:
    """Get unique categories."""
    return list(set(t.category for t in TOOLS))


def get_servers() -> List[str]:
    """Get unique servers."""
    return list(set(t.server for t in TOOLS))


def export_manifest(filepath: str):
    """Export manifest to JSON file."""
    data = {
        "version": "1.0",
        "tool_count": len(TOOLS),
        "categories": get_categories(),
        "servers": get_servers(),
        "tools": [t.to_dict() for t in TOOLS],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Total tools: {len(TOOLS)}")
    print(f"Categories: {get_categories()}")
    print(f"Servers: {get_servers()}")
    print(f"Core tools: {len(get_core_tools())}")

    print("\nSample embedding texts:")
    for tool in TOOLS[:3]:
        print(f"\n{tool.name}:")
        print(f"  {tool.embedding_text()[:100]}...")
