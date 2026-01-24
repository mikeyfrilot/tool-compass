"""
Tool Compass - Indexer Module
Builds and manages the HNSW index for semantic tool discovery.
"""

import hnswlib
import sqlite3
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import time

from embedder import Embedder, EMBEDDING_DIM
from tool_manifest import ToolDefinition, get_all_tools

logger = logging.getLogger(__name__)

# Configuration
DB_DIR = Path(__file__).parent / "db"
HNSW_INDEX_PATH = DB_DIR / "compass.hnsw"
SQLITE_DB_PATH = DB_DIR / "tools.db"

# HNSW Parameters (tuned for ~100-1000 tools)
HNSW_M = 16  # Number of connections per element
HNSW_EF_CONSTRUCTION = 200  # Size of dynamic candidate list during construction
HNSW_EF_SEARCH = 50  # Size of dynamic candidate list during search


@dataclass
class SearchResult:
    """Result from compass search."""

    tool: ToolDefinition
    score: float  # Cosine similarity (higher = better)
    rank: int


class CompassIndex:
    """
    HNSW-based index for semantic tool discovery.

    Architecture:
    - HNSW index stores tool embeddings for O(log n) search
    - SQLite stores tool metadata for retrieval
    - Embedder generates vectors via Ollama
    """

    def __init__(
        self,
        index_path: Path = HNSW_INDEX_PATH,
        db_path: Path = SQLITE_DB_PATH,
        embedder: Optional[Embedder] = None,
    ):
        self.index_path = Path(index_path)
        self.db_path = Path(db_path)
        self.embedder = embedder or Embedder()

        self.index: Optional[hnswlib.Index] = None
        self.db: Optional[sqlite3.Connection] = None
        self._id_to_name: Dict[int, str] = {}

        # Ensure db directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize SQLite database for tool metadata."""
        self.db = sqlite3.connect(str(self.db_path))
        self.db.row_factory = sqlite3.Row

        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                server TEXT NOT NULL,
                parameters TEXT,  -- JSON
                examples TEXT,    -- JSON
                is_core INTEGER DEFAULT 0,
                embedding_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
            CREATE INDEX IF NOT EXISTS idx_tools_server ON tools(server);
            CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
            
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.db.commit()

    def _load_id_mapping(self):
        """Load ID to name mapping from database."""
        cursor = self.db.execute("SELECT id, name FROM tools")
        self._id_to_name = {row["id"]: row["name"] for row in cursor.fetchall()}

    async def build_index(self, tools: Optional[List[ToolDefinition]] = None):
        """
        Build HNSW index from tool definitions.

        Args:
            tools: List of tools to index. Uses manifest if not provided.
        """
        if tools is None:
            tools = get_all_tools()

        logger.info(f"Building index for {len(tools)} tools...")
        start_time = time.time()

        # Initialize database
        self._init_db()

        # Clear existing data
        self.db.execute("DELETE FROM tools")
        self.db.commit()

        # Insert tools and collect texts for embedding
        embedding_texts = []
        tool_ids = []

        for i, tool in enumerate(tools):
            embedding_text = tool.embedding_text()
            embedding_texts.append(embedding_text)

            # Insert into SQLite
            cursor = self.db.execute(
                """
                INSERT INTO tools (name, description, category, server, parameters, examples, is_core, embedding_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tool.name,
                    tool.description,
                    tool.category,
                    tool.server,
                    json.dumps(tool.parameters),
                    json.dumps(tool.examples),
                    1 if tool.is_core else 0,
                    embedding_text,
                ),
            )
            tool_ids.append(cursor.lastrowid)

        self.db.commit()
        logger.info(f"Inserted {len(tools)} tools into database")

        # Generate embeddings
        logger.info("Generating embeddings via Ollama...")
        embed_start = time.time()
        embeddings = await self.embedder.embed_batch(embedding_texts)
        embed_time = time.time() - embed_start
        logger.info(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")

        # Build HNSW index
        logger.info("Building HNSW index...")
        self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self.index.init_index(
            max_elements=max(len(tools) * 2, 1000),  # Room to grow
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )

        # Add vectors with tool IDs
        self.index.add_items(embeddings, tool_ids)
        self.index.set_ef(HNSW_EF_SEARCH)

        # Save index
        self.index.save_index(str(self.index_path))

        # Update metadata
        self.db.execute(
            """
            INSERT OR REPLACE INTO index_meta (key, value) VALUES
            ('tool_count', ?),
            ('embedding_dim', ?),
            ('hnsw_m', ?),
            ('hnsw_ef_construction', ?),
            ('build_time', ?)
        """,
            (
                str(len(tools)),
                str(EMBEDDING_DIM),
                str(HNSW_M),
                str(HNSW_EF_CONSTRUCTION),
                str(time.time() - start_time),
            ),
        )
        self.db.commit()

        # Load ID mapping
        self._load_id_mapping()

        total_time = time.time() - start_time
        logger.info(f"Index built in {total_time:.2f}s")

        return {
            "tools_indexed": len(tools),
            "embedding_time": embed_time,
            "total_time": total_time,
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
        }

    def load_index(self) -> bool:
        """
        Load existing index from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.index_path.exists() or not self.db_path.exists():
            logger.warning("Index files not found")
            return False

        try:
            # Load database
            self._init_db()
            self._load_id_mapping()

            # Load HNSW index
            self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
            self.index.load_index(str(self.index_path))
            self.index.set_ef(HNSW_EF_SEARCH)

            logger.info(f"Loaded index with {len(self._id_to_name)} tools")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def _get_tool_by_id(self, tool_id: int) -> Optional[ToolDefinition]:
        """Retrieve tool definition by ID."""
        cursor = self.db.execute(
            """
            SELECT name, description, category, server, parameters, examples, is_core
            FROM tools WHERE id = ?
        """,
            (tool_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return ToolDefinition(
            name=row["name"],
            description=row["description"],
            category=row["category"],
            server=row["server"],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            examples=json.loads(row["examples"]) if row["examples"] else [],
            is_core=bool(row["is_core"]),
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        server_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for tools matching the query intent.

        Args:
            query: Natural language description of task/intent
            top_k: Number of results to return
            category_filter: Optional category to filter by
            server_filter: Optional server to filter by

        Returns:
            List of SearchResult ordered by relevance
        """
        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() or build_index() first."
            )

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Search HNSW (get more than needed for filtering)
        search_k = min(top_k * 3, self.index.get_current_count())
        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1), k=search_k
        )

        # Convert distances to similarities (hnswlib returns 1 - cosine for cosine space)
        similarities = 1 - distances[0]

        results = []
        for label, similarity in zip(labels[0], similarities):
            tool = self._get_tool_by_id(int(label))
            if tool is None:
                continue

            # Apply filters
            if category_filter and tool.category != category_filter:
                continue
            if server_filter and tool.server != server_filter:
                continue

            results.append(
                SearchResult(tool=tool, score=float(similarity), rank=len(results) + 1)
            )

            if len(results) >= top_k:
                break

        return results

    def search_sync(self, query: str, top_k: int = 5, **kwargs) -> List[SearchResult]:
        """Synchronous search wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.search(query, top_k, **kwargs))
        finally:
            loop.close()

    def get_stats(self) -> Dict:
        """Get index statistics."""
        if self.db is None:
            self._init_db()

        stats = {}

        # Tool counts
        cursor = self.db.execute("SELECT COUNT(*) as count FROM tools")
        stats["total_tools"] = cursor.fetchone()["count"]

        cursor = self.db.execute(
            "SELECT COUNT(*) as count FROM tools WHERE is_core = 1"
        )
        stats["core_tools"] = cursor.fetchone()["count"]

        # Category breakdown
        cursor = self.db.execute(
            "SELECT category, COUNT(*) as count FROM tools GROUP BY category"
        )
        stats["by_category"] = {
            row["category"]: row["count"] for row in cursor.fetchall()
        }

        # Server breakdown
        cursor = self.db.execute(
            "SELECT server, COUNT(*) as count FROM tools GROUP BY server"
        )
        stats["by_server"] = {row["server"]: row["count"] for row in cursor.fetchall()}

        # Index metadata
        cursor = self.db.execute("SELECT key, value FROM index_meta")
        stats["index_meta"] = {row["key"]: row["value"] for row in cursor.fetchall()}

        # HNSW stats
        if self.index:
            stats["hnsw"] = {
                "current_count": self.index.get_current_count(),
                "max_elements": self.index.get_max_elements(),
                "ef": self.index.ef,
            }

        return stats

    async def add_single_tool(self, tool: ToolDefinition) -> bool:
        """
        Add a single tool to the index without full rebuild.
        HNSW supports dynamic element addition.

        Args:
            tool: The tool definition to add

        Returns:
            True if added successfully, False otherwise
        """
        if self.index is None or self.db is None:
            logger.error("Index not initialized. Call load_index() first.")
            return False

        try:
            # Check if tool already exists
            cursor = self.db.execute(
                "SELECT id FROM tools WHERE name = ?", (tool.name,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing tool
                tool_id = existing["id"]
                embedding_text = tool.embedding_text()

                self.db.execute(
                    """
                    UPDATE tools SET
                        description = ?, category = ?, server = ?,
                        parameters = ?, examples = ?, is_core = ?,
                        embedding_text = ?
                    WHERE id = ?
                """,
                    (
                        tool.description,
                        tool.category,
                        tool.server,
                        json.dumps(tool.parameters),
                        json.dumps(tool.examples),
                        1 if tool.is_core else 0,
                        embedding_text,
                        tool_id,
                    ),
                )
            else:
                # Insert new tool
                embedding_text = tool.embedding_text()

                cursor = self.db.execute(
                    """
                    INSERT INTO tools (name, description, category, server, parameters, examples, is_core, embedding_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        tool.name,
                        tool.description,
                        tool.category,
                        tool.server,
                        json.dumps(tool.parameters),
                        json.dumps(tool.examples),
                        1 if tool.is_core else 0,
                        embedding_text,
                    ),
                )
                tool_id = cursor.lastrowid

            self.db.commit()

            # Generate embedding
            embedding_text = tool.embedding_text()
            embedding = await self.embedder.embed(embedding_text)

            # Check if we need to resize the index
            if self.index.get_current_count() >= self.index.get_max_elements() - 1:
                # Need to resize - HNSW doesn't support dynamic resize, so we extend
                new_max = self.index.get_max_elements() * 2
                self.index.resize_index(new_max)
                logger.info(f"Resized HNSW index to {new_max} elements")

            # Add to HNSW index
            self.index.add_items(embedding.reshape(1, -1), [tool_id])

            # Update ID mapping
            self._id_to_name[tool_id] = tool.name

            # Save index
            self.index.save_index(str(self.index_path))

            logger.info(f"Added tool to index: {tool.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add tool {tool.name}: {e}")
            return False

    async def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the database.
        Note: HNSW doesn't support element removal, so the vector remains
        but won't be returned in searches (no matching DB entry).

        For full cleanup, rebuild the index with build_index().

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if removed from DB, False otherwise
        """
        if self.db is None:
            logger.error("Database not initialized")
            return False

        try:
            cursor = self.db.execute(
                "SELECT id FROM tools WHERE name = ?", (tool_name,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Tool not found: {tool_name}")
                return False

            tool_id = row["id"]

            # Remove from database
            self.db.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
            self.db.commit()

            # Remove from ID mapping
            self._id_to_name.pop(tool_id, None)

            logger.info(f"Removed tool from index: {tool_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove tool {tool_name}: {e}")
            return False

    async def close(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
            self.db = None
        await self.embedder.close()


async def build_compass_index():
    """Build the compass index from scratch."""
    logging.basicConfig(level=logging.INFO)

    index = CompassIndex()

    # Check Ollama
    print("Checking Ollama availability...")
    if not await index.embedder.health_check():
        print("ERROR: Ollama not available or nomic-embed-text not loaded")
        print("Run: ollama pull nomic-embed-text")
        return

    # Build index
    print("\nBuilding Tool Compass index...")
    result = await index.build_index()

    print("\n✓ Index built successfully!")
    print(f"  Tools indexed: {result['tools_indexed']}")
    print(f"  Embedding time: {result['embedding_time']:.2f}s")
    print(f"  Total time: {result['total_time']:.2f}s")
    print(f"  Index path: {result['index_path']}")
    print(f"  Database path: {result['db_path']}")

    # Test search
    print("\n--- Testing search ---")
    test_queries = [
        "read a file from disk",
        "generate an image with AI",
        "search for text in documents",
        "check git status",
        "analyze code quality",
    ]

    for query in test_queries:
        results = await index.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  {r.rank}. {r.tool.name} (score: {r.score:.3f})")

    await index.close()


if __name__ == "__main__":
    asyncio.run(build_compass_index())
