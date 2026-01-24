"""
Tool Compass - Chain Indexer
Makes tool chains (workflows) searchable via semantic search.
"""

import json
import sqlite3
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from embedder import Embedder
    from analytics import CompassAnalytics

try:
    import hnswlib
except ImportError:
    hnswlib = None

logger = logging.getLogger(__name__)

# Paths
DB_DIR = Path(__file__).parent / "db"
CHAIN_INDEX_PATH = DB_DIR / "chains.hnsw"
ANALYTICS_DB_PATH = DB_DIR / "compass_analytics.db"

# HNSW settings for chains (smaller than main index)
CHAIN_HNSW_M = 12
CHAIN_HNSW_EF_CONSTRUCTION = 100
CHAIN_HNSW_EF_SEARCH = 30
EMBEDDING_DIM = 768


@dataclass
class ToolChain:
    """A sequence of tools that form a workflow."""

    id: int
    name: str
    tools: List[str]  # ["bridge:read_file", "bridge:write_file"]
    description: str
    use_count: int
    is_auto_detected: bool
    embedding: Optional[np.ndarray] = None
    embedding_text: Optional[str] = None


@dataclass
class ChainSearchResult:
    """Result from chain search."""

    chain: ToolChain
    score: float  # Similarity score 0-1


class ChainIndexer:
    """
    Manages tool chains as first-class searchable entities.

    Features:
    - Auto-detect chains from usage patterns
    - Index chains in HNSW for semantic search
    - Cache top 5 most-used chains
    """

    def __init__(
        self,
        embedder: "Embedder",
        analytics: Optional["CompassAnalytics"] = None,
        top_chains_cache_size: int = 5,
    ):
        self.embedder = embedder
        self.analytics = analytics
        self.top_chains_cache_size = top_chains_cache_size

        self.index: Optional["hnswlib.Index"] = None
        self._chain_cache: List[ToolChain] = []
        self._id_to_chain: Dict[int, ToolChain] = {}
        self._db: Optional[sqlite3.Connection] = None

        # Ensure db directory exists
        DB_DIR.mkdir(parents=True, exist_ok=True)

    def _get_db(self) -> sqlite3.Connection:
        """Get database connection (uses analytics DB)."""
        if self._db is None:
            self._db = sqlite3.connect(str(ANALYTICS_DB_PATH))
            self._db.row_factory = sqlite3.Row
        return self._db

    def create_chain_embedding_text(self, chain: ToolChain) -> str:
        """
        Generate rich text for chain embedding.
        Combines workflow name, steps, and tool names for better semantic matching.
        """
        # Extract simple tool names
        tool_names = [t.split(":")[-1].replace("_", " ") for t in chain.tools]

        parts = [
            f"Workflow: {chain.name.replace('_', ' ')}",
            f"Steps: {', '.join(tool_names)}",
            f"Description: {chain.description}",
            f"Tools: {', '.join(chain.tools)}",
            f"Use cases: {' then '.join(tool_names)}",
        ]

        return " | ".join(parts)

    async def load_chains_from_db(self) -> List[ToolChain]:
        """Load all chains from database."""
        db = self._get_db()

        cursor = db.execute("""
            SELECT id, chain_name, chain_tools, description, use_count, is_auto_detected, embedding_text
            FROM tool_chains
            ORDER BY use_count DESC
        """)

        chains = []
        for row in cursor.fetchall():
            chain = ToolChain(
                id=row["id"],
                name=row["chain_name"],
                tools=json.loads(row["chain_tools"]),
                description=row["description"] or "",
                use_count=row["use_count"],
                is_auto_detected=bool(row["is_auto_detected"]),
                embedding_text=row["embedding_text"],
            )
            chains.append(chain)

        return chains

    async def build_chain_index(self, chains: Optional[List[ToolChain]] = None):
        """
        Build HNSW index for chains.
        If chains not provided, loads from database.
        """
        if hnswlib is None:
            logger.error("hnswlib not installed - chain indexing disabled")
            return

        if chains is None:
            chains = await self.load_chains_from_db()

        if not chains:
            logger.info("No chains to index")
            return

        logger.info(f"Building chain index with {len(chains)} chains...")

        # Generate embeddings for chains without them
        for chain in chains:
            if chain.embedding is None:
                embedding_text = (
                    chain.embedding_text or self.create_chain_embedding_text(chain)
                )
                embedding = await self.embedder.embed(embedding_text)
                chain.embedding = embedding
                chain.embedding_text = embedding_text

        # Initialize HNSW index
        self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
        self.index.init_index(
            max_elements=max(len(chains) * 2, 100),
            M=CHAIN_HNSW_M,
            ef_construction=CHAIN_HNSW_EF_CONSTRUCTION,
        )
        self.index.set_ef(CHAIN_HNSW_EF_SEARCH)

        # Add chains to index
        self._id_to_chain = {}
        embeddings = []
        ids = []

        for chain in chains:
            self._id_to_chain[chain.id] = chain
            embeddings.append(chain.embedding)
            ids.append(chain.id)

        if embeddings:
            embeddings_array = np.vstack(embeddings).astype(np.float32)
            self.index.add_items(embeddings_array, ids)

        # Save index
        self.index.save_index(str(CHAIN_INDEX_PATH))
        logger.info(f"Chain index saved to {CHAIN_INDEX_PATH}")

        # Update cache
        await self.refresh_chain_cache()

    async def load_chain_index(self) -> bool:
        """Load existing chain index from disk."""
        if hnswlib is None:
            return False

        if not CHAIN_INDEX_PATH.exists():
            return False

        try:
            chains = await self.load_chains_from_db()
            if not chains:
                return False

            self.index = hnswlib.Index(space="cosine", dim=EMBEDDING_DIM)
            self.index.load_index(str(CHAIN_INDEX_PATH))
            self.index.set_ef(CHAIN_HNSW_EF_SEARCH)

            # Build ID mapping
            self._id_to_chain = {chain.id: chain for chain in chains}

            await self.refresh_chain_cache()
            logger.info(f"Loaded chain index with {len(chains)} chains")
            return True
        except Exception as e:
            logger.error(f"Failed to load chain index: {e}")
            return False

    async def search_chains(
        self, query: str, top_k: int = 3, min_confidence: float = 0.3
    ) -> List[ChainSearchResult]:
        """
        Search for relevant tool chains.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            min_confidence: Minimum similarity threshold

        Returns:
            List of ChainSearchResult sorted by score
        """
        if self.index is None or self.index.get_current_count() == 0:
            return []

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Search HNSW
        search_k = min(top_k * 2, self.index.get_current_count())
        labels, distances = self.index.knn_query(
            query_embedding.reshape(1, -1), k=search_k
        )

        # Convert to results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            # Convert cosine distance to similarity
            similarity = 1 - distance

            if similarity < min_confidence:
                continue

            chain = self._id_to_chain.get(label)
            if chain:
                results.append(ChainSearchResult(chain=chain, score=similarity))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def refresh_chain_cache(self):
        """Update cache with top N most-used chains."""
        chains = await self.load_chains_from_db()

        # Sort by use count and take top N
        chains.sort(key=lambda c: c.use_count, reverse=True)
        self._chain_cache = chains[: self.top_chains_cache_size]

        logger.debug(f"Refreshed chain cache with {len(self._chain_cache)} chains")

    async def add_chain(
        self,
        name: str,
        tools: List[str],
        description: Optional[str] = None,
        is_auto_detected: bool = False,
    ) -> ToolChain:
        """
        Add a new chain to the index.

        Args:
            name: Unique chain name
            tools: List of tool names in order
            description: Human-readable description
            is_auto_detected: Whether this was auto-detected from patterns

        Returns:
            The created ToolChain
        """
        db = self._get_db()

        # Generate description if not provided
        if not description:
            tool_names = [t.split(":")[-1].replace("_", " ") for t in tools]
            description = f"Workflow: {' → '.join(tool_names)}"

        # Generate embedding text
        embedding_text = f"Workflow: {name.replace('_', ' ')} | Steps: {description} | Tools: {', '.join(tools)}"

        # Generate embedding (use embed() for documents, embed_query() for searches)
        embedding = await self.embedder.embed(embedding_text)

        # Insert into DB
        cursor = db.execute(
            """
            INSERT INTO tool_chains (chain_name, chain_tools, description, embedding_text, is_auto_detected)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chain_name) DO UPDATE SET
                chain_tools = excluded.chain_tools,
                description = excluded.description,
                embedding_text = excluded.embedding_text
        """,
            (
                name,
                json.dumps(tools),
                description,
                embedding_text,
                1 if is_auto_detected else 0,
            ),
        )

        chain_id = cursor.lastrowid
        db.commit()

        chain = ToolChain(
            id=chain_id,
            name=name,
            tools=tools,
            description=description,
            use_count=0,
            is_auto_detected=is_auto_detected,
            embedding=embedding,
            embedding_text=embedding_text,
        )

        # Add to index if it exists
        if self.index:
            self._id_to_chain[chain_id] = chain
            self.index.add_items(
                embedding.reshape(1, -1).astype(np.float32), [chain_id]
            )
            self.index.save_index(str(CHAIN_INDEX_PATH))

        logger.info(f"Added chain: {name} with {len(tools)} tools")
        return chain

    async def record_chain_use(self, chain_name: str):
        """Record that a chain was used (for ranking)."""
        db = self._get_db()
        db.execute(
            """
            UPDATE tool_chains
            SET use_count = use_count + 1, last_used_at = CURRENT_TIMESTAMP
            WHERE chain_name = ?
        """,
            (chain_name,),
        )
        db.commit()

        # Update cache if this chain is in it
        for chain in self._chain_cache:
            if chain.name == chain_name:
                chain.use_count += 1
                break

    async def get_chain(self, chain_name: str) -> Optional[ToolChain]:
        """Get a specific chain by name."""
        # Check cache first
        for chain in self._chain_cache:
            if chain.name == chain_name:
                return chain

        # Check DB
        db = self._get_db()
        row = db.execute(
            """
            SELECT id, chain_name, chain_tools, description, use_count, is_auto_detected, embedding_text
            FROM tool_chains
            WHERE chain_name = ?
        """,
            (chain_name,),
        ).fetchone()

        if row:
            return ToolChain(
                id=row["id"],
                name=row["chain_name"],
                tools=json.loads(row["chain_tools"]),
                description=row["description"] or "",
                use_count=row["use_count"],
                is_auto_detected=bool(row["is_auto_detected"]),
                embedding_text=row["embedding_text"],
            )

        return None

    async def seed_default_chains(self):
        """Add predefined common tool chains."""
        default_chains = [
            {
                "name": "file_modify",
                "tools": ["bridge:read_file", "bridge:write_file"],
                "description": "Read a file, modify its contents, and write it back",
            },
            {
                "name": "git_commit",
                "tools": ["bridge:git_status", "bridge:git_add", "bridge:git_commit"],
                "description": "Check status, stage changes, and commit to git",
            },
            {
                "name": "code_analysis",
                "tools": ["doc:scan_codebase", "doc:generate_report"],
                "description": "Analyze codebase and generate a health report",
            },
            {
                "name": "image_generation",
                "tools": [
                    "comfy:comfy_status",
                    "comfy:comfy_generate",
                    "comfy:comfy_history",
                ],
                "description": "Check ComfyUI status, generate an image, and view history",
            },
            {
                "name": "database_query",
                "tools": [
                    "bridge:db_list_tables",
                    "bridge:db_inspect_table",
                    "bridge:db_execute",
                ],
                "description": "List tables, inspect schema, and run queries",
            },
        ]

        for chain_def in default_chains:
            existing = await self.get_chain(chain_def["name"])
            if not existing:
                await self.add_chain(
                    name=chain_def["name"],
                    tools=chain_def["tools"],
                    description=chain_def["description"],
                    is_auto_detected=False,
                )

        logger.info(f"Seeded {len(default_chains)} default chains")

    def get_cached_chains(self) -> List[ToolChain]:
        """Get the cached top chains."""
        return self._chain_cache

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None


# Singleton instance
_chain_indexer_instance: Optional[ChainIndexer] = None


def get_chain_indexer(
    embedder: "Embedder", analytics: Optional["CompassAnalytics"] = None
) -> ChainIndexer:
    """Get or create the chain indexer singleton."""
    global _chain_indexer_instance
    if _chain_indexer_instance is None:
        _chain_indexer_instance = ChainIndexer(embedder, analytics)
    return _chain_indexer_instance
