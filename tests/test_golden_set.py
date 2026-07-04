"""Golden-set semantic-search regression — TS-B-006.

This is the only meaningful correctness signal for the semantic-search
gateway. Existing search tests are type-only (assert `len(results) > 0`,
assert `isinstance(r, SearchResult)`) and would pass even if retrieval
silently regressed to "always return the first tool by id."

The benchmark:

  1. Build a CompassIndex against the GOLDEN_TOOLS fixture corpus using
     a DETERMINISTIC concept-basis embedder (no Ollama dependency).
  2. For each (query, expected) pair in queries.yaml, run
     `index.search(query, top_k=K)` and record:
       - Recall@k     = |expected ∩ retrieved_top_k| / |expected|
       - nDCG@k       = sum(rel_i / log2(i+2)) / IDCG
       - Hit@k        = 1 if any expected appears in top-k else 0
       - top-1 hit    = 1 if expected[0] == retrieved[0]
  3. Aggregate across all queries and assert the floor.

Research basis:

  - TREC-style golden-set evaluation (https://trec.nist.gov/) — the
    canonical correctness signal for IR systems since 1992.
  - BEIR benchmark (Thakur et al. 2021, https://github.com/beir-cellar/beir)
    operationalizes Recall@k + nDCG@k for retrieval-system regression.
  - Karpukhin et al. 2020 (DPR, https://arxiv.org/abs/2004.04906)
    establishes MRR@10 + Recall@k as the canonical regression metric for
    dense retrieval.

The deterministic embedder cannot reflect every nuance of a real model,
but it WILL catch:

  - Tokenization changes that drop key concepts before embedding.
  - Normalization regressions (dropping L2-normalize would shift scores).
  - Wrong `top_k` clamping (e.g., always returning 1 result).
  - HNSW build failures that silently produce an empty index.
  - Distance-to-similarity conversion bugs.
  - Filtering bugs (category/server filters that exclude valid hits).

Mark `@pytest.mark.golden` so this can be run on a separate cadence —
included in `pytest` default, but skippable via `pytest -m "not golden"`
in tight dev loops.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from config import CompassConfig
from indexer import CompassIndex
from tests.golden_set.deterministic_embedder import (
    build_deterministic_embedder,
    embed_text,
)
from tests.golden_set.fixture_corpus import GOLDEN_TOOLS, all_tool_names


_QUERIES_PATH = Path(__file__).parent / "golden_set" / "queries.yaml"


def _load_queries() -> list[dict]:
    """Load the frozen golden-set queries from queries.yaml."""
    with open(_QUERIES_PATH, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    return data["queries"]


def _recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Recall@k = |expected ∩ retrieved_top_k| / |expected|."""
    if not expected:
        return 1.0
    hit = sum(1 for e in expected if e in retrieved[:k])
    return hit / len(expected)


def _ndcg_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """nDCG@k with binary relevance. Higher == better; max == 1.0."""
    expected_set = set(expected)
    dcg = 0.0
    for i, name in enumerate(retrieved[:k]):
        rel = 1.0 if name in expected_set else 0.0
        if rel:
            dcg += rel / math.log2(i + 2)  # i=0 → log2(2)=1
    # Ideal DCG: all `min(len(expected), k)` relevant docs at the top.
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


@pytest.fixture
async def golden_index(temp_index_path, temp_db_path):
    """A CompassIndex built against the golden-set corpus + deterministic embedder."""
    embedder = build_deterministic_embedder()
    index = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=embedder,
    )
    await index.build_index(GOLDEN_TOOLS)
    try:
        yield index
    finally:
        await index.close()


# -----------------------------------------------------------------------------
# Sanity checks on the fixture itself — surface drift early
# -----------------------------------------------------------------------------


@pytest.mark.golden
class TestGoldenSetFixture:
    """Fixture integrity: every `expected` tool in queries.yaml exists in the
    corpus. Drift surfaces as a clear failure, not as a mystery low recall."""

    def test_all_expected_tools_exist_in_corpus(self):
        queries = _load_queries()
        names = set(all_tool_names())
        missing: list[str] = []
        for q in queries:
            for tool in q["expected"]:
                if tool not in names:
                    missing.append(f"{tool} (referenced by query: {q['query']!r})")
        assert not missing, (
            "queries.yaml references tools not in the golden corpus:\n  "
            + "\n  ".join(missing)
        )

    def test_corpus_has_at_least_seventeen_tools(self):
        """Corpus size pins the baseline difficulty. Shrinking it makes the
        benchmark too easy."""
        assert len(GOLDEN_TOOLS) >= 17

    def test_query_set_has_at_least_thirty_queries(self):
        """Query count pins benchmark coverage. Shrinking it lets a regression
        slip through on the queries that happen to remain."""
        queries = _load_queries()
        assert len(queries) >= 30

    def test_deterministic_embedder_is_deterministic(self):
        """Two calls with the same text must produce byte-identical vectors,
        otherwise the benchmark loses its 'frozen' property."""
        a = embed_text("read a file from disk")
        b = embed_text("read a file from disk")
        assert (a == b).all()


# -----------------------------------------------------------------------------
# Top-k correctness — the actual benchmark
# -----------------------------------------------------------------------------


@pytest.mark.golden
class TestGoldenSetRetrieval:
    """Run the full golden set against the index and assert the floor."""

    @pytest.mark.asyncio
    async def test_recall_at_5_meets_floor(self, golden_index):
        """Average Recall@5 over the entire golden set must clear 0.80.

        Below the floor signals retrieval-pipeline regression — tokenization,
        normalization, HNSW build, or filter logic likely broke.
        """
        queries = _load_queries()
        recalls: list[float] = []
        misses: list[str] = []

        for q in queries:
            results = await golden_index.search(q["query"], top_k=5)
            retrieved = [r.tool.name for r in results]
            r = _recall_at_k(q["expected"], retrieved, 5)
            recalls.append(r)
            if r < 1.0:
                misses.append(
                    f"  {q['query']!r}\n"
                    f"    expected: {q['expected']}\n"
                    f"    got:      {retrieved}"
                )

        avg_recall = sum(recalls) / len(recalls)
        # Floor: 0.80. The benchmark is deterministic so this is a tight bound.
        # If the suite passes today, a regression that drops avg recall below
        # 0.80 represents a meaningful retrieval-quality loss.
        assert avg_recall >= 0.80, (
            f"Recall@5 = {avg_recall:.3f} < 0.80 floor. Misses:\n"
            + "\n".join(misses)
        )

    @pytest.mark.asyncio
    async def test_ndcg_at_5_meets_floor(self, golden_index):
        """nDCG@5 average must clear 0.70. Rewards correct top placement."""
        queries = _load_queries()
        ndcgs: list[float] = []
        for q in queries:
            results = await golden_index.search(q["query"], top_k=5)
            retrieved = [r.tool.name for r in results]
            ndcgs.append(_ndcg_at_k(q["expected"], retrieved, 5))
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        assert avg_ndcg >= 0.70, f"nDCG@5 = {avg_ndcg:.3f} < 0.70 floor"

    @pytest.mark.asyncio
    async def test_hit_rate_at_5_meets_floor(self, golden_index):
        """Hit@5 (at least one expected tool in top-5) must clear 0.85."""
        queries = _load_queries()
        hits = 0
        for q in queries:
            results = await golden_index.search(q["query"], top_k=5)
            retrieved = {r.tool.name for r in results}
            if any(e in retrieved for e in q["expected"]):
                hits += 1
        hit_rate = hits / len(queries)
        assert hit_rate >= 0.85, f"Hit@5 = {hit_rate:.3f} < 0.85 floor"

    @pytest.mark.asyncio
    async def test_empty_query_returns_no_crash(self, golden_index):
        """Edge case: empty query must not crash the pipeline."""
        # Empty query goes through the deterministic embedder which falls
        # back to basis[0], so search will return *some* result rather than
        # raise. The contract is: list returned, no exception.
        results = await golden_index.search("", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, golden_index):
        """Production contract: results <= top_k for every k in [1..10]."""
        for k in (1, 3, 5, 10):
            results = await golden_index.search("read a file", top_k=k)
            assert len(results) <= k, f"top_k={k}, got {len(results)} results"


# -----------------------------------------------------------------------------
# GW-DISC-002 (DEFECT 2) — the FUSED (shipped-default) compass() ranking path
# -----------------------------------------------------------------------------
# The benchmark above calls golden_index.search() directly — the pure-semantic
# path. But hybrid_search + exact_name_boost both DEFAULT True, so production
# actually serves the FUSED compass() ranking (RRF fusion + exact-name boost).
# Without exercising that path, a ranking regression on the actually-served
# path passes CI. These tests drive compass() end-to-end over the golden corpus
# with the default (fusion+boost ON) config and assert the same floors, so a
# fused-path regression fails CI. Floors are pinned at the pure-semantic floors
# — fusion must never regress the served path below what pure semantics clears.


def _mrr(expected: list[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first relevant hit (Karpukhin et al. 2020 / DPR
    canonical dense-retrieval metric). 0.0 if no expected tool retrieved."""
    exp = set(expected)
    for i, name in enumerate(retrieved):
        if name in exp:
            return 1.0 / (i + 1)
    return 0.0


@pytest.fixture
async def fused_gateway(golden_index):
    """Wire gateway module globals to the golden index with the DEFAULT
    (shipped) fusion+boost config, then restore via the autouse
    _reset_gateway_globals fixture in conftest.py."""
    import gateway

    cfg = CompassConfig(
        backends={},
        auto_sync=False,
        analytics_enabled=False,
        chain_indexing_enabled=False,
        progressive_disclosure=True,
        min_confidence=0.3,
    )
    # Assert we are exercising the SHIPPED defaults, not a hand-tuned config.
    assert cfg.hybrid_search is True
    assert cfg.exact_name_boost is True

    gateway._compass_index = golden_index
    gateway._config = cfg
    gateway._startup_sync_done = True
    gateway._analytics = None
    gateway._health_state["ollama_available"] = True
    gateway._health_state["index_available"] = True
    return gateway


async def _fused_retrieved(query: str, k: int = 5) -> list[str]:
    """Run the FUSED compass() path and return the ordered tool names."""
    from gateway import compass

    result = await compass(intent=query, top_k=k, include_chains=False)
    return [m["tool"] for m in result["matches"]]


@pytest.mark.golden
# The deterministic golden embedder is an AsyncMock; compass()'s health-probe
# surface touches an unconfigured async attribute on it, leaking a harmless
# "coroutine never awaited" RuntimeWarning. That is a property of the test
# double, not production code (the real embedder is not a mock), so scope-ignore
# it here rather than let it add noise to the suite.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestGoldenSetFusedPathDISC02:
    """DEFECT 2: the fused (RRF fusion + exact-name boost) compass() path — the
    one production actually serves — has its own golden-benchmark coverage.

    Floors are set AT the pure-semantic floors (Recall@5>=0.80, nDCG@5>=0.70,
    Hit@5>=0.85) plus an MRR floor. Fusion currently clears them with margin
    (Recall/Hit ~0.95, nDCG ~0.87, MRR ~0.85); pinning at the semantic floor
    means a fusion regression that drops the SERVED path below the pure-semantic
    quality bar fails CI, while day-to-day fusion noise does not flake.
    """

    @pytest.mark.asyncio
    async def test_fused_recall_at_5_meets_floor(self, fused_gateway):
        queries = _load_queries()
        recalls: list[float] = []
        misses: list[str] = []
        for q in queries:
            retrieved = await _fused_retrieved(q["query"], 5)
            r = _recall_at_k(q["expected"], retrieved, 5)
            recalls.append(r)
            if r < 1.0:
                misses.append(
                    f"  {q['query']!r}\n"
                    f"    expected: {q['expected']}\n"
                    f"    got:      {retrieved}"
                )
        avg_recall = sum(recalls) / len(recalls)
        assert avg_recall >= 0.80, (
            f"FUSED Recall@5 = {avg_recall:.3f} < 0.80 floor (fusion regressed "
            f"the served path below the pure-semantic bar). Misses:\n"
            + "\n".join(misses)
        )

    @pytest.mark.asyncio
    async def test_fused_ndcg_at_5_meets_floor(self, fused_gateway):
        queries = _load_queries()
        ndcgs: list[float] = []
        for q in queries:
            retrieved = await _fused_retrieved(q["query"], 5)
            ndcgs.append(_ndcg_at_k(q["expected"], retrieved, 5))
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        assert avg_ndcg >= 0.70, f"FUSED nDCG@5 = {avg_ndcg:.3f} < 0.70 floor"

    @pytest.mark.asyncio
    async def test_fused_hit_rate_at_5_meets_floor(self, fused_gateway):
        queries = _load_queries()
        hits = 0
        for q in queries:
            retrieved = set(await _fused_retrieved(q["query"], 5))
            if any(e in retrieved for e in q["expected"]):
                hits += 1
        hit_rate = hits / len(queries)
        assert hit_rate >= 0.85, f"FUSED Hit@5 = {hit_rate:.3f} < 0.85 floor"

    @pytest.mark.asyncio
    async def test_fused_mrr_meets_floor(self, fused_gateway):
        """MRR rewards correct TOP placement — the metric most sensitive to a
        fusion/boost reorder regression on the served path."""
        queries = _load_queries()
        rrs = [
            _mrr(q["expected"], await _fused_retrieved(q["query"], 5))
            for q in queries
        ]
        avg_mrr = sum(rrs) / len(rrs)
        assert avg_mrr >= 0.75, f"FUSED MRR = {avg_mrr:.3f} < 0.75 floor"

    @pytest.mark.asyncio
    async def test_fused_exact_name_paste_pins_tool_at_top(self, fused_gateway):
        """The exact-name boost (default ON) must pin an exact tool-name paste
        to rank #1 at exact_match_confidence on the served path — a behavior the
        pure-semantic tests never exercise. The confidence assertion is the
        boost-specific signal: with the boost OFF the top match would carry its
        raw semantic score (~0.77), not the stamped 1.0."""
        from gateway import compass

        result = await compass(intent="git:commit", top_k=5, include_chains=False)
        matches = result["matches"]
        assert matches, "expected the boosted exact match"
        assert matches[0]["tool"] == "git:commit", (
            "exact-name paste must pin the tool at #1 on the fused path; got "
            + str([m["tool"] for m in matches])
        )
        # exact_match_confidence default is 1.0 — the boost stamp, not a
        # semantic score. This distinguishes the boost from pure semantics.
        assert matches[0]["confidence"] == fused_gateway._config.exact_match_confidence
        assert matches[0]["confidence"] == 1.0
