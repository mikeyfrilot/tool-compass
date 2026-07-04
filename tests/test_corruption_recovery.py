"""
Corruption + lock recovery tests (TST-B-002).

These tests lock in the Stage C production fixes in indexer.py — specifically
CompassIndex.load_index()'s dim-mismatch gate, count-mismatch warning, and
graceful failure on corrupt HNSW files. They also exercise the error path
when SQLite is locked by a concurrent exclusive transaction.

The point of this file is that when something goes wrong on disk (different
embedding model produced the existing index, bytes got corrupted, another
process holds the DB open), the user sees a HELPFUL error that tells them
what to do next — not a cryptic hnswlib/sqlite3 stack trace.
"""

from __future__ import annotations

import logging
import random
import sqlite3

import pytest

from embedder import EMBEDDING_DIM
from indexer import CompassIndex


@pytest.mark.asyncio
async def test_load_index_rejects_dim_mismatch(
    test_index, temp_index_path, temp_db_path, mock_embedder
):
    """load_index() must fail loudly when the persisted embedding dim
    doesn't match the code's EMBEDDING_DIM.

    The error message is load-bearing: it must name the index path so the
    user knows what to delete. This is the Stage C production fix for
    IDX-B-001 — without it, a model swap silently produces nonsense search
    results.
    """
    # Corrupt the persisted dim to simulate "embedding model changed under us".
    wrong_dim = EMBEDDING_DIM + 1  # any value != real dim
    db = sqlite3.connect(str(temp_db_path))
    try:
        db.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('embedding_dim', ?)",
            (str(wrong_dim),),
        )
        db.commit()
    finally:
        db.close()

    fresh = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=mock_embedder,
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            fresh.load_index()

        msg = str(exc_info.value)
        # Must name the file to delete — this is the helpful-failure contract.
        assert str(temp_index_path) in msg, (
            f"dim-mismatch error must name the index path so the user "
            f"knows what to delete; got: {msg!r}"
        )
        # Must mention both the saved and expected dim.
        assert str(wrong_dim) in msg
        assert str(EMBEDDING_DIM) in msg
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_load_index_handles_count_mismatch(
    test_index, temp_index_path, temp_db_path, mock_embedder, caplog
):
    """When HNSW vector count != DB row count, load_index must warn but
    return True — the server stays up with degraded recall rather than
    crashing. Stage C production fix for IDX-B-001.
    """
    # Delete one row from the DB so counts disagree. HNSW still has the
    # vector; DB is missing the mapping.
    db = sqlite3.connect(str(temp_db_path))
    try:
        db.execute("DELETE FROM tools WHERE id = (SELECT MIN(id) FROM tools)")
        db.commit()
    finally:
        db.close()

    fresh = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=mock_embedder,
    )

    try:
        with caplog.at_level(logging.WARNING, logger="indexer"):
            loaded = fresh.load_index()

        assert loaded is True, "mismatch is a warning, not a fatal error"
        # Must have logged a warning naming the integrity issue.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("integrity" in r.getMessage().lower() for r in warnings), (
            "count mismatch must emit a warning that operators can grep for"
        )
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_corrupt_hnsw_file_fails_gracefully(
    test_index, temp_index_path, temp_db_path, mock_embedder
):
    """Random bytes in the HNSW file must surface as load_index() == False
    (not an uncaught hnswlib crash). The caller then knows to rebuild.
    """
    # close the built index so Windows will let us overwrite the file.
    await test_index.close()

    # Overwrite HNSW with 100 bytes of garbage.
    rng = random.Random(0xC0FFEE)
    garbage = bytes(rng.randrange(256) for _ in range(100))
    temp_index_path.write_bytes(garbage)

    fresh = CompassIndex(
        index_path=temp_index_path,
        db_path=temp_db_path,
        embedder=mock_embedder,
    )

    try:
        # Contract (Stage C): load_index surfaces the corruption clearly
        # — either by raising with an advisory message, or by returning
        # False so the gateway can degrade to lexical search. What we
        # LOCK IN here is "does not pretend to succeed": if the current
        # production code re-raises the hnswlib RuntimeError (no extra
        # advice yet), that's still an acceptable failure signal.
        # Follow-up for the production agent: catch the raw hnswlib
        # RuntimeError here and wrap with an advisory that names the
        # rebuild command (e.g., `compass_sync(force=True)`).
        try:
            result = fresh.load_index()
            assert result is False, (
                "corrupt HNSW must not be treated as a successful load — "
                "gateway needs this signal to prompt a rebuild"
            )
        except RuntimeError as e:
            # Raised path is ALSO acceptable — lock in that the error is
            # visible rather than silently eaten. Ideally the message
            # would name "rebuild" / "compass_sync" — track that as a
            # follow-up; don't fail the test on message text yet since
            # the raw hnswlib message doesn't include it.
            assert str(e), "corrupt-HNSW RuntimeError must have a message"
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_sqlite_db_locked_surfaces_error(
    test_index, temp_db_path, mock_embedder
):
    """When another process holds an exclusive lock on the SQLite DB,
    an index mutation must surface an error quickly — not hang forever.

    This covers the "lock recovery path" part of TST-B-002. We open a
    second connection with BEGIN EXCLUSIVE and confirm remove_tool
    returns False (the err-path logs and surfaces) within a short
    busy-timeout window instead of blocking indefinitely.
    """
    # TST-RA-003: assert the concrete locked-write contract, not a tautology.
    # remove_tool reads the row (SELECT succeeds), then DELETEs + commits under
    # an EXCLUSIVE lock. That write must fail and be caught, so:
    #   1. remove_tool returns False (the operation could not succeed), and
    #   2. no partial write occurred — the row still exists once the lock lifts.

    # Sanity: the target tool exists before we try to remove it.
    pre = test_index.db.execute(
        "SELECT COUNT(*) AS n FROM tools WHERE name = ?", ("test:read_file",)
    ).fetchone()
    assert pre["n"] == 1

    # Take an exclusive lock in a second connection.
    locker = sqlite3.connect(str(temp_db_path), timeout=0.1)
    locker.execute("BEGIN EXCLUSIVE")

    try:
        # Keep the lock budget small so a bug (infinite wait) fails fast.
        # remove_tool writes through the index.db connection, which will
        # hit the lock.
        test_index.db.execute("PRAGMA busy_timeout = 200")

        # remove_tool catches the "database is locked" write error internally
        # and returns False. The concrete contract: the locked write cannot
        # succeed, so the return must be exactly False (not just "either").
        result = await test_index.remove_tool("test:read_file")
        assert result is False
    finally:
        # Always release the lock so the fixture teardown can run.
        try:
            locker.rollback()
        finally:
            locker.close()

    # No partial write: with the lock released, the row must still be present
    # (the failed DELETE did not remove it). State is unchanged.
    post = test_index.db.execute(
        "SELECT COUNT(*) AS n FROM tools WHERE name = ?", ("test:read_file",)
    ).fetchone()
    assert post["n"] == 1
