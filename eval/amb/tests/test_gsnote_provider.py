"""Tests for the gsnote AMB provider. Skipped unless AMB (memory_bench) is installed."""
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("memory_bench")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsnote_provider import GsnoteMemoryProvider, space_for_user  # noqa: E402
from memory_bench.models import Document  # noqa: E402
from memory_bench.utils import chunk_text  # noqa: E402

SPACE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
_ENV = ("DB_PATH", "EMBEDDING_MODEL", "EMBEDDING_DIM", "RETRIEVAL_CANDIDATE_N")


@pytest.fixture
def provider(tmp_path, monkeypatch):
    # Record + clear the env the provider writes, so teardown restores it and the main
    # gsnote suite is unaffected when both run in one process.
    for name in _ENV:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    p = GsnoteMemoryProvider()
    p.prepare(tmp_path / "store")
    yield p
    p.cleanup()
    from app.config import get_settings

    get_settings.cache_clear()


def _docs():
    return [
        Document(id="d-db", content="We decided to use PostgreSQL for the primary database.", user_id="alice"),
        Document(id="d-cat", content="The cat sleeps on the warm windowsill all afternoon.", user_id="alice"),
        Document(id="d-bob", content="Bob prefers PostgreSQL migrations written in plain SQL.", user_id="bob"),
    ]


def test_roundtrip(provider):
    provider.ingest(_docs())
    docs, raw = provider.retrieve("which database did we pick", k=3, user_id="alice")
    assert docs[0].id == "d-db"
    assert docs[0].source_ids == ["d-db"]
    assert docs[0].user_id == "alice"
    assert "PostgreSQL" in docs[0].content
    assert raw is not None and raw["results"]


def test_user_isolation(provider):
    provider.ingest(_docs())
    alice, _ = provider.retrieve("PostgreSQL", k=10, user_id="alice")
    bob, _ = provider.retrieve("PostgreSQL", k=10, user_id="bob")
    assert {d.id for d in alice} <= {"d-db", "d-cat"}
    assert {d.id for d in bob} == {"d-bob"}
    nobody, _ = provider.retrieve("PostgreSQL", k=10, user_id="carol")
    assert nobody == []
    # user_id=None searches every space
    everyone, _ = provider.retrieve("PostgreSQL", k=10)
    assert {"d-db", "d-bob"} <= {d.id for d in everyone}


def test_chunk_dedupe_back_to_doc_id(provider):
    long_text = " ".join(f"filler{i}" for i in range(1500)) + " the zebra crossing uses a pelican signal"
    assert len(chunk_text(long_text)) > 1
    provider.ingest([
        Document(id="long", content=long_text, user_id="u"),
        Document(id="short", content="Unrelated note about gardening tomatoes.", user_id="u"),
    ])
    from app.retrieval import search

    note_ids = {r["id"] for r in search("pelican signal", space="u", top_k=20)}
    assert {"long::0", "long::1"} <= note_ids  # stored as several chunk notes
    docs, _ = provider.retrieve("filler zebra pelican signal", k=5, user_id="u")
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))  # one hit per original document
    assert set(ids) == {"long", "short"}
    assert all("::" not in i for i in ids)


def test_k_distinct_docs_despite_chunks(provider):
    # Every doc is 3 chunks that all match; k=4 must still yield 4 distinct docs.
    word = "quokka"
    docs = [Document(id=f"doc{i}", content=" ".join([word] * 1400), user_id="u") for i in range(6)]
    provider.ingest(docs)
    got, _ = provider.retrieve(word, k=4, user_id="u")
    assert len(got) == 4 and len({d.id for d in got}) == 4


def test_timestamps_stored(provider):
    provider.ingest([
        Document(id="t1", content="Dated note about the launch.", user_id="u", timestamp="2024-03-05T10:00:00Z"),
        Document(id="t2", content="Undated note about the launch party.", user_id="u"),
        Document(id="t3", content="Weird date note about the launch.", user_id="u", timestamp="5 March, 2024"),
    ])
    got = {d.id: d for d in provider.retrieve("launch", k=5, user_id="u")[0]}
    assert got["t1"].timestamp.startswith("2024-03-05T10:00:00")
    assert got["t2"].timestamp  # defaults to ingest time
    assert got["t3"].timestamp  # unparseable -> ingest time, not an error


def test_reset_behaviour(tmp_path):
    store = tmp_path / "store"
    p = GsnoteMemoryProvider()
    try:
        p.prepare(store)
        p.ingest(_docs())
        p.prepare(store, reset=False)  # keeps data (skip-ingestion)
        kept, _ = p.retrieve("PostgreSQL", k=5, user_id="alice")
        assert kept and kept[0].id == "d-db"
        p.prepare(store, reset=True)  # wipes it
        assert p.retrieve("PostgreSQL", k=5, user_id="alice")[0] == []
        assert os.path.exists(store / "gsnote.db")
        assert os.environ["DB_PATH"] == str(store / "gsnote.db")
    finally:
        p.cleanup()
        from app.config import get_settings

        get_settings.cache_clear()


def test_reingest_is_idempotent_but_cross_user_id_clash_fails(provider):
    provider.ingest(_docs())
    provider.ingest(_docs())
    docs, _ = provider.retrieve("PostgreSQL", k=10, user_id="alice")
    assert len([d for d in docs if d.id == "d-db"]) == 1
    with pytest.raises(ValueError, match="already ingested under another user"):
        provider.ingest([Document(id="d-db", content="Same id, different user.", user_id="bob")])


@pytest.mark.parametrize(
    "user_id",
    [None, "test-user", "other-user", "Alice", "a_b c", "user@example.com", "ąę", "", "---", "x" * 80, "conv-26/speaker_a"],
)
def test_space_names_valid_and_deterministic(user_id):
    space = space_for_user(user_id)
    assert SPACE_RE.fullmatch(space)
    assert space == space_for_user(user_id)
    from app.spaces import normalize_space

    assert normalize_space(space) == space


def test_space_names_do_not_collide():
    ids = ["Alice", "alice", "a_b", "a-b", "a b", "x" * 40, "x" * 41, "", "---", "ąę", "ęą"]
    spaces = [space_for_user(i) for i in ids]
    assert len(set(spaces)) == len(spaces)
    assert space_for_user("test-user") == "test-user"  # already-valid names pass through
    assert space_for_user(None) == "default"
