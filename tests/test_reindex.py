"""Embedding config tracking (startup check) and the `python -m app.reindex` command."""
import hashlib
import json
import math
import re

import pytest
from fastapi.testclient import TestClient

from app import db, embeddings, reindex
from app.capture import capture_note, delete_note, import_notes
from app.config import get_settings
from app.db import EmbeddingConfigMismatch
from app.retrieval import search


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Deterministic bag-of-words hashing embeddings of the configured dimension (no model download)."""

    def fake_embed_many(texts, task=None):
        dim = get_settings().embedding_dim
        out = []
        for text in texts:
            vec = [0.0] * dim
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    monkeypatch.setattr(embeddings, "embed_many", fake_embed_many)


def _switch_config(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()


def _vector_count() -> int:
    return db.get_conn().execute("SELECT COUNT(*) FROM note_vectors").fetchone()[0]


def _seed():
    capture_note("Pay the electricity bill on Friday")
    capture_note("Dentist appointment moved to March")
    capture_note("Buy oat milk and coffee")


def test_fresh_db_records_the_current_config():
    assert db.recorded_embedding_config(db.get_conn()) == db.current_embedding_config()
    assert db.recorded_embedding_config(db.get_conn())["embedding_dim"] == "384"
    db.require_vectors_ready()


def test_legacy_db_without_meta_gets_current_config_recorded():
    conn = db.get_conn()
    _seed()
    conn.execute("DROP TABLE meta")
    conn.execute("DELETE FROM schema_migrations WHERE version = '002_meta'")
    conn.commit()
    assert db.recorded_embedding_config(conn) == {}

    db.run_migrations()

    assert db.recorded_embedding_config(conn) == db.current_embedding_config()
    db.require_vectors_ready()
    assert len(search("electricity")) == 1


def test_legacy_db_with_other_vector_dim_is_a_mismatch(monkeypatch):
    conn = db.get_conn()
    conn.execute("DELETE FROM meta")
    conn.commit()
    _switch_config(monkeypatch, EMBEDDING_DIM="16")

    db.run_migrations()

    assert db.recorded_embedding_config(conn) == {}  # nothing recorded: we cannot assume it matches
    with pytest.raises(EmbeddingConfigMismatch):
        db.require_vectors_ready()


@pytest.mark.parametrize(
    "env",
    [
        {"EMBEDDING_MODEL": "some/other-model"},
        {"EMBEDDING_DIM": "16"},
        {"EMBEDDING_DOCUMENT_PREFIX": "passage: "},
    ],
)
def test_startup_mismatch_disables_vector_features_with_reindex_hint(monkeypatch, env):
    _seed()
    _switch_config(monkeypatch, **env)

    db.run_migrations()

    for call in (
        lambda: capture_note("new note"),
        lambda: import_notes({"version": 2, "notes": [{"id": "x", "content": "hello"}]}),
        lambda: search("electricity"),
    ):
        with pytest.raises(EmbeddingConfigMismatch, match=r"python -m app\.reindex"):
            call()
    # Recorded config is left alone, and reads/deletes that need no vectors still work.
    assert db.recorded_embedding_config(db.get_conn())["embedding_dim"] == "384"
    listed = search()
    assert len(listed) == 3
    assert delete_note(listed[0]["id"])


def test_mismatch_is_reported_over_http_as_503(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok")
    _switch_config(monkeypatch, EMBEDDING_MODEL="some/other-model")
    from app.main import app

    headers = {"Authorization": "Bearer tok"}
    with TestClient(app) as client:  # startup must not fail
        assert client.get("/health").status_code == 200
        assert client.get("/search", headers=headers).status_code == 200  # list: no vectors needed
        for resp in (
            client.post("/capture", json={"content": "x"}, headers=headers),
            client.get("/search", params={"q": "x"}, headers=headers),
        ):
            assert resp.status_code == 503
            assert "python -m app.reindex" in resp.json()["detail"]


def test_reindex_after_dim_and_model_change(monkeypatch):
    _seed()
    ids = {n["id"] for n in search()}
    _switch_config(monkeypatch, EMBEDDING_MODEL="some/other-model", EMBEDDING_DIM="16")
    db.run_migrations()
    with pytest.raises(EmbeddingConfigMismatch):
        capture_note("blocked")

    summary = reindex.reindex(batch_size=2)

    assert summary["notes"] == 3
    assert summary["previous"]["embedding_dim"] == "384"
    conn = db.get_conn()
    assert db.recorded_embedding_config(conn) == db.current_embedding_config()
    assert db.recorded_embedding_config(conn)["embedding_dim"] == "16"
    assert db.declared_vector_dim(conn) == 16
    assert {r[0] for r in conn.execute("SELECT note_id FROM note_vectors")} == ids
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (reindex.STAGING,)).fetchone()
    # Search and capture work with the new dimension, in this process and after a restart.
    assert search("electricity bill")[0]["content"].startswith("Pay the electricity")
    capture_note("Water the plants on Sunday")
    assert _vector_count() == 4
    db.run_migrations()
    db.require_vectors_ready()
    assert search("plants")[0]["content"].startswith("Water the plants")


def test_reindex_after_document_prefix_change(monkeypatch):
    _seed()
    seen = []
    real = embeddings.embed_many
    monkeypatch.setattr(embeddings, "embed_many", lambda texts, task=None: seen.extend(texts) or real(texts, task))
    _switch_config(monkeypatch, EMBEDDING_DOCUMENT_PREFIX="passage: ")
    db.run_migrations()
    with pytest.raises(EmbeddingConfigMismatch):
        search("electricity")

    reindex.reindex()

    assert all(t.startswith("passage: ") for t in seen) and len(seen) == 3
    assert db.recorded_embedding_config(db.get_conn())["embedding_document_prefix"] == "passage: "
    assert search("electricity")


def test_reindex_is_idempotent_and_handles_an_empty_db(monkeypatch):
    assert reindex.reindex()["notes"] == 0
    _seed()
    reindex.reindex()
    reindex.reindex()
    assert _vector_count() == 3
    assert len(search("coffee")) >= 1


def test_dry_run_changes_nothing(monkeypatch, capsys):
    _seed()
    _switch_config(monkeypatch, EMBEDDING_MODEL="some/other-model", EMBEDDING_DIM="16")
    conn = db.get_conn()
    before = (db.recorded_embedding_config(conn), db.declared_vector_dim(conn), _vector_count())
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}

    assert reindex.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "BAAI/bge-small-en-v1.5" in out and "some/other-model" in out
    assert "Notes: 3" in out and "Reindex needed: yes" in out and "nothing changed" in out
    assert (db.recorded_embedding_config(conn), db.declared_vector_dim(conn), _vector_count()) == before
    assert {r[0] for r in conn.execute("SELECT name FROM sqlite_master")} == tables


def test_dry_run_reports_unchanged_config(capsys):
    assert reindex.main(["--dry-run"]) == 0
    assert "Reindex needed: no" in capsys.readouterr().out


def test_dry_run_on_pre_meta_database_does_not_migrate(capsys):
    conn = db.get_conn()
    conn.execute("DROP TABLE meta")
    conn.execute("DELETE FROM schema_migrations WHERE version = '002_meta'")
    conn.commit()

    assert reindex.main(["--dry-run"]) == 0

    assert "(not recorded)" in capsys.readouterr().out
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone()


def test_interrupted_reindex_leaves_the_old_vectors_intact_and_can_be_rerun(monkeypatch):
    _seed()
    _switch_config(monkeypatch, EMBEDDING_DIM="16")
    db.run_migrations()
    conn = db.get_conn()
    real = embeddings.embed_many
    calls = {"n": 0}

    def flaky(texts, task=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(texts, task)

    monkeypatch.setattr(embeddings, "embed_many", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        reindex.reindex(batch_size=1)

    assert db.declared_vector_dim(conn) == 384 and _vector_count() == 3
    assert db.recorded_embedding_config(conn)["embedding_dim"] == "384"
    assert conn.execute("SELECT COUNT(*) FROM " + reindex.STAGING).fetchone()[0] == 1  # partial staging only

    monkeypatch.setattr(embeddings, "embed_many", real)
    reindex.reindex(batch_size=1)

    assert db.declared_vector_dim(conn) == 16 and _vector_count() == 3
    assert db.recorded_embedding_config(conn)["embedding_dim"] == "16"


def test_failed_swap_rolls_back_completely(monkeypatch):
    _seed()
    _switch_config(monkeypatch, EMBEDDING_DIM="16")
    db.run_migrations()
    conn = db.get_conn()

    def boom(*args, **kwargs):
        raise RuntimeError("meta write failed")

    monkeypatch.setattr(db, "record_embedding_config", boom)
    with pytest.raises(RuntimeError, match="meta write failed"):
        reindex.reindex()

    assert not conn.in_transaction
    assert db.declared_vector_dim(conn) == 384 and _vector_count() == 3
    assert db.recorded_embedding_config(conn)["embedding_dim"] == "384"
    # The old index still answers with the old dimension.
    assert conn.execute(
        "SELECT note_id FROM note_vectors WHERE embedding MATCH ? AND k = 1", (json.dumps([0.0] * 384),)
    ).fetchall()


def test_reindex_cli_main_end_to_end(monkeypatch, capsys):
    _seed()
    _switch_config(monkeypatch, EMBEDDING_DIM="32")
    assert reindex.main(["--batch-size", "2"]) == 0
    out = capsys.readouterr().out
    assert "Embedding 3/3" in out and "Reindexed 3 notes" in out
    assert db.declared_vector_dim(db.get_conn()) == 32
