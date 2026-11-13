import pytest

from app import db


def test_schema_has_only_notes_tables():
    names = {r[0] for r in db.get_conn().execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"notes", "notes_fts", "note_vectors", "schema_migrations"} <= names
    assert not names & {"edges", "notifications", "user_settings"}
    cols = {r[1] for r in db.get_conn().execute("PRAGMA table_info(notes)")}
    assert cols == {"id", "content", "category", "source", "space", "created_at", "updated_at"}


def test_run_migrations_is_idempotent():
    db.run_migrations()
    db.run_migrations()
    versions = [r[0] for r in db.get_conn().execute("SELECT version FROM schema_migrations")]
    assert versions == ["001_init", "002_meta"]


def test_legacy_bot_database_is_rejected():
    conn = db.get_conn()
    for v in ("002_spaces", "011_schedule_reviews_and_observability"):
        conn.execute("INSERT INTO schema_migrations VALUES (?, '2024-01-01')", (v,))
    conn.commit()

    with pytest.raises(RuntimeError) as exc:
        db.run_migrations()
    msg = str(exc.value)
    assert "v0-bot" in msg and "GET /export" in msg and "DB_PATH" in msg and "gsnote import" in msg
    assert "002_spaces" in msg


def test_legacy_guard_fires_on_startup(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    db.get_conn().execute("INSERT INTO schema_migrations VALUES ('002_spaces', '2024-01-01')")
    db.get_conn().commit()
    with pytest.raises(RuntimeError, match="v0-bot"):
        with TestClient(app):
            pass
