import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import get_settings
from app.main import app

AUTH = {"Authorization": "Bearer s3cret"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


def _insert_note(note_id: str, content: str, **overrides) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    note = {
        "id": note_id,
        "content": content,
        "category": "note",
        "importance": 3,
        "source": "user",
        "space": "personal",
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": None,
        "access_count": 0,
        **overrides,
    }
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space,"
            " created_at, updated_at, last_accessed_at, access_count)"
            " VALUES (:id, :content, :category, :importance, :source, :space,"
            " :created_at, :updated_at, :last_accessed_at, :access_count)",
            note,
        )
    return note


def test_export_requires_token(client):
    assert client.get("/export").status_code == 401
    assert client.get("/export", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_export_matches_stored_notes(client):
    note = _insert_note("n1", "remember the milk", category="todo", space="work")
    _insert_note("n2", "soft deleted", deleted_at=None)
    with db.cursor() as cur:
        cur.execute("UPDATE notes SET deleted_at = ? WHERE id = 'n2'", (note["created_at"],))

    resp = client.get("/export", headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert "attachment" in resp.headers["content-disposition"]

    payload = json.loads(resp.content)
    assert payload["version"] == 1
    assert payload["exported_at"]
    with db.cursor() as cur:
        stored = dict(
            cur.execute(
                "SELECT id, content, category, importance, source, space,"
                " created_at, updated_at, last_accessed_at, access_count"
                " FROM notes WHERE id = 'n1'"
            ).fetchone()
        )
    assert payload["notes"] == [stored]  # export round-trips the stored record exactly
    assert "n2" not in {n["id"] for n in payload["notes"]}  # soft-deleted excluded
