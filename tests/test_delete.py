import pytest
from fastapi.testclient import TestClient

from app import capture
from app.config import get_settings
from app.db import get_conn
from app.main import app
from app.retrieval import search

AUTH = {"Authorization": "Bearer s3cret"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


def _count(table: str, col: str, note_id: str) -> int:
    return get_conn().execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (note_id,)).fetchone()[0]


def test_delete_note_removes_everywhere():
    keep = capture.capture_note("quarterly roadmap planning for engineering")
    gone = capture.capture_note("quarterly roadmap planning for sales")

    assert capture.delete_note(gone["id"]) is True
    assert capture.delete_note(gone["id"]) is False

    assert _count("notes", "id", gone["id"]) == 0
    assert _count("notes_fts", "note_id", gone["id"]) == 0
    assert _count("note_vectors", "note_id", gone["id"]) == 0
    ids = [n["id"] for n in search("quarterly roadmap planning")]
    assert ids == [keep["id"]]


def test_delete_endpoint(client):
    note = capture.capture_note("temporary secret note")
    assert client.delete(f"/notes/{note['id']}").status_code == 401

    resp = client.delete(f"/notes/{note['id']}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": note["id"]}
    assert client.get("/search", headers=AUTH, params={"q": "temporary secret"}).json() == {"notes": []}
    assert client.delete(f"/notes/{note['id']}", headers=AUTH).status_code == 404
