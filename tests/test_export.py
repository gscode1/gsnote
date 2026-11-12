import json

import pytest
from fastapi.testclient import TestClient

from app import capture, db
from app.config import get_settings
from app.db import get_conn
from app.main import app

AUTH = {"Authorization": "Bearer s3cret"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


def _export(client) -> dict:
    return json.loads(client.get("/export", headers=AUTH).content)


def _fresh_db(tmp_path):
    db.get_conn().close()
    db.reset_db_for_tests(str(tmp_path / "other.db"))


def test_export_requires_token(client):
    assert client.get("/export").status_code == 401
    assert client.get("/export", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_export_v2_format(client):
    note = capture.capture_note("remember the milk", category="todo", space="work", source="cli")

    resp = client.get("/export", headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert "attachment" in resp.headers["content-disposition"]

    payload = json.loads(resp.content)
    assert payload["version"] == 2
    assert payload["exported_at"]
    stored = dict(get_conn().execute(
        "SELECT id, content, category, source, space, created_at, updated_at FROM notes"
    ).fetchone())
    assert payload["notes"] == [stored]
    assert stored["id"] == note["id"]
    assert set(stored) == {"id", "content", "category", "source", "space", "created_at", "updated_at"}


def test_export_includes_all_spaces_and_sources(client):
    capture.capture_note("cli note", source="cli", space="default")
    capture.capture_note("work note", source="claude-code", space="work")
    assert len(_export(client)["notes"]) == 2


def test_round_trip_export_import_into_fresh_db(client, tmp_path):
    a = capture.capture_note("alpha note about gardening", category="idea", space="home", source="cli")
    b = capture.capture_note("beta note about budgets", space="work")
    payload = _export(client)

    _fresh_db(tmp_path)
    with TestClient(app) as c2:
        resp = c2.post("/import", headers=AUTH, json=payload)
        assert resp.status_code == 200
        assert resp.json() == {"imported": 2, "skipped": 0}
        assert _export(c2)["notes"] == payload["notes"]  # ids, created_at, space, category preserved

        hits = c2.get("/search", headers=AUTH, params={"q": "gardening", "space": "home"}).json()["notes"]
        assert [n["id"] for n in hits] == [a["id"]]  # re-embedded + FTS indexed
        assert b["id"] in {n["id"] for n in c2.get("/search", headers=AUTH, params={"q": "budgets", "space": "work"}).json()["notes"]}

        # importing again skips everything
        assert c2.post("/import", headers=AUTH, json=payload).json() == {"imported": 0, "skipped": 2}


def test_import_v1_ignores_old_fields_and_skips_soft_deleted(client):
    payload = {
        "version": 1,
        "notes": [
            {"id": "v1-a", "content": "old bot note", "category": "Idea", "importance": 5,
             "source": "telegram", "space": "Work", "created_at": "2024-01-01T00:00:00+00:00",
             "updated_at": "2024-01-02T00:00:00+00:00", "last_accessed_at": "2024-02-01", "access_count": 9},
            {"id": "v1-b", "content": "deleted one", "category": "note", "importance": 3,
             "source": "user", "space": "default", "created_at": "2024-01-01T00:00:00+00:00",
             "updated_at": "2024-01-01T00:00:00+00:00", "deleted_at": "2024-03-01T00:00:00+00:00"},
            {"id": "v1-c", "content": "bad values", "category": "!!", "source": "user", "space": "??",
             "created_at": "2024-01-03T00:00:00+00:00", "updated_at": "2024-01-03T00:00:00+00:00"},
        ],
    }
    resp = client.post("/import", headers=AUTH, json=payload)
    assert resp.json() == {"imported": 2, "skipped": 1}

    notes = {n["id"]: n for n in _export(client)["notes"]}
    assert set(notes) == {"v1-a", "v1-c"}
    assert notes["v1-a"] == {
        "id": "v1-a", "content": "old bot note", "category": "idea", "source": "telegram",
        "space": "work", "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-02T00:00:00+00:00",
    }
    assert (notes["v1-c"]["category"], notes["v1-c"]["space"]) == ("note", "default")


@pytest.mark.parametrize("payload", [{"version": 3, "notes": []}, {"notes": []}, {"version": 2}])
def test_import_rejects_unknown_version_or_shape(client, payload):
    assert client.post("/import", headers=AUTH, json=payload).status_code == 422
    with pytest.raises(ValueError):
        capture.import_notes(payload)


def test_import_requires_token(client):
    assert client.post("/import", json={"version": 2, "notes": []}).status_code == 401
