import pytest
from fastapi.testclient import TestClient

from app import capture
from app.config import get_settings
from app.db import get_conn
from app.main import app


def test_capture_note_stores_everywhere():
    note = capture.capture_note("  Build a note app with great recall  ", source="alice")

    assert note["category"] == "note"
    assert note["space"] == "default"
    assert note["content"] == "Build a note app with great recall"

    conn = get_conn()

    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note["id"],)).fetchone()
    assert row is not None
    assert row["content"] == "Build a note app with great recall"
    assert row["source"] == "alice"

    fts_row = conn.execute("SELECT * FROM notes_fts WHERE note_id = ?", (note["id"],)).fetchone()
    assert fts_row is not None

    vec_row = conn.execute("SELECT * FROM note_vectors WHERE note_id = ?", (note["id"],)).fetchone()
    assert vec_row is not None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Idea", "idea"),
        ("  Meeting Notes ", "meeting-notes"),
        ("to_do", "to-do"),
        ("x" * 32, "x" * 32),
    ],
)
def test_category_is_normalized(raw, expected):
    assert capture.capture_note("hello", category=raw)["category"] == expected


@pytest.mark.parametrize("bad", ["", "   ", "-lead", "!!", "x" * 33, "naïve"])
def test_invalid_category_rejected(bad):
    with pytest.raises(ValueError):
        capture.capture_note("hello", category=bad)


@pytest.mark.parametrize("bad", ["", "!!bad name!!", "y" * 33])
def test_invalid_space_rejected(bad):
    with pytest.raises(ValueError):
        capture.capture_note("hello", space=bad)


@pytest.mark.parametrize("empty", ["", "   \n\t"])
def test_empty_content_rejected(empty):
    with pytest.raises(ValueError):
        capture.capture_note(empty)
    assert get_conn().execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0


def test_space_is_normalized():
    assert capture.capture_note("hello", space="  Side Project_X ")["space"] == "side-project-x"


def test_capture_works_with_no_third_party_env(monkeypatch):
    # Nothing but the embedding/db defaults from conftest: no third-party keys at all.
    for key in ("API_TOKEN", "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    assert capture.capture_note("works offline")["content"] == "works offline"


def test_capture_endpoint_category_and_validation(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    auth = {"Authorization": "Bearer s3cret"}
    with TestClient(app) as client:
        ok = client.post("/capture", headers=auth, json={"content": "buy milk", "category": "To Do"})
        assert ok.status_code == 200
        assert ok.json()["category"] == "to-do"

        assert client.post("/capture", headers=auth, json={"content": "x", "category": "!!"}).status_code == 422
        assert client.post("/capture", headers=auth, json={"content": "x", "space": "!!"}).status_code == 422
        assert client.post("/capture", headers=auth, json={"content": "   "}).status_code == 422
