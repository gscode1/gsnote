"""Tests for named note spaces."""
import pytest
from fastapi.testclient import TestClient

from app import capture, spaces
from app.config import get_settings
from app.main import app
from app.retrieval import search


def test_normalize_space():
    assert spaces.normalize_space("  Side Project_X ") == "side-project-x"
    with pytest.raises(ValueError):
        spaces.normalize_space("!!bad name!!")


def test_scope_filter():
    assert spaces.scope_filter(None) == ("", [])
    assert spaces.scope_filter("work") == (" AND space = ?", ["work"])
    assert spaces.scope_filter("work", alias="n") == (" AND n.space = ?", ["work"])


def test_list_spaces_counts_sorted():
    assert spaces.list_spaces() == []
    capture.capture_note("work thing", space="work")
    capture.capture_note("another work thing", space="work")
    capture.capture_note("hobby thing", space="hobby")
    gone = capture.capture_note("deleted thing", space="trash")
    capture.delete_note(gone["id"])

    assert spaces.list_spaces() == [
        {"space": "hobby", "notes": 1},
        {"space": "work", "notes": 2},
    ]


def test_spaces_endpoint_requires_token_and_lists(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    capture.capture_note("a", space="work")
    capture.capture_note("b", space="work")
    capture.capture_note("c")
    with TestClient(app) as client:
        assert client.get("/spaces").status_code == 401
        resp = client.get("/spaces", headers={"Authorization": "Bearer s3cret"})
        assert resp.status_code == 200
        assert resp.json() == [{"space": "default", "notes": 1}, {"space": "work", "notes": 2}]


def test_custom_space_isolates_notes():
    capture.capture_note("hobby guitar practice", space="hobby")
    capture.capture_note("hobby project deadline", space="work")

    hobby_hits = [n["content"] for n in search("hobby", space="hobby")]
    assert hobby_hits == ["hobby guitar practice"]


def test_search_is_space_scoped():
    capture.capture_note("quarterly budget planning deck", space="work")
    capture.capture_note("plan a weekend hiking trip", space="personal")

    work_hits = [n["content"] for n in search("plan", space="work")]
    personal_hits = [n["content"] for n in search("plan", space="personal")]

    assert any("budget" in c for c in work_hits)
    assert all("hiking" not in c for c in work_hits)  # personal note must not leak into work
    assert any("hiking" in c for c in personal_hits)
    assert all("budget" not in c for c in personal_hits)


def test_search_across_all_spaces_when_none():
    capture.capture_note("quarterly budget planning deck", space="work")
    capture.capture_note("plan a weekend hiking trip", space="personal")

    all_hits = [n["content"] for n in search("plan", space=None)]
    assert any("budget" in c for c in all_hits)
    assert any("hiking" in c for c in all_hits)
