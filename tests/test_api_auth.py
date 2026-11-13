import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_open_without_token(client):
    assert client.get("/health").status_code == 200


def test_data_routes_disabled_when_token_unset(client, monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    get_settings.cache_clear()
    assert client.get("/search", params={"q": "x"}).status_code == 503
    assert client.post("/capture", json={"content": "x"}).status_code == 503
    assert client.get("/spaces").status_code == 503
    assert client.get("/export").status_code == 503
    assert client.post("/import", json={"version": 2, "notes": []}).status_code == 503
    assert client.delete("/notes/x").status_code == 503


def test_missing_and_wrong_token_rejected(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    assert client.get("/search", params={"q": "x"}).status_code == 401
    assert client.get(
        "/search", params={"q": "x"}, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401
    assert client.get("/spaces").status_code == 401


def test_correct_token_allowed(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    resp = client.get(
        "/search", params={"q": "x"}, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"notes": []}


def test_search_endpoint_broad_param_controls_the_gate(client, monkeypatch):
    import app.main as main

    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    seen = []
    monkeypatch.setattr(main, "search", lambda *a, **kw: seen.append(kw["gate"]) or [])
    headers = {"Authorization": "Bearer s3cret"}
    assert client.get("/search", params={"q": "x"}, headers=headers).status_code == 200
    assert client.get("/search", params={"q": "x", "broad": "false"}, headers=headers).status_code == 200
    assert client.get("/search", params={"q": "x", "broad": "true"}, headers=headers).status_code == 200
    assert client.get("/search", params={"q": "x", "broad": "maybe"}, headers=headers).status_code == 422
    assert seen == [None, None, False]
