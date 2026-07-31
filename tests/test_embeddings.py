"""Tests for the api embedding provider (mocked HTTP — no real endpoint)."""
import pytest

from app import embeddings
from app.config import get_settings


@pytest.fixture(autouse=True)
def _api_config(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embed.test/v1/")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("EMBEDDING_DIM", "3")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_post(monkeypatch, payload):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return FakeResp()

    monkeypatch.setattr(embeddings.httpx, "post", post)
    return captured


def test_api_provider_posts_and_orders_by_index(monkeypatch):
    captured = _fake_post(
        monkeypatch,
        {"data": [
            {"index": 1, "embedding": [4, 5, 6]},
            {"index": 0, "embedding": [1, 2, 3]},
        ]},
    )

    assert embeddings.embed_many(["a", "b"]) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert captured["url"] == "http://embed.test/v1/embeddings"
    assert captured["json"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert captured["headers"] == {}  # no key configured -> no Authorization


def test_api_key_sets_auth_header(monkeypatch):
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    get_settings.cache_clear()
    captured = _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    embeddings.embed("a")
    assert captured["headers"] == {"Authorization": "Bearer sk-test"}


def test_dim_mismatch_raises(monkeypatch):
    _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2]}]})

    with pytest.raises(ValueError, match="dim mismatch"):
        embeddings.embed("a")


def test_api_without_base_url_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="EMBEDDING_BASE_URL"):
        embeddings.embed("a")
