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
            {"index": 1, "embedding": [0, 1, 0]},
            {"index": 0, "embedding": [1, 0, 0]},
        ]},
    )

    assert embeddings.embed_many(["a", "b"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
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


# --- query prefix ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "baai/BGE-Large-EN-V1.5"],
)
def test_query_prefix_auto_detects_bge_v15_english(monkeypatch, model):
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    get_settings.cache_clear()
    assert embeddings.query_prefix() == embeddings.BGE_QUERY_PREFIX


@pytest.mark.parametrize(
    "model",
    ["nomic-embed-text", "BAAI/bge-m3", "BAAI/bge-small-en", "BAAI/bge-small-zh-v1.5",
     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
)
def test_query_prefix_auto_is_empty_for_other_models(monkeypatch, model):
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    get_settings.cache_clear()
    assert embeddings.query_prefix() == ""


def test_query_prefix_explicit_value_wins(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    get_settings.cache_clear()
    captured = _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    embeddings.embed_query("hello")
    assert captured["json"]["input"] == ["query: hello"]


def test_query_prefix_explicit_empty_disables_autodetect(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    get_settings.cache_clear()
    assert embeddings.query_prefix() == ""
    captured = _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    embeddings.embed_query("hello")
    assert captured["json"]["input"] == ["hello"]


def test_embed_query_prefixes_but_embed_does_not(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    get_settings.cache_clear()
    captured = _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    embeddings.embed_query("hello")
    assert captured["json"]["input"] == [embeddings.BGE_QUERY_PREFIX + "hello"]
    embeddings.embed("hello")
    assert captured["json"]["input"] == ["hello"]


# --- document prefix / e5 family ------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["intfloat/multilingual-e5-large", "intfloat/multilingual-e5-small", "intfloat/e5-base-v2", "Intfloat/E5-Large"],
)
def test_e5_family_auto_detects_both_prefixes(monkeypatch, model):
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    get_settings.cache_clear()
    assert embeddings.query_prefix() == "query: "
    assert embeddings.document_prefix() == "passage: "


@pytest.mark.parametrize(
    "model",
    ["BAAI/bge-large-en-v1.5", "BAAI/bge-m3", "jinaai/jina-embeddings-v3", "nomic-embed-text",
     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "someone/not-e5-but-intfloat"],
)
def test_document_prefix_auto_is_empty_for_other_models(monkeypatch, model):
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    get_settings.cache_clear()
    assert embeddings.document_prefix() == ""


def test_document_prefix_explicit_value_and_empty_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "doc: ")
    get_settings.cache_clear()
    assert embeddings.document_prefix() == "doc: "
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "")
    get_settings.cache_clear()
    assert embeddings.document_prefix() == ""
    assert embeddings.query_prefix() == "query: "  # independent of the document setting


def test_embed_document_prefixes_notes_but_not_queries(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    get_settings.cache_clear()
    captured = _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2, 3]}]})

    embeddings.embed_document("hello")
    assert captured["json"]["input"] == ["passage: hello"]
    embeddings.embed_query("hello")
    assert captured["json"]["input"] == ["query: hello"]
    embeddings.embed("hello")
    assert captured["json"]["input"] == ["hello"]


class _FakeLocalModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(("embed", list(texts)))
        return [[0.0, 0.0, 1.0] for _ in texts]

    def query_embed(self, texts):
        self.calls.append(("query_embed", list(texts)))
        return [[0.0, 0.0, 1.0] for _ in texts]

    def passage_embed(self, texts):
        self.calls.append(("passage_embed", list(texts)))
        return [[0.0, 0.0, 1.0] for _ in texts]


def test_local_provider_uses_task_specific_fastembed_entry_points(monkeypatch):
    """jina-embeddings-v3 has query/passage adapters; fastembed's plain embed() means passage."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
    get_settings.cache_clear()
    fake = _FakeLocalModel()
    monkeypatch.setattr(embeddings, "_model", lambda: fake)

    embeddings.embed_query("q")
    embeddings.embed_document("d")
    embeddings.embed("raw")
    assert fake.calls == [("query_embed", ["q"]), ("passage_embed", ["d"]), ("embed", ["raw"])]


def test_vectors_are_normalised_to_unit_length(monkeypatch):
    """L2 KNN equals cosine ranking only for unit vectors; unit input is returned untouched."""
    _fake_post(monkeypatch, {"data": [{"index": 0, "embedding": [30, 0, 40]}, {"index": 1, "embedding": [1, 0, 0]}]})

    scaled, unit = embeddings.embed_many(["a", "b"])
    assert scaled == pytest.approx([0.6, 0.0, 0.8])
    assert unit == [1.0, 0.0, 0.0]
