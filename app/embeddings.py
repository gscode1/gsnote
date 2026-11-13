"""Embeddings: in-process fastembed (ONNX, CPU) by default, or any OpenAI-compatible endpoint."""
import math
import re
from functools import lru_cache

import httpx

from app.config import get_settings


# bge v1.5 English models are trained to see this instruction before *queries* (not documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_V15_EN = re.compile(r"BAAI/bge-[a-z0-9]+-en-v1\.5", re.IGNORECASE)

# The e5 family (e5-*, multilingual-e5-*) is trained with "query: " / "passage: " markers on both sides.
E5_QUERY_PREFIX = "query: "
E5_DOCUMENT_PREFIX = "passage: "
_E5 = re.compile(r"intfloat/[a-z0-9-]*e5[a-z0-9-]*", re.IGNORECASE)


# Default absolute cosine floor for bge v1.5 English models (see app.retrieval._min_similarity).
BGE_MIN_SIMILARITY = 0.45


def is_bge_v15_en(model: str) -> bool:
    return _BGE_V15_EN.fullmatch(model.strip()) is not None


def is_e5(model: str) -> bool:
    return _E5.fullmatch(model.strip()) is not None


def query_prefix() -> str:
    """The prefix for query embeddings: the explicit setting, else auto-detected from the model."""
    settings = get_settings()
    if settings.embedding_query_prefix is not None:
        return settings.embedding_query_prefix
    if is_bge_v15_en(settings.embedding_model):
        return BGE_QUERY_PREFIX
    return E5_QUERY_PREFIX if is_e5(settings.embedding_model) else ""


def document_prefix() -> str:
    """The prefix for stored-note embeddings: the explicit setting, else auto-detected from the model."""
    settings = get_settings()
    if settings.embedding_document_prefix is not None:
        return settings.embedding_document_prefix
    return E5_DOCUMENT_PREFIX if is_e5(settings.embedding_model) else ""


@lru_cache
def _model():
    from fastembed import TextEmbedding

    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def _embed_api(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if not settings.embedding_base_url:
        raise ValueError("EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=api")
    headers = {"Authorization": f"Bearer {settings.embedding_api_key}"} if settings.embedding_api_key else {}
    resp = httpx.post(
        f"{settings.embedding_base_url.rstrip('/')}/embeddings",
        json={"model": settings.embedding_model, "input": texts},
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()
    # Servers may return data out of order; index is authoritative per the OpenAI spec.
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [[float(x) for x in d["embedding"]] for d in data]


def embed_many(texts: list[str], task: str | None = None) -> list[list[float]]:
    """Embed texts as given. `task` ("query" | "document" | None) only selects fastembed's
    task-specific entry point (jina-embeddings-v3 has separate query and passage adapters, and its
    plain embed() silently means "passage"); text prefixes are the callers' business."""
    settings = get_settings()
    if settings.embedding_provider == "api":
        vectors = _embed_api(texts)
    else:
        model = _model()
        run = {"query": model.query_embed, "document": model.passage_embed}.get(task, model.embed)
        vectors = [[float(x) for x in v] for v in run(texts)]
    for vec in vectors:
        if len(vec) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: model produced {len(vec)}, config expects {settings.embedding_dim}"
            )
    return [_unit(vec) for vec in vectors]


def _unit(vec: list[float]) -> list[float]:
    """Scale to unit length. sqlite-vec ranks by L2 distance, which only matches cosine ranking for
    unit vectors; some models (multilingual-e5, paraphrase-multilingual-mpnet) return unnormalised
    ones (norm ~29 and ~2.6 in fastembed). A no-op for bge/jina, whose vectors are already unit."""
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0.0 or abs(norm - 1.0) < 1e-6 else [x / norm for x in vec]


def embed(text: str) -> list[float]:
    return embed_many([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a search query (with the model's query prefix, if any). Stored notes use embed_document()."""
    return embed_many([query_prefix() + text], task="query")[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed several notes for storage in one call (document prefix applied to each)."""
    prefix = document_prefix()
    return embed_many([prefix + t for t in texts], task="document")


def embed_document(text: str) -> list[float]:
    """Embed a note for storage (with the model's document prefix, if any). Queries use embed_query()."""
    return embed_documents([text])[0]
