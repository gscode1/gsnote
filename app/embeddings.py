"""Embeddings: in-process fastembed (ONNX, CPU) by default, or any OpenAI-compatible endpoint."""
import re
from functools import lru_cache

import httpx

from app.config import get_settings


# bge v1.5 English models are trained to see this instruction before *queries* (not documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_V15_EN = re.compile(r"BAAI/bge-[a-z0-9]+-en-v1\.5", re.IGNORECASE)


# Default absolute cosine floor for bge v1.5 English models (see app.retrieval._min_similarity).
BGE_MIN_SIMILARITY = 0.45


def is_bge_v15_en(model: str) -> bool:
    return _BGE_V15_EN.fullmatch(model.strip()) is not None


def query_prefix() -> str:
    """The prefix for query embeddings: the explicit setting, else auto-detected from the model."""
    settings = get_settings()
    if settings.embedding_query_prefix is not None:
        return settings.embedding_query_prefix
    return BGE_QUERY_PREFIX if is_bge_v15_en(settings.embedding_model) else ""


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


def embed_many(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if settings.embedding_provider == "api":
        vectors = _embed_api(texts)
    else:
        vectors = [[float(x) for x in v] for v in _model().embed(texts)]
    for vec in vectors:
        if len(vec) != settings.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: model produced {len(vec)}, config expects {settings.embedding_dim}"
            )
    return vectors


def embed(text: str) -> list[float]:
    return embed_many([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a search query (with the model's query prefix, if any). Stored notes use embed()."""
    return embed(query_prefix() + text)
