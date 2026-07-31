"""Embeddings: in-process fastembed (ONNX, CPU) by default, or any OpenAI-compatible endpoint."""
from functools import lru_cache

import httpx

from app.config import get_settings


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
