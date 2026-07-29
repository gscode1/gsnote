"""In-process embeddings via fastembed (ONNX, CPU). No network, no Ollama required."""
from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding

    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def embed(text: str) -> list[float]:
    settings = get_settings()
    vec = [float(x) for x in next(iter(_model().embed([text])))]
    if len(vec) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dim mismatch: model produced {len(vec)}, config expects {settings.embedding_dim}"
        )
    return vec


def embed_many(texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in v] for v in _model().embed(texts)]
