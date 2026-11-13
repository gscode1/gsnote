from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    db_path: str = "./data/gsnote.db"

    # --- embeddings ---
    # local = in-process fastembed/ONNX; api = any OpenAI-compatible /embeddings endpoint
    # (Ollama, TEI, vLLM, LiteLLM, OpenAI). "local model over HTTP" is just api + a localhost base_url.
    embedding_provider: str = "local"  # local | api
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_base_url: str = ""  # required when provider=api, e.g. http://ollama:11434/v1
    embedding_api_key: str = ""  # optional; many local servers need none
    # Prefix prepended to QUERIES only (never to stored notes). None = auto: the bge-v1.5 English
    # instruction for BAAI/bge-*-en-v1.5, nothing for other models. "" = explicitly no prefix.
    embedding_query_prefix: str | None = None
    # Same for stored NOTES only (never queries): None = auto ("passage: " for intfloat/*e5*, nothing
    # otherwise), "" = explicitly no prefix. Changing it on a populated DB needs `python -m app.reindex`.
    embedding_document_prefix: str | None = None

    # --- HTTP API auth ---
    # ponytail: fail closed — empty token disables the data routes entirely
    api_token: str = ""

    # --- time ---
    # IANA name; date-only since/until filters mean whole days in this timezone
    timezone: str = "UTC"

    # --- retrieval ---
    retrieval_candidate_n: int = 50
    retrieval_top_k: int = 8
    rrf_k: int = 60
    # Relevance gate (query search only): drop results that are not plausibly relevant, by cosine
    # similarity between query and note embeddings. RETRIEVAL_GATE=false turns it off entirely.
    retrieval_gate: bool = True
    # Absolute cosine floor. None = auto (0.45 for bge-*-en-v1.5, 0 = off for other models, whose
    # score scales differ). 0 disables.
    retrieval_min_similarity: float | None = None
    # Drop results whose similarity is more than this below the best match. 0 disables.
    retrieval_relative_gap: float = 0.2

    # --- backup ---
    litestream_enabled: bool = False
    s3_bucket: str = ""
    s3_endpoint: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (KeyError, ValueError):
            raise ValueError(f"unknown IANA timezone: {v!r}") from None
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
