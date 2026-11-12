from functools import lru_cache
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

    # --- HTTP API auth ---
    # ponytail: fail closed — empty token disables the data routes entirely
    api_token: str = ""

    # --- retrieval ---
    retrieval_candidate_n: int = 20
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # --- backup ---
    litestream_enabled: bool = False
    s3_bucket: str = ""
    s3_endpoint: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_region: str = "us-east-1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
