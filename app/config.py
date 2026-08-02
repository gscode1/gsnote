from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    db_path: str = "./data/gsnote.db"

    # --- LLM provider (BYOM via Pydantic AI) ---
    # openrouter | anthropic | <any OpenAI-compatible provider>
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    classifier_model: str = "openai/gpt-4o-mini"
    answer_model: str = "openai/gpt-4o"

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

    # --- channel ---
    channel: str = "telegram"  # telegram | slack | none
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""  # comma-separated; required when channel=telegram
    slack_bot_token: str = ""  # xoxb-...; required when channel=slack
    slack_app_token: str = ""  # xapp-... (Socket Mode); required when channel=slack
    slack_allowed_user_ids: str = ""  # comma-separated Slack user ids (U...); required when channel=slack

    # --- speech-to-text (voice input) ---
    stt_enabled: bool = False
    stt_base_url: str = ""  # OpenAI-compatible base, e.g. http://host:8000/v1
    stt_model: str = "whisper-large-v3-mlx"
    stt_api_key: str = ""  # optional; many local servers need none

    # --- resurfacing ---
    resurfacing_enabled: bool = True
    resurfacing_cron: str = "0 9 * * MON"  # weekly Monday 09:00
    resurfacing_budget: int = 3
    resurfacing_threshold: float = 0.6
    resurfacing_cooldown_days: int = 30

    # --- reminders ---
    reminder_cron: str = "* * * * *"  # fixed worker tick; reminder times live in SQLite

    # --- graph ---
    semantic_knn_k: int = 5
    semantic_similarity_threshold: float = 0.75
    temporal_proximity_hours: int = 24
    # RRF is rank-based and ignores edge weight magnitude, so weak edges (e.g. two notes
    # ~24h apart, near-zero temporal_proximity weight) must be filtered out here before they
    # get promoted to a full-strength rank-1 graph candidate at retrieval time.
    graph_min_edge_weight: float = 0.3

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

    @property
    def allowed_telegram_ids(self) -> set[int]:
        raw = self.telegram_allowed_user_ids.strip()
        if not raw:
            return set()
        return {int(x) for x in raw.split(",") if x.strip()}

    @property
    def allowed_slack_ids(self) -> set[str]:
        return {x.strip() for x in self.slack_allowed_user_ids.split(",") if x.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
