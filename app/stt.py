"""Speech-to-text via an OpenAI-compatible /audio/transcriptions endpoint.

Used for Telegram voice notes: transcribe audio -> text, then the transcript flows
through the normal message pipeline. The endpoint is configurable (e.g. a local
whisper-large-v3 MLX server); no provider lock-in.
"""
import httpx

from app.config import get_settings


class STTError(RuntimeError):
    pass


async def transcribe(audio: bytes, filename: str = "voice.ogg", content_type: str = "audio/ogg") -> str:
    """Transcribe audio bytes to text. Raises STTError on failure."""
    settings = get_settings()
    if not settings.stt_base_url:
        raise STTError("STT is not configured (stt_base_url is empty).")

    url = settings.stt_base_url.rstrip("/") + "/audio/transcriptions"
    headers = {}
    if settings.stt_api_key:
        headers["Authorization"] = f"Bearer {settings.stt_api_key}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                headers=headers,
                files={"file": (filename, audio, content_type)},
                data={"model": settings.stt_model},
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise STTError(f"STT request failed: {e}") from e

    text = (resp.json().get("text") or "").strip()
    if not text:
        raise STTError("STT returned no transcript.")
    return text
