"""Tests for the STT client (mocked HTTP — no real endpoint)."""
import httpx
import pytest

from app import stt
from app.config import get_settings


@pytest.fixture(autouse=True)
def _stt_config(monkeypatch):
    monkeypatch.setenv("STT_BASE_URL", "http://stt.test/v1")
    monkeypatch.setenv("STT_MODEL", "whisper-large-v3-mlx")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_transcribe_returns_text(monkeypatch):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": "  buy oat milk  "}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, headers=None, files=None, data=None):
            captured["url"] = url
            captured["model"] = data["model"]
            return FakeResp()

    monkeypatch.setattr(stt.httpx, "AsyncClient", FakeClient)

    text = await stt.transcribe(b"oggbytes")
    assert text == "buy oat milk"  # trimmed
    assert captured["url"] == "http://stt.test/v1/audio/transcriptions"
    assert captured["model"] == "whisper-large-v3-mlx"


@pytest.mark.anyio
async def test_transcribe_empty_raises(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": "   "}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(stt.httpx, "AsyncClient", FakeClient)

    with pytest.raises(stt.STTError):
        await stt.transcribe(b"oggbytes")


@pytest.mark.anyio
async def test_transcribe_unconfigured_raises(monkeypatch):
    monkeypatch.setenv("STT_BASE_URL", "")
    get_settings.cache_clear()
    with pytest.raises(stt.STTError):
        await stt.transcribe(b"x")
