import os

import pytest


def test_telegram_requires_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.channels.telegram import TelegramChannel

    with pytest.raises(ValueError, match="TELEGRAM_ALLOWED_USER_IDS"):
        TelegramChannel()
    get_settings.cache_clear()
