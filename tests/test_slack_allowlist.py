import pytest


def _slack_channel():
    from app.config import get_settings

    get_settings.cache_clear()
    from app.channels.slack import SlackChannel

    return SlackChannel, get_settings


def test_slack_requires_tokens(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "")
    monkeypatch.setenv("SLACK_APP_TOKEN", "")
    monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "U123")
    SlackChannel, get_settings = _slack_channel()

    with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
        SlackChannel()
    get_settings.cache_clear()


def test_slack_requires_allowlist(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-fake")
    monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "")
    SlackChannel, get_settings = _slack_channel()

    with pytest.raises(ValueError, match="SLACK_ALLOWED_USER_IDS"):
        SlackChannel()
    get_settings.cache_clear()


async def test_slack_nudge_buttons_require_notification_id(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-fake")
    monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "U123")
    SlackChannel, get_settings = _slack_channel()
    channel = SlackChannel()

    with pytest.raises(ValueError, match="notification_id"):
        await channel.send("U123", "hi", with_nudge_buttons=True)
    get_settings.cache_clear()
