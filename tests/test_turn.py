"""Conversation turn module — history, commands, digest → button response seam."""
import json
import uuid
from datetime import datetime, timezone

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app import capture
from app.agents import Classification, memory_agent
from app.channels import Channel
from app.db import cursor, get_conn
from app.resurfacing import Digest
from app.spaces import get_space
from app.turn import (
    _history_by_user,
    handle_command,
    handle_message,
    handle_response,
    make_digest_sender,
)


class FakeChannel(Channel):
    def __init__(self, recipients: list[str] | None = None) -> None:
        self.sent: list[tuple] = []
        self._recipients = recipients or []

    async def send(
        self,
        user_id: str,
        message: str,
        *,
        with_nudge_buttons: bool = False,
        notification_id: str | None = None,
    ) -> None:
        self.sent.append((user_id, message, with_nudge_buttons, notification_id))

    def recipients(self) -> list[str]:
        return list(self._recipients)

    def on_message(self, handler) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _mock_classifier(monkeypatch):
    async def fake_classify(content: str) -> Classification:
        return Classification(category="note", importance=2, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)


@pytest.mark.asyncio
async def test_space_switch_clears_history():
    _history_by_user["u1"] = ["prior turn"]
    reply = await handle_command("u1", "work", "")
    assert "work" in reply
    assert get_space("u1") == "work"
    assert "u1" not in _history_by_user


@pytest.mark.asyncio
async def test_handle_response_records_against_notification_id():
    notif_id = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) VALUES (?, ?, 'digest', 'telegram', ?)",
            (notif_id, json.dumps([]), datetime.now(timezone.utc).isoformat()),
        )
    await handle_response("u1", "dismissed", notif_id)
    row = get_conn().execute(
        "SELECT user_response FROM notifications WHERE id = ?", (notif_id,)
    ).fetchone()
    assert row["user_response"] == "dismissed"


@pytest.mark.asyncio
async def test_digest_sender_pushes_buttons_with_notification_id():
    channel = FakeChannel(recipients=["42"])
    send_fn = make_digest_sender(channel)
    digest = Digest(message="nudge", notification_id="nid-1")
    await send_fn(digest)

    assert channel.sent == [("42", "nudge", True, "nid-1")]


@pytest.mark.anyio
async def test_handle_message_runs_agent_and_replies(monkeypatch):
    channel = FakeChannel()

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("got it")])

    agent = memory_agent()
    with agent.override(model=FunctionModel(fn)):
        await handle_message(channel, "u1", "hello")

    assert channel.sent == [("u1", "got it", False, None)]
    assert "u1" in _history_by_user
