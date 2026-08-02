import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app import reminders
from app.db import cursor, get_conn
from app.turn import memory_agent


def _insert_note(
    content: str,
    source: str = "alice",
    days_ago: int = 0,
    space: str = "default",
    category: str = "idea",
) -> str:
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=days_ago)).isoformat()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at) "
            "VALUES (?, ?, ?, 3, ?, ?, ?, ?)",
            (note_id, content, category, source, space, created_at, created_at),
        )
    return note_id


def test_action_type_migration_defaults_to_notify():
    sid = reminders.create_schedule(
        "alice", "default", "drink water", "daily", action_type="notify"
    )
    row = dict(get_conn().execute("SELECT * FROM reminders WHERE id = ?", (sid,)).fetchone())
    assert row["action_type"] == "notify"


def test_invalid_action_type_raises_value_error():
    with pytest.raises(ValueError, match="action_type must be one of"):
        reminders.create_schedule(
            "alice", "default", "invalid action", "daily", action_type="unknown"
        )


def test_digest_schedule_stores_window_semantics_and_digest_action():
    sid = reminders.create_digest(
        "alice", "work", "weekly recap", "weekly", weekday=0,
        window_mode="rolling_days", window_days=7, category="task"
    )
    row = dict(get_conn().execute("SELECT * FROM reminders WHERE id = ?", (sid,)).fetchone())
    assert row["action_type"] == "digest"
    assert row["space"] == "work"
    assert row["window_mode"] == "rolling_days"
    assert row["window_days"] == 7
    assert row["category"] == "task"


@pytest.mark.anyio
async def test_digest_worker_uses_reporting_summarizer_not_nudge(monkeypatch):
    _insert_note("finish architecture doc", source="alice", space="work")
    sid = reminders.create_digest(
        "alice", "work", "weekly doc recap", "daily", window_mode="rolling_days", window_days=7
    )
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET next_run_at = ? WHERE id = ?",
            ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(), sid),
        )

    summarizer_called = []

    async def fake_summarize(notes, window_desc=""):
        summarizer_called.append(notes)
        return "Summary of " + ", ".join(n["content"] for n in notes)

    async def fake_nudge(notes):
        raise AssertionError("phrase_nudge should not be called for digest schedules")

    monkeypatch.setattr(reminders, "summarize_notes", fake_summarize)
    monkeypatch.setattr(reminders, "phrase_nudge", fake_nudge)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    result = await reminders.run_reminders(fake_send)
    assert len(summarizer_called) == 1
    assert len(sent) == 1
    assert sent[0][0] == "alice"
    assert "📊 Digest (weekly doc recap):" in sent[0][1]
    assert "finish architecture doc" in sent[0][1]
    assert result["fired"][0]["sent"] is True


def test_legacy_reminder_tool_names_not_exposed_on_agent():
    agent = memory_agent()
    # Check registered tools on the memory agent
    tool_names = [t.name for t in agent._function_toolset.tools.values()]
    assert "create_schedule" in tool_names
    assert "create_digest" in tool_names
    assert "list_schedules" in tool_names
    assert "cancel_schedule" in tool_names

    assert "create_reminder" not in tool_names
    assert "list_reminders" not in tool_names
    assert "cancel_reminder" not in tool_names
