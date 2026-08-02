import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from app.db import cursor, get_conn
from app.reminders import _is_due, cancel_reminder, create_reminder, list_reminders, run_reminders


def _reminder(
    kind="daily",
    user_id="alice",
    space="default",
    message="ping",
    weekday=None,
    fire_date=None,
    window_days=None,
    category=None,
    last_fired_on=None,
) -> dict:
    rid = create_reminder(
        user_id, space, message, kind, weekday=weekday,
        fire_date=fire_date, window_days=window_days, category=category,
    )
    if last_fired_on:
        with cursor() as cur:
            cur.execute("UPDATE reminders SET last_fired_on = ? WHERE id = ?", (last_fired_on, rid))
    return dict(get_conn().execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone())


def _insert_note(content: str, source: str = "alice", days_ago: int = 0, space: str = "default") -> str:
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=days_ago)).isoformat()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at) "
            "VALUES (?, ?, 'idea', 3, ?, ?, ?, ?)",
            (note_id, content, source, space, created_at, created_at),
        )
    return note_id


def test_due_rules():
    today = date(2026, 8, 4)  # a Tuesday, weekday()==1
    assert _is_due(_reminder(kind="daily"), today)
    assert _is_due(_reminder(kind="weekly", weekday=1), today)
    assert not _is_due(_reminder(kind="weekly", weekday=2), today)
    assert _is_due(_reminder(kind="once", fire_date="2026-08-04"), today)
    assert _is_due(_reminder(kind="once", fire_date="2026-08-01"), today)  # missed day fires late
    assert not _is_due(_reminder(kind="once", fire_date="2026-08-05"), today)
    assert not _is_due(_reminder(kind="daily", last_fired_on=today.isoformat()), today)


@pytest.mark.anyio
async def test_message_reminder_sends_once_per_day():
    _reminder(kind="daily", message="drink water")
    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    await run_reminders(fake_send)
    await run_reminders(fake_send)  # same day: idempotent

    assert len(sent) == 1
    assert sent[0][0] == "alice" and "drink water" in sent[0][1]


@pytest.mark.anyio
async def test_query_reminder_scopes_to_owner_and_window(monkeypatch):
    import app.reminders as mod

    _insert_note("alice fresh idea", source="alice", days_ago=2)
    _insert_note("bob idea", source="bob", days_ago=2)
    _insert_note("alice stale idea", source="alice", days_ago=30)
    _reminder(kind="daily", user_id="alice", window_days=7, category="idea")

    async def fake_phrase(notes):
        return " / ".join(n["content"] for n in notes)

    monkeypatch.setattr(mod, "phrase_nudge", fake_phrase)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    result = await run_reminders(fake_send)
    assert len(sent) == 1
    assert sent[0][0] == "alice"
    assert "alice fresh idea" in sent[0][1]
    assert "bob idea" not in sent[0][1] and "alice stale idea" not in sent[0][1]

    # Fired notes are logged into notifications (enters resurfacing's cooldown set).
    row = get_conn().execute(
        "SELECT * FROM notifications WHERE kind LIKE 'reminder:%'"
    ).fetchone()
    assert row is not None


@pytest.mark.anyio
async def test_empty_window_sends_nothing_but_marks_fired(monkeypatch):
    import app.reminders as mod

    async def fake_phrase(notes):
        return "should not be called"

    monkeypatch.setattr(mod, "phrase_nudge", fake_phrase)
    r = _reminder(kind="daily", window_days=7)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    result = await run_reminders(fake_send)
    assert sent == []
    assert result["fired"][0]["sent"] is False

    row = get_conn().execute("SELECT last_fired_on FROM reminders WHERE id = ?", (r["id"],)).fetchone()
    assert row["last_fired_on"] == date.today().isoformat()


@pytest.mark.anyio
async def test_once_reminder_is_consumed_after_firing():
    r = _reminder(kind="once", fire_date=date.today().isoformat(), message="one shot")
    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    await run_reminders(fake_send)
    assert len(sent) == 1

    row = get_conn().execute("SELECT deleted_at FROM reminders WHERE id = ?", (r["id"],)).fetchone()
    assert row["deleted_at"] is not None

    await run_reminders(fake_send)  # consumed: never fires again, even on later days
    assert len(sent) == 1


@pytest.mark.anyio
async def test_failed_send_does_not_block_others_and_retries_next_tick():
    _reminder(kind="daily", user_id="alice", message="first")
    _reminder(kind="daily", user_id="bob", message="second")

    sent = []

    async def flaky_send(user_id, message):
        if user_id == "alice":
            raise RuntimeError("channel down")
        sent.append((user_id, message))

    result = await run_reminders(flaky_send)
    assert sent == [("bob", "⏰ second")]
    assert result["fired"][0]["sent"] is True

    # alice's reminder is unmarked -> fires on the next tick
    rows = get_conn().execute(
        "SELECT user_id, last_fired_on FROM reminders WHERE deleted_at IS NULL"
    ).fetchall()
    by_user = {r["user_id"]: r["last_fired_on"] for r in rows}
    assert by_user["alice"] is None
    assert by_user["bob"] is not None


def test_cancel_reminder_refuses_other_users_id():
    r = _reminder(kind="daily", user_id="alice")
    assert cancel_reminder("bob", r["id"]) is False
    assert cancel_reminder("alice", r["id"]) is True
    assert list_reminders("alice") == []
