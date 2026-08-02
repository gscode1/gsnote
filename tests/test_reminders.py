import asyncio
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from app.db import cursor, get_conn
from app.reminders import (
    cancel_reminder,
    compute_next_run_at,
    create_reminder,
    list_reminders,
    run_reminders,
)


def _reminder(
    kind="daily",
    user_id="alice",
    space="default",
    message="ping",
    weekday=None,
    fire_date=None,
    window_days=None,
    category=None,
    local_time="00:00",
    tz_name="UTC",
    window_mode=None,
    window_value=None,
    due=True,
) -> dict:
    rid = create_reminder(
        user_id, space, message, kind, weekday=weekday,
        fire_date=fire_date, window_days=window_days, category=category,
        local_time=local_time, tz_name=tz_name,
        window_mode=window_mode, window_value=window_value,
    )
    with cursor() as cur:
        if due:
            cur.execute(
                "UPDATE reminders SET next_run_at = ? WHERE id = ?",
                ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(), rid),
            )
    return dict(get_conn().execute("SELECT * FROM reminders WHERE id = ?", (rid,)).fetchone())


def _insert_note(
    content: str,
    source: str = "alice",
    days_ago: int = 0,
    space: str = "default",
    created_at: str | None = None,
) -> str:
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = created_at or (now - timedelta(days=days_ago)).isoformat()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at) "
            "VALUES (?, ?, 'idea', 3, ?, ?, ?, ?)",
            (note_id, content, source, space, created_at, created_at),
        )
    return note_id


def test_migration_defaults_existing_schedule_columns():
    reminder = _reminder(due=False)
    assert reminder["local_time"] == "00:00"  # explicit helper override
    assert reminder["timezone"] == "UTC"
    assert reminder["next_run_at"] is not None
    assert reminder["claim_token"] is None


def test_compute_next_run_at_converts_local_time_to_utc():
    reference = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    assert compute_next_run_at(
        "daily", "21:00", "Europe/Warsaw", ref_utc=reference
    ) == "2026-08-02T19:00:00+00:00"


def test_compute_next_run_at_handles_dst_gap_and_overlap():
    # 02:30 does not exist on the spring-forward day; run at the first valid minute.
    spring = compute_next_run_at(
        "daily", "02:30", "America/New_York",
        ref_utc=datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc),
    )
    assert spring == "2026-03-08T07:00:00+00:00"

    # 01:30 occurs twice on fall-back; choose the first occurrence (fold=0).
    fall = compute_next_run_at(
        "daily", "01:30", "America/New_York",
        ref_utc=datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc),
    )
    assert fall == "2026-11-01T05:30:00+00:00"


def test_window_modes_require_their_values():
    with pytest.raises(ValueError, match="rolling_hours requires"):
        _reminder(kind="daily", window_mode="rolling_hours", due=False)


def test_compute_next_run_at_handles_weekly_and_missed_once():
    reference = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)  # Tuesday
    weekly = compute_next_run_at(
        "weekly", "09:00", "UTC", weekday=0, ref_utc=reference
    )
    assert weekly == "2026-08-10T09:00:00+00:00"

    once = compute_next_run_at(
        "once", "09:00", "UTC", fire_date="2026-08-01", ref_utc=reference
    )
    assert once == "2026-08-01T09:00:00+00:00"


@pytest.mark.anyio
async def test_message_reminder_sends_once_per_occurrence():
    _reminder(kind="daily", message="drink water")
    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    now = datetime.now(timezone.utc)
    await run_reminders(fake_send, now_utc=now)
    await run_reminders(fake_send, now_utc=now)  # same occurrence: idempotent

    assert len(sent) == 1
    assert sent[0][0] == "alice" and "drink water" in sent[0][1]


@pytest.mark.anyio
async def test_previous_local_day_uses_reminder_timezone(monkeypatch):
    import app.reminders as mod

    fixed = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
    yesterday = _insert_note(
        "yesterday note", created_at="2026-08-01T19:30:00+00:00"
    )  # 21:30 on Aug 1 in Europe/Warsaw
    _insert_note("today note", created_at="2026-08-01T23:00:00+00:00")  # 01:00 Aug 2 local
    reminder = _reminder(
        kind="daily", window_mode="previous_local_day", tz_name="Europe/Warsaw",
        local_time="09:00", due=False,
    )
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET next_run_at = ? WHERE id = ?",
            ((fixed - timedelta(minutes=1)).isoformat(), reminder["id"]),
        )

    async def fake_phrase(notes):
        return " / ".join(n["content"] for n in notes)

    monkeypatch.setattr(mod, "phrase_nudge", fake_phrase)
    sent = []

    async def fake_send(user_id, message):
        sent.append(message)

    await run_reminders(fake_send, now_utc=fixed)
    assert len(sent) == 1
    assert "yesterday note" in sent[0]
    assert "today note" not in sent[0]
    assert get_conn().execute(
        "SELECT note_ids FROM notifications WHERE kind LIKE 'reminder:%'"
    ).fetchone()["note_ids"] == f'["{yesterday}"]'


@pytest.mark.anyio
async def test_rolling_hours_window_is_not_calendar_day(monkeypatch):
    import app.reminders as mod

    fixed = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    _insert_note("23 hours old", created_at="2026-08-01T13:00:00+00:00")
    _insert_note("25 hours old", created_at="2026-08-01T11:00:00+00:00")
    reminder = _reminder(
        kind="daily", window_mode="rolling_hours", window_value=24,
        local_time="09:00", due=False,
    )
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET next_run_at = ? WHERE id = ?",
            ((fixed - timedelta(minutes=1)).isoformat(), reminder["id"]),
        )

    async def fake_phrase(notes):
        return " / ".join(n["content"] for n in notes)

    monkeypatch.setattr(mod, "phrase_nudge", fake_phrase)
    sent = []

    async def fake_send(user_id, message):
        sent.append(message)

    await run_reminders(fake_send, now_utc=fixed)
    assert len(sent) == 1
    assert "23 hours old" in sent[0]
    assert "25 hours old" not in sent[0]


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
    assert result["fired"][0]["sent"] is True

    row = get_conn().execute("SELECT * FROM notifications WHERE kind LIKE 'reminder:%'").fetchone()
    assert row is not None


@pytest.mark.anyio
async def test_empty_window_sends_nothing_but_advances_schedule(monkeypatch):
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

    row = get_conn().execute(
        "SELECT last_fired_on, next_run_at FROM reminders WHERE id = ?", (r["id"],)
    ).fetchone()
    assert row["last_fired_on"] == date.today().isoformat()
    assert row["next_run_at"] > datetime.now(timezone.utc).isoformat()


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
    await run_reminders(fake_send)
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

    rows = get_conn().execute(
        "SELECT user_id, last_fired_on, claim_token FROM reminders WHERE deleted_at IS NULL"
    ).fetchall()
    by_user = {r["user_id"]: r for r in rows}
    assert by_user["alice"]["last_fired_on"] is None
    assert by_user["alice"]["claim_token"] is None
    assert by_user["bob"]["last_fired_on"] is not None


@pytest.mark.anyio
async def test_atomic_claim_prevents_duplicate_concurrent_delivery():
    _reminder(kind="daily", message="one copy")
    sent = []

    async def fake_send(user_id, message):
        await asyncio.sleep(0)
        sent.append((user_id, message))

    now = datetime.now(timezone.utc)
    await asyncio.gather(
        run_reminders(fake_send, now_utc=now),
        run_reminders(fake_send, now_utc=now),
    )
    assert sent == [("alice", "⏰ one copy")]


@pytest.mark.anyio
async def test_overdue_recurring_reminder_catches_up_once_and_advances():
    reminder = _reminder(kind="daily", message="catch up")
    now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET next_run_at = ? WHERE id = ?",
            ("2026-08-03T09:00:00+00:00", reminder["id"]),
        )

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    await run_reminders(fake_send, now_utc=now)
    row = get_conn().execute(
        "SELECT next_run_at FROM reminders WHERE id = ?", (reminder["id"],)
    ).fetchone()
    assert len(sent) == 1
    assert row["next_run_at"] > now.isoformat()


def test_cancel_reminder_refuses_other_users_id():
    r = _reminder(kind="daily", user_id="alice")
    assert cancel_reminder("bob", r["id"]) is False
    assert cancel_reminder("alice", r["id"]) is True
    assert list_reminders("alice") == []
