"""Daily commitment briefing (#34): due-date capture, opt-in persistence, delivery."""
import json
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from app import briefing, capture
from app.briefing import briefing_enabled, notes_due_today, run_briefing, set_briefing_enabled
from app.capture import Classification
from app.db import cursor, get_conn


def _insert_note(
    content: str,
    source: str = "alice",
    space: str = "default",
    due_date: str | None = None,
    deleted: bool = False,
) -> str:
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, due_date, "
            "created_at, updated_at, deleted_at) VALUES (?, ?, 'task', 3, ?, ?, ?, ?, ?, ?)",
            (note_id, content, source, space, due_date, now, now, now if deleted else None),
        )
    return note_id


@pytest.fixture
def _classify_with_date(monkeypatch):
    def set_date(due_date: str | None):
        async def fake_classify(content: str) -> Classification:
            return Classification(
                category="task", importance=3, normalized_content=content, due_date=due_date
            )

        monkeypatch.setattr(capture, "classify_note", fake_classify)

    return set_date


@pytest.mark.anyio
async def test_capture_roundtrips_due_date(_classify_with_date):
    _classify_with_date("2026-09-01")
    note = await capture.capture_note("ship the release by 2026-09-01", source="alice")

    assert note["due_date"] == "2026-09-01"
    row = get_conn().execute("SELECT due_date FROM notes WHERE id = ?", (note["id"],)).fetchone()
    assert row["due_date"] == "2026-09-01"


@pytest.mark.anyio
async def test_capture_without_date_stays_null(_classify_with_date):
    _classify_with_date(None)
    note = await capture.capture_note("just a thought", source="alice")

    assert note["due_date"] is None
    row = get_conn().execute("SELECT due_date FROM notes WHERE id = ?", (note["id"],)).fetchone()
    assert row["due_date"] is None


@pytest.mark.anyio
async def test_unparseable_model_date_becomes_null(_classify_with_date):
    _classify_with_date("next someday-ish")  # model garbage -> no date, never invent one
    note = await capture.capture_note("figure things out eventually", source="alice")

    assert note["due_date"] is None


def test_opt_in_defaults_off_and_persists():
    assert briefing_enabled("alice") is False  # default off

    set_briefing_enabled("alice", True)
    assert briefing_enabled("alice") is True

    set_briefing_enabled("alice", False)
    assert briefing_enabled("alice") is False

    # Other users unaffected by alice's toggle.
    assert briefing_enabled("bob") is False


def test_opt_in_preserves_active_space():
    from app.spaces import get_space, set_space

    set_space("alice", "work")
    set_briefing_enabled("alice", True)
    assert get_space("alice") == "work"
    set_briefing_enabled("alice", False)
    assert get_space("alice") == "work"


def test_due_today_filtering():
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    due = _insert_note("due today", due_date=today)
    due_other_space = _insert_note("due today in work", space="work", due_date=today)
    _insert_note("due tomorrow", due_date=tomorrow)
    _insert_note("no date")
    _insert_note("deleted but due", due_date=today, deleted=True)
    _insert_note("bob's note due today", source="bob", due_date=today)

    ids = [n["id"] for n in notes_due_today("alice")]
    assert due in ids
    assert due_other_space in ids  # briefing spans the owner's spaces
    assert len(ids) == 2  # tomorrow, undated, deleted, and bob's are all excluded


@pytest.mark.anyio
async def test_run_briefing_sends_one_owner_scoped_message():
    today = date.today().isoformat()
    _insert_note("call the dentist", source="alice", due_date=today)
    _insert_note("finish report", source="alice", space="work", due_date=today)
    _insert_note("bob's commitment", source="bob", due_date=today)
    set_briefing_enabled("alice", True)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    result = await run_briefing(fake_send)

    assert result["sent"] is True
    assert len(sent) == 1
    user_id, message = sent[0]
    assert user_id == "alice"  # owner-scoped: bob's note never leaves alice's message out
    assert "call the dentist" in message and "finish report" in message
    assert "[work]" in message  # each note is labeled with its space
    assert "bob's commitment" not in message

    # Delivery is logged against the owner, doubling as the cooldown entry.
    row = get_conn().execute(
        "SELECT note_ids FROM notifications WHERE kind = 'briefing:alice'"
    ).fetchone()
    assert row is not None
    assert len(json.loads(row["note_ids"])) == 2


@pytest.mark.anyio
async def test_run_briefing_same_day_idempotent():
    _insert_note("due today", source="alice", due_date=date.today().isoformat())
    set_briefing_enabled("alice", True)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    await run_briefing(fake_send)
    second = await run_briefing(fake_send)  # restart / re-run on the same day

    assert len(sent) == 1
    assert second["sent"] is False


@pytest.mark.anyio
async def test_run_briefing_no_send_when_opted_out_or_nothing_due():
    _insert_note("due today", source="opted_out", due_date=date.today().isoformat())
    _insert_note("due tomorrow", source="opted_in", due_date=(date.today() + timedelta(days=1)).isoformat())
    set_briefing_enabled("opted_in", True)

    sent = []

    async def fake_send(user_id, message):
        sent.append((user_id, message))

    result = await run_briefing(fake_send)

    assert sent == []
    assert result["sent"] is False
