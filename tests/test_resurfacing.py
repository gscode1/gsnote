import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db import cursor, get_conn
from app.resurfacing import record_response, run_digest, score_candidates


def _insert_note(
    content: str,
    importance: int,
    created_days_ago: int,
    access_count: int = 0,
    last_accessed_days_ago: int | None = None,
    space: str = "personal",
) -> str:
    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=created_days_ago)).isoformat()
    last_accessed_at = (
        (now - timedelta(days=last_accessed_days_ago)).isoformat()
        if last_accessed_days_ago is not None
        else None
    )
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at, "
            "last_accessed_at, access_count) VALUES (?, ?, 'idea', ?, 'alice', ?, ?, ?, ?, ?)",
            (note_id, content, importance, space, created_at, created_at, last_accessed_at, access_count),
        )
    return note_id


def test_open_loop_note_scores_above_threshold():
    note_id = _insert_note("An important idea never revisited", importance=5, created_days_ago=10)

    candidates = score_candidates()
    ids = [c["id"] for c in candidates]
    assert note_id in ids


def test_recently_accessed_note_is_not_an_open_loop():
    _insert_note(
        "A note I check often",
        importance=5,
        created_days_ago=10,
        access_count=5,
        last_accessed_days_ago=0,
    )

    candidates = score_candidates()
    # accessed today -> staleness ~0, open_loop=0 -> shouldn't clear threshold
    assert len(candidates) == 0


def test_dismissed_note_is_suppressed():
    note_id = _insert_note("An idea I dismissed", importance=5, created_days_ago=10)

    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (id, note_ids, kind, channel, sent_at, user_response) "
            "VALUES (?, ?, 'digest', 'telegram', ?, 'dismissed')",
            (str(uuid.uuid4()), json.dumps([note_id]), datetime.now(timezone.utc).isoformat()),
        )

    candidates = score_candidates()
    assert note_id not in [c["id"] for c in candidates]


def test_recently_notified_note_is_in_cooldown():
    note_id = _insert_note("An idea notified yesterday", importance=5, created_days_ago=10)

    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) "
            "VALUES (?, ?, 'digest', 'telegram', ?)",
            (
                str(uuid.uuid4()),
                json.dumps([note_id]),
                (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            ),
        )

    candidates = score_candidates()
    assert note_id not in [c["id"] for c in candidates]


def test_budget_caps_candidate_count():
    from app.config import get_settings

    get_settings.cache_clear()
    import os

    os.environ["RESURFACING_BUDGET"] = "2"
    get_settings.cache_clear()

    for i in range(5):
        _insert_note(f"Open loop idea number {i}", importance=5, created_days_ago=10)

    candidates = score_candidates()
    assert len(candidates) <= 2

    del os.environ["RESURFACING_BUDGET"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_run_digest_sends_and_logs_notification(monkeypatch):
    _insert_note("Open loop worth a nudge", importance=5, created_days_ago=10)

    import app.resurfacing as resurfacing_module

    async def fake_phrase_nudge(notes):
        return "You captured an idea worth revisiting!"

    monkeypatch.setattr(resurfacing_module, "phrase_nudge", fake_phrase_nudge)

    sent = []

    async def fake_send(digest):
        sent.append(digest)

    result = await run_digest(fake_send)

    assert result["sent"] is True
    # Per-space digest: the personal note produces one labeled personal message.
    assert len(sent) == 1
    assert "You captured an idea worth revisiting!" in sent[0].message
    assert "Personal" in sent[0].message
    assert sent[0].notification_id  # seam carries the id for button responses

    conn = get_conn()
    row = conn.execute("SELECT * FROM notifications WHERE kind = 'digest:personal'").fetchone()
    assert row is not None
    assert row["id"] == sent[0].notification_id
    assert row["user_response"] is None


@pytest.mark.asyncio
async def test_run_digest_is_per_space(monkeypatch):
    _insert_note("work open loop", importance=5, created_days_ago=10, space="work")
    _insert_note("personal open loop", importance=5, created_days_ago=10, space="personal")

    import app.resurfacing as resurfacing_module

    async def fake_phrase_nudge(notes):
        # echo the (single-space) note content so we can assert isolation
        return notes[0]["content"]

    monkeypatch.setattr(resurfacing_module, "phrase_nudge", fake_phrase_nudge)

    sent = []

    async def fake_send(digest):
        sent.append(digest)

    result = await run_digest(fake_send)
    assert result["sent"] is True
    assert len(sent) == 2  # one per space

    work_msg = next(d.message for d in sent if d.message.startswith("🗂 Work"))
    personal_msg = next(d.message for d in sent if d.message.startswith("🗂 Personal"))
    # Each space's digest contains only its own note.
    assert "work open loop" in work_msg and "personal open loop" not in work_msg
    assert "personal open loop" in personal_msg and "work open loop" not in personal_msg


@pytest.mark.asyncio
async def test_run_digest_does_not_resend_within_cooldown(monkeypatch):
    _insert_note("Open loop worth a nudge", importance=5, created_days_ago=10)

    import app.resurfacing as resurfacing_module

    async def fake_phrase_nudge(notes):
        return "Nudge message"

    monkeypatch.setattr(resurfacing_module, "phrase_nudge", fake_phrase_nudge)

    sent = []

    async def fake_send(digest):
        sent.append(digest)

    first = await run_digest(fake_send)
    assert first["sent"] is True

    second = await run_digest(fake_send)
    assert second["sent"] is False
    assert len(sent) == 1


def test_record_response_updates_notification():
    note_id = _insert_note("idea", importance=5, created_days_ago=10)
    notif_id = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) "
            "VALUES (?, ?, 'digest', 'telegram', ?)",
            (notif_id, json.dumps([note_id]), datetime.now(timezone.utc).isoformat()),
        )

    record_response(notif_id, "dismissed")

    row = get_conn().execute("SELECT user_response FROM notifications WHERE id = ?", (notif_id,)).fetchone()
    assert row["user_response"] == "dismissed"
