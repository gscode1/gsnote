"""User-defined reminders: one daily tick job, no per-reminder scheduler jobs.

Two flavors share the table: plain message pings (window_days NULL) and
recurring query digests (window_days set -> attach phrased notes, reusing
the reporting query and the nudge phrasing agent).
"""
import json
import logging
import uuid
from datetime import date, datetime, timezone

from app.config import get_settings
from app.db import cursor, get_conn
from app.reporting import notes_in_window
from app.resurfacing import phrase_nudge

logger = logging.getLogger(__name__)

KINDS = {"once", "daily", "weekly"}


def _today() -> date:
    # ponytail: server-local date — the same clock APScheduler's cron fires on.
    # Non-UTC mornings: set TZ on the container (single app timezone).
    return datetime.now().date()


def create_reminder(
    user_id: str,
    space: str,
    message: str,
    kind: str,
    weekday: int | None = None,
    fire_date: str | None = None,
    window_days: int | None = None,
    category: str | None = None,
) -> str:
    reminder_id = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO reminders
              (id, user_id, space, message, kind, weekday, fire_date,
               window_days, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder_id, user_id, space, message, kind, weekday,
                fire_date, window_days, category,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return reminder_id


def list_reminders(user_id: str) -> list[dict]:
    rows = get_conn().execute(
        "SELECT * FROM reminders WHERE user_id = ? AND deleted_at IS NULL ORDER BY created_at",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def cancel_reminder(user_id: str, reminder_id: str) -> bool:
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET deleted_at = ? WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            (datetime.now(timezone.utc).isoformat(), reminder_id, user_id),
        )
        return cur.rowcount == 1


def _is_due(r: dict, today: date) -> bool:
    if r["last_fired_on"] == today.isoformat():
        return False  # at most one fire per reminder per day; restart-safe
    if r["kind"] == "daily":
        return True
    if r["kind"] == "weekly":
        return r["weekday"] == today.weekday()
    # once: <= so a day the app was down fires late instead of never
    return r["fire_date"] is not None and r["fire_date"] <= today.isoformat()


def _mark_fired(reminder_id: str, today: date, note_ids: list[str] | None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET last_fired_on = ?, "
            "deleted_at = CASE WHEN kind = 'once' THEN ? ELSE deleted_at END "
            "WHERE id = ?",
            (today.isoformat(), now, reminder_id),
        )
        if note_ids:
            # Logging into notifications also puts these notes into resurfacing's
            # cooldown set — the weekly digest won't re-nudge what a reminder just showed.
            cur.execute(
                "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), json.dumps(note_ids), f"reminder:{reminder_id}",
                 get_settings().channel, now),
            )


async def _fire(send_fn, r: dict, today: date) -> dict:
    message = f"⏰ {r['message']}"
    note_ids: list[str] = []
    if r["window_days"] is not None:
        notes = notes_in_window(
            r["window_days"], r["category"], space=r["space"], owner=r["user_id"]
        )
        if not notes:
            # repo convention (digest): nothing matching -> send nothing
            _mark_fired(r["id"], today, note_ids=None)
            return {"id": r["id"], "sent": False, "reason": "no notes in window"}
        body = await phrase_nudge(notes)
        message = f"⏰ {r['message']}\n{body}"
        note_ids = [n["id"] for n in notes]

    result = send_fn(r["user_id"], message)
    if hasattr(result, "__await__"):  # same sync-or-async seam as run_digest
        await result
    _mark_fired(r["id"], today, note_ids)
    return {"id": r["id"], "sent": True, "message": message}


async def run_reminders(send_fn) -> dict:
    """Daily tick: fire everything due today. send_fn(user_id, message), sync or async."""
    today = _today()
    rows = [dict(r) for r in get_conn().execute(
        "SELECT * FROM reminders WHERE deleted_at IS NULL"
    )]
    results = []
    for r in sorted(rows, key=lambda r: (r["user_id"], r["created_at"])):
        if not _is_due(r, today):
            continue
        try:
            results.append(await _fire(send_fn, r, today))
        except Exception:
            # Send failed -> last_fired_on untouched -> daily retries tomorrow, once
            # retries next tick, weekly waits for its next weekday. One bad reminder
            # never blocks the rest.
            logger.exception("reminder %s failed", r["id"])
    return {"date": today.isoformat(), "fired": results}
