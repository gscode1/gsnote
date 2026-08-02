"""Opt-in daily commitment briefing (#34): notes due today, one owner-scoped message.

Opt-in lives in user_settings (same row as the active space). The daily job sends
at most one message per opted-in owner, only when that owner has notes due today;
delivery reuses the plain channel send_fn shared with reminders.
"""
import json
import logging
import uuid
from datetime import date, datetime, timezone

from app.config import get_settings
from app.db import cursor, get_conn

logger = logging.getLogger(__name__)


def _today() -> date:
    # ponytail: server-local date — the same clock APScheduler's cron fires on.
    return datetime.now().date()


def briefing_enabled(user_id: str) -> bool:
    row = get_conn().execute(
        "SELECT briefing_enabled FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return bool(row and row["briefing_enabled"])


def set_briefing_enabled(user_id: str, enabled: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:  # connection is also a context manager for commit
        conn.execute(
            "INSERT INTO user_settings (user_id, briefing_enabled, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET briefing_enabled = excluded.briefing_enabled, "
            "updated_at = excluded.updated_at",
            (user_id, int(enabled), now),
        )


def notes_due_today(owner: str) -> list[dict]:
    """Non-deleted notes of one owner due today, across all their spaces."""
    rows = get_conn().execute(
        "SELECT * FROM notes WHERE source = ? AND due_date = ? AND deleted_at IS NULL "
        "ORDER BY space, created_at",
        (owner, _today().isoformat()),
    ).fetchall()
    return [dict(r) for r in rows]


def _already_sent_today(owner: str, today: date) -> bool:
    row = get_conn().execute(
        "SELECT 1 FROM notifications WHERE kind = ? AND date(sent_at) = ?",
        (f"briefing:{owner}", today.isoformat()),
    ).fetchone()
    return row is not None


async def run_briefing(send_fn) -> dict:
    """Send each opted-in owner one briefing with their notes due today.

    Nothing is sent to opted-out users or owners with nothing due. The
    notifications row doubles as the same-day idempotency guard (restart-safe)
    and puts the notes into resurfacing's cooldown set.
    send_fn(user_id, message), sync or async (same seam as run_reminders).
    """
    today = _today()
    rows = get_conn().execute(
        "SELECT user_id FROM user_settings WHERE briefing_enabled = 1"
    ).fetchall()
    results = []
    for r in sorted(rows, key=lambda r: r["user_id"]):
        owner = r["user_id"]
        if _already_sent_today(owner, today):
            results.append({"owner": owner, "sent": False, "reason": "already sent today"})
            continue
        notes = notes_due_today(owner)
        if not notes:
            results.append({"owner": owner, "sent": False, "reason": "nothing due today"})
            continue

        lines = "\n".join(f"- [{n['space']}] {n['content']}" for n in notes)
        message = f"📋 Due today ({today.isoformat()}):\n{lines}"
        result = send_fn(owner, message)
        if hasattr(result, "__await__"):
            await result

        with cursor() as cur:
            cur.execute(
                "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    json.dumps([n["id"] for n in notes]),
                    f"briefing:{owner}",
                    get_settings().channel,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        results.append({"owner": owner, "sent": True, "note_ids": [n["id"] for n in notes]})
    return {"date": today.isoformat(), "sent": any(r["sent"] for r in results), "owners": results}
