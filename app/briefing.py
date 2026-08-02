"""Daily commitment briefing with per-user local morning times."""
import json
import uuid
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.db import cursor, get_conn
from app.reminders import DEFAULT_LOCAL_TIME, normalize_local_time
from app.spaces import get_space, get_timezone


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _today(now_utc: datetime | None = None, tz_name: str = "UTC") -> date:
    return _utc(now_utc).astimezone(ZoneInfo(tz_name)).date()


def briefing_enabled(user_id: str) -> bool:
    row = get_conn().execute(
        "SELECT briefing_enabled FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return bool(row and row["briefing_enabled"])


def get_briefing_time(user_id: str) -> str:
    row = get_conn().execute(
        "SELECT briefing_time FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["briefing_time"] if row and row["briefing_time"] else DEFAULT_LOCAL_TIME


def set_briefing_enabled(
    user_id: str, enabled: bool, local_time: str | None = None
) -> None:
    local_time = normalize_local_time(local_time or get_briefing_time(user_id))
    now = _utc()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_settings "
            "(user_id, space, briefing_enabled, briefing_time, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET briefing_enabled = excluded.briefing_enabled, "
            "briefing_time = excluded.briefing_time, updated_at = excluded.updated_at",
            (user_id, get_space(user_id), int(enabled), local_time, now.isoformat()),
        )


def notes_due_today(owner: str, today: date | None = None) -> list[dict]:
    """Non-deleted notes of one owner due on the supplied local date."""
    today = today or _today()
    rows = get_conn().execute(
        "SELECT * FROM notes WHERE source = ? AND due_date = ? AND deleted_at IS NULL "
        "ORDER BY space, created_at",
        (owner, today.isoformat()),
    ).fetchall()
    return [dict(r) for r in rows]


def _already_sent_today(owner: str, today: date, tz_name: str) -> bool:
    tz = ZoneInfo(tz_name)
    rows = get_conn().execute(
        "SELECT sent_at FROM notifications WHERE kind = ?",
        (f"briefing:{owner}",),
    ).fetchall()
    for row in rows:
        try:
            sent_at = datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00"))
            if _utc(sent_at).astimezone(tz).date() == today:
                return True
        except (TypeError, ValueError):
            continue
    return False


async def run_briefing(send_fn, now_utc: datetime | None = None) -> dict:
    """Minute-worker pass: send due briefings at each owner's local time."""
    now = _utc(now_utc)
    rows = get_conn().execute(
        "SELECT user_id, briefing_time FROM user_settings WHERE briefing_enabled = 1"
    ).fetchall()
    results = []
    for row in sorted(rows, key=lambda r: r["user_id"]):
        owner = row["user_id"]
        tz_name = get_timezone(owner) or "UTC"
        tz = ZoneInfo(tz_name)
        local_now = now.astimezone(tz)
        today = local_now.date()
        local_time = normalize_local_time(row["briefing_time"] or DEFAULT_LOCAL_TIME)
        target_hour, target_minute = (int(part) for part in local_time.split(":"))
        if (local_now.hour, local_now.minute) < (target_hour, target_minute):
            results.append({"owner": owner, "sent": False, "reason": "before local briefing time"})
            continue
        if _already_sent_today(owner, today, tz_name):
            results.append({"owner": owner, "sent": False, "reason": "already sent today"})
            continue
        notes = notes_due_today(owner, today)
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
                    str(uuid.uuid4()), json.dumps([n["id"] for n in notes]),
                    f"briefing:{owner}", get_settings().channel, now.isoformat(),
                ),
            )
        results.append({"owner": owner, "sent": True, "note_ids": [n["id"] for n in notes]})
    return {"at": now.isoformat(), "sent": any(r["sent"] for r in results), "owners": results}
