"""Database-driven reminders with per-reminder local schedules.

The scheduler is deliberately one fixed worker tick. Reminder state lives in
SQLite, so Kubernetes does not need a ConfigMap update or one scheduler job per
user reminder.
"""
import json
import logging
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.db import cursor, get_conn
from app.reporting import notes_in_window
from app.resurfacing import phrase_nudge
from app.spaces import get_timezone, normalize_timezone

logger = logging.getLogger(__name__)

KINDS = {"once", "daily", "weekly"}
DEFAULT_LOCAL_TIME = "08:00"
DEFAULT_TIMEZONE = "UTC"
CLAIM_MINUTES = 10
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def normalize_local_time(value: str) -> str:
    """Validate a 24-hour HH:MM time and return a zero-padded value."""
    value = value.strip()
    if not _TIME_RE.fullmatch(value):
        raise ValueError("Time must use 24-hour HH:MM format, for example 21:00")
    hour, minute = (int(part) for part in value.split(":"))
    if hour > 23 or minute > 59:
        raise ValueError("Time must use 24-hour HH:MM format, for example 21:00")
    return f"{hour:02d}:{minute:02d}"


def _localize(naive: datetime, tz: ZoneInfo) -> datetime:
    """Attach a zone, choosing first fall-back occurrence and next valid gap time."""
    candidate = naive.replace(tzinfo=tz, fold=0)
    round_trip = candidate.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)
    if round_trip == naive:
        return candidate

    # A spring-forward wall time does not exist. Move to the first valid minute.
    for minutes in range(1, 24 * 60 + 1):
        shifted = naive + timedelta(minutes=minutes)
        candidate = shifted.replace(tzinfo=tz, fold=0)
        round_trip = candidate.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)
        if round_trip == shifted:
            return candidate
    raise ValueError(f"Could not resolve local time {naive!s} in {tz.key}")


def compute_next_run_at(
    kind: str,
    local_time: str = DEFAULT_LOCAL_TIME,
    tz_name: str = DEFAULT_TIMEZONE,
    weekday: int | None = None,
    fire_date: str | None = None,
    ref_utc: datetime | None = None,
) -> str:
    """Return the next scheduled occurrence as an ISO UTC timestamp."""
    if kind not in KINDS:
        raise ValueError("kind must be one of: once, daily, weekly")
    local_time = normalize_local_time(local_time)
    tz_name = normalize_timezone(tz_name)
    tz = ZoneInfo(tz_name)
    hour, minute = (int(part) for part in local_time.split(":"))
    ref = _utc(ref_utc)
    local_now = ref.astimezone(tz)

    if kind == "once":
        if not fire_date:
            raise ValueError("once reminders need fire_date as YYYY-MM-DD")
        target_date = date.fromisoformat(fire_date)
        target = _localize(datetime.combine(target_date, time(hour, minute)), tz)
        return _iso(target.astimezone(timezone.utc))

    if kind == "weekly":
        if weekday is None or not 0 <= weekday <= 6:
            raise ValueError("weekly reminders need weekday 0 (Monday) .. 6 (Sunday)")
        days = (weekday - local_now.weekday()) % 7
    else:
        days = 0

    target_date = local_now.date() + timedelta(days=days)
    target = _localize(datetime.combine(target_date, time(hour, minute)), tz)
    target_utc = target.astimezone(timezone.utc)
    if target_utc <= ref:
        target_date += timedelta(days=7 if kind == "weekly" else 1)
        target = _localize(datetime.combine(target_date, time(hour, minute)), tz)
        target_utc = target.astimezone(timezone.utc)
    return _iso(target_utc)


def create_reminder(
    user_id: str,
    space: str,
    message: str,
    kind: str,
    weekday: int | None = None,
    fire_date: str | None = None,
    window_days: int | None = None,
    category: str | None = None,
    local_time: str = DEFAULT_LOCAL_TIME,
    tz_name: str | None = None,
) -> str:
    now = _utc()
    tz_name = normalize_timezone(tz_name or get_timezone(user_id) or DEFAULT_TIMEZONE)
    local_time = normalize_local_time(local_time)
    next_run_at = compute_next_run_at(
        kind, local_time, tz_name, weekday=weekday, fire_date=fire_date, ref_utc=now
    )
    reminder_id = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO reminders
              (id, user_id, space, message, kind, weekday, fire_date,
               window_days, category, local_time, timezone, next_run_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reminder_id, user_id, space, message, kind, weekday, fire_date,
                window_days, category, local_time, tz_name, next_run_at, _iso(now),
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
            (_iso(datetime.now(timezone.utc)), reminder_id, user_id),
        )
        return cur.rowcount == 1


def _claim(reminder_id: str, now: datetime) -> str | None:
    token = str(uuid.uuid4())
    now_iso = _iso(now)
    claim_until = _iso(now + timedelta(minutes=CLAIM_MINUTES))
    with cursor() as cur:
        cur.execute(
            """
            UPDATE reminders
            SET claim_token = ?, claim_until = ?
            WHERE id = ? AND deleted_at IS NULL AND next_run_at <= ?
              AND (claim_until IS NULL OR claim_until <= ?)
            """,
            (token, claim_until, reminder_id, now_iso, now_iso),
        )
        return token if cur.rowcount == 1 else None


def _release_claim(reminder_id: str, token: str) -> None:
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET claim_token = NULL, claim_until = NULL "
            "WHERE id = ? AND claim_token = ?",
            (reminder_id, token),
        )


def _complete(
    reminder: dict,
    token: str,
    now: datetime,
    note_ids: list[str],
) -> None:
    now_iso = _iso(now)
    local_date = now.astimezone(ZoneInfo(reminder["timezone"])).date().isoformat()
    next_run_at = None if reminder["kind"] == "once" else compute_next_run_at(
        reminder["kind"],
        reminder["local_time"],
        reminder["timezone"],
        weekday=reminder["weekday"],
        fire_date=reminder["fire_date"],
        ref_utc=now,
    )
    with cursor() as cur:
        cur.execute(
            """
            UPDATE reminders
            SET next_run_at = ?, last_fired_on = ?, deleted_at =
                CASE WHEN kind = 'once' THEN ? ELSE deleted_at END,
                claim_token = NULL, claim_until = NULL
            WHERE id = ? AND claim_token = ?
            """,
            (next_run_at, local_date, now_iso if reminder["kind"] == "once" else None,
             reminder["id"], token),
        )
        if note_ids:
            cur.execute(
                "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), json.dumps(note_ids), f"reminder:{reminder['id']}",
                    get_settings().channel, now_iso,
                ),
            )


async def _fire(send_fn, reminder: dict, token: str, now: datetime) -> dict:
    message = f"⏰ {reminder['message']}"
    note_ids: list[str] = []
    if reminder["window_days"] is not None:
        notes = notes_in_window(
            reminder["window_days"], reminder["category"],
            space=reminder["space"], owner=reminder["user_id"],
        )
        if not notes:
            _complete(reminder, token, now, note_ids=[])
            return {"id": reminder["id"], "sent": False, "reason": "no notes in window"}
        body = await phrase_nudge(notes)
        message = f"⏰ {reminder['message']}\n{body}"
        note_ids = [n["id"] for n in notes]

    result = send_fn(reminder["user_id"], message)
    if hasattr(result, "__await__"):
        await result
    _complete(reminder, token, now, note_ids)
    return {"id": reminder["id"], "sent": True, "message": message}


def _initialize_schedule(reminder: dict, now: datetime) -> dict:
    if reminder["next_run_at"] is not None:
        return reminder
    next_run_at = compute_next_run_at(
        reminder["kind"], reminder["local_time"], reminder["timezone"],
        weekday=reminder["weekday"], fire_date=reminder["fire_date"], ref_utc=now,
    )
    with cursor() as cur:
        cur.execute(
            "UPDATE reminders SET next_run_at = ? WHERE id = ? AND next_run_at IS NULL",
            (next_run_at, reminder["id"]),
        )
    reminder["next_run_at"] = next_run_at
    return reminder


async def run_reminders(send_fn, now_utc: datetime | None = None) -> dict:
    """Minute tick: claim and fire every reminder whose UTC occurrence is due."""
    now = _utc(now_utc)
    now_iso = _iso(now)
    rows = [dict(r) for r in get_conn().execute(
        "SELECT * FROM reminders WHERE deleted_at IS NULL "
        "AND (next_run_at IS NULL OR next_run_at <= ?) ORDER BY next_run_at, created_at",
        (now_iso,),
    )]
    results = []
    for reminder in rows:
        try:
            reminder = _initialize_schedule(reminder, now)
            if reminder["next_run_at"] > now_iso:
                continue
            token = _claim(reminder["id"], now)
            if token is None:
                continue
            try:
                results.append(await _fire(send_fn, reminder, token, now))
            except Exception:
                _release_claim(reminder["id"], token)
                raise
        except Exception:
            logger.exception("reminder %s failed", reminder["id"])
    return {"at": now_iso, "fired": results}
