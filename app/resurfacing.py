"""Proactive resurfacing — the soul (PRD §8). Weekly digest, conservative, dedup-aware."""
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.agents import phrase_nudge
from app.config import get_settings
from app.db import cursor, get_conn


@dataclass(frozen=True)
class Digest:
    """What resurfacing hands the channel: text + the notification row to attach responses to."""

    message: str
    notification_id: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _suppressed_note_ids(conn, cooldown_days: int) -> set[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cooldown_days)).isoformat()
    rows = conn.execute(
        """
        SELECT note_ids, user_response, snooze_until FROM notifications
        WHERE sent_at >= ? OR user_response = 'dismissed' OR
              (snooze_until IS NOT NULL AND snooze_until > ?)
        """,
        (cutoff, _now()),
    ).fetchall()
    suppressed: set[str] = set()
    for row in rows:
        for nid in json.loads(row["note_ids"]):
            suppressed.add(nid)
    return suppressed


def score_candidates(space: str | None = None) -> list[dict]:
    """Heuristic SQL scoring — no LLM needed to select.

    score = w1*open_loop + w2*(staleness * importance) + w3*recurrence - suppressors

    `space` scopes scoring to one space (work|personal); None scores across all spaces.
    """
    conn = get_conn()
    settings = get_settings()
    suppressed = _suppressed_note_ids(conn, settings.resurfacing_cooldown_days)

    sql = """
        SELECT n.*, (
            SELECT COUNT(*) FROM edges e WHERE (e.from_id = n.id OR e.to_id = n.id) AND e.type = 'semantic'
        ) AS semantic_degree
        FROM notes n
        WHERE n.deleted_at IS NULL
    """
    params: list = []
    if space is not None:
        sql += " AND n.space = ?"
        params.append(space)
    rows = conn.execute(sql, params).fetchall()

    now = datetime.now(timezone.utc)
    candidates = []
    for row in rows:
        note = dict(row)
        if note["id"] in suppressed:
            continue

        created = datetime.fromisoformat(note["created_at"])
        last_accessed = (
            datetime.fromisoformat(note["last_accessed_at"]) if note["last_accessed_at"] else created
        )
        staleness_days = (now - last_accessed).total_seconds() / 86400

        open_loop = 1.0 if note["access_count"] == 0 else 0.0
        staleness_score = min(staleness_days / 30.0, 1.0) * (note["importance"] / 5.0)
        recurrence = min(note["semantic_degree"] / 3.0, 1.0)

        score = 0.5 * open_loop + 0.3 * staleness_score + 0.2 * recurrence
        if score >= settings.resurfacing_threshold and staleness_days >= 3:
            note["_score"] = score
            candidates.append(note)

    candidates.sort(key=lambda n: n["_score"], reverse=True)
    return candidates[: settings.resurfacing_budget]


async def _send_digest_for_space(send_fn, space: str) -> dict:
    """Score, phrase, send, and log a digest for a single space."""
    candidates = score_candidates(space)
    if not candidates:
        return {"space": space, "sent": False, "reason": "no candidates above threshold"}

    body = await phrase_nudge(candidates)
    message = f"🗂 {space.capitalize()} — {body}"
    notification_id = str(uuid.uuid4())
    digest = Digest(message=message, notification_id=notification_id)

    result = send_fn(digest)
    if hasattr(result, "__await__"):
        await result

    with cursor() as cur:
        cur.execute(
            "INSERT INTO notifications (id, note_ids, kind, channel, sent_at) VALUES (?, ?, ?, ?, ?)",
            (
                notification_id,
                json.dumps([c["id"] for c in candidates]),
                f"digest:{space}",
                get_settings().channel,
                _now(),
            ),
        )
    return {
        "space": space,
        "sent": True,
        "note_ids": [c["id"] for c in candidates],
        "message": message,
        "notification_id": notification_id,
    }


async def run_digest(send_fn) -> dict:
    """Run the weekly digest per space: each space gets its own scored, labeled nudge.

    Keeping spaces separate avoids mixing work and personal open loops in one message.
    send_fn(digest: Digest) -> None (sync or async).
    """
    from app.db import get_conn

    rows = get_conn().execute("SELECT DISTINCT space FROM notes").fetchall()
    results = []
    for space in sorted(r["space"] for r in rows):
        results.append(await _send_digest_for_space(send_fn, space))
    sent_any = any(r["sent"] for r in results)
    return {"sent": sent_any, "spaces": results}


def record_response(notification_id: str, response: str, snooze_until: str | None = None) -> None:
    with cursor() as cur:
        cur.execute(
            "UPDATE notifications SET user_response = ?, snooze_until = ? WHERE id = ?",
            (response, snooze_until, notification_id),
        )
