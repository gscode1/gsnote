"""Work/personal space separation.

Each note belongs to a space; each user has an active space that new notes and
queries scope to. Toggled via /work and /personal in the channel.
"""
from datetime import datetime, timezone

from app.db import get_conn

VALID_SPACES = {"personal", "work"}
DEFAULT_SPACE = "personal"


def get_space(user_id: str) -> str:
    row = get_conn().execute(
        "SELECT space FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["space"] if row else DEFAULT_SPACE


def set_space(user_id: str, space: str) -> None:
    if space not in VALID_SPACES:
        raise ValueError(f"Unknown space: {space}")
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:  # connection is also a context manager for commit
        conn.execute(
            "INSERT INTO user_settings (user_id, space, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET space = excluded.space, updated_at = excluded.updated_at",
            (user_id, space, now),
        )
