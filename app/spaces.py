"""Named note spaces.

Each note belongs to a space; each user has an active space that new notes and
queries scope to. A space is just a name — using it creates it implicitly.
"""
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.db import get_conn

DEFAULT_SPACE = "default"

_SPACE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")


def normalize_space(name: str) -> str:
    """Lowercase, hyphenate separators, validate. Raises ValueError on junk."""
    name = re.sub(r"[\s_]+", "-", name.strip().lower())
    if not _SPACE_RE.fullmatch(name):
        raise ValueError(f"Invalid space name: {name!r} (use letters, digits, dashes)")
    return name


def scope_filter(owner: str | None = None, space: str | None = None, alias: str = "") -> tuple[str, list]:
    """The one authoritative note read scope: owner (notes.source) + space.

    Every note read path applies this. owner=None/space=None means unscoped —
    reserved for admin surfaces (the token-protected HTTP API and export).
    """
    col = f"{alias}." if alias else ""
    sql, params = "", []
    if owner is not None:
        sql += f" AND {col}source = ?"
        params.append(owner)
    if space is not None:
        sql += f" AND {col}space = ?"
        params.append(space)
    return sql, params

def list_spaces(user_id: str) -> list[str]:
    """Spaces the user has notes in, plus their active one."""
    rows = get_conn().execute(
        "SELECT DISTINCT space FROM notes WHERE source = ?", (user_id,)
    ).fetchall()
    return sorted({r["space"] for r in rows} | {get_space(user_id)})


def get_space(user_id: str) -> str:
    row = get_conn().execute(
        "SELECT space FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["space"] if row else DEFAULT_SPACE


def set_space(user_id: str, space: str) -> None:
    space = normalize_space(space)
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:  # connection is also a context manager for commit
        conn.execute(
            "INSERT INTO user_settings (user_id, space, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET space = excluded.space, updated_at = excluded.updated_at",
            (user_id, space, now),
        )


def normalize_timezone(name: str) -> str:
    """Validate and normalize an IANA timezone name."""
    name = name.strip()
    if not name:
        raise ValueError("Timezone is required; use an IANA name such as Europe/Warsaw")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone: {name!r} (use an IANA name such as Europe/Warsaw)")
    return name


def get_timezone(user_id: str) -> str | None:
    row = get_conn().execute(
        "SELECT timezone FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["timezone"] if row and row["timezone"] else None


def set_timezone(user_id: str, name: str) -> str:
    """Persist a validated IANA timezone and return its canonical input."""
    name = normalize_timezone(name)
    now = datetime.now(timezone.utc).isoformat()
    # Include the current/default space so creating preferences does not revive
    # the obsolete user_settings default ('personal') from migration 002.
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO user_settings (user_id, space, timezone, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET timezone = excluded.timezone, "
            "updated_at = excluded.updated_at",
            (user_id, get_space(user_id), name, now),
        )
    return name
