"""Named note spaces.

Each note belongs to a space. A space is just a name — using it creates it implicitly.
"""
import re

from app.db import get_conn

DEFAULT_SPACE = "default"

_SPACE_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")


def normalize_space(name: str) -> str:
    """Lowercase, hyphenate separators, validate. Raises ValueError on junk."""
    name = re.sub(r"[\s_]+", "-", name.strip().lower())
    if not _SPACE_RE.fullmatch(name):
        raise ValueError(f"Invalid space name: {name!r} (use letters, digits, dashes)")
    return name


def scope_filter(space: str | None, alias: str = "") -> tuple[str, list]:
    """The one authoritative note read scope: space.

    Every note read path applies this. space=None means unscoped (all spaces).
    """
    col = f"{alias}." if alias else ""
    if space is None:
        return "", []
    return f" AND {col}space = ?", [space]


def list_spaces() -> list[dict]:
    """Every space that has non-deleted notes, with its note count, sorted by name."""
    rows = get_conn().execute(
        "SELECT space, COUNT(*) AS notes FROM notes WHERE deleted_at IS NULL "
        "GROUP BY space ORDER BY space"
    ).fetchall()
    return [{"space": r["space"], "notes": r["notes"]} for r in rows]
