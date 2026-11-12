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


def list_spaces() -> list[dict]:
    """Every space that has notes, with its note count, sorted by name."""
    rows = get_conn().execute(
        "SELECT space, COUNT(*) AS notes FROM notes "
        "GROUP BY space ORDER BY space"
    ).fetchall()
    return [{"space": r["space"], "notes": r["notes"]} for r in rows]
