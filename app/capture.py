"""Capture pipeline: validate -> store -> embed -> graph edges. No LLM involved."""
import json
import re
import uuid
from datetime import datetime, timezone

from app import graph
from app.db import cursor
from app.embeddings import embed
from app.spaces import normalize_space

DEFAULT_CATEGORY = "note"

_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")


def normalize_category(name: str) -> str:
    """Lowercase, hyphenate separators, validate. Raises ValueError on junk."""
    name = re.sub(r"[\s_]+", "-", name.strip().lower())
    if not _CATEGORY_RE.fullmatch(name):
        raise ValueError(f"Invalid category: {name!r} (use letters, digits, dashes; max 32 chars)")
    return name


def capture_note(
    content: str,
    source: str = "user",
    space: str = "default",
    category: str = DEFAULT_CATEGORY,
) -> dict:
    content = (content or "").strip()
    if not content:
        raise ValueError("Note content must not be empty")
    space = normalize_space(space)
    category = normalize_category(category)

    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    vector = embed(content)

    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (id, content, category, source, space, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (note_id, content, category, source, space, now, now),
        )
        cur.execute("INSERT INTO notes_fts (note_id, content) VALUES (?, ?)", (note_id, content))
        cur.execute(
            "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
            (note_id, json.dumps(vector)),
        )
        graph.build_temporal_edges(cur.connection, note_id, source, now)
        graph.build_semantic_edges(cur.connection, note_id, vector)

    return {
        "id": note_id,
        "content": content,
        "category": category,
        "source": source,
        "space": space,
        "created_at": now,
    }
