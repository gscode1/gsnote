"""Capture pipeline: validate -> store -> embed. No LLM involved."""
import json
import re
import uuid
from datetime import datetime, timezone

from app.db import cursor, get_conn, require_vectors_ready
from app.embeddings import embed_document
from app.spaces import DEFAULT_SPACE, normalize_space

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
    require_vectors_ready()
    content = (content or "").strip()
    if not content:
        raise ValueError("Note content must not be empty")
    space = normalize_space(space)
    category = normalize_category(category)

    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    vector = embed_document(content)

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

    return {
        "id": note_id,
        "content": content,
        "category": category,
        "source": source,
        "space": space,
        "created_at": now,
    }


def delete_note(note_id: str) -> bool:
    """Hard delete from notes, FTS and vector index. Returns False if the note does not exist."""
    with cursor() as cur:
        cur.execute("DELETE FROM notes_fts WHERE note_id = ?", (note_id,))
        cur.execute("DELETE FROM note_vectors WHERE note_id = ?", (note_id,))
        deleted = cur.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount
    return deleted > 0


def import_notes(payload: dict) -> dict:
    """Import an export payload (version 1 = old bot format, version 2). Skips existing ids."""
    require_vectors_ready()
    version = payload.get("version") if isinstance(payload, dict) else None
    if version not in (1, 2):
        raise ValueError(f"Unsupported export version: {version!r} (expected 1 or 2)")
    notes = payload.get("notes")
    if not isinstance(notes, list):
        raise ValueError("Export payload must contain a 'notes' list")

    imported = skipped = 0
    for raw in notes:
        if not isinstance(raw, dict):
            raise ValueError("Each note must be an object")
        content = (raw.get("content") or "").strip()
        note_id = raw.get("id")
        if not content or not note_id or raw.get("deleted_at"):  # v1 soft-deleted notes
            skipped += 1
            continue
        if get_conn().execute("SELECT 1 FROM notes WHERE id = ?", (note_id,)).fetchone():
            skipped += 1
            continue
        try:
            space = normalize_space(raw.get("space") or "")
        except ValueError:
            space = DEFAULT_SPACE
        try:
            category = normalize_category(raw.get("category") or "")
        except ValueError:
            category = DEFAULT_CATEGORY
        now = datetime.now(timezone.utc).isoformat()
        created_at = raw.get("created_at") or now
        updated_at = raw.get("updated_at") or created_at
        vector = embed_document(content)
        with cursor() as cur:
            cur.execute(
                "INSERT INTO notes (id, content, category, source, space, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (note_id, content, category, raw.get("source") or "user", space, created_at, updated_at),
            )
            cur.execute("INSERT INTO notes_fts (note_id, content) VALUES (?, ?)", (note_id, content))
            cur.execute(
                "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
                (note_id, json.dumps(vector)),
            )
        imported += 1
    return {"imported": imported, "skipped": skipped}
