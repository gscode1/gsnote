"""Capture pipeline (PRD §5): classify -> store -> embed -> graph edges -> ack."""
import json
import uuid
from datetime import datetime, timezone

from app import graph
from app.agents import classify_note
from app.db import cursor
from app.embeddings import embed


async def capture_note(content: str, source: str = "user", space: str = "personal") -> dict:
    classification = await classify_note(content)

    note_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    vector = embed(classification.normalized_content)

    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                classification.normalized_content,
                classification.category,
                classification.importance,
                source,
                space,
                now,
                now,
            ),
        )
        cur.execute(
            "INSERT INTO notes_fts (note_id, content) VALUES (?, ?)",
            (note_id, classification.normalized_content),
        )
        cur.execute(
            "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
            (note_id, json.dumps(vector)),
        )
        graph.build_temporal_edges(cur.connection, note_id, source, now)
        graph.build_semantic_edges(cur.connection, note_id, vector)

    return {
        "id": note_id,
        "content": classification.normalized_content,
        "category": classification.category,
        "importance": classification.importance,
        "space": space,
        "created_at": now,
    }
