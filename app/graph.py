"""Graph builder: temporal edges (free, at insert) + semantic kNN edges (cheap, at insert)."""
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from app.config import get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insert_edge(conn: sqlite3.Connection, from_id: str, to_id: str, edge_type: str, weight: float) -> None:
    conn.execute(
        "INSERT INTO edges (id, from_id, to_id, type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), from_id, to_id, edge_type, weight, _now()),
    )


def build_temporal_edges(conn: sqlite3.Connection, note_id: str, source: str, created_at: str) -> None:
    """temporal_backbone: same source, sequential. temporal_proximity: within N hours, any source."""
    settings = get_settings()

    prev = conn.execute(
        "SELECT id, created_at FROM notes WHERE source = ? AND id != ? AND deleted_at IS NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (source, note_id),
    ).fetchone()
    if prev is not None:
        _insert_edge(conn, prev["id"], note_id, "temporal_backbone", 1.0)

    window_start = (
        datetime.fromisoformat(created_at) - timedelta(hours=settings.temporal_proximity_hours)
    ).isoformat()
    nearby = conn.execute(
        "SELECT id, created_at FROM notes WHERE id != ? AND deleted_at IS NULL "
        "AND created_at >= ? AND created_at <= ?",
        (note_id, window_start, created_at),
    ).fetchall()
    for row in nearby:
        hours_diff = max(
            (datetime.fromisoformat(created_at) - datetime.fromisoformat(row["created_at"])).total_seconds() / 3600,
            0.01,
        )
        weight = 1 / (1 + hours_diff)
        _insert_edge(conn, row["id"], note_id, "temporal_proximity", weight)


def build_semantic_edges(conn: sqlite3.Connection, note_id: str, embedding: list[float]) -> None:
    """kNN semantic edges using the just-computed embedding, above a similarity threshold."""
    settings = get_settings()
    k = settings.semantic_knn_k
    threshold = settings.semantic_similarity_threshold

    rows = conn.execute(
        """
        SELECT nv.note_id AS id, nv.distance AS distance
        FROM note_vectors nv
        WHERE nv.embedding MATCH ? AND k = ?
        ORDER BY nv.distance
        """,
        (json.dumps(embedding), k + 1),  # +1 because the note itself may already be inserted
    ).fetchall()

    for row in rows:
        if row["id"] == note_id:
            continue
        similarity = 1 - row["distance"]  # cosine distance -> similarity
        if similarity >= threshold:
            _insert_edge(conn, row["id"], note_id, "semantic", similarity)
