"""Hybrid retrieval: vector + keyword (FTS5 BM25) candidates -> plain Reciprocal Rank Fusion -> top-K."""
import json
import re
import sqlite3

from app.config import get_settings
from app.db import get_conn
from app.embeddings import embed
from app.spaces import scope_filter


def _vector_candidates(
    conn: sqlite3.Connection,
    query: str,
    n: int,
    space: str | None = None,
) -> list[str]:
    vector = embed(query)
    # ponytail: vec0 applies k before SQL filters, so oversample then scope
    fetch_k = n if space is None else min(max(n * 20, n), 500)
    rows = conn.execute(
        """
        SELECT note_id FROM note_vectors
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (json.dumps(vector), fetch_k),
    ).fetchall()
    ids = [r["note_id"] for r in rows]
    scope_sql, scope_params = scope_filter(space)
    if not scope_sql:
        return ids[:n]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    scoped = {
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM notes WHERE id IN ({placeholders}){scope_sql}",
            (*ids, *scope_params),
        ).fetchall()
    }
    return [i for i in ids if i in scoped][:n]


_STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "myself", "we", "you", "he", "she", "it", "they",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "what", "when", "who", "where", "why", "how", "which",
    "to", "of", "in", "on", "at", "for", "with", "about", "from", "by", "as",
    "and", "or", "but", "if", "this", "that", "these", "those",
    "set", "get", "any", "all", "some",
}


def _fts5_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from free text (OR of quoted tokens, stopwords dropped)."""
    tokens = [t for t in re.findall(r"[\w']+", query, re.UNICODE) if t.lower() not in _STOPWORDS]
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _keyword_candidates(
    conn: sqlite3.Connection, query: str, n: int, space: str | None = None
) -> list[str]:
    fts_query = _fts5_query(query)
    if fts_query is None:
        return []
    # FTS5 bm25: lower score = better match -> ascending order. Join notes to scope.
    sql = (
        "SELECT f.note_id, bm25(notes_fts) AS score "
        "FROM notes_fts f JOIN notes n ON n.id = f.note_id "
        "WHERE notes_fts MATCH ?"
    )
    params: list = [fts_query]
    scope_sql, scope_params = scope_filter(space, alias="n")
    sql += scope_sql
    params.extend(scope_params)
    sql += " ORDER BY score LIMIT ?"
    params.append(n)
    rows = conn.execute(sql, params).fetchall()
    return [r["note_id"] for r in rows]


def _rrf_fuse(ranked_lists: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Plain unweighted Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, note_id in enumerate(ids, start=1):
            scores[note_id] = scores.get(note_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def search(
    query: str,
    top_k: int | None = None,
    space: str | None = "default",
) -> list[dict]:
    """Hybrid retrieval. `space` scopes to one space (None = all spaces).

    Both candidate generators apply the same scope; the final fetch still filters as a
    defensive check against cross-scope leakage. Each result carries its fused `score`.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    n = settings.retrieval_candidate_n
    conn = get_conn()

    fused = _rrf_fuse(
        [_vector_candidates(conn, query, n, space), _keyword_candidates(conn, query, n, space)],
        settings.rrf_k,
    )
    if not fused:
        return []

    ids = [note_id for note_id, _ in fused]
    placeholders = ",".join("?" for _ in ids)
    scope_sql, scope_params = scope_filter(space)
    rows = conn.execute(
        f"SELECT * FROM notes WHERE id IN ({placeholders}){scope_sql}", [*ids, *scope_params]
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}

    results = []
    for note_id, score in fused:
        note = by_id.get(note_id)
        if note is not None:
            note["score"] = score
            results.append(note)
    return results[:top_k]
