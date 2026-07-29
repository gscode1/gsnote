"""Hybrid retrieval pipeline (PRD §6) — the core feature.

Candidate generators (vector, keyword, recency, graph) -> Reciprocal Rank Fusion
-> importance boost -> top-K -> bump access stats.
"""
import json
import re
import sqlite3
from datetime import datetime, timezone

from app.config import get_settings
from app.db import cursor, get_conn
from app.embeddings import embed
from app.intent import FUSION_WEIGHTS, detect_intent


def _vector_candidates(
    conn: sqlite3.Connection, query: str, n: int, space: str | None = None
) -> list[str]:
    vector = embed(query)
    # ponytail: vec0 applies k before SQL filters, so oversample then scope by space
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
    if space is None:
        return ids[:n]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    scoped = {
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM notes WHERE id IN ({placeholders}) AND space = ? AND deleted_at IS NULL",
            (*ids, space),
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


def _keyword_candidates(conn: sqlite3.Connection, query: str, n: int, space: str | None = None) -> list[str]:
    fts_query = _fts5_query(query)
    if fts_query is None:
        return []
    # FTS5 bm25: lower score = better match -> ascending order. Join notes to scope by space.
    sql = (
        "SELECT f.note_id, bm25(notes_fts) AS score "
        "FROM notes_fts f JOIN notes n ON n.id = f.note_id "
        "WHERE notes_fts MATCH ? AND n.deleted_at IS NULL "
    )
    params: list = [fts_query]
    if space is not None:
        sql += "AND n.space = ? "
        params.append(space)
    sql += "ORDER BY score LIMIT ?"
    params.append(n)
    rows = conn.execute(sql, params).fetchall()
    return [r["note_id"] for r in rows]


def _recency_candidates(conn: sqlite3.Connection, n: int, space: str | None = None) -> list[str]:
    sql = "SELECT id FROM notes WHERE deleted_at IS NULL "
    params: list = []
    if space is not None:
        sql += "AND space = ? "
        params.append(space)
    sql += "ORDER BY created_at DESC LIMIT ?"
    params.append(n)
    rows = conn.execute(sql, params).fetchall()
    return [r["id"] for r in rows]


def _graph_candidates(
    conn: sqlite3.Connection, seed_ids: list[str], n: int, space: str | None = None
) -> list[str]:
    """1-hop expansion from seed notes along relevance-bearing edges, weighted by edge weight.

    temporal_backbone is excluded here: it links every sequential same-source note regardless
    of topic (a bookkeeping/resurfacing signal, PRD §7), so including it in topical retrieval
    floods results with chronologically-adjacent but unrelated notes.
    """
    if not seed_ids:
        return []
    min_weight = get_settings().graph_min_edge_weight
    placeholders = ",".join("?" for _ in seed_ids)
    # Over-fetch when space-scoped so post-filter still yields up to n neighbors.
    fetch_n = n if space is None else min(max(n * 20, n), 500)
    rows = conn.execute(
        f"""
        SELECT to_id AS neighbor, weight FROM edges
        WHERE from_id IN ({placeholders}) AND type IN ('semantic', 'temporal_proximity') AND weight >= ?
        UNION ALL
        SELECT from_id AS neighbor, weight FROM edges
        WHERE to_id IN ({placeholders}) AND type IN ('semantic', 'temporal_proximity') AND weight >= ?
        ORDER BY weight DESC
        LIMIT ?
        """,
        (*seed_ids, min_weight, *seed_ids, min_weight, fetch_n),
    ).fetchall()
    seen: list[str] = []
    for r in rows:
        if r["neighbor"] not in seen and r["neighbor"] not in seed_ids:
            seen.append(r["neighbor"])
    if space is None:
        return seen[:n]
    if not seen:
        return []
    ph = ",".join("?" for _ in seen)
    scoped = {
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM notes WHERE id IN ({ph}) AND space = ? AND deleted_at IS NULL",
            (*seen, space),
        ).fetchall()
    }
    return [i for i in seen if i in scoped][:n]


def _rrf_fuse(ranked_lists: dict[str, list[str]], weights: dict[str, float], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for signal, ids in ranked_lists.items():
        w = weights.get(signal, 0.0)
        for rank, note_id in enumerate(ids, start=1):
            scores[note_id] = scores.get(note_id, 0.0) + w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _bump_access(conn: sqlite3.Connection, note_ids: list[str]) -> None:
    if not note_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" for _ in note_ids)
    conn.execute(
        f"UPDATE notes SET access_count = access_count + 1, last_accessed_at = ? "
        f"WHERE id IN ({placeholders})",
        (now, *note_ids),
    )
    conn.commit()


def search(query: str, top_k: int | None = None, space: str | None = "personal") -> list[dict]:
    """Hybrid retrieval. `space` scopes results (work|personal); None searches across all spaces.

    All candidate generators scope by space when set; the final fetch still filters as a
    defensive check against cross-space leakage.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    n = settings.retrieval_candidate_n

    intent = detect_intent(query)
    weights = FUSION_WEIGHTS[intent]
    conn = get_conn()

    vector_ids = _vector_candidates(conn, query, n, space)
    keyword_ids = _keyword_candidates(conn, query, n, space)
    recency_ids = _recency_candidates(conn, n, space)
    seeds = (vector_ids[:5] + keyword_ids[:5])
    graph_ids = _graph_candidates(conn, seeds, n, space)

    fused = _rrf_fuse(
        {"vector": vector_ids, "keyword": keyword_ids, "recency": recency_ids, "graph": graph_ids},
        weights,
        settings.rrf_k,
    )

    if not fused:
        return []

    ids = [note_id for note_id, _ in fused]
    placeholders = ",".join("?" for _ in ids)
    sql = f"SELECT * FROM notes WHERE id IN ({placeholders}) AND deleted_at IS NULL"
    params: list = list(ids)
    if space is not None:
        sql += " AND space = ?"
        params.append(space)
    rows = conn.execute(sql, params).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}

    boosted = []
    for note_id, rrf_score in fused:
        note = by_id.get(note_id)
        if note is None:
            continue
        # A gentle tiebreaker, not a relevance override: importance boosts ranking among
        # similarly-relevant notes, but must not let a high-importance/low-relevance note
        # outrank a genuinely on-topic one (RRF score differences are often tiny).
        importance_multiplier = 1.0 + (note["importance"] - 3) * 0.03
        note["_score"] = rrf_score * importance_multiplier
        boosted.append(note)

    boosted.sort(key=lambda n: n["_score"], reverse=True)
    top = boosted[:top_k]

    _bump_access(conn, [n["id"] for n in top])
    return top
