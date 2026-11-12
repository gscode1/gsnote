"""Hybrid retrieval: vector + keyword (FTS5 BM25) candidates -> plain Reciprocal Rank Fusion -> top-K.

Filters (space, category, since, until) are hard filters applied inside both candidate
generators and in the final fetch. Without a query, search lists the newest matching notes.
"""
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from app.capture import normalize_category
from app.config import get_settings
from app.db import get_conn
from app.embeddings import embed
from app.spaces import normalize_space

# Compare created_at through SQLite datetime(): it normalizes any ISO-8601 string (with or
# without fractional seconds, any UTC offset) to a fixed-width UTC "YYYY-MM-DD HH:MM:SS",
# so plain string comparison is correct. Bounds are inclusive at one-second granularity.
_CREATED = "datetime({col}created_at)"
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_UTC_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class Filters:
    """Normalized note filters. since/until are UTC "YYYY-MM-DD HH:MM:SS" strings, inclusive."""

    space: str | None = None
    category: str | None = None
    since: str | None = None
    until: str | None = None

    def sql(self, alias: str = "") -> tuple[str, list]:
        """` AND ...` clauses for a notes query (empty when no filter is set)."""
        col = f"{alias}." if alias else ""
        sql, params = "", []
        if self.space is not None:
            sql += f" AND {col}space = ?"
            params.append(self.space)
        if self.category is not None:
            sql += f" AND {col}category = ?"
            params.append(self.category)
        if self.since is not None:
            sql += f" AND {_CREATED.format(col=col)} >= ?"
            params.append(self.since)
        if self.until is not None:
            sql += f" AND {_CREATED.format(col=col)} <= ?"
            params.append(self.until)
        return sql, params

    def __bool__(self) -> bool:
        return any(v is not None for v in (self.space, self.category, self.since, self.until))


def _parse_bound(value: str, tz: ZoneInfo, end_of_day: bool) -> datetime:
    """YYYY-MM-DD (whole local day) or ISO-8601 datetime (naive = local) -> UTC, whole seconds."""
    value = value.strip()
    try:
        if _DATE_RE.fullmatch(value):
            dt = datetime.combine(
                date.fromisoformat(value), time(23, 59, 59) if end_of_day else time.min, tzinfo=tz
            )
        else:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc).replace(microsecond=0)
    except (ValueError, OverflowError):
        raise ValueError(f"Invalid date: {value!r} (use YYYY-MM-DD or an ISO-8601 datetime)") from None


def build_filters(
    space: str | None = None,
    category: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> Filters:
    """Normalize raw filter inputs. Raises ValueError on invalid values or since > until."""
    tz = ZoneInfo(get_settings().timezone)
    start = _parse_bound(since, tz, end_of_day=False) if since else None
    end = _parse_bound(until, tz, end_of_day=True) if until else None
    if start and end and start > end:
        raise ValueError(f"since ({since}) is after until ({until})")
    return Filters(
        space=normalize_space(space) if space else None,
        category=normalize_category(category) if category else None,
        since=start.strftime(_UTC_FMT) if start else None,
        until=end.strftime(_UTC_FMT) if end else None,
    )


def _vector_candidates(
    conn: sqlite3.Connection,
    query: str,
    n: int,
    filters: Filters = Filters(),
) -> list[str]:
    vector = embed(query)
    # ponytail: vec0 applies k before SQL filters, so oversample then filter
    fetch_k = min(max(n * 20, n), 500) if filters else n
    rows = conn.execute(
        """
        SELECT note_id FROM note_vectors
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (json.dumps(vector), fetch_k),
    ).fetchall()
    ids = [r["note_id"] for r in rows]
    if not filters:
        return ids[:n]
    if not ids:
        return []
    filter_sql, filter_params = filters.sql()
    placeholders = ",".join("?" for _ in ids)
    kept = {
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM notes WHERE id IN ({placeholders}){filter_sql}",
            (*ids, *filter_params),
        ).fetchall()
    }
    return [i for i in ids if i in kept][:n]


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
    conn: sqlite3.Connection, query: str, n: int, filters: Filters = Filters()
) -> list[str]:
    fts_query = _fts5_query(query)
    if fts_query is None:
        return []
    # FTS5 bm25: lower score = better match -> ascending order. Join notes to filter.
    sql = (
        "SELECT f.note_id, bm25(notes_fts) AS score "
        "FROM notes_fts f JOIN notes n ON n.id = f.note_id "
        "WHERE notes_fts MATCH ?"
    )
    params: list = [fts_query]
    filter_sql, filter_params = filters.sql(alias="n")
    sql += filter_sql
    params.extend(filter_params)
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
    query: str | None = None,
    *,
    space: str | None = None,
    category: str | None = None,
    since: str | None = None,
    until: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Hybrid retrieval over notes matching the filters (space=None = all spaces).

    since/until: YYYY-MM-DD (whole day in the configured TIMEZONE) or an ISO-8601 datetime.
    Raises ValueError on invalid filters. With an empty query, lists the newest matching
    notes (score is None). Otherwise each result carries its fused `score`.
    """
    filters = build_filters(space, category, since, until)
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    conn = get_conn()
    filter_sql, filter_params = filters.sql()

    if not (query and query.strip()):
        rows = conn.execute(
            f"SELECT * FROM notes WHERE 1=1{filter_sql}"
            f" ORDER BY {_CREATED.format(col='')} DESC, created_at DESC LIMIT ?",
            [*filter_params, top_k],
        ).fetchall()
        return [{**dict(r), "score": None} for r in rows]

    n = settings.retrieval_candidate_n
    fused = _rrf_fuse(
        [_vector_candidates(conn, query, n, filters), _keyword_candidates(conn, query, n, filters)],
        settings.rrf_k,
    )
    if not fused:
        return []

    ids = [note_id for note_id, _ in fused]
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM notes WHERE id IN ({placeholders}){filter_sql}", [*ids, *filter_params]
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}

    results = []
    for note_id, score in fused:
        note = by_id.get(note_id)
        if note is not None:
            note["score"] = score
            results.append(note)
    return results[:top_k]
