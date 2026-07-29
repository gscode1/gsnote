"""Precision@k eval harness (PRD §6, M2).

Seeds a small fixed (query -> expected notes) eval set, then compares:
  - hybrid RRF (vector + keyword + recency + graph)
  - keyword-only (FTS5 BM25)
  - vector-only (sqlite-vec cosine)

Run with: python -m eval.run_eval
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("DB_PATH", "./data/eval.db")

from app import db, graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import cursor, get_conn  # noqa: E402
from app.embeddings import embed  # noqa: E402
from app.retrieval import _keyword_candidates, _vector_candidates, search  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
TOP_K = 5


def seed(eval_set: dict) -> None:
    db.reset_db_for_tests(get_settings().db_path)
    now = datetime.now(timezone.utc)

    for i, note in enumerate(eval_set["notes"]):
        # Spread creation times across days so temporal_proximity (24h window) only links
        # genuinely nearby notes, instead of flooding the graph with every note pair.
        created_at = (now - timedelta(days=len(eval_set["notes"]) - i)).isoformat()
        vector = embed(note["content"])
        with cursor() as cur:
            cur.execute(
                "INSERT INTO notes (id, content, category, importance, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'eval', ?, ?)",
                (note["id"], note["content"], note["category"], note["importance"], created_at, created_at),
            )
            cur.execute("INSERT INTO notes_fts (note_id, content) VALUES (?, ?)", (note["id"], note["content"]))
            cur.execute(
                "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
                (note["id"], json.dumps(vector)),
            )
            graph.build_temporal_edges(cur.connection, note["id"], "eval", created_at)
            graph.build_semantic_edges(cur.connection, note["id"], vector)


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Standard precision@k: hits in the top-k / k (missing slots count as non-relevant)."""
    top = retrieved_ids[:k]
    hits = sum(1 for note_id in top if note_id in expected_ids)
    return hits / k


def run() -> None:
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    seed(eval_set)
    conn = get_conn()

    hybrid_scores, keyword_scores, vector_scores = [], [], []
    graph_only_hits = []

    print(f"{'query':<55} {'hybrid':>8} {'keyword':>8} {'vector':>8}")
    print("-" * 82)

    for q in eval_set["queries"]:
        expected = set(q["expected_ids"])

        hybrid_results = [n["id"] for n in search(q["query"], top_k=TOP_K)]
        keyword_results = _keyword_candidates(conn, q["query"], TOP_K)
        vector_results = _vector_candidates(conn, q["query"], TOP_K)

        p_hybrid = precision_at_k(hybrid_results, expected, TOP_K)
        p_keyword = precision_at_k(keyword_results, expected, TOP_K)
        p_vector = precision_at_k(vector_results, expected, TOP_K)

        hybrid_scores.append(p_hybrid)
        keyword_scores.append(p_keyword)
        vector_scores.append(p_vector)

        # Did hybrid find an expected note that neither keyword nor vector found directly?
        direct = set(keyword_results) | set(vector_results)
        graph_contribution = (set(hybrid_results) & expected) - direct
        if graph_contribution:
            graph_only_hits.append((q["query"], graph_contribution))

        print(f"{q['query'][:53]:<55} {p_hybrid:>8.2f} {p_keyword:>8.2f} {p_vector:>8.2f}")

    print("-" * 82)
    avg = lambda xs: sum(xs) / len(xs)
    print(f"{'AVERAGE':<55} {avg(hybrid_scores):>8.2f} {avg(keyword_scores):>8.2f} {avg(vector_scores):>8.2f}")

    print()
    if graph_only_hits:
        print("Graph expansion surfaced notes neither keyword nor vector found directly:")
        for query, ids in graph_only_hits:
            print(f"  - {query!r}: {sorted(ids)}")
    else:
        print("No query in this run was won purely by graph expansion.")

    print()
    assert avg(hybrid_scores) >= avg(keyword_scores), "hybrid should beat or match keyword-only on average"
    assert avg(hybrid_scores) >= avg(vector_scores), "hybrid should beat or match vector-only on average"
    print("PASS: hybrid RRF >= keyword-only and >= vector-only on precision@%d" % TOP_K)


if __name__ == "__main__":
    run()
