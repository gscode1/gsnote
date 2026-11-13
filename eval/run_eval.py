"""Precision@k eval harness (PRD §6, M2).

Seeds a small fixed (query -> expected notes) eval set, then compares:
  - hybrid RRF (vector + keyword)
  - keyword-only (FTS5 BM25)
  - vector-only (sqlite-vec cosine)

Besides precision@k it reports, for the hybrid search as configured, recall and the average
number of results returned (the relevance gate can return fewer than k). Compare the gate
with RETRIEVAL_GATE=false.

Run with: python -m eval.run_eval [path/to/eval_set.json]   (default: eval/eval_set.json;
eval/eval_set_mixed.json is a mixed Polish/English set that includes no-answer queries)
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

from app import db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import cursor, get_conn  # noqa: E402
from app.embeddings import embed_document  # noqa: E402
from app.retrieval import _keyword_candidates, _vector_candidates, search  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
TOP_K = 5


def seed(eval_set: dict) -> None:
    db.reset_db_for_tests(get_settings().db_path)
    now = datetime.now(timezone.utc)

    for i, note in enumerate(eval_set["notes"]):
        created_at = (now - timedelta(days=len(eval_set["notes"]) - i)).isoformat()
        vector = embed_document(note["content"])
        with cursor() as cur:
            cur.execute(
                "INSERT INTO notes (id, content, category, source, created_at, updated_at) "
                "VALUES (?, ?, ?, 'eval', ?, ?)",
                (note["id"], note["content"], note["category"], created_at, created_at),
            )
            cur.execute("INSERT INTO notes_fts (note_id, content) VALUES (?, ?)", (note["id"], note["content"]))
            cur.execute(
                "INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)",
                (note["id"], json.dumps(vector)),
            )


def precision_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> float:
    """Standard precision@k: hits in the top-k / k (missing slots count as non-relevant)."""
    top = retrieved_ids[:k]
    hits = sum(1 for note_id in top if note_id in expected_ids)
    return hits / k


def run(path: Path = EVAL_SET_PATH) -> None:
    eval_set = json.loads(path.read_text())
    seed(eval_set)
    conn = get_conn()

    hybrid_scores, keyword_scores, vector_scores = [], [], []
    recalls, returned, noise_free, set_precisions = [], [], [], []
    no_answer = no_answer_empty = 0

    print(f"{'query':<55} {'hybrid':>8} {'keyword':>8} {'vector':>8}")
    print("-" * 82)

    for q in eval_set["queries"]:
        expected = set(q["expected_ids"])

        hybrid_results = [n["id"] for n in search(q["query"], top_k=TOP_K)]
        returned.append(len(hybrid_results))
        if expected:
            recalls.append(len(expected & set(hybrid_results)) / len(expected))
            set_precisions.append(len(expected & set(hybrid_results)) / len(hybrid_results) if hybrid_results else 0.0)
            noise_free.append(bool(hybrid_results) and set(hybrid_results) <= expected)
        else:
            no_answer += 1
            no_answer_empty += not hybrid_results
        keyword_results = _keyword_candidates(conn, q["query"], TOP_K)
        vector_results = _vector_candidates(conn, q["query"], TOP_K)

        p_hybrid = precision_at_k(hybrid_results, expected, TOP_K)
        p_keyword = precision_at_k(keyword_results, expected, TOP_K)
        p_vector = precision_at_k(vector_results, expected, TOP_K)

        hybrid_scores.append(p_hybrid)
        keyword_scores.append(p_keyword)
        vector_scores.append(p_vector)

        print(f"{q['query'][:53]:<55} {p_hybrid:>8.2f} {p_keyword:>8.2f} {p_vector:>8.2f}")

    print("-" * 82)
    avg = lambda xs: sum(xs) / len(xs)
    print(f"{'AVERAGE':<55} {avg(hybrid_scores):>8.2f} {avg(keyword_scores):>8.2f} {avg(vector_scores):>8.2f}")

    print()
    gate = "on" if get_settings().retrieval_gate else "off"
    print(f"hybrid (relevance gate {gate}): recall@{TOP_K} {avg(recalls):.3f}   "
          f"precision of returned results {avg(set_precisions):.3f}   avg results {avg(returned):.2f}   "
          f"result sets with no noise {sum(noise_free)}/{len(noise_free)}")
    if no_answer:
        print(f"queries without a relevant note: {no_answer_empty}/{no_answer} returned nothing")
    print()
    assert avg(hybrid_scores) >= avg(keyword_scores), "hybrid should beat or match keyword-only on average"
    assert avg(hybrid_scores) >= avg(vector_scores), "hybrid should beat or match vector-only on average"
    print("PASS: hybrid RRF >= keyword-only and >= vector-only on precision@%d" % TOP_K)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_SET_PATH)
