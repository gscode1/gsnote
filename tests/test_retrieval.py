from app import capture
from app.db import get_conn
from app.retrieval import Filters, search


def _seed(items: list[tuple[str, str]]):
    """items: list of (content, category)"""
    return [capture.capture_note(content, source="alice", category=category) for content, category in items]


def test_keyword_search_finds_lexical_match():
    _seed(
        [
            ("Buy oat milk and bananas at the grocery store", "task"),
            ("Brainstorm names for the new memory agent product", "idea"),
        ],
    )

    results = search("grocery store milk")
    contents = [r["content"] for r in results]
    assert any("oat milk" in c for c in contents)


def test_results_carry_fused_score_in_descending_order():
    _seed(
        [
            ("A note about the quarterly roadmap planning meeting", "meeting"),
            ("Buy oat milk at the grocery store", "task"),
        ],
    )

    results = search("quarterly roadmap planning meeting")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert "roadmap" in results[0]["content"]


def test_search_does_not_mutate_notes():
    notes = _seed([("Plan a trip to the mountains next spring", "idea")])
    before = dict(get_conn().execute("SELECT * FROM notes WHERE id = ?", (notes[0]["id"],)).fetchone())
    search("trip to the mountains")
    after = dict(get_conn().execute("SELECT * FROM notes WHERE id = ?", (notes[0]["id"],)).fetchone())
    assert before == after


def test_rrf_fuse_is_unweighted():
    from app.retrieval import _rrf_fuse

    fused = dict(_rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == 1 / 62 + 1 / 61
    assert fused["a"] == 1 / 61 and fused["c"] == 1 / 62
    assert max(fused, key=fused.get) == "b"


def test_vector_candidates_respect_space_when_other_space_dominates():
    """Space-scoped vector search must not starve when another space owns global top-N."""
    from app.config import get_settings
    from app.retrieval import _vector_candidates

    settings = get_settings()
    n = settings.retrieval_candidate_n

    # Flood personal with near-duplicate semantic notes so they dominate global knn.
    for i in range(n + 5):
        capture.capture_note(
            f"quarterly roadmap planning meeting notes draft {i}",
            source="alice",
            space="personal",
        )
    work = capture.capture_note(
        "quarterly roadmap planning meeting for the engineering org",
        source="alice",
        space="work",
    )

    ids = _vector_candidates(get_conn(), "quarterly roadmap planning meeting", n, Filters(space="work"))
    assert work["id"] in ids


def test_fts5_query_drops_polish_stopwords():
    from app.retrieval import _fts5_query

    q = _fts5_query("Pomysł NA ogród")
    assert '"Pomysł"' in q and '"ogród"' in q
    assert '"NA"' not in q and '"na"' not in q


def test_polish_stopword_does_not_yield_keyword_candidate():
    from app.retrieval import _keyword_candidates

    notes = _seed(
        [
            ("Zdecydowaliśmy przenieść API na Postgresa", "decision"),
            ("Pomysł: domek dla ptaków w ogrodzie", "idea"),
        ]
    )
    ids = _keyword_candidates(get_conn(), "pomysł na ogród", 10)
    assert notes[0]["id"] not in ids
    assert notes[1]["id"] in ids
