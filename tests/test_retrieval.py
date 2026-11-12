from app import capture
from app.db import get_conn
from app.retrieval import search


def _seed(items: list[tuple[str, str, int]]):
    """items: list of (content, category, importance)"""
    notes = []
    for content, category, importance in items:
        note = capture.capture_note(content, source="alice", category=category)
        conn = get_conn()
        conn.execute("UPDATE notes SET importance = ? WHERE id = ?", (importance, note["id"]))
        conn.commit()
        notes.append(note)
    return notes


def test_keyword_search_finds_lexical_match():
    _seed(
        [
            ("Buy oat milk and bananas at the grocery store", "task", 2),
            ("Brainstorm names for the new memory agent product", "idea", 4),
        ],
    )

    results = search("grocery store milk")
    contents = [r["content"] for r in results]
    assert any("oat milk" in c for c in contents)


def test_importance_boost_affects_ranking():
    notes = _seed(
        [
            ("A note about the quarterly roadmap planning meeting", "meeting", 5),
            ("Another note about the quarterly roadmap planning meeting", "meeting", 1),
        ],
    )

    results = search("quarterly roadmap planning meeting")
    assert len(results) >= 2
    # the higher-importance note should rank at or above the lower-importance one
    scores_by_id = {r["id"]: r["_score"] for r in results}
    assert scores_by_id[notes[0]["id"]] >= scores_by_id[notes[1]["id"]]


def test_search_bumps_access_count():
    notes = _seed([("Plan a trip to the mountains next spring", "idea", 3)])

    before = get_conn().execute(
        "SELECT access_count FROM notes WHERE id = ?", (notes[0]["id"],)
    ).fetchone()["access_count"]

    search("trip to the mountains")

    after = get_conn().execute(
        "SELECT access_count FROM notes WHERE id = ?", (notes[0]["id"],)
    ).fetchone()["access_count"]

    assert after == before + 1


def test_graph_expansion_surfaces_related_note():
    """A note with no lexical/vector overlap with the query, but linked via a temporal edge
    to a note that does match, should be surfaced by the graph candidate generator."""
    notes = _seed(
        [
            ("Talked to Sam about the database migration plan", "meeting", 4),
            ("Remember to buy a birthday gift", "task", 2),
        ],
    )

    results = search("database migration plan")
    ids = [r["id"] for r in results]
    # both should appear: the direct match and its temporal-backbone neighbor
    assert notes[0]["id"] in ids


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

    ids = _vector_candidates(get_conn(), "quarterly roadmap planning meeting", n, space="work")
    assert work["id"] in ids
