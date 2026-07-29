import pytest

from app import capture
from app.agents import Classification
from app.retrieval import search


async def _seed(monkeypatch, items: list[tuple[str, str, int]]):
    """items: list of (content, category, importance)"""

    async def fake_classify(content: str) -> Classification:
        for c, cat, imp in items:
            if c == content:
                return Classification(category=cat, importance=imp, normalized_content=content)
        return Classification(category="note", importance=3, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)

    notes = []
    for content, _, _ in items:
        notes.append(await capture.capture_note(content, source="alice"))
    return notes


@pytest.mark.asyncio
async def test_keyword_search_finds_lexical_match(monkeypatch):
    await _seed(
        monkeypatch,
        [
            ("Buy oat milk and bananas at the grocery store", "task", 2),
            ("Brainstorm names for the new memory agent product", "idea", 4),
        ],
    )

    results = search("grocery store milk")
    contents = [r["content"] for r in results]
    assert any("oat milk" in c for c in contents)


@pytest.mark.asyncio
async def test_importance_boost_affects_ranking(monkeypatch):
    notes = await _seed(
        monkeypatch,
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


@pytest.mark.asyncio
async def test_search_bumps_access_count(monkeypatch):
    notes = await _seed(monkeypatch, [("Plan a trip to the mountains next spring", "idea", 3)])

    from app.db import get_conn

    before = get_conn().execute(
        "SELECT access_count FROM notes WHERE id = ?", (notes[0]["id"],)
    ).fetchone()["access_count"]

    search("trip to the mountains")

    after = get_conn().execute(
        "SELECT access_count FROM notes WHERE id = ?", (notes[0]["id"],)
    ).fetchone()["access_count"]

    assert after == before + 1


@pytest.mark.asyncio
async def test_graph_expansion_surfaces_related_note(monkeypatch):
    """A note with no lexical/vector overlap with the query, but linked via a temporal edge
    to a note that does match, should be surfaced by the graph candidate generator."""
    notes = await _seed(
        monkeypatch,
        [
            ("Talked to Sam about the database migration plan", "meeting", 4),
            ("Remember to buy a birthday gift", "task", 2),
        ],
    )

    results = search("database migration plan")
    ids = [r["id"] for r in results]
    # both should appear: the direct match and its temporal-backbone neighbor
    assert notes[0]["id"] in ids


@pytest.mark.asyncio
async def test_vector_candidates_respect_space_when_other_space_dominates(monkeypatch):
    """Space-scoped vector search must not starve when another space owns global top-N."""
    from app.config import get_settings
    from app.retrieval import _vector_candidates
    from app.db import get_conn

    settings = get_settings()
    n = settings.retrieval_candidate_n

    async def fake_classify(content: str) -> Classification:
        return Classification(category="note", importance=3, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)

    # Flood personal with near-duplicate semantic notes so they dominate global knn.
    for i in range(n + 5):
        await capture.capture_note(
            f"quarterly roadmap planning meeting notes draft {i}",
            source="alice",
            space="personal",
        )
    work = await capture.capture_note(
        "quarterly roadmap planning meeting for the engineering org",
        source="alice",
        space="work",
    )

    ids = _vector_candidates(get_conn(), "quarterly roadmap planning meeting", n, space="work")
    assert work["id"] in ids
