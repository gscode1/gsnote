from app import capture
from app.config import get_settings
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


# --- query prefix reaches only the query side -------------------------------------------------


def test_stored_notes_are_embedded_without_prefix_and_queries_with_it(monkeypatch):
    from app import embeddings

    seen: list[str] = []

    def fake_embed_many(texts, task=None):
        seen.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]

    monkeypatch.setattr(embeddings, "embed_many", fake_embed_many)
    capture.capture_note("Pay the electricity bill", source="alice")
    search("electricity bill")

    assert seen == ["Pay the electricity bill", embeddings.BGE_QUERY_PREFIX + "electricity bill"]


def test_e5_model_prefixes_stored_notes_and_imports_with_passage_and_queries_with_query(monkeypatch):
    from app import embeddings
    from app.config import get_settings

    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    get_settings.cache_clear()
    seen: list[str] = []

    def fake_embed_many(texts, task=None):
        seen.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]

    monkeypatch.setattr(embeddings, "embed_many", fake_embed_many)
    capture.capture_note("Kup mleko", source="alice")
    capture.import_notes({"version": 2, "notes": [{"id": "n1", "content": "Zadzwon do mamy"}]})
    search("mleko")

    assert seen == ["passage: Kup mleko", "passage: Zadzwon do mamy", "query: mleko"]


# --- relevance gate -------------------------------------------------------------------------


NOISE = [
    "Buy oat milk, eggs and bananas at the grocery store",
    "Renew the passport before the trip to Japan in spring",
    "Idea: gamify note capture with streaks",
    "Call the dentist to reschedule next week's appointment",
    "Pick up the dry cleaning before Friday",
    "Intention: start running three times a week",
]
ROADMAP = "Meeting about the Q3 roadmap: ship the MCP server first, then the web UI"


def _seed_roadmap_and_noise():
    _seed([(ROADMAP, "meeting"), *[(text, "note") for text in NOISE]])


def test_gate_drops_unrelated_notes_but_keeps_the_relevant_one(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_GATE", "false")
    get_settings.cache_clear()
    _seed_roadmap_and_noise()
    ungated = search("plans for the product roadmap", top_k=20)
    assert len(ungated) == 1 + len(NOISE)  # without the gate every note comes back

    monkeypatch.setenv("RETRIEVAL_GATE", "true")
    get_settings.cache_clear()
    gated = search("plans for the product roadmap", top_k=20)
    assert gated[0]["content"] == ROADMAP and 0.0 < gated[0]["similarity"] <= 1.0
    assert len(gated) < len(ungated)  # the default (lenient) gate already drops the weakest notes

    monkeypatch.setenv("RETRIEVAL_RELATIVE_GAP", "0.1")
    get_settings.cache_clear()
    assert [r["content"] for r in search("plans for the product roadmap", top_k=20)] == [ROADMAP]


def test_gate_param_overrides_the_setting(monkeypatch):
    _seed_roadmap_and_noise()  # setting default: gate on
    q = "plans for the product roadmap"
    gated = search(q, top_k=20)
    assert len(gated) < 1 + len(NOISE)
    assert search(q, top_k=20, gate=None) == gated  # None follows the setting
    ungated = search(q, top_k=20, gate=False)
    assert len(ungated) == 1 + len(NOISE)  # broad: nothing dropped
    assert ungated[0]["content"] == ROADMAP and [r["id"] for r in ungated[: len(gated)]] == [r["id"] for r in gated]
    assert [r["id"] for r in search(q, top_k=20, gate=True)] == [r["id"] for r in gated]

    monkeypatch.setenv("RETRIEVAL_GATE", "false")
    get_settings.cache_clear()
    assert len(search(q, top_k=20)) == 1 + len(NOISE)
    assert len(search(q, top_k=20, gate=True)) < 1 + len(NOISE)  # forced on despite the setting


def test_gate_can_be_disabled_by_setting(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_GATE", "false")
    get_settings.cache_clear()
    _seed_roadmap_and_noise()
    assert len(search("plans for the product roadmap", top_k=20)) == 1 + len(NOISE)


def test_gate_min_similarity_setting_is_a_floor(monkeypatch):
    _seed_roadmap_and_noise()
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "0.99")
    get_settings.cache_clear()
    assert search("plans for the product roadmap", top_k=20) == []  # nothing is that similar: abstain


def test_top_k_stays_a_cap_when_the_gate_keeps_more():
    _seed([(f"Quarterly roadmap planning meeting, take {i}", "meeting") for i in range(6)])
    assert len(search("quarterly roadmap planning meeting", top_k=20)) == 6
    assert len(search("quarterly roadmap planning meeting", top_k=3)) == 3


def test_empty_query_listing_ignores_the_gate(monkeypatch):
    _seed_roadmap_and_noise()
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "0.99")
    get_settings.cache_clear()
    listed = search(top_k=20)
    assert len(listed) == 1 + len(NOISE)
    assert all(r["score"] is None and r["similarity"] is None for r in listed)
    assert len(search("   ", top_k=20)) == 1 + len(NOISE)
    assert len(search(space="default", category="meeting", top_k=20)) == 1


def test_gate_respects_filters(monkeypatch):
    work = capture.capture_note(ROADMAP, source="alice", space="work")
    capture.capture_note(NOISE[0], source="alice", space="work")
    capture.capture_note(ROADMAP + " (copy)", source="alice", space="home")
    results = search("plans for the product roadmap", space="work", top_k=20)
    assert [r["id"] for r in results] == [work["id"]]


def test_keyword_match_of_every_query_term_survives_the_gate(monkeypatch):
    """A note with every query term is exempt from the similarity cutoff (rare names, IDs)."""
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "0.99")
    get_settings.cache_clear()
    hit = capture.capture_note("Keychron Q1 purchase receipt", source="alice")
    capture.capture_note(NOISE[0], source="alice")
    results = search("keychron", top_k=20)
    assert [r["id"] for r in results] == [hit["id"]]


def test_gate_results_unit():
    from app.retrieval import gate_results

    def rs(*sims):
        return [{"id": f"n{i}", "similarity": s} for i, s in enumerate(sims)]

    ids = lambda results: [r["id"] for r in results]
    assert ids(gate_results(rs(0.9, 0.8, 0.6), min_similarity=0.0, relative_gap=0.2)) == ["n0", "n1"]
    assert ids(gate_results(rs(0.9, 0.8, 0.6), min_similarity=0.85, relative_gap=0.0)) == ["n0"]
    assert ids(gate_results(rs(0.4, 0.38), min_similarity=0.45, relative_gap=0.2)) == []
    # both checks 0 -> disabled; no similarity -> nothing to judge, kept
    assert ids(gate_results(rs(0.9, 0.1), min_similarity=0.0, relative_gap=0.0)) == ["n0", "n1"]
    assert ids(gate_results([{"id": "x", "similarity": None}], min_similarity=0.9, relative_gap=0.1)) == ["x"]
    # exempt notes survive, and an order-preserving result is kept
    assert ids(gate_results(rs(0.9, 0.2, 0.85), min_similarity=0.5, relative_gap=0.1, exempt={"n1"})) == [
        "n0", "n1", "n2"
    ]


def test_min_similarity_auto_default_is_for_bge_only(monkeypatch):
    from app.retrieval import _min_similarity

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    get_settings.cache_clear()
    assert _min_similarity(get_settings()) == 0.45
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    get_settings.cache_clear()
    assert _min_similarity(get_settings()) == 0.0
    monkeypatch.setenv("RETRIEVAL_MIN_SIMILARITY", "0.3")
    get_settings.cache_clear()
    assert _min_similarity(get_settings()) == 0.3
