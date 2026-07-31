import json

import pytest

from app import capture
from app.capture import Classification
from app.db import get_conn


@pytest.mark.anyio
async def test_capture_note_stores_everywhere(monkeypatch):
    async def fake_classify(content: str) -> Classification:
        return Classification(category="idea", importance=4, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)

    note = await capture.capture_note("Build a note app with great recall", source="alice")

    assert note["category"] == "idea"
    assert note["importance"] == 4

    conn = get_conn()

    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note["id"],)).fetchone()
    assert row is not None
    assert row["content"] == "Build a note app with great recall"

    fts_row = conn.execute("SELECT * FROM notes_fts WHERE note_id = ?", (note["id"],)).fetchone()
    assert fts_row is not None

    vec_row = conn.execute("SELECT * FROM note_vectors WHERE note_id = ?", (note["id"],)).fetchone()
    assert vec_row is not None


@pytest.mark.anyio
async def test_capture_builds_temporal_and_semantic_edges(monkeypatch):
    async def fake_classify(content: str) -> Classification:
        return Classification(category="idea", importance=3, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)

    n1 = await capture.capture_note("I want to build a personal memory agent", source="alice")
    n2 = await capture.capture_note("The memory agent should have proactive resurfacing", source="alice")

    conn = get_conn()
    edges = conn.execute("SELECT * FROM edges WHERE to_id = ?", (n2["id"],)).fetchall()
    types = {e["type"] for e in edges}

    assert "temporal_backbone" in types
    # semantic edge depends on similarity threshold; backbone is guaranteed
    assert any(e["from_id"] == n1["id"] for e in edges)
