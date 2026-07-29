"""Tests for work/personal space separation."""
import pytest

from app import capture, spaces
from app.agents import Classification
from app.retrieval import search
from app.reporting import notes_in_window


@pytest.fixture(autouse=True)
def _mock_classifier(monkeypatch):
    async def fake_classify(content: str) -> Classification:
        return Classification(category="note", importance=3, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)


def test_active_space_default_and_toggle():
    assert spaces.get_space("u1") == "personal"
    spaces.set_space("u1", "work")
    assert spaces.get_space("u1") == "work"
    spaces.set_space("u1", "personal")
    assert spaces.get_space("u1") == "personal"


def test_set_invalid_space_raises():
    with pytest.raises(ValueError):
        spaces.set_space("u1", "bogus")


@pytest.mark.anyio
async def test_search_is_space_scoped():
    await capture.capture_note("quarterly budget planning deck", source="u1", space="work")
    await capture.capture_note("plan a weekend hiking trip", source="u1", space="personal")

    work_hits = [n["content"] for n in search("plan", space="work")]
    personal_hits = [n["content"] for n in search("plan", space="personal")]

    assert any("budget" in c for c in work_hits)
    assert all("hiking" not in c for c in work_hits)  # personal note must not leak into work
    assert any("hiking" in c for c in personal_hits)
    assert all("budget" not in c for c in personal_hits)


@pytest.mark.anyio
async def test_search_across_all_spaces_when_none():
    await capture.capture_note("quarterly budget planning deck", source="u1", space="work")
    await capture.capture_note("plan a weekend hiking trip", source="u1", space="personal")

    all_hits = [n["content"] for n in search("plan", space=None)]
    assert any("budget" in c for c in all_hits)
    assert any("hiking" in c for c in all_hits)


@pytest.mark.anyio
async def test_window_report_is_space_scoped():
    await capture.capture_note("work task one", source="u1", space="work")
    await capture.capture_note("personal task one", source="u1", space="personal")

    work = [n["content"] for n in notes_in_window(7, space="work")]
    assert work == ["work task one"]
