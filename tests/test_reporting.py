from datetime import datetime, timedelta, timezone

import pytest

from app.db import cursor
from app.reporting import notes_in_window, parse_window_days, report


def _insert_raw_note(content: str, category: str, created_at: str) -> str:
    import uuid

    note_id = str(uuid.uuid4())
    with cursor() as cur:
        cur.execute(
            "INSERT INTO notes (id, content, category, importance, source, space, created_at, updated_at) "
            "VALUES (?, ?, ?, 3, 'alice', 'default', ?, ?)",
            (note_id, content, category, created_at, created_at),
        )
    return note_id


def test_parse_window_days_recognizes_phrases():
    assert parse_window_days("summarize my ideas from last week") == 7
    assert parse_window_days("what happened last month") == 30
    assert parse_window_days("recap last 3 days") == 3
    assert parse_window_days("just a normal question") == 7  # default


def test_notes_in_window_excludes_old_note():
    now = datetime.now(timezone.utc)
    recent_id = _insert_raw_note("Recent idea about search quality", "idea", now.isoformat())
    old_id = _insert_raw_note(
        "Old idea from two months ago", "idea", (now - timedelta(days=60)).isoformat()
    )

    results = notes_in_window(days=7)
    ids = [r["id"] for r in results]

    assert recent_id in ids
    assert old_id not in ids


def test_notes_in_window_filters_by_category():
    now = datetime.now(timezone.utc)
    idea_id = _insert_raw_note("An idea note", "idea", now.isoformat())
    task_id = _insert_raw_note("A task note", "task", now.isoformat())

    results = notes_in_window(days=7, category="idea")
    ids = [r["id"] for r in results]

    assert idea_id in ids
    assert task_id not in ids


@pytest.mark.anyio
async def test_report_synthesizes_via_llm(monkeypatch):
    now = datetime.now(timezone.utc)
    _insert_raw_note("Idea: build a better note app", "idea", now.isoformat())

    class FakeResult:
        output = "You captured 1 idea last week: a better note app."

    class FakeAgent:
        async def run(self, prompt):
            assert "Idea: build a better note app" in prompt
            return FakeResult()

    import app.reporting as reporting_module

    monkeypatch.setattr(reporting_module, "answer_agent", lambda: FakeAgent())

    summary = await report("summarize my ideas from last week")
    assert "idea" in summary.lower()


@pytest.mark.anyio
async def test_report_handles_empty_window():
    summary = await report("summarize last week")
    assert "No notes found" in summary
