"""Temporal / reporting queries (PRD §6 temporal, M3): native SQL date window + LLM synthesis."""
import re

from app.agents import answer_agent
from app.db import get_conn
from app.spaces import scope_filter

_WINDOW_PATTERNS = [
    (re.compile(r"last week", re.IGNORECASE), 7),
    (re.compile(r"this week", re.IGNORECASE), 7),
    (re.compile(r"last month", re.IGNORECASE), 30),
    (re.compile(r"this month", re.IGNORECASE), 30),
    (re.compile(r"yesterday", re.IGNORECASE), 1),
    (re.compile(r"today", re.IGNORECASE), 1),
    (re.compile(r"last (\d+) days?", re.IGNORECASE), None),  # captured group is days
]


def parse_window_days(text: str, default_days: int = 7) -> int:
    for pattern, days in _WINDOW_PATTERNS:
        m = pattern.search(text)
        if m:
            if days is None:
                return int(m.group(1))
            return days
    return default_days


def notes_in_window(
    days: int,
    category: str | None = None,
    space: str | None = "default",
    owner: str | None = None,
) -> list[dict]:
    conn = get_conn()
    sql = (
        "SELECT * FROM notes WHERE deleted_at IS NULL "
        "AND created_at >= datetime('now', ?)"
    )
    params: list = [f"-{days} days"]
    if category:
        sql += " AND category = ?"
        params.append(category)
    scope_sql, scope_params = scope_filter(owner, space)
    sql += scope_sql
    params.extend(scope_params)
    sql += " ORDER BY created_at DESC"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


async def report(
    query: str,
    category: str | None = None,
    space: str | None = "default",
    owner: str | None = None,
) -> str:
    days = parse_window_days(query)
    notes = notes_in_window(days, category, space=space, owner=owner)

    if not notes:
        return "No notes found in that time window."

    context = "\n".join(f"- [{n['category']}] {n['content']} ({n['created_at']})" for n in notes)
    prompt = (
        f"Summarize the following notes captured in the last {days} day(s). "
        f"Be concise and group by theme if helpful.\n\n{context}"
    )
    result = await answer_agent().run(prompt)
    return result.output
