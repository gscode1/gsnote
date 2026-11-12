"""MCP server: four thin tools over capture/retrieval/spaces, served at /mcp (see app/main.py)."""
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from app import capture, spaces
from app.config import get_settings
from app.retrieval import search

mcp = MCPServer(
    "gsnote",
    instructions=(
        "Personal notes memory. Save things the user wants to remember with save_note; recall them "
        "with search_notes (semantic + keyword, optional space/category/date filters)."
    ),
)


def _present(note: dict) -> dict[str, Any]:
    """Compact note for the caller: always includes the created_at timestamp and local calendar date."""
    created_at = note["created_at"]
    try:
        date = datetime.fromisoformat(created_at).astimezone(ZoneInfo(get_settings().timezone)).date().isoformat()
    except ValueError:
        date = created_at[:10]
    return {
        "id": note["id"],
        "date": date,
        "created_at": created_at,
        "space": note["space"],
        "category": note["category"],
        "content": note["content"],
    }


@mcp.tool()
def save_note(
    content: str,
    space: str = "default",
    category: str = "note",
) -> dict[str, Any]:
    """Save a note to the user's personal notes memory.

    Use it when the user asks to remember, note down or save something. Write the content as a
    self-contained statement (it is recalled later without the conversation around it).

    space: a named collection of notes, such as "work", "personal" or a project name. It is created
    on first use. Use "default" when the user names none.
    category: free-form label. Suggested: note, idea, task, decision, meeting.
    """
    try:
        note = capture.capture_note(content, source="mcp", space=space, category=category)
    except ValueError as e:  # ToolError text reaches the calling LLM; other exceptions are masked
        raise ToolError(str(e)) from e
    return _present(note)


@mcp.tool()
def search_notes(
    query: str | None = None,
    space: str | None = None,
    category: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> dict[str, Any]:
    """Search the user's saved notes by meaning and keywords, optionally filtered; newest first when
    there is no query.

    query: what to look for, in natural language. Omit it to just list notes matching the filters,
    newest first (e.g. "what did I note last week").
    space: only this space (e.g. "work", "personal"). Omit to search ALL spaces; use list_spaces to
    see which exist.
    category: only this category (e.g. note, idea, task, decision, meeting).
    since / until: inclusive date range on when the note was saved. Format YYYY-MM-DD (a whole day) or
    a full ISO-8601 datetime. This server does not understand relative dates: convert phrases like
    "last week" or "in March" into explicit dates yourself, using the `today` and `timezone` fields
    of this tool's result (call once without a range if you need them). Dates are in that timezone.
    limit: maximum number of notes to return.

    Returns today's date and timezone, plus notes (id, date, created_at, space, category, content).
    """
    try:
        notes = search(query, space=space, category=category, since=since, until=until, top_k=limit)
    except ValueError as e:
        raise ToolError(str(e)) from e
    now = datetime.now(ZoneInfo(get_settings().timezone))
    return {
        "today": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "timezone": get_settings().timezone,
        "notes": [_present(n) for n in notes],
    }


@mcp.tool()
def list_spaces() -> list[dict]:
    """List the user's spaces (named collections of notes such as "work" or "personal") with their
    note counts. Use it to pick a space for save_note or search_notes."""
    return spaces.list_spaces()


@mcp.tool()
def delete_note(note_id: str) -> dict[str, Any]:
    """Permanently delete one note by id (ids come from save_note and search_notes results).
    Only on the user's explicit request; it cannot be undone."""
    return {"id": note_id, "deleted": capture.delete_note(note_id)}


def new_session_manager() -> StreamableHTTPSessionManager:
    """A fresh stateless, JSON-response session manager for the HTTP transport.

    A manager can only run() once, so the app lifespan builds one per start. DNS-rebinding
    host checks are off: the endpoint is bearer-token protected and normally sits behind a proxy.
    """
    mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return mcp.session_manager
