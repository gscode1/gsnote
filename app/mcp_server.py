"""MCP server: four thin tools over capture/retrieval/spaces, served at /mcp (see app/main.py)."""
from datetime import datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

from app import capture, spaces
from app.config import get_settings
from app.retrieval import search

INSTRUCTIONS = """\
Personal notes memory shared across the user's agents.

Saving (save_note):
- Only when the user asks to remember something or confirms a decision or fact worth keeping.
- Never save chit-chat, code, logs, secrets, or what the repo or git history already holds.
- One self-contained fact per note: names and absolute dates, not "this" or "today". Keep the user's language.
- Search first; skip duplicates.
- Space: the one the user names, else the project's default (e.g. from CLAUDE.md or AGENTS.md), else ask.

Recall (search_notes):
- Turn relative dates into since/until using the returned today and timezone.
- If notes conflict, trust the newest and mention the change."""

mcp = MCPServer("gsnote", instructions=INSTRUCTIONS)


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


@mcp.tool(
    title="Save note",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False),
)
def save_note(
    content: Annotated[
        str, Field(description="Self-contained statement with names and absolute dates, in the user's language.")
    ],
    space: Annotated[
        str, Field(description="Collection such as work, personal or a project name; created on first use.")
    ] = "default",
    category: Annotated[
        str, Field(description="Free-form label, e.g. note, idea, task, decision, meeting.")
    ] = "note",
) -> dict[str, Any]:
    """Save one note. Only when the user asks to remember something or confirms a decision worth keeping."""
    try:
        note = capture.capture_note(content, source="mcp", space=space, category=category)
    except ValueError as e:  # ToolError text reaches the calling LLM; other exceptions are masked
        raise ToolError(str(e)) from e
    return _present(note)


@mcp.tool(
    title="Search notes",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
def search_notes(
    query: Annotated[str | None, Field(description="What to find, in natural language. Omit to list.")] = None,
    space: Annotated[str | None, Field(description="Only this space. Omit to search all.")] = None,
    category: Annotated[str | None, Field(description="Only this category.")] = None,
    since: Annotated[
        str | None,
        Field(description="Inclusive start: YYYY-MM-DD or ISO-8601 datetime, in the server timezone."),
    ] = None,
    until: Annotated[
        str | None,
        Field(description="Inclusive end: YYYY-MM-DD or ISO-8601 datetime, in the server timezone."),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=50, description="Maximum notes to return.")] = 10,
) -> dict[str, Any]:
    """Search notes by meaning and keywords. Without a query, list matching notes newest first. Returns today's date and timezone."""
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


@mcp.tool(
    title="List spaces",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
)
def list_spaces() -> list[dict]:
    """List spaces with note counts."""
    return spaces.list_spaces()


@mcp.tool(
    title="Delete note",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False),
)
def delete_note(
    note_id: Annotated[str, Field(description="Note id from save_note or search_notes.")],
) -> dict[str, Any]:
    """Permanently delete a note by id. Only when the user asks."""
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
