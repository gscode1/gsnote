"""MCP server: tool functions, /mcp auth, and a real streamable-HTTP protocol round trip."""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.server.mcpserver.exceptions import ToolError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from app import capture, mcp_server
from app.config import get_settings
from app.db import get_conn
from app.main import app


# --- tool functions -------------------------------------------------------------------------


def test_save_note_tool_sets_source_and_returns_date():
    out = mcp_server.save_note("Decided to use SQLite", space="Work", category="decision")
    assert out["space"] == "work" and out["category"] == "decision"
    assert out["date"] == out["created_at"][:10]
    row = get_conn().execute("SELECT source FROM notes WHERE id = ?", (out["id"],)).fetchone()
    assert row["source"] == "mcp"


def test_save_note_defaults_and_validation():
    out = mcp_server.save_note("plain note")
    assert (out["space"], out["category"]) == ("default", "note")
    with pytest.raises(ToolError):
        mcp_server.save_note("   ")
    with pytest.raises(ToolError):
        mcp_server.save_note("x", space="!!bad!!")


def test_search_notes_tool_filters_and_reports_today(monkeypatch):
    monkeypatch.setenv("TIMEZONE", "Europe/Warsaw")
    get_settings.cache_clear()
    old = mcp_server.save_note("budget review", space="work")
    new = mcp_server.save_note("budget planning", space="personal")
    get_conn().execute("UPDATE notes SET created_at = '2026-01-10T10:00:00+00:00' WHERE id = ?", (old["id"],))
    get_conn().commit()

    res = mcp_server.search_notes("budget", since="2026-03-01")
    assert res["timezone"] == "Europe/Warsaw"
    assert len(res["today"]) == 10 and res["weekday"]
    assert [n["id"] for n in res["notes"]] == [new["id"]]
    assert set(res["notes"][0]) == {"id", "date", "created_at", "space", "category", "content"}

    assert [n["id"] for n in mcp_server.search_notes("budget", space="work")["notes"]] == [old["id"]]
    assert mcp_server.search_notes(space="work")["notes"][0]["date"] == "2026-01-10"
    assert len(mcp_server.search_notes()["notes"]) == 2  # no query, no space: newest of all spaces
    assert len(mcp_server.search_notes(limit=1)["notes"]) == 1
    with pytest.raises(ToolError):
        mcp_server.search_notes("budget", since="last week")


def test_list_spaces_and_delete_tools():
    n = mcp_server.save_note("a", space="work")
    mcp_server.save_note("b", space="work")
    assert mcp_server.list_spaces() == [{"space": "work", "notes": 2}]
    assert mcp_server.delete_note(n["id"]) == {"id": n["id"], "deleted": True}
    assert mcp_server.delete_note(n["id"]) == {"id": n["id"], "deleted": False}
    assert mcp_server.list_spaces() == [{"space": "work", "notes": 1}]


# --- /mcp auth ------------------------------------------------------------------------------

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "t", "version": "0"}},
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def test_mcp_fails_closed_when_token_unset(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as c:
        assert c.post("/mcp", json=INIT, headers=MCP_HEADERS).status_code == 503


def test_mcp_rejects_missing_and_wrong_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with TestClient(app) as c:
        assert c.post("/mcp", json=INIT, headers=MCP_HEADERS).status_code == 401
        bad = {**MCP_HEADERS, "Authorization": "Bearer wrong"}
        assert c.post("/mcp", json=INIT, headers=bad).status_code == 401
        good = {**MCP_HEADERS, "Authorization": "Bearer s3cret"}
        assert c.post("/mcp", json=INIT, headers=good).status_code == 200


# --- real protocol round trip ---------------------------------------------------------------


@pytest.fixture
def server_url(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


def test_mcp_protocol_list_save_search(server_url):
    async def run():
        http = create_mcp_http_client(headers={"Authorization": "Bearer s3cret"})
        async with http, streamable_http_client(server_url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                assert set(tools) == {"save_note", "search_notes", "list_spaces", "delete_note"}

                saved = await session.call_tool(
                    "save_note",
                    {"content": "Quarterly roadmap meeting moved to Thursday", "space": "work",
                     "category": "meeting"},
                )
                assert not saved.is_error
                assert saved.structured_content["space"] == "work"

                found = await session.call_tool(
                    "search_notes", {"query": "roadmap meeting", "space": "work"}
                )
                assert not found.is_error
                data = found.structured_content
                assert [n["content"] for n in data["notes"]] == [
                    "Quarterly roadmap meeting moved to Thursday"
                ]
                assert data["notes"][0]["date"] == data["today"]
                assert data["timezone"] == "UTC"

                bad = await session.call_tool("search_notes", {"since": "last week"})
                assert bad.is_error
                assert "Invalid date: 'last week'" in bad.content[0].text
                return saved.structured_content["id"]

    note_id = asyncio.run(run())
    assert get_conn().execute("SELECT source FROM notes WHERE id = ?", (note_id,)).fetchone()["source"] == "mcp"
    assert len(capture.get_conn().execute("SELECT 1 FROM notes").fetchall()) == 1


def test_mcp_protocol_guidance_titles_annotations(server_url):
    async def run():
        http = create_mcp_http_client(headers={"Authorization": "Bearer s3cret"})
        async with http, streamable_http_client(server_url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                return init.instructions, tools

    instructions, tools = asyncio.run(run())
    assert "Never save" in instructions and "search_notes" in instructions

    expected = {
        "save_note": ("Save note", dict(readOnlyHint=False, destructiveHint=False, idempotentHint=False)),
        "search_notes": ("Search notes", dict(readOnlyHint=True)),
        "list_spaces": ("List spaces", dict(readOnlyHint=True)),
        "delete_note": ("Delete note", dict(readOnlyHint=False, destructiveHint=True, idempotentHint=True)),
    }
    for name, (title, hints) in expected.items():
        t = tools[name]
        assert t.title == title
        assert t.annotations.model_dump(by_alias=True, exclude_none=True) == {**hints, "openWorldHint": False}

    props = {n: tools[n].input_schema["properties"] for n in ("save_note", "search_notes", "delete_note")}
    for n, params in props.items():
        for pname, schema in params.items():
            assert schema.get("description"), f"{n}.{pname} lacks a description"
    assert props["search_notes"]["limit"]["maximum"] == 50 and props["search_notes"]["limit"]["minimum"] == 1
    assert tools["list_spaces"].input_schema.get("properties", {}) == {}
