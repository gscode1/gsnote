"""Deterministic tests for the single memory agent's tool branching.

Uses FunctionModel to simulate the model choosing a specific tool, so we can verify
store-vs-retrieve wiring without any real LLM call (ALLOW_MODEL_REQUESTS is False).
"""
import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app import capture
from app.capture import Classification
from app.db import get_conn
from app.spaces import set_space
from app.turn import NoteDeps, memory_agent


@pytest.fixture(autouse=True)
def _mock_classifier(monkeypatch):
    """save_note -> capture_note -> classify_note hits a separate LLM agent; stub it out."""

    async def fake_classify(content: str) -> Classification:
        return Classification(category="task", importance=2, normalized_content=content)

    monkeypatch.setattr(capture, "classify_note", fake_classify)


def _first_call_then_text(tool_name: str, args: dict):
    """Build a FunctionModel fn: call `tool_name` on first turn, then return text."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # The agent loop sends the tool result back; on the second model call we finish.
        called = any(
            getattr(p, "part_kind", None) == "tool-return"
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if called:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name, args)])

    return fn


@pytest.mark.anyio
async def test_save_intent_calls_save_note():
    agent = memory_agent()
    with agent.override(model=FunctionModel(_first_call_then_text("save_note", {"content": "buy milk"}))):
        result = await agent.run("remember to buy milk", deps=NoteDeps(user_id="u1"))
    assert result.output == "done"
    # The note was actually persisted by the save_note tool.
    row = get_conn().execute("SELECT content FROM notes").fetchone()
    assert row is not None
    assert "milk" in row["content"].lower()


@pytest.mark.anyio
async def test_ask_intent_calls_search_note():
    agent = memory_agent()
    # Seed a note directly so search has something to find.
    with agent.override(model=FunctionModel(_first_call_then_text("save_note", {"content": "home lab docs need work"}))):
        await agent.run("remember home lab docs", deps=NoteDeps(user_id="u1"))

    captured: dict = {}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called = any(
            getattr(p, "part_kind", None) == "tool-return"
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if called:
            # capture the tool-return content the model would see
            for m in messages:
                for p in getattr(m, "parts", []):
                    if getattr(p, "part_kind", None) == "tool-return":
                        captured["result"] = p.content
            return ModelResponse(parts=[TextPart("answered")])
        return ModelResponse(parts=[ToolCallPart("search_notes", {"query": "home lab"})])

    with agent.override(model=FunctionModel(fn)):
        result = await agent.run("what about my home lab?", deps=NoteDeps(user_id="u1"))

    assert result.output == "answered"
    assert "home lab" in captured["result"].lower()


def _capture_tool_return(tool_name: str, args: dict, captured: dict):
    """FunctionModel fn: call `tool_name`, record the tool-return content, then finish."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called = any(
            getattr(p, "part_kind", None) == "tool-return"
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if called:
            for m in messages:
                for p in getattr(m, "parts", []):
                    if getattr(p, "part_kind", None) == "tool-return":
                        captured["result"] = p.content
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name, args)])

    return fn


@pytest.mark.anyio
async def test_list_spaces_tool_returns_real_state():
    agent = memory_agent()
    # u1 is active in "work" (no notes there) and has notes in "home";
    # u2's "secret" space must not leak into u1's listing.
    set_space("u1", "work")
    await capture.capture_note("fix the tap", source="u1", space="home")
    await capture.capture_note("u2 private stuff", source="u2", space="secret")

    captured: dict = {}
    with agent.override(model=FunctionModel(_capture_tool_return("list_spaces", {}, captured))):
        await agent.run("which spaces do I have?", deps=NoteDeps(user_id="u1"))

    assert "home" in captured["result"]
    assert "work (active)" in captured["result"]
    assert "secret" not in captured["result"]


@pytest.mark.anyio
async def test_get_current_space_tool_defaults_and_switch():
    agent = memory_agent()

    captured: dict = {}
    with agent.override(model=FunctionModel(_capture_tool_return("get_current_space", {}, captured))):
        await agent.run("which space am I in?", deps=NoteDeps(user_id="u1"))
    assert captured["result"] == "default"

    set_space("u1", "work")
    captured.clear()
    with agent.override(model=FunctionModel(_capture_tool_return("get_current_space", {}, captured))):
        await agent.run("which space am I in?", deps=NoteDeps(user_id="u1"))
    assert captured["result"] == "work"
