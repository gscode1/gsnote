"""Tests for boundary-safe conversation-history trimming."""
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from app.turn import trim_history


def _user_turn(text: str) -> list:
    """A full tool-using turn: user -> tool call -> tool return -> assistant text."""
    return [
        ModelRequest(parts=[UserPromptPart(content=text)]),
        ModelResponse(parts=[ToolCallPart("search_notes", {"query": text}, tool_call_id="t1")]),
        ModelRequest(parts=[ToolReturnPart("search_notes", "result", tool_call_id="t1")]),
        ModelResponse(parts=[TextPart("answer")]),
    ]


def test_short_history_unchanged():
    msgs = _user_turn("hello")
    assert trim_history(msgs, keep=20) == msgs


def test_trim_starts_on_clean_user_turn():
    # Five turns (20 messages); keep=6 would naively start mid tool-pair.
    msgs = sum((_user_turn(f"q{i}") for i in range(5)), [])
    trimmed = trim_history(msgs, keep=6)
    # Must begin with a ModelRequest carrying a UserPromptPart, never an orphan tool-return.
    first = trimmed[0]
    assert isinstance(first, ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in first.parts)


def test_no_orphan_tool_return_at_head():
    msgs = sum((_user_turn(f"q{i}") for i in range(5)), [])
    for keep in range(2, 16):
        trimmed = trim_history(msgs, keep=keep)
        if not trimmed:
            continue  # empty history is always valid
        head = trimmed[0]
        # The head must not be a tool-return continuation.
        if isinstance(head, ModelRequest):
            assert not any(isinstance(p, ToolReturnPart) for p in head.parts), f"orphan at keep={keep}"
