"""One conversation turn: message, command, nudge response.

Owns the memory conversation agent and its in-memory history: the agent's tools
(save/search/list) are the turn's way into capture, retrieval, and reporting.
Notification identity rides in channel button payloads (not process memory),
so digest responses survive restart.
"""
from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from app import reminders
from app.capture import VALID_CATEGORIES, capture_note
from app.config import get_settings
from app.llm import build_model
from app.prompts import MEMORY, prompt
from app.reporting import notes_in_window
from app.resurfacing import Digest, record_response
from app.retrieval import search
from app import spaces
from app.spaces import get_space, list_spaces, set_space

if TYPE_CHECKING:
    from app.channels import Channel

logger = logging.getLogger(__name__)

# Short-term conversation context window (in pydantic-ai messages, not user turns —
# one tool-using turn is ~4 messages). Long-term memory lives in the DB via tools.
HISTORY_KEEP_MESSAGES = 20

# user_id -> recent pydantic-ai message history
_history_by_user: dict[str, list] = {}


def reset_for_tests() -> None:
    _history_by_user.clear()


def trim_history(messages: list[ModelMessage], keep: int = HISTORY_KEEP_MESSAGES) -> list[ModelMessage]:
    """Keep only the most recent turns, trimming on a clean boundary.

    A naive `messages[-keep:]` can start the window in the middle of a tool
    call/return pair, leaving an orphaned ToolReturnPart that some providers reject.
    We slice to the last `keep`, then advance to the first message that begins a
    fresh user turn (a ModelRequest carrying a UserPromptPart), so history always
    re-enters cleanly.
    """
    if len(messages) <= keep:
        return messages
    # Forward-search from the start of the kept window for the first clean user-turn
    # boundary. If the window contains none (only a partial tool sequence), return no
    # history — empty is always valid, malformed (orphan tool-return) is not.
    for i in range(len(messages) - keep, len(messages)):
        m = messages[i]
        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts):
            return messages[i:]
    return []


@dataclass
class NoteDeps:
    """Per-request dependencies injected into the memory agent's tools."""

    user_id: str
    space: str = "default"  # active note space for this user's turn


@lru_cache
def memory_agent() -> Agent[NoteDeps, str]:
    """Single agent with tools; the model decides whether to store or retrieve.

    Instantiated once and reused; safe to run concurrently with different `deps`.
    """
    settings = get_settings()
    agent: Agent[NoteDeps, str] = Agent(
        build_model(settings.answer_model),
        deps_type=NoteDeps,
        output_type=str,
        # Trim each model request to recent turns on a clean boundary, so the context
        # window stays bounded regardless of how long the conversation runs.
        capabilities=[ProcessHistory(lambda msgs: trim_history(msgs))],
        # Callable so data/prompts/memory.md overrides are re-read each run.
        # Today's date is injected so the model can resolve relative dates
        # ("next Tuesday") into tool arguments.
        instructions=lambda: (
            prompt("memory", MEMORY)
            + f"\nToday is {datetime.now().strftime('%A, %Y-%m-%d')}."
        ),
    )

    @agent.tool
    async def save_note(ctx: RunContext[NoteDeps], content: str) -> str:
        """Store a new note for the user. Use when the user gives you something to remember.

        Args:
            content: The note text to save (the user's new information).
        """
        if not content.strip():
            raise ModelRetry("Cannot save an empty note. Ask the user what to record.")
        note = await capture_note(content, source=ctx.deps.user_id, space=ctx.deps.space)
        # Return the stored (normalized) text so the model can confirm exactly what
        # went into memory — important since we lightly normalize, and for voice input.
        return (
            f"Saved as {note['category']} (importance {note['importance']}). "
            f"Stored text: \"{note['content']}\""
        )

    @agent.tool
    async def search_notes(ctx: RunContext[NoteDeps], query: str, limit: int = 8) -> str:
        """Search the user's stored notes by topic/meaning (hybrid semantic + keyword).

        Args:
            query: What to look for, in natural language.
            limit: Max number of notes to return.
        """
        if not query.strip():
            raise ModelRetry("Search query is empty. Provide a topic or keyword to search for.")
        notes = search(query, top_k=limit, space=ctx.deps.space, owner=ctx.deps.user_id)
        if not notes:
            return "No matching notes found."
        return "\n".join(f"- [{n['category']}] {n['content']} ({n['created_at']})" for n in notes)

    @agent.tool
    async def list_recent_notes(ctx: RunContext[NoteDeps], days: int = 7, category: str | None = None) -> str:
        """List all notes saved in the last N days. Use for 'everything saved' or time-window requests.

        Args:
            days: How many days back to include.
            category: Optional filter — one of idea, intention, meeting, task, note.
        """
        notes = notes_in_window(days, category, space=ctx.deps.space, owner=ctx.deps.user_id)
        if not notes:
            return f"No notes found in the last {days} day(s)."
        return "\n".join(f"- [{n['category']}] {n['content']} ({n['created_at']})" for n in notes)

    @agent.tool
    async def list_spaces(ctx: RunContext[NoteDeps]) -> str:
        """List the user's note spaces and which one is currently active."""
        active = get_space(ctx.deps.user_id)
        # spaces.list_spaces: the module-level import is shadowed by this tool's name.
        listing = ", ".join(f"{s} (active)" if s == active else s for s in spaces.list_spaces(ctx.deps.user_id))
        return f"Spaces: {listing}. You cannot switch spaces — tell the user to send /space <name>."

    @agent.tool
    async def get_current_space(ctx: RunContext[NoteDeps]) -> str:
        """Report the user's currently active note space."""
        return get_space(ctx.deps.user_id)

    @agent.tool
    async def create_reminder(
        ctx: RunContext[NoteDeps],
        message: str,
        kind: str,
        weekday: int | None = None,
        fire_date: str | None = None,
        window_days: int | None = None,
        category: str | None = None,
    ) -> str:
        """Schedule a reminder delivered at the daily reminder hour. Use when the user
        asks to be reminded. For "remind me about my <category> notes from the last
        N days", set window_days (and category) so the reminder includes those notes.

        Args:
            message: What to remind about, in the user's own words.
            kind: 'once' (needs fire_date), 'daily', or 'weekly' (needs weekday).
            weekday: 0=Monday .. 6=Sunday; required when kind='weekly'.
            fire_date: YYYY-MM-DD; required when kind='once'; today or later.
            window_days: If set, attach the user's notes from the last N days.
            category: Optional filter — one of idea, intention, meeting, task, note.
        """
        if not message.strip():
            raise ModelRetry("Cannot set a reminder with an empty message. Ask what to remind about.")
        if kind not in reminders.KINDS:
            raise ModelRetry("kind must be one of: once, daily, weekly")
        if window_days is not None and window_days < 1:
            raise ModelRetry("window_days must be at least 1.")
        if kind == "weekly" and (weekday is None or not 0 <= weekday <= 6):
            raise ModelRetry("weekly reminders need weekday 0 (Monday) .. 6 (Sunday)")
        if kind == "once":
            try:
                when = date.fromisoformat(fire_date or "")
            except ValueError:
                raise ModelRetry("once reminders need fire_date as YYYY-MM-DD")
            if when < date.today():
                raise ModelRetry(f"{fire_date} is in the past — today is {date.today()}")
        if category is not None and category not in VALID_CATEGORIES:
            raise ModelRetry(f"category must be one of {sorted(VALID_CATEGORIES)}")
        rid = reminders.create_reminder(
            ctx.deps.user_id, ctx.deps.space, message.strip(), kind,
            weekday=weekday, fire_date=fire_date,
            window_days=window_days, category=category,
        )
        return f"Reminder set ({kind}, id {rid}). It fires at the morning reminder hour."

    @agent.tool
    async def list_reminders(ctx: RunContext[NoteDeps]) -> str:
        """List the user's active reminders with their ids."""
        rows = reminders.list_reminders(ctx.deps.user_id)
        if not rows:
            return "No active reminders."

        def when(r: dict) -> str:
            if r["kind"] == "weekly":
                return f"weekly on {calendar.day_name[r['weekday']]}"
            if r["kind"] == "once":
                return f"once on {r['fire_date']}"
            return "daily"

        return "\n".join(f"- {r['id']}: {r['message']} ({when(r)})" for r in rows)

    @agent.tool
    async def cancel_reminder(ctx: RunContext[NoteDeps], reminder_id: str) -> str:
        """Cancel one of the user's reminders by id (get ids from list_reminders)."""
        ok = reminders.cancel_reminder(ctx.deps.user_id, reminder_id)
        return "Cancelled." if ok else "No active reminder with that id."

    return agent


async def handle_message(channel: Channel, user_id: str, text: str) -> None:
    space = get_space(user_id)
    history = _history_by_user.get(user_id, [])
    result = await memory_agent().run(
        text, deps=NoteDeps(user_id=user_id, space=space), message_history=history
    )
    _history_by_user[user_id] = trim_history(result.all_messages())
    # Tag with the space captured before the run, so the label matches the operation.
    await channel.send(user_id, f"{result.output} [{space}]")


async def handle_command(user_id: str, command: str, args: str) -> str:
    if command == "space":
        if args:
            return _switch_space(user_id, args)
        active = get_space(user_id)
        others = ", ".join(s for s in list_spaces(user_id) if s != active)
        listing = f" Your spaces: {others}." if others else ""
        return (
            f"Active space: {active}.{listing} "
            "Switch with /space <name> — a new name creates a new space."
        )
    if command == "start":
        return (
            "Hi! I'm your memory bot. Send a note to save it, ask a question to recall, "
            "or send a voice message.\n\n"
            f"Spaces keep your notes apart (current: {get_space(user_id)}):\n"
            "• /space <name> — switch to (or create) a space\n"
            "• /space — show active space and your spaces"
        )
    return "Unknown command. Try /space."


def _switch_space(user_id: str, name: str) -> str:
    try:
        set_space(user_id, name)
    except ValueError as e:
        return str(e)
    space = get_space(user_id)
    _history_by_user.pop(user_id, None)  # don't carry context across a space switch
    return f"Switched to your {space} space. New notes and questions now use {space}."


async def handle_response(user_id: str, response: str, notification_id: str) -> None:
    record_response(notification_id, response)


def make_reminder_sender(channel: Channel | None):
    """send_fn for reminders: plain owner-scoped message, no buttons."""

    async def send_fn(user_id: str, message: str) -> None:
        if channel is None:
            logger.warning("No channel configured; reminder not sent: %s", message)
            return
        await channel.send(user_id, message)

    return send_fn


def make_digest_sender(channel: Channel | None):
    """send_fn for resurfacing: push Digest with buttons to its owning user only."""

    async def send_fn(digest: Digest) -> None:
        if channel is None:
            logger.warning("No channel configured; resurfacing message not sent: %s", digest.message)
            return
        await channel.send(
            digest.user_id,
            digest.message,
            with_nudge_buttons=True,
            notification_id=digest.notification_id,
        )

    return send_fn
