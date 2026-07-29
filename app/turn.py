"""One conversation turn: message, command, nudge response.

Owns in-memory history. Notification identity rides in channel button payloads
(not process memory), so digest responses survive restart.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.resurfacing import Digest, record_response
from app.spaces import VALID_SPACES, get_space, set_space

if TYPE_CHECKING:
    from app.channels import Channel

logger = logging.getLogger(__name__)

# user_id -> recent pydantic-ai message history
_history_by_user: dict[str, list] = {}


def reset_for_tests() -> None:
    _history_by_user.clear()


async def handle_message(channel: Channel, user_id: str, text: str) -> None:
    from app.agents import NoteDeps, memory_agent, trim_history

    space = get_space(user_id)
    history = _history_by_user.get(user_id, [])
    result = await memory_agent().run(
        text, deps=NoteDeps(user_id=user_id, space=space), message_history=history
    )
    _history_by_user[user_id] = trim_history(result.all_messages())
    await channel.send(user_id, result.output)


async def handle_command(user_id: str, command: str, args: str) -> str:
    if command in VALID_SPACES:  # /work, /personal
        set_space(user_id, command)
        _history_by_user.pop(user_id, None)  # don't carry context across a space switch
        return f"Switched to your {command} space. New notes and questions now use {command}."
    if command == "space":
        return f"Active space: {get_space(user_id)}. Switch with /work or /personal."
    if command == "start":
        return (
            "Hi! I'm your memory bot. Send a note to save it, ask a question to recall, "
            "or send a voice message.\n\n"
            f"Spaces keep work and personal separate (current: {get_space(user_id)}):\n"
            "• /work — switch to work\n"
            "• /personal — switch to personal\n"
            "• /space — show current space"
        )
    return "Unknown command. Try /work, /personal, /space."


async def handle_response(user_id: str, response: str, notification_id: str) -> None:
    record_response(notification_id, response)


def make_digest_sender(channel: Channel | None):
    """send_fn for resurfacing: push Digest with buttons via the channel's recipients."""

    async def send_fn(digest: Digest) -> None:
        if channel is None:
            logger.warning("No channel configured; resurfacing message not sent: %s", digest.message)
            return
        targets = channel.recipients()
        if not targets:
            logger.warning("Channel has no digest recipients; not sent: %s", digest.message)
            return
        for uid in targets:
            await channel.send(
                uid,
                digest.message,
                with_nudge_buttons=True,
                notification_id=digest.notification_id,
            )

    return send_fn
