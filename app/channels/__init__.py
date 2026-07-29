"""Pluggable channel abstraction (PRD §10).

One interface, multiple adapters. The deciding requirement is proactive push:
resurfacing needs the server to *initiate* messages anytime.
"""
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

MessageHandler = Callable[[str, str], Awaitable[None]]  # (user_id, text) -> None
# (user_id, response, notification_id) — notification_id comes from button payload
ResponseHandler = Callable[[str, str, str], Awaitable[None]]
CommandHandler = Callable[[str, str, str], Awaitable[str]]  # (user_id, command, args) -> reply text


class Channel(ABC):
    @abstractmethod
    async def send(
        self,
        user_id: str,
        message: str,
        *,
        with_nudge_buttons: bool = False,
        notification_id: str | None = None,
    ) -> None:
        """Push a message to the user — used for replies and proactive nudges.

        with_nudge_buttons: when True, adapters that support it attach engaged/dismiss/snooze.
        notification_id: encoded into button payloads so responses survive process restart.
        """

    def recipients(self) -> list[str]:
        """Users who should receive proactive digests. Default: none."""
        return []

    @abstractmethod
    def on_message(self, handler: MessageHandler) -> None:
        """Register the handler invoked when a user sends a message in."""

    def on_response(self, handler: ResponseHandler) -> None:
        """Register the handler invoked when a user responds to a nudge (engaged/dismissed/snoozed).

        Default no-op; adapters that support inline buttons/quick-replies override this.
        """

    def on_command(self, handler: CommandHandler) -> None:
        """Register the handler invoked for slash commands (e.g. /work, /personal).

        Default no-op; adapters that support commands override this.
        """

    @abstractmethod
    async def start(self) -> None:
        """Start receiving messages (polling or webhook server)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop receiving messages, release resources."""
