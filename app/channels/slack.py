"""Slack channel adapter (PRD §10).

Socket Mode: outbound websocket, no public webhook — same deployment shape as
Telegram polling. Proactive push is unrestricted (chat.postMessage anytime),
nudge buttons are Block Kit actions. DMs only: a personal bot doesn't need
channel routing.
"""
import asyncio
import logging

import httpx
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

from app.channels import Channel, CommandHandler, MessageHandler, ResponseHandler
from app.config import get_settings

logger = logging.getLogger(__name__)

_NUDGE_ACTIONS = [("Engaged", "engaged"), ("Dismiss", "dismissed"), ("Snooze", "snoozed")]


class SlackChannel(Channel):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.slack_bot_token or not settings.slack_app_token:
            raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required for the slack channel")
        self._bot_token = settings.slack_bot_token
        self._allowed = settings.allowed_slack_ids
        # ponytail: fail closed — empty allowlist must not mean "allow all"
        if not self._allowed:
            raise ValueError(
                "SLACK_ALLOWED_USER_IDS is required when CHANNEL=slack "
                "(comma-separated Slack user ids, e.g. U0123ABCDEF)"
            )
        self._app = AsyncApp(token=settings.slack_bot_token)
        self._socket = AsyncSocketModeHandler(self._app, settings.slack_app_token)
        self._message_handler: MessageHandler | None = None
        self._response_handler: ResponseHandler | None = None
        self._command_handler: CommandHandler | None = None

        @self._app.event("message")
        async def _message(event: dict) -> None:
            # bolt acks only after the listener returns, and Slack retries unacked
            # events after ~3s — an LLM turn takes longer, so process in the
            # background or every note gets saved twice.
            task = asyncio.create_task(self._on_message(event))
            task.add_done_callback(self._log_task_failure)

        @self._app.command("/space")
        async def _space(ack, command, respond) -> None:
            await ack()
            user_id = command.get("user_id")
            if user_id not in self._allowed:
                await respond("Sorry, you're not authorized to use this bot.")
                return
            if self._command_handler:
                reply = await self._command_handler(user_id, "space", (command.get("text") or "").strip())
                if reply:
                    await respond(reply)

        @self._app.action("nudge")
        async def _nudge(ack, body, client) -> None:
            await ack()
            await self._on_nudge(body, client)

    @staticmethod
    def _log_task_failure(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            logger.error("message handling failed", exc_info=task.exception())

    def on_message(self, handler: MessageHandler) -> None:
        self._message_handler = handler

    def on_response(self, handler: ResponseHandler) -> None:
        self._response_handler = handler

    def on_command(self, handler: CommandHandler) -> None:
        self._command_handler = handler

    def recipients(self) -> list[str]:
        return list(self._allowed)

    async def send(
        self,
        user_id: str,
        message: str,
        *,
        with_nudge_buttons: bool = False,
        notification_id: str | None = None,
    ) -> None:
        blocks = None
        if with_nudge_buttons:
            if not notification_id:
                raise ValueError("notification_id is required when with_nudge_buttons=True")
            # ponytail: response:id goes in the button value (≤2000 chars); UUID fits
            blocks = [
                {"type": "section", "text": {"type": "mrkdwn", "text": message}},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": label},
                            "action_id": "nudge",
                            "value": f"{response}:{notification_id}",
                        }
                        for label, response in _NUDGE_ACTIONS
                    ],
                },
            ]
        # chat.postMessage accepts a user id as channel and opens the DM
        await self._app.client.chat_postMessage(channel=user_id, text=message, blocks=blocks)

    async def _on_message(self, event: dict) -> None:
        if event.get("channel_type") != "im":
            return
        # file_share = audio clip upload; drop edits/deletes/etc.
        subtype = event.get("subtype")
        if event.get("bot_id") or (subtype is not None and subtype != "file_share"):
            return
        user_id = event.get("user")
        if user_id is None:
            return
        if user_id not in self._allowed:
            await self.send(user_id, "Sorry, you're not authorized to use this bot.")
            return

        files = event.get("files") or []
        audio = next((f for f in files if f.get("mimetype", "").startswith("audio/")), None)
        if audio is not None:
            await self._on_voice(user_id, audio)
            return

        text = (event.get("text") or "").strip()
        if not text:
            return
        # ponytail: no typed-"/" branch — Slack intercepts "/cmd" client-side and
        # never posts it as a message; /space arrives via the slash-command handler.
        if self._message_handler:
            await self._message_handler(user_id, text)

    async def _on_voice(self, user_id: str, file_info: dict) -> None:
        if not get_settings().stt_enabled:
            await self.send(user_id, "Voice input isn't enabled. Please send text for now.")
            return
        url = file_info.get("url_private_download") or file_info.get("url_private")
        if not url:
            return

        from app.stt import STTError, transcribe

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url, headers={"Authorization": f"Bearer {self._bot_token}"}, timeout=60
                )
                resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("voice download failed: %s", e)
            await self.send(user_id, "Sorry, I couldn't download that voice message.")
            return

        try:
            # Slack clips are usually m4a — pass the real type, not the ogg default
            transcript = await transcribe(
                resp.content,
                filename=file_info.get("name") or "clip",
                content_type=file_info.get("mimetype") or "audio/ogg",
            )
        except STTError as e:
            logger.warning("transcription failed: %s", e)
            await self.send(user_id, "Sorry, I couldn't transcribe that voice message.")
            return

        # Show what was heard (STT can mishear), then run the normal pipeline on the text.
        await self.send(user_id, f'🎤 "{transcript}"')
        if self._message_handler:
            await self._message_handler(user_id, transcript)

    async def _on_nudge(self, body: dict, client) -> None:
        user_id = body.get("user", {}).get("id")
        if user_id not in self._allowed:
            return
        try:
            response, notification_id = body["actions"][0]["value"].split(":", 1)
        except (KeyError, IndexError, ValueError):
            return
        if self._response_handler:
            await self._response_handler(user_id, response, notification_id)
        # Drop the buttons, keep the text (same as Telegram clearing reply_markup).
        message = body.get("message", {})
        channel = body.get("channel", {})
        if message.get("ts") and channel.get("id"):
            await client.chat_update(
                channel=channel["id"], ts=message["ts"], text=message.get("text", ""), blocks=[]
            )

    async def start(self) -> None:
        await self._socket.connect_async()

    async def stop(self) -> None:
        await self._socket.close_async()
