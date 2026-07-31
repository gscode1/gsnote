"""Telegram channel adapter — reference adapter for the OSS release (PRD §10).

Trivial bot token, no ban risk, pushes freely. Runs in the same Python process
(no sidecar needed), unlike Baileys/Signal.
"""
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler as PTBMessageHandler,
    filters,
)

from app.channels import Channel, CommandHandler, MessageHandler, ResponseHandler
from app.config import get_settings

logger = logging.getLogger(__name__)


class TelegramChannel(Channel):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required for the telegram channel")
        self._allowed = settings.allowed_telegram_ids
        # ponytail: fail closed — empty allowlist used to mean "allow all"
        if not self._allowed:
            raise ValueError(
                "TELEGRAM_ALLOWED_USER_IDS is required when CHANNEL=telegram "
                "(comma-separated numeric Telegram user ids)"
            )
        self._app: Application = Application.builder().token(settings.telegram_bot_token).build()
        self._message_handler: MessageHandler | None = None
        self._response_handler: ResponseHandler | None = None
        self._command_handler: CommandHandler | None = None

    def _authorized(self, user_id: int) -> bool:
        return user_id in self._allowed

    def on_message(self, handler: MessageHandler) -> None:
        self._message_handler = handler

    def on_response(self, handler: ResponseHandler) -> None:
        self._response_handler = handler

    def on_command(self, handler: CommandHandler) -> None:
        self._command_handler = handler

    def recipients(self) -> list[str]:
        return [str(uid) for uid in self._allowed]

    async def send(
        self,
        user_id: str,
        message: str,
        *,
        with_nudge_buttons: bool = False,
        notification_id: str | None = None,
    ) -> None:
        reply_markup = None
        if with_nudge_buttons:
            if not notification_id:
                raise ValueError("notification_id is required when with_nudge_buttons=True")
            # ponytail: encode id in callback_data (≤64 bytes); UUID fits
            reply_markup = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("Engaged", callback_data=f"nudge:engaged:{notification_id}"),
                        InlineKeyboardButton("Dismiss", callback_data=f"nudge:dismissed:{notification_id}"),
                        InlineKeyboardButton("Snooze", callback_data=f"nudge:snoozed:{notification_id}"),
                    ]
                ]
            )
        await self._app.bot.send_message(chat_id=int(user_id), text=message, reply_markup=reply_markup)

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message
        if user is None or message is None or message.text is None:
            return
        if not self._authorized(user.id):
            await message.reply_text("Sorry, you're not authorized to use this bot.")
            return
        if self._message_handler:
            await self._message_handler(str(user.id), message.text)

    async def _on_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message
        if user is None or message is None or message.text is None:
            return
        if not self._authorized(user.id):
            await message.reply_text("Sorry, you're not authorized to use this bot.")
            return
        # "/space foo bar" -> command="space", args="foo bar"
        parts = message.text[1:].split(maxsplit=1)
        command = parts[0].split("@")[0].lower()  # strip optional @botname
        args = parts[1] if len(parts) > 1 else ""
        if self._command_handler:
            reply = await self._command_handler(str(user.id), command, args)
            if reply:
                await message.reply_text(reply)

    async def _on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.message
        if user is None or message is None:
            return
        if not self._authorized(user.id):
            await message.reply_text("Sorry, you're not authorized to use this bot.")
            return

        settings = get_settings()
        if not settings.stt_enabled:
            await message.reply_text("Voice input isn't enabled. Please send text for now.")
            return

        voice = message.voice or message.audio
        if voice is None:
            return

        from app.stt import STTError, transcribe

        try:
            tg_file = await context.bot.get_file(voice.file_id)
            audio = bytes(await tg_file.download_as_bytearray())
            transcript = await transcribe(audio)
        except STTError as e:
            logger.warning("transcription failed: %s", e)
            await message.reply_text("Sorry, I couldn't transcribe that voice message.")
            return

        # Show what was heard (STT can mishear), then run the normal pipeline on the text.
        await message.reply_text(f'🎤 "{transcript}"')
        if self._message_handler:
            await self._message_handler(str(user.id), transcript)

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        # nudge:<response>:<notification_id>
        parts = query.data.split(":", 2)
        if len(parts) != 3 or parts[0] != "nudge":
            return
        _, response, notification_id = parts
        user = update.effective_user
        if user and self._response_handler:
            await self._response_handler(str(user.id), response, notification_id)
        if query.message:
            await query.edit_message_reply_markup(reply_markup=None)

    async def start(self) -> None:
        self._app.add_handler(PTBMessageHandler(filters.COMMAND, self._on_command))
        self._app.add_handler(PTBMessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))
        self._app.add_handler(PTBMessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
