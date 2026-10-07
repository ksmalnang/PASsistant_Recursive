"""Telegram update adapter for PASsistant."""

import logging
from typing import TYPE_CHECKING, Any

from telegram import Bot, Message, Update
from telegram.constants import ChatAction

from src.config.settings import Settings
from src.guardrails.input_guard import InputGuard
from src.telegram_bot.files import (
    TelegramFileDownloadError,
    TelegramFileError,
    TelegramFileTooLargeError,
    TelegramUnsupportedFileTypeError,
    extract_telegram_files,
    get_effective_prompt,
)
from src.telegram_bot.formatting import format_telegram_response, split_telegram_messages

if TYPE_CHECKING:
    from src.api.services import ChatRouteService

logger = logging.getLogger(__name__)
_guard = InputGuard()

WELCOME_MESSAGE = (
    "Halo! 👋 Butuh bantuan soal layanan akademik atau data mahasiswa? Saya juga bisa membaca PDF atau gambar yang Anda kirim, "
    "dengan atau tanpa caption. 📄"
)
UNSUPPORTED_MESSAGE = "Oops! 😅 Saat ini aku cuma bisa baca teks, PDF, atau gambar ya. 📄"
GENERIC_ERROR_MESSAGE = "Waduh, lagi error nih 😅 Coba kirim ulang sebentar lagi ya!"
REJECTED_MESSAGE = (
    "Permintaan itu tidak dapat diproses 🙂 Coba pertanyaan lain yang lebih relevan ya."
)
DOWNLOAD_ERROR_MESSAGE = "I could not download that file from Telegram."
EMPTY_RESPONSE_MESSAGE = "I processed your request but have no response to provide."


class TelegramBotAdapter:
    """Handle Telegram updates using the shared chat service."""

    def __init__(self, chat_service: "ChatRouteService", bot: Bot, settings: Settings):
        self._chat_service = chat_service
        self._bot = bot
        self._settings = settings

    async def handle_update(self, update: Update) -> None:
        """Process a Telegram update. Failures are reported to the user, not raised."""
        message = update.effective_message
        if message is None or message.chat is None:
            logger.info(
                "Ignoring unsupported Telegram update",
                extra={"channel": "telegram", "update_id": update.update_id},
            )
            return

        chat_id = message.chat.id
        thread_id = f"telegram:{chat_id}"
        log_extra = {
            "channel": "telegram",
            "telegram_chat_id": chat_id,
            "telegram_user_id": message.from_user.id if message.from_user else None,
            "thread_id": thread_id,
            "update_id": update.update_id,
        }
        logger.info("Received Telegram update", extra=log_extra)

        try:
            await self._handle_message(message, thread_id, log_extra)
        except (TelegramFileTooLargeError, TelegramUnsupportedFileTypeError) as exc:
            await self._send_text(chat_id, str(exc))
        except TelegramFileDownloadError:
            await self._send_text(chat_id, DOWNLOAD_ERROR_MESSAGE)
        except TelegramFileError:
            await self._send_text(chat_id, UNSUPPORTED_MESSAGE)
        except Exception:
            logger.exception("Telegram adapter failed", extra=log_extra)
            await self._send_text(chat_id, GENERIC_ERROR_MESSAGE)

    async def _handle_message(
        self, message: Message, thread_id: str, log_extra: dict[str, Any]
    ) -> None:
        chat_id = message.chat.id
        text = (message.text or "").strip()

        if text.startswith(("/start", "/help")):
            await self._send_text(chat_id, WELCOME_MESSAGE)
            return

        # Text wins; otherwise look for attached files (caption becomes the prompt).
        files = [] if text else await extract_telegram_files(message, self._bot, self._settings)
        if not text and not files:
            await self._send_text(chat_id, UNSUPPORTED_MESSAGE)
            return

        prompt = text or get_effective_prompt(message)
        verdict = _guard.validate(prompt)
        if not verdict.safe:
            await self._send_text(chat_id, REJECTED_MESSAGE)
            return
        prompt = verdict.sanitized or prompt.strip()

        await self._bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        if files:
            response = await self._chat_service.handle_chat_upload(
                message=prompt, files=files, thread_id=thread_id
            )
        else:
            response = await self._chat_service.handle_chat_message(
                message=prompt, thread_id=thread_id
            )

        await self._send_reply(chat_id, response.response)
        logger.info(
            "Telegram update handled",
            extra={
                **log_extra,
                "intent": response.intent,
                "documents_processed": response.documents_processed,
            },
        )

    async def _send_reply(self, chat_id: int, text: str) -> None:
        formatted = format_telegram_response(text) or EMPTY_RESPONSE_MESSAGE
        for chunk in split_telegram_messages(formatted):
            await self._send_text(chat_id, chunk)

    async def _send_text(self, chat_id: int, text: str) -> None:
        if text.strip():
            await self._bot.send_message(chat_id=chat_id, text=text)
