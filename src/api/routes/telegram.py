"""Telegram webhook endpoint."""

import logging
from functools import lru_cache
from hmac import compare_digest
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, status
from telegram import Bot, Update

from src.api.models import ErrorResponse
from src.api.services import chat_service
from src.config.settings import get_settings
from src.telegram_bot.adapter import TelegramBotAdapter

logger = logging.getLogger(__name__)


def _require_telegram_enabled() -> None:
    if not get_settings().TELEGRAM_ENABLED:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Telegram is disabled.")


def _verify_webhook_secret(
    x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
) -> None:
    """Check the secret header. Refuses all requests if no secret is set."""
    expected = get_settings().TELEGRAM_WEBHOOK_SECRET_TOKEN
    if not expected:
        logger.error("TELEGRAM_WEBHOOK_SECRET_TOKEN is not set; rejecting request.")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Telegram webhook secret is not configured.",
        )
    if not compare_digest(
        (x_telegram_bot_api_secret_token or "").encode(), expected.encode()
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Invalid Telegram webhook secret."
        )


router = APIRouter()


@lru_cache
def get_telegram_bot() -> Bot:
    """Build a reusable Telegram bot client."""
    token = get_settings().TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN must be set when Telegram support is enabled."
        )
    return Bot(token=token)


@lru_cache
def get_telegram_adapter() -> TelegramBotAdapter:
    """Build a reusable Telegram adapter."""
    return TelegramBotAdapter(
        chat_service=chat_service,
        bot=get_telegram_bot(),
        settings=get_settings(),
    )


async def _process_update(update: Update) -> None:
    """Run the chat turn after Telegram already got its 200."""
    try:
        await get_telegram_adapter().handle_update(update)
    except Exception:
        logger.error(
            "Failed to handle Telegram update %s", update.update_id, exc_info=True
        )


@router.post(
    "/telegram/webhook",
    dependencies=[
        Depends(_require_telegram_enabled),
        Depends(_verify_webhook_secret),
    ],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid Telegram update."},
        403: {"model": ErrorResponse, "description": "Invalid webhook secret."},
        404: {"model": ErrorResponse, "description": "Telegram support is disabled."},
        503: {"model": ErrorResponse, "description": "Webhook secret not configured."},
    },
)
async def telegram_webhook(
    payload: dict[str, Any], background_tasks: BackgroundTasks
) -> dict[str, bool]:
    """Acknowledge a Telegram update immediately and process it in the background."""
    bot = get_telegram_bot()
    try:
        update = Update.de_json(payload, bot)
    except Exception:
        logger.error("Failed to deserialize Telegram update", exc_info=True)
        update = None
    if update is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Malformed Telegram update payload."
        )

    background_tasks.add_task(_process_update, update)
    return {"ok": True}
