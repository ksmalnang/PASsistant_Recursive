"""Telegram webhook endpoint."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException, Request, status
from telegram import Bot, Update

from src.api.models import ErrorResponse, TelegramWebhookHealthResponse
from src.api.services import chat_service
from src.config.settings import Settings, get_settings
from src.telegram_bot.adapter import TelegramBotAdapter

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache
def get_telegram_bot() -> Bot:
    """Build a reusable Telegram bot client."""
    settings = get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set when Telegram support is enabled.")
    return Bot(token=settings.TELEGRAM_BOT_TOKEN)


@lru_cache
def get_telegram_adapter() -> TelegramBotAdapter:
    """Build a reusable Telegram adapter."""
    return TelegramBotAdapter(
        chat_service=chat_service,
        bot=get_telegram_bot(),
        settings=get_settings(),
    )


def _verify_telegram_request(
    settings: Settings,
    secret_token: str | None,
) -> None:
    if not settings.TELEGRAM_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Telegram is disabled.")

    expected_secret = settings.TELEGRAM_WEBHOOK_SECRET_TOKEN
    if not expected_secret:
        return

    if secret_token != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Telegram webhook secret.",
        )


@router.post(
    "/telegram/webhook",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "The Telegram update payload is invalid.",
        },
        status.HTTP_403_FORBIDDEN: {
            "model": ErrorResponse,
            "description": "The Telegram webhook secret is invalid or missing.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Telegram support is disabled.",
        },
    },
)
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, bool]:
    """Handle Telegram webhook updates."""
    settings = get_settings()
    _verify_telegram_request(settings, x_telegram_bot_api_secret_token)

    try:
        raw_body = await request.body()
    except Exception as exc:
        logger.error("Failed to read Telegram webhook body: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read request body.",
        ) from exc

    if not raw_body:
        logger.warning(
            "Received empty request body from %s. Headers: %s",
            request.client,
            request.headers,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty request body.",
        )

    try:
        payload = await request.json()
    except Exception as exc:
        logger.error(
            "Failed to parse Telegram update JSON: %s | raw body: %s",
            exc,
            raw_body[:500],
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed Telegram update payload.",
        ) from exc

    try:
        update = Update.de_json(payload, get_telegram_bot())
    except Exception as exc:
        logger.error(
            "Failed to deserialize Telegram update: %s | payload keys: %s",
            exc,
            list(payload.keys()) if isinstance(payload, dict) else type(payload),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to deserialize Telegram update: {exc}",
        ) from exc

    await get_telegram_adapter().handle_update(update)
    return {"ok": True}


@router.get(
    "/telegram/webhook",
    response_model=TelegramWebhookHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Telegram Webhook Status",
    description="Get the current health and status of the Telegram webhook.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Telegram support is disabled.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get webhook info.",
        },
    },
)
async def get_telegram_webhook_health() -> TelegramWebhookHealthResponse:
    """Get the current health and status of the Telegram webhook."""
    settings = get_settings()
    if not settings.TELEGRAM_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Telegram is disabled.",
        )

    try:
        bot = get_telegram_bot()
        info = await bot.get_webhook_info()
        return TelegramWebhookHealthResponse(
            url=info.url,
            has_custom_certificate=info.has_custom_certificate,
            pending_update_count=info.pending_update_count,
            ip_address=info.ip_address,
            last_error_date=info.last_error_date,
            last_error_message=info.last_error_message,
            max_connections=info.max_connections,
            allowed_updates=info.allowed_updates,
        )
    except Exception as exc:
        logger.error("Failed to get Telegram webhook info: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Telegram webhook info: {exc}",
        ) from exc
