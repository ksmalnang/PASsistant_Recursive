from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.models import ChatResponse
from src.telegram_bot.adapter import TelegramBotAdapter


def make_adapter(chat_service: SimpleNamespace) -> TelegramBotAdapter:
    bot = SimpleNamespace(
        send_chat_action=AsyncMock(),
        send_message=AsyncMock(),
    )
    return TelegramBotAdapter(chat_service, bot, SimpleNamespace())


@pytest.mark.asyncio
async def test_text_turn_passes_thread_id_to_chat_service() -> None:
    chat_service = SimpleNamespace(
        handle_chat_message=AsyncMock(
            return_value=ChatResponse(response="Answer", thread_id="telegram:123")
        ),
        handle_chat_upload=AsyncMock(),
    )
    adapter = make_adapter(chat_service)
    message = SimpleNamespace(
        text="What are the graduation requirements?",
        chat=SimpleNamespace(id=123),
        from_user=None,
    )

    await adapter.handle_update(SimpleNamespace(update_id=1, effective_message=message))

    chat_service.handle_chat_message.assert_awaited_once_with(
        message="What are the graduation requirements?",
        thread_id="telegram:123",
    )


@pytest.mark.asyncio
async def test_file_turn_passes_thread_id_to_chat_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_service = SimpleNamespace(
        handle_chat_message=AsyncMock(),
        handle_chat_upload=AsyncMock(
            return_value=ChatResponse(response="Processed", thread_id="telegram:456")
        ),
    )
    adapter = make_adapter(chat_service)
    message = SimpleNamespace(
        text=None,
        caption="Process this document.",
        chat=SimpleNamespace(id=456),
        from_user=None,
    )

    async def extract_files(*args: object) -> list[tuple[str, bytes]]:
        return [("record.pdf", b"pdf bytes")]

    monkeypatch.setattr("src.telegram_bot.adapter.extract_telegram_files", extract_files)

    await adapter.handle_update(SimpleNamespace(update_id=2, effective_message=message))

    chat_service.handle_chat_upload.assert_awaited_once_with(
        message="Process this document.",
        files=[("record.pdf", b"pdf bytes")],
        thread_id="telegram:456",
    )
