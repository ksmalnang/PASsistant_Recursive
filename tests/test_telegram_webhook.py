"""Telegram webhook route tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


def test_disabled_telegram_route_returns_404(monkeypatch):
    monkeypatch.setattr("src.api.routes.telegram.get_settings", lambda: _settings(False))
    client = TestClient(app)

    response = client.post("/telegram/webhook", json={})

    assert response.status_code == 404
    assert response.json() == {"detail": "Telegram is disabled."}


def test_valid_webhook_accepts_update_and_returns_200(monkeypatch):
    calls = []

    class FakeAdapter:
        async def handle_update(self, update):
            calls.append(update.update_id)

    monkeypatch.setattr(
        "src.api.routes.telegram.get_settings",
        lambda: _settings(True, APP_ENV="production"),
    )
    monkeypatch.setattr("src.api.routes.telegram.get_telegram_bot", lambda: object())
    monkeypatch.setattr("src.api.routes.telegram.get_telegram_adapter", lambda: FakeAdapter())
    monkeypatch.setattr(
        "src.api.routes.telegram.Update.de_json",
        classmethod(lambda cls, payload, bot: _update(payload)),
    )
    client = TestClient(app)

    response = client.post(
        "/telegram/webhook",
        json={"update_id": 77},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == [77]


def test_malformed_json_returns_400(monkeypatch):
    monkeypatch.setattr("src.api.routes.telegram.get_settings", lambda: _settings(True))
    client = TestClient(app)

    response = client.post(
        "/telegram/webhook",
        content="{invalid",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Malformed Telegram update payload."}


def test_route_delegates_to_adapter_once(monkeypatch):
    calls = []

    class FakeAdapter:
        async def handle_update(self, update):
            calls.append(update.update_id)

    monkeypatch.setattr("src.api.routes.telegram.get_settings", lambda: _settings(True))
    monkeypatch.setattr("src.api.routes.telegram.get_telegram_bot", lambda: object())
    monkeypatch.setattr("src.api.routes.telegram.get_telegram_adapter", lambda: FakeAdapter())
    monkeypatch.setattr(
        "src.api.routes.telegram.Update.de_json",
        classmethod(lambda cls, payload, bot: _update(payload)),
    )
    client = TestClient(app)

    response = client.post("/telegram/webhook", json={"update_id": 88})

    assert response.status_code == 200
    assert calls == [88]


def _settings(enabled: bool, **overrides):
    from src.config.settings import Settings

    base = {
        "TELEGRAM_ENABLED": enabled,
        "TELEGRAM_BOT_TOKEN": "token",
        "TELEGRAM_WEBHOOK_SECRET_TOKEN": None,
        "APP_ENV": "development",
    }
    base.update(overrides)
    return Settings(**base)

def test_webhook_with_invalid_secret_token_returns_403(monkeypatch):
    monkeypatch.setattr(
        "src.api.routes.telegram.get_settings",
        lambda: _settings(True, TELEGRAM_WEBHOOK_SECRET_TOKEN="valid_secret_123")
    )
    client = TestClient(app)

    # Missing token
    response1 = client.post("/telegram/webhook", json={"update_id": 1})
    assert response1.status_code == 403

    # Wrong token
    response2 = client.post(
        "/telegram/webhook",
        json={"update_id": 1},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"}
    )
    assert response2.status_code == 403


def _update(payload: dict):
    from types import SimpleNamespace

    return SimpleNamespace(update_id=payload.get("update_id", 0))
