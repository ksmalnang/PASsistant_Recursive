"""Shared GLM-OCR client."""

from zai import ZaiClient

from src.config import get_settings

_zai_client: ZaiClient | None = None


def get_zai_client() -> ZaiClient | None:
    """Return the process-wide GLM-OCR client, or None without an API key."""
    global _zai_client
    if _zai_client is None:
        api_key = get_settings().ZHIPU_API_KEY
        if not api_key:
            return None
        _zai_client = ZaiClient(api_key=api_key, timeout=120.0)
    return _zai_client


def close_zai_client() -> None:
    """Close the process-wide GLM-OCR client if one was created."""
    global _zai_client
    if _zai_client is not None:
        _zai_client.close()
        _zai_client = None
