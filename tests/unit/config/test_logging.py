"""Unit tests for centralized logging formatters, context filter, and dictConfig."""

import json
import logging
from types import SimpleNamespace
from typing import Any

from src.config.logging import (
    ConsoleFormatter,
    ContextFilter,
    JSONFormatter,
    RFC5424Formatter,
    build_logging_config,
    clear_log_context,
    configure_logging,
    get_log_context,
    reset_log_context,
    set_log_context,
)


def _make_record(
    msg: str = "Test message",
    level: int = logging.INFO,
    name: str = "test.logger",
    exc_info: Any = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    return record


def test_unit_context_vars_and_filter() -> None:
    """ContextFilter injects active context variables into records."""
    clear_log_context()
    assert get_log_context() == {}

    token = set_log_context(session_id="sess-001", user_id="user-abc")
    assert get_log_context() == {"session_id": "sess-001", "user_id": "user-abc"}

    record = _make_record("Hello world")
    flt = ContextFilter()
    assert flt.filter(record) is True
    assert record.session_id == "sess-001"  # type: ignore[attr-defined]
    assert record.user_id == "user-abc"  # type: ignore[attr-defined]
    assert record._context_data == {"session_id": "sess-001", "user_id": "user-abc"}  # type: ignore[attr-defined]

    reset_log_context(token)
    assert get_log_context() == {}

    record2 = _make_record("After reset")
    flt.filter(record2)
    assert record2._context_data == {}  # type: ignore[attr-defined]


def test_unit_console_formatter_plain() -> None:
    """ConsoleFormatter renders clean plaintext without ANSI escape codes."""
    formatter = ConsoleFormatter(use_colors=False)
    record = _make_record("User query processed")
    record._context_data = {"session_id": "sess-xyz"}  # type: ignore[attr-defined]

    formatted = formatter.format(record)
    assert "[INFO ]" in formatted
    assert "[test.logger]: User query processed" in formatted
    assert "[session_id=sess-xyz]" in formatted
    assert "\033[" not in formatted


def test_unit_console_formatter_with_colors() -> None:
    """ConsoleFormatter includes ANSI escape sequences when use_colors is True."""
    formatter = ConsoleFormatter(use_colors=True)
    record = _make_record("Warning occurred", level=logging.WARNING)

    formatted = formatter.format(record)
    assert "\033[33m" in formatted  # Yellow for WARNING
    assert "WARNING" in formatted
    assert "Warning occurred" in formatted


def test_unit_console_formatter_exception() -> None:
    """ConsoleFormatter formats tracebacks cleanly."""
    formatter = ConsoleFormatter(use_colors=False)
    try:
        raise ValueError("Something broke")
    except ValueError:
        import sys

        record = _make_record("Operation failed", level=logging.ERROR, exc_info=sys.exc_info())

    formatted = formatter.format(record)
    assert "Operation failed" in formatted
    assert "ValueError: Something broke" in formatted


def test_unit_json_formatter() -> None:
    """JSONFormatter renders valid single-line JSON records with metadata."""
    formatter = JSONFormatter(environment="test-env")
    record = _make_record("Structured message")
    record._context_data = {"request_id": "req-999"}  # type: ignore[attr-defined]

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert data["message"] == "Structured message"
    assert data["environment"] == "test-env"
    assert data["context"] == {"request_id": "req-999"}
    assert "timestamp" in data
    assert "line" in data
    assert "module" in data


def test_unit_json_formatter_exception() -> None:
    """JSONFormatter captures exception details in the exception key."""
    formatter = JSONFormatter()
    try:
        raise RuntimeError("Failure in task")
    except RuntimeError:
        import sys

        record = _make_record("Failed", level=logging.ERROR, exc_info=sys.exc_info())

    data = json.loads(formatter.format(record))
    assert "RuntimeError: Failure in task" in data["exception"]


def test_unit_rfc5424_formatter() -> None:
    """RFC5424Formatter formats standard syslog format records."""
    formatter = RFC5424Formatter(app_name="TestApp", facility=16, environment="development")
    record = _make_record("Syslog message")
    record._context_data = {"tag": "val"}  # type: ignore[attr-defined]

    formatted = formatter.format(record)
    assert formatted.startswith("<134>1 ")
    assert "TestApp" in formatted
    assert "[meta " in formatted
    assert 'tag="val"' in formatted
    assert "Syslog message" in formatted


def test_unit_build_logging_config_modes() -> None:
    """build_logging_config selects the requested formatter and quietens third parties."""
    settings_console = SimpleNamespace(
        LOG_FORMAT="console",
        LOG_LEVEL="INFO",
        LOG_FILE=None,
        DEBUG=False,
        APP_ENV="test",
        LOG_APP_NAME="app",
        LOG_SYSLOG_FACILITY=16,
    )
    config = build_logging_config(settings_console)  # type: ignore[arg-type]
    assert config["handlers"]["default"]["formatter"] == "console"
    assert config["loggers"]["httpx"]["level"] == "WARNING"
    assert config["loggers"]["qdrant_client"]["level"] == "WARNING"

    settings_json = SimpleNamespace(
        LOG_FORMAT="json",
        LOG_LEVEL="DEBUG",
        LOG_FILE=None,
        DEBUG=True,
        APP_ENV="test",
        LOG_APP_NAME="app",
        LOG_SYSLOG_FACILITY=16,
    )
    config_json = build_logging_config(settings_json)  # type: ignore[arg-type]
    assert config_json["handlers"]["default"]["formatter"] == "json"
    assert config_json["loggers"]["httpx"]["level"] == "DEBUG"

    settings_file = SimpleNamespace(
        LOG_FORMAT="rfc5424",
        LOG_LEVEL="INFO",
        LOG_FILE="data/test_logs/app.log",
        DEBUG=False,
        APP_ENV="test",
        LOG_APP_NAME="app",
        LOG_SYSLOG_FACILITY=16,
    )
    config_file = build_logging_config(settings_file)  # type: ignore[arg-type]
    assert config_file["handlers"]["default"]["formatter"] == "rfc5424"
    assert "file" in config_file["handlers"]
    assert config_file["handlers"]["file"]["class"] == "logging.handlers.RotatingFileHandler"


def test_unit_configure_logging_integration() -> None:
    """configure_logging executes dictConfig cleanly."""
    settings = SimpleNamespace(
        LOG_FORMAT="console",
        LOG_LEVEL="INFO",
        LOG_FILE=None,
        DEBUG=False,
        APP_ENV="test",
        LOG_APP_NAME="app",
        LOG_SYSLOG_FACILITY=16,
    )
    configure_logging(settings)  # type: ignore[arg-type]
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) > 0
