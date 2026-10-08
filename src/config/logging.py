"""Centralized logging configuration with console, JSON, and RFC 5424 formats."""

from __future__ import annotations

import json
import logging
import logging.config
import os
import socket
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

from src.config.settings import Settings, get_settings

_SEQUENCE = count(1)
_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


def set_log_context(**kwargs: Any) -> Token[dict[str, Any] | None]:
    """Set key-value metadata for the current async/execution context."""
    current = dict(_LOG_CONTEXT.get() or {})
    current.update({k: v for k, v in kwargs.items() if v is not None})
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: Token[dict[str, Any] | None]) -> None:
    """Reset the log context to a previously saved token."""
    _LOG_CONTEXT.reset(token)


def clear_log_context() -> None:
    """Clear all contextual metadata for the current context."""
    _LOG_CONTEXT.set(None)


def get_log_context() -> dict[str, Any]:
    """Return a copy of the current contextual metadata dictionary."""
    return dict(_LOG_CONTEXT.get() or {})


class ContextFilter(logging.Filter):
    """Inject contextvars metadata into every processed log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get() or {}
        if context:
            record._context_data = context  # type: ignore[attr-defined]
            for key, value in context.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        else:
            record._context_data = {}  # type: ignore[attr-defined]
        return True


class ConsoleFormatter(logging.Formatter):
    """Render human-friendly, colorized log messages for terminal readability."""

    _COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;35m",  # Bold Magenta
    }
    _DIM = "\033[90m"
    _RESET = "\033[0m"

    def __init__(self, use_colors: bool | None = None):
        super().__init__()
        if use_colors is None:
            # Check NO_COLOR spec and terminal capability
            no_color = bool(os.getenv("NO_COLOR"))
            is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            self.use_colors = is_tty and not no_color
        else:
            self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        timestamp_str = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        levelname = record.levelname.ljust(5)

        # Contextual tags (e.g. session_id, request_id)
        ctx: dict[str, Any] = getattr(record, "_context_data", {})
        ctx_str = ""
        if ctx:
            pairs = " ".join(f"{k}={v}" for k, v in ctx.items())
            ctx_str = f" [{pairs}]"

        message = record.getMessage()

        if self.use_colors:
            color = self._COLORS.get(record.levelno, "")
            dim = self._DIM
            reset = self._RESET
            formatted = (
                f"{dim}{timestamp_str}{reset} "
                f"{color}[{levelname}]{reset} "
                f"{dim}[{record.name}]{reset}: "
                f"{message}"
                f"{dim}{ctx_str}{reset}"
            )
        else:
            formatted = f"{timestamp_str} [{levelname}] [{record.name}]: {message}{ctx_str}"

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            if exception_text:
                formatted = f"{formatted}\n{exception_text}"

        return formatted


class JSONFormatter(logging.Formatter):
    """Render log records as single-line JSON objects for structured APM / aggregators."""

    def __init__(self, environment: str = "development"):
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        timestamp = (
            datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "environment": self.environment,
        }

        # Include contextual metadata
        ctx: dict[str, Any] = getattr(record, "_context_data", {})
        if ctx:
            payload["context"] = ctx

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            if exception_text:
                payload["exception"] = exception_text

        return json.dumps(payload, default=str)


class RFC5424Formatter(logging.Formatter):
    """Render log records as RFC 5424 syslog messages."""

    _SEVERITY_MAP = {
        logging.CRITICAL: 2,
        logging.ERROR: 3,
        logging.WARNING: 4,
        logging.INFO: 6,
        logging.DEBUG: 7,
    }

    def __init__(self, app_name: str, facility: int, environment: str):
        super().__init__()
        self.app_name = self._clean_header_value(app_name, max_length=48)
        self.facility = facility
        self.environment = environment
        self.hostname = self._clean_header_value(socket.gethostname(), max_length=255)
        self.procid = str(os.getpid())

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record according to RFC 5424."""
        severity = self._get_severity(record.levelno)
        priority = (self.facility * 8) + severity
        timestamp = self._format_timestamp(record.created)
        msgid = self._resolve_msgid(record)
        structured_data = self._build_structured_data(record)
        message = self._sanitize_message(record.getMessage())

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            if exception_text:
                message = f"{message}\n{exception_text}" if message else exception_text

        return (
            f"<{priority}>1 {timestamp} {self.hostname} {self.app_name} "
            f"{self.procid} {msgid} {structured_data} {message}"
        )

    def _get_severity(self, levelno: int) -> int:
        """Map Python levels to syslog severities."""
        if levelno >= logging.CRITICAL:
            return self._SEVERITY_MAP[logging.CRITICAL]
        if levelno >= logging.ERROR:
            return self._SEVERITY_MAP[logging.ERROR]
        if levelno >= logging.WARNING:
            return self._SEVERITY_MAP[logging.WARNING]
        if levelno >= logging.INFO:
            return self._SEVERITY_MAP[logging.INFO]
        return self._SEVERITY_MAP[logging.DEBUG]

    def _format_timestamp(self, created: float) -> str:
        """Render a RFC 3339 timestamp with UTC offset."""
        return (
            datetime.fromtimestamp(created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _resolve_msgid(self, record: logging.LogRecord) -> str:
        """Resolve a safe RFC 5424 MSGID value."""
        raw_msgid = getattr(record, "msgid", record.levelname)
        return self._clean_header_value(str(raw_msgid), max_length=32)

    def _build_structured_data(self, record: logging.LogRecord) -> str:
        """Build the structured-data element for the record."""
        sequence_id = getattr(record, "sequence_id", next(_SEQUENCE))
        params: dict[str, Any] = {
            "sequenceId": str(sequence_id),
            "logger": record.name,
            "module": record.module,
            "line": str(record.lineno),
            "environment": self.environment,
        }
        ctx: dict[str, Any] = getattr(record, "_context_data", {})
        if ctx:
            for k, v in ctx.items():
                params[k] = str(v)

        rendered = " ".join(
            f'{key}="{self._escape_sd_param(value)}"'
            for key, value in params.items()
            if value
        )
        return f"[meta {rendered}]"

    def _sanitize_message(self, message: str) -> str:
        """Ensure the message is printable for the syslog payload."""
        if not message:
            return "-"
        return "".join(
            character if character.isprintable() or character in "\n\t" else " "
            for character in message
        )

    def _escape_sd_param(self, value: str) -> str:
        """Escape structured-data parameter values."""
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("]", "\\]")
        )

    def _clean_header_value(self, value: str, max_length: int) -> str:
        """Normalize header fields to RFC 5424-safe ASCII tokens."""
        cleaned = "".join(
            character if 33 <= ord(character) <= 126 and character != " " else "_"
            for character in value
        ).strip("_")
        if not cleaned:
            return "-"
        return cleaned[:max_length]


def build_logging_config(settings: Settings | None = None) -> dict[str, Any]:
    """Build a dictConfig payload for application logging."""
    resolved = settings or get_settings()
    log_format = getattr(resolved, "LOG_FORMAT", "console")
    is_debug = bool(getattr(resolved, "DEBUG", False)) or (
        resolved.LOG_LEVEL.upper() == "DEBUG"
    )

    formatters: dict[str, Any] = {
        "console": {
            "()": "src.config.logging.ConsoleFormatter",
        },
        "json": {
            "()": "src.config.logging.JSONFormatter",
            "environment": resolved.APP_ENV,
        },
        "rfc5424": {
            "()": "src.config.logging.RFC5424Formatter",
            "app_name": resolved.LOG_APP_NAME,
            "facility": resolved.LOG_SYSLOG_FACILITY,
            "environment": resolved.APP_ENV,
        },
    }

    selected_formatter = log_format if log_format in formatters else "console"

    handlers: dict[str, Any] = {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": selected_formatter,
            "filters": ["context_filter"],
            "stream": "ext://sys.__stdout__",
        }
    }

    active_handlers = ["default"]

    # Optional rotating file handler
    log_file = getattr(resolved, "LOG_FILE", None)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(file_path),
            "formatter": selected_formatter,
            "filters": ["context_filter"],
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        active_handlers.append("file")

    # Keep verbose third-party loggers at WARNING unless LOG_LEVEL is explicitly DEBUG
    third_party_level = (
        "DEBUG" if resolved.LOG_LEVEL.upper() == "DEBUG" else "WARNING"
    )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context_filter": {
                "()": "src.config.logging.ContextFilter",
            }
        },
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": resolved.LOG_LEVEL.upper(),
            "handlers": active_handlers,
        },
        "loggers": {
            "uvicorn": {
                "level": resolved.LOG_LEVEL.upper(),
                "handlers": active_handlers,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": resolved.LOG_LEVEL.upper(),
                "handlers": active_handlers,
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO" if is_debug else "WARNING",
                "handlers": active_handlers,
                "propagate": False,
            },
            # Quiet noisy HTTP and DB clients
            "httpx": {"level": third_party_level, "propagate": True},
            "httpcore": {"level": third_party_level, "propagate": True},
            "qdrant_client": {"level": third_party_level, "propagate": True},
            "urllib3": {"level": third_party_level, "propagate": True},
            "watchfiles": {"level": third_party_level, "propagate": True},
        },
    }


def configure_logging(settings: Settings | None = None) -> None:
    """Apply centralized logging configuration."""
    logging.config.dictConfig(build_logging_config(settings))
