"""
Structured Logging Configuration
==================================

Configures application-wide logging with:
  - JSON-structured output in production (environment=production)
  - Human-readable coloured output in development
  - Request ID propagation via contextvars
  - Consistent field names for log aggregation (Datadog, CloudWatch, etc.)

Usage:
    from app.core.logging import configure_logging, get_logger

    configure_logging()  # Call once at startup

    logger = get_logger(__name__)
    logger.info("Processing request | request_id=%s", request_id)

Context Variables (for async propagation):
    from app.core.logging import request_id_var
    token = request_id_var.set("uuid-here")
    # ... do work ...
    request_id_var.reset(token)
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# Context variable for propagating request_id through async call chains
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = False,
) -> None:
    """
    Configure application-wide logging.

    Parameters
    ----------
    level :
        Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_logs :
        If True, emit JSON lines for log aggregation systems.
        If False (default), emit human-readable coloured output.

    Call once at application startup:
        configure_logging(level="INFO", json_logs=settings.environment == "production")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Remove existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if json_logs:
        formatter = _JsonFormatter()
    else:
        formatter = _DevFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Convenience wrapper around ``logging.getLogger`` to keep imports tidy.

    Parameters
    ----------
    name :
        Logger name — use ``__name__`` for module-level loggers.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _DevFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    LEVEL_COLOURS = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
        "CRITICAL": "\033[35m", # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelname, "")
        reset = self.RESET
        request_id = request_id_var.get("")
        rid_str = f" [{request_id[:8]}]" if request_id else ""

        formatted = super().format(record)
        return (
            f"{colour}{record.levelname:8}{reset} "
            f"\033[90m{record.name}{reset}{rid_str} — {record.getMessage()}"
        )


class _JsonFormatter(logging.Formatter):
    """JSON-line formatter for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        payload = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(""),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)
