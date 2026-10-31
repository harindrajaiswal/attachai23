"""Structured JSON/key-value logging with automatic PII and secret redaction."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from app.utils.pii import redact_data, redact_text


class RedactingFormatter(logging.Formatter):
    """Structured log formatter that redacts emails, phones, tokens, and secrets."""

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self.extra_secrets = list(extra_secrets)

    def format(self, record: logging.LogRecord) -> str:
        msg = redact_text(record.getMessage(), extra_secrets=self.extra_secrets)
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        }
        for attr in ("request_id", "tenant_id", "operation", "status_code", "latency_ms"):
            val = getattr(record, attr, None)
            if val is not None:
                payload[attr] = redact_data(val, extra_secrets=self.extra_secrets)

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            payload["exception"] = redact_text(exc_text, extra_secrets=self.extra_secrets)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", extra_secrets: Iterable[str] = ()) -> logging.Logger:
    """Configure the orbit_ai logger with secret and PII redaction."""
    logger = logging.getLogger("orbit_ai")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter(extra_secrets=extra_secrets))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a child logger under orbit_ai."""
    base = logging.getLogger("orbit_ai")
    if not base.handlers:
        configure_logging()
    return base.getChild(name) if name else base
