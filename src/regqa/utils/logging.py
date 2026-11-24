"""Structured logging utilities with secret redaction.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §0.1 & §13).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

# Mask known sensitive environment variables
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|auth)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?"),
    re.compile(r"\b(gsk_[a-zA-Z0-9]{20,})\b"),
    re.compile(r"\b(AIza[a-zA-Z0-9_\-]{35})\b"),
]


class SecretRedactingFormatter(logging.Formatter):
    """Logging formatter that redacts detected API keys and secrets."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern in _SECRET_PATTERNS:
            msg = pattern.sub(r"\1: [REDACTED]", msg)
        return msg


def get_logger(name: str = "regqa", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with secret redaction enabled."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = SecretRedactingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
