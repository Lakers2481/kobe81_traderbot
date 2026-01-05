"""
Secrets Masking Utility for Kobe Trading System.

FIX (2026-01-05): Added to prevent API keys from appearing in logs.

This module provides utilities to:
1. Mask secrets in log messages and exception traces
2. Identify common secret patterns
3. Sanitize output before logging

Usage:
    from core.secrets import mask_secrets, sanitize_exception

    # Mask secrets in a string
    safe_text = mask_secrets("API key is APCA123456789")
    # Output: "API key is ***MASKED***"

    # Sanitize exception before logging
    try:
        risky_operation()
    except Exception as e:
        logger.error(sanitize_exception(e))
"""
from __future__ import annotations

import re
import logging
from typing import List, Pattern, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Secret Patterns
# =============================================================================

# Patterns to detect and mask (pattern, replacement)
SECRET_PATTERNS: List[Tuple[Pattern, str]] = [
    # Alpaca API keys
    (re.compile(r"APCA-API-KEY-ID[=:]\s*\S+", re.IGNORECASE), "APCA-API-KEY-ID=***MASKED***"),
    (re.compile(r"APCA-API-SECRET-KEY[=:]\s*\S+", re.IGNORECASE), "APCA-API-SECRET-KEY=***MASKED***"),
    (re.compile(r"ALPACA_API_KEY_ID[=:]\s*\S+", re.IGNORECASE), "ALPACA_API_KEY_ID=***MASKED***"),
    (re.compile(r"ALPACA_API_SECRET_KEY[=:]\s*\S+", re.IGNORECASE), "ALPACA_API_SECRET_KEY=***MASKED***"),
    # Alpaca key patterns (PK/SK prefix)
    (re.compile(r"\bPK[A-Z0-9]{18,}\b"), "***ALPACA_KEY***"),
    (re.compile(r"\bSK[A-Z0-9]{18,}\b"), "***ALPACA_SECRET***"),

    # Polygon API keys
    (re.compile(r"POLYGON_API_KEY[=:]\s*\S+", re.IGNORECASE), "POLYGON_API_KEY=***MASKED***"),
    (re.compile(r"apiKey=\S+", re.IGNORECASE), "apiKey=***MASKED***"),

    # Generic API key patterns
    (re.compile(r"api[_-]?key[=:]\s*['\"]?\S+['\"]?", re.IGNORECASE), "api_key=***MASKED***"),
    (re.compile(r"secret[_-]?key[=:]\s*['\"]?\S+['\"]?", re.IGNORECASE), "secret_key=***MASKED***"),
    (re.compile(r"access[_-]?token[=:]\s*['\"]?\S+['\"]?", re.IGNORECASE), "access_token=***MASKED***"),
    (re.compile(r"bearer\s+\S+", re.IGNORECASE), "Bearer ***MASKED***"),

    # Webhook secrets
    (re.compile(r"WEBHOOK_HMAC[=:]\s*\S+", re.IGNORECASE), "WEBHOOK_HMAC=***MASKED***"),
    (re.compile(r"WEBHOOK_SECRET[=:]\s*\S+", re.IGNORECASE), "WEBHOOK_SECRET=***MASKED***"),

    # Telegram bot token
    (re.compile(r"TELEGRAM_BOT_TOKEN[=:]\s*\S+", re.IGNORECASE), "TELEGRAM_BOT_TOKEN=***MASKED***"),
    (re.compile(r"\b\d{10}:[A-Za-z0-9_-]{35}\b"), "***TELEGRAM_TOKEN***"),

    # Discord webhook
    (re.compile(r"https://discord\.com/api/webhooks/\S+"), "https://discord.com/api/webhooks/***MASKED***"),

    # Database connection strings
    (re.compile(r"postgres://\S+@\S+"), "postgres://***:***@***"),
    (re.compile(r"mysql://\S+@\S+"), "mysql://***:***@***"),

    # AWS credentials
    (re.compile(r"AWS_ACCESS_KEY_ID[=:]\s*\S+", re.IGNORECASE), "AWS_ACCESS_KEY_ID=***MASKED***"),
    (re.compile(r"AWS_SECRET_ACCESS_KEY[=:]\s*\S+", re.IGNORECASE), "AWS_SECRET_ACCESS_KEY=***MASKED***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***AWS_KEY***"),

    # Generic password patterns
    (re.compile(r"password[=:]\s*['\"]?\S+['\"]?", re.IGNORECASE), "password=***MASKED***"),
    (re.compile(r"passwd[=:]\s*['\"]?\S+['\"]?", re.IGNORECASE), "passwd=***MASKED***"),
]


# =============================================================================
# Masking Functions
# =============================================================================

def mask_secrets(text: str) -> str:
    """
    Mask all known secret patterns in a string.

    Args:
        text: String that may contain secrets

    Returns:
        String with secrets masked
    """
    if not text:
        return text

    result = text
    for pattern, replacement in SECRET_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def sanitize_dict(data: dict) -> dict:
    """
    Recursively sanitize a dictionary, masking secret values.

    Args:
        data: Dictionary that may contain secrets

    Returns:
        Dictionary with secrets masked
    """
    if not data:
        return data

    result = {}
    sensitive_keys = {
        "api_key", "apikey", "api-key",
        "secret", "secret_key", "secretkey",
        "password", "passwd", "pwd",
        "token", "access_token", "bearer",
        "hmac", "webhook_secret",
        "alpaca_api_key_id", "alpaca_api_secret_key",
        "polygon_api_key",
    }

    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")

        if key_lower in sensitive_keys or any(s in key_lower for s in ["key", "secret", "password", "token"]):
            result[key] = "***MASKED***"
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, str):
            result[key] = mask_secrets(value)
        else:
            result[key] = value

    return result


def sanitize_exception(exc: Exception) -> str:
    """
    Sanitize an exception message for safe logging.

    Args:
        exc: Exception that may contain secrets

    Returns:
        Sanitized exception string
    """
    exc_str = str(exc)
    return mask_secrets(exc_str)


def sanitize_traceback(tb_str: str) -> str:
    """
    Sanitize a traceback string for safe logging.

    Args:
        tb_str: Traceback string that may contain secrets

    Returns:
        Sanitized traceback string
    """
    return mask_secrets(tb_str)


# =============================================================================
# Logging Filter
# =============================================================================

class SecretsMaskingFilter(logging.Filter):
    """
    Logging filter that masks secrets in log messages.

    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(SecretsMaskingFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask secrets in log record."""
        # Mask the message
        if record.msg:
            record.msg = mask_secrets(str(record.msg))

        # Mask args if present
        if record.args:
            if isinstance(record.args, dict):
                record.args = sanitize_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    mask_secrets(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

        return True


def install_secrets_filter(logger_name: str = None) -> None:
    """
    Install the secrets masking filter on a logger.

    Args:
        logger_name: Name of logger to filter (None = root logger)
    """
    target_logger = logging.getLogger(logger_name)

    # Add filter to all handlers
    for handler in target_logger.handlers:
        if not any(isinstance(f, SecretsMaskingFilter) for f in handler.filters):
            handler.addFilter(SecretsMaskingFilter())

    # Also add to the logger itself
    if not any(isinstance(f, SecretsMaskingFilter) for f in target_logger.filters):
        target_logger.addFilter(SecretsMaskingFilter())


# =============================================================================
# Environment Variable Safety
# =============================================================================

def get_safe_env_summary() -> dict:
    """
    Get a summary of environment variables with secrets masked.

    Returns:
        Dict with env var names and masked values
    """
    import os

    result = {}
    for key, value in os.environ.items():
        key_lower = key.lower()
        if any(s in key_lower for s in ["key", "secret", "password", "token", "hmac"]):
            result[key] = "***MASKED***"
        else:
            result[key] = value if len(value) < 100 else value[:50] + "..."

    return result
