"""
Tests for secrets masking utility.

FIX (2026-01-05): Added for logging hygiene.
"""
from __future__ import annotations

import pytest
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.secrets import (
    mask_secrets,
    sanitize_dict,
    sanitize_exception,
    sanitize_traceback,
    SecretsMaskingFilter,
    install_secrets_filter,
    get_safe_env_summary,
)


class TestMaskSecrets:
    """Tests for mask_secrets function."""

    def test_masks_alpaca_api_key(self):
        """Masks Alpaca API key."""
        text = "ALPACA_API_KEY_ID=PKABCDEFGHIJKLMNOP"
        result = mask_secrets(text)
        assert "PKABCDEFGHIJKLMNOP" not in result
        assert "MASKED" in result

    def test_masks_alpaca_secret_key(self):
        """Masks Alpaca secret key."""
        text = "ALPACA_API_SECRET_KEY=SKABCDEFGHIJKLMNOPQRS"
        result = mask_secrets(text)
        assert "SKABCDEFGHIJKLMNOPQRS" not in result
        assert "MASKED" in result

    def test_masks_polygon_api_key(self):
        """Masks Polygon API key."""
        text = "POLYGON_API_KEY=abc123xyz789"
        result = mask_secrets(text)
        assert "abc123xyz789" not in result
        assert "MASKED" in result

    def test_masks_api_key_in_url(self):
        """Masks API key in URL."""
        text = "https://api.polygon.io/v2/aggs?apiKey=mysecretkey123"
        result = mask_secrets(text)
        assert "mysecretkey123" not in result
        assert "MASKED" in result

    def test_masks_bearer_token(self):
        """Masks Bearer token."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = mask_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "MASKED" in result

    def test_masks_telegram_token(self):
        """Masks Telegram bot token."""
        text = "TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz12345678901"
        result = mask_secrets(text)
        assert "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz12345678901" not in result
        assert "MASKED" in result

    def test_preserves_non_secret_text(self):
        """Preserves normal text."""
        text = "Processing AAPL with entry at 150.50"
        result = mask_secrets(text)
        assert result == text

    def test_handles_empty_string(self):
        """Handles empty string."""
        assert mask_secrets("") == ""
        assert mask_secrets(None) is None


class TestSanitizeDict:
    """Tests for sanitize_dict function."""

    def test_masks_sensitive_keys(self):
        """Masks values of sensitive keys."""
        data = {
            "api_key": "secret123",
            "symbol": "AAPL",
            "password": "mypassword",
        }
        result = sanitize_dict(data)

        assert result["api_key"] == "***MASKED***"
        assert result["password"] == "***MASKED***"
        assert result["symbol"] == "AAPL"

    def test_handles_nested_dicts(self):
        """Handles nested dictionaries."""
        data = {
            "config": {
                "api_key": "secret123",
                "timeout": 30,
            }
        }
        result = sanitize_dict(data)

        assert result["config"]["api_key"] == "***MASKED***"
        assert result["config"]["timeout"] == 30

    def test_handles_empty_dict(self):
        """Handles empty dict."""
        assert sanitize_dict({}) == {}
        assert sanitize_dict(None) is None


class TestSanitizeException:
    """Tests for sanitize_exception function."""

    def test_masks_secrets_in_exception(self):
        """Masks secrets in exception message."""
        exc = Exception("Failed with POLYGON_API_KEY=secret123")
        result = sanitize_exception(exc)

        assert "secret123" not in result
        assert "MASKED" in result

    def test_preserves_normal_exception(self):
        """Preserves normal exception message."""
        exc = ValueError("Invalid price: -5.0")
        result = sanitize_exception(exc)

        assert "Invalid price: -5.0" in result


class TestSecretsMaskingFilter:
    """Tests for logging filter."""

    def test_filter_masks_log_messages(self):
        """Filter masks secrets in log messages."""
        filter = SecretsMaskingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="API key is POLYGON_API_KEY=secret123",
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "secret123" not in record.msg
        assert "MASKED" in record.msg

    def test_filter_always_returns_true(self):
        """Filter always returns True (doesn't block messages)."""
        filter = SecretsMaskingFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Normal message",
            args=(),
            exc_info=None,
        )

        assert filter.filter(record) is True


class TestGetSafeEnvSummary:
    """Tests for get_safe_env_summary function."""

    def test_masks_secret_env_vars(self):
        """Masks environment variables with sensitive names."""
        import os

        # Set a test env var
        os.environ["TEST_API_KEY_123"] = "supersecret"

        try:
            result = get_safe_env_summary()

            # Check that secret-like keys are masked
            if "TEST_API_KEY_123" in result:
                assert result["TEST_API_KEY_123"] == "***MASKED***"
        finally:
            # Clean up
            del os.environ["TEST_API_KEY_123"]
