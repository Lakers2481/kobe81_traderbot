"""
Tests for rate limiting in monitor/health_endpoints.py
"""
from __future__ import annotations

import pytest
import time
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from monitor.health_endpoints import (
    TokenBucketRateLimiter,
    get_metrics_rate_limiter,
    is_rate_limiting_enabled,
)


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_allows_requests_under_limit(self):
        """Allows requests when under the limit."""
        limiter = TokenBucketRateLimiter(max_requests=5, window_seconds=60)

        for _ in range(5):
            assert limiter.is_allowed() is True

    def test_blocks_requests_over_limit(self):
        """Blocks requests when over the limit."""
        limiter = TokenBucketRateLimiter(max_requests=3, window_seconds=60)

        # Use up the limit
        for _ in range(3):
            assert limiter.is_allowed() is True

        # Next request should be blocked
        assert limiter.is_allowed() is False

    def test_allows_requests_after_window_expires(self):
        """Allows requests after the time window expires."""
        limiter = TokenBucketRateLimiter(max_requests=2, window_seconds=1)

        # Use up the limit
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.is_allowed() is True

    def test_get_retry_after(self):
        """Returns appropriate retry-after time."""
        limiter = TokenBucketRateLimiter(max_requests=1, window_seconds=60)

        limiter.is_allowed()  # Use the single request
        limiter.is_allowed()  # This is blocked

        retry_after = limiter.get_retry_after()
        assert 0 < retry_after <= 60


class TestRateLimitingEnabled:
    """Tests for rate limiting enable/disable logic."""

    def test_disabled_in_paper_mode(self):
        """Rate limiting is disabled in paper mode."""
        with patch('config.settings_loader.get_setting', return_value='paper'):
            assert is_rate_limiting_enabled() is False

    def test_enabled_in_live_mode(self):
        """Rate limiting is enabled in live mode."""
        with patch('config.settings_loader.get_setting', return_value='live'):
            assert is_rate_limiting_enabled() is True


class TestGetMetricsRateLimiter:
    """Tests for global rate limiter singleton."""

    def test_returns_same_instance(self):
        """Returns the same instance on repeated calls."""
        limiter1 = get_metrics_rate_limiter()
        limiter2 = get_metrics_rate_limiter()
        assert limiter1 is limiter2

    def test_default_limits(self):
        """Default rate limiter has expected limits."""
        limiter = get_metrics_rate_limiter()
        assert limiter.max_requests == 60
        assert limiter.window_seconds == 60
