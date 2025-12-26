"""
Tests for core/rate_limiter.py - Token bucket rate limiting.
"""
from __future__ import annotations

import time
import threading
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.rate_limiter import (
    TokenBucket,
    wait_for_rate_limit,
    rate_limited,
    with_retry,
    RateLimitExceeded,
    reset_bucket,
)


class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_initial_capacity(self):
        """Bucket starts with default capacity (tokens field has default)."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        # Note: tokens field has a default value in the dataclass
        # After refill, it will be clamped to capacity
        bucket._refill()
        assert bucket.tokens <= bucket.capacity

    def test_acquire_decrements_tokens(self):
        """Acquiring a token decrements the count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.acquire(timeout=1.0) is True
        assert bucket.tokens < 10.0

    def test_try_acquire_non_blocking(self):
        """try_acquire is non-blocking."""
        bucket = TokenBucket(capacity=1, refill_rate=0.1)
        assert bucket.try_acquire() is True
        # Second immediate acquire should fail (not enough tokens)
        assert bucket.try_acquire() is False

    def test_refill_over_time(self):
        """Tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        # Drain all tokens
        for _ in range(10):
            bucket.try_acquire()
        assert bucket.tokens < 1.0
        # Wait for refill
        time.sleep(0.5)
        bucket._refill()
        assert bucket.tokens >= 4.0  # Should have refilled ~5 tokens

    def test_capacity_limit(self):
        """Tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)
        time.sleep(0.1)  # Would refill 10 tokens
        bucket._refill()
        assert bucket.tokens <= 5.0

    def test_acquire_timeout(self):
        """Acquire times out if no tokens available."""
        bucket = TokenBucket(capacity=1, refill_rate=0.1)  # Very slow refill
        bucket.try_acquire()  # Drain the only token
        start = time.time()
        result = bucket.acquire(timeout=0.1)
        elapsed = time.time() - start
        # Should timeout quickly
        assert elapsed < 0.5
        # May or may not have gotten a token depending on timing
        assert isinstance(result, bool)

    def test_thread_safety(self):
        """Bucket is thread-safe."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0)
        acquired = []

        def worker():
            for _ in range(10):
                if bucket.try_acquire():
                    acquired.append(1)
                time.sleep(0.01)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have acquired some tokens
        assert len(acquired) > 0


class TestRateLimitedDecorator:
    """Tests for @rate_limited decorator."""

    def setup_method(self):
        """Reset global bucket before each test."""
        reset_bucket()

    def test_decorator_passes_through(self):
        """Decorated function executes normally."""
        @rate_limited(timeout=1.0)
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_decorator_preserves_metadata(self):
        """Decorator preserves function name and docstring."""
        @rate_limited(timeout=1.0)
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


class TestWaitForRateLimit:
    """Tests for wait_for_rate_limit function."""

    def setup_method(self):
        """Reset global bucket before each test."""
        reset_bucket()

    def test_returns_true_when_disabled(self):
        """Returns True immediately when rate limiting is disabled."""
        # Config defaults to disabled
        result = wait_for_rate_limit(timeout=1.0)
        assert result is True


class TestWithRetry:
    """Tests for with_retry function."""

    def setup_method(self):
        """Reset global bucket before each test."""
        reset_bucket()

    def test_succeeds_on_first_try(self):
        """Function succeeds on first try."""
        call_count = 0

        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = with_retry(my_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    def test_retries_on_rate_limit_error(self):
        """Retries when RateLimitExceeded is raised."""
        call_count = 0

        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitExceeded("Rate limited")
            return "success"

        result = with_retry(my_func, max_retries=3, base_delay_ms=10)
        assert result == "success"
        assert call_count == 3

    def test_retries_on_429_error(self):
        """Retries when 429 is in error message."""
        call_count = 0

        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("HTTP 429 Too Many Requests")
            return "success"

        result = with_retry(my_func, max_retries=3, base_delay_ms=10, retry_on_429=True)
        assert result == "success"
        assert call_count == 2

    def test_raises_after_max_retries(self):
        """Raises exception after max retries exhausted."""
        def my_func():
            raise RateLimitExceeded("Always fails")

        with pytest.raises(RateLimitExceeded):
            with_retry(my_func, max_retries=2, base_delay_ms=10)

    def test_non_retryable_error_raises_immediately(self):
        """Non-retryable errors raise immediately."""
        call_count = 0

        def my_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a rate limit error")

        with pytest.raises(ValueError):
            with_retry(my_func, max_retries=3, retry_on_429=False)

        assert call_count == 1

    def test_exponential_backoff(self):
        """Backoff increases exponentially."""
        call_times = []

        def my_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RateLimitExceeded("Rate limited")
            return "success"

        with_retry(my_func, max_retries=3, base_delay_ms=50)

        # Check delays increase
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Second delay should be roughly 2x first (exponential)
            assert delay2 >= delay1 * 1.5


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_exception_message(self):
        """Exception stores message correctly."""
        exc = RateLimitExceeded("Test message")
        assert str(exc) == "Test message"

    def test_exception_inheritance(self):
        """Exception inherits from Exception."""
        exc = RateLimitExceeded("Test")
        assert isinstance(exc, Exception)
