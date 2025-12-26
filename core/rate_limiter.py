"""
Token Bucket Rate Limiter for Kobe81 Trading Bot.
Config-gated: only active when execution.rate_limiter.enabled = true.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional
from functools import wraps

from config.settings_loader import get_rate_limiter_config


T = TypeVar("T")


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter implementation.
    Tokens are refilled at a constant rate up to capacity.
    """
    capacity: int = 120  # Max tokens (orders per minute)
    refill_rate: float = 2.0  # Tokens per second (120/60 = 2)
    tokens: float = field(default=120.0, init=False)
    last_refill: float = field(default_factory=time.time, init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire a token. Blocks until token is available or timeout.

        Args:
            timeout: Max seconds to wait for a token

        Returns:
            True if token acquired, False if timeout
        """
        deadline = time.time() + timeout
        with self.lock:
            while True:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.refill_rate
                if time.time() + wait_time > deadline:
                    return False
                time.sleep(min(wait_time, 0.1))

    def try_acquire(self) -> bool:
        """Non-blocking token acquisition."""
        with self.lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


# Global singleton bucket
_bucket: Optional[TokenBucket] = None
_bucket_lock = threading.Lock()


def _get_bucket() -> TokenBucket:
    """Get or create the global rate limiter bucket."""
    global _bucket
    if _bucket is None:
        with _bucket_lock:
            if _bucket is None:
                cfg = get_rate_limiter_config()
                capacity = cfg.get("max_orders_per_minute", 120)
                _bucket = TokenBucket(
                    capacity=capacity,
                    refill_rate=capacity / 60.0,
                )
    return _bucket


def wait_for_rate_limit(timeout: float = 10.0) -> bool:
    """
    Wait for rate limit token (config-gated).
    Returns immediately if rate limiting is disabled.

    Args:
        timeout: Max seconds to wait

    Returns:
        True if ready to proceed, False if timed out
    """
    cfg = get_rate_limiter_config()
    if not cfg.get("enabled", False):
        return True
    return _get_bucket().acquire(timeout)


def rate_limited(timeout: float = 10.0):
    """
    Decorator to rate-limit a function.
    Raises RateLimitExceeded if timeout expires.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not wait_for_rate_limit(timeout):
                raise RateLimitExceeded(f"Rate limit timeout after {timeout}s")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit cannot be satisfied within timeout."""
    pass


def with_retry(
    func: Callable[..., T],
    *args,
    max_retries: Optional[int] = None,
    base_delay_ms: Optional[int] = None,
    retry_on_429: Optional[bool] = None,
    **kwargs,
) -> T:
    """
    Execute a function with exponential backoff retry on rate limit errors.
    Config-gated: uses settings from execution.rate_limiter if not overridden.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        max_retries: Max retry attempts (default from config)
        base_delay_ms: Initial backoff delay in ms (default from config)
        retry_on_429: Whether to retry on 429 errors (default from config)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries exhausted
    """
    cfg = get_rate_limiter_config()

    if max_retries is None:
        max_retries = cfg.get("max_retries", 3)
    if base_delay_ms is None:
        base_delay_ms = cfg.get("base_delay_ms", 500)
    if retry_on_429 is None:
        retry_on_429 = cfg.get("retry_on_429", True)

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            # Wait for rate limit token before each attempt
            wait_for_rate_limit()
            return func(*args, **kwargs)
        except RateLimitExceeded as e:
            last_exc = e
            if not retry_on_429:
                raise
            if attempt < max_retries:
                delay = (base_delay_ms / 1000.0) * (2 ** attempt)
                time.sleep(delay)
        except Exception as e:
            # Check for HTTP 429 in exception message
            if retry_on_429 and "429" in str(e):
                last_exc = e
                if attempt < max_retries:
                    delay = (base_delay_ms / 1000.0) * (2 ** attempt)
                    time.sleep(delay)
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Retry loop exited unexpectedly")


def reset_bucket() -> None:
    """Reset the global bucket (for testing)."""
    global _bucket
    with _bucket_lock:
        _bucket = None
