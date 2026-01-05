"""
Unified HTTP Client
====================

Centralized HTTP client with consistent User-Agent, timeouts, and retry logic.
Replaces fragmented requests.get() calls throughout the codebase.

FIX (2026-01-05): Created to standardize HTTP request handling.

Features:
- Consistent User-Agent header
- Configurable default timeout (30s)
- Automatic retry on transient errors (429, 503, 504)
- Exponential backoff
- Request logging

Usage:
    from core.http_client import get_http_client

    client = get_http_client()
    response = client.get("https://api.example.com/data")

    # Custom timeout
    response = client.get("https://api.example.com/data", timeout=60)
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Centralized HTTP client with User-Agent, timeouts, and retry logic.

    Thread-safe (uses requests.Session which is thread-safe for most operations).
    """

    DEFAULT_USER_AGENT = "Kobe/1.0 (+https://github.com/kobe81_traderbot)"
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize the HTTP client.

        Args:
            user_agent: User-Agent header value
            timeout: Default timeout in seconds
            max_retries: Maximum retry attempts for transient errors
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries

        # Create session with retry logic
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

        # Configure retry for transient errors
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,  # 0.5s, 1s, 2s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            raise_on_status=False,  # Don't raise on retry exhaustion
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.debug(f"HTTPClient initialized: timeout={timeout}s, retries={max_retries}")

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make a GET request.

        Args:
            url: Request URL
            params: Query parameters
            headers: Additional headers (merged with defaults)
            timeout: Override default timeout
            **kwargs: Additional requests arguments

        Returns:
            requests.Response object
        """
        timeout = timeout if timeout is not None else self.timeout

        # Merge headers
        request_headers = dict(self.session.headers)
        if headers:
            request_headers.update(headers)

        logger.debug(f"GET {url} (timeout={timeout}s)")

        try:
            response = self.session.get(
                url,
                params=params,
                headers=request_headers,
                timeout=timeout,
                **kwargs
            )
            logger.debug(f"GET {url} -> {response.status_code}")
            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout on GET {url}: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error on GET {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Request failed on GET {url}: {e}")
            raise

    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make a POST request.

        Args:
            url: Request URL
            json: JSON body (auto-serialized)
            data: Form data
            headers: Additional headers
            timeout: Override default timeout
            **kwargs: Additional requests arguments

        Returns:
            requests.Response object
        """
        timeout = timeout if timeout is not None else self.timeout

        # Merge headers
        request_headers = dict(self.session.headers)
        if headers:
            request_headers.update(headers)

        logger.debug(f"POST {url} (timeout={timeout}s)")

        try:
            response = self.session.post(
                url,
                json=json,
                data=data,
                headers=request_headers,
                timeout=timeout,
                **kwargs
            )
            logger.debug(f"POST {url} -> {response.status_code}")
            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout on POST {url}: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error on POST {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Request failed on POST {url}: {e}")
            raise

    def close(self) -> None:
        """Close the underlying session."""
        self.session.close()

    def __enter__(self) -> "HTTPClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Global singleton instance
_http_client: Optional[HTTPClient] = None


def get_http_client(
    user_agent: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> HTTPClient:
    """
    Get or create the global HTTP client instance.

    Args:
        user_agent: Override default User-Agent (only affects first call)
        timeout: Override default timeout (only affects first call)
        max_retries: Override default retries (only affects first call)

    Returns:
        HTTPClient instance
    """
    global _http_client
    if _http_client is None:
        _http_client = HTTPClient(
            user_agent=user_agent or HTTPClient.DEFAULT_USER_AGENT,
            timeout=timeout or HTTPClient.DEFAULT_TIMEOUT,
            max_retries=max_retries or HTTPClient.DEFAULT_MAX_RETRIES,
        )
    return _http_client


def reset_http_client() -> None:
    """Reset the global HTTP client (for testing)."""
    global _http_client
    if _http_client is not None:
        _http_client.close()
        _http_client = None
