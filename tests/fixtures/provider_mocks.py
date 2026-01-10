"""
Data provider mocks for testing.

Provides mock implementations for:
- Polygon API
- Stooq API
- YFinance API (fallback)
- Failure scenarios (500, 429, timeout)
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd


def mock_polygon_response_data(
    symbol: str = "TEST",
    days: int = 250,
    start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate Polygon API response format.

    Args:
        symbol: Stock symbol
        days: Number of days of data
        start_date: Start date (YYYY-MM-DD)

    Returns:
        Dict in Polygon API response format
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    dates = pd.date_range(start=start_date, periods=days, freq="B")

    results = []
    base_price = 100.0

    for i, date in enumerate(dates):
        # Simple price progression
        close_price = base_price * (1 + 0.0001 * i + (i % 5 - 2) * 0.01)

        results.append({
            "v": 5000000 + (i % 10) * 500000,  # volume
            "vw": close_price,  # volume weighted average
            "o": close_price * 0.998,  # open
            "c": close_price,  # close
            "h": close_price * 1.01,  # high
            "l": close_price * 0.99,  # low
            "t": int(date.timestamp() * 1000),  # timestamp ms
            "n": 10000 + i,  # number of trades
        })

    return {
        "ticker": symbol,
        "queryCount": days,
        "resultsCount": days,
        "adjusted": True,
        "results": results,
        "status": "OK",
        "request_id": "test-request-id",
    }


def mock_polygon_api(
    requests_mock,
    symbols: Optional[List[str]] = None,
    days: int = 250,
) -> None:
    """
    Set up Polygon API mock endpoints.

    Args:
        requests_mock: pytest requests_mock fixture
        symbols: List of symbols to mock (defaults to TEST)
        days: Number of days of data
    """
    if symbols is None:
        symbols = ["TEST"]

    base_url = "https://api.polygon.io"

    # Mock aggs endpoint for each symbol
    for symbol in symbols:
        response_data = mock_polygon_response_data(symbol, days)

        # Match any date range in URL
        requests_mock.get(
            f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day",
            json=response_data,
        )

    # Generic handler for any symbol
    def generic_handler(request, context):
        # Extract symbol from URL
        parts = request.path.split("/")
        for i, part in enumerate(parts):
            if part == "ticker" and i + 1 < len(parts):
                symbol = parts[i + 1]
                return mock_polygon_response_data(symbol, days)
        return mock_polygon_response_data("UNKNOWN", days)

    requests_mock.register_uri(
        "GET",
        f"{base_url}/v2/aggs/ticker",
        json=generic_handler,
    )


def mock_stooq_api(
    requests_mock,
    symbols: Optional[List[str]] = None,
    days: int = 250,
) -> None:
    """
    Set up Stooq API mock (CSV download).

    Args:
        requests_mock: pytest requests_mock fixture
        symbols: List of symbols to mock
        days: Number of days of data
    """
    if symbols is None:
        symbols = ["TEST"]

    base_url = "https://stooq.com"

    def csv_handler(request, context):
        # Parse symbol from query params
        symbol = request.qs.get("s", ["TEST"])[0].replace(".us", "").upper()

        # Generate CSV data
        start = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start, periods=days, freq="B")

        lines = ["Date,Open,High,Low,Close,Volume"]
        base_price = 100.0

        for i, date in enumerate(dates):
            close_price = base_price * (1 + 0.0001 * i)
            line = f"{date.strftime('%Y-%m-%d')},{close_price*0.998:.2f},{close_price*1.01:.2f},{close_price*0.99:.2f},{close_price:.2f},{5000000+i*1000}"
            lines.append(line)

        context.headers["Content-Type"] = "text/csv"
        return "\n".join(lines)

    requests_mock.get(
        f"{base_url}/q/d/l/",
        text=csv_handler,
    )


def failing_polygon_api(
    requests_mock,
    status_code: int = 500,
    error_message: str = "Internal Server Error",
) -> None:
    """
    Set up Polygon API to return errors.

    Args:
        requests_mock: pytest requests_mock fixture
        status_code: HTTP status code to return
        error_message: Error message
    """
    # Use regex to match any symbol in the URL pattern
    pattern = re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/[A-Z]+/range/.*")

    requests_mock.register_uri(
        "GET",
        pattern,
        status_code=status_code,
        json={
            "status": "ERROR",
            "error": error_message,
            "request_id": "error-request-id",
        },
    )


def rate_limited_polygon_api(requests_mock) -> None:
    """
    Set up Polygon API to return 429 rate limit errors.

    Args:
        requests_mock: pytest requests_mock fixture
    """
    # Use regex to match any symbol in the URL pattern
    pattern = re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/[A-Z]+/range/.*")

    requests_mock.register_uri(
        "GET",
        pattern,
        status_code=429,
        json={
            "status": "ERROR",
            "error": "Rate limit exceeded. Please slow down your requests.",
            "request_id": "rate-limit-request-id",
        },
        headers={
            "Retry-After": "60",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(datetime.now().timestamp()) + 60),
        },
    )


def mock_empty_polygon_response(requests_mock, symbol: str = "TEST") -> None:
    """
    Set up Polygon API to return empty results.

    Args:
        requests_mock: pytest requests_mock fixture
        symbol: Symbol to mock
    """
    base_url = "https://api.polygon.io"

    requests_mock.get(
        f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day",
        json={
            "ticker": symbol,
            "queryCount": 0,
            "resultsCount": 0,
            "adjusted": True,
            "results": [],
            "status": "OK",
            "request_id": "empty-request-id",
        },
    )


def mock_malformed_polygon_response(requests_mock, symbol: str = "TEST") -> None:
    """
    Set up Polygon API to return malformed data.

    Args:
        requests_mock: pytest requests_mock fixture
        symbol: Symbol to mock
    """
    base_url = "https://api.polygon.io"

    # Missing required fields
    requests_mock.get(
        f"{base_url}/v2/aggs/ticker/{symbol}/range/1/day",
        json={
            "ticker": symbol,
            "status": "OK",
            "results": [
                {"v": 1000000},  # Missing OHLC fields
                {"o": 100, "c": 101},  # Missing high, low, timestamp
            ],
        },
    )
