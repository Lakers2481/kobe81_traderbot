import pytest
import pandas as pd
from datetime import datetime, timedelta
import re

from data.providers.polygon_eod import fetch_daily_bars_polygon
from execution.broker_alpaca import get_best_ask, get_order_by_id, get_order_by_client_id

# Define a fixture for a temporary cache directory
@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "cache"

# Define a fixture for mocking Alpaca API requests
@pytest.fixture
def alpaca_requests_mock(requests_mock):
    paper_api_base = "https://paper-api.alpaca.markets"
    data_api_base = "https://data.alpaca.markets"

    # Callback for dynamic Alpaca Quote responses
    def alpaca_quote_callback(request, context):
        symbol_match = re.search(r"symbols=([A-Z]+)", request.url)
        symbol = symbol_match.group(1) if symbol_match else "AAPL"
        context.status_code = 200
        if symbol == "UNKNOWN": # This is what test_alpaca_get_best_ask_failure checks
            return {"quotes": {symbol: []}}
        return {"quotes": {symbol: [{"ap": 170.0, "bp": 169.9, "as": 100, "bs": 200}]}}

    # Callback for dynamic Alpaca Order by Client ID responses
    def alpaca_client_order_callback(request, context):
        # Extract client_order_id from query string
        client_order_ids = request.qs.get('client_order_id')
        client_order_id = client_order_ids[0] if client_order_ids else "test-client-id"
        
        context.status_code = 200
        if "non-existent-client-id" in client_order_id:
            context.status_code = 404 # Set status code to 404
            return {}
        context.status_code = 200
        return {"id": "alpaca-order-id-123", "status": "filled", "filled_avg_price": 170.1, "filled_qty": 10, "client_order_id": client_order_id}

    # Callback for dynamic Alpaca Positions responses
    def alpaca_positions_callback(request, context):
        context.status_code = 200
        if request.headers.get("X-Test-Empty") == "True": # Check custom header for empty response
            return []
        if request.headers.get("X-Test-Api-Failure") == "True": # Check custom header for API failure
            context.status_code = 500
            return {}
        return [
            {"symbol": "AAPL", "qty": "10", "current_price": "170.0", "market_value": "1700.0", "sector": "Technology"},
            {"symbol": "MSFT", "qty": "5", "current_price": "300.0", "market_value": "1500.0", "sector": "Technology"}
        ]

    # Callback for dynamic Alpaca Order by ID responses
    def alpaca_order_id_callback(request, context):
        order_id = re.search(r"orders/([A-Za-z0-9-]+)", request.url).group(1)
        context.status_code = 200
        if order_id == "non-existent-id": # Test non-existent ID
            context.status_code = 404
            return {}
        return {"id": order_id, "status": "filled", "filled_avg_price": 170.1, "filled_qty": 10}


    # Mock Alpaca Quote endpoint (any symbol)
    requests_mock.get(
        re.compile(f"{data_api_base}/v2/stocks/quotes"),
        json=alpaca_quote_callback,
        status_code=200,
        complete_qs=False
    )

    # Mock Alpaca Order endpoint (for placing orders)
    requests_mock.post(
        f"{paper_api_base}/v2/orders",
        json={"id": "alpaca-order-id-123", "status": "new", "client_order_id": "test-client-id"},
        status_code=200
    )

    # Mock Alpaca Order by ID endpoint (any ID)
    requests_mock.get(
        re.compile(f"{paper_api_base}/v2/orders/[A-Za-z0-9-]+"), # Matches any order ID
        json=alpaca_order_id_callback,
        status_code=200,
        complete_qs=False
    )
    
    # Mock Alpaca Order by client ID endpoint (any client ID)
    requests_mock.get(
        re.compile(f"{paper_api_base}/v2/orders:by_client_order_id"),
        json=alpaca_client_order_callback,
        status_code=200,
        complete_qs=False
    )

    # Mock Alpaca Positions endpoint (dynamic responses)
    requests_mock.get(
        f"{paper_api_base}/v2/positions",
        json=alpaca_positions_callback,
        status_code=200, # Initial status, can be overridden by callback
    )
    
    return requests_mock

# Define a fixture for mocking Polygon API requests
@pytest.fixture
def polygon_requests_mock(requests_mock):
    
    # Mock Polygon EOD bars - Success
    requests_mock.get(
        re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/.*"), # Match any ticker and anything after
        json={
            "ticker": "AAPL", # Default ticker for success
            "results": [
                {"t": 1672531200000, "o": 170, "h": 172, "l": 169, "c": 171, "v": 100000},
                {"t": 1672617600000, "o": 171, "h": 173, "l": 170, "c": 172, "v": 120000},
            ],
            "status": "OK"
        },
        status_code=200,
        complete_qs=False
    )
    
    # Mock Polygon EOD bars - No data
    requests_mock.get(
        re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/NODATA/.*"), # Match NODATA and anything after
        json={"ticker": "NODATA", "results": [], "status": "OK"},
        status_code=200,
        complete_qs=False
    )
    
    # Mock Polygon EOD bars - API error
    requests_mock.get(
        re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/APIFAIL/.*"), # Match APIFAIL and anything after
        status_code=500,
        complete_qs=False
    )
    
    return requests_mock


# ============================================================================
# Test Cases
# ============================================================================

class TestApiConnectivity:

    def test_polygon_fetch_daily_bars_success(self, polygon_requests_mock, temp_cache_dir, mock_env_vars):
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        df = fetch_daily_bars_polygon("AAPL", start_date, end_date, cache_dir=temp_cache_dir)
        
        assert not df.empty
        assert len(df) == 2
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["close"].iloc[1] == 172.0
        
        # Ensure it was cached
        cache_file = temp_cache_dir / "polygon" / f"AAPL_{start_date}_{end_date}.csv"
        assert cache_file.exists()

    def test_polygon_fetch_daily_bars_no_data(self, polygon_requests_mock, temp_cache_dir, mock_env_vars):
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        df = fetch_daily_bars_polygon("NODATA", start_date, end_date, cache_dir=temp_cache_dir)
        
        assert df.empty
        # Ensure it was not cached (or cached empty)
        cache_file = temp_cache_dir / "polygon" / "NODATA_daily.csv"
        assert not cache_file.exists() or pd.read_csv(cache_file).empty

    def test_polygon_fetch_daily_bars_api_failure(self, polygon_requests_mock, temp_cache_dir, mock_env_vars):
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        df = fetch_daily_bars_polygon("APIFAIL", start_date, end_date, cache_dir=temp_cache_dir)
        
        assert df.empty
        cache_file = temp_cache_dir / "polygon" / "APIFAIL_daily.csv"
        assert not cache_file.exists() # Should not cache failed responses


    def test_alpaca_get_best_ask_success(self, alpaca_requests_mock, mock_env_vars):
        ask = get_best_ask("AAPL")
        assert ask == 170.0

    def test_alpaca_get_best_ask_failure(self, alpaca_requests_mock, mock_env_vars):
        # To simulate failure, we need to remove the mock or add a specific one for failure
        # For now, let's just test a symbol not mocked
        ask = get_best_ask("UNKNOWN")
        assert ask is None
        
    def test_alpaca_get_order_by_id_success(self, alpaca_requests_mock, mock_env_vars):
        order_data = get_order_by_id("alpaca-order-id-123")
        assert order_data is not None
        assert order_data["status"] == "filled"
        
    def test_alpaca_get_order_by_id_failure(self, alpaca_requests_mock, mock_env_vars):
        # Unmocked ID should return None
        order_data = get_order_by_id("non-existent-id")
        assert order_data is None

    def test_alpaca_get_order_by_client_id_success(self, alpaca_requests_mock, mock_env_vars):
        order_data = get_order_by_client_id("test-client-id")
        assert order_data is not None
        assert order_data["status"] == "filled"

    def test_alpaca_get_order_by_client_id_failure(self, alpaca_requests_mock, mock_env_vars):
        order_data = get_order_by_client_id("non-existent-client-id")
        assert order_data is None
        
    def test_alpaca_get_positions_success(self, alpaca_requests_mock, mock_env_vars):
        from risk.position_limit_gate import PositionLimitGate
        gate = PositionLimitGate()
        positions = gate._fetch_positions() # Accessing internal method for testing
        
        assert len(positions) == 2
        assert positions[0]["symbol"] == "AAPL"
        assert positions[1]["symbol"] == "MSFT"
        assert positions[0]["market_value"] == 1700.0 # Check if market_value was added
        assert positions[1]["sector"] == "Technology" # Check if sector was added (assuming mock or _get_symbol_sector for MSFT)

    def test_alpaca_get_positions_empty(self, alpaca_requests_mock, mock_env_vars):
        # Configure the mock to return an empty list for positions
        alpaca_requests_mock.get(
            "https://paper-api.alpaca.markets/v2/positions",
            json=[],
            status_code=200
        )
        
        from risk.position_limit_gate import PositionLimitGate
        gate = PositionLimitGate()
        positions = gate._fetch_positions()
        
        assert len(positions) == 0

    def test_alpaca_get_positions_api_failure(self, alpaca_requests_mock, mock_env_vars):
        # Configure the mock to return an API error for positions
        alpaca_requests_mock.get(
            "https://paper-api.alpaca.markets/v2/positions",
            status_code=500
        )
        
        from risk.position_limit_gate import PositionLimitGate
        gate = PositionLimitGate()
        positions = gate._fetch_positions()
        
        assert len(positions) == 0
