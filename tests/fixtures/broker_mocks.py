"""
Alpaca broker mocks for testing.

Provides mock implementations for:
- Full broker class mock
- API endpoint mocks (orders, quotes, account, positions)
- Success/failure response generators
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock
import uuid


class MockAlpacaBroker:
    """
    Full mock implementation of Alpaca broker for testing.

    Tracks orders placed, fills, and positions without real API calls.
    """

    def __init__(
        self,
        account_equity: float = 100000.0,
        buying_power: float = 200000.0,
    ):
        self.account_equity = account_equity
        self.buying_power = buying_power
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: List[Dict[str, Any]] = []
        self.order_count = 0
        self.should_reject = False
        self.rejection_reason = ""
        self.fill_delay_bars = 0
        self.quotes: Dict[str, Dict[str, float]] = {}

    def set_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: int = 100,
        ask_size: int = 100,
    ):
        """Set quote for a symbol."""
        self.quotes[symbol] = {
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    def get_best_ask(self, symbol: str) -> Optional[float]:
        """Get best ask price for symbol."""
        if symbol in self.quotes:
            return self.quotes[symbol]["ask"]
        return None

    def get_best_bid(self, symbol: str) -> Optional[float]:
        """Get best bid price for symbol."""
        if symbol in self.quotes:
            return self.quotes[symbol]["bid"]
        return None

    def place_ioc_limit(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
    ) -> Dict[str, Any]:
        """Place IOC limit order."""
        self.order_count += 1
        order_id = str(uuid.uuid4())

        if self.should_reject:
            return {
                "id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "limit_price": limit_price,
                "status": "rejected",
                "reject_reason": self.rejection_reason or "Insufficient buying power",
                "filled_qty": 0,
                "filled_avg_price": None,
            }

        # Simulate fill
        fill_price = limit_price if side == "buy" else limit_price
        if symbol in self.quotes:
            fill_price = self.quotes[symbol]["ask"] if side == "buy" else self.quotes[symbol]["bid"]

        order = {
            "id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "limit_price": limit_price,
            "status": "filled",
            "filled_qty": qty,
            "filled_avg_price": fill_price,
            "filled_at": datetime.now().isoformat(),
        }

        self.orders.append(order)

        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "qty": 0,
                "avg_entry_price": 0,
            }

        pos = self.positions[symbol]
        if side == "buy":
            new_qty = pos["qty"] + qty
            if new_qty > 0:
                pos["avg_entry_price"] = (
                    (pos["avg_entry_price"] * pos["qty"] + fill_price * qty) / new_qty
                )
            pos["qty"] = new_qty
        else:
            pos["qty"] -= qty
            if pos["qty"] == 0:
                del self.positions[symbol]

        return order

    def get_account(self) -> Dict[str, Any]:
        """Get account info."""
        return {
            "equity": self.account_equity,
            "buying_power": self.buying_power,
            "cash": self.account_equity * 0.5,
            "portfolio_value": self.account_equity,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)

    def set_reject_orders(self, reject: bool, reason: str = ""):
        """Configure broker to reject orders."""
        self.should_reject = reject
        self.rejection_reason = reason

    def reset(self):
        """Reset broker state."""
        self.positions.clear()
        self.orders.clear()
        self.order_count = 0
        self.should_reject = False
        self.rejection_reason = ""


def mock_alpaca_api(requests_mock) -> MockAlpacaBroker:
    """
    Set up all Alpaca API endpoint mocks.

    Args:
        requests_mock: pytest requests_mock fixture

    Returns:
        MockAlpacaBroker instance for assertions
    """
    broker = MockAlpacaBroker()
    base_url = "https://paper-api.alpaca.markets"

    # Account endpoint
    requests_mock.get(
        f"{base_url}/v2/account",
        json=mock_account_response(),
    )

    # Orders endpoint
    def orders_callback(request, context):
        if request.method == "POST":
            body = json.loads(request.body)
            result = broker.place_ioc_limit(
                symbol=body.get("symbol"),
                qty=int(body.get("qty", 0)),
                side=body.get("side"),
                limit_price=float(body.get("limit_price", 0)),
            )
            context.status_code = 200 if result["status"] == "filled" else 422
            return result
        return broker.orders

    requests_mock.register_uri(
        "POST",
        f"{base_url}/v2/orders",
        json=orders_callback,
    )

    requests_mock.get(
        f"{base_url}/v2/orders",
        json=lambda req, ctx: broker.orders,
    )

    # Positions endpoint
    requests_mock.get(
        f"{base_url}/v2/positions",
        json=lambda req, ctx: broker.get_positions(),
    )

    # Latest quote endpoint
    def quote_callback(request, context):
        symbol = request.path.split("/")[-1].replace("/latest/quote", "")
        if symbol in broker.quotes:
            return {
                "symbol": symbol,
                "quote": broker.quotes[symbol],
            }
        # Default quote
        return {
            "symbol": symbol,
            "quote": {"bid": 99.0, "ask": 100.0, "bid_size": 100, "ask_size": 100},
        }

    requests_mock.get(
        f"{base_url}/v2/stocks/TEST/quotes/latest",
        json=quote_callback,
    )

    return broker


def mock_order_success(
    symbol: str = "TEST",
    qty: int = 10,
    side: str = "buy",
    filled_price: float = 100.0,
) -> Dict[str, Any]:
    """Generate successful order response."""
    return {
        "id": str(uuid.uuid4()),
        "client_order_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "submitted_at": datetime.now().isoformat(),
        "filled_at": datetime.now().isoformat(),
        "symbol": symbol,
        "asset_class": "us_equity",
        "qty": str(qty),
        "filled_qty": str(qty),
        "filled_avg_price": str(filled_price),
        "type": "limit",
        "side": side,
        "time_in_force": "ioc",
        "status": "filled",
    }


def mock_order_rejected(
    symbol: str = "TEST",
    reason: str = "insufficient buying power",
) -> Dict[str, Any]:
    """Generate rejected order response."""
    return {
        "id": str(uuid.uuid4()),
        "client_order_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "symbol": symbol,
        "status": "rejected",
        "reject_reason": reason,
        "filled_qty": "0",
    }


def mock_quote_response(
    bid: float = 99.0,
    ask: float = 100.0,
    symbol: str = "TEST",
) -> Dict[str, Any]:
    """Generate quote response."""
    return {
        "symbol": symbol,
        "quote": {
            "ap": ask,
            "as": 100,
            "bp": bid,
            "bs": 100,
            "t": datetime.now().isoformat(),
        },
    }


def mock_account_response(
    equity: float = 100000.0,
    buying_power: float = 200000.0,
) -> Dict[str, Any]:
    """Generate account response."""
    return {
        "id": str(uuid.uuid4()),
        "account_number": "PA123456",
        "status": "ACTIVE",
        "currency": "USD",
        "cash": str(equity * 0.5),
        "portfolio_value": str(equity),
        "pattern_day_trader": False,
        "trading_blocked": False,
        "transfers_blocked": False,
        "account_blocked": False,
        "equity": str(equity),
        "buying_power": str(buying_power),
        "daytrading_buying_power": str(buying_power * 2),
    }


def mock_positions_response(
    positions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate positions response."""
    if positions is None:
        return []

    return [
        {
            "asset_id": str(uuid.uuid4()),
            "symbol": pos.get("symbol", "TEST"),
            "qty": str(pos.get("qty", 10)),
            "avg_entry_price": str(pos.get("avg_entry_price", 100.0)),
            "market_value": str(pos.get("qty", 10) * pos.get("current_price", 100.0)),
            "unrealized_pl": str(
                pos.get("qty", 10) * (pos.get("current_price", 100.0) - pos.get("avg_entry_price", 100.0))
            ),
            "side": "long" if pos.get("qty", 10) > 0 else "short",
        }
        for pos in positions
    ]
