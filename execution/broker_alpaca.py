from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

import requests

from oms.order_state import OrderRecord, OrderStatus
from oms.idempotency_store import IdempotencyStore
from core.rate_limiter import with_retry
from config.settings_loader import (
    is_clamp_enabled,
    get_clamp_max_pct,
    get_clamp_use_atr,
    get_clamp_atr_multiple,
)


ALPACA_ORDERS_URL = "/v2/orders"
ALPACA_QUOTES_URL = "/v2/stocks/quotes"


@dataclass
class AlpacaConfig:
    base_url: str
    key_id: str
    secret: str


def _alpaca_cfg() -> AlpacaConfig:
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    return AlpacaConfig(
        base_url=base.rstrip("/"),
        key_id=os.getenv("ALPACA_API_KEY_ID", ""),
        secret=os.getenv("ALPACA_API_SECRET_KEY", ""),
    )


def _auth_headers(cfg: AlpacaConfig) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": cfg.key_id,
        "APCA-API-SECRET-KEY": cfg.secret,
        "Content-Type": "application/json",
    }


def get_best_ask(symbol: str, timeout: int = 5) -> Optional[float]:
    """Fetch best ask from Alpaca market data. Fallback to None if unavailable."""
    cfg = _alpaca_cfg()
    if not cfg.key_id or not cfg.secret:
        return None
    url = f"{cfg.base_url}{ALPACA_QUOTES_URL}?symbols={symbol.upper()}"
    try:
        r = requests.get(url, headers=_auth_headers(cfg), timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        quotes = data.get("quotes") or data.get("quotes_by_symbol")
        # API shapes may vary; attempt a couple of shapes
        if isinstance(quotes, dict):
            arr = quotes.get(symbol.upper()) or []
        else:
            arr = quotes or []
        if not arr:
            return None
        q = arr[-1]
        ask = q.get("ap") or q.get("ask_price")
        return float(ask) if ask is not None else None
    except Exception:
        return None


def place_ioc_limit(order: OrderRecord) -> OrderRecord:
    """Place an IOC LIMIT order via Alpaca. Returns updated OrderRecord."""
    cfg = _alpaca_cfg()
    store = IdempotencyStore()
    # Idempotency guard
    if store.exists(order.decision_id):
        order.status = OrderStatus.CLOSED
        order.notes = "duplicate_decision_id"
        return order

    url = f"{cfg.base_url}{ALPACA_ORDERS_URL}"
    payload = {
        "symbol": order.symbol.upper(),
        "qty": order.qty,
        "side": "buy" if order.side.lower() == "long" or order.side.upper() == "BUY" else "sell",
        "type": "limit",
        "time_in_force": "ioc",
        "limit_price": float(order.limit_price),
        "client_order_id": order.idempotency_key,
        "extended_hours": False,
    }
    try:
        def _post():
            return requests.post(url, json=payload, headers=_auth_headers(cfg), timeout=10)

        r = with_retry(_post)
        if r.status_code not in (200, 201):
            order.status = OrderStatus.REJECTED
            order.notes = f"alpaca_http_{r.status_code}"
            return order
        data = r.json()
        order.broker_order_id = data.get("id")
        order.status = OrderStatus.SUBMITTED
        store.put(order.decision_id, order.idempotency_key)
        return order
    except Exception as e:
        order.status = OrderStatus.REJECTED
        order.notes = f"exception:{e}"
        return order


def _apply_clamp(raw_limit: float, best_quote: float, atr_value: Optional[float] = None) -> float:
    """
    Apply LULD/volatility clamp to limit price (config-gated).
    Clamps the limit price to be within a reasonable band from the quote.

    Args:
        raw_limit: The raw limit price (e.g., best_ask * 1.001)
        best_quote: The current best quote (ask for buys, bid for sells)
        atr_value: Optional ATR(14) value for ATR-based clamping

    Returns:
        Clamped limit price
    """
    if not is_clamp_enabled():
        return raw_limit

    if get_clamp_use_atr() and atr_value is not None and atr_value > 0:
        # ATR-based clamp: quote ± (ATR × multiple)
        atr_multiple = get_clamp_atr_multiple()
        max_deviation = atr_value * atr_multiple
        lower_bound = best_quote - max_deviation
        upper_bound = best_quote + max_deviation
    else:
        # Fixed percentage clamp
        max_pct = get_clamp_max_pct()
        lower_bound = best_quote * (1 - max_pct)
        upper_bound = best_quote * (1 + max_pct)

    # Clamp the limit price within bounds
    clamped = max(lower_bound, min(raw_limit, upper_bound))
    return round(clamped, 2)


def construct_decision(
    symbol: str,
    side: str,
    qty: int,
    best_ask: Optional[float],
    atr_value: Optional[float] = None,
) -> OrderRecord:
    """
    Construct an order decision with optional LULD/volatility clamping.

    Args:
        symbol: Stock symbol
        side: "BUY" or "SELL"
        qty: Number of shares
        best_ask: Best ask price from quote
        atr_value: Optional ATR(14) for ATR-based clamping
    """
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    decision_id = f"DEC_{now}_{symbol.upper()}_{uuid.uuid4().hex[:6].upper()}"
    idk = decision_id  # idempotency = decision id

    # Limit: best ask + 0.1% (fallback to None -> caller must guard)
    if best_ask:
        raw_limit = best_ask * 1.001
        limit_price = _apply_clamp(raw_limit, best_ask, atr_value)
    else:
        limit_price = None

    return OrderRecord(
        decision_id=decision_id,
        signal_id=decision_id.replace("DEC_", "SIG_"),
        symbol=symbol.upper(),
        side=side.upper(),
        qty=int(qty),
        limit_price=float(limit_price) if limit_price else 0.0,
        tif="IOC",
        order_type="IOC_LIMIT",
        idempotency_key=idk,
        created_at=datetime.utcnow(),
    )
