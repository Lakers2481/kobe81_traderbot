from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
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
from risk.liquidity_gate import LiquidityGate, LiquidityCheck

logger = logging.getLogger(__name__)


ALPACA_ORDERS_URL = "/v2/orders"
ALPACA_QUOTES_URL = "/v2/stocks/quotes"
ALPACA_BARS_URL = "/v2/stocks/bars"

# Default liquidity gate - can be overridden via set_liquidity_gate()
_liquidity_gate: Optional[LiquidityGate] = None
_liquidity_gate_enabled: bool = True  # Global toggle


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


def _fetch_quotes(symbol: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
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
        return q
    except Exception:
        return None


def get_best_ask(symbol: str, timeout: int = 5) -> Optional[float]:
    """Fetch best ask from Alpaca market data. Fallback to None if unavailable."""
    q = _fetch_quotes(symbol, timeout=timeout)
    if not q:
        return None
    ask = q.get("ap") or q.get("ask_price")
    try:
        return float(ask) if ask is not None else None
    except Exception:
        return None


def get_best_bid(symbol: str, timeout: int = 5) -> Optional[float]:
    """Fetch best bid from Alpaca market data. Fallback to None if unavailable."""
    q = _fetch_quotes(symbol, timeout=timeout)
    if not q:
        return None
    bid = q.get("bp") or q.get("bid_price")
    try:
        return float(bid) if bid is not None else None
    except Exception:
        return None


def get_quote_with_sizes(symbol: str, timeout: int = 5) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Fetch bid, ask, bid size, and ask size from Alpaca.

    Returns:
        Tuple of (bid, ask, bid_size, ask_size) - any can be None if unavailable
    """
    q = _fetch_quotes(symbol, timeout=timeout)
    if not q:
        return None, None, None, None

    try:
        bid = float(q.get("bp") or q.get("bid_price") or 0) or None
        ask = float(q.get("ap") or q.get("ask_price") or 0) or None
        bid_size = int(q.get("bs") or q.get("bid_size") or 0) or None
        ask_size = int(q.get("as") or q.get("ask_size") or 0) or None
        return bid, ask, bid_size, ask_size
    except Exception:
        return None, None, None, None


def get_avg_volume(symbol: str, lookback_days: int = 20, timeout: int = 10) -> Optional[int]:
    """
    Fetch average daily volume from Alpaca bars.

    Args:
        symbol: Stock symbol
        lookback_days: Number of days to average (default: 20)
        timeout: Request timeout in seconds

    Returns:
        Average daily volume in shares, or None if unavailable
    """
    cfg = _alpaca_cfg()
    if not cfg.key_id or not cfg.secret:
        return None

    # Use data API for historical bars
    # Paper API uses same data endpoint
    data_base = cfg.base_url.replace("paper-api", "data").replace("api", "data")
    if "data.alpaca.markets" not in data_base:
        data_base = "https://data.alpaca.markets"

    url = f"{data_base}{ALPACA_BARS_URL}"
    params = {
        "symbols": symbol.upper(),
        "timeframe": "1Day",
        "limit": lookback_days,
    }

    try:
        r = requests.get(url, params=params, headers=_auth_headers(cfg), timeout=timeout)
        if r.status_code != 200:
            return None

        data = r.json()
        bars = data.get("bars", {}).get(symbol.upper(), [])

        if not bars:
            return None

        # Calculate average volume
        volumes = [bar.get("v", 0) for bar in bars if bar.get("v")]
        if not volumes:
            return None

        return int(sum(volumes) / len(volumes))
    except Exception as e:
        logger.debug(f"Failed to fetch avg volume for {symbol}: {e}")
        return None


# ============================================================================
# Liquidity Gate Integration
# ============================================================================

def get_liquidity_gate() -> LiquidityGate:
    """Get the liquidity gate instance (creates default if needed)."""
    global _liquidity_gate
    if _liquidity_gate is None:
        _liquidity_gate = LiquidityGate(
            min_adv_usd=100_000,
            max_spread_pct=0.50,
            max_pct_of_adv=1.0,
        )
    return _liquidity_gate


def set_liquidity_gate(gate: LiquidityGate) -> None:
    """Set a custom liquidity gate instance."""
    global _liquidity_gate
    _liquidity_gate = gate


def enable_liquidity_gate(enabled: bool = True) -> None:
    """Enable or disable liquidity checking globally."""
    global _liquidity_gate_enabled
    _liquidity_gate_enabled = enabled
    logger.info(f"Liquidity gate {'enabled' if enabled else 'disabled'}")


def is_liquidity_gate_enabled() -> bool:
    """Check if liquidity gate is enabled."""
    return _liquidity_gate_enabled


def check_liquidity_for_order(
    symbol: str,
    qty: int,
    price: Optional[float] = None,
    strict: bool = True,
) -> LiquidityCheck:
    """
    Check if an order passes liquidity requirements.

    Args:
        symbol: Stock symbol
        qty: Number of shares
        price: Price per share (fetches current quote if None)
        strict: If True, any issue fails. If False, only critical issues fail.

    Returns:
        LiquidityCheck with pass/fail result and details
    """
    gate = get_liquidity_gate()

    # Fetch current quote
    bid, ask, _, _ = get_quote_with_sizes(symbol)

    # Use mid price if price not provided
    if price is None:
        if bid and ask:
            price = (bid + ask) / 2
        elif ask:
            price = ask
        elif bid:
            price = bid
        else:
            # Can't check without price
            return LiquidityCheck(
                symbol=symbol,
                passed=False,
                reason="Unable to fetch quote data",
            )

    # Fetch average volume
    avg_volume = get_avg_volume(symbol)

    # Run liquidity check
    return gate.check_liquidity(
        symbol=symbol,
        price=price,
        shares=qty,
        bid=bid,
        ask=ask,
        avg_volume=avg_volume,
        strict=strict,
    )


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


# ============================================================================
# Liquidity-Gated Order Placement
# ============================================================================

@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    order: OrderRecord
    liquidity_check: Optional[LiquidityCheck] = None
    blocked_by_liquidity: bool = False

    @property
    def success(self) -> bool:
        """True if order was submitted (not rejected or blocked)."""
        return (
            not self.blocked_by_liquidity
            and self.order.status == OrderStatus.SUBMITTED
        )


def place_order_with_liquidity_check(
    order: OrderRecord,
    strict: bool = True,
    bypass_if_disabled: bool = True,
) -> OrderResult:
    """
    Place an order with liquidity gate validation.

    This is the recommended entry point for order placement. It:
    1. Checks liquidity requirements (ADV, spread, order impact)
    2. Rejects orders that fail liquidity checks
    3. Places order via Alpaca if checks pass

    Args:
        order: The OrderRecord to place
        strict: If True, any liquidity issue blocks the order.
                If False, only critical issues (low ADV, wide spread) block.
        bypass_if_disabled: If True and liquidity gate is disabled, skip check.

    Returns:
        OrderResult with order status and liquidity check details
    """
    # Check if liquidity gate is enabled
    if not is_liquidity_gate_enabled() and bypass_if_disabled:
        logger.debug(f"Liquidity gate disabled, bypassing check for {order.symbol}")
        placed_order = place_ioc_limit(order)
        return OrderResult(order=placed_order, liquidity_check=None, blocked_by_liquidity=False)

    # Run liquidity check
    liq_check = check_liquidity_for_order(
        symbol=order.symbol,
        qty=order.qty,
        price=order.limit_price if order.limit_price > 0 else None,
        strict=strict,
    )

    # Log the check result
    if liq_check.passed:
        logger.info(
            f"Liquidity check PASSED for {order.symbol}: "
            f"ADV=${liq_check.adv_usd:,.0f}, spread={liq_check.spread_pct:.2f}%"
        )
    else:
        logger.warning(
            f"Liquidity check FAILED for {order.symbol}: {liq_check.reason}"
        )
        # Reject the order
        order.status = OrderStatus.REJECTED
        order.notes = f"liquidity_gate:{liq_check.reason}"
        return OrderResult(
            order=order,
            liquidity_check=liq_check,
            blocked_by_liquidity=True,
        )

    # Liquidity check passed - place the order
    placed_order = place_ioc_limit(order)

    return OrderResult(
        order=placed_order,
        liquidity_check=liq_check,
        blocked_by_liquidity=False,
    )


def execute_signal(
    symbol: str,
    side: str,
    qty: int,
    atr_value: Optional[float] = None,
    check_liquidity: bool = True,
    strict_liquidity: bool = True,
) -> OrderResult:
    """
    High-level function to execute a trading signal with all safety checks.

    This is the recommended entry point for signal execution. It:
    1. Fetches current market quote
    2. Constructs order with LULD clamping
    3. Validates liquidity requirements
    4. Places order via Alpaca

    Args:
        symbol: Stock symbol
        side: "BUY" or "SELL"
        qty: Number of shares
        atr_value: Optional ATR for price clamping
        check_liquidity: Whether to run liquidity checks (default: True)
        strict_liquidity: If True, any issue blocks. If False, only critical.

    Returns:
        OrderResult with full execution details

    Example:
        result = execute_signal("AAPL", "BUY", 100)
        if result.success:
            print(f"Order placed: {result.order.broker_order_id}")
        elif result.blocked_by_liquidity:
            print(f"Blocked: {result.liquidity_check.reason}")
        else:
            print(f"Rejected: {result.order.notes}")
    """
    # Fetch best ask for limit price construction
    best_ask = get_best_ask(symbol)

    if best_ask is None:
        # Can't proceed without a quote
        order = OrderRecord(
            decision_id=f"DEC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{symbol}_NOQUOTE",
            signal_id="",
            symbol=symbol.upper(),
            side=side.upper(),
            qty=qty,
            limit_price=0.0,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key="",
            created_at=datetime.utcnow(),
            status=OrderStatus.REJECTED,
            notes="no_quote_available",
        )
        return OrderResult(order=order, liquidity_check=None, blocked_by_liquidity=False)

    # Construct the order
    order = construct_decision(
        symbol=symbol,
        side=side,
        qty=qty,
        best_ask=best_ask,
        atr_value=atr_value,
    )

    # Place with liquidity check
    if check_liquidity and is_liquidity_gate_enabled():
        return place_order_with_liquidity_check(
            order=order,
            strict=strict_liquidity,
        )
    else:
        placed_order = place_ioc_limit(order)
        return OrderResult(order=placed_order, liquidity_check=None, blocked_by_liquidity=False)
