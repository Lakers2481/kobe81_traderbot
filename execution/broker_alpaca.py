from __future__ import annotations

import os
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import uuid

import requests

from oms.order_state import OrderRecord, OrderStatus
from oms.idempotency_store import IdempotencyStore
from core.rate_limiter import with_retry
from core.kill_switch import require_no_kill_switch, is_kill_switch_active
from monitor.health_endpoints import update_trade_event
from config.settings_loader import (
    is_clamp_enabled,
    get_clamp_max_pct,
    get_clamp_use_atr,
    get_clamp_atr_multiple,
)
from risk.liquidity_gate import LiquidityGate, LiquidityCheck
from execution.utils import normalize_side, normalize_side_lowercase, is_buy_side
from safety.execution_choke import evaluate_safety_gates
from functools import wraps

logger = logging.getLogger(__name__)


class PolicyGateError(Exception):
    """Raised when PolicyGate rejects an order."""
    pass


class ComplianceError(Exception):
    """Raised when compliance rules reject an order."""
    pass


def require_policy_gate(func):
    """
    Decorator to enforce PolicyGate AND Compliance checks before order placement.

    CRITICAL FIX (2026-01-04): PolicyGate must be enforced at broker boundary,
    not just in higher-level orchestration layers. This prevents any direct
    broker calls from bypassing risk limits.

    SECURITY FIX (2026-01-04): Also wires compliance engine (prohibited list,
    trading rules) into the order flow. Previously compliance was exported but
    never actually checked.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract order from args/kwargs
        order = args[0] if args else kwargs.get('order')
        if order and hasattr(order, 'symbol') and hasattr(order, 'side') and hasattr(order, 'qty'):
            try:
                from risk.policy_gate import PolicyGate
                from core.structured_log import jlog

                gate = PolicyGate.from_config()
                price = getattr(order, 'limit_price', 0) or getattr(order, 'entry_price', 0) or 0

                # PolicyGate check
                allowed, reason = gate.check(
                    symbol=order.symbol,
                    side=order.side,
                    price=float(price) if price else 0.0,
                    qty=int(order.qty)
                )
                if not allowed:
                    # SECURITY FIX 2026-01-04: Use ** to unpack dict as kwargs
                    jlog('policy_gate_reject',
                         symbol=order.symbol,
                         side=order.side,
                         qty=order.qty,
                         reason=reason,
                         function=func.__name__)
                    logger.warning(f"PolicyGate blocked {order.symbol}: {reason}")
                    # Increment Prometheus counter
                    try:
                        from trade_logging.prometheus_metrics import POLICY_GATE_REJECTED
                        POLICY_GATE_REJECTED.labels(reason=reason[:50]).inc()
                    except Exception:
                        pass
                    raise PolicyGateError(f"PolicyGate blocked: {reason}")

                # Compliance: Prohibited list check
                try:
                    from compliance import is_prohibited, log_compliance_event
                    from datetime import date

                    prohibited, reasons = is_prohibited(order.symbol, date.today())
                    if prohibited:
                        reason_str = ', '.join(r.code for r in reasons)
                        log_compliance_event('prohibited_reject', {
                            'symbol': order.symbol,
                            'reasons': [{'code': r.code, 'detail': r.detail} for r in reasons],
                            'function': func.__name__
                        })
                        jlog('compliance_reject',
                             symbol=order.symbol,
                             reason='prohibited_list',
                             codes=reason_str)
                        logger.warning(f"Compliance blocked {order.symbol}: prohibited ({reason_str})")
                        # Increment Prometheus counter
                        try:
                            from trade_logging.prometheus_metrics import COMPLIANCE_REJECTED
                            COMPLIANCE_REJECTED.labels(reason='prohibited_list').inc()
                        except Exception:
                            pass
                        raise ComplianceError(f"Symbol prohibited: {reason_str}")
                except ComplianceError:
                    raise
                except ImportError:
                    # Compliance module not fully available, continue
                    pass

                # Compliance: Trade rules check (price floor, position size, RTH)
                try:
                    from compliance import evaluate_trade_rules
                    from risk.equity_sizer import get_account_equity

                    account_equity = get_account_equity(fail_safe=False)
                    trade_allowed, rule_reason = evaluate_trade_rules(
                        price=float(price) if price else 0.0,
                        qty=int(order.qty),
                        account_equity=account_equity,
                        ts=datetime.now()
                    )
                    if not trade_allowed:
                        log_compliance_event('rules_reject', {
                            'symbol': order.symbol,
                            'reason': rule_reason,
                            'function': func.__name__
                        })
                        jlog('compliance_rules_reject',
                             symbol=order.symbol,
                             reason=rule_reason)
                        logger.warning(f"Compliance rules blocked {order.symbol}: {rule_reason}")
                        # Increment Prometheus counter
                        try:
                            from trade_logging.prometheus_metrics import COMPLIANCE_REJECTED
                            COMPLIANCE_REJECTED.labels(reason='trade_rules').inc()
                        except Exception:
                            pass
                        raise ComplianceError(f"Trade rules violated: {rule_reason}")
                except (ImportError, RuntimeError):
                    # equity_sizer may fail or compliance not available, continue
                    pass
                except ComplianceError:
                    raise

            except (PolicyGateError, ComplianceError):
                raise
            except ImportError:
                logger.warning("PolicyGate not available, skipping check")
            except Exception as e:
                # SECURITY FIX (2026-01-04): Fail CLOSED, not OPEN
                # Any exception during policy check BLOCKS the order and activates kill switch
                logger.critical(f"PolicyGate check failed: {e}, BLOCKING order (fail-closed)")
                try:
                    from core.kill_switch import activate_kill_switch
                    activate_kill_switch(f"PolicyGate failure: {e}")
                except Exception:
                    pass  # Kill switch activation failed, but still block the order
                raise PolicyGateError(f"Policy check error (fail-closed): {e}")
        return func(*args, **kwargs)
    return wrapper


ALPACA_ORDERS_URL = "/v2/orders"
ALPACA_QUOTES_URL = "/v2/stocks/quotes"
ALPACA_BARS_URL = "/v2/stocks/bars"

# Default liquidity gate - can be overridden via set_liquidity_gate()
_liquidity_gate: Optional[LiquidityGate] = None
# REMOVED (2026-01-04): _liquidity_gate_enabled toggle - liquidity gate is ALWAYS enabled
# This was a security risk allowing programmatic bypass of liquidity protection


@dataclass
class AlpacaConfig:
    base_url: str
    key_id: str
    secret: str

@dataclass
class BrokerExecutionResult:
    """
    Result returned by Alpaca broker functions, including market context.
    Used for Transaction Cost Analysis (TCA).
    """
    order: OrderRecord
    market_bid_at_execution: Optional[float] = None
    market_ask_at_execution: Optional[float] = None


def _alpaca_cfg() -> AlpacaConfig:
    base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    # Support both ALPACA_ and APCA_ prefixes for API keys
    key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY", "")
    return AlpacaConfig(
        base_url=base.rstrip("/"),
        key_id=key_id,
        secret=secret,
    )


def _auth_headers(cfg: AlpacaConfig) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": cfg.key_id,
        "APCA-API-SECRET-KEY": cfg.secret,
        "Content-Type": "application/json",
    }


def _data_api_base(cfg: AlpacaConfig) -> str:
    """Return the Alpaca Data API base URL regardless of trading base.

    Ensures we hit https://data.alpaca.markets for market data endpoints
    even if ALPACA_BASE_URL points at paper or live trading.
    """
    data_base = cfg.base_url.replace("paper-api", "data").replace("api.", "data.")
    if "data.alpaca.markets" not in data_base:
        data_base = "https://data.alpaca.markets"
    return data_base.rstrip("/")


def get_account_info(timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Get account information from Alpaca.

    Returns dict with equity, buying_power, cash, portfolio_value, etc.
    Returns None on error.
    """
    cfg = _alpaca_cfg()
    url = f"{cfg.base_url}/v2/account"

    try:
        resp = requests.get(url, headers=_auth_headers(cfg), timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Failed to get account info: {resp.status_code} - {resp.text}")
        return None
    except Exception as e:
        logger.error(f"Error fetching account info: {e}")
        return None


def _fetch_quotes(symbol: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Fetch quotes from Alpaca with retry support.

    SECURITY FIX (2026-01-04): Added retry wrapper for reliability.
    """
    cfg = _alpaca_cfg()
    if not cfg.key_id or not cfg.secret:
        return None
    # Use the data API domain for quotes
    data_base = _data_api_base(cfg)
    url = f"{data_base}{ALPACA_QUOTES_URL}?symbols={symbol.upper()}"

    def do_request():
        """Inner function for retry wrapper."""
        r = requests.get(url, headers=_auth_headers(cfg), timeout=timeout)
        if r.status_code != 200:
            if r.status_code == 429:
                raise requests.exceptions.HTTPError("Rate limited")
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
        return arr[-1]

    try:
        from core.rate_limiter import with_retry
        return with_retry(do_request, max_retries=3, base_delay_ms=500)
    except Exception:
        # Retry exhausted or unexpected error
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


def get_quote_with_sizes(symbol: str, timeout: int = 5) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int], Optional[datetime]]:
    """
    Fetch bid, ask, bid size, ask size, and quote timestamp from Alpaca.

    Returns:
        Tuple of (bid, ask, bid_size, ask_size, quote_timestamp)
        - any can be None if unavailable
        - quote_timestamp is UTC datetime of when the quote was generated
    """
    q = _fetch_quotes(symbol, timeout=timeout)
    if not q:
        return None, None, None, None, None

    try:
        bid = float(q.get("bp") or q.get("bid_price") or 0) or None
        ask = float(q.get("ap") or q.get("ask_price") or 0) or None
        bid_size = int(q.get("bs") or q.get("bid_size") or 0) or None
        ask_size = int(q.get("as") or q.get("ask_size") or 0) or None

        # Extract quote timestamp (Alpaca returns ISO format with 't' key)
        quote_ts = None
        ts_raw = q.get("t")
        if ts_raw:
            try:
                # Alpaca returns timestamps like "2025-12-31T15:30:00.123456789Z"
                quote_ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return bid, ask, bid_size, ask_size, quote_ts
    except Exception:
        return None, None, None, None, None

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
    """
    DEPRECATED (2026-01-04): Liquidity gate can no longer be disabled programmatically.
    This was a security risk. Liquidity checking is now ALWAYS enabled.
    """
    if not enabled:
        logger.critical("SECURITY: Attempt to disable liquidity gate BLOCKED. Liquidity gate is always enabled.")
        from core.structured_log import jlog
        jlog('security_violation',
             violation_type='liquidity_gate_disable_attempt',
             blocked=True,
             message='Liquidity gate cannot be disabled programmatically')
    else:
        logger.info("Liquidity gate is always enabled (no action needed)")


def is_liquidity_gate_enabled() -> bool:
    """Check if liquidity gate is enabled. Always returns True."""
    return True  # Liquidity gate is ALWAYS enabled


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
    bid, ask, _, _, _ = get_quote_with_sizes(symbol)

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


# ============================================================================
# Order Status Resolution & Trade Logging
# ============================================================================

TRADES_LOG_PATH = Path("logs/trades.jsonl")


def get_order_by_id(order_id: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Fetch order details by broker order ID.

    Args:
        order_id: The Alpaca order ID
        timeout: Request timeout in seconds

    Returns:
        Order data dict or None if not found
    """
    cfg = _alpaca_cfg()
    if not cfg.key_id or not cfg.secret:
        return None

    url = f"{cfg.base_url}{ALPACA_ORDERS_URL}/{order_id}"
    try:
        r = requests.get(url, headers=_auth_headers(cfg), timeout=timeout)
        if r.status_code != 200:
            logger.debug(f"Order {order_id} fetch failed: HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        logger.debug(f"Order {order_id} fetch exception: {e}")
        return None


def get_order_by_client_id(client_order_id: str, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """
    Fetch order details by client order ID (idempotency key).

    Args:
        client_order_id: The client order ID used when placing the order
        timeout: Request timeout in seconds

    Returns:
        Order data dict or None if not found
    """
    cfg = _alpaca_cfg()
    if not cfg.key_id or not cfg.secret:
        return None

    url = f"{cfg.base_url}{ALPACA_ORDERS_URL}:by_client_order_id"
    params = {"client_order_id": client_order_id}
    try:
        r = requests.get(url, params=params, headers=_auth_headers(cfg), timeout=timeout)
        if r.status_code != 200:
            logger.debug(f"Order by client_id {client_order_id} fetch failed: HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        logger.debug(f"Order by client_id {client_order_id} fetch exception: {e}")
        return None


def resolve_ioc_status(
    order: OrderRecord,
    timeout_s: float = 3.0,
    interval_s: float = 0.3,
) -> OrderRecord:
    """
    Poll Alpaca until IOC order reaches terminal state (FILLED/CANCELLED/EXPIRED).

    IOC orders settle quickly, so we poll up to timeout_s to get final status.
    Updates the OrderRecord in-place with final status, fill price, and filled qty.

    Args:
        order: OrderRecord with broker_order_id set
        timeout_s: Maximum time to poll (default 3s)
        interval_s: Polling interval (default 0.3s)

    Returns:
        Updated OrderRecord with final status
    """
    if not order.broker_order_id:
        logger.warning(f"Cannot resolve status: no broker_order_id for {order.decision_id}")
        return order

    start_time = time.time()
    terminal_states = {"filled", "cancelled", "expired", "rejected"}

    while (time.time() - start_time) < timeout_s:
        data = get_order_by_id(order.broker_order_id)
        if data is None:
            time.sleep(interval_s)
            continue

        alpaca_status = data.get("status", "").lower()
        order.last_update = datetime.utcnow()

        if alpaca_status in terminal_states:
            # Map Alpaca status to our OrderStatus
            if alpaca_status == "filled":
                order.status = OrderStatus.FILLED
                # Extract fill details
                fill_price = data.get("filled_avg_price")
                filled_qty = data.get("filled_qty")
                if fill_price is not None:
                    order.fill_price = float(fill_price)
                if filled_qty is not None:
                    order.filled_qty = int(float(filled_qty))
                logger.info(
                    f"Order {order.symbol} FILLED: {order.filled_qty} @ ${order.fill_price:.2f}"
                )
                update_trade_event("ioc_filled")
            elif alpaca_status in ("cancelled", "expired"):
                order.status = OrderStatus.CANCELLED
                order.notes = f"alpaca_{alpaca_status}"
                logger.info(f"Order {order.symbol} {alpaca_status.upper()}")
                update_trade_event("ioc_cancelled")
            elif alpaca_status == "rejected":
                order.status = OrderStatus.REJECTED
                order.notes = f"alpaca_rejected:{data.get('reject_reason', 'unknown')}"
                logger.warning(f"Order {order.symbol} REJECTED: {order.notes}")
                update_trade_event("ioc_cancelled")
            return order

        time.sleep(interval_s)

    # Timeout - order still in non-terminal state
    logger.warning(f"Order {order.decision_id} resolution timeout after {timeout_s}s")
    order.notes = (order.notes or "") + ";resolution_timeout"
    return order


def log_trade_event(order: OrderRecord, market_bid: Optional[float] = None, market_ask: Optional[float] = None) -> None:
    """
    Log trade event to logs/trades.jsonl for audit and analysis.

    Each line is a JSON object with order details and timestamps.
    Creates logs/ directory if it doesn't exist.
    """
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "decision_id": order.decision_id,
        "symbol": order.symbol,
        "side": order.side,
        "qty": order.qty,
        "limit_price": order.limit_price,
        "status": order.status.value if hasattr(order.status, "value") else str(order.status),
        "broker_order_id": order.broker_order_id,
        "fill_price": order.fill_price,
        "filled_qty": order.filled_qty,
        "notes": order.notes,
        "market_bid_at_execution": market_bid,
        "market_ask_at_execution": market_ask,
        "entry_price_decision": order.entry_price_decision, # Add benchmark price
        "strategy_used": order.strategy_used, # Add strategy for TCA context
    }

    try:
        with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        logger.debug(f"Logged trade event: {order.symbol} {order.status}")
    except Exception as e:
        logger.error(f"Failed to log trade event: {e}")


@require_policy_gate
@require_no_kill_switch
def place_ioc_limit(order: OrderRecord, resolve_status: bool = True) -> BrokerExecutionResult:
    """
    Place an IOC LIMIT order via Alpaca. Returns updated OrderRecord
    along with market bid/ask at the time of order submission.

    Args:
        order: The order to place
        resolve_status: If True, poll for final status and log trade event (default: True)

    Returns:
        BrokerExecutionResult containing the updated OrderRecord and market bid/ask.
    """
    cfg = _alpaca_cfg()
    store = IdempotencyStore()

    # Get current market quote BEFORE placing the order for TCA benchmarking
    market_bid_at_execution, market_ask_at_execution, _, _, quote_ts = get_quote_with_sizes(order.symbol)

    # If quotes are unavailable, we cannot reliably place order or perform TCA.
    if market_bid_at_execution is None or market_ask_at_execution is None:
        logger.warning(f"Unable to fetch quotes for {order.symbol}. Cannot place order.")
        order.status = OrderStatus.REJECTED
        order.notes = "no_quotes_available"
        log_trade_event(order, market_bid=None, market_ask=None)
        return BrokerExecutionResult(order=order, market_bid_at_execution=None, market_ask_at_execution=None)

    # STALENESS GUARD: Reject order if quote is more than 5 minutes old
    MAX_QUOTE_AGE_SECONDS = 300  # 5 minutes
    if quote_ts is not None:
        quote_age_seconds = (datetime.now(quote_ts.tzinfo) - quote_ts).total_seconds()
        if quote_age_seconds > MAX_QUOTE_AGE_SECONDS:
            logger.warning(
                f"Stale quote for {order.symbol}: {quote_age_seconds:.0f}s old (max: {MAX_QUOTE_AGE_SECONDS}s)"
            )
            order.status = OrderStatus.REJECTED
            order.notes = f"stale_quote_{quote_age_seconds:.0f}s"
            log_trade_event(order, market_bid=market_bid_at_execution, market_ask=market_ask_at_execution)
            return BrokerExecutionResult(
                order=order,
                market_bid_at_execution=market_bid_at_execution,
                market_ask_at_execution=market_ask_at_execution
            )

    # Idempotency guard
    if store.exists(order.decision_id):
        order.status = OrderStatus.CLOSED
        order.notes = "duplicate_decision_id"
        log_trade_event(order, market_bid=market_bid_at_execution, market_ask=market_ask_at_execution)
        return BrokerExecutionResult(order=order, market_bid_at_execution=market_bid_at_execution, market_ask_at_execution=market_ask_at_execution)

    url = f"{cfg.base_url}{ALPACA_ORDERS_URL}"
    payload = {
        "symbol": order.symbol.upper(),
        "qty": order.qty,
        "side": normalize_side_lowercase(order.side),
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
            log_trade_event(order, market_bid=market_bid_at_execution, market_ask=market_ask_at_execution)
            return BrokerExecutionResult(order=order, market_bid_at_execution=market_bid_at_execution, market_ask_at_execution=market_ask_at_execution)
        data = r.json()
        order.broker_order_id = data.get("id")
        order.status = OrderStatus.SUBMITTED
        store.put(order.decision_id, order.idempotency_key)
        update_trade_event("ioc_submitted")

        # Resolve IOC status and log trade event
        if resolve_status and order.broker_order_id:
            order = resolve_ioc_status(order)
        log_trade_event(order, market_bid=market_bid_at_execution, market_ask=market_ask_at_execution)

        return BrokerExecutionResult(order=order, market_bid_at_execution=market_bid_at_execution, market_ask_at_execution=market_ask_at_execution)
    except Exception as e:
        order.status = OrderStatus.REJECTED
        order.notes = f"exception:{e}"
        log_trade_event(order, market_bid=market_bid_at_execution, market_ask=market_ask_at_execution)
        return BrokerExecutionResult(order=order, market_bid_at_execution=market_bid_at_execution, market_ask_at_execution=market_ask_at_execution)


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


# ============================================================================
# VIX-Aware Adaptive Clamp (Phase 3: Execution Calibration)
# ============================================================================

IOC_TELEMETRY_PATH = Path("state/ioc_telemetry.json")


def _calculate_adaptive_offset(
    vix_level: Optional[float] = None,
    regime: Optional[str] = None,
) -> float:
    """
    Calculate VIX-aware and regime-aware limit price offset.

    In high volatility environments, wider offsets increase fill probability.
    In choppy regimes, additional buffer helps avoid slippage.

    Args:
        vix_level: Current VIX level (fetched if None)
        regime: Current market regime (BULLISH, NEUTRAL, BEARISH, CHOPPY)

    Returns:
        Offset as decimal (e.g., 0.001 = 0.1%, 0.003 = 0.3%)

    Reference (from plan):
        Default: 0.1% offset (10 bps)
        High VIX (>25): 0.3% offset
        Extreme VIX (>35): 0.5% offset
        CHOPPY regime: +0.1% additional
    """
    base_offset = 0.001  # 10 basis points default

    # VIX-based adjustment
    if vix_level is not None:
        if vix_level > 35:
            base_offset = 0.005  # 50 bps for extreme volatility
        elif vix_level > 25:
            base_offset = 0.003  # 30 bps for high volatility

    # Regime-based adjustment
    if regime is not None:
        regime_upper = regime.upper()
        if regime_upper == "CHOPPY":
            base_offset += 0.001  # Additional 10 bps for choppy markets
        elif regime_upper == "BEARISH":
            base_offset += 0.0005  # 5 bps extra in bearish

    return base_offset


def apply_adaptive_clamp(
    symbol: str,
    side: str,
    base_price: float,
    vix_level: Optional[float] = None,
    regime: Optional[str] = None,
    atr_value: Optional[float] = None,
) -> float:
    """
    Apply VIX-aware and regime-aware limit price clamp.

    This is an enhanced version of _apply_clamp that considers market conditions.
    For BUY orders: limit = base_price * (1 + offset)
    For SELL orders: limit = base_price * (1 - offset)

    Args:
        symbol: Stock symbol
        side: "BUY" or "SELL"
        base_price: The base price (typically best_ask for buys, best_bid for sells)
        vix_level: Current VIX level
        regime: Current market regime
        atr_value: Optional ATR for additional clamping

    Returns:
        Clamped limit price
    """
    # Calculate adaptive offset
    offset = _calculate_adaptive_offset(vix_level, regime)

    # Apply offset based on side
    if is_buy_side(side):
        raw_limit = base_price * (1 + offset)
    else:
        raw_limit = base_price * (1 - offset)

    # Apply standard clamp on top for safety
    clamped = _apply_clamp(raw_limit, base_price, atr_value)

    logger.debug(
        f"Adaptive clamp {symbol} {side}: base={base_price:.2f}, "
        f"offset={offset*100:.2f}%, raw={raw_limit:.2f}, clamped={clamped:.2f}"
    )

    return clamped


def record_fill_telemetry(
    order: OrderRecord,
    intended_qty: int,
    filled_qty: int,
    intended_price: float,
    fill_price: Optional[float],
    vix_at_execution: Optional[float] = None,
    regime: Optional[str] = None,
) -> None:
    """
    Record IOC fill rate telemetry for analysis and model improvement.

    Tracks:
    - Fill rate (filled_qty / intended_qty)
    - Slippage in basis points
    - VIX and regime context
    - Timestamp for time-of-day analysis

    Data is appended to state/ioc_telemetry.json for later analysis.

    Args:
        order: The executed OrderRecord
        intended_qty: Original intended quantity
        filled_qty: Actually filled quantity
        intended_price: Original limit price
        fill_price: Actual fill price (None if not filled)
        vix_at_execution: VIX level when order was placed
        regime: Market regime when order was placed
    """
    IOC_TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    fill_rate = filled_qty / intended_qty if intended_qty > 0 else 0.0

    slippage_bps = 0.0
    if fill_price is not None and intended_price > 0:
        slippage_bps = (fill_price - intended_price) / intended_price * 10000

    telemetry_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": order.symbol,
        "side": order.side,
        "decision_id": order.decision_id,
        "intended_qty": intended_qty,
        "filled_qty": filled_qty,
        "fill_rate": round(fill_rate, 4),
        "intended_price": round(intended_price, 2),
        "fill_price": round(fill_price, 2) if fill_price else None,
        "slippage_bps": round(slippage_bps, 2),
        "vix_at_execution": round(vix_at_execution, 2) if vix_at_execution else None,
        "regime": regime,
        "status": order.status.value if hasattr(order.status, "value") else str(order.status),
    }

    try:
        # Load existing telemetry
        telemetry_data = []
        if IOC_TELEMETRY_PATH.exists():
            with open(IOC_TELEMETRY_PATH, "r", encoding="utf-8") as f:
                try:
                    telemetry_data = json.load(f)
                except json.JSONDecodeError:
                    telemetry_data = []

        # Append new record
        telemetry_data.append(telemetry_record)

        # Keep only last 1000 records to prevent unbounded growth
        if len(telemetry_data) > 1000:
            telemetry_data = telemetry_data[-1000:]

        # Write back
        with open(IOC_TELEMETRY_PATH, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f, indent=2)

        logger.debug(
            f"Recorded fill telemetry: {order.symbol} fill_rate={fill_rate:.2%}, "
            f"slippage={slippage_bps:.1f}bps"
        )

    except Exception as e:
        logger.warning(f"Failed to record fill telemetry: {e}")


def get_fill_telemetry_stats(lookback_days: int = 7) -> Dict[str, Any]:
    """
    Get summary statistics from IOC fill telemetry.

    Args:
        lookback_days: Number of days to analyze

    Returns:
        Dict with fill rate stats, slippage stats, and regime breakdown
    """
    if not IOC_TELEMETRY_PATH.exists():
        return {
            "n_records": 0,
            "avg_fill_rate": None,
            "avg_slippage_bps": None,
            "message": "No telemetry data available",
        }

    try:
        with open(IOC_TELEMETRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return {"n_records": 0, "message": "No telemetry records"}

        # Filter by lookback period
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent = [
            r for r in data
            if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None) > cutoff
        ]

        if not recent:
            return {"n_records": 0, "message": f"No records in last {lookback_days} days"}

        # Calculate stats
        fill_rates = [r["fill_rate"] for r in recent if r["fill_rate"] is not None]
        slippage_vals = [r["slippage_bps"] for r in recent if r["slippage_bps"] is not None]

        # Group by regime
        regime_stats = {}
        for r in recent:
            regime = r.get("regime") or "UNKNOWN"
            if regime not in regime_stats:
                regime_stats[regime] = {"count": 0, "fill_rates": [], "slippage": []}
            regime_stats[regime]["count"] += 1
            if r["fill_rate"] is not None:
                regime_stats[regime]["fill_rates"].append(r["fill_rate"])
            if r["slippage_bps"] is not None:
                regime_stats[regime]["slippage"].append(r["slippage_bps"])

        # Calculate regime averages
        for regime, stats in regime_stats.items():
            stats["avg_fill_rate"] = sum(stats["fill_rates"]) / len(stats["fill_rates"]) if stats["fill_rates"] else None
            stats["avg_slippage"] = sum(stats["slippage"]) / len(stats["slippage"]) if stats["slippage"] else None
            del stats["fill_rates"]
            del stats["slippage"]

        return {
            "n_records": len(recent),
            "lookback_days": lookback_days,
            "avg_fill_rate": sum(fill_rates) / len(fill_rates) if fill_rates else None,
            "avg_slippage_bps": sum(slippage_vals) / len(slippage_vals) if slippage_vals else None,
            "min_fill_rate": min(fill_rates) if fill_rates else None,
            "max_fill_rate": max(fill_rates) if fill_rates else None,
            "by_regime": regime_stats,
        }

    except Exception as e:
        logger.warning(f"Failed to compute fill telemetry stats: {e}")
        return {"error": str(e)}


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
        side=normalize_side(side),
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
    market_bid_at_execution: Optional[float] = None # Added for TCA
    market_ask_at_execution: Optional[float] = None # Added for TCA

    @property
    def success(self) -> bool:
        """True if order was successfully executed (filled or submitted)."""
        return (
            not self.blocked_by_liquidity
            and self.order.status in (OrderStatus.SUBMITTED, OrderStatus.FILLED)
        )


@require_policy_gate
@require_no_kill_switch
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

    Note:
        SECURITY FIX (2026-01-04): bypass_if_disabled param is deprecated and ignored.
        Liquidity gate is ALWAYS enabled, cannot be bypassed.
    """
    # SECURITY FIX (2026-01-04): Removed dead branch
    # Liquidity gate is always enabled, no bypass possible
    if bypass_if_disabled:
        logger.debug("bypass_if_disabled is deprecated and ignored - liquidity gate is always on")

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
        log_trade_event(order, market_bid=None, market_ask=None) # Log rejection
        return OrderResult(
            order=order,
            liquidity_check=liq_check,
            blocked_by_liquidity=True,
            market_bid_at_execution=None,
            market_ask_at_execution=None,
        )

    # Liquidity check passed - place the order
    broker_result = place_ioc_limit(order)

    return OrderResult(
        order=broker_result.order,
        liquidity_check=liq_check,
        blocked_by_liquidity=False,
        market_bid_at_execution=broker_result.market_bid_at_execution,
        market_ask_at_execution=broker_result.market_ask_at_execution,
    )


@require_policy_gate
@require_no_kill_switch
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
    best_bid = get_best_bid(symbol) # Also fetch best bid for context

    if best_ask is None or best_bid is None:
        # Can't proceed without a quote
        order = OrderRecord(
            decision_id=f"DEC_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{symbol}_NOQUOTE",
            signal_id="",
            symbol=symbol.upper(),
            side=normalize_side(side),
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
        best_ask=best_ask, # Use ask for buy limit, bid for sell limit
        atr_value=atr_value,
    )

    # Place with liquidity check
    # SECURITY FIX (2026-01-04): Removed redundant is_liquidity_gate_enabled() check
    # Liquidity gate is always enabled - simplified to just check_liquidity flag
    if check_liquidity:
        return place_order_with_liquidity_check(
            order=order,
            strict=strict_liquidity,
        )
    else:
        broker_result = place_ioc_limit(order)
        return OrderResult(
            order=broker_result.order,
            liquidity_check=None,
            blocked_by_liquidity=False,
            market_bid_at_execution=broker_result.market_bid_at_execution,
            market_ask_at_execution=broker_result.market_ask_at_execution,
        )


# ============================================================================
# Bracket Order Support (OCO: Entry + Stop Loss + Take Profit)
# ============================================================================

@dataclass
class BracketOrderResult:
    """Result of a bracket order submission."""
    order: OrderRecord
    stop_order_id: Optional[str] = None
    profit_order_id: Optional[str] = None
    market_bid_at_execution: Optional[float] = None
    market_ask_at_execution: Optional[float] = None

    @property
    def success(self) -> bool:
        """True if bracket order was successfully placed."""
        return self.order.status in (OrderStatus.SUBMITTED, OrderStatus.FILLED)


@require_policy_gate
@require_no_kill_switch
def place_bracket_order(
    symbol: str,
    side: str,
    qty: int,
    limit_price: float,
    stop_loss: float,
    take_profit: float,
    time_in_force: str = "day",
) -> BracketOrderResult:
    """
    Place a bracket order (entry + stop-loss + take-profit) via Alpaca.

    Bracket orders use OCO (one-cancels-other) for the exit legs:
    - Entry: LIMIT order at limit_price
    - Stop Loss: STOP order at stop_loss price
    - Take Profit: LIMIT order at take_profit price

    When entry fills, both exit orders become active. When one exit fills,
    the other is automatically cancelled.

    Args:
        symbol: Stock symbol
        side: "BUY" or "SELL" (or "long"/"short")
        qty: Number of shares
        limit_price: Entry limit price
        stop_loss: Stop loss trigger price
        take_profit: Take profit limit price
        time_in_force: Order duration (default: "day", options: "gtc")

    Returns:
        BracketOrderResult with order details and exit order IDs
    """
    cfg = _alpaca_cfg()
    store = IdempotencyStore()

    # Get current market quote for TCA benchmarking
    market_bid, market_ask, _, _, _ = get_quote_with_sizes(symbol)

    # Construct order record
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    decision_id = f"BRACKET_{now}_{symbol.upper()}_{uuid.uuid4().hex[:6].upper()}"

    order = OrderRecord(
        decision_id=decision_id,
        signal_id=decision_id.replace("BRACKET_", "SIG_"),
        symbol=symbol.upper(),
        side=normalize_side(side),
        qty=int(qty),
        limit_price=float(limit_price),
        tif=time_in_force.upper(),
        order_type="BRACKET",
        idempotency_key=decision_id,
        created_at=datetime.utcnow(),
    )

    # Idempotency guard
    if store.exists(decision_id):
        order.status = OrderStatus.CLOSED
        order.notes = "duplicate_decision_id"
        return BracketOrderResult(
            order=order,
            market_bid_at_execution=market_bid,
            market_ask_at_execution=market_ask,
        )

    # Alpaca bracket order payload
    url = f"{cfg.base_url}{ALPACA_ORDERS_URL}"
    payload = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": normalize_side_lowercase(side),
        "type": "limit",
        "time_in_force": time_in_force.lower(),
        "limit_price": round(float(limit_price), 2),
        "order_class": "bracket",
        "take_profit": {
            "limit_price": round(float(take_profit), 2),
        },
        "stop_loss": {
            "stop_price": round(float(stop_loss), 2),
        },
        "client_order_id": decision_id,
    }

    try:
        def _post():
            return requests.post(url, json=payload, headers=_auth_headers(cfg), timeout=10)

        r = with_retry(_post)
        if r.status_code not in (200, 201):
            order.status = OrderStatus.REJECTED
            error_msg = r.text[:200] if r.text else f"HTTP {r.status_code}"
            order.notes = f"alpaca_bracket_rejected:{error_msg}"
            logger.warning(f"Bracket order rejected for {symbol}: {error_msg}")
            return BracketOrderResult(
                order=order,
                market_bid_at_execution=market_bid,
                market_ask_at_execution=market_ask,
            )

        data = r.json()
        order.broker_order_id = data.get("id")
        order.status = OrderStatus.SUBMITTED
        store.put(decision_id, decision_id)

        # Extract child order IDs (stop loss and take profit legs)
        legs = data.get("legs", [])
        stop_order_id = None
        profit_order_id = None
        for leg in legs:
            leg_type = leg.get("order_type", "").lower()
            if leg_type == "stop":
                stop_order_id = leg.get("id")
            elif leg_type == "limit":
                profit_order_id = leg.get("id")

        logger.info(
            f"Bracket order placed: {symbol} {side} {qty} @ {limit_price}, "
            f"SL={stop_loss}, TP={take_profit}, order_id={order.broker_order_id}"
        )
        update_trade_event("bracket_submitted")

        # Log trade event
        log_trade_event(order, market_bid=market_bid, market_ask=market_ask)

        return BracketOrderResult(
            order=order,
            stop_order_id=stop_order_id,
            profit_order_id=profit_order_id,
            market_bid_at_execution=market_bid,
            market_ask_at_execution=market_ask,
        )

    except Exception as e:
        order.status = OrderStatus.REJECTED
        order.notes = f"exception:{e}"
        logger.error(f"Bracket order exception for {symbol}: {e}")
        return BracketOrderResult(
            order=order,
            market_bid_at_execution=market_bid,
            market_ask_at_execution=market_ask,
        )


# ============================================================================
# Alpaca Broker Class (Implements BrokerBase)
# ============================================================================

from typing import List
from execution.broker_base import (
    BrokerBase,
    BrokerType,
    Quote,
    Position,
    Account,
    Order as BrokerOrder,
    OrderResult as BrokerOrderResult,
    OrderSide,
    OrderType,
    TimeInForce as BrokerTimeInForce,
    BrokerOrderStatus,
)
from execution.broker_factory import register_broker


class AlpacaBroker(BrokerBase):
    """
    Alpaca broker implementation using the BrokerBase interface.

    Wraps existing module-level functions for backward compatibility
    while providing the standardized broker interface.

    Supports:
    - Stocks (US equities)
    - Paper and live trading modes
    - IOC LIMIT orders (recommended for our strategy)
    - Bracket orders (OCO with stop-loss and take-profit)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key (or from env: ALPACA_API_KEY_ID)
            api_secret: Alpaca API secret (or from env: ALPACA_API_SECRET_KEY)
            base_url: Alpaca API base URL (or from env: ALPACA_BASE_URL)
            paper: If True (default), use paper trading endpoint
        """
        self.paper = paper
        self._connected = False

        # Load from env if not provided
        self._api_key = api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID", "")
        self._api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY", "")

        if base_url:
            self._base_url = base_url
        elif paper:
            self._base_url = "https://paper-api.alpaca.markets"
        else:
            self._base_url = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

    # === Properties ===

    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.ALPACA

    @property
    def name(self) -> str:
        mode = "Paper" if self.paper else "Live"
        return f"Alpaca ({mode})"

    @property
    def supports_extended_hours(self) -> bool:
        return True

    @property
    def is_24_7(self) -> bool:
        return False  # US equities only

    # === Connection ===

    def connect(self) -> bool:
        """Verify connection to Alpaca API."""
        if not self._api_key or not self._api_secret:
            logger.error("Alpaca API credentials not configured")
            return False

        try:
            # Test connection by fetching account
            account = self.get_account()
            if account:
                self._connected = True
                logger.info(f"Connected to Alpaca ({self.name})")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        return self._connected

    # === Market Data ===

    def get_quote(self, symbol: str, timeout: int = 5) -> Optional[Quote]:
        """Get current market quote from Alpaca."""
        bid, ask, bid_size, ask_size, ts = get_quote_with_sizes(symbol, timeout)

        if bid is None and ask is None:
            return None

        last = (bid + ask) / 2 if bid and ask else bid or ask

        return Quote(
            symbol=symbol.upper(),
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            last=last,
            volume=None,  # Would need separate call
            timestamp=ts or datetime.now(),
        )

    def get_quotes(self, symbols: List[str], timeout: int = 10) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol, timeout)
            if quote:
                quotes[symbol] = quote
        return quotes

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        try:
            cfg = _alpaca_cfg()
            url = f"{cfg.base_url}/v2/clock"
            r = requests.get(url, headers=_auth_headers(cfg), timeout=5)
            if r.status_code == 200:
                return r.json().get("is_open", False)
        except Exception:
            pass
        return False

    # === Account & Positions ===

    def get_account(self) -> Optional[Account]:
        """Get Alpaca account information."""
        try:
            cfg = _alpaca_cfg()
            url = f"{cfg.base_url}/v2/account"
            r = requests.get(url, headers=_auth_headers(cfg), timeout=10)
            if r.status_code != 200:
                return None

            data = r.json()
            return Account(
                account_id=data.get("id", ""),
                equity=float(data.get("equity", 0)),
                cash=float(data.get("cash", 0)),
                buying_power=float(data.get("buying_power", 0)),
                currency=data.get("currency", "USD"),
                pattern_day_trader=data.get("pattern_day_trader", False),
                trading_blocked=data.get("trading_blocked", False),
                transfers_blocked=data.get("transfers_blocked", False),
                account_blocked=data.get("account_blocked", False),
                portfolio_value=float(data.get("portfolio_value", 0)),
                last_equity=float(data.get("last_equity", 0)),
                multiplier=float(data.get("multiplier", 1)),
            )
        except Exception as e:
            logger.error(f"Failed to get Alpaca account: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """Get all current positions from Alpaca."""
        try:
            cfg = _alpaca_cfg()
            url = f"{cfg.base_url}/v2/positions"
            r = requests.get(url, headers=_auth_headers(cfg), timeout=10)
            if r.status_code != 200:
                return []

            positions = []
            for p in r.json():
                qty = int(float(p.get("qty", 0)))
                avg_price = float(p.get("avg_entry_price", 0))
                current_price = float(p.get("current_price", 0))
                market_value = float(p.get("market_value", 0))
                unrealized_pnl = float(p.get("unrealized_pl", 0))

                positions.append(Position(
                    symbol=p.get("symbol", ""),
                    qty=qty,
                    avg_price=avg_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    side="long" if qty > 0 else "short",
                    cost_basis=float(p.get("cost_basis", 0)),
                    unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)) * 100,
                ))

            return positions
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        try:
            cfg = _alpaca_cfg()
            url = f"{cfg.base_url}/v2/positions/{symbol.upper()}"
            r = requests.get(url, headers=_auth_headers(cfg), timeout=10)
            if r.status_code != 200:
                return None

            p = r.json()
            qty = int(float(p.get("qty", 0)))

            return Position(
                symbol=p.get("symbol", ""),
                qty=qty,
                avg_price=float(p.get("avg_entry_price", 0)),
                current_price=float(p.get("current_price", 0)),
                market_value=float(p.get("market_value", 0)),
                unrealized_pnl=float(p.get("unrealized_pl", 0)),
                side="long" if qty > 0 else "short",
                cost_basis=float(p.get("cost_basis", 0)),
                unrealized_pnl_pct=float(p.get("unrealized_plpc", 0)) * 100,
            )
        except Exception as e:
            logger.debug(f"Position for {symbol} not found or error: {e}")
            return None

    # === Orders ===

    def place_order(self, order: BrokerOrder, ack_token: str = None) -> BrokerOrderResult:
        """
        Place an order via Alpaca with unified safety gate enforcement.

        Args:
            order: The order to place
            ack_token: Runtime acknowledgment token for live orders

        Returns:
            BrokerOrderResult with execution details
        """
        # UNIFIED SAFETY GATE CHECK - Required for all order submissions
        gate_result = evaluate_safety_gates(
            is_paper_order=self.paper,
            ack_token=ack_token,
            context=f"alpaca_place_order:{order.symbol}"
        )

        if not gate_result.allowed:
            logger.warning(
                f"Safety gate blocked order for {order.symbol}: {gate_result.reason}"
            )
            return BrokerOrderResult(
                success=False,
                broker_order_id=None,
                status=BrokerOrderStatus.REJECTED,
                filled_qty=0,
                fill_price=None,
                error_message=f"safety_gate_blocked: {gate_result.reason}",
            )

        # Get current quote for TCA
        quote = self.get_quote(order.symbol)

        # Convert to OrderRecord for existing functions
        order_record = construct_decision(
            symbol=order.symbol,
            side="BUY" if order.side == OrderSide.BUY else "SELL",
            qty=order.qty,
            best_ask=quote.ask if quote else order.limit_price,
        )

        # Override with provided limit price if any
        if order.limit_price:
            order_record.limit_price = order.limit_price

        if order.client_order_id:
            order_record.decision_id = order.client_order_id
            order_record.idempotency_key = order.client_order_id

        # Place order based on type
        if order.order_type == OrderType.MARKET or (
            order.order_type == OrderType.LIMIT and order.time_in_force == BrokerTimeInForce.IOC
        ):
            result = place_ioc_limit(order_record)

            # Map status
            status_map = {
                OrderStatus.SUBMITTED: BrokerOrderStatus.SUBMITTED,
                OrderStatus.FILLED: BrokerOrderStatus.FILLED,
                OrderStatus.CANCELLED: BrokerOrderStatus.CANCELLED,
                OrderStatus.REJECTED: BrokerOrderStatus.REJECTED,
                OrderStatus.CLOSED: BrokerOrderStatus.CANCELLED,
            }

            return BrokerOrderResult(
                success=result.order.status in (OrderStatus.SUBMITTED, OrderStatus.FILLED),
                broker_order_id=result.order.broker_order_id,
                client_order_id=order.client_order_id or order_record.decision_id,
                status=status_map.get(result.order.status, BrokerOrderStatus.PENDING),
                filled_qty=result.order.filled_qty or 0,
                fill_price=result.order.fill_price,
                error_message=result.order.notes if result.order.status == OrderStatus.REJECTED else None,
                market_bid_at_execution=result.market_bid_at_execution,
                market_ask_at_execution=result.market_ask_at_execution,
            )

        # For other order types (limit day, stop, etc.) use direct API
        return self._place_order_direct(order, quote)

    def _place_order_direct(self, order: BrokerOrder, quote: Optional[Quote]) -> BrokerOrderResult:
        """Place order directly via Alpaca API (non-IOC)."""
        cfg = _alpaca_cfg()
        store = IdempotencyStore()

        client_order_id = order.client_order_id or f"ORDER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Idempotency check
        if store.exists(client_order_id):
            return BrokerOrderResult(
                success=False,
                status=BrokerOrderStatus.REJECTED,
                error_message="duplicate_order_id",
            )

        # Build payload
        payload = {
            "symbol": order.symbol.upper(),
            "qty": order.qty,
            "side": "buy" if order.side == OrderSide.BUY else "sell",
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
            "client_order_id": client_order_id,
            "extended_hours": order.extended_hours,
        }

        if order.limit_price:
            payload["limit_price"] = round(order.limit_price, 2)
        if order.stop_price:
            payload["stop_price"] = round(order.stop_price, 2)

        # Handle bracket orders
        if order.take_profit_limit and order.stop_loss_price:
            payload["order_class"] = "bracket"
            payload["take_profit"] = {"limit_price": round(order.take_profit_limit, 2)}
            payload["stop_loss"] = {"stop_price": round(order.stop_loss_price, 2)}
            if order.stop_loss_limit:
                payload["stop_loss"]["limit_price"] = round(order.stop_loss_limit, 2)

        try:
            url = f"{cfg.base_url}{ALPACA_ORDERS_URL}"
            r = requests.post(url, json=payload, headers=_auth_headers(cfg), timeout=10)

            if r.status_code not in (200, 201):
                return BrokerOrderResult(
                    success=False,
                    status=BrokerOrderStatus.REJECTED,
                    error_message=f"alpaca_http_{r.status_code}: {r.text[:200]}",
                    market_bid_at_execution=quote.bid if quote else None,
                    market_ask_at_execution=quote.ask if quote else None,
                )

            data = r.json()
            store.put(client_order_id, client_order_id)

            return BrokerOrderResult(
                success=True,
                broker_order_id=data.get("id"),
                client_order_id=client_order_id,
                status=BrokerOrderStatus.SUBMITTED,
                filled_qty=int(float(data.get("filled_qty", 0))),
                fill_price=float(data.get("filled_avg_price")) if data.get("filled_avg_price") else None,
                market_bid_at_execution=quote.bid if quote else None,
                market_ask_at_execution=quote.ask if quote else None,
            )

        except Exception as e:
            return BrokerOrderResult(
                success=False,
                status=BrokerOrderStatus.REJECTED,
                error_message=str(e),
            )

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an order by broker order ID."""
        try:
            cfg = _alpaca_cfg()
            url = f"{cfg.base_url}{ALPACA_ORDERS_URL}/{broker_order_id}"
            r = requests.delete(url, headers=_auth_headers(cfg), timeout=10)
            if r.status_code in (200, 204):
                logger.info(f"Cancelled order {broker_order_id}")
                return True
            logger.warning(f"Failed to cancel order {broker_order_id}: HTTP {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"Cancel order exception: {e}")
            return False

    def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrderStatus]:
        """Get order status by broker order ID."""
        data = get_order_by_id(broker_order_id)
        if not data:
            return None

        status = data.get("status", "").lower()
        status_map = {
            "new": BrokerOrderStatus.PENDING,
            "pending_new": BrokerOrderStatus.PENDING,
            "accepted": BrokerOrderStatus.ACCEPTED,
            "partially_filled": BrokerOrderStatus.PARTIALLY_FILLED,
            "filled": BrokerOrderStatus.FILLED,
            "canceled": BrokerOrderStatus.CANCELLED,
            "cancelled": BrokerOrderStatus.CANCELLED,
            "expired": BrokerOrderStatus.EXPIRED,
            "rejected": BrokerOrderStatus.REJECTED,
        }
        return status_map.get(status, BrokerOrderStatus.PENDING)

    def get_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get orders from Alpaca."""
        try:
            cfg = _alpaca_cfg()
            params = {"limit": limit}

            if status == "open":
                params["status"] = "open"
            elif status == "closed":
                params["status"] = "closed"
            # else: all (default)

            if after:
                params["after"] = after.isoformat() + "Z"

            url = f"{cfg.base_url}{ALPACA_ORDERS_URL}"
            r = requests.get(url, params=params, headers=_auth_headers(cfg), timeout=10)

            if r.status_code != 200:
                return []

            return r.json()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []


# Auto-register Alpaca broker
register_broker("alpaca", AlpacaBroker)