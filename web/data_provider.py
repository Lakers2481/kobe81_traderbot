"""
Dashboard Data Provider for Kobe Trading System.

Provides real-time data for the web dashboard including:
- Account and position data from Alpaca
- Performance metrics calculation
- Signal loading from scanner results
- Market context (VIX, indices)
- Kill switch status monitoring
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Live performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    current_equity: float = 100000.0
    initial_capital: float = 100000.0
    cumulative_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    open_positions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "current_equity": round(self.current_equity, 2),
            "initial_capital": self.initial_capital,
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "cumulative_pnl_pct": round((self.cumulative_pnl / self.initial_capital) * 100, 2) if self.initial_capital > 0 else 0,
            "daily_pnl": round(self.daily_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "current_drawdown": round(self.current_drawdown * 100, 2),
            "open_positions": self.open_positions,
        }


@dataclass
class KillSwitchStatus:
    """Kill switch monitoring status."""
    kill_switch_active: bool = False
    daily_loss_pct: float = 0.0
    daily_loss_triggered: bool = False
    position_count: int = 0
    position_limit_triggered: bool = False
    drawdown_pct: float = 0.0
    drawdown_triggered: bool = False
    alpaca_connected: bool = True
    polygon_connected: bool = True
    all_clear: bool = True
    triggered_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kill_switch_active": self.kill_switch_active,
            "layers": [
                {
                    "name": "Kill Switch File",
                    "current": "ACTIVE" if self.kill_switch_active else "OFF",
                    "limit": "OFF",
                    "triggered": self.kill_switch_active,
                },
                {
                    "name": "Daily Loss",
                    "current": f"{self.daily_loss_pct:.2f}%",
                    "limit": "-3.0%",
                    "triggered": self.daily_loss_triggered,
                },
                {
                    "name": "Drawdown",
                    "current": f"{self.drawdown_pct:.2f}%",
                    "limit": "-15.0%",
                    "triggered": self.drawdown_triggered,
                },
                {
                    "name": "Position Limit",
                    "current": str(self.position_count),
                    "limit": "5",
                    "triggered": self.position_limit_triggered,
                },
                {
                    "name": "Alpaca API",
                    "current": "OK" if self.alpaca_connected else "ERROR",
                    "limit": "Connected",
                    "triggered": not self.alpaca_connected,
                },
                {
                    "name": "Polygon API",
                    "current": "OK" if self.polygon_connected else "ERROR",
                    "limit": "Connected",
                    "triggered": not self.polygon_connected,
                },
            ],
            "all_clear": self.all_clear,
            "triggered_reasons": self.triggered_reasons,
        }


@dataclass
class MarketContext:
    """Market context data."""
    vix: float = 18.0  # Safe default VIX value (historical average)
    vix_regime: str = "NORMAL"
    spy_price: float = 0.0
    qqq_price: float = 0.0
    iwm_price: float = 0.0
    market_open: bool = False
    market_status: str = "CLOSED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vix": {
                "current": round(self.vix, 2),
                "regime": self.vix_regime,
            },
            "indices": {
                "SPY": round(self.spy_price, 2),
                "QQQ": round(self.qqq_price, 2),
                "IWM": round(self.iwm_price, 2),
            },
            "market": {
                "is_open": self.market_open,
                "status": self.market_status,
            }
        }


@dataclass
class SignalData:
    """Trading signal data."""
    symbol: str
    direction: str = "long"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    rsi_2: float = 0.0
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        risk = abs(self.entry_price - self.stop_loss) if self.entry_price > 0 else 1
        reward = abs(self.take_profit - self.entry_price) if self.entry_price > 0 else 1
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "confidence": round(self.confidence * 100, 1),
            "reason": self.reason,
            "rsi_2": round(self.rsi_2, 2),
            "rr_ratio": round(rr_ratio, 2),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# =============================================================================
# DATA PROVIDER
# =============================================================================

class DashboardDataProvider:
    """
    Data provider for Kobe dashboard.

    Fetches and calculates all data needed for real-time dashboard display.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize data provider."""
        self.project_root = project_root or Path(__file__).parent.parent
        self._price_cache: Dict[str, tuple] = {}
        self._last_cache_clear = time.time()
        self._broker = None
        self._init_broker()

    def _init_broker(self):
        """Initialize broker connection."""
        try:
            from execution.broker_alpaca import AlpacaBroker
            self._broker = AlpacaBroker()
            logger.info("Alpaca broker initialized for dashboard")
        except Exception as e:
            logger.warning(f"Could not initialize broker: {e}")

    def get_live_price(self, symbol: str, cache_seconds: int = 30) -> float:
        """Get live price with caching."""
        import requests

        now = time.time()

        # Clear cache periodically
        if now - self._last_cache_clear > 300:
            self._price_cache.clear()
            self._last_cache_clear = now

        # Check cache
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if now - ts < cache_seconds:
                return price

        api_key = os.getenv("POLYGON_API_KEY", "")
        if not api_key:
            return self._get_fallback_price(symbol)

        # Handle VIX separately
        if symbol.upper() == "VIX":
            return self._get_vix_price(api_key)

        try:
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={api_key}"
            resp = requests.get(url, timeout=5)

            if resp.status_code == 200:
                data = resp.json()
                ticker = data.get('ticker', {})

                price = 0
                if ticker.get('day', {}).get('c'):
                    price = float(ticker['day']['c'])
                elif ticker.get('lastTrade', {}).get('p'):
                    price = float(ticker['lastTrade']['p'])
                elif ticker.get('prevDay', {}).get('c'):
                    price = float(ticker['prevDay']['c'])

                if price > 0:
                    self._price_cache[symbol] = (price, now)
                    return price
        except Exception as e:
            logger.debug(f"Price fetch error for {symbol}: {e}")

        # Return cached if available
        if symbol in self._price_cache:
            return self._price_cache[symbol][0]
        return self._get_fallback_price(symbol)

    def _get_vix_price(self, api_key: str) -> float:
        """Get VIX price from Polygon."""
        import requests

        now = time.time()

        if "VIX" in self._price_cache:
            price, ts = self._price_cache["VIX"]
            if now - ts < 60:
                return price

        vix_price = 18.0  # Default

        try:
            # Try indices endpoint
            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/indices/tickers/I:VIX?apiKey={api_key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                session = data.get('session', {})
                vix_price = float(session.get('close', 0) or session.get('previous_close', 0) or 18.0)
        except Exception:
            pass

        if vix_price < 5 or vix_price > 100:
            vix_price = 18.0

        self._price_cache["VIX"] = (vix_price, now)
        return vix_price

    def _get_fallback_price(self, symbol: str) -> float:
        """Get fallback prices for common symbols."""
        fallbacks = {
            "SPY": 585.0,
            "QQQ": 520.0,
            "IWM": 230.0,
            "VIX": 18.0,
        }
        return fallbacks.get(symbol.upper(), 0)

    def get_account_data(self) -> Dict[str, Any]:
        """Get account data from Alpaca."""
        default = {
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 100000.0,
            "unrealized_pnl": 0.0,
        }

        if not self._broker:
            return default

        try:
            account = self._broker.get_account()
            return {
                "equity": float(getattr(account, 'equity', 100000)),
                "cash": float(getattr(account, 'cash', 100000)),
                "buying_power": float(getattr(account, 'buying_power', 100000)),
                "unrealized_pnl": float(getattr(account, 'unrealized_pl', 0) or 0),
            }
        except Exception as e:
            logger.debug(f"Account fetch error: {e}")
            return default

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from Alpaca."""
        if not self._broker:
            return []

        try:
            positions = self._broker.get_positions()
            result = []
            for pos in positions:
                result.append({
                    "symbol": getattr(pos, 'symbol', ''),
                    "qty": int(getattr(pos, 'qty', 0)),
                    "side": "long" if int(getattr(pos, 'qty', 0)) > 0 else "short",
                    "entry_price": float(getattr(pos, 'avg_entry_price', 0)),
                    "current_price": float(getattr(pos, 'current_price', 0)),
                    "unrealized_pnl": float(getattr(pos, 'unrealized_pl', 0) or 0),
                    "unrealized_pnl_pct": float(getattr(pos, 'unrealized_plpc', 0) or 0) * 100,
                    "market_value": float(getattr(pos, 'market_value', 0)),
                })
            return result
        except Exception as e:
            logger.debug(f"Positions fetch error: {e}")
            return []

    def check_alpaca_connection(self) -> bool:
        """Check if Alpaca is connected."""
        if not self._broker:
            return False
        try:
            self._broker.get_account()
            return True
        except Exception:
            return False

    def check_polygon_connection(self) -> bool:
        """Check if Polygon is connected."""
        try:
            api_key = os.getenv("POLYGON_API_KEY", "")
            if not api_key:
                return False
            import requests
            resp = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey={api_key}",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def check_kill_switch(self) -> bool:
        """Check if kill switch file exists."""
        kill_file = self.project_root / "state" / "KILL_SWITCH"
        return kill_file.exists()

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        account = self.get_account_data()
        positions = self.get_positions()

        metrics = PerformanceMetrics()
        metrics.timestamp = datetime.now()
        metrics.current_equity = account["equity"]
        metrics.initial_capital = 100000.0
        metrics.cumulative_pnl = metrics.current_equity - metrics.initial_capital
        metrics.unrealized_pnl = account["unrealized_pnl"]
        metrics.open_positions = len(positions)

        # Load trade history for win rate calculation
        trade_file = self.project_root / "state" / "trade_history.json"
        if trade_file.exists():
            try:
                with open(trade_file) as f:
                    trades = json.load(f)

                if isinstance(trades, list) and trades:
                    metrics.total_trades = len(trades)
                    metrics.winning_trades = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
                    metrics.losing_trades = metrics.total_trades - metrics.winning_trades

                    if metrics.total_trades > 0:
                        metrics.win_rate = metrics.winning_trades / metrics.total_trades

                    total_wins = sum(float(t.get("pnl", 0)) for t in trades if float(t.get("pnl", 0)) > 0)
                    total_losses = abs(sum(float(t.get("pnl", 0)) for t in trades if float(t.get("pnl", 0)) < 0))
                    if total_losses > 0:
                        metrics.profit_factor = total_wins / total_losses

            except Exception as e:
                logger.debug(f"Trade history load error: {e}")

        # Calculate drawdown
        if metrics.current_equity < metrics.initial_capital:
            metrics.current_drawdown = (metrics.current_equity - metrics.initial_capital) / metrics.initial_capital

        return metrics

    def get_kill_switch_status(self, metrics: PerformanceMetrics) -> KillSwitchStatus:
        """Get kill switch monitoring status."""
        status = KillSwitchStatus()

        # Check kill switch file
        status.kill_switch_active = self.check_kill_switch()
        if status.kill_switch_active:
            status.triggered_reasons.append("KILL_SWITCH_FILE")

        # Daily loss check
        status.daily_loss_pct = (metrics.cumulative_pnl / metrics.initial_capital) * 100 if metrics.initial_capital > 0 else 0
        status.daily_loss_triggered = status.daily_loss_pct <= -3.0
        if status.daily_loss_triggered:
            status.triggered_reasons.append("DAILY_LOSS")

        # Position limit
        status.position_count = metrics.open_positions
        status.position_limit_triggered = metrics.open_positions >= 5
        if status.position_limit_triggered:
            status.triggered_reasons.append("POSITION_LIMIT")

        # Drawdown
        status.drawdown_pct = metrics.current_drawdown * 100
        status.drawdown_triggered = metrics.current_drawdown <= -0.15
        if status.drawdown_triggered:
            status.triggered_reasons.append("DRAWDOWN")

        # API health
        status.alpaca_connected = self.check_alpaca_connection()
        status.polygon_connected = self.check_polygon_connection()
        if not status.alpaca_connected:
            status.triggered_reasons.append("ALPACA_DISCONNECTED")
        if not status.polygon_connected:
            status.triggered_reasons.append("POLYGON_DISCONNECTED")

        # Overall status
        status.all_clear = len(status.triggered_reasons) == 0

        return status

    def get_market_context(self) -> MarketContext:
        """Get current market context."""
        context = MarketContext()

        # VIX
        context.vix = self.get_live_price("VIX")
        if context.vix < 15:
            context.vix_regime = "LOW"
        elif context.vix < 25:
            context.vix_regime = "NORMAL"
        else:
            context.vix_regime = "HIGH"

        # Indices
        context.spy_price = self.get_live_price("SPY")
        context.qqq_price = self.get_live_price("QQQ")
        context.iwm_price = self.get_live_price("IWM")

        # Market status
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        if weekday >= 5:
            context.market_status = "CLOSED"
            context.market_open = False
        elif hour < 4:
            context.market_status = "CLOSED"
            context.market_open = False
        elif hour < 9 or (hour == 9 and minute < 30):
            context.market_status = "PRE-MARKET"
            context.market_open = False
        elif hour < 16:
            context.market_status = "OPEN"
            context.market_open = True
        elif hour < 20:
            context.market_status = "AFTER-HOURS"
            context.market_open = False
        else:
            context.market_status = "CLOSED"
            context.market_open = False

        return context

    def load_signals(self, limit: int = 10) -> List[SignalData]:
        """Load recent signals from scanner results."""
        signals = []

        # Try loading from state/signals.json
        signals_file = self.project_root / "state" / "signals.json"
        if signals_file.exists():
            try:
                with open(signals_file) as f:
                    data = json.load(f)

                for sig in data[:limit]:
                    signals.append(SignalData(
                        symbol=sig.get("symbol", ""),
                        direction=sig.get("direction", sig.get("side", "long")),
                        entry_price=float(sig.get("entry_price", 0)),
                        stop_loss=float(sig.get("stop_loss", 0)),
                        take_profit=float(sig.get("take_profit", 0)),
                        confidence=float(sig.get("confidence", 0.5)),
                        reason=sig.get("reason", ""),
                        rsi_2=float(sig.get("rsi_2", 50)),
                        timestamp=datetime.fromisoformat(sig["timestamp"]) if sig.get("timestamp") else None,
                    ))
            except Exception as e:
                logger.debug(f"Signals load error: {e}")

        return signals

    def get_full_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data bundle."""
        metrics = self.get_performance_metrics()
        kill_switches = self.get_kill_switch_status(metrics)
        market = self.get_market_context()
        signals = self.load_signals()
        positions = self.get_positions()

        return {
            "generated_at": datetime.now().isoformat(),
            "system_name": "Kobe Trading System",
            "performance": metrics.to_dict(),
            "kill_switches": kill_switches.to_dict(),
            "market": market.to_dict(),
            "signals": [s.to_dict() for s in signals],
            "positions": positions,
            "system_healthy": kill_switches.all_clear,
        }


# Singleton instance
_data_provider: Optional[DashboardDataProvider] = None


def get_data_provider() -> DashboardDataProvider:
    """Get or create the global data provider instance."""
    global _data_provider
    if _data_provider is None:
        _data_provider = DashboardDataProvider()
    return _data_provider
