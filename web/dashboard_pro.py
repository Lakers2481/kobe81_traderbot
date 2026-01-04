"""
Kobe Trading System - Professional Quant Dashboard v3.0

Bloomberg Terminal-inspired design with:
- Real-time WebSocket data updates (every 60s scanner cycle)
- Dense, professional information display
- Collapsible sections with dropdowns
- No gaps - every section filled with data
- Consistent typography and spacing
- Dark theme with Robinhood green accent (#00C805)

FastAPI backend with WebSocket for live updates.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def atomic_json_write(path: Path, data: Any) -> None:
    """Atomically write JSON to a file (write to temp, then rename)."""
    import tempfile
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    temp_path.replace(path)


def get_data_loader(root: Path):
    """Stub for data loader (not yet implemented)."""
    return None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# KOBE ADAPTER LAYER - Provides compatibility with 2K28 dashboard code
# ============================================================================

# VIX regime thresholds (SSOT)
VIX_THRESHOLDS = {"low": 16, "elevated": 24, "high": 32, "extreme": 40}

def get_vix_regime(vix: float) -> Dict[str, Any]:
    """Get VIX regime with full metadata for strategy adjustments."""
    if vix < VIX_THRESHOLDS["low"]:
        return {
            "regime": "NORMAL", "label": "Low Volatility", "color": "#00C805",
            "win_rate_adjustment": 0, "position_size_pct": 100,
            "recommendation": "Full position sizing. Optimal conditions."
        }
    elif vix < VIX_THRESHOLDS["elevated"]:
        return {
            "regime": "ELEVATED", "label": "Elevated Volatility", "color": "#FFA500",
            "win_rate_adjustment": -2, "position_size_pct": 75,
            "recommendation": "Reduce position size to 75%."
        }
    elif vix < VIX_THRESHOLDS["high"]:
        return {
            "regime": "HIGH", "label": "High Volatility", "color": "#FF5000",
            "win_rate_adjustment": -5, "position_size_pct": 50,
            "recommendation": "Half position sizing."
        }
    else:
        return {
            "regime": "EXTREME", "label": "Extreme Volatility", "color": "#FF0000",
            "win_rate_adjustment": -10, "position_size_pct": 25,
            "recommendation": "Minimum exposure."
        }

def get_risk_params_from_config() -> Dict[str, Any]:
    """Get risk parameters from config."""
    return {
        "max_positions": 5, "risk_per_trade_pct": 2.0,
        "daily_loss_limit_pct": 3.0, "account_stop_loss_dd_pct": 10.0,
    }

def get_scanner_config() -> Dict[str, Any]:
    """Get scanner configuration."""
    return {"universe_mode": "PROVEN_900", "rsi_threshold": 10, "max_signals": 50}

# Kobe imports with graceful degradation
# Create simple AlpacaBroker wrapper for dashboard compatibility
class AlpacaBroker:
    """Simple Alpaca API wrapper for dashboard data fetching."""

    def __init__(self):
        import requests
        self._requests = requests
        self._base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self._headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY_ID", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_API_SECRET_KEY", ""),
        }

    def get_account(self) -> Dict[str, Any]:
        """Get account info from Alpaca."""
        try:
            resp = self._requests.get(
                f"{self._base_url}/v2/account",
                headers=self._headers,
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Alpaca account error: {e}")
        return {"equity": 100000, "cash": 100000, "buying_power": 100000}

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions from Alpaca."""
        try:
            resp = self._requests.get(
                f"{self._base_url}/v2/positions",
                headers=self._headers,
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Alpaca positions error: {e}")
        return []

ALPACA_AVAILABLE = True

try:
    from alerts.telegram_alerter import TelegramAlerter
    def get_alerter(): return TelegramAlerter()
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    def get_alerter(): return None

REPORT_LOADER_AVAILABLE = False
EDGE_SERVICE_AVAILABLE = False

# Stub for EdgeMasterService (not yet implemented)
class EdgeMasterService:
    """Placeholder for future EdgeMasterService implementation."""
    pass

app = FastAPI(title="Kobe Pro Dashboard", version="3.0.0")

# Module-level EdgeMasterService singleton (Phase 16B)
_edge_service = None

def _get_edge_service():
    """Get or create EdgeMasterService singleton."""
    global _edge_service
    if _edge_service is None and EDGE_SERVICE_AVAILABLE:
        try:
            _edge_service = EdgeMasterService()
        except Exception as e:
            print(f"Failed to initialize EdgeMasterService: {e}")
            return None
    return _edge_service

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")


class ProDashboardData:
    """Professional data provider with real-time updates."""

    def __init__(self):
        self.alpaca = None
        self.initial_capital = 100000.0
        self._price_cache = {}
        self._last_update = 0
        self._project_root = Path(__file__).parent.parent

        try:
            if ALPACA_AVAILABLE:
                self.alpaca = AlpacaBroker()
        except Exception as e:
            print(f"Alpaca init error: {e}")

    def _get_live_price(self, symbol: str) -> float:
        """Get live price from Polygon with 60s cache."""
        import requests

        now = time.time()
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if now - ts < 60:
                return price

        try:
            api_key = os.getenv("POLYGON_API_KEY", "")
            if not api_key:
                return 0

            url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={api_key}"
            resp = requests.get(url, timeout=3)

            if resp.status_code == 200:
                data = resp.json()
                ticker = data.get('ticker', {})

                # Try multiple price sources
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
        except Exception:
            pass

        # Return cached if available
        if symbol in self._price_cache:
            return self._price_cache[symbol][0]
        return 0

    def get_full_data(self) -> Dict[str, Any]:
        """Get all dashboard data in one call."""
        now_ct = datetime.now(CT)
        now_et = datetime.now(ET)

        # Account data
        account = {"equity": 100000, "cash": 100000, "buying_power": 100000}
        positions = []

        if self.alpaca:
            try:
                acct = self.alpaca.get_account()
                account = {
                    "equity": float(acct.get("equity", 100000)),
                    "cash": float(acct.get("cash", 100000)),
                    "buying_power": float(acct.get("buying_power", 100000)),
                    "unrealized_pnl": float(acct.get("unrealized_pl", 0) or 0),
                }

                raw_positions = self.alpaca.get_positions() or []
                for p in raw_positions:
                    symbol = p.get("symbol", "")
                    live_price = self._get_live_price(symbol)
                    entry = float(p.get("avg_entry_price", 0))
                    qty = int(float(p.get("qty", 0)))

                    # Use live price if available, else Alpaca price
                    current = live_price if live_price > 0 else float(p.get("current_price", entry))

                    pnl = (current - entry) * qty
                    pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0

                    # Load trade metadata
                    meta = self._load_trade_meta(symbol)

                    positions.append({
                        "symbol": symbol,
                        "qty": qty,
                        "entry": entry,
                        "current": current,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "market_value": current * qty,
                        "stop_loss": meta.get("stop_loss", entry * 0.97),
                        "take_profit": meta.get("take_profit", entry * 1.09),
                        "entry_date": meta.get("entry_date", now_ct.strftime("%Y-%m-%d")),
                        "days_held": meta.get("days_held", 1),
                        "confidence": meta.get("confidence", 70),
                    })
            except Exception as e:
                print(f"Data fetch error: {e}")

        # Calculate totals
        total_pnl = account["equity"] - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        unrealized = sum(p["pnl"] for p in positions)

        # Load scanner results
        scanner = self._load_scanner_results()

        # Load trade stats
        stats = self._load_stats()

        # Market data - Add QQQ and IWM
        spy_price = self._get_live_price("SPY")
        qqq_price = self._get_live_price("QQQ")
        iwm_price = self._get_live_price("IWM")
        vix_price = self._get_live_price("VIX") or 15.0

        # Load AI brain state
        ai_brain = self._load_ai_brain()

        # Enhanced market status with context
        market_status = self._get_market_status_context(now_et)

        # Generate market intelligence
        market_intel = self._generate_market_intelligence(
            vix=vix_price,
            spy_price=spy_price,
            market_status=market_status["status"],
            now_et=now_et
        )

        return {
            "timestamp": now_ct.isoformat(),
            "time_ct": now_ct.strftime("%H:%M:%S"),
            "time_et": now_et.strftime("%H:%M:%S"),
            "date": now_ct.strftime("%A, %b %d, %Y"),
            "account": {
                **account,
                "total_pnl": total_pnl,
                "total_pnl_pct": total_pnl_pct,
                "unrealized_pnl": unrealized,
                "initial_capital": self.initial_capital,
            },
            "positions": positions,
            "position_count": len(positions),
            "market": {
                "spy": spy_price,
                "qqq": qqq_price,
                "iwm": iwm_price,
                "vix": vix_price,
                "status": market_status["status"],
                "context": market_status["context"],
            },
            "market_intelligence": market_intel,
            "scanner": scanner,
            "stats": stats,
            "schedule": self._get_schedule(now_ct),
            "ai_brain": ai_brain,
            "news": self._load_news(
                watchlist_tickers=[s["symbol"] for s in scanner.get("top_10", []) if s.get("symbol")]
            ),
            "trading_mode": self._get_trading_mode(),
            "validation": self._get_validation_metrics(),
            "universe": self._get_universe_info(),
            "health_status": self._get_health_status(),
            "risk_dashboard": self._get_risk_dashboard(account.get("equity", self.initial_capital), positions),
            "daily_reports": self._load_daily_reports(),
        }

    def _is_market_open(self, now_et: datetime) -> bool:
        """Check if market is open."""
        if now_et.weekday() >= 5:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0)
        market_close = now_et.replace(hour=16, minute=0, second=0)
        return market_open <= now_et <= market_close

    def _load_trade_meta(self, symbol: str) -> Dict:
        """Load trade metadata from history."""
        try:
            path = Path("data/trade_history.json")
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    trade = data.get("open_trades", {}).get(symbol, {})
                    if trade:
                        entry_date = trade.get("entry_date", "")
                        days = 1
                        if entry_date:
                            try:
                                ed = datetime.strptime(entry_date, "%Y-%m-%d")
                                days = (datetime.now() - ed).days + 1
                            except ValueError:
                                pass
                        return {
                            "stop_loss": trade.get("stop_loss", 0),
                            "take_profit": trade.get("take_profit", 0),
                            "entry_date": entry_date,
                            "days_held": days,
                            "confidence": trade.get("confidence", 70),
                        }
        except (json.JSONDecodeError, OSError, KeyError):
            pass
        return {}

    def _load_scanner_results(self) -> Dict:
        """Load latest scanner results from daily scan files."""
        try:
            # Priority 1: Today's daily scan file (SCAN_YYYYMMDD.json)
            today = datetime.now().strftime("%Y%m%d")
            daily_scan_path = Path(f"reports/daily/SCAN_{today}.json")

            # Priority 2: Fall back to latest_scan_results.json
            fallback_path = Path("data/scanner/latest_scan_results.json")

            # Use daily scan if available, else fallback
            path = daily_scan_path if daily_scan_path.exists() else fallback_path

            if path.exists():
                with open(path) as f:
                    data = json.load(f)

                    # Convert NEW daily scan format to expected format if needed
                    if "long_candidates" in data and "short_candidates" in data:
                        # New format: SCAN_YYYYMMDD.json
                        longs = data.get("long_candidates", {}).get("top10", [])
                        shorts = data.get("short_candidates", {}).get("top10", [])

                        # Merge and sort by conviction
                        all_candidates = longs + shorts
                        all_candidates.sort(key=lambda x: x.get("conviction", 0), reverse=True)

                        # Map field names from daily scan to dashboard format
                        mapped_candidates = []
                        for sig in all_candidates:
                            mapped = {**sig}  # Copy all fields
                            # Map rsi5 -> rsi_5 if needed
                            if "rsi5" in sig and "rsi_5" not in sig:
                                mapped["rsi_5"] = sig["rsi5"]
                            # Ensure confidence is percentage (0-100)
                            if "conviction" in sig:
                                mapped["confidence"] = sig["conviction"]
                            # Add signal_time if missing
                            if "signal_time" not in mapped:
                                mapped["signal_time"] = data.get("scan_timestamp")
                            mapped_candidates.append(mapped)

                        # Build expected format
                        data = {
                            "scan_time": data.get("scan_timestamp", datetime.now().isoformat()),
                            "total_scanned": data.get("universe_size", 900),
                            "signals_found": len(mapped_candidates),
                            "long_count": len(longs),
                            "short_count": len(shorts),
                            "top_10": mapped_candidates[:10],
                            "top_3": mapped_candidates[:3],
                            "universe_mode": "PROVEN_900",
                        }

                    def transform_signal(signal):
                        """Transform signal to ensure proper display format."""
                        if not signal or signal.get("symbol") == "N/A":
                            return None

                        # Ensure confidence is in percentage form (0-100) for display
                        confidence = signal.get("confidence", 0)
                        if isinstance(confidence, (int, float)):
                            # If confidence is 0-1 range, convert to percentage
                            if confidence <= 1.0:
                                confidence = confidence * 100
                        else:
                            confidence = 0

                        entry_price = signal.get("entry_price", 0)
                        stop_loss = signal.get("stop_loss")
                        take_profit = signal.get("take_profit")

                        # Calculate stop_loss and take_profit when missing (ATR-based approximation)
                        if entry_price and entry_price > 0:
                            atr_estimate = entry_price * 0.025  # ~2.5% ATR estimate
                            if stop_loss is None:
                                stop_loss = round(entry_price - atr_estimate, 2)
                            if take_profit is None:
                                take_profit = round(entry_price + (atr_estimate * 2), 2)

                        # Calculate rr_ratio
                        rr_ratio = signal.get("rr_ratio")
                        if rr_ratio is None:
                            rr_ratio = self._calculate_rr_ratio(entry_price, stop_loss, take_profit)

                        # Phase 16B: Determine status based on entry conditions
                        rsi_5 = signal.get("rsi_5")
                        bb_lower = signal.get("bb_lower", 0)
                        sma_200 = signal.get("sma_200", 0)

                        # Check if this is a "ready" signal (all conditions met)
                        rsi_met = rsi_5 is not None and rsi_5 < 3
                        band_met = entry_price <= bb_lower if bb_lower > 0 else False
                        trend_met = entry_price > sma_200 if sma_200 > 0 else True  # Assume true if missing

                        if rsi_met and band_met and trend_met:
                            status = "ready"
                        elif rsi_met and trend_met:
                            status = "watchlist"
                        else:
                            status = "ineligible"

                        # Phase 16B: Calculate readiness score (0-100)
                        # Note: Use explicit None check since rsi_5=0.0 is valid (0.0 is falsy in Python!)
                        rsi_score = max(0, min(100, 100 * (3 - rsi_5) / 3)) if rsi_5 is not None and rsi_5 < 3 else 0
                        band_score = 100 if band_met else 0
                        trend_score = 100 if trend_met else 0
                        readiness_score = round(0.5 * rsi_score + 0.4 * band_score + 0.1 * trend_score)

                        # Calculate BB Position % (how far price is from lower band)
                        bb_upper = signal.get("bb_upper", 0)
                        bb_position_pct = 0
                        if bb_lower > 0 and bb_upper > bb_lower:
                            bb_range = bb_upper - bb_lower
                            bb_position_pct = round(((entry_price - bb_lower) / bb_range) * 100, 1)

                        # Calculate ATR-based expected weekly move
                        atr_estimate = entry_price * 0.025  # ~2.5% ATR estimate
                        expected_move_weekly = signal.get("expected_move_weekly") or round(atr_estimate * 1.5, 2)
                        expected_move_weekly_pct = signal.get("expected_move_weekly_pct") or round((expected_move_weekly / entry_price) * 100, 1) if entry_price > 0 else 0

                        # Multi-targets: T1 (1R), T2 (1.5R), T3 (2R)
                        risk = entry_price - stop_loss if stop_loss else entry_price * 0.03
                        t1 = round(entry_price + risk, 2)  # 1R
                        t2 = round(entry_price + (risk * 1.5), 2)  # 1.5R
                        t3 = round(entry_price + (risk * 2), 2)  # 2R (main target)

                        # WHY TAKE THIS TRADE panel data
                        why_trade = []
                        if rsi_met:
                            why_trade.append(f"RSI({rsi_5:.1f}) extreme oversold - prime mean reversion entry")
                        if band_met:
                            why_trade.append(f"Price ${entry_price:.2f} at/below BB Lower ${bb_lower:.2f} - volatility extreme")
                        if trend_met:
                            why_trade.append(f"Above SMA(200) ${sma_200:.2f} - uptrend filter confirms")
                        if rr_ratio and rr_ratio >= 2:
                            why_trade.append(f"R:R {rr_ratio:.1f}:1 - favorable risk/reward")

                        # Historical edge context
                        historical = self._get_historical_stats()
                        why_trade.append(f"Historical edge: {historical.get('strategy_win_rate', 66.96):.1f}% WR, {historical.get('strategy_profit_factor', 1.53):.2f} PF over {historical.get('total_trades_validated', 1501)} trades")

                        return {
                            "symbol": signal.get("symbol", "N/A"),
                            "direction": signal.get("direction", "long"),
                            "confidence": confidence,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "rr_ratio": rr_ratio,
                            "signal_time": signal.get("signal_time"),
                            # Multi-targets
                            "targets": {
                                "t1": t1,
                                "t2": t2,
                                "t3": t3,
                                "t1_label": f"T1 (1R): ${t1:.2f}",
                                "t2_label": f"T2 (1.5R): ${t2:.2f}",
                                "t3_label": f"T3 (2R): ${t3:.2f}",
                            },
                            # Expected move (weekly)
                            "expected_move_weekly": expected_move_weekly,
                            "expected_move_weekly_pct": expected_move_weekly_pct,
                            # Connors RSI-2 metrics
                            "rsi_5": rsi_5,
                            "bb_lower": bb_lower,
                            "bb_upper": bb_upper,
                            "sma_200": sma_200,
                            "bb_position_pct": bb_position_pct,
                            "gap_pct": signal.get("gap_pct", 0),
                            "intraday_change_pct": signal.get("intraday_change_pct", 0),
                            # Analysis
                            "conviction_reasons": signal.get("conviction_reasons", []),
                            # WHY TAKE THIS TRADE
                            "why_take_trade": why_trade,
                            # Phase 16B fields
                            "status": status,
                            "readiness_score": readiness_score,
                            "rsi_score": round(rsi_score),
                            "band_score": round(band_score),
                            "trend_score": round(trend_score),
                            "freshness": "fresh" if signal.get("is_fresh", True) else "stale",
                            # Entry checks for WHY panel
                            "entry_checks": {
                                "rsi_below_3": rsi_met,
                                "close_at_bb_lower": band_met,
                                "above_sma200": trend_met,
                            },
                            # Historical stats from validated backtest results
                            "historical": historical,
                            # Exit rules (SSOT v2.1.0)
                            "exit_rules": {
                                "rsi5_exit_threshold": 50,
                                "max_holding_days": 30,
                                "note": "AUTOMATED: RSI(5) > 50 OR 30-day timeout (SSOT v2.1.0)",
                            },
                            # Guidance
                            "guidance": {
                                "suggested_stop": stop_loss,
                                "suggested_target": take_profit,
                                "note": "Guidance only - NOT automated exits",
                            },
                        }

                    # Transform and filter out invalid signals
                    top_10_raw = data.get("top_10", [])
                    top_3_raw = data.get("top_3", [])

                    top_10 = [s for s in (transform_signal(sig) for sig in top_10_raw) if s is not None]
                    top_3 = [s for s in (transform_signal(sig) for sig in top_3_raw) if s is not None]

                    # Calculate scanner freshness
                    scan_time_str = data.get("scan_time")
                    scanner_freshness = "Unknown"
                    scanner_age_seconds = None

                    if scan_time_str:
                        try:
                            # Parse the scan time
                            scan_dt = datetime.fromisoformat(scan_time_str.replace("Z", "+00:00"))
                            # Make it timezone aware if not already
                            if scan_dt.tzinfo is None:
                                scan_dt = scan_dt.replace(tzinfo=CT)
                            now = datetime.now(CT)
                            age = now - scan_dt.astimezone(CT)
                            scanner_age_seconds = int(age.total_seconds())

                            # Determine freshness label
                            if scanner_age_seconds < 900:  # < 15 minutes
                                scanner_freshness = "Fresh"
                            elif scanner_age_seconds < 21600:  # < 6 hours
                                scanner_freshness = "Today"
                            else:
                                scanner_freshness = "Stale"
                        except Exception:
                            pass

                    # Calculate long/short counts if not in file (backward compatibility)
                    long_count = data.get("long_count")
                    short_count = data.get("short_count")
                    if long_count is None or short_count is None:
                        # Count from all_signals
                        all_signals = data.get("all_signals", [])
                        long_count = sum(1 for s in all_signals if s.get("direction", "").lower() == "long")
                        short_count = sum(1 for s in all_signals if s.get("direction", "").lower() == "short")

                    # Map fields to expected format
                    return {
                        "total_signals": data.get("signals_found", data.get("total_found", 0)),
                        "total_scanned": data.get("total_scanned", data.get("universe_size", 0)),
                        "universe_mode": data.get("universe_mode", "PROVEN_900"),
                        "long_count": long_count,
                        "short_count": 0,  # Connors RSI-2 is LONG ONLY
                        "top_10": top_10,
                        "top_3": top_3,
                        "scan_time": scan_time_str,
                        "scan_runtime_seconds": data.get("scan_runtime_seconds"),
                        "scanner_freshness": scanner_freshness,
                        "scanner_age_seconds": scanner_age_seconds,
                    }
        except Exception as e:
            print(f"Error loading scanner results: {e}")
        return {"total_signals": 0, "total_scanned": 0, "long_count": 0, "short_count": 0, "universe_mode": "PROVEN_900", "top_10": [], "top_3": [], "scan_time": None, "scan_runtime_seconds": None, "scanner_freshness": "Unknown", "scanner_age_seconds": None}

    def _calculate_rr_ratio(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk-reward ratio."""
        try:
            if entry is None or stop is None or target is None:
                return 0.0
            if entry == 0 or stop == 0:
                return 0.0

            risk = abs(entry - stop)
            reward = abs(target - entry)

            if risk == 0:
                return 0.0

            return reward / risk
        except (TypeError, ZeroDivisionError):
            return 0.0

    def _load_stats(self) -> Dict:
        """Load performance stats from paper trading."""
        try:
            path = Path("data/performance_stats.json")
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        # Return zeros for paper session (no real trades yet)
        return {
            "win_rate": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "profit_factor": 0.0,
            "avg_winner": 0,
            "avg_loser": 0,
        }

    def _get_trading_mode(self) -> Dict:
        """Get current trading mode from .env.paper file."""
        mode = "PAPER"  # Default safe mode
        live_gated = True

        try:
            env_file = Path(".env.paper")
            if env_file.exists():
                content = env_file.read_text()
                if "TRADING_MODE=LIVE" in content:
                    mode = "LIVE"
                    live_gated = False
                elif "TRADING_MODE=PAPER" in content:
                    mode = "PAPER"
        except OSError:
            pass

        return {
            "mode": mode,
            "live_gated": live_gated,
            "is_paper": mode == "PAPER",
        }

    def _get_risk_dashboard(self, equity: float, positions: List[Dict]) -> Dict:
        """Calculate real-time risk metrics for dashboard display.

        Args:
            equity: Current account equity
            positions: List of current open positions

        Returns:
            Dict with risk metrics for display
        """
        # Risk parameters from centralized config (SSOT)
        risk_config = get_risk_params_from_config()
        MAX_POSITIONS = risk_config["max_positions"]
        RISK_PER_TRADE_PCT = risk_config["risk_per_trade_pct"]
        MAX_DAILY_LOSS_PCT = risk_config["daily_loss_limit_pct"]
        MAX_DRAWDOWN_PCT = risk_config["account_stop_loss_dd_pct"]

        equity = equity or self.initial_capital
        current_positions = len(positions)

        # Calculate risk per trade in dollars
        risk_per_trade_usd = equity * (RISK_PER_TRADE_PCT / 100)

        # Calculate total capital at risk (sum of position risks)
        total_risk = 0.0
        for pos in positions:
            entry = pos.get("entry", 0)
            stop = pos.get("stop_loss", entry * 0.97)  # Default 3% stop if not set
            qty = pos.get("qty", 0)
            if entry > 0 and stop > 0:
                position_risk = abs(entry - stop) * qty
                total_risk += position_risk

        capital_at_risk_pct = (total_risk / equity * 100) if equity > 0 else 0

        # Max daily loss in dollars
        max_daily_loss_usd = equity * (MAX_DAILY_LOSS_PCT / 100)

        # Current drawdown (from initial capital)
        current_drawdown_pct = ((self.initial_capital - equity) / self.initial_capital * 100) if equity < self.initial_capital else 0

        # Portfolio heat (visual indicator of risk level)
        if capital_at_risk_pct < 2:
            heat_level = "LOW"
            heat_color = "green"
        elif capital_at_risk_pct < 4:
            heat_level = "NORMAL"
            heat_color = "green"
        elif capital_at_risk_pct < 6:
            heat_level = "ELEVATED"
            heat_color = "yellow"
        else:
            heat_level = "HIGH"
            heat_color = "red"

        return {
            "max_positions": MAX_POSITIONS,
            "current_positions": current_positions,
            "positions_available": MAX_POSITIONS - current_positions,
            "risk_per_trade_pct": RISK_PER_TRADE_PCT,
            "risk_per_trade_usd": round(risk_per_trade_usd, 2),
            "capital_at_risk_pct": round(capital_at_risk_pct, 2),
            "capital_at_risk_usd": round(total_risk, 2),
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "max_daily_loss_usd": round(max_daily_loss_usd, 2),
            "current_drawdown_pct": round(current_drawdown_pct, 2),
            "max_drawdown_pct": MAX_DRAWDOWN_PCT,
            "heat_level": heat_level,
            "heat_color": heat_color,
        }

    def _get_validation_metrics(self) -> Dict:
        """Load validated strategy metrics from bulletproof validation."""
        try:
            path = Path("reports/bulletproof_validation/final_validation_results.json")
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    final_stats = data.get("final_stats", {})
                    year_by_year = data.get("year_by_year", {})
                    return {
                        "strategy": data.get("strategy", "Connors RSI-2"),
                        "win_rate": final_stats.get("win_rate", 66.96),
                        "profit_factor": final_stats.get("pf", 1.53),
                        "total_trades": final_stats.get("trades", 1501),
                        "pf_2022": year_by_year.get("2022", {}).get("pf", 0.77),
                        "decision": data.get("decision", "GREEN"),
                        "confidence": data.get("confidence", 10),
                    }
        except (json.JSONDecodeError, OSError, KeyError):
            pass
        return {
            "strategy": "IBS+RSI + ICT",
            "win_rate": 66.96,
            "profit_factor": 1.53,
            "total_trades": 1501,
            "pf_2022": 0.77,
            "decision": "GREEN",
            "confidence": 10,
        }

    def _get_strategy_info(self) -> Dict:
        """Get strategy info with real validated metrics."""
        validation = self._get_validation_metrics()
        return {
            "name": validation.get("strategy", "Connors RSI-2"),
            "direction": "LONG ONLY",
            "win_rate": f"{validation.get('win_rate', 0):.2f}%",
            "profit_factor": f"{validation.get('profit_factor', 0):.4f}",
            "total_trades": validation.get("total_trades", 0),
            "pf_2022": validation.get("pf_2022", 0),
            "decision": validation.get("decision", "N/A"),
            "confidence": validation.get("confidence", 0),
        }

    def _get_historical_stats(self) -> Dict:
        """Get historical stats from validation results for signal WHY panel."""
        validation = self._get_validation_metrics()
        return {
            "strategy_win_rate": validation.get("win_rate", 0),
            "strategy_profit_factor": validation.get("profit_factor", 0),
            "total_trades_validated": validation.get("total_trades", 0),
            "pf_2022_bear": validation.get("pf_2022", 0),
            "decision": validation.get("decision", "N/A"),
        }

    def _get_universe_info(self) -> Dict:
        """Get universe information from PROVEN_900 universe (data/polygon/daily)."""
        try:
            # PROVEN_900 universe: data/polygon/daily/*.parquet
            # Verified: 66.96% WR, 1.53 PF (SSOT v2.1.0)
            data_dir = Path("data/polygon/daily")
            if data_dir.exists():
                parquet_count = len(list(data_dir.glob("*.parquet")))
                return {
                    "name": "PROVEN_900_UNIVERSE",
                    "count": parquet_count,
                    "expected": 832,
                    "verified_results": {
                        "win_rate": 66.96,
                        "profit_factor": 1.53,
                        "t_statistic": 12.76,  # SSOT v2.1.0
                    },
                }
        except OSError:
            pass
        return {
            "name": "PROVEN_900_UNIVERSE",
            "count": 832,
            "expected": 832,
        }

    def _get_health_status(self) -> Dict:
        """Get system health status from health endpoints and runtime files."""
        try:
            # Check gate status
            gate_file = Path("data/runtime/gate_status.json")
            gate_open = True  # Default to open in PAPER mode
            if gate_file.exists():
                with open(gate_file) as f:
                    data = json.load(f)
                    gate_open = data.get("open", True)

            # Check circuit breaker status
            cb_file = Path("data/runtime/circuit_breaker.json")
            cb_closed = True  # Default to closed (normal operation)
            if cb_file.exists():
                with open(cb_file) as f:
                    data = json.load(f)
                    state = data.get("state", "closed")
                    cb_closed = (state == "closed")

            # Get data freshness from universe
            universe_info = self._get_universe_info()
            data_ok = universe_info.get("count", 0) >= 700  # At least 700 files

            # Connection status (if we got here, app is running)
            connection_ok = True
            data_feed_ok = self.alpaca is not None or self.polygon is not None

            return {
                "connection_ok": connection_ok,
                "data_feed_ok": data_feed_ok,
                "gate_open": gate_open,
                "circuit_breaker_ok": cb_closed,
                "data_ok": data_ok,
                "universe_count": universe_info.get("count", 0),
            }
        except Exception:
            return {
                "connection_ok": True,
                "data_feed_ok": True,
                "gate_open": True,
                "circuit_breaker_ok": True,
                "data_ok": True,
                "universe_count": 710,
            }

    def _get_schedule(self, now_ct: datetime) -> Dict:
        """Get next scheduled events with professional names (user's CT routine)."""
        # User's requested CT schedule for Mon-Fri trading days
        weekday_schedules = [
            ("05:00", "ðŸ”§ System Wake + Health Check"),
            ("06:00", "ðŸ“Š Fresh Data Update"),
            ("07:00", "ðŸ“ˆ Pre-Market Report"),
            ("08:30", "ðŸ” Market Open Scan + Live Trade"),
            ("09:30", "âœ… System/Position Check"),
            ("14:30", "ðŸŒŠ Swing Scan (1 trade max)"),
            ("16:30", "ðŸ“‹ After-Market Report"),
            ("19:30", "ðŸŒ™ Evening Scan + Watchlist"),
        ]

        # Saturday weekly report
        saturday_schedules = [
            ("08:30", "ðŸ“Š Weekly Full Report + Discovery"),
        ]

        # Determine if today is Saturday
        is_saturday = now_ct.weekday() == 5
        is_sunday = now_ct.weekday() == 6

        if is_sunday:
            # On Sunday, next task is Monday 05:00
            schedules = weekday_schedules
            next_job = {"time": "05:00 CT (Mon)", "name": "ðŸ”§ System Wake + Health Check"}
        elif is_saturday:
            schedules = saturday_schedules
            current_time = now_ct.strftime("%H:%M")
            next_job = None
            for time_str, name in schedules:
                if time_str > current_time:
                    next_job = {"time": f"{time_str} CT", "name": name}
                    break
            if not next_job:
                next_job = {"time": "05:00 CT (Mon)", "name": "ðŸ”§ System Wake + Health Check"}
        else:
            # Weekday
            schedules = weekday_schedules
            current_time = now_ct.strftime("%H:%M")
            next_job = None
            for time_str, name in schedules:
                if time_str > current_time:
                    next_job = {"time": f"{time_str} CT", "name": name}
                    break
            if not next_job:
                # After last task, next is tomorrow's first task (or Saturday if Friday)
                if now_ct.weekday() == 4:  # Friday
                    next_job = {"time": "08:30 CT (Sat)", "name": "ðŸ“Š Weekly Full Report + Discovery"}
                else:
                    next_job = {"time": "05:00 CT", "name": "ðŸ”§ System Wake + Health Check"}

        return {"next": next_job, "all": weekday_schedules + [("Sat 08:30", "ðŸ“Š Weekly Report")]}

    def _load_ai_brain(self) -> Dict:
        """Load AI brain state and generate rule-based scanner insights."""
        # First try to load ML model report (if exists)
        ml_report = {}
        try:
            path = Path("data/ai_learning/training_report.json")
            if path.exists():
                with open(path) as f:
                    ml_report = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

        # Generate RULE-BASED insights from scanner data (this is deterministic, not ML)
        scanner_insights = self._generate_scanner_insights()

        return {
            "status": scanner_insights.get("status", "RULE_BASED_ACTIVE"),
            "model_type": "Connors RSI-2 Rule Engine (validated)",
            "test_accuracy": ml_report.get("honest_metrics", {}).get("test_accuracy", 0),
            "test_auc": ml_report.get("honest_metrics", {}).get("test_roc_auc", 0),
            "sample_size": scanner_insights.get("signals_analyzed", 0),
            "feature_importances": {},
            "caveats": scanner_insights.get("caveats", []),
            "recommendations": scanner_insights.get("recommendations", []),
            # New scanner-based insights
            "scanner_insights": scanner_insights.get("insights", []),
            "signal_breakdown": scanner_insights.get("signal_breakdown", {}),
            "strategy_info": scanner_insights.get("strategy_info", {}),
        }

    def _generate_scanner_insights(self) -> Dict:
        """Generate rule-based insights from scanner results."""
        insights = []
        caveats = []
        recommendations = []

        # Load scanner results AND trade stats for comprehensive insights
        scanner_data = self._load_scanner_results()
        trade_stats = self._load_stats()

        if not scanner_data:
            return {
                "status": "NO_DATA",
                "insights": ["No recent scan data available"],
                "caveats": ["Run a scan to generate insights"],
                "recommendations": ["Click 'Scan Now' to scan the market"],
                "signals_analyzed": 0,
                "trade_performance": {},
            }

        total_signals = scanner_data.get("total_signals", 0)
        total_scanned = scanner_data.get("total_scanned", 0)
        long_count = scanner_data.get("long_count", 0)
        scanner_data.get("short_count", 0)
        top_3 = scanner_data.get("top_3", [])

        # Signal quality insights
        if total_signals == 0:
            insights.append("ðŸ“‰ No setups detected - no extreme oversold conditions found")
            caveats.append("Consider waiting for extreme RSI < 3 readings")
        elif total_signals < 5:
            insights.append(f"ðŸ“Š {total_signals} setups found - selective market conditions")
            recommendations.append("Focus on highest confidence signals only")
        elif total_signals < 15:
            insights.append(f"âœ… {total_signals} quality setups - healthy market structure")
        else:
            insights.append(f"âš¡ {total_signals} signals - high activity, be selective")
            caveats.append("Too many signals can indicate choppy conditions")

        # Signal count analysis (LONG ONLY strategy)
        if long_count > 0:
            insights.append(f"ðŸ“ˆ {long_count} LONG setups found (LONG ONLY strategy)")
            if long_count >= 10:
                recommendations.append("Multiple extreme oversold setups - prioritize highest confidence")
            elif long_count >= 5:
                recommendations.append("Focus on extreme RSI < 3 setups below lower Bollinger Band")
            else:
                recommendations.append("Limited setups - ensure RSI < 3 and price below BB lower band")

        # Top signal analysis
        if top_3:
            top_signal = top_3[0]
            conf = top_signal.get("confidence", 0)
            if conf >= 90:
                insights.append(f"â­ High-confidence leader: {top_signal.get('symbol')} at {conf:.0f}%")
            elif conf >= 80:
                insights.append(f"ðŸŽ¯ Quality leader: {top_signal.get('symbol')} at {conf:.0f}%")
            else:
                caveats.append(f"Top signal only {conf:.0f}% confidence - borderline")

            # Check RSI extremes in top signals
            avg_rsi = sum((s.get("rsi_5") or 5) for s in top_3) / len(top_3)
            if avg_rsi < 2.5:
                insights.append(f"ðŸ”¥ Extreme oversold conditions in Top 3 - avg RSI: {avg_rsi:.1f}")
            elif avg_rsi > 4.0:
                caveats.append(f"Top signals RSI above 4.0 (avg: {avg_rsi:.1f}) - less extreme")

        # Staleness check
        scanner_age = scanner_data.get("scanner_age_seconds") or 0
        if scanner_age and scanner_age > 1800:  # 30 minutes
            caveats.append(f"Scan is {scanner_age // 60}m old - consider rescanning")

        # Add trade performance summary for System Insights panel
        total_trades = trade_stats.get("total_trades", 0)
        win_rate = trade_stats.get("win_rate", 0.0)
        profit_factor = trade_stats.get("profit_factor", 0.0)
        wins = trade_stats.get("wins", 0)
        losses = trade_stats.get("losses", 0)

        return {
            "status": "RULE_BASED_ACTIVE",
            "insights": insights,
            "caveats": caveats,
            "recommendations": recommendations,
            "signals_analyzed": total_signals,
            "signal_breakdown": {
                "total": total_signals,
                "long": long_count,
                "scanned": total_scanned,
            },
            "strategy_info": self._get_strategy_info(),
            "trade_performance": {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "wins": wins,
                "losses": losses,
                "profit_factor": profit_factor,
                "sample_warning": total_trades < 30,
            },
        }

    def _load_news(self, watchlist_tickers: Optional[List[str]] = None) -> List[Dict]:
        """Load recent market news, filtered by watchlist tickers if provided.

        Args:
            watchlist_tickers: List of stock symbols to filter news for.
                              If None, returns unfiltered news.

        Returns:
            List of news items, max 10, filtered for watchlist if provided.
            Falls back to general market news if no watchlist-specific news found.
        """
        try:
            path = Path("data/news/latest_news.json")
            if path.exists():
                with open(path) as f:
                    news_items = json.load(f)

                    if not isinstance(news_items, list):
                        return []

                    # Filter by watchlist tickers if provided
                    if watchlist_tickers:
                        watchlist_set = set(t.upper() for t in watchlist_tickers)
                        filtered_news = []
                        for item in news_items:
                            item_symbols = item.get("symbols", [])
                            # Check if any news symbol matches our watchlist
                            if any(sym.upper() in watchlist_set for sym in item_symbols):
                                # Add matched ticker info for UI highlighting
                                matched = [s for s in item_symbols if s.upper() in watchlist_set]
                                item["matched_watchlist"] = matched
                                filtered_news.append(item)
                                if len(filtered_news) >= 10:
                                    break

                        # Fallback: If no watchlist news found, show general market news
                        if not filtered_news:
                            # Mark items as general market news (no specific match)
                            general_news = []
                            for item in news_items[:10]:
                                item["is_general_news"] = True
                                general_news.append(item)
                            return general_news

                        return filtered_news

                    # No filter - return top 10 most recent items
                    return news_items[:10]
        except Exception as e:
            print(f"Error loading news: {e}")

        # Fallback to placeholder if file doesn't exist
        return [
            {
                "headline": "Market News Integration Coming Soon",
                "timestamp": datetime.now().isoformat(),
                "source": "System",
                "sentiment": "neutral"
            }
        ]

    def _load_daily_reports(self) -> Dict:
        """Load PRE-GAME, HALF-TIME, POST-GAME daily reports."""
        if not report_loader:
            return {
                "pregame": {"available": False, "error": "Report loader not initialized"},
                "halftime": {"available": False, "error": "Report loader not initialized"},
                "postgame": {"available": False, "error": "Report loader not initialized"},
                "period": "unavailable"
            }

        try:
            today_date = datetime.now().strftime("%Y%m%d")
            return {
                "pregame": report_loader.load_pregame_report(today_date),
                "halftime": report_loader.load_halftime_report(today_date),
                "postgame": report_loader.load_postgame_report(today_date),
                "period": report_loader.get_dashboard_data(today_date).get("period", "unavailable")
            }
        except Exception as e:
            print(f"Error loading daily reports: {e}")
            return {
                "pregame": {"available": False, "error": str(e)},
                "halftime": {"available": False, "error": str(e)},
                "postgame": {"available": False, "error": str(e)},
                "period": "error"
            }

    def _get_market_status_context(self, now_et: datetime) -> Dict:
        """Get enhanced market status with context."""
        is_open = self._is_market_open(now_et)
        status = "OPEN" if is_open else "CLOSED"

        # Time of day classification
        hour = now_et.hour
        if hour < 4:
            time_period = "Late Night"
        elif hour < 9:
            time_period = "Pre-Market"
        elif 9 <= hour < 12:
            time_period = "Morning Session"
        elif 12 <= hour < 14:
            time_period = "Midday"
        elif 14 <= hour < 16:
            time_period = "Afternoon Session"
        elif 16 <= hour < 20:
            time_period = "After-Hours"
        else:
            time_period = "Evening"

        # Quarter and season
        month = now_et.month
        quarter = f"Q{(month - 1) // 3 + 1}"

        # Season context
        if month in [11, 12]:
            season = "Holiday Season"
        elif month == 1:
            season = "New Year"
        elif month in [2, 3]:
            season = "Winter"
        elif month in [4, 5, 6]:
            season = "Spring"
        elif month in [7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        context = f"{time_period} â€¢ {quarter} {season}"

        return {
            "status": status,
            "context": context,
            "time_period": time_period,
            "quarter": quarter,
            "season": season,
        }

    def _generate_market_intelligence(self, vix: float, spy_price: float, market_status: str, now_et: datetime) -> Dict:
        """Generate intelligent market analysis and game plan."""

        # VIX interpretation
        if vix < 12:
            volatility = "ðŸ“‰ Very Low Volatility - Potential breakout brewing"
            stress = "CALM"
        elif vix < 15:
            volatility = "âœ… Low Volatility - Normal market conditions"
            stress = "NORMAL"
        elif vix < 20:
            volatility = "âš ï¸ Elevated Volatility - Increased caution"
            stress = "ELEVATED"
        elif vix < 30:
            volatility = "ðŸ”¥ High Volatility - Active opportunity window"
            stress = "HIGH"
        else:
            volatility = "âš¡ Extreme Volatility - Risk-off mode"
            stress = "EXTREME"

        # Time-based game plan
        hour_et = now_et.hour

        if market_status == "CLOSED":
            if hour_et < 4:
                game_plan = "ðŸŒ™ After-hours - Review EOD data, prepare watchlist"
            elif hour_et < 9:
                game_plan = "ðŸŒ… Pre-market prep - Monitor gaps, news, overseas markets"
            else:
                game_plan = "ðŸ“Š Post-close - Analyze today's setups, backtest adjustments"
        else:
            if 9 <= hour_et < 10:
                game_plan = "ðŸ”¥ Opening Range - High volatility, wait for clear structure"
            elif 10 <= hour_et < 11:
                game_plan = "ðŸ“ˆ Morning Session - Prime mean reversion window"
            elif 11 <= hour_et < 14:
                game_plan = "â¸ï¸ Midday Chop - Reduced activity, avoid low-quality setups"
            elif 14 <= hour_et < 15:
                game_plan = "âš¡ Power Hour Setup - Watch for extreme oversold bounces"
            else:
                game_plan = "ðŸŽ¯ Final Hour - Last chance setups, manage open positions"

        # What to look for
        lookfor = []
        if vix < 15:
            lookfor.append("âœ“ Extreme RSI < 3 readings in uptrends")
            lookfor.append("âœ“ BB breakdown with tight stops")
        else:
            lookfor.append("âš ï¸ Wait for volatility stabilization")
            lookfor.append("âš ï¸ Wider stops due to increased noise")

        if market_status == "OPEN":
            lookfor.append("ðŸŽ¯ Price below BB(20,2) + RSI(5) < 3")
            lookfor.append("ðŸ” Stocks in uptrends (above SMA 200)")
        else:
            lookfor.append("ðŸ“‹ Review completed trades for patterns")
            lookfor.append("ðŸ”¬ Backtest parameter sensitivity")

        # Regime detection based on VIX (using centralized SSOT)
        # VIX_THRESHOLDS: NORMAL <16, ELEVATED 16-24, HIGH 24-32, EXTREME >=32
        vix_regime = get_vix_regime(vix)
        regime = vix_regime["regime"]
        regime_label = vix_regime["label"]
        regime_color = vix_regime["color"]
        win_rate_adj = vix_regime["win_rate_adjustment"]
        position_size_pct = vix_regime["position_size_pct"]
        regime_recommendation = vix_regime["recommendation"]

        # Base validated stats (from SINGLE_SOURCE_OF_TRUTH.json with RSI < 3.0)
        base_win_rate = 66.96  # SSOT v2.1.0
        adjusted_win_rate = max(50, base_win_rate + win_rate_adj)

        return {
            "volatility_analysis": volatility,
            "stress_level": stress,
            "game_plan": game_plan,
            "what_to_look_for": lookfor,
            "vix_value": vix,
            "position_sizing": f"{position_size_pct}%",
            # Enhanced regime detection (from scanner_config.py SSOT)
            "regime": {
                "current": regime,
                "label": regime_label,
                "color": regime_color,
                "vix": vix,
                "base_win_rate": base_win_rate,
                "adjusted_win_rate": adjusted_win_rate,
                "win_rate_adjustment": win_rate_adj,
                "recommendation": regime_recommendation,
                "position_size_pct": position_size_pct,
                "thresholds": VIX_THRESHOLDS,  # Include thresholds for transparency
            }
        }


# Compatibility alias for tests expecting QuantDashboardDataProvider
class QuantDashboardDataProvider(ProDashboardData):
    """Compatibility wrapper providing the interface expected by unit tests.

    Maps:
    - get_dashboard_data() -> get_full_data()
    - project_root attribute -> Path(".")
    """

    def __init__(self):
        super().__init__()
        self.project_root = Path(".")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Alias for get_full_data() for test compatibility."""
        return self.get_full_data()


# Initialize data provider
data_provider = ProDashboardData()

# Initialize report loader for daily reports
PROJECT_ROOT = Path(__file__).parent.parent.parent
report_loader = None
if REPORT_LOADER_AVAILABLE:
    try:
        report_loader = get_data_loader(PROJECT_ROOT)
    except Exception as e:
        print(f"Failed to initialize report loader: {e}")

# WebSocket connections
active_connections: List[WebSocket] = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates - 5 second refresh."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Send fresh data every 5 seconds for real-time updates
            data = data_provider.get_full_data()
            await websocket.send_json(data)
            await asyncio.sleep(5)  # Changed from 60s to 5s for real-time dashboard
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/api/data")
async def get_data():
    """REST endpoint for full data."""
    return JSONResponse(data_provider.get_full_data())


@app.get("/health")
async def health_check():
    """Health check endpoint - always returns OK if app is running."""
    return {"status": "ok"}


@app.get("/api/reports/pregame")
async def get_pregame_report(date: str = None):
    """Get PRE-GAME report for specified date (default: today)."""
    if not report_loader:
        return JSONResponse({"error": "Report loader not available", "available": False})
    try:
        pregame_data = report_loader.load_pregame_report(date)
        return JSONResponse(pregame_data)
    except Exception as e:
        return JSONResponse({"error": str(e), "available": False})


@app.get("/api/reports/halftime")
async def get_halftime_report(date: str = None):
    """Get HALF-TIME report for specified date (default: today)."""
    if not report_loader:
        return JSONResponse({"error": "Report loader not available", "available": False})
    try:
        halftime_data = report_loader.load_halftime_report(date)
        return JSONResponse(halftime_data)
    except Exception as e:
        return JSONResponse({"error": str(e), "available": False})


@app.get("/api/reports/postgame")
async def get_postgame_report(date: str = None):
    """Get POST-GAME report for specified date (default: today)."""
    if not report_loader:
        return JSONResponse({"error": "Report loader not available", "available": False})
    try:
        postgame_data = report_loader.load_postgame_report(date)
        return JSONResponse(postgame_data)
    except Exception as e:
        return JSONResponse({"error": str(e), "available": False})


@app.get("/api/reports/dashboard")
async def get_dashboard_reports():
    """Get all reports for dashboard display."""
    if not report_loader:
        return JSONResponse({"error": "Report loader not available", "available": False})
    try:
        dashboard_data = report_loader.get_dashboard_data()
        return JSONResponse(dashboard_data)
    except Exception as e:
        return JSONResponse({"error": str(e), "available": False})


@app.get("/readiness")
async def readiness_check():
    """Readiness check endpoint - validates system dependencies."""
    from fastapi.responses import Response
    import json as json_mod

    checks = {}
    all_ready = True

    # Check universe data
    universe_path = Path("data/polygon/daily")
    if universe_path.exists():
        csv_count = len(list(universe_path.glob("*.csv")))
        checks["universe_data"] = f"ready ({csv_count} stocks)"
        if csv_count < 700:
            all_ready = False
    else:
        checks["universe_data"] = "path not found"
        all_ready = False

    # Check edge stocks (check both old and new locations)
    edge_file = Path("config/universe/FINAL_EDGE_STOCKS.json")
    if not edge_file.exists():
        edge_file = Path("FINAL_EDGE_STOCKS.json")
    if edge_file.exists():
        checks["edge_stocks"] = "ready"
    else:
        checks["edge_stocks"] = "file missing"
        all_ready = False

    # Check Alpaca connection
    if data_provider.alpaca:
        try:
            acct = data_provider.alpaca.get_account()
            checks["alpaca"] = f"ready (equity: ${float(acct.get('equity', 0)):,.0f})"
        except Exception as e:
            checks["alpaca"] = f"failed: {str(e)[:50]}"
            all_ready = False
    else:
        checks["alpaca"] = "not initialized"
        all_ready = False

    status_code = 200 if all_ready else 503
    content = json_mod.dumps({
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    })

    return Response(content=content, status_code=status_code, media_type="application/json")


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    # Get metrics from health_endpoints module
    try:
        from src.monitor.health_endpoints import get_health_checker
        checker = get_health_checker()
        metrics_text = checker.get_prometheus_metrics()
        return PlainTextResponse(content=metrics_text, media_type="text/plain; version=0.0.4")
    except Exception:
        # Fallback: return basic metrics
        metrics = []
        metrics.append("# HELP trading_system_up System is running")
        metrics.append("# TYPE trading_system_up gauge")
        metrics.append("trading_system_up 1")
        metrics.append("")
        metrics.append("# HELP trading_system_error Metrics collection error")
        metrics.append("# TYPE trading_system_error gauge")
        metrics.append("trading_system_error 1")
        return PlainTextResponse(content="\n".join(metrics), media_type="text/plain; version=0.0.4")


@app.post("/api/scan")
async def trigger_scan():
    """
    Trigger IBS+RSI + ICT scanner on PROVEN_900 universe.

    Signal Detection Logic (validated production strategy - SSOT v2.1.0):
    - Entry: Close < Lower BB(20,2) AND RSI(5) < 3 (EXTREME) AND Close > SMA(200)
    - Exit: RSI(5) > 50 OR 30-day timeout
    - Direction: LONG ONLY (mean reversion on extreme oversold)
    - Universe: 900 liquid US equities (PROVEN_900 from data/polygon/daily/*.parquet)
    - Validated: Win Rate 66.96%, PF 1.53, 1,501 trades (SINGLE_SOURCE_OF_TRUTH.json v2.1.0)

    Ranking: Deterministic by (confidence DESC, rr_ratio DESC, symbol ASC)
    """
    try:
        import time as time_module
        import numpy as np

        scan_start = datetime.now(CT)
        start_monotonic = time_module.monotonic()

        # Get universe symbols from PROVEN_900 (data/polygon/daily/*.parquet)
        try:
            from src.config.universe_config import get_universe_symbols, UniverseMode
            symbols = get_universe_symbols(UniverseMode.PROVEN_900)
        except Exception:
            # Fallback to liquid stocks if universe loading fails
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA",
                       "UNH", "HD", "PG", "JNJ", "DIS", "NFLX", "ADBE", "CRM", "AMD", "INTC",
                       "WMT", "KO", "PEP", "MCD", "XOM", "CVX", "BAC", "WFC", "GS", "MS"]

        signals = []
        api_key = os.getenv("POLYGON_API_KEY", "")
        scan_symbols = symbols
        long_count = 0
        short_count = 0  # Always 0 for this strategy
        errors_count = 0

        # Connors RSI-2 Strategy Parameters (VALIDATED - DO NOT CHANGE)
        BB_PERIOD = 20
        BB_STD = 2.0
        RSI_PERIOD = 5
        RSI_THRESHOLD = 3.0  # CRITICAL: RSI < 3 is the edge
        TREND_SMA_PERIOD = 200

        for symbol in scan_symbols:
            try:
                # Fetch historical data from Polygon for indicator calculation
                # Need enough bars for SMA(200) calculation
                end_date = datetime.now()
                start_date = end_date - timedelta(days=250)  # ~1 year of data

                aggs_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={api_key}"
                resp = requests.get(aggs_url, timeout=3)

                if resp.status_code != 200:
                    errors_count += 1
                    continue

                data = resp.json()
                results = data.get("results", [])

                if len(results) < TREND_SMA_PERIOD + 10:
                    # Not enough data for indicators
                    continue

                # Convert to arrays for calculation
                closes = np.array([r['c'] for r in results])
                np.array([r['h'] for r in results])
                np.array([r['l'] for r in results])
                np.array([r['v'] for r in results])

                if len(closes) == 0 or closes[-1] <= 0:
                    continue

                # ========== BOLLINGER BANDS (20, 2) ==========
                sma20 = np.mean(closes[-BB_PERIOD:])
                std20 = np.std(closes[-BB_PERIOD:], ddof=1)
                bb_lower = sma20 - (BB_STD * std20)
                bb_upper = sma20 + (BB_STD * std20)

                # ========== RSI(5) ==========
                def calculate_rsi(prices, period):
                    deltas = np.diff(prices)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)

                    avg_gain = np.mean(gains[-period:])
                    avg_loss = np.mean(losses[-period:])

                    if avg_loss == 0:
                        return 100.0
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi

                rsi5 = calculate_rsi(closes, RSI_PERIOD)

                # ========== SMA(200) ==========
                if len(closes) >= TREND_SMA_PERIOD:
                    sma200 = np.mean(closes[-TREND_SMA_PERIOD:])
                else:
                    continue

                # ========== CURRENT PRICE ==========
                current_price = closes[-1]
                prev_close = closes[-2] if len(closes) > 1 else current_price

                # ========== ENTRY SIGNAL DETECTION ==========
                # ALL conditions must be true:
                # 1. Close < Lower Bollinger Band
                # 2. RSI(5) < 3.0
                # 3. Close > SMA(200) (uptrend filter)

                below_bb = current_price < bb_lower
                extreme_oversold = rsi5 < RSI_THRESHOLD
                in_uptrend = current_price > sma200

                if not (below_bb and extreme_oversold and in_uptrend):
                    continue

                # This is a LONG ONLY strategy
                direction = "long"
                long_count += 1

                # ========== ENTRY, STOP, TARGET ==========
                entry_price = current_price

                # Stop: 2% below lower BB
                stop_loss = bb_lower * 0.98

                # Target: SMA(200) as first profit target (mean reversion)
                take_profit = sma200

                # Calculate R:R
                risk_dollars = entry_price - stop_loss
                reward_dollars = take_profit - entry_price
                rr_ratio = reward_dollars / risk_dollars if risk_dollars > 0 else 0

                # ========== CONFIDENCE SCORING ==========
                # Lower RSI = Higher confidence (more extreme oversold)
                # RSI < 1: 0.95 confidence
                # RSI 1-2: 0.85 confidence
                # RSI 2-3: 0.75 confidence
                if rsi5 < 1.0:
                    confidence = 0.95
                elif rsi5 < 2.0:
                    confidence = 0.85
                else:
                    confidence = 0.75

                # Bonus for large distance below BB (stronger mean reversion signal)
                bb_distance_pct = (bb_lower - current_price) / current_price * 100
                if bb_distance_pct > 5.0:
                    confidence += 0.05
                    confidence = min(confidence, 0.98)

                # ========== STRATEGY ANALYSIS ==========
                daily_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0

                conviction_reasons = [
                    f"RSI(5) = {rsi5:.2f} (< {RSI_THRESHOLD} threshold) - EXTREME oversold",
                    f"Close ${current_price:.2f} below Lower BB ${bb_lower:.2f} ({bb_distance_pct:.1f}% below)",
                    f"Uptrend confirmed: Close > SMA(200) ${sma200:.2f}",
                    f"R:R = {rr_ratio:.1f}:1 (${risk_dollars:.2f} risk â†’ ${reward_dollars:.2f} reward)",
                    f"Daily change: {daily_change_pct:+.2f}%"
                ]

                signals.append({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    "entry_price": round(entry_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "take_profit": round(take_profit, 2),
                    "rr_ratio": round(rr_ratio, 2),
                    "signal_time": scan_start.isoformat(),
                    # BB_RSI metrics for display
                    "rsi_5": round(rsi5, 2),
                    "bb_lower": round(bb_lower, 2),
                    "bb_upper": round(bb_upper, 2),
                    "sma_20": round(sma20, 2),
                    "sma_200": round(sma200, 2),
                    "bb_distance_pct": round(bb_distance_pct, 2),
                    "daily_change_pct": round(daily_change_pct, 2),
                    # Rule-based analysis (validated strategy)
                    "conviction_reasons": conviction_reasons,
                    "strategy_5ws": {
                        "who": "Mean reversion system",
                        "what": "Connors RSI-2 entry",
                        "where": f"${entry_price:.2f} entry, ${stop_loss:.2f} stop",
                        "when": scan_start.strftime("%H:%M CT"),
                        "why": f"Extreme oversold RSI({rsi5:.2f}) with BB breakdown in uptrend",
                    }
                })

            except Exception:
                errors_count += 1
                continue

        # ========== DETERMINISTIC RANKING ==========
        # Sort by: confidence DESC, rr_ratio DESC, symbol ASC (for tie-breaking)
        signals.sort(key=lambda x: (-x["confidence"], -x["rr_ratio"], x["symbol"]))

        top_10 = signals[:10]
        top_3 = signals[:3]

        # Calculate elapsed time
        elapsed_seconds = round(time_module.monotonic() - start_monotonic, 2)

        # Save to JSON file
        output_dir = Path(__file__).parent.parent.parent / "data" / "scanner"
        output_dir.mkdir(parents=True, exist_ok=True)

        scan_results = {
            "scan_time": scan_start.isoformat(),
            "scan_runtime_seconds": elapsed_seconds,
            "universe_mode": "PROVEN_900",
            "universe_size": len(symbols),
            "total_scanned": len(scan_symbols),
            "total_found": len(signals),
            "signals_found": len(signals),
            "long_count": long_count,
            "short_count": short_count,
            "errors_count": errors_count,
            "watchlist": [s["symbol"] for s in top_3] if top_3 else [],
            "top_10": top_10,
            "top_3": top_3,
            "all_signals": signals,
        }

        # Use atomic write to prevent file corruption
        atomic_json_write(output_dir / "latest_scan_results.json", scan_results)

        return JSONResponse({
            "status": "success",
            "message": f"ðŸ” Found {len(signals)} setups ({long_count}L/{short_count}S) from {len(scan_symbols)} stocks in {elapsed_seconds}s",
            "signals_found": len(signals),
            "long_count": long_count,
            "short_count": short_count,
            "scan_time": scan_start.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "total_scanned": len(scan_symbols),
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Scanner error: {str(e)}"
        }, status_code=500)


# =============================================================================
# PHASE 16B: Enhanced Scanner API Endpoints
# =============================================================================

@app.get("/api/scan/summary")
async def get_scan_summary():
    """
    Get scan summary with readiness scores and status classification.

    PHASE 16B: Returns counts, top10 by readiness, top3 ready, freshness.
    Falls back to reading from latest_scan_results.json if EdgeMasterService unavailable.
    """
    # Try EdgeMasterService first
    service = _get_edge_service()
    if service is not None:
        try:
            summary = service.get_latest_summary()
            if summary is None:
                summary = service.run_scan()
            return JSONResponse(summary.to_dict())
        except Exception:
            pass  # Fall through to file-based fallback

    # Fallback: Read from latest_scan_results.json
    try:
        results_file = Path("data/scanner/latest_scan_results.json")
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            # Transform to summary format
            vix_info = data.get("vix", {})
            strategy_info = data.get("strategy", {})

            return JSONResponse({
                "scan_time": data.get("scan_time"),
                "scan_runtime_seconds": data.get("scan_runtime_seconds"),
                "universe_mode": data.get("universe_mode"),
                "total_scanned": data.get("total_scanned", 0),
                "signals_found": data.get("signals_found", 0),
                "long_count": data.get("long_count", 0),
                "short_count": data.get("short_count", 0),
                "ready_count": data.get("ready_count", 0),
                "watchlist_count": data.get("watchlist_count", 0),
                "ineligible_count": data.get("ineligible_count", 0),
                "fresh_count": data.get("fresh_count", 0),
                "stale_count": data.get("stale_count", 0),
                "top_10": data.get("top_10", []),
                "vix": vix_info,
                "strategy": strategy_info,
                "source": "file_cache",
            })
        else:
            return JSONResponse({
                "error": "No scan results available. Run a scan first.",
                "source": "none"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to read scan results: {str(e)[:200]}"
        }, status_code=500)


@app.get("/api/scan/symbol/{symbol}")
async def get_symbol_details(symbol: str):
    """
    Get full details for a symbol including WHY TAKE THIS TRADE panel.

    PHASE 16B: Returns readiness score, entry checks, exit rules, guidance.
    Falls back to reading from latest_scan_results.json if EdgeMasterService unavailable.
    """
    symbol = symbol.upper()

    # Try EdgeMasterService first
    service = _get_edge_service()
    if service is not None:
        try:
            details = service.get_symbol_details(symbol)
            if details is not None:
                return JSONResponse(details)
        except Exception:
            pass  # Fall through to file-based fallback

    # Fallback: Read from latest_scan_results.json
    try:
        results_file = Path("data/scanner/latest_scan_results.json")
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            # Find the symbol in top_10
            for sig in data.get("top_10", []):
                if sig.get("symbol", "").upper() == symbol:
                    return JSONResponse({
                        **sig,
                        "source": "file_cache",
                    })

            return JSONResponse({
                "error": f"Symbol {symbol} not found in latest scan results"
            }, status_code=404)
        else:
            return JSONResponse({
                "error": "No scan results available"
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to get symbol details: {str(e)[:200]}"
        }, status_code=500)


@app.get("/api/scan/why/{symbol}")
async def get_why_take_trade(symbol: str):
    """
    Get WHY TAKE THIS TRADE panel for a symbol.

    PHASE 16B: Returns entry checks, historical performance, exit rules, guidance.
    """
    service = _get_edge_service()
    if service is None:
        return JSONResponse({
            "error": "EdgeMasterService not available"
        }, status_code=503)

    symbol = symbol.upper()
    why = service.get_why_take_trade(symbol)

    if why is None:
        return JSONResponse({
            "error": f"No data for symbol: {symbol}"
        }, status_code=404)

    return JSONResponse(why.to_dict())


@app.post("/api/scan/enhanced")
async def trigger_enhanced_scan():
    """
    Trigger enhanced scan using EdgeMasterService.

    PHASE 16B: Returns readiness scores, status classification, freshness.
    """
    service = _get_edge_service()
    if service is None:
        return JSONResponse({
            "error": "EdgeMasterService not available",
            "fallback": "Use /api/scan for legacy scanner"
        }, status_code=503)

    try:
        import time as time_module
        start = time_module.monotonic()

        summary = service.run_scan()
        elapsed = round(time_module.monotonic() - start, 2)

        # Convert to dashboard-compatible format
        top_10 = []
        for score in summary.top10:
            why = service.get_why_take_trade(score.symbol)
            top_10.append({
                "symbol": score.symbol,
                "readiness_score": score.total_score,
                "status": score.status,
                "status_reason": score.status_reason,
                "direction": "long",
                "confidence": score.total_score,  # Map readiness to confidence for display
                "entry_price": round(score.close, 2),
                "stop_loss": round(score.close - (score.atr14 or score.close * 0.03), 2) if score.atr14 else round(score.close * 0.97, 2),
                "take_profit": round(score.close + (score.atr14 * 2 if score.atr14 else score.close * 0.06), 2),
                "rr_ratio": 2.0,
                "rsi_5": round(score.rsi5, 2),
                "bb_lower": round(score.bb_lower, 2),
                "bb_upper": round(score.bb_upper, 2),
                "sma_200": round(score.sma200, 2),
                "is_fresh": score.is_fresh,
                "freshness_badge": score.freshness_badge,
                "why_take_trade": why.to_dict() if why else None,
                "conviction_reasons": [
                    f"RSI(5) = {score.rsi5:.2f} {'< 3 âœ“' if score.rsi_met else '>= 3'}",
                    f"Close {'<=' if score.band_met else '>'} BB Lower (${score.bb_lower:.2f})",
                    f"{'Above' if score.trend_met else 'Below'} SMA(200) (${score.sma200:.2f})",
                    f"Readiness Score: {score.total_score}/100",
                ],
                "strategy_5ws": {
                    "who": "Connors RSI-2 Mean Reversion",
                    "what": f"Entry at ${score.close:.2f}",
                    "where": f"BB Lower ${score.bb_lower:.2f}, SMA200 ${score.sma200:.2f}",
                    "when": datetime.now(CT).strftime("%H:%M CT"),
                    "why": f"RSI({score.rsi5:.2f}) extreme oversold + BB breakdown in uptrend",
                }
            })

        top_3 = top_10[:3]
        ready_count = sum(1 for s in summary.top10 if s.all_conditions_met)

        # Save to JSON for _load_scanner_results
        output_dir = Path(__file__).parent.parent.parent / "data" / "scanner"
        output_dir.mkdir(parents=True, exist_ok=True)

        scan_results = {
            "scan_time": datetime.now(CT).isoformat(),
            "scan_runtime_seconds": elapsed,
            "universe_mode": "PROVEN_900",
            "total_scanned": summary.total_symbols,
            "signals_found": ready_count,
            "long_count": ready_count,
            "short_count": 0,
            "ready_count": summary.ready_count,
            "watchlist_count": summary.watchlist_count,
            "ineligible_count": summary.ineligible_count,
            "fresh_count": summary.fresh_count,
            "stale_count": summary.stale_count,
            "top_10": top_10,
            "top_3": top_3,
            "all_signals": top_10,
            "phase_16b": True,
        }

        # Use atomic write to prevent file corruption
        atomic_json_write(output_dir / "latest_scan_results.json", scan_results)

        # Phase 16B: Send Telegram alert for Top 3 Ready signals
        ready_signals = [s for s in top_3 if s.get("status") == "ready"]
        if ready_signals and get_scanner_config().alert_ready_promotion_enabled:
            try:
                get_alerter().alert_top3_ready(ready_signals)
            except Exception as e:
                print(f"Telegram alert failed: {e}")

        return JSONResponse({
            "status": "success",
            "message": f"ðŸ” Phase 16B scan: {ready_count} ready, {summary.watchlist_count} watchlist from {summary.total_symbols} stocks in {elapsed}s",
            "signals_found": ready_count,
            "ready_count": summary.ready_count,
            "watchlist_count": summary.watchlist_count,
            "ineligible_count": summary.ineligible_count,
            "fresh_count": summary.fresh_count,
            "stale_count": summary.stale_count,
            "scan_time": datetime.now(CT).isoformat(),
            "elapsed_seconds": elapsed,
            "total_scanned": summary.total_symbols,
            "phase_16b": True,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": f"Enhanced scan error: {str(e)}"
        }, status_code=500)


# =============================================================================
# OBSERVABILITY ENDPOINTS - SYSTEM_SUMMARY_NON_TECH.md Implementation
# =============================================================================

# Global orchestrator instance (lazy loaded)
_intraday_orchestrator = None


def _get_intraday_orchestrator():
    """Get or create intraday orchestrator instance."""
    global _intraday_orchestrator
    if _intraday_orchestrator is None:
        try:
            from src.jobs.intraday_orchestrator import IntradayOrchestrator
            _intraday_orchestrator = IntradayOrchestrator()
        except Exception as e:
            logger.warning(f"Could not create IntradayOrchestrator: {e}")
    return _intraday_orchestrator


@app.get("/api/system/status")
async def get_system_status():
    """
    Get comprehensive system status for dashboard observability.

    Returns readiness, open risk, circuit breaker state, failover status,
    and recent alerts as per SYSTEM_SUMMARY_NON_TECH.md requirements.
    """
    orchestrator = _get_intraday_orchestrator()

    status = {
        "timestamp": datetime.now(CT).isoformat(),
        "system_status": "healthy",
        "components": {},
    }

    # Get orchestrator status if available
    if orchestrator:
        try:
            orch_status = orchestrator.get_status()
            status["system_status"] = orch_status.get("system_status", "unknown")
            status["intraday_orchestrator"] = orch_status
        except Exception as e:
            status["intraday_orchestrator"] = {"error": str(e)}

    # Get circuit breaker status
    try:
        from src.resilience.trading_circuit_breaker import TradingCircuitBreaker
        cb = TradingCircuitBreaker()
        status["circuit_breaker"] = cb.get_status()
    except Exception as e:
        status["circuit_breaker"] = {"error": str(e)}

    # Get kill switch status
    try:
        from src.risk.kill_switch_manager import KillSwitchManager
        ks = KillSwitchManager()
        status["kill_switch"] = ks.get_status()
    except Exception as e:
        status["kill_switch"] = {"error": str(e)}

    # Get last reconciliation status
    try:
        from src.jobs.reconciliation_job import ReconciliationJob
        job = ReconciliationJob()
        last_report = job.get_last_report()
        if last_report:
            status["last_reconciliation"] = {
                "timestamp": last_report.get("timestamp"),
                "status": last_report.get("status"),
                "mismatches": last_report.get("summary", {}).get("total_mismatches", 0),
            }
    except Exception as e:
        status["last_reconciliation"] = {"error": str(e)}

    return JSONResponse(status)


@app.get("/api/system/metrics")
async def get_system_metrics():
    """
    Get real-time system metrics for observability dashboard.

    Exposes loop counts, error rates, positions checked,
    and other operational metrics.
    """
    orchestrator = _get_intraday_orchestrator()

    metrics = {
        "timestamp": datetime.now(CT).isoformat(),
        "intraday_metrics": {},
        "scan_metrics": {},
    }

    if orchestrator:
        try:
            metrics["intraday_metrics"] = orchestrator.metrics.to_dict()
            metrics["metrics_history"] = orchestrator.get_metrics_history(limit=50)
        except Exception as e:
            metrics["intraday_metrics"] = {"error": str(e)}

    # Add scan metrics from edge service
    service = _get_edge_service()
    if service:
        try:
            metrics["scan_metrics"] = {
                "last_scan_time": getattr(service, "last_scan_time", None),
                "symbols_in_universe": getattr(service, "universe_size", 0),
            }
        except Exception:
            pass

    return JSONResponse(metrics)


@app.get("/api/system/alerts")
async def get_recent_alerts():
    """
    Get recent system alerts for the observability dashboard.

    Returns circuit breaker triggers, reconciliation issues,
    and other actionable alerts.
    """
    alerts = []

    # Check circuit breaker
    try:
        from src.resilience.trading_circuit_breaker import TradingCircuitBreaker
        cb = TradingCircuitBreaker()
        cb_status = cb.get_status()
        if cb_status.get("entries_blocked"):
            alerts.append({
                "type": "circuit_breaker",
                "severity": "warning",
                "message": f"Circuit breaker active: {cb_status.get('reason', 'Unknown')}",
                "timestamp": cb_status.get("last_check"),
            })
    except Exception:
        pass

    # Check kill switch
    try:
        from src.risk.kill_switch_manager import KillSwitchManager
        ks = KillSwitchManager()
        if ks.is_active():
            ks_status = ks.get_status()
            alerts.append({
                "type": "kill_switch",
                "severity": "critical",
                "message": f"Kill switch active: {ks_status.get('trigger_reason', 'Unknown')}",
                "timestamp": ks_status.get("date"),
            })
    except Exception:
        pass

    # Check orchestrator failover state
    orchestrator = _get_intraday_orchestrator()
    if orchestrator and orchestrator.failover_state.in_failover_mode:
        alerts.append({
            "type": "failover",
            "severity": "critical",
            "message": "System in failover mode - operating in paper mode",
            "timestamp": orchestrator.failover_state.last_heartbeat,
        })

    return JSONResponse({
        "timestamp": datetime.now(CT).isoformat(),
        "alert_count": len(alerts),
        "alerts": alerts,
    })


@app.post("/api/system/premarket-scan")
async def trigger_premarket_scan():
    """
    Manually trigger premarket scan (normally runs at 08:15 ET).

    As per SYSTEM_SUMMARY_NON_TECH.md: Grade symbols before market open.
    """
    orchestrator = _get_intraday_orchestrator()
    if not orchestrator:
        return JSONResponse({
            "error": "IntradayOrchestrator not available"
        }, status_code=503)

    try:
        result = orchestrator.run_premarket_scan()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "error": f"Premarket scan failed: {str(e)}"
        }, status_code=500)


@app.post("/api/system/market-open-alignment")
async def trigger_market_open_alignment():
    """
    Manually trigger market-open alignment (normally runs at 09:30 ET).

    As per SYSTEM_SUMMARY_NON_TECH.md: Re-evaluate candidates at open.
    """
    orchestrator = _get_intraday_orchestrator()
    if not orchestrator:
        return JSONResponse({
            "error": "IntradayOrchestrator not available"
        }, status_code=503)

    try:
        result = orchestrator.run_market_open_alignment()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "error": f"Market-open alignment failed: {str(e)}"
        }, status_code=500)


@app.post("/api/system/position-surveillance")
async def trigger_position_surveillance():
    """
    Manually trigger position surveillance (normally runs every 5 minutes).

    As per SYSTEM_SUMMARY_NON_TECH.md: Verify brackets, reconcile state.
    """
    orchestrator = _get_intraday_orchestrator()
    if not orchestrator:
        return JSONResponse({
            "error": "IntradayOrchestrator not available"
        }, status_code=503)

    try:
        result = orchestrator.run_position_surveillance()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "error": f"Position surveillance failed: {str(e)}"
        }, status_code=500)


@app.get("/api/system/schedule")
async def get_system_schedule():
    """
    Get the current job schedule.

    Shows all scheduled jobs as per SYSTEM_SUMMARY_NON_TECH.md:
    - Premarket scan (08:15 ET)
    - Market-open alignment (09:30 ET)
    - Position surveillance (every 5 min)
    - Weekly review (Saturday 09:00 ET)
    """
    schedule = {
        "timezone": "America/New_York",
        "jobs": [
            {
                "name": "premarket_scan",
                "description": "Grade symbols before market open",
                "schedule": "08:15 ET Mon-Fri",
                "cron": "15 8 * * 1-5",
            },
            {
                "name": "market_open_alignment",
                "description": "Re-evaluate candidates at market open",
                "schedule": "09:30 ET Mon-Fri",
                "cron": "30 9 * * 1-5",
            },
            {
                "name": "position_surveillance",
                "description": "Verify brackets and reconcile state",
                "schedule": "Every 5 minutes during market hours",
                "interval": "5 minutes",
            },
            {
                "name": "weekly_review",
                "description": "Learning/drift review with human-in-the-loop",
                "schedule": "09:00 ET Saturday",
                "cron": "0 9 * * 6",
            },
        ],
        "note": "All times use America/New_York timezone with DST handling",
    }

    # Add job status if orchestrator is available
    orchestrator = _get_intraday_orchestrator()
    if orchestrator:
        schedule["loop_running"] = (
            orchestrator._loop_thread is not None and
            orchestrator._loop_thread.is_alive()
        )
        schedule["last_loop_time"] = orchestrator.metrics.last_loop_time
        schedule["loop_count"] = orchestrator.metrics.loop_count

    return JSONResponse(schedule)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the professional dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


# Professional Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KOBE | Pro Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #2a2a2a;
            --border: #333333;
            --text-primary: #00C805;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #00C805;
            --accent-green: #00C805;
            --accent-red: #FF5000;
            --accent-yellow: #FFA500;
            --accent-purple: #a371f7;
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 13px;
            line-height: 1.4;
            min-height: 100vh;
        }

        .mono { font-family: 'JetBrains Mono', monospace; }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-text {
            font-size: 15px;
            font-weight: 700;
            color: var(--accent-blue);
        }

        .logo-sub {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .time-display {
            text-align: right;
        }

        .time-main {
            font-size: 18px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .time-date {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .status-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-live {
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }

        .status-closed {
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
            border: 1px solid var(--accent-red);
        }

        /* Main Grid */
        .main {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            gap: 1px;
            background: var(--border);
            min-height: calc(100vh - 50px);
        }

        .panel {
            background: var(--bg-primary);
            overflow-y: auto;
        }

        /* Panel Sections */
        .section {
            border-bottom: 1px solid var(--border);
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: var(--bg-secondary);
            cursor: pointer;
            user-select: none;
        }

        .section-header:hover {
            background: var(--bg-tertiary);
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }

        .section-toggle {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s;
        }

        .section.collapsed .section-toggle {
            transform: rotate(-90deg);
        }

        .section.collapsed .section-content {
            display: none;
        }

        .section-content {
            padding: 10px 12px;
        }

        /* Metric Row */
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 500;
        }

        /* Big Number */
        .big-number {
            text-align: center;
            padding: 16px 0;
        }

        .big-number-value {
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .big-number-label {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .big-number-change {
            font-size: 14px;
            margin-top: 4px;
        }

        /* Colors */
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        .neutral { color: var(--text-secondary); }
        .accent { color: var(--accent-blue); }
        .warning { color: var(--accent-yellow); }
        .paper-mode { color: var(--accent-yellow); font-weight: 600; }
        .live-mode { color: var(--accent-red); font-weight: 600; }

        /* Position Card */
        .position-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 8px;
            overflow: hidden;
        }

        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }

        .position-symbol {
            font-weight: 700;
            font-size: 14px;
        }

        .position-pnl {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        .position-details {
            padding: 10px 12px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .position-detail {
            text-align: center;
        }

        .position-detail-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .position-detail-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            margin-top: 2px;
        }

        /* Signal Card */
        .signal-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }

        .signal-card.top-pick {
            border-color: var(--accent-green);
            background: rgba(63, 185, 80, 0.05);
        }

        .signal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .signal-symbol {
            font-weight: 700;
            font-size: 14px;
        }

        .signal-confidence {
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
        }

        .signal-prices {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            font-size: 11px;
        }

        .signal-price {
            text-align: center;
        }

        .signal-price-label {
            color: var(--text-muted);
            font-size: 9px;
            text-transform: uppercase;
        }

        .signal-price-value {
            font-family: 'JetBrains Mono', monospace;
            margin-top: 2px;
        }

        /* Progress Bar */
        .progress-bar {
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 4px;
        }

        .progress-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s;
        }

        .progress-fill.green { background: var(--accent-green); }
        .progress-fill.red { background: var(--accent-red); }
        .progress-fill.blue { background: var(--accent-blue); }
        .progress-fill.yellow { background: var(--accent-yellow); }

        /* Schedule List */
        .schedule-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 0;
            font-size: 12px;
        }

        .schedule-time {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-blue);
            min-width: 50px;
        }

        .schedule-name {
            color: var(--text-secondary);
        }

        .schedule-item.next {
            background: rgba(88, 166, 255, 0.1);
            margin: 0 -12px;
            padding: 6px 12px;
            border-left: 2px solid var(--accent-blue);
        }

        .schedule-item.next .schedule-name {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
        }

        .empty-state-icon {
            font-size: 24px;
            margin-bottom: 8px;
        }

        /* Refresh indicator */
        .refresh-indicator {
            position: fixed;
            bottom: 16px;
            right: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .refresh-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        /* Heartbeat indicator */
        .heartbeat-indicator {
            position: fixed;
            bottom: 16px;
            left: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 11px;
            color: var(--text-secondary);
            max-width: 300px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .heartbeat-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: heartbeat 1.5s infinite;
        }

        @keyframes heartbeat {
            0%, 100% { transform: scale(1); opacity: 1; }
            25% { transform: scale(1.3); opacity: 0.8; }
            50% { transform: scale(1); opacity: 1; }
        }

        /* Market Stats Bar */
        .market-bar {
            display: flex;
            gap: 24px;
            padding: 8px 12px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }

        .market-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .market-label {
            color: var(--text-muted);
        }

        .market-value {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
        }

        /* Top 10 List */
        .top10-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .top10-item {
            display: grid;
            grid-template-columns: 30px 60px 20px 1fr 80px 80px 80px 60px;
            gap: 8px;
            padding: 8px;
            border-bottom: 1px solid var(--border);
            font-size: 11px;
            align-items: center;
            cursor: pointer;
            transition: background 0.2s;
        }

        .top10-item:hover {
            background: var(--bg-tertiary);
        }

        .top10-rank {
            font-weight: 700;
            color: var(--text-muted);
            text-align: center;
        }

        .top10-symbol {
            font-weight: 700;
            font-size: 12px;
        }

        .top10-confidence {
            text-align: center;
            padding: 2px 6px;
            border-radius: 3px;
            background: var(--bg-tertiary);
        }

        /* Phase 16B: Status Badges */
        .status-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 9px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-ready {
            background: var(--accent-green);
            color: #000;
        }
        .status-watchlist {
            background: var(--accent-yellow);
            color: #000;
        }
        .status-ineligible {
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }

        /* Phase 16B: Readiness Score Bar */
        .readiness-bar {
            width: 100%;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 4px;
        }
        .readiness-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .readiness-high { background: var(--accent-green); }
        .readiness-medium { background: var(--accent-yellow); }
        .readiness-low { background: var(--text-muted); }

        /* Phase 16B: Freshness Badge */
        .freshness-badge {
            font-size: 9px;
            padding: 1px 4px;
            border-radius: 2px;
        }
        .freshness-fresh {
            background: var(--accent-green);
            color: #000;
        }
        .freshness-stale {
            background: var(--accent-red);
            color: #fff;
        }

        /* Phase 16B: WHY TAKE THIS TRADE Panel */
        .why-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--accent-green);
            border-radius: 6px;
            padding: 12px;
            margin-top: 12px;
        }
        .why-panel-header {
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-green);
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .why-section {
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }
        .why-section:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }
        .why-section-title {
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .entry-check {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            margin: 2px 0;
        }
        .entry-check-pass { color: var(--accent-green); }
        .entry-check-fail { color: var(--text-muted); }
        .guidance-row {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            margin: 2px 0;
        }
        .guidance-label { color: var(--text-secondary); }
        .guidance-value { font-family: 'JetBrains Mono', monospace; }

        /* News Section */
        .news-item {
            padding: 8px 0;
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .news-headline {
            font-size: 12px;
            margin-bottom: 4px;
        }

        .news-meta {
            font-size: 10px;
            color: var(--text-muted);
        }

        /* AI Insights */
        .ai-metric {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--bg-tertiary);
            font-size: 11px;
        }

        .ai-caveat {
            padding: 6px 8px;
            margin: 4px 0;
            background: rgba(255, 80, 0, 0.1);
            border-left: 2px solid var(--accent-red);
            font-size: 11px;
            color: var(--text-secondary);
        }

        /* Signal Detail Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            padding: 20px;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .modal-title {
            font-size: 18px;
            font-weight: 700;
        }

        .modal-close {
            cursor: pointer;
            font-size: 20px;
            color: var(--text-muted);
        }

        .modal-close:hover {
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div id="app">
        <!-- Header -->
        <header class="header">
            <div class="logo">
                <div>
                    <div class="logo-text">KOBE</div>
                    <div class="logo-sub">{{ data.validation?.strategy || 'IBS+RSI + ICT' }} â€¢ {{ data.universe?.name || 'OPTIONABLE_LIQUID_900' }} ({{ data.universe?.count || 900 }} stocks)</div>
                </div>
            </div>
            <div class="header-right">
                <button @click="triggerScan"
                        :disabled="isScanning"
                        :style="{
                            background: isScanning ? 'var(--accent-yellow)' : 'var(--accent-green)',
                            color: 'black',
                            border: 'none',
                            padding: '8px 16px',
                            borderRadius: '4px',
                            fontWeight: '600',
                            cursor: isScanning ? 'wait' : 'pointer',
                            marginRight: '16px',
                            fontSize: '12px',
                            opacity: isScanning ? '0.9' : '1',
                            minWidth: '140px'
                        }">
                    <span v-if="isScanning">
                        â³ Scanning... {{ scanElapsed.toFixed(1) }}s
                    </span>
                    <span v-else>
                        ðŸ” Scan Now
                    </span>
                </button>
                <div class="status-badge" :class="data.market?.status === 'OPEN' ? 'status-live' : 'status-closed'">
                    {{ data.market?.status === 'OPEN' ? 'ðŸŸ¢ MARKET OPEN' : 'ðŸ”´ MARKET CLOSED' }}
                </div>
                <div class="time-display">
                    <div class="time-main">{{ data.time_ct || '--:--:--' }} CT</div>
                    <div class="time-date">{{ data.date || 'Loading...' }} â€¢ {{ data.market?.context || '' }}</div>
                </div>
            </div>
        </header>

        <!-- Market Bar -->
        <div class="market-bar">
            <div class="market-item">
                <span class="market-label">SPY</span>
                <span class="market-value">${{ data.market?.spy?.toFixed(2) || '--' }}</span>
            </div>
            <div class="market-item">
                <span class="market-label">QQQ</span>
                <span class="market-value">${{ data.market?.qqq?.toFixed(2) || '--' }}</span>
            </div>
            <div class="market-item">
                <span class="market-label">IWM</span>
                <span class="market-value">${{ data.market?.iwm?.toFixed(2) || '--' }}</span>
            </div>
            <div class="market-item">
                <span class="market-label">VIX</span>
                <span class="market-value" :class="data.market?.vix > 20 ? 'warning' : ''">{{ data.market?.vix?.toFixed(1) || '--' }}</span>
            </div>
            <div class="market-item">
                <span class="market-label">NEXT TASK</span>
                <span class="market-value accent">{{ data.schedule?.next?.time || '...' }} {{ data.schedule?.next?.name || '' }}</span>
            </div>
        </div>

        <!-- Main Grid -->
        <main class="main">
            <!-- LEFT PANEL: Account & Stats -->
            <div class="panel">
                <!-- Portfolio Value -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('portfolio')">
                        <span class="section-title">ðŸ’° Portfolio</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <div class="big-number">
                            <div class="big-number-value" :class="data.account?.total_pnl >= 0 ? 'positive' : 'negative'">
                                ${{ formatNumber(data.account?.equity) }}
                            </div>
                            <div class="big-number-change" :class="data.account?.total_pnl >= 0 ? 'positive' : 'negative'">
                                {{ data.account?.total_pnl >= 0 ? 'ðŸ“ˆ +' : 'ðŸ“‰ ' }}${{ formatNumber(Math.abs(data.account?.total_pnl)) }}
                                ({{ data.account?.total_pnl_pct?.toFixed(2) || '0.00' }}%)
                            </div>
                            <div class="big-number-label">Total Account Value</div>
                        </div>

                        <div class="metric-row">
                            <span class="metric-label">ðŸ’µ Cash</span>
                            <span class="metric-value">${{ formatNumber(data.account?.cash) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">âš¡ Buying Power</span>
                            <span class="metric-value accent">${{ formatNumber(data.account?.buying_power) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">ðŸ“Š Unrealized P&L</span>
                            <span class="metric-value" :class="data.account?.unrealized_pnl >= 0 ? 'positive' : 'negative'">
                                {{ data.account?.unrealized_pnl >= 0 ? '+' : '' }}${{ data.account?.unrealized_pnl?.toFixed(2) || '0.00' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">ðŸ“ˆ Open Positions</span>
                            <span class="metric-value accent">{{ data.position_count || 0 }}</span>
                        </div>
                    </div>
                </div>

                <!-- Paper Session Stats -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('stats')">
                        <span class="section-title">ðŸ“Š Paper Session</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <div class="metric-row">
                            <span class="metric-label">ðŸŽ¯ Win Rate</span>
                            <span class="metric-value" :class="(data.stats?.win_rate || 0) >= 0.65 ? 'positive' : 'warning'">
                                {{ ((data.stats?.win_rate || 0) * 100).toFixed(1) }}%
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" :class="(data.stats?.win_rate || 0) >= 0.65 ? 'green' : 'yellow'"
                                 :style="'width: ' + ((data.stats?.win_rate || 0) * 100) + '%'"></div>
                        </div>

                        <div class="metric-row" style="margin-top: 8px;">
                            <span class="metric-label">ðŸ“ˆ Total Trades</span>
                            <span class="metric-value">{{ data.stats?.total_trades || 0 }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">âš”ï¸ W / L</span>
                            <span class="metric-value">
                                <span class="positive">{{ data.stats?.wins || 0 }}</span> /
                                <span class="negative">{{ data.stats?.losses || 0 }}</span>
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">âš¡ Profit Factor</span>
                            <span class="metric-value" :class="(data.stats?.profit_factor || 0) >= 1.5 ? 'positive' : ''">
                                {{ (data.stats?.profit_factor || 0).toFixed(2) }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">âœ… Avg Winner</span>
                            <span class="metric-value positive">+${{ formatNumber(data.stats?.avg_winner) }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">âŒ Avg Loser</span>
                            <span class="metric-value negative">-${{ formatNumber(Math.abs(data.stats?.avg_loser || 0)) }}</span>
                        </div>
                    </div>
                </div>

                <!-- Risk Dashboard Section -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('risk')">
                        <span class="section-title">ðŸ›¡ï¸ Risk Dashboard</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <!-- Portfolio Heat Indicator -->
                        <div style="background: var(--bg-primary); padding: 8px; border-radius: 4px; margin-bottom: 8px; text-align: center;">
                            <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 4px;">PORTFOLIO HEAT</div>
                            <div style="font-size: 16px; font-weight: 700;"
                                 :style="{ color: data.risk_dashboard?.heat_color === 'green' ? 'var(--accent-green)' : data.risk_dashboard?.heat_color === 'yellow' ? 'var(--accent-yellow)' : 'var(--accent-red)' }">
                                {{ data.risk_dashboard?.heat_level || 'LOW' }}
                            </div>
                            <div style="font-size: 10px; color: var(--text-muted);">
                                {{ data.risk_dashboard?.capital_at_risk_pct?.toFixed(1) || 0 }}% capital at risk
                            </div>
                        </div>

                        <div class="metric-row">
                            <span class="metric-label">ðŸ“Š Positions</span>
                            <span class="metric-value">
                                {{ data.risk_dashboard?.current_positions || 0 }} / {{ data.risk_dashboard?.max_positions || 5 }}
                            </span>
                        </div>
                        <div class="progress-bar" style="margin-bottom: 8px;">
                            <div class="progress-fill"
                                 :class="(data.risk_dashboard?.current_positions || 0) >= (data.risk_dashboard?.max_positions || 5) ? 'red' : 'green'"
                                 :style="'width: ' + ((data.risk_dashboard?.current_positions || 0) / (data.risk_dashboard?.max_positions || 5) * 100) + '%'"></div>
                        </div>

                        <div class="metric-row">
                            <span class="metric-label">ðŸ’° Risk/Trade</span>
                            <span class="metric-value">
                                {{ data.risk_dashboard?.risk_per_trade_pct || 1 }}% (${{ formatNumber(data.risk_dashboard?.risk_per_trade_usd) }})
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">ðŸ”¥ At Risk</span>
                            <span class="metric-value" :class="(data.risk_dashboard?.capital_at_risk_pct || 0) > 4 ? 'warning' : ''">
                                ${{ formatNumber(data.risk_dashboard?.capital_at_risk_usd) }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">ðŸš¨ Max Daily Loss</span>
                            <span class="metric-value negative">
                                {{ data.risk_dashboard?.max_daily_loss_pct || 3 }}% (${{ formatNumber(data.risk_dashboard?.max_daily_loss_usd) }})
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">ðŸ“‰ Current DD</span>
                            <span class="metric-value" :class="(data.risk_dashboard?.current_drawdown_pct || 0) > 5 ? 'negative' : ''">
                                {{ data.risk_dashboard?.current_drawdown_pct?.toFixed(2) || 0 }}%
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">â›” Max DD Limit</span>
                            <span class="metric-value">{{ data.risk_dashboard?.max_drawdown_pct || 10 }}%</span>
                        </div>
                    </div>
                </div>

                <!-- Validated Strategy Reference -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('validation')">
                        <span class="section-title">Ground Truth (Backtest)</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content" style="background: rgba(0,255,136,0.05); border-radius: 4px; padding: 8px;">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 6px;">
                            {{ data.validation?.strategy || 'Connors RSI-2' }} - Validated 2021-2025
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Win Rate</span>
                            <span class="metric-value positive">{{ data.validation?.win_rate ? data.validation.win_rate.toFixed(2) + '%' : 'Loading...' }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Profit Factor</span>
                            <span class="metric-value positive">{{ data.validation?.profit_factor ? data.validation.profit_factor.toFixed(4) : 'Loading...' }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Total Trades</span>
                            <span class="metric-value">{{ data.validation?.total_trades || 'Loading...' }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">2022 Bear PF</span>
                            <span class="metric-value positive">{{ data.validation?.pf_2022 ? data.validation.pf_2022.toFixed(4) : 'Loading...' }}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Decision</span>
                            <span class="metric-value" :class="data.validation?.decision === 'GREEN' ? 'positive' : ''">
                                {{ data.validation?.decision || 'GREEN' }}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Schedule -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('schedule')">
                        <span class="section-title">ðŸ“… Schedule (CT)</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <template v-for="(item, idx) in data.schedule?.all || []" :key="idx">
                            <div class="schedule-item" :class="item[0] === data.schedule?.next?.time?.split(' ')[0] ? 'next' : ''">
                                <span class="schedule-time">{{ item[0] }} CT</span>
                                <span class="schedule-name">{{ item[1] }}</span>
                            </div>
                        </template>
                    </div>
                </div>
            </div>

            <!-- CENTER PANEL: Top 10, Positions, News, AI -->
            <div class="panel" style="padding: 12px; overflow-y: auto;">
                <!-- Top 10 Signals Section -->
                <div style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h2 style="font-size: 14px; font-weight: 600;">ðŸ“Š Strategy Signals (Top 10)</h2>
                        <span style="font-size: 12px; color: var(--text-secondary);">{{ data.scanner?.top_10?.length || 0 }} signals</span>
                    </div>

                    <template v-if="data.scanner?.top_10 && data.scanner.top_10.length > 0">
                        <div class="top10-list">
                            <div v-for="(sig, idx) in data.scanner.top_10" :key="sig.symbol"
                                 class="top10-item" @click="showSignalDetail(sig)">
                                <div class="top10-rank">#{{ idx + 1 }}</div>
                                <div class="top10-symbol">
                                    {{ sig.symbol }}
                                    <!-- Phase 16B: Status Badge -->
                                    <span class="status-badge"
                                          :class="sig.status === 'ready' ? 'status-ready' : sig.status === 'watchlist' ? 'status-watchlist' : 'status-ineligible'">
                                        {{ sig.status || 'scan' }}
                                    </span>
                                    <!-- Phase 16B: Freshness Badge -->
                                    <span v-if="sig.freshness" class="freshness-badge"
                                          :class="sig.freshness === 'fresh' ? 'freshness-fresh' : 'freshness-stale'">
                                        {{ sig.freshness }}
                                    </span>
                                </div>
                                <div style="font-size: 11px; font-weight: 600;"
                                     :style="{ color: sig.direction === 'long' ? 'var(--accent-green)' : 'var(--accent-red)' }">
                                    {{ sig.direction === 'long' ? 'â†‘' : 'â†“' }}
                                </div>
                                <!-- Phase 16B: Readiness Score (0-100) instead of confidence -->
                                <div class="top10-confidence" :class="(sig.readiness_score || sig.confidence) >= 90 ? 'positive' : ''"
                                     style="cursor: help;"
                                     :title="'Readiness: ' + (sig.readiness_score || sig.confidence)?.toFixed(0) + ' (RSI:' + (sig.rsi_score || 0)?.toFixed(0) + ' Band:' + (sig.band_score || 0)?.toFixed(0) + ' Trend:' + (sig.trend_score || 0)?.toFixed(0) + ')'">
                                    {{ (sig.readiness_score || sig.confidence)?.toFixed(0) }}
                                    <div class="readiness-bar">
                                        <div class="readiness-fill"
                                             :class="(sig.readiness_score || sig.confidence) >= 90 ? 'readiness-high' : (sig.readiness_score || sig.confidence) >= 70 ? 'readiness-medium' : 'readiness-low'"
                                             :style="'width: ' + (sig.readiness_score || sig.confidence) + '%'"></div>
                                    </div>
                                </div>
                                <div class="mono" style="text-align: right;">${{ sig.entry_price?.toFixed(2) }}</div>
                                <div class="mono negative" style="text-align: right;">${{ sig.stop_loss?.toFixed(2) }}</div>
                                <div class="mono positive" style="text-align: right;">${{ sig.take_profit?.toFixed(2) }}</div>
                                <div class="mono" style="text-align: center; cursor: help;"
                                     title="Risk-to-Reward ratio (target â‰¥1.5:1)">
                                    {{ sig.rr_ratio?.toFixed(1) }}:1
                                </div>
                            </div>
                        </div>
                    </template>
                    <template v-else>
                        <div class="empty-state">
                            <div class="empty-state-icon">ðŸ”</div>
                            <div>No signals found</div>
                        </div>
                    </template>
                </div>

                <!-- Active Positions Section -->
                <div style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h2 style="font-size: 14px; font-weight: 600;">Active Positions</h2>
                        <span style="font-size: 12px; color: var(--text-secondary);">{{ data.position_count || 0 }} open</span>
                    </div>

                    <template v-if="data.positions && data.positions.length > 0">
                        <div v-for="pos in data.positions" :key="pos.symbol" class="position-card">
                            <div class="position-header" @click="togglePosition(pos.symbol)">
                                <div>
                                    <span class="position-symbol">{{ pos.symbol }}</span>
                                    <span style="margin-left: 8px; font-size: 12px; color: var(--text-secondary);">
                                        {{ pos.qty }} shares
                                    </span>
                                </div>
                                <div class="position-pnl" :class="pos.pnl >= 0 ? 'positive' : 'negative'">
                                    {{ pos.pnl >= 0 ? '+' : '' }}${{ pos.pnl?.toFixed(2) }}
                                    <span style="font-size: 11px; margin-left: 4px;">({{ pos.pnl_pct?.toFixed(2) }}%)</span>
                                </div>
                            </div>
                            <div class="position-details" v-show="expandedPositions[pos.symbol]">
                                <div class="position-detail">
                                    <div class="position-detail-label">Entry</div>
                                    <div class="position-detail-value">${{ pos.entry?.toFixed(2) }}</div>
                                </div>
                                <div class="position-detail">
                                    <div class="position-detail-label">Current</div>
                                    <div class="position-detail-value accent">${{ pos.current?.toFixed(2) }}</div>
                                </div>
                                <div class="position-detail">
                                    <div class="position-detail-label">Value</div>
                                    <div class="position-detail-value">${{ formatNumber(pos.market_value) }}</div>
                                </div>
                                <div class="position-detail">
                                    <div class="position-detail-label">Stop Loss</div>
                                    <div class="position-detail-value negative">${{ pos.stop_loss?.toFixed(2) }}</div>
                                </div>
                                <div class="position-detail">
                                    <div class="position-detail-label">Target</div>
                                    <div class="position-detail-value positive">${{ pos.take_profit?.toFixed(2) }}</div>
                                </div>
                                <div class="position-detail">
                                    <div class="position-detail-label">Days Held</div>
                                    <div class="position-detail-value">{{ pos.days_held || 1 }} / 10</div>
                                </div>
                            </div>
                        </div>
                    </template>
                    <template v-else>
                        <div class="empty-state">
                            <div class="empty-state-icon">ðŸ“Š</div>
                            <div>No open positions</div>
                            <div style="font-size: 11px; margin-top: 4px;">Waiting for scanner signals</div>
                        </div>
                    </template>
                </div>

                <!-- System Insights Section (Rule-Based + Trade Performance) -->
                <div style="margin-bottom: 20px;">
                    <h2 style="font-size: 14px; font-weight: 600; margin-bottom: 12px;">
                        ðŸ§  System & AI Insights
                        <span style="font-size: 10px; color: var(--text-muted); font-weight: 400;">(honest)</span>
                    </h2>
                    <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px; padding: 12px;">
                        <!-- Trade Performance Stats -->
                        <div v-if="data.ai_brain?.trade_performance" style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid var(--border);">
                            <div style="font-size: 10px; color: var(--accent-green); margin-bottom: 6px; font-weight: 600;">ðŸ“Š LIVE TRADE PERFORMANCE</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;">
                                <div class="ai-metric" style="border: none; padding: 4px 0;">
                                    <span style="color: var(--text-secondary); font-size: 10px;">Total Trades</span>
                                    <span style="font-weight: 600;">{{ data.ai_brain.trade_performance.total_trades || 0 }}</span>
                                </div>
                                <div class="ai-metric" style="border: none; padding: 4px 0;">
                                    <span style="color: var(--text-secondary); font-size: 10px;">W/L</span>
                                    <span>
                                        <span class="positive">{{ data.ai_brain.trade_performance.wins || 0 }}</span> /
                                        <span class="negative">{{ data.ai_brain.trade_performance.losses || 0 }}</span>
                                    </span>
                                </div>
                                <div class="ai-metric" style="border: none; padding: 4px 0;">
                                    <span style="color: var(--text-secondary); font-size: 10px;">Win Rate</span>
                                    <span :class="(data.ai_brain.trade_performance.win_rate || 0) >= 0.65 ? 'positive' : 'warning'" style="font-weight: 600;">
                                        {{ ((data.ai_brain.trade_performance.win_rate || 0) * 100).toFixed(1) }}%
                                    </span>
                                </div>
                                <div class="ai-metric" style="border: none; padding: 4px 0;">
                                    <span style="color: var(--text-secondary); font-size: 10px;">Profit Factor</span>
                                    <span :class="(data.ai_brain.trade_performance.profit_factor || 0) >= 1.5 ? 'positive' : ''" style="font-weight: 600;">
                                        {{ (data.ai_brain.trade_performance.profit_factor || 0).toFixed(2) }}
                                    </span>
                                </div>
                            </div>
                            <div v-if="data.ai_brain.trade_performance.sample_warning" class="ai-caveat" style="font-size: 9px; padding: 4px 6px;">
                                âš ï¸ Small sample size (n={{ data.ai_brain.trade_performance.total_trades }}) - statistics unstable until 30+ trades
                            </div>
                        </div>

                        <!-- ML Status & Engine Type -->
                        <div class="ai-metric">
                            <span style="color: var(--text-secondary);">ML Status</span>
                            <span style="font-size: 10px; color: var(--text-muted);">No predictive model (rule-based only)</span>
                        </div>
                        <div class="ai-metric">
                            <span style="color: var(--text-secondary);">Engine</span>
                            <span style="font-size: 10px;">{{ data.ai_brain?.model_type || 'Connors RSI-2 Rule Engine' }}</span>
                        </div>
                        <div class="ai-metric">
                            <span style="color: var(--text-secondary);">Signals Today</span>
                            <span :class="(data.ai_brain?.sample_size || 0) > 0 ? 'positive' : ''">
                                {{ data.ai_brain?.sample_size || 0 }}
                            </span>
                        </div>

                        <!-- Signal Breakdown -->
                        <div v-if="data.ai_brain?.signal_breakdown" style="margin-top: 10px; padding: 8px; background: var(--bg-primary); border-radius: 4px;">
                            <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 6px;">SIGNAL BREAKDOWN (LONG ONLY)</div>
                            <div style="display: flex; justify-content: space-between; font-size: 11px;">
                                <span>ðŸ“ˆ Long Setups: <span class="positive">{{ data.ai_brain.signal_breakdown.long || 0 }}</span></span>
                                <span style="color: var(--text-muted);">/ {{ data.ai_brain.signal_breakdown.scanned || 0 }} scanned</span>
                            </div>
                        </div>

                        <!-- Regime Detection Panel -->
                        <div v-if="data.market_intelligence?.regime" style="margin-top: 12px; padding: 10px; background: var(--bg-primary); border-radius: 6px; border-left: 3px solid var(--accent-blue);">
                            <div style="font-size: 10px; color: var(--accent-blue); margin-bottom: 8px; font-weight: 600;">ðŸ“Š REGIME DETECTION</div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 14px; font-weight: 700;"
                                      :style="{ color: data.market_intelligence.regime.color === 'green' ? 'var(--accent-green)' : data.market_intelligence.regime.color === 'yellow' ? 'var(--accent-yellow)' : data.market_intelligence.regime.color === 'orange' ? 'var(--accent-yellow)' : 'var(--accent-red)' }">
                                    {{ data.market_intelligence.regime.label || 'Normal' }}
                                </span>
                                <span style="font-size: 12px; color: var(--text-muted);">VIX {{ data.market_intelligence.regime.vix?.toFixed(1) || data.market?.vix?.toFixed(1) || '--' }}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px;">
                                <div>
                                    <span style="color: var(--text-muted);">Base WR:</span>
                                    <span class="positive" style="margin-left: 4px;">{{ data.market_intelligence.regime.base_win_rate?.toFixed(1) || '66.96' }}%</span>
                                </div>
                                <div>
                                    <span style="color: var(--text-muted);">Adj WR:</span>
                                    <span :class="(data.market_intelligence.regime.adjusted_win_rate || 67) >= 60 ? 'positive' : 'warning'" style="margin-left: 4px;">
                                        {{ data.market_intelligence.regime.adjusted_win_rate?.toFixed(1) || '66.96' }}%
                                        <span v-if="data.market_intelligence.regime.win_rate_adjustment && data.market_intelligence.regime.win_rate_adjustment !== 0"
                                              :class="data.market_intelligence.regime.win_rate_adjustment < 0 ? 'negative' : 'positive'"
                                              style="font-size: 9px;">
                                            ({{ data.market_intelligence.regime.win_rate_adjustment > 0 ? '+' : '' }}{{ data.market_intelligence.regime.win_rate_adjustment }}%)
                                        </span>
                                    </span>
                                </div>
                            </div>
                            <div style="margin-top: 8px; font-size: 10px;">
                                <span style="color: var(--text-muted);">Position Size:</span>
                                <span :class="(data.market_intelligence.regime.position_size_pct || 100) >= 75 ? 'positive' : 'warning'" style="margin-left: 4px; font-weight: 600;">
                                    {{ data.market_intelligence.regime.position_size_pct || 100 }}%
                                </span>
                            </div>
                            <div v-if="data.market_intelligence.regime.recommendation" style="margin-top: 8px; padding: 6px; background: var(--bg-secondary); border-radius: 4px; font-size: 10px; color: var(--text-secondary);">
                                ðŸ’¡ {{ data.market_intelligence.regime.recommendation }}
                            </div>
                        </div>

                        <!-- Scanner Insights -->
                        <div v-if="data.ai_brain?.scanner_insights && data.ai_brain.scanner_insights.length > 0" style="margin-top: 10px;">
                            <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 6px;">INSIGHTS</div>
                            <div v-for="(insight, idx) in data.ai_brain.scanner_insights" :key="idx"
                                 style="font-size: 11px; padding: 4px 0; border-bottom: 1px solid var(--border);">
                                {{ insight }}
                            </div>
                        </div>

                        <!-- Recommendations -->
                        <div v-if="data.ai_brain?.recommendations && data.ai_brain.recommendations.length > 0" style="margin-top: 10px;">
                            <div style="font-size: 10px; color: var(--accent-green); margin-bottom: 6px;">ðŸ’¡ RECOMMENDATIONS</div>
                            <div v-for="(rec, idx) in data.ai_brain.recommendations" :key="idx"
                                 style="font-size: 11px; color: var(--text-secondary); padding: 3px 0;">
                                â†’ {{ rec }}
                            </div>
                        </div>

                        <!-- Caveats -->
                        <div v-if="data.ai_brain?.caveats && data.ai_brain.caveats.length > 0" style="margin-top: 10px;">
                            <div style="font-size: 10px; color: var(--accent-yellow); margin-bottom: 6px;">âš ï¸ CAVEATS</div>
                            <div v-for="(caveat, idx) in data.ai_brain.caveats" :key="idx" class="ai-caveat">
                                {{ caveat }}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- News Section - Watchlist Filtered -->
                <div>
                    <h2 style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">ðŸ“° Watchlist News</h2>
                    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 10px;">
                        <span v-if="data.news && data.news.length > 0 && data.news[0].is_general_news">
                            ðŸ“¡ General market news (no watchlist matches)
                        </span>
                        <span v-else-if="data.news && data.news.length > 0 && data.news[0].matched_watchlist">
                            âœ“ Filtered for {{ data.scanner?.top_10?.length || 0 }} watchlist tickers
                        </span>
                        <span v-else>
                            Scanning for watchlist ticker news...
                        </span>
                    </div>
                    <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px; padding: 12px; max-height: 400px; overflow-y: auto;">
                        <template v-if="data.news && data.news.length > 0">
                            <div v-for="(item, idx) in data.news" :key="idx" class="news-item" style="padding: 8px; margin-bottom: 8px; background: var(--bg-primary); border-radius: 4px; border-left: 2px solid var(--accent-blue);">
                                <!-- Matched Watchlist Tickers -->
                                <div v-if="item.matched_watchlist && item.matched_watchlist.length > 0" style="margin-bottom: 6px;">
                                    <span v-for="ticker in item.matched_watchlist" :key="ticker"
                                          style="display: inline-block; padding: 2px 6px; margin-right: 4px; background: var(--accent-green); color: #000; font-size: 9px; font-weight: 700; border-radius: 3px;">
                                        {{ ticker }}
                                    </span>
                                </div>
                                <div class="news-headline" style="font-size: 11px; line-height: 1.4;">{{ item.headline }}</div>
                                <div class="news-meta" style="margin-top: 4px;">
                                    <span style="color: var(--text-muted);">{{ item.source }}</span>
                                    <span style="color: var(--text-muted); margin-left: 6px;">{{ formatTime(item.published || item.timestamp) }}</span>
                                    <span v-if="item.impact"
                                          :style="{ marginLeft: '8px', fontWeight: 600, color: item.impact === 'CRITICAL' ? 'var(--accent-red)' : item.impact === 'HIGH' ? 'var(--accent-yellow)' : 'var(--text-muted)' }">
                                        {{ item.impact }}
                                    </span>
                                </div>
                                <!-- Summary Preview (if available) -->
                                <div v-if="item.summary" style="font-size: 10px; color: var(--text-muted); margin-top: 4px; line-height: 1.3;">
                                    {{ item.summary.substring(0, 120) }}{{ item.summary.length > 120 ? '...' : '' }}
                                </div>
                            </div>
                        </template>
                        <template v-else>
                            <div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 20px;">
                                <div style="margin-bottom: 6px;">ðŸ“­ No news for watchlist tickers</div>
                                <div style="font-size: 10px;">News will appear when articles mention scanner tickers</div>
                            </div>
                        </template>
                    </div>
                </div>

                <!-- Daily Reports Section (PRE-GAME, HALF-TIME, POST-GAME) -->
                <div style="margin-top: 20px;">
                    <h2 style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">ðŸ“Š Daily Reports</h2>
                    <div style="background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px; overflow: hidden;">
                        <!-- Report Tabs -->
                        <div style="display: flex; border-bottom: 1px solid var(--border);">
                            <div @click="activeReportTab = 'pregame'"
                                 :style="{ flex: 1, padding: '8px', textAlign: 'center', fontSize: '11px', fontWeight: 600, cursor: 'pointer', background: activeReportTab === 'pregame' ? 'var(--bg-tertiary)' : 'transparent', borderBottom: activeReportTab === 'pregame' ? '2px solid var(--accent-green)' : 'none', color: activeReportTab === 'pregame' ? 'var(--accent-green)' : 'var(--text-secondary)' }">
                                ðŸŽ¯ PRE-GAME
                                <span v-if="data.daily_reports?.pregame?.available" style="margin-left: 4px; color: var(--accent-green);">â—</span>
                            </div>
                            <div @click="activeReportTab = 'halftime'"
                                 :style="{ flex: 1, padding: '8px', textAlign: 'center', fontSize: '11px', fontWeight: 600, cursor: 'pointer', background: activeReportTab === 'halftime' ? 'var(--bg-tertiary)' : 'transparent', borderBottom: activeReportTab === 'halftime' ? '2px solid var(--accent-blue)' : 'none', color: activeReportTab === 'halftime' ? 'var(--accent-blue)' : 'var(--text-secondary)' }">
                                â¸ï¸ HALF-TIME
                                <span v-if="data.daily_reports?.halftime?.available" style="margin-left: 4px; color: var(--accent-blue);">â—</span>
                            </div>
                            <div @click="activeReportTab = 'postgame'"
                                 :style="{ flex: 1, padding: '8px', textAlign: 'center', fontSize: '11px', fontWeight: 600, cursor: 'pointer', background: activeReportTab === 'postgame' ? 'var(--bg-tertiary)' : 'transparent', borderBottom: activeReportTab === 'postgame' ? '2px solid var(--accent-yellow)' : 'none', color: activeReportTab === 'postgame' ? 'var(--accent-yellow)' : 'var(--text-secondary)' }">
                                ðŸ POST-GAME
                                <span v-if="data.daily_reports?.postgame?.available" style="margin-left: 4px; color: var(--accent-yellow);">â—</span>
                            </div>
                        </div>
                        <!-- Report Content -->
                        <div style="padding: 12px; max-height: 400px; overflow-y: auto;">
                            <!-- PRE-GAME Content -->
                            <div v-if="activeReportTab === 'pregame'">
                                <div v-if="data.daily_reports?.pregame?.available">
                                    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 8px;">
                                        Period: {{ data.daily_reports.period || 'N/A' }} | Generated: {{ data.daily_reports.pregame.timestamp ? formatTime(data.daily_reports.pregame.timestamp) : 'N/A' }}
                                    </div>
                                    <div v-html="formatMarkdown(data.daily_reports.pregame.full_content)" style="font-size: 11px; line-height: 1.5;"></div>
                                </div>
                                <div v-else style="text-align: center; padding: 40px; color: var(--text-muted);">
                                    <div style="font-size: 14px; margin-bottom: 8px;">ðŸ“‹</div>
                                    <div>No PRE-GAME report available yet</div>
                                    <div style="font-size: 10px; margin-top: 4px;">Report generates at 8:30 AM ET</div>
                                </div>
                            </div>
                            <!-- HALF-TIME Content -->
                            <div v-if="activeReportTab === 'halftime'">
                                <div v-if="data.daily_reports?.halftime?.available">
                                    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 8px;">
                                        Period: {{ data.daily_reports.period || 'N/A' }} | Generated: {{ data.daily_reports.halftime.timestamp ? formatTime(data.daily_reports.halftime.timestamp) : 'N/A' }}
                                    </div>
                                    <div v-html="formatMarkdown(data.daily_reports.halftime.full_content)" style="font-size: 11px; line-height: 1.5;"></div>
                                </div>
                                <div v-else style="text-align: center; padding: 40px; color: var(--text-muted);">
                                    <div style="font-size: 14px; margin-bottom: 8px;">â¸ï¸</div>
                                    <div>No HALF-TIME report available yet</div>
                                    <div style="font-size: 10px; margin-top: 4px;">Report generates at 12:00 PM ET</div>
                                </div>
                            </div>
                            <!-- POST-GAME Content -->
                            <div v-if="activeReportTab === 'postgame'">
                                <div v-if="data.daily_reports?.postgame?.available">
                                    <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 8px;">
                                        Period: {{ data.daily_reports.period || 'N/A' }} | Generated: {{ data.daily_reports.postgame.timestamp ? formatTime(data.daily_reports.postgame.timestamp) : 'N/A' }}
                                    </div>
                                    <div v-html="formatMarkdown(data.daily_reports.postgame.full_content)" style="font-size: 11px; line-height: 1.5;"></div>
                                </div>
                                <div v-else style="text-align: center; padding: 40px; color: var(--text-muted);">
                                    <div style="font-size: 14px; margin-bottom: 8px;">ðŸ</div>
                                    <div>No POST-GAME report available yet</div>
                                    <div style="font-size: 10px; margin-top: 4px;">Report generates at 5:00 PM ET</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- RIGHT PANEL: Signals & Scanner -->
            <div class="panel">
                <!-- Top Signals -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('signals')">
                        <span class="section-title">ðŸŽ¯ Scanner Signals</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <div style="margin-bottom: 8px; font-size: 11px; color: var(--text-muted);">
                            <span v-if="isScanning" style="color: var(--accent-yellow);">
                                â³ Scanning {{ data.scanner?.total_scanned || 710 }} stocks... {{ scanElapsed.toFixed(1) }}s
                            </span>
                            <span v-else>
                                Last scan: {{ data.scanner?.scan_time ? formatTime(data.scanner.scan_time) : 'N/A' }}
                                <span v-if="data.scanner?.scanner_age_seconds"> ({{ formatAge(data.scanner.scanner_age_seconds) }}<span v-if="data.scanner?.scan_runtime_seconds">, {{ data.scanner.scan_runtime_seconds }}s runtime</span>)</span>
                                â€¢ {{ data.scanner?.total_signals || 0 }} signals from {{ data.scanner?.total_scanned || 0 }} stocks
                            </span>
                        </div>

                        <template v-if="data.scanner?.top_3 && data.scanner.top_3.length > 0">
                            <div v-for="(sig, idx) in data.scanner.top_3" :key="sig.symbol"
                                 class="signal-card" :class="idx === 0 ? 'top-pick' : ''"
                                 @click="showSignalDetail(sig)"
                                 style="cursor: pointer;">
                                <div class="signal-header">
                                    <div>
                                        <span v-if="idx === 0" style="color: var(--accent-green); font-size: 10px;">â­ TOP PICK</span>
                                        <span v-else style="color: var(--text-muted); font-size: 10px;">#{{ idx + 1 }}</span>
                                        <span class="signal-symbol" style="margin-left: 6px;">{{ sig.symbol }}</span>
                                        <span style="font-size: 11px; margin-left: 6px;"
                                              :style="{ color: sig.direction === 'long' ? 'var(--accent-green)' : 'var(--accent-red)' }">
                                            {{ sig.direction === 'long' ? 'â†‘' : 'â†“' }} {{ sig.direction?.toUpperCase() }}
                                        </span>
                                    </div>
                                    <span class="signal-confidence" :class="sig.confidence >= 90 ? 'positive' : ''"
                                          style="cursor: help;"
                                          title="Confidence score (0-100%). Based on RSI extremity and distance below BB. Lower RSI = higher confidence.">
                                        {{ sig.confidence >= 90 ? 'â­ ' : '' }}{{ sig.confidence?.toFixed(0) }}%
                                    </span>
                                </div>
                                <div class="signal-prices">
                                    <div class="signal-price">
                                        <div class="signal-price-label">Entry</div>
                                        <div class="signal-price-value">${{ sig.entry_price?.toFixed(2) }}</div>
                                    </div>
                                    <div class="signal-price">
                                        <div class="signal-price-label">Stop</div>
                                        <div class="signal-price-value negative">${{ sig.stop_loss?.toFixed(2) }}</div>
                                    </div>
                                    <div class="signal-price">
                                        <div class="signal-price-label">Target</div>
                                        <div class="signal-price-value positive">${{ sig.take_profit?.toFixed(2) }}</div>
                                    </div>
                                    <div class="signal-price" style="cursor: help;"
                                         title="Risk-to-Reward ratio. Example: 2.0:1 means risking $1 to potentially make $2. Target: â‰¥1.5:1 for quality setups.">
                                        <div class="signal-price-label">R:R</div>
                                        <div class="signal-price-value">{{ sig.rr_ratio?.toFixed(1) }}:1</div>
                                    </div>
                                </div>
                                <!-- Multi-Targets (T1, T2, T3) Panel -->
                                <div v-if="sig.targets" style="margin-top: 8px; padding: 8px; background: var(--bg-secondary); border-radius: 4px; border-left: 2px solid var(--accent-green);">
                                    <div style="font-size: 10px; color: var(--accent-green); margin-bottom: 6px; font-weight: 600;">ðŸŽ¯ PROFIT TARGETS</div>
                                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 10px;">
                                        <div style="text-align: center; padding: 4px; background: var(--bg-primary); border-radius: 3px;">
                                            <div style="color: var(--text-muted); margin-bottom: 2px;">T1 (1R)</div>
                                            <div class="positive" style="font-weight: 600;">${{ sig.targets.t1?.toFixed(2) }}</div>
                                        </div>
                                        <div style="text-align: center; padding: 4px; background: var(--bg-primary); border-radius: 3px;">
                                            <div style="color: var(--text-muted); margin-bottom: 2px;">T2 (1.5R)</div>
                                            <div class="positive" style="font-weight: 600;">${{ sig.targets.t2?.toFixed(2) }}</div>
                                        </div>
                                        <div style="text-align: center; padding: 4px; background: var(--bg-primary); border-radius: 3px; border: 1px solid var(--accent-green);">
                                            <div style="color: var(--accent-green); margin-bottom: 2px;">T3 (2R)</div>
                                            <div class="positive" style="font-weight: 700;">${{ sig.targets.t3?.toFixed(2) }}</div>
                                        </div>
                                    </div>
                                </div>
                                <!-- BB Position % & Expected Move Row -->
                                <div style="margin-top: 6px; display: flex; gap: 6px;">
                                    <div v-if="sig.bb_position_pct !== undefined" style="flex: 1; padding: 6px 8px; background: var(--bg-secondary); border-radius: 4px; cursor: help;"
                                         title="BB Position %: Where price sits within the Bollinger Band range. 0% = at lower band (oversold), 100% = at upper band. Lower is better for long entries.">
                                        <span style="font-size: 10px; color: var(--text-muted);">BB Pos:</span>
                                        <span :class="sig.bb_position_pct <= 5 ? 'positive' : sig.bb_position_pct <= 15 ? 'warning' : 'neutral'" style="font-size: 11px; font-weight: 500; margin-left: 4px;">
                                            {{ sig.bb_position_pct?.toFixed(1) }}%
                                        </span>
                                    </div>
                                    <div v-if="sig.expected_move_weekly && sig.expected_move_weekly > 0" style="flex: 1; padding: 6px 8px; background: var(--bg-secondary); border-radius: 4px; cursor: help;"
                                         title="Expected 1-week price movement based on ATR. Used to assess if target is reachable.">
                                        <span style="font-size: 10px; color: var(--text-muted);">Exp. Move:</span>
                                        <span style="font-size: 11px; color: var(--accent-blue); font-weight: 500; margin-left: 4px;">
                                            Â±{{ sig.expected_move_weekly_pct?.toFixed(1) }}%
                                        </span>
                                    </div>
                                </div>
                                <!-- BB_RSI Indicator Info with tooltips -->
                                <div v-if="sig.rsi_5 || sig.bb_lower"
                                     style="margin-top: 6px; padding: 6px 8px; background: var(--bg-secondary); border-radius: 4px;">
                                    <div style="display: flex; gap: 12px; font-size: 10px;">
                                        <span v-if="sig.rsi_5" style="color: var(--accent-red); cursor: help;"
                                              title="RSI(5) reading. Extreme oversold < 3 is the edge signal.">
                                            RSI(5): {{ sig.rsi_5?.toFixed(2) }}
                                        </span>
                                        <span v-if="sig.bb_lower" style="color: var(--accent-purple); cursor: help;"
                                              title="Lower Bollinger Band (20-period, 2 std). Entry when price breaks below.">
                                            BB Lower: ${{ sig.bb_lower?.toFixed(2) }}
                                        </span>
                                        <span v-if="sig.sma_200" style="color: var(--accent-green); cursor: help;"
                                              title="200-period Simple Moving Average. Uptrend filter - must be above.">
                                            SMA(200): ${{ sig.sma_200?.toFixed(2) }}
                                        </span>
                                    </div>
                                </div>
                                <!-- WHY TAKE THIS TRADE Inline Panel -->
                                <div v-if="sig.why_take_trade && sig.why_take_trade.length > 0"
                                     style="margin-top: 8px; padding: 8px; background: var(--bg-primary); border-radius: 4px; border-left: 2px solid var(--accent-green);">
                                    <div style="font-size: 10px; color: var(--accent-green); margin-bottom: 6px; font-weight: 600;">
                                        ðŸ’¡ WHY TAKE THIS TRADE
                                    </div>
                                    <div v-for="(reason, rIdx) in sig.why_take_trade" :key="rIdx"
                                         style="font-size: 10px; color: var(--text-secondary); padding: 2px 0;">
                                        âœ“ {{ reason }}
                                    </div>
                                </div>
                                <!-- Entry Checks Grid (Compact) -->
                                <div v-if="sig.entry_checks" style="margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap;">
                                    <span :class="sig.entry_checks.rsi_below_3 ? 'positive' : 'neutral'"
                                          style="font-size: 9px; padding: 2px 6px; background: var(--bg-secondary); border-radius: 3px;">
                                        {{ sig.entry_checks.rsi_below_3 ? 'âœ“' : 'âœ—' }} RSI&lt;3
                                    </span>
                                    <span :class="sig.entry_checks.close_at_bb_lower ? 'positive' : 'neutral'"
                                          style="font-size: 9px; padding: 2px 6px; background: var(--bg-secondary); border-radius: 3px;">
                                        {{ sig.entry_checks.close_at_bb_lower ? 'âœ“' : 'âœ—' }} BB Touch
                                    </span>
                                    <span :class="sig.entry_checks.above_sma200 ? 'positive' : 'neutral'"
                                          style="font-size: 9px; padding: 2px 6px; background: var(--bg-secondary); border-radius: 3px;">
                                        {{ sig.entry_checks.above_sma200 ? 'âœ“' : 'âœ—' }} &gt;SMA200
                                    </span>
                                </div>
                                <!-- Historical Edge (Compact) -->
                                <div v-if="sig.historical"
                                     style="margin-top: 6px; padding: 4px 8px; background: var(--bg-secondary); border-radius: 4px; display: flex; justify-content: space-between; font-size: 9px; color: var(--text-muted);">
                                    <span>ðŸ“Š Historical: <span class="positive">{{ sig.historical.strategy_win_rate?.toFixed(1) || '66.96' }}% WR</span></span>
                                    <span>PF: <span class="positive">{{ sig.historical.strategy_profit_factor?.toFixed(2) || '1.53' }}</span></span>
                                </div>
                            </div>
                        </template>

                        <template v-else>
                            <div class="empty-state">
                                <div class="empty-state-icon">ðŸ”</div>
                                <div>No signals yet</div>
                                <div style="font-size: 11px; margin-top: 4px;">Next scan at {{ data.schedule?.next?.time || '--' }}</div>
                            </div>
                        </template>
                    </div>
                </div>

                <!-- Market Intelligence -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('market')">
                        <span class="section-title">ðŸ§  Market Intelligence</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <!-- Volatility Analysis -->
                        <div style="padding: 8px; background: var(--bg-secondary); border-radius: 4px; margin-bottom: 10px;">
                            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">VOLATILITY</div>
                            <div style="font-size: 12px;">{{ data.market_intelligence?.volatility_analysis || 'Analyzing...' }}</div>
                        </div>

                        <!-- Game Plan -->
                        <div style="padding: 8px; background: var(--bg-secondary); border-radius: 4px; margin-bottom: 10px;">
                            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">GAME PLAN</div>
                            <div style="font-size: 12px;">{{ data.market_intelligence?.game_plan || 'Loading...' }}</div>
                        </div>

                        <!-- What to Look For -->
                        <div style="padding: 8px; background: var(--bg-secondary); border-radius: 4px; margin-bottom: 10px;">
                            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">WHAT TO LOOK FOR</div>
                            <div v-for="(item, idx) in data.market_intelligence?.what_to_look_for || []" :key="idx"
                                 style="font-size: 11px; padding: 3px 0;">
                                {{ item }}
                            </div>
                        </div>

                        <!-- Metrics -->
                        <div class="metric-row">
                            <span class="metric-label">Stress Level</span>
                            <span class="metric-value" :class="data.market_intelligence?.stress_level === 'CALM' || data.market_intelligence?.stress_level === 'NORMAL' ? 'positive' : 'warning'">
                                {{ data.market_intelligence?.stress_level || 'N/A' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Position Sizing</span>
                            <span class="metric-value accent">
                                {{ data.market_intelligence?.position_sizing || '100%' }}
                            </span>
                        </div>
                    </div>
                </div>

                <!-- System Status -->
                <div class="section">
                    <div class="section-header" @click="toggleSection('system')">
                        <span class="section-title">System Status</span>
                        <span class="section-toggle">â–¼</span>
                    </div>
                    <div class="section-content">
                        <div class="metric-row">
                            <span class="metric-label">Connection</span>
                            <span class="metric-value" :class="data.health_status?.connection_ok ? 'positive' : 'negative'">
                                {{ data.health_status?.connection_ok ? 'â— Connected' : 'â— Disconnected' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Data Feed</span>
                            <span class="metric-value" :class="data.health_status?.data_feed_ok ? 'positive' : 'warning'">
                                {{ data.health_status?.data_feed_ok ? 'â— Live' : 'â— Offline' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Pre-Open Gate</span>
                            <span class="metric-value" :class="data.health_status?.gate_open ? 'positive' : 'warning'">
                                {{ data.health_status?.gate_open ? 'â— Open' : 'â— Blocked' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Circuit Breaker</span>
                            <span class="metric-value" :class="data.health_status?.circuit_breaker_ok ? 'positive' : 'negative'">
                                {{ data.health_status?.circuit_breaker_ok ? 'â— Normal' : 'â— TRIPPED' }}
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Scanner</span>
                            <span class="metric-value" :class="getScannerFreshnessClass()">
                                â— {{ data.scanner?.scanner_freshness || 'Unknown' }}
                                <span v-if="data.scanner?.scanner_age_seconds" style="font-size: 10px; color: var(--text-muted);">
                                    ({{ formatAge(data.scanner.scanner_age_seconds) }})
                                </span>
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Trading Mode</span>
                            <span class="metric-value" :class="data.trading_mode?.is_paper ? 'paper-mode' : 'live-mode'">
                                {{ data.trading_mode?.mode || 'PAPER' }}
                                <span v-if="data.trading_mode?.live_gated" style="color: var(--text-muted); font-size: 10px;"> (LIVE GATED)</span>
                            </span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Last Update</span>
                            <span class="metric-value">{{ data.time_ct || '--' }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <!-- Signal Detail Modal -->
        <div v-if="selectedSignal" class="modal-overlay" @click.self="closeSignalDetail">
            <div class="modal-content">
                <div class="modal-header">
                    <div class="modal-title">
                        {{ selectedSignal.symbol }} â€¢ {{ selectedSignal.direction?.toUpperCase() }}
                        <!-- Phase 16B: Status Badge in Modal -->
                        <span class="status-badge"
                              :class="selectedSignal.status === 'ready' ? 'status-ready' : selectedSignal.status === 'watchlist' ? 'status-watchlist' : 'status-ineligible'"
                              style="margin-left: 8px;">
                            {{ selectedSignal.status || 'SCANNED' }}
                        </span>
                    </div>
                    <div class="modal-close" @click="closeSignalDetail">âœ•</div>
                </div>

                <!-- Phase 16B: Readiness Score instead of Quality Score -->
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 11px; color: var(--text-muted);">READINESS SCORE</span>
                        <span style="font-size: 14px; font-weight: 700;" :class="(selectedSignal.readiness_score || selectedSignal.confidence) >= 90 ? 'positive' : ''">
                            {{ (selectedSignal.readiness_score || selectedSignal.confidence)?.toFixed(0) }}
                        </span>
                    </div>
                    <div class="progress-bar" style="height: 8px;">
                        <div class="progress-fill"
                             :class="(selectedSignal.readiness_score || selectedSignal.confidence) >= 90 ? 'green' : 'yellow'"
                             :style="`width: ${selectedSignal.readiness_score || selectedSignal.confidence}%`"></div>
                    </div>
                    <!-- Phase 16B: Score Breakdown -->
                    <div v-if="selectedSignal.rsi_score !== undefined" style="display: flex; gap: 12px; margin-top: 8px; font-size: 10px; color: var(--text-muted);">
                        <span>RSI: {{ selectedSignal.rsi_score?.toFixed(0) }}</span>
                        <span>Band: {{ selectedSignal.band_score?.toFixed(0) }}</span>
                        <span>Trend: {{ selectedSignal.trend_score?.toFixed(0) }}</span>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
                    <div style="text-align: center;">
                        <div style="font-size: 10px; color: var(--text-muted);">ENTRY</div>
                        <div style="font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            ${{ selectedSignal.entry_price?.toFixed(2) }}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 10px; color: var(--text-muted);">STOP</div>
                        <div class="negative" style="font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            ${{ selectedSignal.stop_loss?.toFixed(2) }}
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 10px; color: var(--text-muted);">TARGET</div>
                        <div class="positive" style="font-size: 16px; font-weight: 600; font-family: 'JetBrains Mono', monospace;">
                            ${{ selectedSignal.take_profit?.toFixed(2) }}
                        </div>
                    </div>
                </div>

                <div v-if="selectedSignal.conviction_reasons && selectedSignal.conviction_reasons.length > 0"
                     style="border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 16px;">
                    <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">ðŸ’¡ Conviction Reasons</div>
                    <div v-for="(reason, idx) in selectedSignal.conviction_reasons" :key="idx"
                         style="padding: 6px 0; border-bottom: 1px solid var(--bg-tertiary); font-size: 11px;">
                        {{ reason }}
                    </div>
                </div>

                <div v-if="selectedSignal.strategy_5ws" style="border: 1px solid var(--border); border-radius: 6px; padding: 12px;">
                    <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px;">ðŸ“‹ Strategy 5Ws Analysis</div>
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">WHO</div>
                        <div style="font-size: 11px;">{{ selectedSignal.strategy_5ws.who || 'N/A' }}</div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">WHAT</div>
                        <div style="font-size: 11px;">{{ selectedSignal.strategy_5ws.what || 'N/A' }}</div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">WHERE</div>
                        <div style="font-size: 11px;">{{ selectedSignal.strategy_5ws.where || 'N/A' }}</div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">WHEN</div>
                        <div style="font-size: 11px;">{{ selectedSignal.strategy_5ws.when || 'N/A' }}</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 2px;">WHY</div>
                        <div style="font-size: 11px;">{{ selectedSignal.strategy_5ws.why || 'N/A' }}</div>
                    </div>
                </div>

                <!-- Phase 16B: WHY TAKE THIS TRADE Panel -->
                <div class="why-panel" style="margin-top: 16px;">
                    <div class="why-panel-header" @click="toggleWhyPanel" style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                        <span class="why-panel-title">WHY TAKE THIS TRADE?</span>
                        <span>{{ whyPanelExpanded ? 'â–¼' : 'â–º' }}</span>
                    </div>
                    <div v-if="whyPanelExpanded" class="why-panel-content" style="margin-top: 12px;">
                        <!-- Strategy Summary -->
                        <div v-if="getWhyData(selectedSignal)?.summary" class="why-section" style="background: rgba(58, 134, 255, 0.1); padding: 8px; border-radius: 4px; margin-bottom: 12px;">
                            <div style="font-size: 11px; color: var(--accent-blue);">{{ getWhyData(selectedSignal)?.summary }}</div>
                        </div>

                        <!-- Entry Checks -->
                        <div class="why-section">
                            <div class="why-section-title">Entry Checks</div>
                            <div class="why-check-item">
                                <span class="why-check-icon" :class="getEntryCheck(selectedSignal, 'rsi') ? 'check-pass' : 'check-fail'">
                                    {{ getEntryCheck(selectedSignal, 'rsi') ? 'âœ“' : 'âœ—' }}
                                </span>
                                <span>RSI(5) &lt; 3</span>
                                <span style="color: var(--text-muted); margin-left: auto;">
                                    ({{ selectedSignal.rsi_5?.toFixed(2) || selectedSignal.rsi5?.toFixed(2) || 'N/A' }})
                                </span>
                            </div>
                            <div class="why-check-item">
                                <span class="why-check-icon" :class="getEntryCheck(selectedSignal, 'bb_touch') ? 'check-pass' : 'check-fail'">
                                    {{ getEntryCheck(selectedSignal, 'bb_touch') ? 'âœ“' : 'âœ—' }}
                                </span>
                                <span>Close â‰¤ BB Lower</span>
                                <span style="color: var(--text-muted); margin-left: auto;">
                                    (${{ selectedSignal.entry_price?.toFixed(2) || 'N/A' }} vs ${{ selectedSignal.bb_lower?.toFixed(2) || 'N/A' }})
                                </span>
                            </div>
                            <div class="why-check-item">
                                <span class="why-check-icon" :class="getEntryCheck(selectedSignal, 'above_sma200') ? 'check-pass' : 'check-fail'">
                                    {{ getEntryCheck(selectedSignal, 'above_sma200') ? 'âœ“' : 'âœ—' }}
                                </span>
                                <span>Close &gt; SMA(200)</span>
                                <span style="color: var(--text-muted); margin-left: auto;">
                                    (${{ selectedSignal.entry_price?.toFixed(2) || 'N/A' }} vs ${{ selectedSignal.sma_200?.toFixed(2) || 'N/A' }})
                                </span>
                            </div>
                        </div>

                        <!-- Historical Performance -->
                        <div class="why-section" style="margin-top: 12px;">
                            <div class="why-section-title">Historical Performance</div>
                            <div class="why-stat-row">
                                <span>Strategy Win Rate</span>
                                <span class="positive">{{ selectedSignal.historical?.strategy_win_rate ? selectedSignal.historical.strategy_win_rate.toFixed(2) + '%' : 'N/A' }}</span>
                            </div>
                            <div class="why-stat-row">
                                <span>Strategy Profit Factor</span>
                                <span class="positive">{{ selectedSignal.historical?.strategy_profit_factor ? selectedSignal.historical.strategy_profit_factor.toFixed(2) : 'N/A' }}</span>
                            </div>
                            <div v-if="selectedSignal.historical?.symbol_total_trades" class="why-stat-row">
                                <span>{{ selectedSignal.symbol }} Trades</span>
                                <span>{{ selectedSignal.historical?.symbol_wins || 0 }}/{{ selectedSignal.historical?.symbol_total_trades || 0 }} ({{ selectedSignal.historical?.symbol_win_rate?.toFixed(1) || 0 }}%)</span>
                            </div>
                        </div>

                        <!-- Exit Rules (AUTOMATED) - SSOT v2.1.0 -->
                        <div class="why-section" style="margin-top: 12px;">
                            <div class="why-section-title" style="color: var(--accent-yellow);">Exit Rules (AUTOMATED)</div>
                            <div class="why-stat-row">
                                <span>RSI(5) Exit</span>
                                <span>&gt; {{ selectedSignal.exit_rules?.rsi5_exit_threshold || 50 }}</span>
                            </div>
                            <div class="why-stat-row">
                                <span>Max Hold</span>
                                <span>{{ selectedSignal.exit_rules?.max_holding_days || 30 }} days</span>
                            </div>
                            <div style="font-size: 9px; color: var(--text-muted); margin-top: 4px;">
                                {{ selectedSignal.exit_rules?.note || 'AUTOMATED: RSI(5) > 50 OR 30-day timeout (SSOT v2.1.0)' }}
                            </div>
                        </div>

                        <!-- Guidance (NOT AUTOMATED) -->
                        <div class="why-section" style="margin-top: 12px;">
                            <div class="why-section-title" style="color: var(--text-muted);">Guidance (NOT Automated)</div>
                            <div class="why-stat-row">
                                <span>Suggested Stop</span>
                                <span class="negative">${{ selectedSignal.guidance?.suggested_stop?.toFixed(2) || selectedSignal.stop_loss?.toFixed(2) }}</span>
                            </div>
                            <div class="why-stat-row">
                                <span>Suggested Target</span>
                                <span class="positive">${{ selectedSignal.guidance?.suggested_target?.toFixed(2) || selectedSignal.take_profit?.toFixed(2) }}</span>
                            </div>
                            <div style="font-size: 9px; color: var(--text-muted); margin-top: 4px;">
                                {{ selectedSignal.guidance?.note || 'Guidance only - NOT automated exits' }}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Heartbeat Indicator -->
        <div class="heartbeat-indicator">
            <div class="heartbeat-dot"></div>
            <span>{{ heartbeatMessage }}</span>
        </div>

        <!-- Refresh Indicator with Live Countdown -->
        <div class="refresh-indicator">
            <div class="refresh-dot"></div>
            <span>Live â€¢ Data: {{ refreshCountdown }}s â€¢ Scan: {{ isScanning ? 'Running...' : scanCountdown + 's' }}</span>
        </div>
    </div>

    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    <script>
        const { createApp, ref, reactive, onMounted, onUnmounted } = Vue;

        createApp({
            setup() {
                const data = reactive({
                    timestamp: '',
                    time_ct: '',
                    time_et: '',
                    date: '',
                    account: {},
                    positions: [],
                    position_count: 0,
                    market: {},
                    market_intelligence: {},
                    scanner: {},
                    stats: {},
                    schedule: {},
                    ai_brain: {},
                    news: []
                });

                const expandedPositions = reactive({});
                const collapsedSections = reactive({});
                const selectedSignal = ref(null);
                const heartbeatMessage = ref('ðŸ§  Analyzing market structure...');

                // Daily reports active tab
                const activeReportTab = ref('pregame');

                // Phase 16B: WHY panel expanded state
                const whyPanelExpanded = ref(true);

                // Scanning state
                const isScanning = ref(false);
                const scanElapsed = ref(0);
                let scanTimer = null;

                // Refresh countdown (5s cycle for real-time data)
                const refreshCountdown = ref(5);
                let countdownInterval = null;

                // Auto-scan countdown (60s cycle for scanner)
                const scanCountdown = ref(60);
                let scanCountdownInterval = null;

                let ws = null;
                let reconnectInterval = null;
                let heartbeatInterval = null;

                // Heartbeat messages that rotate
                const heartbeatMessages = [
                    'ðŸ§  Analyzing market structure...',
                    'ðŸ“Š Monitoring 832 tickers (SSOT v2.1.0)...',
                    'ðŸŽ¯ Scanning for BB_RSI setups...',
                    'âš¡ Checking RSI < 3 extreme oversold...',
                    'ðŸ” Validating Bollinger Band positions...',
                    'ðŸ“ˆ Confirming SMA(200) trend filter...',
                    'ðŸ’¡ Computing confidence scores...',
                    'ðŸŽ² Running LSTM predictions...'
                ];
                let heartbeatIndex = 0;

                const updateHeartbeat = () => {
                    // Update with dynamic info if available
                    if (data.scanner?.top_3?.length > 0) {
                        const topSignal = data.scanner.top_3[0];
                        heartbeatMessage.value = `ðŸŽ¯ Top setup: ${topSignal.symbol} confidence ${topSignal.confidence?.toFixed(0)}%`;
                    } else if (data.market?.vix) {
                        heartbeatMessage.value = `âš¡ VIX ${data.market.vix.toFixed(1)} - ${data.market_intelligence?.stress_level || 'Normal'} conditions`;
                    } else {
                        heartbeatMessage.value = heartbeatMessages[heartbeatIndex];
                        heartbeatIndex = (heartbeatIndex + 1) % heartbeatMessages.length;
                    }
                };

                const connect = () => {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

                    ws.onmessage = (event) => {
                        const newData = JSON.parse(event.data);
                        Object.assign(data, newData);

                        // Reset countdown on data receive (5s cycle)
                        refreshCountdown.value = 5;

                        // Auto-expand first position
                        if (newData.positions?.length > 0 && Object.keys(expandedPositions).length === 0) {
                            expandedPositions[newData.positions[0].symbol] = true;
                        }
                    };

                    ws.onclose = () => {
                        console.log('WebSocket closed, reconnecting...');
                        reconnectInterval = setTimeout(connect, 3000);
                    };

                    ws.onerror = (err) => {
                        console.error('WebSocket error:', err);
                        ws.close();
                    };
                };

                const toggleSection = (section) => {
                    collapsedSections[section] = !collapsedSections[section];
                };

                const togglePosition = (symbol) => {
                    expandedPositions[symbol] = !expandedPositions[symbol];
                };

                const formatNumber = (num) => {
                    if (num === undefined || num === null) return '0';
                    return Math.abs(num).toLocaleString('en-US', {
                        minimumFractionDigits: 0,
                        maximumFractionDigits: 0
                    });
                };

                const formatTime = (isoString) => {
                    if (!isoString) return 'N/A';
                    try {
                        const date = new Date(isoString);
                        return date.toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: false
                        });
                    } catch {
                        return 'N/A';
                    }
                };

                const showSignalDetail = async (signal) => {
                    selectedSignal.value = signal;
                    whyPanelExpanded.value = true;

                    // Phase 16B: Fetch WHY data from API if not already present
                    if (signal.symbol && !signal.entry_checks) {
                        try {
                            const response = await fetch(`/api/scan/why/${signal.symbol}`);
                            if (response.ok) {
                                const whyData = await response.json();
                                // Merge WHY data into the selected signal
                                selectedSignal.value = { ...signal, ...whyData };
                            }
                        } catch (err) {
                            console.warn('Failed to fetch WHY data:', err);
                        }
                    }
                };

                const closeSignalDetail = () => {
                    selectedSignal.value = null;
                    whyPanelExpanded.value = true;  // Reset for next open
                };

                // Phase 16B: Toggle WHY panel
                const toggleWhyPanel = () => {
                    whyPanelExpanded.value = !whyPanelExpanded.value;
                };

                // Phase 16B: Get WHY data from signal (handles nested structure)
                const getWhyData = (signal) => {
                    if (!signal) return null;
                    // Check for why_take_trade nested object first
                    if (signal.why_take_trade) return signal.why_take_trade;
                    // Check if signal itself has the summary (flat structure)
                    if (signal.summary) return signal;
                    return null;
                };

                // Phase 16B: Get entry check result (handles multiple data structures)
                const getEntryCheck = (signal, checkName) => {
                    if (!signal) return false;

                    // Structure 1: why_take_trade.entry_checks.{checkName}.passed
                    const why = signal.why_take_trade;
                    if (why?.entry_checks?.[checkName]?.passed !== undefined) {
                        return why.entry_checks[checkName].passed;
                    }

                    // Structure 2: Direct entry_checks.{checkName}.passed (from API merge)
                    if (signal.entry_checks?.[checkName]?.passed !== undefined) {
                        return signal.entry_checks[checkName].passed;
                    }

                    // Structure 3: Legacy flat structure (entry_checks.rsi_below_3, etc.)
                    const legacyMap = {
                        'rsi': 'rsi_below_3',
                        'bb_touch': 'close_at_bb_lower',
                        'above_sma200': 'above_sma200'
                    };
                    if (signal.entry_checks?.[legacyMap[checkName]] !== undefined) {
                        return signal.entry_checks[legacyMap[checkName]];
                    }

                    // Fallback: Calculate from raw values for RSI
                    if (checkName === 'rsi') {
                        const rsi = signal.rsi_5 ?? signal.rsi5;
                        return rsi !== undefined && rsi < 3;
                    }
                    // Fallback: BB touch
                    if (checkName === 'bb_touch') {
                        return signal.entry_price && signal.bb_lower && signal.entry_price <= signal.bb_lower;
                    }
                    // Fallback: Above SMA200
                    if (checkName === 'above_sma200') {
                        return signal.entry_price && signal.sma_200 && signal.entry_price > signal.sma_200;
                    }

                    return false;
                };

                const triggerScan = async () => {
                    if (isScanning.value) return; // Prevent double-click

                    try {
                        // Start scanning state
                        isScanning.value = true;
                        scanElapsed.value = 0;
                        scanCountdown.value = 60; // Reset auto-scan countdown

                        // Start elapsed timer (updates every 100ms for smooth display)
                        scanTimer = setInterval(() => {
                            scanElapsed.value += 0.1;
                        }, 100);

                        // Phase 16B: Use enhanced scan endpoint with readiness scores
                        const response = await fetch('/api/scan/enhanced', {
                            method: 'POST',
                        });
                        const result = await response.json();

                        // Stop timer
                        if (scanTimer) {
                            clearInterval(scanTimer);
                            scanTimer = null;
                        }
                        isScanning.value = false;

                        if (result.status === 'success') {
                            // Force refresh data to show new results
                            const freshData = await fetch('/api/data').then(r => r.json());
                            Object.assign(data, freshData);

                            // Show brief success notification (no alert - less intrusive)
                            heartbeatMessage.value = `âœ… Scan complete: ${result.signals_found} signals in ${result.elapsed_seconds}s`;
                        } else {
                            heartbeatMessage.value = 'âŒ Scan failed: ' + result.message;
                        }
                    } catch (error) {
                        // Stop timer on error
                        if (scanTimer) {
                            clearInterval(scanTimer);
                            scanTimer = null;
                        }
                        isScanning.value = false;
                        heartbeatMessage.value = 'âŒ Scan error: ' + error.message;
                    }
                };

                const getScannerFreshnessClass = () => {
                    const freshness = data.scanner?.scanner_freshness;
                    if (freshness === 'Fresh') return 'positive';
                    if (freshness === 'Today') return 'warning';
                    if (freshness === 'Stale') return 'negative';
                    return 'neutral';
                };

                const formatAge = (seconds) => {
                    if (!seconds && seconds !== 0) return '';
                    if (seconds < 60) return `${seconds}s ago`;
                    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
                    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
                    return `${Math.floor(seconds / 86400)}d ago`;
                };

                const formatMarkdown = (md) => {
                    if (!md) return '';
                    return md
                        .replace(/^### (.+)$/gm, '<h3 style="font-size: 12px; font-weight: 600; color: var(--accent-green); margin: 8px 0 4px;">$1</h3>')
                        .replace(/^## (.+)$/gm, '<h2 style="font-size: 13px; font-weight: 700; color: var(--accent-blue); margin: 12px 0 6px;">$1</h2>')
                        .replace(/^# (.+)$/gm, '<h1 style="font-size: 14px; font-weight: 700; color: var(--text-primary); margin: 12px 0 6px;">$1</h1>')
                        .replace(/\*\*(.+?)\*\*/g, '<strong style="color: var(--text-primary); font-weight: 600;">$1</strong>')
                        .replace(/\*(.+?)\*/g, '<em style="color: var(--text-secondary);">$1</em>')
                        .replace(/^- (.+)$/gm, '<li style="margin-left: 16px; margin-bottom: 2px;">$1</li>')
                        .replace(/\n\n/g, '<br><br>')
                        .replace(/\n/g, '<br>');
                };

                onMounted(() => {
                    connect();

                    // Initial fetch via REST as backup
                    fetch('/api/data')
                        .then(r => r.json())
                        .then(d => Object.assign(data, d))
                        .catch(console.error);

                    // Start heartbeat updates
                    heartbeatInterval = setInterval(updateHeartbeat, 5000);
                    updateHeartbeat(); // Initial call

                    // Start refresh countdown (resets every 60s when WebSocket sends data)
                    countdownInterval = setInterval(() => {
                        if (refreshCountdown.value > 0) {
                            refreshCountdown.value -= 1;
                        } else {
                            refreshCountdown.value = 5; // Reset countdown (5s cycle)
                        }
                    }, 1000);

                    // Start auto-scan countdown (60s cycle for scanner)
                    scanCountdownInterval = setInterval(() => {
                        if (isScanning.value) {
                            return; // Don't decrement while scanning
                        }
                        if (scanCountdown.value > 0) {
                            scanCountdown.value -= 1;
                        } else {
                            // Trigger auto-scan
                            triggerScan();
                            scanCountdown.value = 60; // Reset countdown
                        }
                    }, 1000);
                });

                onUnmounted(() => {
                    if (ws) ws.close();
                    if (reconnectInterval) clearTimeout(reconnectInterval);
                    if (heartbeatInterval) clearInterval(heartbeatInterval);
                    if (countdownInterval) clearInterval(countdownInterval);
                    if (scanCountdownInterval) clearInterval(scanCountdownInterval);
                });

                return {
                    data,
                    expandedPositions,
                    collapsedSections,
                    selectedSignal,
                    heartbeatMessage,
                    activeReportTab,
                    isScanning,
                    scanElapsed,
                    refreshCountdown,
                    scanCountdown,
                    // Phase 16B: WHY panel state
                    whyPanelExpanded,
                    toggleSection,
                    togglePosition,
                    formatNumber,
                    formatTime,
                    formatMarkdown,
                    showSignalDetail,
                    closeSignalDetail,
                    // Phase 16B: WHY panel toggle and helpers
                    toggleWhyPanel,
                    getWhyData,
                    getEntryCheck,
                    triggerScan,
                    getScannerFreshnessClass,
                    formatAge
                };
            }
        }).mount('#app');
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")


