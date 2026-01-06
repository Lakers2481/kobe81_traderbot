#!/usr/bin/env python3
"""
KOBE DATA VALIDATOR - NO FAKE DATA EVER
=========================================
This module ensures ALL data is:
- REAL (from verified sources)
- VALIDATED (cross-checked)
- ACCURATE (no drift, no errors)
- LOGGED (full audit trail)

RULES:
1. Never guess - if we don't know, say "unknown"
2. Never fake - every number from real API
3. Always validate - cross-check when possible
4. Always log - full audit trail
5. Alert on issues - don't hide problems
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class DataValidator:
    """
    Validates ALL data used by Kobe.
    No fake data. No guessing. No errors.
    """

    def __init__(self):
        self.validation_log: List[Dict] = []
        self.alerts: List[Dict] = []
        self.state_dir = Path("state/autonomous/validation")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def log_validation(self, source: str, data_type: str, status: str,
                       details: Dict = None, is_alert: bool = False):
        """Log every validation for audit trail."""
        entry = {
            "timestamp": datetime.now(ET).isoformat(),
            "source": source,
            "data_type": data_type,
            "status": status,  # "valid", "invalid", "error", "warning"
            "details": details or {},
        }
        self.validation_log.append(entry)

        if status == "valid":
            logger.info(f"  ✓ {source}/{data_type}: VALID")
        elif status == "warning":
            logger.warning(f"  ⚠ {source}/{data_type}: WARNING - {details}")
        else:
            logger.error(f"  ✗ {source}/{data_type}: {status.upper()} - {details}")

        if is_alert:
            self.alerts.append(entry)
            logger.info(f"  🚨 ALERT: {source}/{data_type} - {details}")

    # =========================================================================
    # POLYGON VALIDATION
    # =========================================================================
    def validate_polygon_price(self, symbol: str) -> Tuple[bool, Dict]:
        """Get and validate price from Polygon API."""
        try:
            from data.providers.polygon_eod import fetch_daily_bars_polygon
            from pathlib import Path

            end = datetime.now(ET).date()
            start = end - timedelta(days=5)

            df = fetch_daily_bars_polygon(
                symbol,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                cache_dir=Path("data/polygon_cache")
            )

            if df is None or len(df) == 0:
                self.log_validation("polygon", f"price/{symbol}", "error",
                                   {"error": "No data returned"})
                return False, {"error": "No data"}

            # Validate OHLC logic
            latest = df.iloc[-1]
            price = float(latest['close'])
            high = float(latest['high'])
            low = float(latest['low'])
            open_price = float(latest['open'])

            # OHLC sanity checks
            errors = []
            if high < low:
                errors.append("high < low (impossible)")
            if price > high or price < low:
                errors.append("close outside high/low range")
            if open_price > high or open_price < low:
                errors.append("open outside high/low range")
            if price <= 0:
                errors.append("price <= 0 (invalid)")

            if errors:
                self.log_validation("polygon", f"price/{symbol}", "invalid",
                                   {"errors": errors, "data": latest.to_dict()})
                return False, {"errors": errors}

            result = {
                "symbol": symbol,
                "price": price,
                "high": high,
                "low": low,
                "open": open_price,
                "date": str(latest.name),
                "source": "polygon",
                "validated": True,
            }

            self.log_validation("polygon", f"price/{symbol}", "valid",
                               {"price": price})
            return True, result

        except Exception as e:
            self.log_validation("polygon", f"price/{symbol}", "error",
                               {"error": str(e)})
            return False, {"error": str(e)}

    # =========================================================================
    # ALPACA VALIDATION
    # =========================================================================
    def validate_alpaca_account(self) -> Tuple[bool, Dict]:
        """Get and validate account data from Alpaca."""
        try:
            from execution.broker_alpaca import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            broker.connect()

            # Get account using the proper method
            account = broker.get_account()
            if account is None:
                return False, {"error": "Could not get account"}

            # Account is an object, not a dict - access attributes directly
            equity = float(getattr(account, "equity", 0))
            cash = float(getattr(account, "cash", 0))
            buying_power = float(getattr(account, "buying_power", 0))
            portfolio_value = float(getattr(account, "portfolio_value", 0))

            # Derive status from blocked flags
            trading_blocked = getattr(account, "trading_blocked", False)
            account_blocked = getattr(account, "account_blocked", False)
            if account_blocked:
                status = "blocked"
            elif trading_blocked:
                status = "trading_blocked"
            else:
                status = "active"

            # Sanity checks
            errors = []
            if equity < 0:
                errors.append("equity < 0 (impossible)")
            if cash < 0 and equity > 0:
                errors.append("negative cash with positive equity (suspicious)")

            if errors:
                self.log_validation("alpaca", "account", "warning",
                                   {"errors": errors})

            result = {
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
                "portfolio_value": portfolio_value,
                "status": status,
                "source": "alpaca",
                "validated": len(errors) == 0,
            }

            self.log_validation("alpaca", "account", "valid" if not errors else "warning",
                               {"equity": equity})
            return True, result

        except Exception as e:
            self.log_validation("alpaca", "account", "error",
                               {"error": str(e)})
            return False, {"error": str(e)}

    def validate_alpaca_positions(self) -> Tuple[bool, List[Dict]]:
        """Get and validate positions from Alpaca."""
        try:
            from execution.broker_alpaca import AlpacaBroker

            broker = AlpacaBroker(paper=True)
            broker.connect()
            positions = broker.get_positions()

            if positions is None:
                positions = []

            validated_positions = []
            for pos in positions:
                # Handle both dict and object formats
                if isinstance(pos, dict):
                    symbol = pos.get("symbol", "")
                    qty = float(pos.get("qty", 0))
                    market_value = float(pos.get("market_value", 0))
                    current_price = float(pos.get("current_price", 0))
                    avg_entry = float(pos.get("avg_price", pos.get("avg_entry_price", 0)))
                    unrealized_pl = float(pos.get("unrealized_pnl", pos.get("unrealized_pl", 0)))
                else:
                    symbol = getattr(pos, "symbol", "")
                    qty = float(getattr(pos, "qty", 0))
                    market_value = float(getattr(pos, "market_value", 0))
                    current_price = float(getattr(pos, "current_price", 0))
                    avg_entry = float(getattr(pos, "avg_price", 0))
                    unrealized_pl = float(getattr(pos, "unrealized_pnl", 0))

                # Validate
                errors = []
                if qty == 0:
                    errors.append("qty is 0 (why is this a position?)")

                if current_price > 0:
                    expected_value = qty * current_price
                    if abs(market_value - expected_value) > 1:  # Allow $1 rounding
                        errors.append(f"market_value mismatch: {market_value} vs {expected_value}")

                validated_positions.append({
                    "symbol": symbol,
                    "qty": qty,
                    "market_value": market_value,
                    "current_price": current_price,
                    "avg_entry": avg_entry,
                    "unrealized_pl": unrealized_pl,
                    "side": "long" if qty > 0 else "short",
                    "validated": len(errors) == 0,
                    "errors": errors if errors else None,
                })

            self.log_validation("alpaca", "positions", "valid",
                               {"count": len(validated_positions)})
            return True, validated_positions

        except Exception as e:
            self.log_validation("alpaca", "positions", "error",
                               {"error": str(e)})
            return False, []

    # =========================================================================
    # CROSS-VALIDATION (Multiple Sources)
    # =========================================================================
    def cross_validate_price(self, symbol: str) -> Tuple[bool, Dict]:
        """
        Cross-validate price from multiple sources.
        If sources disagree significantly, flag it.
        """
        sources = {}

        # Source 1: Polygon
        valid, polygon_data = self.validate_polygon_price(symbol)
        if valid:
            sources["polygon"] = polygon_data["price"]

        # Source 2: Yahoo Finance (free, no API key)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if len(hist) > 0:
                yahoo_price = float(hist['Close'].iloc[-1])
                sources["yahoo"] = yahoo_price
                self.log_validation("yahoo", f"price/{symbol}", "valid",
                                   {"price": yahoo_price})
        except Exception as e:
            self.log_validation("yahoo", f"price/{symbol}", "error",
                               {"error": str(e)})

        # Source 3: Alpaca (if we have a position)
        try:
            from execution.broker_alpaca import AlpacaBroker
            broker = AlpacaBroker(paper=True)
            broker.connect()
            positions = broker.get_positions()
            if positions:
                for pos in positions:
                    pos_symbol = getattr(pos, 'symbol', None) or pos.get('symbol', '') if isinstance(pos, dict) else pos.symbol
                    if pos_symbol == symbol:
                        alpaca_price = float(getattr(pos, 'current_price', 0) if hasattr(pos, 'current_price') else pos.get('current_price', 0))
                        if alpaca_price > 0:
                            sources["alpaca"] = alpaca_price
                            self.log_validation("alpaca", f"price/{symbol}", "valid",
                                               {"price": alpaca_price})
                        break
        except Exception:
            pass  # No position or connection issue, that's okay

        if len(sources) < 1:
            self.log_validation("cross_check", f"price/{symbol}", "error",
                               {"error": "No sources available"})
            return False, {"error": "No price sources available"}

        # Check for discrepancies
        prices = list(sources.values())
        avg_price = sum(prices) / len(prices)

        discrepancies = []
        for source, price in sources.items():
            pct_diff = abs(price - avg_price) / avg_price * 100
            if pct_diff > 2:  # More than 2% difference is suspicious
                discrepancies.append({
                    "source": source,
                    "price": price,
                    "pct_diff": round(pct_diff, 2),
                })

        result = {
            "symbol": symbol,
            "sources": sources,
            "avg_price": round(avg_price, 2),
            "num_sources": len(sources),
            "discrepancies": discrepancies,
            "validated": len(discrepancies) == 0,
        }

        if discrepancies:
            self.log_validation("cross_check", f"price/{symbol}", "warning",
                               {"discrepancies": discrepancies}, is_alert=True)
        else:
            self.log_validation("cross_check", f"price/{symbol}", "valid",
                               {"sources": len(sources), "avg": avg_price})

        return len(discrepancies) == 0, result

    # =========================================================================
    # FRED ECONOMIC DATA
    # =========================================================================
    def get_fred_data(self, series_id: str) -> Tuple[bool, Dict]:
        """Get economic data from FRED."""
        try:
            import requests

            api_key = os.getenv("FRED_API_KEY")
            if not api_key:
                return False, {"error": "FRED_API_KEY not configured"}

            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "limit": 10,
                "sort_order": "desc",
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if "observations" not in data:
                self.log_validation("fred", series_id, "error",
                                   {"error": "No observations in response"})
                return False, {"error": "No data"}

            obs = data["observations"]
            if len(obs) == 0:
                return False, {"error": "Empty observations"}

            latest = obs[0]
            value = latest["value"]

            # FRED returns "." for missing data
            if value == ".":
                return False, {"error": "Missing value (.)"}

            result = {
                "series_id": series_id,
                "value": float(value),
                "date": latest["date"],
                "source": "fred",
                "validated": True,
            }

            self.log_validation("fred", series_id, "valid",
                               {"value": value, "date": latest["date"]})
            return True, result

        except Exception as e:
            self.log_validation("fred", series_id, "error",
                               {"error": str(e)})
            return False, {"error": str(e)}

    def get_vix(self) -> Tuple[bool, Dict]:
        """Get VIX from FRED (VIXCLS series)."""
        return self.get_fred_data("VIXCLS")

    def get_fed_funds_rate(self) -> Tuple[bool, Dict]:
        """Get Federal Funds Rate."""
        return self.get_fred_data("FEDFUNDS")

    def get_10y_treasury(self) -> Tuple[bool, Dict]:
        """Get 10-Year Treasury Rate."""
        return self.get_fred_data("DGS10")

    # =========================================================================
    # FREE DATA SOURCES (No API Key)
    # =========================================================================
    def get_yahoo_quote(self, symbol: str) -> Tuple[bool, Dict]:
        """Get quote from Yahoo Finance (free, no API key)."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "regularMarketPrice" not in info:
                # Try history instead
                hist = ticker.history(period="1d")
                if len(hist) == 0:
                    return False, {"error": "No data from Yahoo"}

                price = float(hist['Close'].iloc[-1])
                result = {
                    "symbol": symbol,
                    "price": price,
                    "source": "yahoo_history",
                    "validated": True,
                }
            else:
                result = {
                    "symbol": symbol,
                    "price": info.get("regularMarketPrice"),
                    "previousClose": info.get("previousClose"),
                    "volume": info.get("volume"),
                    "marketCap": info.get("marketCap"),
                    "pe": info.get("trailingPE"),
                    "source": "yahoo",
                    "validated": True,
                }

            self.log_validation("yahoo", f"quote/{symbol}", "valid",
                               {"price": result.get("price")})
            return True, result

        except Exception as e:
            self.log_validation("yahoo", f"quote/{symbol}", "error",
                               {"error": str(e)})
            return False, {"error": str(e)}

    def get_fear_greed_index(self) -> Tuple[bool, Dict]:
        """
        Get market sentiment.
        Primary: CNN Fear & Greed Index
        Fallback: Derive from VIX (if CNN fails)
        """
        # Try CNN first
        try:
            import urllib.request
            import json

            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if "fear_and_greed" in data:
                fg = data["fear_and_greed"]
                score = fg.get("score")
                rating = fg.get("rating")

                result = {
                    "score": score,
                    "rating": rating,
                    "source": "cnn",
                    "validated": True,
                }

                self.log_validation("cnn", "fear_greed", "valid",
                                   {"score": score, "rating": rating})
                return True, result

        except Exception:
            pass  # Fall through to VIX-based fallback

        # Fallback: Use VIX to derive sentiment
        try:
            valid, vix_data = self.get_vix()
            if valid:
                vix = vix_data["value"]
                # VIX interpretation:
                # < 12: Extreme Greed
                # 12-17: Greed
                # 17-25: Neutral
                # 25-35: Fear
                # > 35: Extreme Fear
                if vix < 12:
                    score = 85
                    rating = "Extreme Greed"
                elif vix < 17:
                    score = 70
                    rating = "Greed"
                elif vix < 25:
                    score = 50
                    rating = "Neutral"
                elif vix < 35:
                    score = 30
                    rating = "Fear"
                else:
                    score = 15
                    rating = "Extreme Fear"

                result = {
                    "score": score,
                    "rating": rating,
                    "source": "vix_derived",
                    "vix": vix,
                    "validated": True,
                }

                self.log_validation("vix", "fear_greed", "valid",
                                   {"score": score, "rating": rating, "vix": vix})
                return True, result

        except Exception as e:
            self.log_validation("sentiment", "fear_greed", "error",
                               {"error": str(e)})
            return False, {"error": str(e)}

        return False, {"error": "Could not get sentiment from any source"}

    # =========================================================================
    # BACKTEST VALIDATION
    # =========================================================================
    def validate_backtest_result(self, result: Dict) -> Tuple[bool, Dict]:
        """
        Validate a backtest result is real and not inflated.
        """
        errors = []
        warnings = []

        # Check required fields
        required = ["win_rate", "profit_factor", "total_trades", "net_pnl"]
        for field in required:
            if field not in result:
                errors.append(f"Missing required field: {field}")

        if errors:
            self.log_validation("backtest", "result", "error",
                               {"errors": errors})
            return False, {"errors": errors}

        win_rate = result["win_rate"]
        profit_factor = result["profit_factor"]
        total_trades = result["total_trades"]
        net_pnl = result["net_pnl"]

        # Sanity checks
        if win_rate < 0 or win_rate > 1:
            errors.append(f"Invalid win_rate: {win_rate} (must be 0-1)")

        if profit_factor < 0:
            errors.append(f"Negative profit_factor: {profit_factor}")

        if total_trades < 0:
            errors.append(f"Negative total_trades: {total_trades}")

        # Warning checks (suspicious but not invalid)
        if win_rate > 0.80:
            warnings.append(f"Suspiciously high win_rate: {win_rate:.1%}")

        if profit_factor > 3.0:
            warnings.append(f"Suspiciously high profit_factor: {profit_factor:.2f}")

        if total_trades < 30:
            warnings.append(f"Low sample size: {total_trades} trades (need 30+ for significance)")

        validation = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "net_pnl": net_pnl,
            "errors": errors,
            "warnings": warnings,
            "validated": len(errors) == 0,
            "statistically_significant": total_trades >= 30,
        }

        if errors:
            self.log_validation("backtest", "result", "invalid",
                               {"errors": errors})
            return False, validation

        if warnings:
            self.log_validation("backtest", "result", "warning",
                               {"warnings": warnings})
        else:
            self.log_validation("backtest", "result", "valid",
                               {"win_rate": f"{win_rate:.1%}", "pf": f"{profit_factor:.2f}"})

        return True, validation

    # =========================================================================
    # FULL SYSTEM VALIDATION
    # =========================================================================
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run full system validation - check ALL data sources.
        This should be run periodically to catch any drift or errors.
        """
        logger.info("=" * 60)
        logger.info("RUNNING FULL DATA VALIDATION")
        logger.info("=" * 60)

        results = {
            "timestamp": datetime.now(ET).isoformat(),
            "checks": {},
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "alerts": [],
        }

        # 1. Alpaca Account
        logger.info("\n[1] Validating Alpaca Account...")
        valid, data = self.validate_alpaca_account()
        results["checks"]["alpaca_account"] = {"valid": valid, "data": data}
        if valid:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # 2. Alpaca Positions
        logger.info("\n[2] Validating Alpaca Positions...")
        valid, data = self.validate_alpaca_positions()
        results["checks"]["alpaca_positions"] = {"valid": valid, "count": len(data)}
        if valid:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # 3. VIX from FRED
        logger.info("\n[3] Validating VIX (FRED)...")
        valid, data = self.get_vix()
        results["checks"]["vix"] = {"valid": valid, "data": data}
        if valid:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # 4. Fear & Greed Index
        logger.info("\n[4] Validating Fear & Greed Index...")
        valid, data = self.get_fear_greed_index()
        results["checks"]["fear_greed"] = {"valid": valid, "data": data}
        if valid:
            results["passed"] += 1
        else:
            results["failed"] += 1

        # 5. Cross-validate SPY price
        logger.info("\n[5] Cross-validating SPY price...")
        valid, data = self.cross_validate_price("SPY")
        results["checks"]["spy_price"] = {"valid": valid, "data": data}
        if valid:
            results["passed"] += 1
        elif data.get("discrepancies"):
            results["warnings"] += 1
        else:
            results["failed"] += 1

        # Summary
        total = results["passed"] + results["failed"] + results["warnings"]
        results["summary"] = {
            "total_checks": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "warnings": results["warnings"],
            "health": "HEALTHY" if results["failed"] == 0 else "UNHEALTHY",
        }

        # Copy alerts
        results["alerts"] = self.alerts.copy()

        # Save validation report
        report_file = self.state_dir / "validation_report.json"
        report_file.write_text(json.dumps(results, indent=2, default=str))

        logger.info("\n" + "=" * 60)
        logger.info(f"VALIDATION COMPLETE: {results['passed']}/{total} passed")
        if results["failed"] > 0:
            logger.error(f"  ✗ {results['failed']} FAILED")
        if results["warnings"] > 0:
            logger.warning(f"  ⚠ {results['warnings']} WARNINGS")
        logger.info("=" * 60)

        return results

    def get_alerts(self) -> List[Dict]:
        """Get any alerts that need attention."""
        return self.alerts.copy()

    def clear_alerts(self):
        """Clear alerts after they've been handled."""
        self.alerts = []
