#!/usr/bin/env python3
"""
Kobe Signal Explainer (Deprecated)

This script previously explained RSI2/IBS signals. The system is now
standardized to two strategies: IBS+RSI and ICT Turtle Soup.

For step-by-step reasoning on current strategies, use:
  python scripts/debugger.py --strategy ibs_rsi --symbol AAPL ...
  python scripts/debugger.py --strategy turtle_soup --symbol AAPL ...
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.env_loader import load_env


def fetch_price_data(symbol: str, start: str, end: str) -> Optional[Any]:
    """Fetch price data for analysis."""
    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        return fetch_daily_bars_polygon(symbol, start, end)
    except ImportError:
        return None


def calculate_rsi(prices: List[float], period: int = 2) -> List[Optional[float]]:
    """Calculate RSI values."""
    if len(prices) < period + 1:
        return [None] * len(prices)

    rsi_values: List[Optional[float]] = [None] * period
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

        if i >= period:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

    return rsi_values


def calculate_ibs(highs: List[float], lows: List[float], closes: List[float]) -> List[Optional[float]]:
    """Calculate IBS (Internal Bar Strength) values."""
    ibs_values = []
    for i in range(len(closes)):
        high_low_range = highs[i] - lows[i]
        if high_low_range > 0:
            ibs = (closes[i] - lows[i]) / high_low_range
        else:
            ibs = 0.5  # Neutral if no range
        ibs_values.append(ibs)
    return ibs_values


def explain_rsi2_signal(
    df: Any,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explain Connors RSI(2) signal for a specific date.
    RSI(2) typically signals:
    - Long when RSI(2) < 10 (oversold)
    - Short when RSI(2) > 90 (overbought)
    """
    if df is None or df.empty:
        return {"error": "No price data available"}

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Find target date row
    if target_date:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        mask = df["timestamp"].dt.date == target_dt
        if not mask.any():
            # Find closest date
            df["date_diff"] = abs(df["timestamp"].dt.date.apply(lambda x: (x - target_dt).days))
            idx = df["date_diff"].idxmin()
        else:
            idx = mask.idxmax()
    else:
        idx = len(df) - 1  # Latest

    # Calculate RSI(2)
    closes = df["close"].tolist()
    rsi_values = calculate_rsi(closes, period=2)

    # Get context window
    start_idx = max(0, idx - 5)
    end_idx = min(len(df), idx + 2)

    context = []
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        rsi_val = rsi_values[i] if i < len(rsi_values) else None
        is_target = i == idx
        context.append({
            "date": str(row["timestamp"].date()) if hasattr(row["timestamp"], "date") else str(row["timestamp"])[:10],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "rsi2": round(rsi_val, 2) if rsi_val else None,
            "is_signal_date": is_target,
        })

    target_row = df.iloc[idx]
    target_rsi = rsi_values[idx] if idx < len(rsi_values) else None

    # Determine signal
    signal = "NONE"
    reason = "RSI(2) in neutral zone"
    if target_rsi is not None:
        if target_rsi < 10:
            signal = "LONG"
            reason = f"RSI(2) = {target_rsi:.2f} < 10 (oversold, expect mean reversion bounce)"
        elif target_rsi > 90:
            signal = "SHORT"
            reason = f"RSI(2) = {target_rsi:.2f} > 90 (overbought, expect pullback)"
        else:
            reason = f"RSI(2) = {target_rsi:.2f} between 10-90 (no signal)"

    # Calculate potential entry/stop
    entry_price = float(target_row["close"])
    stop_loss = entry_price * 0.97 if signal == "LONG" else entry_price * 1.03
    target_price = entry_price * 1.03 if signal == "LONG" else entry_price * 0.97

    return {
        "strategy": "connors_rsi2",
        "symbol": str(target_row.get("symbol", "UNKNOWN")),
        "date": str(target_row["timestamp"].date()) if hasattr(target_row["timestamp"], "date") else str(target_row["timestamp"])[:10],
        "signal": signal,
        "reason": reason,
        "indicators": {
            "rsi2": round(target_rsi, 2) if target_rsi else None,
            "close": entry_price,
        },
        "suggested_trade": {
            "entry": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target_price, 2),
            "risk_reward": "1:1",
        } if signal != "NONE" else None,
        "context": context,
    }


def explain_ibs_signal(
    df: Any,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Explain IBS signal for a specific date.
    IBS signals:
    - Long when IBS < 0.2 (closed near low)
    - Short when IBS > 0.8 (closed near high)
    """
    if df is None or df.empty:
        return {"error": "No price data available"}

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Find target date row
    if target_date:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        mask = df["timestamp"].dt.date == target_dt
        if not mask.any():
            df["date_diff"] = abs(df["timestamp"].dt.date.apply(lambda x: (x - target_dt).days))
            idx = df["date_diff"].idxmin()
        else:
            idx = mask.idxmax()
    else:
        idx = len(df) - 1

    # Calculate IBS
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    ibs_values = calculate_ibs(highs, lows, closes)

    # Get context window
    start_idx = max(0, idx - 5)
    end_idx = min(len(df), idx + 2)

    context = []
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        ibs_val = ibs_values[i] if i < len(ibs_values) else None
        is_target = i == idx
        context.append({
            "date": str(row["timestamp"].date()) if hasattr(row["timestamp"], "date") else str(row["timestamp"])[:10],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "ibs": round(ibs_val, 3) if ibs_val else None,
            "is_signal_date": is_target,
        })

    target_row = df.iloc[idx]
    target_ibs = ibs_values[idx] if idx < len(ibs_values) else None

    # Determine signal
    signal = "NONE"
    reason = "IBS in neutral zone"
    if target_ibs is not None:
        if target_ibs < 0.2:
            signal = "LONG"
            reason = f"IBS = {target_ibs:.3f} < 0.2 (closed near low, expect bounce)"
        elif target_ibs > 0.8:
            signal = "SHORT"
            reason = f"IBS = {target_ibs:.3f} > 0.8 (closed near high, expect pullback)"
        else:
            reason = f"IBS = {target_ibs:.3f} between 0.2-0.8 (no signal)"

    entry_price = float(target_row["close"])
    stop_loss = entry_price * 0.97 if signal == "LONG" else entry_price * 1.03
    target_price = entry_price * 1.02 if signal == "LONG" else entry_price * 0.98

    return {
        "strategy": "ibs",
        "symbol": str(target_row.get("symbol", "UNKNOWN")),
        "date": str(target_row["timestamp"].date()) if hasattr(target_row["timestamp"], "date") else str(target_row["timestamp"])[:10],
        "signal": signal,
        "reason": reason,
        "indicators": {
            "ibs": round(target_ibs, 3) if target_ibs else None,
            "high": float(target_row["high"]),
            "low": float(target_row["low"]),
            "close": entry_price,
        },
        "suggested_trade": {
            "entry": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target": round(target_price, 2),
            "risk_reward": "1.5:1",
        } if signal != "NONE" else None,
        "context": context,
    }


def print_explanation(explanation: Dict[str, Any]) -> None:
    """Pretty-print signal explanation."""
    if "error" in explanation:
        print(f"Error: {explanation['error']}")
        return

    print(f"\n{'=' * 70}")
    print(f"SIGNAL EXPLANATION: {explanation['symbol']}")
    print(f"{'=' * 70}")

    print(f"\nStrategy: {explanation['strategy'].upper()}")
    print(f"Date: {explanation['date']}")
    print(f"Signal: {explanation['signal']}")
    print(f"\nReason: {explanation['reason']}")

    print(f"\nIndicator Values:")
    for key, val in explanation.get("indicators", {}).items():
        print(f"  {key}: {val}")

    if explanation.get("suggested_trade"):
        trade = explanation["suggested_trade"]
        print(f"\nSuggested Trade Parameters:")
        print(f"  Entry: ${trade['entry']}")
        print(f"  Stop Loss: ${trade['stop_loss']}")
        print(f"  Target: ${trade['target']}")
        print(f"  Risk/Reward: {trade['risk_reward']}")

    if explanation.get("context"):
        print(f"\nPrice Context:")
        print(f"{'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Indicator':>12}")
        print("-" * 70)
        for row in explanation["context"]:
            marker = " <<" if row.get("is_signal_date") else ""
            indicator_key = "rsi2" if "rsi2" in row else "ibs"
            indicator_val = row.get(indicator_key, "N/A")
            if indicator_val is not None and indicator_val != "N/A":
                indicator_str = f"{indicator_val:.2f}" if indicator_key == "rsi2" else f"{indicator_val:.3f}"
            else:
                indicator_str = "N/A"
            print(f"{row['date']:<12} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} {row['close']:>10.2f} {indicator_str:>12}{marker}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kobe Signal Explainer (deprecated)")
    ap.add_argument("--dotenv", type=str, default="./.env", help="Path to .env file")
    ap.add_argument("--symbol", type=str, required=False, help="Symbol to explain")
    ap.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD), defaults to latest")
    ap.add_argument("--lookback", type=int, default=30, help="Days of price history to fetch (default: 30)")

    args = ap.parse_args()

    print("This script is deprecated. Use scripts/debugger.py for IBS+RSI signal traces.")
    sys.exit(0)


if __name__ == "__main__":
    main()
