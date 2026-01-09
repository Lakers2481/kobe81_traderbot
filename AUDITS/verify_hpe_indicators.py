#!/usr/bin/env python3
"""
Independent verification of HPE indicators (IBS and RSI).
Recomputes from raw OHLC data to validate thesis claims.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from data.providers.multi_source import fetch_daily_bars_multi
from datetime import datetime, timedelta

def compute_ibs(df: pd.DataFrame) -> pd.Series:
    """
    IBS (Internal Bar Strength) = (Close - Low) / (High - Low)
    Range: 0.0 (close at low) to 1.0 (close at high)
    """
    ibs = (df['close'] - df['low']) / (df['high'] - df['low'])
    return ibs.fillna(0.5)  # Neutral if high == low

def compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """
    RSI (Relative Strength Index) with period=2 for mean reversion.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.fillna(50.0)

def main():
    print("=" * 80)
    print("INDEPENDENT VALIDATION: HPE Indicators (IBS and RSI)")
    print("=" * 80)
    print()

    # Fetch HPE data
    symbol = "HPE"
    end_date = "2026-01-08"  # The scan date
    start_date = "2025-11-01"  # Get enough history for RSI

    print(f"Fetching {symbol} data from {start_date} to {end_date}...")

    try:
        df = fetch_daily_bars_multi(symbol, start_date, end_date)

        if df.empty:
            print(f"ERROR: No data returned for {symbol}")
            sys.exit(1)

        df = df.sort_values('timestamp')

        # Compute indicators
        df['ibs'] = compute_ibs(df)
        df['rsi2'] = compute_rsi(df['close'], period=2)

        # Get last 10 bars for context
        print(f"\nLast 10 bars for {symbol}:")
        print("=" * 80)

        cols = ['timestamp', 'close', 'high', 'low', 'ibs', 'rsi2']
        last_10 = df[cols].tail(10)

        for idx, row in last_10.iterrows():
            print(f"{row['timestamp'].date()} | Close: ${row['close']:.2f} | "
                  f"High: ${row['high']:.2f} | Low: ${row['low']:.2f} | "
                  f"IBS: {row['ibs']:.4f} | RSI(2): {row['rsi2']:.4f}")

        # Get the signal bar (2026-01-08)
        from datetime import datetime
        signal_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        signal_bar = df[df['timestamp'].dt.date == signal_date]

        if signal_bar.empty:
            print(f"\nERROR: No data for signal date {signal_date}")
            sys.exit(1)

        signal_bar = signal_bar.iloc[-1]

        print()
        print("=" * 80)
        print(f"SIGNAL BAR VERIFICATION: {signal_bar['timestamp'].date()}")
        print("=" * 80)
        print(f"Close: ${signal_bar['close']:.2f}")
        print(f"High:  ${signal_bar['high']:.2f}")
        print(f"Low:   ${signal_bar['low']:.2f}")
        print()
        print(f"COMPUTED IBS:  {signal_bar['ibs']:.6f}")
        print(f"COMPUTED RSI:  {signal_bar['rsi2']:.6f}")
        print()
        print("THESIS CLAIMS:")
        print("  IBS = 0.00")
        print("  RSI = 0.0")
        print()

        # Validation
        ibs_match = abs(signal_bar['ibs'] - 0.00) < 0.01  # Within 1%
        rsi_match = abs(signal_bar['rsi2'] - 0.0) < 5.0   # Within 5 points

        print("=" * 80)
        print("VALIDATION RESULT:")
        print("=" * 80)

        if ibs_match:
            print(f"✅ IBS MATCH: {signal_bar['ibs']:.6f} ≈ 0.00 (PASS)")
        else:
            print(f"❌ IBS MISMATCH: {signal_bar['ibs']:.6f} ≠ 0.00 (FAIL)")

        if rsi_match:
            print(f"✅ RSI MATCH: {signal_bar['rsi2']:.6f} ≈ 0.0 (PASS)")
        else:
            print(f"❌ RSI MISMATCH: {signal_bar['rsi2']:.6f} ≠ 0.0 (FAIL)")

        print()

        if ibs_match and rsi_match:
            print("VERDICT: PASS - Indicators independently verified")
            return 0
        else:
            print("VERDICT: FAIL - Indicators do not match thesis claims")
            return 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
