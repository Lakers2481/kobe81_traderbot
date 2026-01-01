#!/usr/bin/env python3
"""Analyze current market regime - continuous monitoring."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import numpy as np

def analyze_regime(dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=120)

    try:
        # Use SPY as market proxy
        df = fetch_daily_bars_polygon('SPY', start.isoformat(), end.isoformat())
        if df is None or len(df) < 50:
            print("[REGIME] Insufficient SPY data")
            return 1

        # Calculate regime indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        returns = df['close'].pct_change()
        df['vol_20'] = returns.rolling(20).std() * np.sqrt(252)

        latest = df.iloc[-1]

        # Determine regime
        trend = "BULL" if latest['close'] > latest['sma_50'] else "BEAR"
        vol_level = "HIGH" if latest['vol_20'] > 0.20 else ("NORMAL" if latest['vol_20'] > 0.12 else "LOW")

        # Momentum
        mom_20 = (latest['close'] / df['close'].iloc[-20] - 1)
        momentum = "STRONG" if mom_20 > 0.03 else ("WEAK" if mom_20 < -0.03 else "NEUTRAL")

        print(f"[REGIME] Current: {trend} | Volatility: {vol_level} ({latest['vol_20']:.1%}) | Momentum: {momentum} ({mom_20:.1%})")

        # Check VIX if available
        try:
            vix = fetch_daily_bars_polygon('VIXY', start.isoformat(), end.isoformat())
            if vix is not None and len(vix) > 0:
                vix_change = vix['close'].iloc[-1] / vix['close'].iloc[-5] - 1
                print(f"[REGIME] VIX proxy 5d change: {vix_change:.1%}")
        except:
            pass

        return 0

    except Exception as e:
        print(f"[REGIME] ERROR - {e}")
        return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return analyze_regime(args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
