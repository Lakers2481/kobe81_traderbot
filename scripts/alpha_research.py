#!/usr/bin/env python3
"""Alpha factor research - continuous discovery."""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import numpy as np

def research_alpha(symbol: str, dotenv: str) -> dict:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=365)

    try:
        df = fetch_daily_bars_polygon(symbol, start.isoformat(), end.isoformat())
        if df is None or len(df) < 60:
            return {'symbol': symbol, 'status': 'insufficient_data'}

        # Research various alpha factors
        factors = {}

        # Momentum factors
        factors['mom_5d'] = float(df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
        factors['mom_20d'] = float(df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
        factors['mom_60d'] = float(df['close'].iloc[-1] / df['close'].iloc[-60] - 1)

        # Mean reversion factors
        factors['distance_sma20'] = float((df['close'].iloc[-1] / df['close'].rolling(20).mean().iloc[-1]) - 1)

        # Volatility factors
        returns = df['close'].pct_change()
        factors['vol_20d'] = float(returns.tail(20).std() * np.sqrt(252))
        factors['vol_ratio'] = float(returns.tail(5).std() / returns.tail(20).std()) if returns.tail(20).std() > 0 else 1

        # Volume factors
        factors['vol_surge'] = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])

        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'factors': factors,
            'status': 'success'
        }

        print(f"[ALPHA] {symbol}: mom5d={factors['mom_5d']:.1%}, vol_surge={factors['vol_surge']:.1f}x")
        return result

    except Exception as e:
        print(f"[ALPHA] {symbol}: ERROR - {e}")
        return {'symbol': symbol, 'status': 'error'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    result = research_alpha(args.symbol, args.dotenv)
    return 0 if result.get('status') == 'success' else 1

if __name__ == '__main__':
    sys.exit(main())
