#!/usr/bin/env python3
"""
Analyze a single stock - comprehensive analysis for continuous learning.

This script runs deep analysis on one stock at a time, cycling through
the entire 900-stock universe systematically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_daily_bars_polygon
from core.clock.tz_utils import now_et
import pandas as pd
import numpy as np
import json


def analyze_stock(symbol: str, dotenv: str) -> dict:
    """Run comprehensive analysis on a single stock."""
    load_env(Path(dotenv))

    end_date = now_et().date()
    start_date = end_date - timedelta(days=365)

    try:
        df = fetch_daily_bars_polygon(symbol, start_date.isoformat(), end_date.isoformat())
        if df is None or df.empty:
            return {'symbol': symbol, 'status': 'no_data'}

        # Basic statistics
        returns = df['close'].pct_change().dropna()

        analysis = {
            'symbol': symbol,
            'status': 'analyzed',
            'timestamp': datetime.now().isoformat(),
            'bars': len(df),
            'latest_close': float(df['close'].iloc[-1]),
            'avg_volume': float(df['volume'].mean()),

            # Return statistics
            'return_1d': float(returns.iloc[-1]) if len(returns) > 0 else 0,
            'return_5d': float(df['close'].iloc[-1] / df['close'].iloc[-5] - 1) if len(df) > 5 else 0,
            'return_20d': float(df['close'].iloc[-1] / df['close'].iloc[-20] - 1) if len(df) > 20 else 0,
            'return_252d': float(df['close'].iloc[-1] / df['close'].iloc[0] - 1) if len(df) > 252 else 0,

            # Volatility
            'volatility_20d': float(returns.tail(20).std() * np.sqrt(252)) if len(returns) > 20 else 0,
            'volatility_60d': float(returns.tail(60).std() * np.sqrt(252)) if len(returns) > 60 else 0,

            # Technical levels
            'sma_20': float(df['close'].tail(20).mean()),
            'sma_50': float(df['close'].tail(50).mean()) if len(df) > 50 else 0,
            'sma_200': float(df['close'].tail(200).mean()) if len(df) > 200 else 0,

            # Position in range
            'high_52w': float(df['high'].tail(252).max()) if len(df) > 252 else float(df['high'].max()),
            'low_52w': float(df['low'].tail(252).min()) if len(df) > 252 else float(df['low'].min()),

            # IBS (Internal Bar Strength)
            'ibs': float((df['close'].iloc[-1] - df['low'].iloc[-1]) /
                        (df['high'].iloc[-1] - df['low'].iloc[-1]))
                   if df['high'].iloc[-1] != df['low'].iloc[-1] else 0.5,
        }

        # Calculate RSI(2)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi2 = 100 - (100 / (1 + rs))
        analysis['rsi2'] = float(rsi2.iloc[-1]) if not pd.isna(rsi2.iloc[-1]) else 50

        # Save analysis
        output_dir = ROOT / 'outputs' / 'stock_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{symbol}_{end_date.isoformat()}.json'
        output_file.write_text(json.dumps(analysis, indent=2))

        print(f"[ANALYZE] {symbol}: close=${analysis['latest_close']:.2f}, "
              f"vol={analysis['volatility_20d']:.1%}, IBS={analysis['ibs']:.2f}, RSI2={analysis['rsi2']:.0f}")

        return analysis

    except Exception as e:
        print(f"[ANALYZE] {symbol}: ERROR - {e}")
        return {'symbol': symbol, 'status': 'error', 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='Analyze single stock')
    ap.add_argument('--symbol', required=True, help='Stock symbol to analyze')
    ap.add_argument('--dotenv', default='.env', help='Path to .env file')
    args = ap.parse_args()

    result = analyze_stock(args.symbol, args.dotenv)
    return 0 if result.get('status') == 'analyzed' else 1


if __name__ == '__main__':
    sys.exit(main())
