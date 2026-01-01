#!/usr/bin/env python3
"""
Hunt for patterns in a stock - continuous pattern discovery.

Looks for:
- Failed breakouts (turtle soup setups)
- Mean reversion setups (IBS + RSI)
- Momentum breakouts
- Volume anomalies
- Price compression patterns
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


def hunt_patterns(symbol: str, dotenv: str) -> dict:
    """Hunt for trading patterns in a stock."""
    load_env(Path(dotenv))

    end_date = now_et().date()
    start_date = end_date - timedelta(days=100)

    try:
        df = fetch_daily_bars_polygon(symbol, start_date.isoformat(), end_date.isoformat())
        if df is None or len(df) < 25:
            return {'symbol': symbol, 'patterns': [], 'status': 'insufficient_data'}

        patterns = []

        # Calculate indicators
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['vol_avg'] = df['volume'].rolling(20).mean()

        # IBS
        df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)

        # RSI(2)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi2'] = 100 - (100 / (1 + rs))

        # Check latest bar for patterns
        latest = df.iloc[-1]

        # 1. Mean Reversion Setup (IBS + RSI)
        if latest['ibs'] < 0.1 and latest['rsi2'] < 5:
            patterns.append({
                'type': 'MEAN_REVERSION_EXTREME',
                'strength': 'STRONG',
                'ibs': float(latest['ibs']),
                'rsi2': float(latest['rsi2']),
                'description': f"Extreme oversold: IBS={latest['ibs']:.2f}, RSI2={latest['rsi2']:.0f}"
            })
        elif latest['ibs'] < 0.2 and latest['rsi2'] < 10:
            patterns.append({
                'type': 'MEAN_REVERSION',
                'strength': 'MODERATE',
                'ibs': float(latest['ibs']),
                'rsi2': float(latest['rsi2']),
                'description': f"Oversold: IBS={latest['ibs']:.2f}, RSI2={latest['rsi2']:.0f}"
            })

        # 2. Failed Breakout (Turtle Soup)
        prev = df.iloc[-2]
        if prev['high'] > df['high_20'].iloc[-3] and latest['close'] < prev['low']:
            sweep_size = (prev['high'] - df['high_20'].iloc[-3]) / latest['atr']
            if sweep_size > 0.3:
                patterns.append({
                    'type': 'TURTLE_SOUP_LONG',
                    'strength': 'STRONG' if sweep_size > 0.5 else 'MODERATE',
                    'sweep_atr': float(sweep_size),
                    'description': f"Failed breakout above 20-day high, sweep={sweep_size:.2f} ATR"
                })

        if prev['low'] < df['low_20'].iloc[-3] and latest['close'] > prev['high']:
            sweep_size = (df['low_20'].iloc[-3] - prev['low']) / latest['atr']
            if sweep_size > 0.3:
                patterns.append({
                    'type': 'TURTLE_SOUP_SHORT',
                    'strength': 'STRONG' if sweep_size > 0.5 else 'MODERATE',
                    'sweep_atr': float(sweep_size),
                    'description': f"Failed breakout below 20-day low, sweep={sweep_size:.2f} ATR"
                })

        # 3. Volume Anomaly
        if latest['volume'] > latest['vol_avg'] * 2:
            patterns.append({
                'type': 'VOLUME_SPIKE',
                'strength': 'STRONG' if latest['volume'] > latest['vol_avg'] * 3 else 'MODERATE',
                'volume_ratio': float(latest['volume'] / latest['vol_avg']),
                'description': f"Volume {latest['volume']/latest['vol_avg']:.1f}x average"
            })

        # 4. Price Compression (Bollinger Band squeeze)
        df['bb_width'] = (df['close'].rolling(20).std() * 2) / df['sma_20']
        if latest['bb_width'] < df['bb_width'].quantile(0.1):
            patterns.append({
                'type': 'COMPRESSION',
                'strength': 'MODERATE',
                'bb_width_pct': float(df['bb_width'].rank(pct=True).iloc[-1]),
                'description': "Price compression - potential breakout setup"
            })

        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'patterns': patterns,
            'pattern_count': len(patterns),
            'status': 'success'
        }

        if patterns:
            print(f"[PATTERN] {symbol}: Found {len(patterns)} patterns")
            for p in patterns:
                print(f"  - {p['type']} ({p['strength']}): {p['description']}")
        else:
            print(f"[PATTERN] {symbol}: No patterns found")

        # Save if patterns found
        if patterns:
            output_dir = ROOT / 'outputs' / 'patterns'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'{symbol}_{end_date.isoformat()}.json'
            output_file.write_text(json.dumps(result, indent=2))

        return result

    except Exception as e:
        print(f"[PATTERN] {symbol}: ERROR - {e}")
        return {'symbol': symbol, 'patterns': [], 'status': 'error', 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='Hunt for patterns in a stock')
    ap.add_argument('--symbol', required=True, help='Stock symbol')
    ap.add_argument('--dotenv', default='.env', help='Path to .env file')
    args = ap.parse_args()

    result = hunt_patterns(args.symbol, args.dotenv)
    return 0 if result.get('status') == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
