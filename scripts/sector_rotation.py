#!/usr/bin/env python3
"""Analyze sector rotation - continuous market analysis."""
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

# Sector ETFs
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
}

def analyze_rotation(dotenv: str) -> int:
    load_env(Path(dotenv))
    end = now_et().date()
    start = end - timedelta(days=30)

    sector_performance = {}

    for etf, sector in SECTOR_ETFS.items():
        try:
            df = fetch_daily_bars_polygon(etf, start.isoformat(), end.isoformat())
            if df is not None and len(df) > 5:
                ret_5d = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
                ret_20d = df['close'].iloc[-1] / df['close'].iloc[0] - 1
                sector_performance[sector] = {'5d': ret_5d, '20d': ret_20d, 'etf': etf}
        except:
            pass

    if not sector_performance:
        print("[SECTOR] No sector data available")
        return 1

    # Sort by 5-day performance
    sorted_5d = sorted(sector_performance.items(), key=lambda x: x[1]['5d'], reverse=True)

    print("[SECTOR] 5-Day Sector Performance:")
    for sector, data in sorted_5d:
        arrow = "+" if data['5d'] > 0 else ""
        print(f"  {sector:18} ({data['etf']}): {arrow}{data['5d']:.1%}")

    # Identify rotation
    top_3 = [s[0] for s in sorted_5d[:3]]
    bottom_3 = [s[0] for s in sorted_5d[-3:]]

    print(f"\n[SECTOR] Leaders: {', '.join(top_3)}")
    print(f"[SECTOR] Laggards: {', '.join(bottom_3)}")

    # Risk-on vs Risk-off
    risk_on = ['Technology', 'Consumer Disc', 'Financials']
    risk_off = ['Utilities', 'Consumer Staples', 'Healthcare']

    risk_on_perf = sum(sector_performance.get(s, {}).get('5d', 0) for s in risk_on) / 3
    risk_off_perf = sum(sector_performance.get(s, {}).get('5d', 0) for s in risk_off) / 3

    if risk_on_perf > risk_off_perf + 0.005:
        print(f"\n[SECTOR] Market sentiment: RISK-ON")
    elif risk_off_perf > risk_on_perf + 0.005:
        print(f"\n[SECTOR] Market sentiment: RISK-OFF")
    else:
        print(f"\n[SECTOR] Market sentiment: NEUTRAL")

    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dotenv', default='.env')
    args = ap.parse_args()
    return analyze_rotation(args.dotenv)

if __name__ == '__main__':
    sys.exit(main())
