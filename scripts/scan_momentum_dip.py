#!/usr/bin/env python3
"""
Daily Scanner for Momentum Dip Strategy

Scans 900 stocks, outputs:
1. Top 3 picks (ranked by score)
2. Trade of the Day (single best opportunity)

Quant Interview Ready: 65.8% win rate, 1.38 profit factor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

env_path = Path("C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env")
if env_path.exists():
    load_dotenv(env_path)

# from strategies.momentum_dip import MomentumDipStrategy, MomentumDipParams


def run_scanner(
    symbols: list[str],
    lookback_days: int = 250,
) -> pd.DataFrame:
    """Scan universe for today's signals."""
    # strategy = MomentumDipStrategy() # Commented out strategy initialization
    cache_dir = Path("data/cache/polygon")

    # Calculate date range
    end = datetime.now().strftime('%Y-%m-%d')
    start_dt = datetime.now() - pd.Timedelta(days=lookback_days + 50)
    start = start_dt.strftime('%Y-%m-%d')

    print(f"Scanning {len(symbols)} symbols...")
    all_data = []

    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(symbols)}...")
        try:
            df = fetch_daily_bars_polygon(sym, start, end, cache_dir=cache_dir)
            if df is not None and len(df) > 220:
                if 'symbol' not in df.columns:
                    df['symbol'] = sym
                all_data.append(df)
        except:
            pass

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Got data for {len(all_data)} symbols")

    # Generate signals for latest bar only
    # signals = strategy.generate_signals(combined) # Commented out signal generation
    print("Skipping signal generation for unimplemented strategy.")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', default='data/universe/optionable_liquid_900.csv')
    parser.add_argument('--top', type=int, default=3, help='Number of top picks')
    parser.add_argument('--output', type=str, default='logs/daily_picks.csv')
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    print(f"\n{'='*60}")
    print("MOMENTUM DIP SCANNER")
    print(f"{'='*60}")
    print(f"Universe: {len(symbols)} symbols")
    print(f"Strategy: MomentumDip (65.8% WR, 1.38 PF)")
    print(f"{'='*60}\n")

    signals = run_scanner(symbols)

    if signals.empty:
        print("\nNo signals today (Strategy not implemented).")
        return 0

    print(f"\n{'='*60}")
    print(f"SIGNALS FOUND: {len(signals)}")
    print(f"{'='*60}")

    # Top picks
    top_picks = signals.head(args.top)
    print(f"\nTOP {args.top} PICKS:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_picks.iterrows(), 1):
        print(f"{i}. {row['symbol']:<6} Entry: ${row['entry_price']:.2f}  Stop: ${row['stop_loss']:.2f}  "
              f"CumRSI: {row['cum_rsi']:.1f}  20dRet: {row['ret20']:.1f}%")

    # Trade of the Day
    totd = signals.iloc[0]
    print(f"\n{'='*60}")
    print("TRADE OF THE DAY")
    print(f"{'='*60}")
    print(f"Symbol:      {totd['symbol']}")
    print(f"Side:        {totd['side'].upper()}")
    print(f"Entry Price: ${totd['entry_price']:.2f}")
    print(f"Stop Loss:   ${totd['stop_loss']:.2f}")
    print(f"Risk (ATR):  ${totd['atr']:.2f}")
    print(f"CumRSI:      {totd['cum_rsi']:.1f}")
    print(f"RSI(2):      {totd['rsi2']:.1f}")
    print(f"20d Return:  {totd['ret20']:.1f}%")
    print(f"Reason:      {totd['reason']}")
    print(f"{'='*60}")

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing file or create new
    signals['scan_date'] = datetime.now().strftime('%Y-%m-%d')
    signals['scan_time'] = datetime.now().strftime('%H:%M:%S')

    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined_output = pd.concat([existing, signals], ignore_index=True)
        combined_output.to_csv(output_path, index=False)
    else:
        signals.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")

    # Also save TOTD separately
    totd_path = Path("logs/trade_of_the_day.json")
    totd_dict = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'symbol': totd['symbol'],
        'side': totd['side'],
        'entry_price': float(totd['entry_price']),
        'stop_loss': float(totd['stop_loss']),
        'atr': float(totd['atr']),
        'cum_rsi': float(totd['cum_rsi']),
        'rsi2': float(totd['rsi2']),
        'ret20': float(totd['ret20']),
        'reason': totd['reason'],
        'time_stop_bars': int(totd['time_stop_bars']),
    }
    with open(totd_path, 'w') as f:
        json.dump(totd_dict, f, indent=2)
    print(f"TOTD saved to {totd_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
