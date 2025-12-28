#!/usr/bin/env python3
"""Debug script to investigate why signals aren't being generated."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams

# Load cached data
cache = Path("data/cache/polygon")
files = sorted(cache.glob("*_2024-01-01_2025-12-26.csv"))[:50]

print(f"Loading {len(files)} cached files...")
all_data = []
for f in files:
    df = pd.read_csv(f)
    sym = f.stem.split("_")[0]
    df["symbol"] = sym
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
print(f"Total bars: {len(combined)}")
print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

# Check parameters
params = DualStrategyParams()
print(f"\n=== PARAMETERS ===")
print(f"IBS entry threshold: < {params.ibs_entry}")
print(f"RSI entry threshold: < {params.rsi_entry}")
print(f"Min price: ${params.min_price}")

# Run scanner
scanner = DualStrategyScanner(params)

# Get signals over time (for backtesting)
print("\n=== SCANNING ALL HISTORICAL BARS ===")
signals = scanner.scan_signals_over_time(combined)

if signals.empty:
    print("NO SIGNALS FOUND - investigating...")

    # Check indicators manually
    print("\n=== CHECKING INDICATORS ===")
    df = scanner._compute_indicators(combined)

    # Check IBS values
    ibs_vals = df['ibs'].dropna()
    print(f"IBS range: {ibs_vals.min():.3f} to {ibs_vals.max():.3f}")
    print(f"IBS < {params.ibs_entry}: {(ibs_vals < params.ibs_entry).sum()} bars")

    # Check RSI values
    rsi_vals = df['rsi2'].dropna()
    print(f"RSI2 range: {rsi_vals.min():.1f} to {rsi_vals.max():.1f}")
    print(f"RSI2 < {params.rsi_entry}: {(rsi_vals < params.rsi_entry).sum()} bars")

    # Check SMA filter
    above_sma = (df['close'] > df['sma200']).sum()
    print(f"Close > SMA200: {above_sma} bars")

    # Check combined conditions
    ibs_ok = df['ibs'] < params.ibs_entry
    rsi_ok = df['rsi2'] < params.rsi_entry
    sma_ok = df['close'] > df['sma200']
    price_ok = df['close'] >= params.min_price

    all_ok = ibs_ok & rsi_ok & sma_ok & price_ok
    print(f"\nBars meeting ALL IBS+RSI criteria: {all_ok.sum()}")

    # Show some examples
    if all_ok.sum() > 0:
        examples = df[all_ok].head(5)
        print("\nExample bars that SHOULD trigger:")
        print(examples[['timestamp', 'symbol', 'close', 'ibs', 'rsi2', 'sma200']].to_string())
else:
    print(f"Total signals: {len(signals)}")
    print(f"  IBS_RSI: {len(signals[signals['strategy']=='IBS_RSI'])}")
    print(f"  TurtleSoup: {len(signals[signals['strategy']=='TurtleSoup'])}")

    # Show recent signals
    signals['timestamp'] = pd.to_datetime(signals['timestamp'])
    recent = signals.sort_values('timestamp', ascending=False).head(20)

    print("\n=== MOST RECENT SIGNALS ===")
    for _, r in recent.iterrows():
        ts = r['timestamp'].strftime('%Y-%m-%d') if pd.notna(r['timestamp']) else 'N/A'
        print(f"{ts} | {r['symbol']:5} | {r['strategy']:10} | ${r['entry_price']:7.2f} | IBS={r.get('ibs', 'N/A')} RSI={r.get('rsi2', 'N/A')}")
