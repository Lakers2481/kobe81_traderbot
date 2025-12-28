#!/usr/bin/env python3
from __future__ import annotations

"""
Build a modeling dataset from TOTD backtest artifacts (features at entry, label/outcome, return).
Inputs: reports/totd_YYYY/(ibs_rsi|turtle_soup)_trades.csv
Outputs: data/ml/totd_dataset.parquet
"""

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description='Build TOTD dataset from backtest trades')
    ap.add_argument('--year', type=str, required=True)
    args = ap.parse_args()
    src = ROOT / 'reports' / f'totd_{args.year}'
    out = ROOT / 'data' / 'ml' / 'totd_dataset.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)

    dfs = []
    for key in ('ibs_rsi','turtle_soup'):
        p = src / f'{key}_trades.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df['strategy'] = key.upper()
        df['label'] = (df['r_mult'] > 0).astype(int)
        df['ret'] = df['r_mult']
        dfs.append(df[['date','strategy','symbol','label','ret']])
    if not dfs:
        print('No input trades found.')
        return
    all_df = pd.concat(dfs, ignore_index=True)
    # Merge with features at entry date
    # Fetch recent bars and compute features, but for simplicity assume features will be joined later in training
    # Save core dataset now
    all_df.rename(columns={'date':'timestamp'}, inplace=True)
    all_df.to_parquet(out, index=False)
    print('Wrote:', out)


if __name__ == '__main__':
    main()
