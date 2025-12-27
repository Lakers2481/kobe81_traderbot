#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='Validate universe coverage (>=900 and >=10 years each)')
    ap.add_argument('--earliest-file', type=str, default='data/universe/earliest_latest_universe.csv')
    ap.add_argument('--min-years', type=float, default=10.0)
    ap.add_argument('--min-count', type=int, default=900)
    args = ap.parse_args()

    p = Path(args.earliest_file)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if 'symbol' not in df.columns or 'earliest' not in df.columns or 'latest' not in df.columns:
        raise ValueError('CSV must contain symbol, earliest, latest columns')
    df['earliest'] = pd.to_datetime(df['earliest'], errors='coerce')
    df['latest'] = pd.to_datetime(df['latest'], errors='coerce')
    df['years'] = (df['latest'] - df['earliest']).dt.days / 365.25
    ok_rows = df.dropna(subset=['years'])
    count_ok = (ok_rows['years'] >= args.min_years).sum()
    total = len(ok_rows)
    print(f'Coverage OK (>= {args.min_years}y): {count_ok}/{total}')
    if total < args.min_count:
        raise SystemExit(f'Universe count below {args.min_count}: {total}')
    if count_ok < args.min_count:
        raise SystemExit(f'Coverage below {args.min_count}: {count_ok}')
    print('Universe coverage validation PASSED')


if __name__ == '__main__':
    main()
