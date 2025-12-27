#!/usr/bin/env python3
from __future__ import annotations

"""
Build a labeled ML dataset from walk-forward trade outputs for Donchian and ICT.

For each trade, fetch bars up to the entry date to compute rolling features,
and label by realized outcome (win/loss by net PnL if available or forward return).

Outputs: data/ml/signal_dataset.parquet
"""

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.multi_source import fetch_daily_bars_multi
from ml_meta.features import compute_features_frame


def find_trade_lists(wfdir: Path) -> List[Path]:
    paths: List[Path] = []
    for strat in ("donchian", "turtle_soup"):
        sdir = wfdir / strat
        if not sdir.exists():
            continue
        for p in sdir.rglob("trade_list.csv"):
            paths.append(p)
    return sorted(paths)


def load_trades(trade_files: List[Path]) -> pd.DataFrame:
    frames = []
    for tf in trade_files:
        try:
            df = pd.read_csv(tf)
            if df.empty:
                continue
            df.columns = df.columns.str.lower()
            # Expected columns: timestamp, symbol, side, entry_price, exit_price, pnl, strategy (varies by writer)
            if 'strategy' not in df.columns:
                # infer from path
                strategy = 'donchian' if 'donchian' in str(tf).lower() else 'turtle_soup'
                df['strategy'] = strategy.upper()
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Normalize timestamp to datetime (entry time or signal time)
    if 'timestamp' in out.columns:
        out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
    elif 'entry_time' in out.columns:
        out['timestamp'] = pd.to_datetime(out['entry_time'], errors='coerce')
    else:
        out['timestamp'] = pd.NaT
    # Normalized side
    out['side'] = out.get('side', 'long').astype(str).str.lower()
    # Keep essential columns only
    keep = ['timestamp','symbol','side','entry_price','exit_price','pnl','strategy']
    for k in keep:
        if k not in out.columns:
            out[k] = pd.NA
    out = out[keep].dropna(subset=['timestamp','symbol','entry_price'])
    out['symbol'] = out['symbol'].astype(str).str.upper()
    # Compute simple return estimate per trade (used for PF/Sharpe)
    ret = out['pnl']
    if ret.isna().all():
        # Use exit-entry; normalize by entry to get percentage
        if 'exit_price' in out.columns:
            ret = (out.get('exit_price', pd.Series(index=out.index, dtype=float)) - out['entry_price'])
    # Normalize to return % by entry where possible
    with pd.option_context('mode.use_inf_as_na', True):
        out['ret'] = (ret / out['entry_price']).astype(float)
    out['ret'] = out['ret'].fillna(0.0)
    return out


def compute_labels(trades: pd.DataFrame) -> pd.Series:
    """Binary success label from realized PnL if available; fallback to exit- vs entry-price.
    Short trades: invert sign.
    """
    pnl = trades['pnl']
    if pnl.isna().all():
        # Use exit - entry
        pnl = trades.get('exit_price', pd.Series(index=trades.index, dtype=float)) - trades['entry_price']
    sign = pnl.copy().astype(float)
    # Invert for shorts (profit when price declines)
    is_short = trades['side'].astype(str).str.lower().eq('short')
    sign.loc[is_short] = -sign.loc[is_short]
    return (sign > 0).astype(int)


def fetch_history_for_symbols(symbols: List[str], start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = fetch_daily_bars_multi(sym, start, end, cache_dir=cache_dir)
        if df is None or df.empty:
            continue
        if 'symbol' not in df:
            df = df.copy(); df['symbol'] = sym
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description='Build ML signal dataset from WF outputs')
    ap.add_argument('--wfdir', type=str, default='wf_outputs', help='Walk-forward output directory')
    ap.add_argument('--out', type=str, default='data/ml/signal_dataset.parquet', help='Output parquet file')
    ap.add_argument('--cache', type=str, default='data/cache', help='Cache directory for bars')
    ap.add_argument('--dotenv', type=str, default='./.env')
    args = ap.parse_args()

    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    wfdir = Path(args.wfdir)
    trade_files = find_trade_lists(wfdir)
    if not trade_files:
        print('No trade_list.csv files found under', wfdir)
        return

    trades = load_trades(trade_files)
    if trades.empty:
        print('No trades loaded from WF outputs')
        return

    # Collect needed history window (~220 bars back from earliest entry)
    start_date = (trades['timestamp'].min() - pd.Timedelta(days=260)).date().isoformat()
    end_date = (trades['timestamp'].max() + pd.Timedelta(days=1)).date().isoformat()
    symbols = sorted(trades['symbol'].unique().tolist())

    cache_dir = Path(args.cache)
    hist = fetch_history_for_symbols(symbols, start_date, end_date, cache_dir)
    if hist.empty:
        print('No history fetched; aborting.')
        return

    feats = compute_features_frame(hist)
    # Align features to entry timestamp by merging on (symbol,timestamp)
    # Features are computed on bar timestamps; for safety, align to previous day
    t = trades.copy()
    t['timestamp'] = (t['timestamp'] - pd.Timedelta(days=1)).dt.normalize()
    f = feats.copy(); f['timestamp'] = pd.to_datetime(f['timestamp']).dt.normalize()
    merged = pd.merge(t, f, on=['symbol','timestamp'], how='left')
    # Fill NA features with zeros
    feat_cols = [c for c in merged.columns if c in ('atr14','sma20_over_200','rv20','don20_width','pos_in_don20','ret5','log_vol')]
    for c in feat_cols:
        merged[c] = merged[c].astype(float).fillna(0.0)

    merged['label'] = compute_labels(merged)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f'Wrote dataset: {args.out} rows={len(merged)}')


if __name__ == '__main__':
    main()
