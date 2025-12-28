#!/usr/bin/env python3
from __future__ import annotations

"""
Build a labeled ML dataset from walk-forward trade outputs for IBS+RSI and ICT.

For each trade, fetch bars up to the entry date to compute rolling features,
and label by realized outcome (win/loss by net PnL if available or forward return).

Outputs: data/ml/signal_dataset.parquet
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


# Directory name mapping: old WF output names -> canonical strategy name
DIR_TO_STRATEGY = {
    'ibs_rsi': 'IBS_RSI',
    'ibs': 'IBS_RSI',
    'rsi2': 'IBS_RSI',
    'and': 'IBS_RSI',
    'turtle_soup': 'TURTLE_SOUP',
    'ict': 'TURTLE_SOUP',
}

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.multi_source import fetch_daily_bars_multi
from ml_meta.features import compute_features_frame


def find_trade_lists(wfdir: Path) -> List[Tuple[Path, str]]:
    """Find all trade_list.csv files and their canonical strategy names."""
    paths: List[Tuple[Path, str]] = []
    if not wfdir.exists():
        return paths
    for subdir in wfdir.iterdir():
        if not subdir.is_dir():
            continue
        dir_name = subdir.name.lower()
        if dir_name not in DIR_TO_STRATEGY:
            continue
        strategy = DIR_TO_STRATEGY[dir_name]
        for p in subdir.rglob("trade_list.csv"):
            paths.append((p, strategy))
    return sorted(paths, key=lambda x: str(x[0]))


def _pair_buy_sell(df: pd.DataFrame, strategy: str) -> List[dict]:
    """Pair consecutive BUY/SELL rows for the same symbol into trade records.

    The backtester outputs BUY row (entry) followed by SELL row (exit) for longs.
    """
    trades: List[dict] = []
    pending: Dict[str, dict] = {}  # symbol -> pending BUY row

    for _, row in df.sort_values('timestamp').iterrows():
        sym = str(row.get('symbol', '')).upper()
        side = str(row.get('side', '')).upper()

        if side == 'BUY':
            # Entry: store for later pairing
            pending[sym] = {
                'timestamp': row['timestamp'],
                'symbol': sym,
                'qty': int(row.get('qty', 1)),
                'entry_price': float(row['price']),
                'strategy': strategy,
            }
        elif side == 'SELL' and sym in pending:
            # Exit: pair with pending BUY
            entry = pending.pop(sym)
            exit_price = float(row['price'])
            qty = entry['qty']
            pnl = (exit_price - entry['entry_price']) * qty
            trades.append({
                'timestamp': entry['timestamp'],
                'symbol': sym,
                'side': 'long',
                'qty': qty,
                'entry_price': entry['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'strategy': strategy,
            })
    return trades


def load_trades(trade_files: List[Tuple[Path, str]]) -> pd.DataFrame:
    """Load trades from WF outputs, handling both BUY/SELL pair format and legacy format."""
    all_trades: List[dict] = []

    for tf, strategy in trade_files:
        try:
            df = pd.read_csv(tf)
            if df.empty:
                continue
            df.columns = df.columns.str.lower()
            df['timestamp'] = pd.to_datetime(df.get('timestamp', pd.NaT), errors='coerce')

            # Detect format: BUY/SELL pairs (price column) vs legacy (entry_price column)
            if 'price' in df.columns and 'entry_price' not in df.columns:
                # BUY/SELL pair format from backtester
                trades = _pair_buy_sell(df, strategy)
                all_trades.extend(trades)
            else:
                # Legacy format with entry_price, exit_price, pnl already present
                for _, row in df.iterrows():
                    all_trades.append({
                        'timestamp': row.get('timestamp'),
                        'symbol': str(row.get('symbol', '')).upper(),
                        'side': str(row.get('side', 'long')).lower(),
                        'qty': int(row.get('qty', 1)),
                        'entry_price': float(row.get('entry_price', 0)),
                        'exit_price': float(row.get('exit_price', 0)) if pd.notna(row.get('exit_price')) else None,
                        'pnl': float(row.get('pnl', 0)) if pd.notna(row.get('pnl')) else None,
                        'strategy': strategy,
                    })
        except Exception:
            continue

    if not all_trades:
        return pd.DataFrame()

    out = pd.DataFrame(all_trades)
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce')
    out = out.dropna(subset=['timestamp', 'symbol', 'entry_price'])

    # Compute return percentage
    if 'pnl' in out.columns and 'entry_price' in out.columns:
        with pd.option_context('mode.use_inf_as_na', True):
            out['ret'] = (out['pnl'] / (out['entry_price'] * out.get('qty', 1))).astype(float)
        out['ret'] = out['ret'].fillna(0.0)
    else:
        out['ret'] = 0.0

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
