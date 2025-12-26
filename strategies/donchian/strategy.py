#!/usr/bin/env python3
"""
Donchian Breakout Strategy (long-only, daily)

Proven trend-following component inspired by Turtle rules:
- Entry: Close breaks above the highest high of the past N days (excluding today)
- Stop: ATR(14) * stop_mult below entry
- Exit: time stop (default 20 bars); trailing can be added later if needed

Signals computed at close(t), fills assumed at open(t+1).
Indicators are shifted one bar to avoid lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


@dataclass
class DonchianParams:
    lookback: int = 55
    atr_period: int = 14
    stop_mult: float = 2.0
    time_stop_bars: int = 20
    min_price: float = 5.0
    r_multiple: float = 2.5


class DonchianBreakoutStrategy:
    def __init__(self, params: Optional[DonchianParams] = None):
        self.params = params or DonchianParams()

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['symbol','timestamp']).copy()
        parts: List[pd.DataFrame] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp').copy()
            # Highest high of prior N days (exclude current bar)
            g['donchian_hi'] = g['high'].shift(1).rolling(window=self.params.lookback, min_periods=self.params.lookback).max()
            g['atr14'] = atr(g, self.params.atr_period)
            g['atr14_sig'] = g['atr14'].shift(1)
            parts.append(g)
        return pd.concat(parts, ignore_index=True) if parts else df.copy()

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute(df)
        rows: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < max(self.params.lookback, self.params.atr_period) + 5:
                continue
            cond_long = (g['close'] > g['donchian_hi'])
            for idx in g.index[cond_long]:
                row = g.loc[idx]
                if any(pd.isna(row.get(c)) for c in ['close','donchian_hi','atr14']):
                    continue
                if float(row['close']) < self.params.min_price:
                    continue
                entry = float(row['close'])
                atrv = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else None
                stop = entry - self.params.stop_mult * atrv if atrv is not None else None
                take_profit = None
                if stop is not None and entry > stop and self.params.r_multiple and self.params.r_multiple > 0:
                    r = entry - stop
                    take_profit = entry + self.params.r_multiple * r
                rows.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2) if stop is not None else None,
                    'take_profit': round(take_profit, 2) if take_profit is not None else None,
                    'reason': f"Donchian{self.params.lookback} breakout",
                    'donchian_hi': round(float(row['donchian_hi']), 2) if pd.notna(row['donchian_hi']) else None,
                    'time_stop_bars': int(self.params.time_stop_bars),
                    'trail_atr_mult': float(self.params.stop_mult),
                })
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','donchian_hi','time_stop_bars']
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Most recent bar signals only (no lookahead)."""
        df = self._compute(df)
        out: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < max(self.params.lookback, self.params.atr_period) + 5:
                continue
            row = g.iloc[-1]
            if any(pd.isna(row.get(c)) for c in ['close','donchian_hi','atr14_sig']):
                continue
            if float(row['close']) < self.params.min_price:
                continue
            if float(row['close']) > float(row['donchian_hi']):
                entry = float(row['close'])
                atrv = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else None
                stop = entry - self.params.stop_mult * atrv if atrv is not None else None
                tp = None
                if stop is not None and entry > stop and self.params.r_multiple and self.params.r_multiple > 0:
                    r = entry - stop
                    tp = entry + self.params.r_multiple * r
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2) if stop is not None else None,
                    'take_profit': round(tp, 2) if tp is not None else None,
                    'reason': f"Donchian{self.params.lookback} breakout",
                    'donchian_hi': round(float(row['donchian_hi']), 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                    'trail_atr_mult': float(self.params.stop_mult),
                })
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','donchian_hi','time_stop_bars']
        return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)
