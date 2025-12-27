#!/usr/bin/env python3
"""
Connors RSI (CRSI) Composite Strategy

Components (classic formulation):
- RSI(close, 3)
- RSI(streak, 2), where streak is the length of consecutive up/down closes
- PercentRank(ROC(close, 3), 100)

Composite: CRSI = (RSI3 + RSI_streak + PR_ROC3_100) / 3

Entries:
- Long: CRSI <= 10 and Close > SMA(200)
- Short: CRSI >= 90 and Close < SMA(200)

Exits handled by backtest engine (ATR stop + time stop of 5 bars).
All indicators are shifted by 1 bar to avoid lookahead; fills at open(t+1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 3, method: str = "wilder") -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    if method == "wilder":
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


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


def percent_rank(series: pd.Series, window: int) -> pd.Series:
    """PercentRank: fraction of past values below current (0..100)."""
    def pr(a: pd.Series) -> float:
        if len(a) <= 1:
            return np.nan
        x = a.iloc[-1]
        prev = a.iloc[:-1]
        return 100.0 * (prev < x).mean()
    return series.rolling(window=window, min_periods=window).apply(pr, raw=False)


def compute_streak(close: pd.Series) -> pd.Series:
    """
    Compute consecutive up/down streak lengths: positive for up streaks, negative for down.
    """
    diff = close.diff()
    sign = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # Reset streak when sign changes; accumulate otherwise
    streak = []
    cur = 0
    last = 0
    for s in sign.fillna(0).values:
        if s == 0:
            cur = 0
        elif s == last:
            cur = cur + s
        else:
            cur = s
        streak.append(cur)
        last = s
    return pd.Series(streak, index=close.index)


@dataclass
class ConnorsCRSIParams:
    rsi_period: int = 3
    rsi_method: str = "wilder"
    streak_rsi_period: int = 2
    roc_lookback: int = 3
    pr_window: int = 100
    sma_period: int = 200
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    time_stop_bars: int = 5
    long_entry_crsi_max: float = 10.0  # Conservative MR threshold (classic Connors)
    short_entry_crsi_min: float = 90.0  # Conservative MR threshold
    min_price: float = 5.0


class ConnorsCRSIStrategy:
    def __init__(self, params: Optional[ConnorsCRSIParams] = None):
        self.params = params or ConnorsCRSIParams()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['symbol', 'timestamp']).copy()
        # Compute components per symbol to keep windows aligned
        parts = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp').copy()
            g['rsi3'] = rsi(g['close'], self.params.rsi_period, self.params.rsi_method)
            streak = compute_streak(g['close'])
            g['streak'] = streak
            # Classic Connors CRSI uses RSI of signed streak (not abs)
            g['streak_rsi2'] = rsi(streak, self.params.streak_rsi_period, self.params.rsi_method)
            roc3 = g['close'].pct_change(self.params.roc_lookback)
            g['pr_roc3_100'] = percent_rank(roc3, self.params.pr_window)
            g['sma200'] = sma(g['close'], self.params.sma_period)
            g['atr14'] = atr(g, self.params.atr_period)
            # Composite CRSI
            g['crsi_raw'] = (g['rsi3'] + g['streak_rsi2'] + g['pr_roc3_100']) / 3.0
            # Shift to avoid lookahead
            for col in ['rsi3','streak_rsi2','pr_roc3_100','sma200','atr14','crsi_raw']:
                g[f'{col}_sig'] = g[col].shift(1)
            parts.append(g)
        out = pd.concat(parts, ignore_index=True) if parts else df.copy()
        return out

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_indicators(df)
        rows: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < self.params.sma_period + max(5, self.params.pr_window):
                continue
            # Guards
            base_na = ['crsi_raw_sig','sma200_sig','atr14_sig','close']
            # Conditions
            cond_long = (g['close'] > g['sma200_sig']) & (g['crsi_raw_sig'] <= self.params.long_entry_crsi_max)
            cond_short = (g['close'] < g['sma200_sig']) & (g['crsi_raw_sig'] >= self.params.short_entry_crsi_min)
            idxs = g.index[(cond_long | cond_short)]
            for idx in idxs:
                row = g.loc[idx]
                if any(pd.isna(row.get(c)) for c in base_na):
                    continue
                if row['close'] < self.params.min_price:
                    continue
                side = 'long' if cond_long.loc[idx] else 'short'
                entry = float(row['close'])
                atrv = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else None
                stop = entry - self.params.atr_stop_mult * atrv if side == 'long' else entry + self.params.atr_stop_mult * atrv
                rows.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': side,
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2) if stop is not None else None,
                    'take_profit': None,
                    'reason': f"CRSI={row['crsi_raw_sig']:.1f}",
                    'crsi': round(float(row['crsi_raw_sig']), 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','crsi']
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return signals for the most recent bar per symbol (no lookahead)."""
        df = self._compute_indicators(df)
        out: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < self.params.sma_period + max(5, self.params.pr_window):
                continue
            row = g.iloc[-1]
            if any(pd.isna(row.get(c)) for c in ['crsi_raw_sig','sma200_sig','atr14_sig','close']):
                continue
            if row['close'] < self.params.min_price:
                continue
            crsi_v = float(row['crsi_raw_sig'])
            sma200 = float(row['sma200_sig'])
            atrv = float(row['atr14_sig'])
            close = float(row['close'])
            if close > sma200 and crsi_v <= self.params.long_entry_crsi_max:
                entry = close
                stop = entry - self.params.atr_stop_mult * atrv
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': f"CRSI={crsi_v:.1f}<= {self.params.long_entry_crsi_max} & above SMA{self.params.sma_period}",
                    'crsi': round(crsi_v, 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','crsi','time_stop_bars']
        return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)
