#!/usr/bin/env python3
"""
Connors RSI-2 Strategy (Canonical)

- RSI(2) long entry <= 10; short entry >= 90
- Exits: long when RSI >= 70; short when RSI <= 30
- SMA(200) trend filter (longs above, shorts below)
- Protective stop: ATR(14) * 2.0
- Time stop: 5 bars
- Signals computed at close(t), fills assumed at open(t+1)
- No lookahead bias: indicators shifted by 1 bar where applicable
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


# ============================
# Indicators (vectorized)
# ============================

def rsi(series: pd.Series, period: int = 2, method: str = "wilder") -> pd.Series:
    """Relative Strength Index with Wilder smoothing by default."""
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


# ============================
# Parameters and Strategy
# ============================

@dataclass
class ConnorsRSI2Params:
    rsi_period: int = 2
    rsi_method: str = "wilder"
    sma_period: int = 200
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    time_stop_bars: int = 5
    long_entry_rsi_max: float = 10.0
    short_entry_rsi_min: float = 90.0
    long_exit_rsi_min: float = 70.0
    short_exit_rsi_max: float = 30.0
    min_price: float = 5.0


class ConnorsRSI2Strategy:
    def __init__(self, params: Optional[ConnorsRSI2Params] = None):
        self.params = params or ConnorsRSI2Params()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['symbol', 'timestamp']).copy()
        df['rsi2'] = rsi(df['close'], self.params.rsi_period, self.params.rsi_method)
        df['sma200'] = sma(df['close'], self.params.sma_period)
        df['atr14'] = atr(df, self.params.atr_period)
        # Shift to avoid lookahead
        for col in ['rsi2', 'sma200', 'atr14']:
            df[f'{col}_sig'] = df[col].shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df must contain columns:
          ['timestamp','symbol','open','high','low','close','volume']
        Returns a DataFrame with signals per symbol at the most recent bar.
        """
        df = self._compute_indicators(df)
        out: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            if len(g) < self.params.sma_period + 5:
                continue
            row = g.iloc[-1]
            # Basic guards
            if any(pd.isna(row.get(c)) for c in ['rsi2_sig', 'sma200_sig', 'atr14_sig', 'close']):
                continue
            if row['close'] < self.params.min_price:
                continue

            rsi2 = float(row['rsi2_sig'])
            sma200 = float(row['sma200_sig'])
            atrv = float(row['atr14_sig'])
            close = float(row['close'])

            # Long entry: above SMA(200) and RSI(2) <= 10
            if close > sma200 and rsi2 <= self.params.long_entry_rsi_max:
                entry = close
                stop = entry - self.params.atr_stop_mult * atrv
                tp = None  # leave exits to OMS/Policy per blueprint
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': tp,
                    'reason': f"RSI2={rsi2:.1f}<= {self.params.long_entry_rsi_max} & above SMA{self.params.sma_period}",
                    'rsi2': round(rsi2, 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })

            # Short entry: below SMA(200) and RSI(2) >= 90
            elif close < sma200 and rsi2 >= self.params.short_entry_rsi_min:
                entry = close
                stop = entry + self.params.atr_stop_mult * atrv
                tp = None
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'short',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': tp,
                    'reason': f"RSI2={rsi2:.1f}>= {self.params.short_entry_rsi_min} & below SMA{self.params.sma_period}",
                    'rsi2': round(rsi2, 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })

        columns = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','rsi2']
        return pd.DataFrame(out, columns=columns) if out else pd.DataFrame(columns=columns)

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest-friendly variant: returns ALL bars where entry conditions are met
        (computed on prior bar indicators, filled next bar).
        """
        df = self._compute_indicators(df)
        rows: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < self.params.sma_period + 5:
                continue
            # Using shifted indicators to avoid lookahead
            cond_long = (g['close'] > g['sma200_sig']) & (g['rsi2_sig'] <= self.params.long_entry_rsi_max)
            cond_short = (g['close'] < g['sma200_sig']) & (g['rsi2_sig'] >= self.params.short_entry_rsi_min)
            # Iterate rows where a condition is true
            for idx, row in g.loc[cond_long | cond_short].iterrows():
                side = 'long' if cond_long.loc[idx] else 'short'
                atrv = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else None
                if pd.isna(row['close']) or pd.isna(row['sma200_sig']) or pd.isna(row['rsi2_sig']) or pd.isna(row['atr14_sig']):
                    continue
                if row['close'] < self.params.min_price:
                    continue
                entry = float(row['close'])
                stop = entry - self.params.atr_stop_mult * atrv if side == 'long' else entry + self.params.atr_stop_mult * atrv
                rows.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': side,
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': f"RSI2={row['rsi2_sig']:.1f} {'<=' if side=='long' else '>='} {self.params.long_entry_rsi_max if side=='long' else self.params.short_entry_rsi_min}",
                    'rsi2': round(float(row['rsi2_sig']), 2),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })
        columns = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','rsi2']
        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


# Quick verification (synthetic)
if __name__ == '__main__':
    np.random.seed(0)
    dates = pd.date_range(end='2024-06-01', periods=260, freq='B')
    prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, len(dates)))
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': prices * (1 + np.random.uniform(-0.002, 0.002, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'close': prices,
        'volume': np.random.randint(1_000_000, 5_000_000, len(dates)),
    })
    strat = ConnorsRSI2Strategy()
    sigs = strat.generate_signals(df)
    print(sigs.tail(5))
