#!/usr/bin/env python3
"""
IBS (Internal Bar Strength) Strategy (Canonical)

- IBS = (Close - Low) / (High - Low)
- Long entry: IBS < 0.2 and Close > SMA(200)
- Short entry: IBS > 0.8 and Close < SMA(200)
- Protective stop: ATR(14) * 2.0
- Time stop: 5 bars
- Signals at close(t), fills at open(t+1)
- No lookahead bias: indicators shifted by 1 bar
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


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


def ibs(df: pd.DataFrame) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    rng = (high - low).replace(0, np.nan)
    return (close - low) / rng


@dataclass
class IBSParams:
    sma_period: int = 200
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    time_stop_bars: int = 5
    ibs_long_max: float = 0.2
    ibs_short_min: float = 0.8
    min_price: float = 5.0


class IBSStrategy:
    def __init__(self, params: Optional[IBSParams] = None):
        self.params = params or IBSParams()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['symbol','timestamp']).copy()
        df['ibs'] = ibs(df)
        df['sma200'] = sma(df['close'], self.params.sma_period)
        df['atr14'] = atr(df, self.params.atr_period)
        # Shift to avoid lookahead
        for col in ['ibs', 'sma200', 'atr14']:
            df[f'{col}_sig'] = df[col].shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._compute_indicators(df)
        out: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            if len(g) < self.params.sma_period + 5:
                continue
            row = g.iloc[-1]
            if any(pd.isna(row.get(c)) for c in ['ibs_sig','sma200_sig','atr14_sig','close']):
                continue
            if row['close'] < self.params.min_price:
                continue

            ibsv = float(row['ibs_sig'])
            sma200 = float(row['sma200_sig'])
            atrv = float(row['atr14_sig'])
            close = float(row['close'])

            # Long entry: IBS < 0.2 and above SMA(200)
            if close > sma200 and ibsv < self.params.ibs_long_max:
                entry = close
                stop = entry - self.params.atr_stop_mult * atrv
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': f"IBS={ibsv:.3f} < {self.params.ibs_long_max} & above SMA{self.params.sma_period}",
                    'ibs': round(ibsv, 3),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })

            # Short entry: IBS > 0.8 and below SMA(200)
            elif close < sma200 and ibsv > self.params.ibs_short_min:
                entry = close
                stop = entry + self.params.atr_stop_mult * atrv
                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'short',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': f"IBS={ibsv:.3f} > {self.params.ibs_short_min} & below SMA{self.params.sma_period}",
                    'ibs': round(ibsv, 3),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })

        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','ibs']
        return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest-friendly: returns ALL bars where entry conditions are met
        (indicators shifted one bar; fills assumed next bar).
        """
        df = self._compute_indicators(df)
        rows: List[Dict] = []
        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < self.params.sma_period + 5:
                continue
            cond_long = (g['close'] > g['sma200_sig']) & (g['ibs_sig'] < self.params.ibs_long_max)
            cond_short = (g['close'] < g['sma200_sig']) & (g['ibs_sig'] > self.params.ibs_short_min)
            for idx, row in g.loc[cond_long | cond_short].iterrows():
                side = 'long' if cond_long.loc[idx] else 'short'
                if any(pd.isna(row.get(c)) for c in ['ibs_sig','sma200_sig','atr14_sig','close']):
                    continue
                if row['close'] < self.params.min_price:
                    continue
                entry = float(row['close'])
                atrv = float(row['atr14_sig'])
                stop = entry - self.params.atr_stop_mult * atrv if side == 'long' else entry + self.params.atr_stop_mult * atrv
                rows.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': side,
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': f"IBS={row['ibs_sig']:.3f} {'<' if side=='long' else '>'} threshold & trend",
                    'ibs': round(float(row['ibs_sig']), 3),
                    'time_stop_bars': int(self.params.time_stop_bars),
                })
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','ibs']
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


if __name__ == '__main__':
    np.random.seed(1)
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
    strat = IBSStrategy()
    sigs = strat.generate_signals(df)
    print(sigs.tail(5))
