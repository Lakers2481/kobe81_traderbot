from __future__ import annotations

"""
IBS + RSI(2) Mean-Reversion Strategy (long-only, daily)

Entry conditions (evaluated on prior bar to avoid lookahead):
- IBS(prev) < ibs_max (default 0.15)
- RSI2(prev) < rsi_max (default 10)
- Close(prev) > SMA(200) (trend filter)

Stops/targets:
- Stop = entry - atr_mult * ATR(14)
- Target = entry + r_mult * (entry - stop)
- Time exit = time_stop_bars (default 5)

Outputs a DataFrame with columns:
  timestamp, symbol, side, entry_price, stop_loss, take_profit, reason, score, time_stop_bars
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd


@dataclass
class IbsRsiParams:
    # TIGHTENED PARAMETERS (v2.0) - Reduce signals from 50/week to ~5/week
    ibs_max: float = 0.08          # Was 0.15 - 47% tighter for extreme oversold only
    rsi_max: float = 5.0           # Was 10.0 - 50% tighter for severe oversold
    sma200_filter: bool = True
    atr_mult: float = 1.5          # Was 1.0 - wider stop reduces false stop-outs
    r_multiple: float = 2.0
    time_stop_bars: int = 7        # Was 5 - more patience for mean reversion
    min_price: float = 15.0        # Was 5.0 - higher liquidity stocks only


class IbsRsiStrategy:
    def __init__(self, params: Optional[IbsRsiParams] = None):
        self.params = params or IbsRsiParams()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    @staticmethod
    def _rsi2(c: pd.Series) -> pd.Series:
        # Simple RSI(2)
        delta = c.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(2, min_periods=2).mean()
        roll_down = down.rolling(2, min_periods=2).mean()
        rs = roll_up / (roll_down.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.sort_values('timestamp').copy()
        # Normalize timestamps to tz-naive for engine compatibility
        g['timestamp'] = pd.to_datetime(g['timestamp'], errors='coerce')
        if hasattr(g['timestamp'].dt, 'tz_localize'):
            try:
                g['timestamp'] = g['timestamp'].dt.tz_localize(None)
            except Exception:
                pass
        # IBS of previous bar
        rng = (g['high'] - g['low']).replace(0, pd.NA)
        g['ibs_prev'] = ((g['close'] - g['low']) / rng).shift(1)
        # RSI2 of previous bar
        g['rsi2_prev'] = self._rsi2(g['close']).shift(1)
        # SMA200
        g['sma200'] = self._sma(g['close'], 200)
        # ATR14
        g['atr14'] = self._atr(g, 14)
        return g

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(columns=['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','score','time_stop_bars'])

        frames: List[Dict] = []
        for sym, g in data.groupby('symbol'):
            g = self._compute_indicators(g)
            for _, row in g.iterrows():
                try:
                    price = float(row['close'])
                    if price < float(self.params.min_price):
                        continue
                    ibs_ok = (pd.notna(row['ibs_prev']) and float(row['ibs_prev']) < float(self.params.ibs_max))
                    rsi_ok = (pd.notna(row['rsi2_prev']) and float(row['rsi2_prev']) < float(self.params.rsi_max))
                    trend_ok = True
                    if self.params.sma200_filter:
                        trend_ok = (pd.notna(row['sma200']) and float(row['close']) >= float(row['sma200']))
                    atr = float(row['atr14']) if pd.notna(row['atr14']) else None
                    if not (ibs_ok and rsi_ok and trend_ok and atr is not None):
                        continue
                    # Entry at close (next-bar simulation handled by backtester)
                    entry = price
                    stop = round(entry - self.params.atr_mult * atr, 2)
                    r = entry - stop
                    take = round(entry + self.params.r_multiple * r, 2)
                    score = (float(self.params.ibs_max) - float(row['ibs_prev'])) * 100.0 + (float(self.params.rsi_max) - float(row['rsi2_prev']))
                    frames.append({
                        'timestamp': row['timestamp'],
                        'symbol': sym,
                        'side': 'long',
                        'entry_price': round(entry, 2),
                        'stop_loss': round(stop, 2),
                        'take_profit': round(take, 2),
                        'reason': f"IBS_RSI ibs={float(row['ibs_prev']):.2f} rsi2={float(row['rsi2_prev']):.1f}",
                        'score': round(score, 2),
                        'time_stop_bars': int(self.params.time_stop_bars),
                    })
                except Exception:
                    continue

        if not frames:
            return pd.DataFrame(columns=['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','score','time_stop_bars'])
        cols = ['timestamp','symbol','side','entry_price','stop_loss','take_profit','reason','score','time_stop_bars']
        return pd.DataFrame(frames, columns=cols)

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest-friendly: returns ALL bars where prior-bar conditions were met.
        Signals at close(t), fills assumed at open(t+1). Mirrors generate_signals.
        """
        return self.generate_signals(df)
