#!/usr/bin/env python3
"""
Turtle Soup Strategy (ICT Liquidity Sweep)

Original: Linda Bradford Raschke & Larry Connors, "Street Smarts" (1995)
ICT Alignment: Maps to "liquidity raid + revert" concept

Core Concept:
- Exploits failed breakouts by trading reversals after liquidity sweeps
- When price breaks a prior N-day high/low (triggering stops), then reverses
  back inside, it signals a failed breakout and potential reversal

Exact Rules (from Street Smarts):

LONG SETUP:
1. Today's low < prior 20-bar low (new breakout / liquidity sweep)
2. Prior 20-bar low must have been made at least 3 bars earlier
3. Today's close > prior 20-bar low (reverted back inside - failed breakout)
4. Entry: Next bar at open (or buy stop at prior low level)
5. Stop: Below the swept low (today's low) with ATR buffer
6. Exit: 2-6 bars (time stop) or trailing stop

SHORT SETUP (mirror):
1. Today's high > prior 20-bar high
2. Prior 20-bar high at least 3 bars earlier
3. Today's close < prior 20-bar high
4. Stop: Above swept high

Filters (for our system):
- SMA(200) trend filter (longs above, shorts below)
- Min price filter
- Regime filter (applied externally)
- Earnings blackout (applied externally)

No-Lookahead:
- All indicators computed on prior bars (shift by 1)
- Signals generated at close(t), fills at open(t+1)

References:
- Street Smarts: High Probability Short-Term Trading Strategies (1995)
- https://www.mql5.com/en/articles/2717
- https://oxfordstrat.com/trading-strategies/turtle-soup-plus-1/
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
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


def rolling_low_with_offset(series: pd.Series, window: int) -> tuple:
    """
    Compute rolling minimum and the bar offset of when that minimum occurred.
    Returns (rolling_min, bars_since_min).
    """
    rolling_min = series.rolling(window=window, min_periods=window).min()

    # Compute how many bars ago the minimum occurred
    def bars_since_min(arr):
        if len(arr) < window:
            return np.nan
        min_val = arr.min()
        # Find most recent occurrence of min
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] == min_val:
                return len(arr) - 1 - i
        return np.nan

    bars_offset = series.rolling(window=window, min_periods=window).apply(
        bars_since_min, raw=True
    )
    return rolling_min, bars_offset


def rolling_high_with_offset(series: pd.Series, window: int) -> tuple:
    """
    Compute rolling maximum and the bar offset of when that maximum occurred.
    Returns (rolling_max, bars_since_max).
    """
    rolling_max = series.rolling(window=window, min_periods=window).max()

    def bars_since_max(arr):
        if len(arr) < window:
            return np.nan
        max_val = arr.max()
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] == max_val:
                return len(arr) - 1 - i
        return np.nan

    bars_offset = series.rolling(window=window, min_periods=window).apply(
        bars_since_max, raw=True
    )
    return rolling_max, bars_offset


@dataclass
class TurtleSoupParams:
    """
    Parameters for Turtle Soup strategy.

    Defaults based on original Raschke/Connors rules from Street Smarts.
    """
    lookback: int = 20              # N-day channel (original: 20)
    min_bars_since_extreme: int = 3 # Prior extreme must be 3+ bars old (original rule)
    sma_period: int = 200           # Trend filter (longs above, shorts below)
    atr_period: int = 14            # ATR for stop calculation
    stop_buffer_mult: float = 0.5   # Stop = swept_level - ATR * buffer
    r_multiple: float = 2.0         # Take profit at 2R (or use time stop)
    time_stop_bars: int = 5         # Exit after N bars if no TP hit
    min_price: float = 10.0         # Minimum stock price filter
    allow_shorts: bool = False      # Long-only for v1 (safer)


class TurtleSoupStrategy:
    """
    Turtle Soup / ICT Liquidity Sweep Strategy

    Trades failed breakouts: when price sweeps beyond a prior N-day
    high/low (grabbing liquidity) then reverses back inside.
    """

    def __init__(self, params: Optional[TurtleSoupParams] = None):
        self.params = params or TurtleSoupParams()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute indicators per symbol with no lookahead.

        Key indicators:
        - prior_N_low: Lowest low of prior N bars (excluding today)
        - prior_N_high: Highest high of prior N bars (excluding today)
        - bars_since_low: How many bars ago the N-bar low occurred
        - bars_since_high: How many bars ago the N-bar high occurred
        - sma200: Trend filter
        - atr14: For stop calculation
        """
        df = df.sort_values(['symbol', 'timestamp']).copy()
        parts: List[pd.DataFrame] = []

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp').copy()

            # Prior N-bar low/high (shift by 1 to exclude current bar)
            # This is the "old breakout" level from Street Smarts
            prior_lows = g['low'].shift(1)
            prior_highs = g['high'].shift(1)

            # Rolling min/max of prior bars
            prior_N_low, bars_since_low = rolling_low_with_offset(
                prior_lows, self.params.lookback
            )
            prior_N_high, bars_since_high = rolling_high_with_offset(
                prior_highs, self.params.lookback
            )

            g['prior_N_low'] = prior_N_low
            g['prior_N_high'] = prior_N_high
            g['bars_since_low'] = bars_since_low
            g['bars_since_high'] = bars_since_high

            # Trend filter and ATR (shifted by 1 for no lookahead)
            g['sma200'] = sma(g['close'], self.params.sma_period)
            g['sma200_sig'] = g['sma200'].shift(1)
            g['atr14'] = atr(g, self.params.atr_period)
            g['atr14_sig'] = g['atr14'].shift(1)

            parts.append(g)

        return pd.concat(parts, ignore_index=True) if parts else df.copy()

    def _check_long_setup(self, row: pd.Series) -> bool:
        """
        Check if row meets Turtle Soup LONG setup criteria.

        From Street Smarts:
        1. Today's low < prior 20-bar low (sweep below - liquidity grab)
        2. Prior 20-bar low was made at least 3 bars earlier
        3. Today's close > prior 20-bar low (reverted back inside)
        4. Close > SMA(200) (trend filter)
        """
        # Guard against NaN
        required = ['low', 'close', 'prior_N_low', 'bars_since_low', 'sma200_sig']
        if any(pd.isna(row.get(c)) for c in required):
            return False

        low = float(row['low'])
        close = float(row['close'])
        prior_N_low = float(row['prior_N_low'])
        bars_since = float(row['bars_since_low'])
        sma200 = float(row['sma200_sig'])

        # Rule 1: Today's low sweeps below prior N-bar low
        swept_below = low < prior_N_low

        # Rule 2: Prior extreme at least 3 bars old (not a fresh extreme)
        extreme_aged = bars_since >= self.params.min_bars_since_extreme

        # Rule 3: Close reverted back above the prior low (failed breakout)
        reverted_inside = close > prior_N_low

        # Rule 4: Trend filter - above SMA(200)
        above_trend = close > sma200

        # Rule 5: Min price filter
        price_ok = close >= self.params.min_price

        return swept_below and extreme_aged and reverted_inside and above_trend and price_ok

    def _check_short_setup(self, row: pd.Series) -> bool:
        """
        Check if row meets Turtle Soup SHORT setup criteria.

        Mirror of long setup:
        1. Today's high > prior 20-bar high (sweep above)
        2. Prior 20-bar high was made at least 3 bars earlier
        3. Today's close < prior 20-bar high (reverted back inside)
        4. Close < SMA(200) (trend filter)
        """
        if not self.params.allow_shorts:
            return False

        required = ['high', 'close', 'prior_N_high', 'bars_since_high', 'sma200_sig']
        if any(pd.isna(row.get(c)) for c in required):
            return False

        high = float(row['high'])
        close = float(row['close'])
        prior_N_high = float(row['prior_N_high'])
        bars_since = float(row['bars_since_high'])
        sma200 = float(row['sma200_sig'])

        swept_above = high > prior_N_high
        extreme_aged = bars_since >= self.params.min_bars_since_extreme
        reverted_inside = close < prior_N_high
        below_trend = close < sma200
        price_ok = close >= self.params.min_price

        return swept_above and extreme_aged and reverted_inside and below_trend and price_ok

    def _compute_sweep_strength(self, row: pd.Series, side: str) -> float:
        """
        Compute sweep strength = how far price swept beyond the prior extreme,
        normalized by ATR. Higher = more aggressive liquidity grab.

        For ranking signals in Top-N selection.
        """
        atr_val = float(row['atr14_sig']) if pd.notna(row.get('atr14_sig')) else 1.0
        if atr_val <= 0:
            atr_val = 1.0

        if side == 'long':
            prior_level = float(row['prior_N_low'])
            swept_to = float(row['low'])
            sweep_distance = prior_level - swept_to  # How far below
        else:
            prior_level = float(row['prior_N_high'])
            swept_to = float(row['high'])
            sweep_distance = swept_to - prior_level  # How far above

        return round(sweep_distance / atr_val, 3)

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest-friendly: returns ALL bars where entry conditions are met.
        Signals at close(t), fills assumed at open(t+1).
        """
        df = self._compute_indicators(df)
        rows: List[Dict] = []

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            min_bars = max(self.params.lookback, self.params.sma_period) + 10
            if len(g) < min_bars:
                continue

            for idx, row in g.iterrows():
                side = None

                if self._check_long_setup(row):
                    side = 'long'
                    swept_level = float(row['prior_N_low'])
                    entry = float(row['close'])
                    atr_val = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else 0
                    # Stop below the swept low with ATR buffer
                    stop = float(row['low']) - self.params.stop_buffer_mult * atr_val

                elif self._check_short_setup(row):
                    side = 'short'
                    swept_level = float(row['prior_N_high'])
                    entry = float(row['close'])
                    atr_val = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else 0
                    stop = float(row['high']) + self.params.stop_buffer_mult * atr_val

                if side is None:
                    continue

                # Calculate take profit at R-multiple
                risk = abs(entry - stop)
                take_profit = None
                if risk > 0 and self.params.r_multiple > 0:
                    if side == 'long':
                        take_profit = entry + self.params.r_multiple * risk
                    else:
                        take_profit = entry - self.params.r_multiple * risk

                sweep_strength = self._compute_sweep_strength(row, side)

                rows.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': side,
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': round(take_profit, 2) if take_profit else None,
                    'reason': f"TurtleSoup sweep={sweep_strength:.2f}ATR",
                    'swept_level': round(swept_level, 2),
                    'sweep_strength': sweep_strength,
                    'time_stop_bars': int(self.params.time_stop_bars),
                })

        cols = ['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss',
                'take_profit', 'reason', 'swept_level', 'sweep_strength', 'time_stop_bars']
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Real-time: returns signals for the most recent bar per symbol.
        """
        df = self._compute_indicators(df)
        out: List[Dict] = []

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            min_bars = max(self.params.lookback, self.params.sma_period) + 10
            if len(g) < min_bars:
                continue

            row = g.iloc[-1]
            side = None

            if self._check_long_setup(row):
                side = 'long'
                swept_level = float(row['prior_N_low'])
                entry = float(row['close'])
                atr_val = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else 0
                stop = float(row['low']) - self.params.stop_buffer_mult * atr_val

            elif self._check_short_setup(row):
                side = 'short'
                swept_level = float(row['prior_N_high'])
                entry = float(row['close'])
                atr_val = float(row['atr14_sig']) if pd.notna(row['atr14_sig']) else 0
                stop = float(row['high']) + self.params.stop_buffer_mult * atr_val

            if side is None:
                continue

            risk = abs(entry - stop)
            take_profit = None
            if risk > 0 and self.params.r_multiple > 0:
                if side == 'long':
                    take_profit = entry + self.params.r_multiple * risk
                else:
                    take_profit = entry - self.params.r_multiple * risk

            sweep_strength = self._compute_sweep_strength(row, side)

            out.append({
                'timestamp': row['timestamp'],
                'symbol': sym,
                'side': side,
                'entry_price': round(entry, 2),
                'stop_loss': round(stop, 2),
                'take_profit': round(take_profit, 2) if take_profit else None,
                'reason': f"TurtleSoup sweep={sweep_strength:.2f}ATR",
                'swept_level': round(swept_level, 2),
                'sweep_strength': sweep_strength,
                'time_stop_bars': int(self.params.time_stop_bars),
            })

        cols = ['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss',
                'take_profit', 'reason', 'swept_level', 'sweep_strength', 'time_stop_bars']
        return pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)


# Verification with synthetic data
if __name__ == '__main__':
    import numpy as np

    np.random.seed(42)
    n = 300
    dates = pd.date_range('2020-01-01', periods=n, freq='B')

    # Create price series with some sweep patterns
    base = 100.0
    prices = [base]
    for i in range(1, n):
        # Add some mean-reverting behavior
        change = np.random.randn() * 1.0
        if prices[-1] > 110:
            change -= 0.5
        elif prices[-1] < 90:
            change += 0.5
        prices.append(prices[-1] + change)

    prices = np.array(prices)
    high = prices + np.abs(np.random.randn(n)) * 0.8
    low = prices - np.abs(np.random.randn(n)) * 0.8

    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': prices + np.random.randn(n) * 0.3,
        'high': high,
        'low': low,
        'close': prices,
        'volume': np.random.randint(1_000_000, 5_000_000, n),
    })

    strat = TurtleSoupStrategy(TurtleSoupParams(min_price=5.0))
    signals = strat.scan_signals_over_time(df)

    print(f"Turtle Soup Signals: {len(signals)}")
    if not signals.empty:
        print("\nSample signals:")
        print(signals[['timestamp', 'symbol', 'side', 'entry_price', 'stop_loss',
                       'sweep_strength', 'reason']].head(10))
        print(f"\nSweep strength range: {signals['sweep_strength'].min():.2f} - {signals['sweep_strength'].max():.2f}")
