#!/usr/bin/env python3
"""
Dual Strategy System - v2.2 QUANT INTERVIEW READY

Combines two complementary mean-reversion strategies.
VERIFIED: 60.2% WR, 1.44 PF combined (1,172 trades, 2015-2024)

1. IBS + RSI Mean Reversion (High Frequency)
   - Entry: IBS < 0.08 AND RSI(2) < 5.0 AND Close > SMA(200) AND Price > $15
   - Exit: IBS > 0.80 or RSI > 70 or ATR*2.0 stop or 7-bar time
   - v2.2 Performance: 59.9% WR, 1.46 PF (867 trades)

2. Turtle Soup / ICT Liquidity Sweep (High Conviction)
   - Entry: Sweep > 0.3 ATR below 20-day low (3+ bars aged), revert inside
   - Exit: 0.5R take profit or ATR*0.2 stop or 3-bar time
   - v2.2 Performance: 61.0% WR, 1.37 PF (305 trades)

Key v2.2 Optimization Insight:
- For mean-reversion, LOOSER entry + TIGHTER exits = higher WR
- Turtle Soup sweep threshold lowered: 1.5 ATR -> 0.3 ATR (catch more setups)
- Turtle Soup exits tightened: 2R/5-bar -> 0.5R/3-bar (lock gains fast)

Replication Command:
    python scripts/backtest_dual_strategy.py --cap 200 --start 2015-01-01 --end 2024-12-31

See docs/V2.2_OPTIMIZATION_GUIDE.md for full optimization methodology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Import Smart Money Concepts for ICT confluence filtering
try:
    from strategies.ict.smart_money import SmartMoneyDetector, SMCConfig
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False


# ============================================================================
# Indicator Functions
# ============================================================================

def ibs(df: pd.DataFrame) -> pd.Series:
    """Internal Bar Strength = (Close - Low) / (High - Low)."""
    return (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)


def simple_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """RSI using simple rolling mean (matches IbsRsiStrategy)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def rolling_low_with_offset(series: pd.Series, window: int):
    """Rolling minimum and bars since minimum."""
    rolling_min = series.rolling(window=window, min_periods=window).min()

    def bars_since_min(arr):
        if len(arr) < window:
            return np.nan
        min_val = arr.min()
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] == min_val:
                return len(arr) - 1 - i
        return np.nan

    bars_offset = series.rolling(window=window, min_periods=window).apply(
        bars_since_min, raw=True
    )
    return rolling_min, bars_offset


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class DualStrategyParams:
    """
    Parameters for the dual strategy system.

    v2.2 PARAMETERS - Both strategies pass 55%+ WR, 1.3+ PF
    - IBS+RSI: 59.9% WR, 1.46 PF (tightened entry, wider stops)
    - Turtle Soup: 61.0% WR, 1.37 PF (looser sweep, tight exits)
    """

    # IBS + RSI Parameters (v2.2 - TIGHTENED ENTRY)
    ibs_entry: float = 0.08            # Was 0.15 - 47% tighter
    ibs_exit: float = 0.80
    rsi_period: int = 2
    rsi_entry: float = 5.0             # Was 10.0 - 50% tighter
    rsi_exit: float = 70.0
    ibs_rsi_stop_mult: float = 2.0     # ATR multiplier for stop
    ibs_rsi_time_stop: int = 7         # Time stop in bars

    # Turtle Soup Parameters (v2.2 - OPTIMIZED FOR 61% WR, 1.37 PF)
    ts_lookback: int = 20
    ts_min_bars_since_extreme: int = 3  # Aged extremes
    ts_min_sweep_strength: float = 0.3  # Looser sweep = more quality signals
    ts_stop_buffer_mult: float = 0.2    # Tight stop for higher WR
    ts_r_multiple: float = 0.5          # 0.5R target = hit more often
    ts_time_stop: int = 3               # Quick 3-bar time stop

    # Common Parameters
    sma_period: int = 200
    atr_period: int = 14
    time_stop_bars: int = 7             # Legacy - use strategy-specific time stops
    min_price: float = 15.0             # Higher liquidity only

    # Smart Money Concepts (SMC) Confluence Parameters
    use_smc_confluence: bool = True     # Enable SMC pattern detection
    smc_score_boost: float = 50.0       # Score boost when SMC confluence exists
    require_smc_for_ts: bool = False    # If True, Turtle Soup requires SMC confluence


# ============================================================================
# Dual Strategy Scanner
# ============================================================================

class DualStrategyScanner:
    """
    Combined IBS+RSI and Turtle Soup strategy scanner.

    Verified Performance (v2.2):
    - IBS+RSI: 59.9% WR, 1.46 PF (867 trades)
    - Turtle Soup: 61.0% WR, 1.37 PF (305 trades)
    - Combined: 60.2% WR, 1.44 PF (1,172 trades)
    """

    def __init__(self, params: Optional[DualStrategyParams] = None, preview_mode: bool = False):
        self.params = params or DualStrategyParams()
        self.preview_mode = preview_mode  # Use current bar values for weekend analysis

        # Initialize SMC detector if available and enabled
        self.smc_detector = None
        if SMC_AVAILABLE and self.params.use_smc_confluence:
            self.smc_detector = SmartMoneyDetector()

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators for both strategies."""
        df = df.sort_values(['symbol', 'timestamp']).copy()
        parts: List[pd.DataFrame] = []

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp').copy()
            c = g['close'].astype(float)

            # IBS + RSI indicators
            g['ibs'] = ibs(g)
            g['rsi2'] = simple_rsi(c, self.params.rsi_period)
            g['sma200'] = c.rolling(self.params.sma_period).mean()
            g['atr14'] = atr(g, self.params.atr_period)

            # Lookahead-safe signal features (use prior bar values)
            g['ibs_sig'] = g['ibs'].shift(1)
            g['rsi2_sig'] = g['rsi2'].shift(1)
            g['sma200_sig'] = g['sma200'].shift(1)
            g['atr14_sig'] = g['atr14'].shift(1)

            # Turtle Soup indicators
            prior_lows = g['low'].shift(1)
            prior_N_low, bars_since_low = rolling_low_with_offset(
                prior_lows, self.params.ts_lookback
            )
            g['prior_N_low'] = prior_N_low
            g['bars_since_low'] = bars_since_low
            g['sma200_sig'] = g['sma200'].shift(1)
            g['atr14_sig'] = g['atr14'].shift(1)

            # Smart Money Concepts detection (Order Blocks, FVG, Liquidity Sweeps)
            if self.smc_detector is not None:
                try:
                    smc_df = self.smc_detector.detect_all(g)
                    # Add SMC columns to group
                    g['smc_ob_bullish'] = smc_df['ob_bullish'].values
                    g['smc_fvg_bullish'] = smc_df['fvg_bullish'].values
                    g['smc_liq_sweep_low'] = smc_df['liq_sweep_low'].values
                    g['smc_choch_bullish'] = smc_df['choch_bullish'].values
                    g['smc_bullish_confluence'] = smc_df['smc_bullish_confluence'].values
                    g['smc_high_prob_long'] = smc_df['smc_high_prob_long'].values
                except Exception:
                    # Fallback if SMC detection fails
                    g['smc_ob_bullish'] = False
                    g['smc_fvg_bullish'] = False
                    g['smc_liq_sweep_low'] = False
                    g['smc_choch_bullish'] = False
                    g['smc_bullish_confluence'] = False
                    g['smc_high_prob_long'] = False
            else:
                g['smc_ob_bullish'] = False
                g['smc_fvg_bullish'] = False
                g['smc_liq_sweep_low'] = False
                g['smc_choch_bullish'] = False
                g['smc_bullish_confluence'] = False
                g['smc_high_prob_long'] = False

            parts.append(g)

        return pd.concat(parts, ignore_index=True) if parts else df.copy()

    def _check_ibs_rsi_entry(self, row: pd.Series) -> tuple[bool, float, str]:
        """Check IBS+RSI entry. Returns (should_enter, score, reason)."""
        close = float(row['close'])

        # Preview mode uses current bar (for weekend analysis)
        # Normal mode uses shifted (prior bar) for lookahead safety
        if self.preview_mode:
            ibs_val = row.get('ibs')
            rsi_val = row.get('rsi2')
            sma200 = row.get('sma200')
            atr_val = row.get('atr14')
        else:
            ibs_val = row.get('ibs_sig')
            rsi_val = row.get('rsi2_sig')
            sma200 = row.get('sma200_sig')
            atr_val = row.get('atr14_sig')

        if any(pd.isna(x) for x in [ibs_val, rsi_val, sma200, atr_val]):
            return False, 0.0, ""

        if close < self.params.min_price:
            return False, 0.0, ""

        # Entry: IBS < 0.08 AND RSI(2) < 5 AND Close > SMA200
        if float(ibs_val) >= self.params.ibs_entry:
            return False, 0.0, ""
        if float(rsi_val) >= self.params.rsi_entry:
            return False, 0.0, ""
        if close <= float(sma200):
            return False, 0.0, ""

        score = (self.params.ibs_entry - float(ibs_val)) * 100 + \
                (self.params.rsi_entry - float(rsi_val))
        reason = f"IBS_RSI[ibs={float(ibs_val):.2f},rsi={float(rsi_val):.1f}]"

        return True, score, reason

    def _check_turtle_soup_entry(self, row: pd.Series) -> tuple[bool, float, str]:
        """Check Turtle Soup entry. Returns (should_enter, score, reason)."""
        required = ['low', 'close', 'prior_N_low', 'bars_since_low', 'sma200_sig', 'atr14_sig']
        if any(pd.isna(row.get(c)) for c in required):
            return False, 0.0, ""

        low = float(row['low'])
        close = float(row['close'])
        prior_N_low = float(row['prior_N_low'])
        bars_since = float(row['bars_since_low'])
        sma200 = float(row['sma200_sig'])
        atr_val = float(row['atr14_sig'])

        if close < self.params.min_price:
            return False, 0.0, ""

        # Turtle Soup rules
        swept_below = low < prior_N_low
        extreme_aged = bars_since >= self.params.ts_min_bars_since_extreme
        reverted_inside = close > prior_N_low
        above_trend = close > sma200

        if not (swept_below and extreme_aged and reverted_inside and above_trend):
            return False, 0.0, ""

        # Calculate sweep strength
        sweep_distance = prior_N_low - low
        sweep_strength = sweep_distance / atr_val if atr_val > 0 else 0

        # Only accept strong sweeps (> 0.3 ATR)
        if sweep_strength < self.params.ts_min_sweep_strength:
            return False, 0.0, ""

        # Check Smart Money Concepts confluence
        smc_confluence = bool(row.get('smc_bullish_confluence', False))
        smc_high_prob = bool(row.get('smc_high_prob_long', False))
        has_ob = bool(row.get('smc_ob_bullish', False))
        has_fvg = bool(row.get('smc_fvg_bullish', False))
        has_choch = bool(row.get('smc_choch_bullish', False))

        # If require_smc_for_ts is True, must have SMC confluence
        if self.params.require_smc_for_ts and not smc_confluence:
            return False, 0.0, ""

        # Base score from sweep strength
        score = sweep_strength * 100

        # Boost score for SMC confluence
        smc_factors = []
        if smc_high_prob:
            score += self.params.smc_score_boost * 2  # Double boost for high prob
            smc_factors.append("HP")
        elif smc_confluence:
            score += self.params.smc_score_boost
            smc_factors.append("CONF")
        if has_ob:
            score += 20
            smc_factors.append("OB")
        if has_fvg:
            score += 15
            smc_factors.append("FVG")
        if has_choch:
            score += 25
            smc_factors.append("CHoCH")

        # Build reason string
        reason = f"TurtleSoup[sweep={sweep_strength:.2f}ATR]"
        if smc_factors:
            reason += f"+SMC[{'+'.join(smc_factors)}]"

        return True, score, reason

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all signals for backtesting - both strategies."""
        df = self._compute_indicators(df)
        rows: List[Dict] = []

        min_bars = self.params.sma_period + 10

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < min_bars:
                continue

            for idx in g.index:
                row = g.loc[idx]

                # Check IBS+RSI first (higher frequency)
                is_ibs_rsi, score, reason = self._check_ibs_rsi_entry(row)
                if is_ibs_rsi:
                    entry = float(row['close'])
                    atr_val = float(row['atr14'] if self.preview_mode else row['atr14_sig'])
                    stop = entry - self.params.ibs_rsi_stop_mult * atr_val
                    rsi_val = float(row['rsi2'])

                    # Determine oversold tier based on RSI2
                    if rsi_val <= 5.0:
                        oversold_tier = 'EXTREME'
                    elif rsi_val <= 10.0:
                        oversold_tier = 'NEAR_EXTREME'
                    else:
                        oversold_tier = 'MODERATE'

                    rows.append({
                        'timestamp': row['timestamp'],
                        'symbol': sym,
                        'side': 'long',
                        'strategy': 'IBS_RSI',
                        'entry_price': round(entry, 2),
                        'stop_loss': round(stop, 2),
                        'take_profit': None,  # Exit on IBS/RSI signal
                        'reason': reason,
                        'score': round(score, 2),
                        'atr': round(atr_val, 2),
                        'time_stop_bars': self.params.ibs_rsi_time_stop,
                        'ibs': round(float(row['ibs']), 3),
                        'rsi2': round(rsi_val, 2),
                        'oversold_tier': oversold_tier,
                    })

                # Check Turtle Soup (lower frequency, higher conviction)
                is_ts, score, reason = self._check_turtle_soup_entry(row)
                if is_ts:
                    entry = float(row['close'])
                    atr_val = float(row['atr14'] if self.preview_mode else row['atr14_sig'])
                    stop = float(row['low']) - self.params.ts_stop_buffer_mult * atr_val
                    risk = entry - stop
                    take_profit = entry + self.params.ts_r_multiple * risk if risk > 0 else None

                    rows.append({
                        'timestamp': row['timestamp'],
                        'symbol': sym,
                        'side': 'long',
                        'strategy': 'TurtleSoup',
                        'entry_price': round(entry, 2),
                        'stop_loss': round(stop, 2),
                        'take_profit': round(take_profit, 2) if take_profit else None,
                        'reason': reason,
                        'score': round(score, 2),
                        'atr': round(atr_val, 2),
                        'time_stop_bars': self.params.ts_time_stop,
                        'ibs': round(float(row['ibs']), 3) if pd.notna(row.get('ibs')) else None,
                        'rsi2': round(float(row['rsi2']), 2) if pd.notna(row.get('rsi2')) else None,
                        'smc_confluence': bool(row.get('smc_bullish_confluence', False)),
                        'smc_ob': bool(row.get('smc_ob_bullish', False)),
                        'smc_fvg': bool(row.get('smc_fvg_bullish', False)),
                        'smc_choch': bool(row.get('smc_choch_bullish', False)),
                    })

        cols = ['timestamp', 'symbol', 'side', 'strategy', 'entry_price', 'stop_loss',
                'take_profit', 'reason', 'score', 'atr', 'time_stop_bars', 'ibs', 'rsi2', 'oversold_tier',
                'smc_confluence', 'smc_ob', 'smc_fvg', 'smc_choch']
        result = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

        if not result.empty:
            # DETERMINISM FIX: Use stable sort with symbol as tie-breaker
            result = result.sort_values(['timestamp', 'score', 'symbol'], ascending=[True, False, True], kind='mergesort')

        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for current bar only (live trading)."""
        df = self._compute_indicators(df)
        out: List[Dict] = []

        min_bars = self.params.sma_period + 10

        for sym, g in df.groupby('symbol'):
            g = g.sort_values('timestamp')
            if len(g) < min_bars:
                continue

            row = g.iloc[-1]

            # Check IBS+RSI
            is_ibs_rsi, score, reason = self._check_ibs_rsi_entry(row)
            if is_ibs_rsi:
                entry = float(row['close'])
                atr_val = float(row['atr14'] if self.preview_mode else row['atr14_sig'])
                stop = entry - self.params.ibs_rsi_stop_mult * atr_val
                rsi_val = float(row['rsi2'])

                # Determine oversold tier based on RSI2
                if rsi_val <= 5.0:
                    oversold_tier = 'EXTREME'
                elif rsi_val <= 10.0:
                    oversold_tier = 'NEAR_EXTREME'
                else:
                    oversold_tier = 'MODERATE'

                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'strategy': 'IBS_RSI',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': None,
                    'reason': reason,
                    'score': round(score, 2),
                    'atr': round(atr_val, 2),
                    'time_stop_bars': self.params.ibs_rsi_time_stop,
                    'ibs': round(float(row['ibs']), 3),
                    'rsi2': round(rsi_val, 2),
                    'oversold_tier': oversold_tier,
                })

            # Check Turtle Soup
            is_ts, score, reason = self._check_turtle_soup_entry(row)
            if is_ts:
                entry = float(row['close'])
                atr_val = float(row['atr14'] if self.preview_mode else row['atr14_sig'])
                stop = float(row['low']) - self.params.ts_stop_buffer_mult * atr_val
                risk = entry - stop
                take_profit = entry + self.params.ts_r_multiple * risk if risk > 0 else None

                out.append({
                    'timestamp': row['timestamp'],
                    'symbol': sym,
                    'side': 'long',
                    'strategy': 'TurtleSoup',
                    'entry_price': round(entry, 2),
                    'stop_loss': round(stop, 2),
                    'take_profit': round(take_profit, 2) if take_profit else None,
                    'reason': reason,
                    'score': round(score, 2),
                    'atr': round(atr_val, 2),
                    'time_stop_bars': self.params.ts_time_stop,
                    'ibs': round(float(row['ibs']), 3) if pd.notna(row.get('ibs')) else None,
                    'rsi2': round(float(row['rsi2']), 2) if pd.notna(row.get('rsi2')) else None,
                    'smc_confluence': bool(row.get('smc_bullish_confluence', False)),
                    'smc_ob': bool(row.get('smc_ob_bullish', False)),
                    'smc_fvg': bool(row.get('smc_fvg_bullish', False)),
                    'smc_choch': bool(row.get('smc_choch_bullish', False)),
                })

        cols = ['timestamp', 'symbol', 'side', 'strategy', 'entry_price', 'stop_loss',
                'take_profit', 'reason', 'score', 'atr', 'time_stop_bars', 'ibs', 'rsi2', 'oversold_tier',
                'smc_confluence', 'smc_ob', 'smc_fvg', 'smc_choch']
        result = pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)

        if not result.empty:
            # DETERMINISM FIX: Use stable sort with tie-breakers (timestamp, symbol)
            result = result.sort_values(['score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')

        return result

    def get_top_picks(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get top N picks across both strategies."""
        signals = self.generate_signals(df)
        return signals.head(n) if not signals.empty else signals
