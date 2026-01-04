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

    v2.6 PARAMETERS - Autonomous Brain validated discoveries applied
    - IBS+RSI: rsi_entry=10.0 (validated +19.1% improvement)
    - Turtle Soup: ts_lookback=15 (validated +32% more signals)
    - Turtle Soup: ts_r_multiple=0.75 (validated +19% PF improvement)
    - VIX Filter: max_vix=25 (validated +10.1% PF improvement)
    """

    # IBS + RSI Parameters (v2.3 - VALIDATED DISCOVERY)
    ibs_entry: float = 0.08            # Was 0.15 - 47% tighter
    ibs_exit: float = 0.80
    rsi_period: int = 2
    rsi_entry: float = 10.0            # v2.3: Was 5.0 - validated +19.1% improvement
    rsi_exit: float = 70.0
    ibs_rsi_stop_mult: float = 2.0     # ATR multiplier for stop
    ibs_rsi_time_stop: int = 7         # Time stop in bars

    # Turtle Soup Parameters (v2.5 - OPTIMIZED FOR HIGHER PF)
    ts_lookback: int = 15              # v2.4: Was 20 - validated +32% more TS signals
    ts_min_bars_since_extreme: int = 3  # Aged extremes
    ts_min_sweep_strength: float = 0.3  # Looser sweep = more quality signals
    ts_stop_buffer_mult: float = 0.2    # Tight stop for higher WR
    ts_r_multiple: float = 0.75         # v2.5: Was 0.5 - validated +19% PF improvement
    ts_time_stop: int = 3               # Quick 3-bar time stop

    # Common Parameters
    sma_period: int = 200
    atr_period: int = 14
    time_stop_bars: int = 7             # Legacy - use strategy-specific time stops
    min_price: float = 15.0             # Higher liquidity only

    # VIX Filter (v2.6 - VALIDATED +10.1% PF IMPROVEMENT)
    use_vix_filter: bool = True         # Enable VIX-based filtering
    max_vix: float = 25.0               # v2.6: Block trades when VIX > 25

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

    Verified Performance (v2.6):
    - IBS+RSI: 64.8% WR, 1.68 PF
    - Turtle Soup: 61.7% WR, 1.63 PF
    - Combined: 64.5% WR, 1.68 PF
    - VIX Filter: Blocks trades when VIX > 25 (+10.1% PF)

    Neural Integration (v2.7):
    - Semantic memory rule application
    - Confidence boost/penalty based on learned rules
    """

    def __init__(self, params: Optional[DualStrategyParams] = None, preview_mode: bool = False,
                 vix_data: Optional[pd.DataFrame] = None, use_semantic_memory: bool = True):
        self.params = params or DualStrategyParams()
        self.preview_mode = preview_mode  # Use current bar values for weekend analysis
        self.vix_data = vix_data  # Optional VIX data for filtering (columns: timestamp, close)
        self._vix_cache: Dict[str, float] = {}  # Cache VIX values by date string
        self.use_semantic_memory = use_semantic_memory
        self._semantic_memory = None  # Lazy-loaded

        # Pre-process VIX data if provided
        if self.vix_data is not None and len(self.vix_data) > 0:
            self._build_vix_cache()

        # Initialize SMC detector if available and enabled
        self.smc_detector = None
        if SMC_AVAILABLE and self.params.use_smc_confluence:
            self.smc_detector = SmartMoneyDetector()

    @property
    def semantic_memory(self):
        """Lazy-load semantic memory to avoid circular imports."""
        if self._semantic_memory is None and self.use_semantic_memory:
            try:
                from cognitive.semantic_memory import get_semantic_memory
                self._semantic_memory = get_semantic_memory()
            except ImportError:
                pass  # Semantic memory not available
        return self._semantic_memory

    def _apply_semantic_rules(self, signal: Dict, symbol: str, strategy: str) -> Dict:
        """
        Apply learned semantic rules to adjust signal confidence.

        Queries semantic memory for applicable rules and adjusts the signal score
        based on past experience with similar patterns.
        """
        if not self.semantic_memory:
            return signal

        try:
            # Build context for rule matching
            context = {
                'symbol': symbol,
                'strategy': strategy,
                'pattern_type': signal.get('strategy', '').lower(),
            }

            # Query for applicable rules
            rules = self.semantic_memory.get_applicable_rules(context)

            if not rules:
                return signal

            # Apply rules
            score_adjustment = 0.0
            rules_applied = []

            for rule in rules:
                if not rule.is_active:
                    continue

                action = rule.action.lower()

                # Confidence adjustments
                if action == 'increase_confidence':
                    adjustment = rule.parameters.get('multiplier', 0.1) * rule.confidence
                    score_adjustment += adjustment
                    rules_applied.append(f"+{adjustment:.2f} ({rule.rule_id})")
                    # Record application
                    self.semantic_memory.record_rule_application(rule.rule_id)

                elif action == 'decrease_confidence':
                    adjustment = -rule.parameters.get('multiplier', 0.1) * rule.confidence
                    score_adjustment += adjustment
                    rules_applied.append(f"{adjustment:.2f} ({rule.rule_id})")
                    self.semantic_memory.record_rule_application(rule.rule_id)

                elif action == 'skip_trade':
                    # Strong negative signal - mark to skip
                    signal['semantic_skip'] = True
                    rules_applied.append(f"SKIP ({rule.rule_id})")
                    self.semantic_memory.record_rule_application(rule.rule_id)

            # Apply score adjustment (cap at +/- 0.3)
            score_adjustment = max(-0.3, min(0.3, score_adjustment))
            original_score = signal.get('score', 0.5)
            signal['score'] = round(original_score + score_adjustment, 2)
            signal['semantic_adjustment'] = round(score_adjustment, 2)
            signal['semantic_rules'] = rules_applied

        except Exception as e:
            # Log but don't fail - semantic memory is optional
            import logging
            logging.getLogger(__name__).debug(f"Semantic rule application error: {e}")

        return signal

    def _build_vix_cache(self):
        """Build VIX lookup cache by date."""
        if self.vix_data is None:
            return
        for _, row in self.vix_data.iterrows():
            ts = row.get('timestamp') or row.get('date')
            if ts is not None:
                date_str = pd.to_datetime(ts).strftime('%Y-%m-%d')
                self._vix_cache[date_str] = float(row.get('close', row.get('vix', 0)))

    def _get_vix(self, timestamp) -> Optional[float]:
        """Get VIX value for a given timestamp."""
        if not self._vix_cache:
            return None
        date_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d')
        return self._vix_cache.get(date_str)

    def _is_vix_ok(self, timestamp) -> bool:
        """Check if VIX is below threshold (trade allowed)."""
        if not self.params.use_vix_filter:
            return True  # VIX filter disabled
        vix = self._get_vix(timestamp)
        if vix is None:
            return True  # No VIX data, allow trade
        return vix <= self.params.max_vix

    def set_vix_data(self, vix_data: pd.DataFrame):
        """Set VIX data for filtering (can be called after init)."""
        self.vix_data = vix_data
        self._vix_cache = {}
        if vix_data is not None and len(vix_data) > 0:
            self._build_vix_cache()

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

                # VIX Filter: Skip if VIX > max_vix (v2.6)
                if not self._is_vix_ok(row['timestamp']):
                    continue

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

            # VIX Filter: Skip if VIX > max_vix (v2.6)
            if not self._is_vix_ok(row['timestamp']):
                continue

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

                signal = {
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
                }
                # Apply learned semantic rules (v2.7)
                signal = self._apply_semantic_rules(signal, sym, 'IBS_RSI')
                if not signal.get('semantic_skip'):
                    out.append(signal)

            # Check Turtle Soup
            is_ts, score, reason = self._check_turtle_soup_entry(row)
            if is_ts:
                entry = float(row['close'])
                atr_val = float(row['atr14'] if self.preview_mode else row['atr14_sig'])
                stop = float(row['low']) - self.params.ts_stop_buffer_mult * atr_val
                risk = entry - stop
                take_profit = entry + self.params.ts_r_multiple * risk if risk > 0 else None

                signal = {
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
                }
                # Apply learned semantic rules (v2.7)
                signal = self._apply_semantic_rules(signal, sym, 'TurtleSoup')
                if not signal.get('semantic_skip'):
                    out.append(signal)

        cols = ['timestamp', 'symbol', 'side', 'strategy', 'entry_price', 'stop_loss',
                'take_profit', 'reason', 'score', 'atr', 'time_stop_bars', 'ibs', 'rsi2', 'oversold_tier',
                'smc_confluence', 'smc_ob', 'smc_fvg', 'smc_choch', 'semantic_adjustment', 'semantic_rules']
        result = pd.DataFrame(out, columns=cols) if out else pd.DataFrame(columns=cols)

        if not result.empty:
            # DETERMINISM FIX: Use stable sort with tie-breakers (timestamp, symbol)
            result = result.sort_values(['score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')

        return result

    def get_top_picks(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get top N picks across both strategies."""
        signals = self.generate_signals(df)
        return signals.head(n) if not signals.empty else signals
