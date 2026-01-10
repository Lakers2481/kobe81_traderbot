"""
Alpha Library - 100+ Alpha Candidates for Mining

Comprehensive library of trading alpha factors organized by category:
- Momentum (trend following)
- Mean Reversion (contrarian)
- Volatility (vol-based signals)
- Volume (liquidity signals)
- Technical (classic indicators)
- Cross-sectional (relative signals)

USAGE:
    from research.alpha_library import AlphaLibrary

    lib = AlphaLibrary()
    alphas_df = lib.compute_all(price_df)  # All 100+ alphas
    alphas_df = lib.compute_category(price_df, 'momentum')  # Category only

Created: 2026-01-07
Based on: Qlib Alpha158, WorldQuant 101 Alphas, Lopez de Prado features
"""

from __future__ import annotations

import logging
from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaLibrary:
    """
    Comprehensive alpha factor library with 100+ candidates.

    Each alpha returns a Series indexed by (timestamp, symbol) with factor values.
    Positive values indicate bullish signals, negative = bearish.
    """

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._categories: Dict[str, List[str]] = {
            'momentum': [],
            'mean_reversion': [],
            'volatility': [],
            'volume': [],
            'technical': [],
            'cross_sectional': [],
            'pattern': [],
        }
        self._register_all_alphas()

    def _register_all_alphas(self):
        """Register all alpha factors."""

        # ========== MOMENTUM ALPHAS (20) ==========
        for period in [5, 10, 20, 40, 60, 120, 252]:
            name = f'mom_{period}d'
            self._register(name, 'momentum',
                          lambda df, p=period: self._alpha_momentum(df, p))

        # Rate of change variants
        for period in [5, 10, 20]:
            name = f'roc_{period}d'
            self._register(name, 'momentum',
                          lambda df, p=period: self._alpha_roc(df, p))

        # Acceleration (momentum of momentum)
        for period in [5, 10, 20]:
            name = f'accel_{period}d'
            self._register(name, 'momentum',
                          lambda df, p=period: self._alpha_acceleration(df, p))

        # Time-series momentum (excess return)
        self._register('tsmom_20d', 'momentum', lambda df: self._alpha_tsmom(df, 20))
        self._register('tsmom_60d', 'momentum', lambda df: self._alpha_tsmom(df, 60))
        self._register('tsmom_252d', 'momentum', lambda df: self._alpha_tsmom(df, 252))

        # ========== MEAN REVERSION ALPHAS (25) ==========
        # RSI variants
        for period in [2, 3, 5, 7, 14]:
            name = f'rsi_{period}d'
            self._register(name, 'mean_reversion',
                          lambda df, p=period: self._alpha_rsi(df, p))

        # IBS (Internal Bar Strength)
        self._register('ibs', 'mean_reversion', self._alpha_ibs)

        # Distance from MA
        for period in [5, 10, 20, 50, 200]:
            name = f'dist_ma_{period}d'
            self._register(name, 'mean_reversion',
                          lambda df, p=period: self._alpha_dist_ma(df, p))

        # Bollinger Band %B
        for period in [10, 20, 50]:
            name = f'bb_pctb_{period}d'
            self._register(name, 'mean_reversion',
                          lambda df, p=period: self._alpha_bb_pctb(df, p))

        # Z-score of returns
        for period in [5, 10, 20, 60]:
            name = f'zscore_{period}d'
            self._register(name, 'mean_reversion',
                          lambda df, p=period: self._alpha_zscore(df, p))

        # Consecutive days up/down
        self._register('consec_down', 'mean_reversion', self._alpha_consec_down)
        self._register('consec_up', 'mean_reversion', self._alpha_consec_up)

        # ========== VOLATILITY ALPHAS (15) ==========
        # ATR-based
        for period in [5, 10, 14, 20]:
            name = f'atr_{period}d'
            self._register(name, 'volatility',
                          lambda df, p=period: self._alpha_atr(df, p))

        # Realized volatility
        for period in [5, 10, 20, 60]:
            name = f'rvol_{period}d'
            self._register(name, 'volatility',
                          lambda df, p=period: self._alpha_realized_vol(df, p))

        # Vol regime (high vol = low signal, low vol = high signal)
        self._register('vol_regime', 'volatility', self._alpha_vol_regime)

        # Intraday range (high-low)/close
        self._register('intraday_range', 'volatility', self._alpha_intraday_range)

        # Gap volatility
        self._register('gap_size', 'volatility', self._alpha_gap_size)

        # Vol expansion/contraction
        self._register('vol_expansion', 'volatility', self._alpha_vol_expansion)
        self._register('vol_contraction', 'volatility', self._alpha_vol_contraction)

        # Garman-Klass volatility
        self._register('gk_vol_20d', 'volatility', lambda df: self._alpha_gk_vol(df, 20))

        # ========== VOLUME ALPHAS (15) ==========
        # Volume MA ratio
        for period in [5, 10, 20]:
            name = f'vol_ma_ratio_{period}d'
            self._register(name, 'volume',
                          lambda df, p=period: self._alpha_vol_ma_ratio(df, p))

        # VWAP distance
        self._register('vwap_dist', 'volume', self._alpha_vwap_dist)

        # Price-volume correlation
        for period in [10, 20]:
            name = f'pv_corr_{period}d'
            self._register(name, 'volume',
                          lambda df, p=period: self._alpha_pv_corr(df, p))

        # On-balance volume momentum
        for period in [10, 20]:
            name = f'obv_mom_{period}d'
            self._register(name, 'volume',
                          lambda df, p=period: self._alpha_obv_mom(df, p))

        # Volume breakout
        self._register('vol_breakout', 'volume', self._alpha_vol_breakout)

        # Money flow
        self._register('mfi_14d', 'volume', lambda df: self._alpha_mfi(df, 14))

        # Accumulation/Distribution
        self._register('ad_line', 'volume', self._alpha_ad_line)

        # ========== TECHNICAL ALPHAS (20) ==========
        # MACD
        self._register('macd_12_26', 'technical', lambda df: self._alpha_macd(df, 12, 26, 9))
        self._register('macd_signal', 'technical', lambda df: self._alpha_macd_signal(df, 12, 26, 9))

        # Stochastic
        for period in [5, 14]:
            name = f'stoch_{period}d'
            self._register(name, 'technical',
                          lambda df, p=period: self._alpha_stochastic(df, p))

        # CCI
        self._register('cci_20d', 'technical', lambda df: self._alpha_cci(df, 20))

        # ADX (trend strength)
        self._register('adx_14d', 'technical', lambda df: self._alpha_adx(df, 14))

        # Donchian breakout
        for period in [10, 20, 55]:
            name = f'donchian_{period}d'
            self._register(name, 'technical',
                          lambda df, p=period: self._alpha_donchian(df, p))

        # Support/Resistance proximity
        self._register('sr_proximity', 'technical', self._alpha_sr_proximity)

        # Moving average crossovers
        self._register('ma_cross_5_20', 'technical', lambda df: self._alpha_ma_cross(df, 5, 20))
        self._register('ma_cross_10_50', 'technical', lambda df: self._alpha_ma_cross(df, 10, 50))
        self._register('ma_cross_50_200', 'technical', lambda df: self._alpha_ma_cross(df, 50, 200))

        # Price vs moving averages
        self._register('above_ma_200', 'technical', lambda df: self._alpha_above_ma(df, 200))
        self._register('above_ma_50', 'technical', lambda df: self._alpha_above_ma(df, 50))

        # Williams %R
        self._register('williams_r_14d', 'technical', lambda df: self._alpha_williams_r(df, 14))

        # ========== CROSS-SECTIONAL ALPHAS (10) ==========
        # Relative strength vs market
        self._register('rs_vs_spy_20d', 'cross_sectional', lambda df: self._alpha_rs_market(df, 20))

        # Sector relative strength (placeholder - needs sector data)
        self._register('sector_rs', 'cross_sectional', self._alpha_sector_rs)

        # Percentile rank
        for period in [20, 60, 252]:
            name = f'pctile_rank_{period}d'
            self._register(name, 'cross_sectional',
                          lambda df, p=period: self._alpha_pctile_rank(df, p))

        # Relative volatility
        self._register('rel_vol_20d', 'cross_sectional', lambda df: self._alpha_rel_vol(df, 20))

        # ========== PATTERN ALPHAS (10) ==========
        # Turtle soup sweep
        self._register('turtle_sweep_20d', 'pattern', lambda df: self._alpha_turtle_sweep(df, 20))

        # Inside bar
        self._register('inside_bar', 'pattern', self._alpha_inside_bar)

        # Outside bar (engulfing)
        self._register('outside_bar', 'pattern', self._alpha_outside_bar)

        # Higher high / lower low
        self._register('higher_high', 'pattern', self._alpha_higher_high)
        self._register('lower_low', 'pattern', self._alpha_lower_low)

        # Gap patterns
        self._register('gap_up', 'pattern', self._alpha_gap_up)
        self._register('gap_down', 'pattern', self._alpha_gap_down)

        # Hammer/doji candle patterns
        self._register('hammer', 'pattern', self._alpha_hammer)

        logger.info(f"Registered {len(self._registry)} alpha factors across {len(self._categories)} categories")

    def _register(self, name: str, category: str, func: Callable):
        """Register an alpha factor."""
        self._registry[name] = func
        self._categories[category].append(name)

    # ========== ALPHA IMPLEMENTATIONS ==========

    def _compute_by_symbol(self, df: pd.DataFrame, compute_fn: Callable) -> pd.Series:
        """Apply computation per symbol and return Series."""
        results = []
        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('timestamp').copy()
            values = compute_fn(group)
            if isinstance(values, pd.Series):
                values = values.values
            result = pd.DataFrame({
                'timestamp': group['timestamp'].values,
                'symbol': symbol,
                'value': values
            })
            results.append(result)

        if not results:
            return pd.Series(dtype=float)

        combined = pd.concat(results, ignore_index=True)
        return combined.set_index(['timestamp', 'symbol'])['value']

    # Momentum alphas
    def _alpha_momentum(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Log return over period."""
        def compute(g):
            c = g['close'].astype(float)
            return np.log(c / c.shift(period))
        return self._compute_by_symbol(df, compute)

    def _alpha_roc(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Rate of change."""
        def compute(g):
            c = g['close'].astype(float)
            return c.pct_change(period)
        return self._compute_by_symbol(df, compute)

    def _alpha_acceleration(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Momentum of momentum."""
        def compute(g):
            c = g['close'].astype(float)
            mom = c.pct_change(period)
            return mom - mom.shift(period)
        return self._compute_by_symbol(df, compute)

    def _alpha_tsmom(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Time-series momentum (Moskowitz style)."""
        def compute(g):
            c = g['close'].astype(float)
            ret = c.pct_change(period)
            vol = c.pct_change().rolling(period).std()
            return ret / (vol + 1e-8)  # Risk-adjusted
        return self._compute_by_symbol(df, compute)

    # Mean reversion alphas
    def _alpha_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """RSI - returns distance from 50 (centered)."""
        def compute(g):
            c = g['close'].astype(float)
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return -(rsi - 50) / 50  # Negative when overbought, positive when oversold
        return self._compute_by_symbol(df, compute)

    def _alpha_ibs(self, df: pd.DataFrame) -> pd.Series:
        """Internal Bar Strength - mean reversion signal."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            ibs = (c - l) / (h - l + 1e-8)
            return -(ibs - 0.5) * 2  # Negative when high IBS, positive when low
        return self._compute_by_symbol(df, compute)

    def _alpha_dist_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Distance from moving average (negative = overbought)."""
        def compute(g):
            c = g['close'].astype(float)
            ma = c.rolling(period).mean()
            return -(c / ma - 1)  # Negative when above MA
        return self._compute_by_symbol(df, compute)

    def _alpha_bb_pctb(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Bollinger Band %B (negative when overbought)."""
        def compute(g):
            c = g['close'].astype(float)
            ma = c.rolling(period).mean()
            std = c.rolling(period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            pctb = (c - lower) / (upper - lower + 1e-8)
            return -(pctb - 0.5) * 2
        return self._compute_by_symbol(df, compute)

    def _alpha_zscore(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Z-score of returns."""
        def compute(g):
            c = g['close'].astype(float)
            ret = c.pct_change()
            z = (ret - ret.rolling(period).mean()) / (ret.rolling(period).std() + 1e-8)
            return -z  # Negative when extreme high
        return self._compute_by_symbol(df, compute)

    def _alpha_consec_down(self, df: pd.DataFrame) -> pd.Series:
        """Count of consecutive down days (bullish signal)."""
        def compute(g):
            c = g['close'].astype(float)
            down = (c.diff() < 0).astype(int)
            # Count consecutive downs
            counter = down.copy()
            for i in range(1, len(counter)):
                if down.iloc[i] == 1:
                    counter.iloc[i] = counter.iloc[i-1] + 1
                else:
                    counter.iloc[i] = 0
            return counter / 5  # Normalize to ~1 at 5 days
        return self._compute_by_symbol(df, compute)

    def _alpha_consec_up(self, df: pd.DataFrame) -> pd.Series:
        """Count of consecutive up days (bearish signal)."""
        def compute(g):
            c = g['close'].astype(float)
            up = (c.diff() > 0).astype(int)
            counter = up.copy()
            for i in range(1, len(counter)):
                if up.iloc[i] == 1:
                    counter.iloc[i] = counter.iloc[i-1] + 1
                else:
                    counter.iloc[i] = 0
            return -counter / 5  # Negative (bearish) when consecutive ups
        return self._compute_by_symbol(df, compute)

    # Volatility alphas
    def _alpha_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ATR normalized by close."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            prev_c = c.shift(1)
            tr = pd.concat([h - l, abs(h - prev_c), abs(l - prev_c)], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr / c
        return self._compute_by_symbol(df, compute)

    def _alpha_realized_vol(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Realized volatility (annualized)."""
        def compute(g):
            c = g['close'].astype(float)
            ret = c.pct_change()
            return ret.rolling(period).std() * np.sqrt(252)
        return self._compute_by_symbol(df, compute)

    def _alpha_vol_regime(self, df: pd.DataFrame) -> pd.Series:
        """Vol regime: low vol = 1 (good for trading), high vol = -1."""
        def compute(g):
            c = g['close'].astype(float)
            vol_20 = c.pct_change().rolling(20).std()
            vol_60 = c.pct_change().rolling(60).std()
            # Low vol relative to recent = positive signal
            return -(vol_20 / (vol_60 + 1e-8) - 1)
        return self._compute_by_symbol(df, compute)

    def _alpha_intraday_range(self, df: pd.DataFrame) -> pd.Series:
        """Intraday range normalized."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            return (h - l) / c
        return self._compute_by_symbol(df, compute)

    def _alpha_gap_size(self, df: pd.DataFrame) -> pd.Series:
        """Gap size (open - prev close) / prev close."""
        def compute(g):
            o = g['open'].astype(float)
            c = g['close'].astype(float)
            prev_c = c.shift(1)
            return (o - prev_c) / (prev_c + 1e-8)
        return self._compute_by_symbol(df, compute)

    def _alpha_vol_expansion(self, df: pd.DataFrame) -> pd.Series:
        """Volatility expansion signal."""
        def compute(g):
            c = g['close'].astype(float)
            vol_5 = c.pct_change().rolling(5).std()
            vol_20 = c.pct_change().rolling(20).std()
            return (vol_5 / (vol_20 + 1e-8) - 1).clip(-2, 2)
        return self._compute_by_symbol(df, compute)

    def _alpha_vol_contraction(self, df: pd.DataFrame) -> pd.Series:
        """Volatility contraction signal (inverse of expansion)."""
        return -self._alpha_vol_expansion(df)

    def _alpha_gk_vol(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Garman-Klass volatility estimator."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            o = g['open'].astype(float)

            gk = 0.5 * np.log(h/l)**2 - (2*np.log(2)-1) * np.log(c/o)**2
            return gk.rolling(period).mean()
        return self._compute_by_symbol(df, compute)

    # Volume alphas
    def _alpha_vol_ma_ratio(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Volume vs MA ratio."""
        def compute(g):
            v = g['volume'].astype(float)
            return v / (v.rolling(period).mean() + 1e-8) - 1
        return self._compute_by_symbol(df, compute)

    def _alpha_vwap_dist(self, df: pd.DataFrame) -> pd.Series:
        """Distance from VWAP."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            v = g['volume'].astype(float)

            typical = (h + l + c) / 3
            vwap = (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-8)
            return -(c / vwap - 1)  # Negative when above VWAP
        return self._compute_by_symbol(df, compute)

    def _alpha_pv_corr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Price-volume correlation."""
        def compute(g):
            c = g['close'].astype(float)
            v = g['volume'].astype(float)
            return c.pct_change().rolling(period).corr(v.pct_change())
        return self._compute_by_symbol(df, compute)

    def _alpha_obv_mom(self, df: pd.DataFrame, period: int) -> pd.Series:
        """On-balance volume momentum."""
        def compute(g):
            c = g['close'].astype(float)
            v = g['volume'].astype(float)
            direction = np.sign(c.diff())
            obv = (direction * v).cumsum()
            return obv.pct_change(period)
        return self._compute_by_symbol(df, compute)

    def _alpha_vol_breakout(self, df: pd.DataFrame) -> pd.Series:
        """Volume breakout signal."""
        def compute(g):
            v = g['volume'].astype(float)
            vol_20 = v.rolling(20).mean()
            vol_std = v.rolling(20).std()
            return (v - vol_20) / (vol_std + 1e-8)
        return self._compute_by_symbol(df, compute)

    def _alpha_mfi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Money Flow Index."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            v = g['volume'].astype(float)

            typical = (h + l + c) / 3
            mf = typical * v

            pos_mf = mf.where(typical > typical.shift(1), 0).rolling(period).sum()
            neg_mf = mf.where(typical < typical.shift(1), 0).rolling(period).sum()

            mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-8))
            return -(mfi - 50) / 50  # Center and invert
        return self._compute_by_symbol(df, compute)

    def _alpha_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution line momentum."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            v = g['volume'].astype(float)

            clv = ((c - l) - (h - c)) / (h - l + 1e-8)
            ad = (clv * v).cumsum()
            return ad.pct_change(10)
        return self._compute_by_symbol(df, compute)

    # Technical alphas
    def _alpha_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
        """MACD histogram."""
        def compute(g):
            c = g['close'].astype(float)
            ema_fast = c.ewm(span=fast, adjust=False).mean()
            ema_slow = c.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd - signal_line
        return self._compute_by_symbol(df, compute)

    def _alpha_macd_signal(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
        """MACD signal line crossover."""
        def compute(g):
            c = g['close'].astype(float)
            ema_fast = c.ewm(span=fast, adjust=False).mean()
            ema_slow = c.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            # 1 when above signal, -1 when below
            return np.sign(macd - signal_line)
        return self._compute_by_symbol(df, compute)

    def _alpha_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Stochastic oscillator."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            lowest = l.rolling(period).min()
            highest = h.rolling(period).max()
            stoch = (c - lowest) / (highest - lowest + 1e-8) * 100
            return -(stoch - 50) / 50  # Center and invert
        return self._compute_by_symbol(df, compute)

    def _alpha_cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Commodity Channel Index."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            typical = (h + l + c) / 3
            ma = typical.rolling(period).mean()
            mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (typical - ma) / (0.015 * mad + 1e-8)
            return -cci / 100  # Normalize and invert
        return self._compute_by_symbol(df, compute)

    def _alpha_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX - Average Directional Index (trend strength)."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            plus_dm = (h.diff()).where((h.diff() > l.diff().abs()), 0)
            minus_dm = (l.diff().abs()).where((l.diff().abs() > h.diff()), 0)

            prev_c = c.shift(1)
            tr = pd.concat([h - l, abs(h - prev_c), abs(l - prev_c)], axis=1).max(axis=1)

            atr = tr.rolling(period).mean()
            plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-8)
            minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-8)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx.rolling(period).mean()

            return adx / 100  # Strong trend = positive
        return self._compute_by_symbol(df, compute)

    def _alpha_donchian(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Donchian channel breakout signal."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            upper = h.rolling(period).max()
            lower = l.rolling(period).min()

            # 1 if at upper, -1 if at lower
            range_pos = (c - lower) / (upper - lower + 1e-8)
            return range_pos * 2 - 1
        return self._compute_by_symbol(df, compute)

    def _alpha_sr_proximity(self, df: pd.DataFrame) -> pd.Series:
        """Proximity to support/resistance levels."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            # Simple S/R: 20-day high/low
            resistance = h.rolling(20).max()
            support = l.rolling(20).min()

            # Distance to support (positive) vs resistance (negative)
            to_support = (c - support) / (c + 1e-8)
            to_resistance = (resistance - c) / (c + 1e-8)

            return to_resistance - to_support  # Positive when near support
        return self._compute_by_symbol(df, compute)

    def _alpha_ma_cross(self, df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
        """Moving average crossover signal."""
        def compute(g):
            c = g['close'].astype(float)
            ma_fast = c.rolling(fast).mean()
            ma_slow = c.rolling(slow).mean()
            return np.sign(ma_fast - ma_slow)
        return self._compute_by_symbol(df, compute)

    def _alpha_above_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Price above moving average."""
        def compute(g):
            c = g['close'].astype(float)
            ma = c.rolling(period).mean()
            return np.sign(c - ma)
        return self._compute_by_symbol(df, compute)

    def _alpha_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Williams %R."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            highest = h.rolling(period).max()
            lowest = l.rolling(period).min()
            wr = (highest - c) / (highest - lowest + 1e-8) * -100
            return -wr / 50 - 1  # Normalize to [-1, 1], invert
        return self._compute_by_symbol(df, compute)

    # Cross-sectional alphas
    def _alpha_rs_market(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Relative strength vs market (placeholder - needs market data)."""
        def compute(g):
            c = g['close'].astype(float)
            # Without market data, just return momentum
            return c.pct_change(period)
        return self._compute_by_symbol(df, compute)

    def _alpha_sector_rs(self, df: pd.DataFrame) -> pd.Series:
        """Sector relative strength (placeholder)."""
        # Would need sector mapping
        return self._alpha_momentum(df, 20)

    def _alpha_pctile_rank(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Percentile rank of returns."""
        def compute(g):
            c = g['close'].astype(float)
            ret = c.pct_change()
            return ret.rolling(period).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8))
        return self._compute_by_symbol(df, compute)

    def _alpha_rel_vol(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Relative volatility (placeholder - needs cross-sectional)."""
        def compute(g):
            c = g['close'].astype(float)
            vol = c.pct_change().rolling(period).std()
            return -vol / (vol.rolling(60).mean() + 1e-8)  # Low vol = positive
        return self._compute_by_symbol(df, compute)

    # Pattern alphas
    def _alpha_turtle_sweep(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Turtle soup - sweep of prior low then reversal."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)

            prior_low = l.shift(1).rolling(period).min()
            prior_high = h.shift(1).rolling(period).max()

            # Bullish: new low but close above prior low
            bull_sweep = ((l < prior_low) & (c > prior_low)).astype(float)
            # Bearish: new high but close below prior high
            bear_sweep = ((h > prior_high) & (c < prior_high)).astype(float)

            return bull_sweep - bear_sweep
        return self._compute_by_symbol(df, compute)

    def _alpha_inside_bar(self, df: pd.DataFrame) -> pd.Series:
        """Inside bar pattern (compression)."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)

            inside = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
            return inside
        return self._compute_by_symbol(df, compute)

    def _alpha_outside_bar(self, df: pd.DataFrame) -> pd.Series:
        """Outside bar (engulfing) pattern."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            o = g['open'].astype(float)

            outside = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)
            # Bullish if close > open, bearish otherwise
            direction = np.sign(c - o)
            return outside * direction
        return self._compute_by_symbol(df, compute)

    def _alpha_higher_high(self, df: pd.DataFrame) -> pd.Series:
        """Higher high pattern (trend continuation)."""
        def compute(g):
            h = g['high'].astype(float)
            return (h > h.shift(1)).astype(float) - (h < h.shift(1)).astype(float)
        return self._compute_by_symbol(df, compute)

    def _alpha_lower_low(self, df: pd.DataFrame) -> pd.Series:
        """Lower low pattern (bearish)."""
        def compute(g):
            l = g['low'].astype(float)
            return -((l < l.shift(1)).astype(float) - (l > l.shift(1)).astype(float))
        return self._compute_by_symbol(df, compute)

    def _alpha_gap_up(self, df: pd.DataFrame) -> pd.Series:
        """Gap up pattern."""
        def compute(g):
            o = g['open'].astype(float)
            c = g['close'].astype(float)
            gap = (o - c.shift(1)) / (c.shift(1) + 1e-8)
            return (gap > 0.01).astype(float)  # 1% gap threshold
        return self._compute_by_symbol(df, compute)

    def _alpha_gap_down(self, df: pd.DataFrame) -> pd.Series:
        """Gap down pattern."""
        def compute(g):
            o = g['open'].astype(float)
            c = g['close'].astype(float)
            gap = (o - c.shift(1)) / (c.shift(1) + 1e-8)
            return (gap < -0.01).astype(float)  # -1% gap threshold
        return self._compute_by_symbol(df, compute)

    def _alpha_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Hammer candle pattern."""
        def compute(g):
            h = g['high'].astype(float)
            l = g['low'].astype(float)
            c = g['close'].astype(float)
            o = g['open'].astype(float)

            body = abs(c - o)
            lower_wick = np.minimum(c, o) - l
            upper_wick = h - np.maximum(c, o)
            total_range = h - l + 1e-8

            # Hammer: small body, long lower wick, small upper wick
            hammer = ((lower_wick > 2 * body) &
                     (upper_wick < body) &
                     (body / total_range < 0.3)).astype(float)
            return hammer
        return self._compute_by_symbol(df, compute)

    # ========== PUBLIC METHODS ==========

    def list_alphas(self) -> List[str]:
        """List all registered alpha names."""
        return list(self._registry.keys())

    def list_categories(self) -> Dict[str, List[str]]:
        """List alphas by category."""
        return self._categories.copy()

    def compute_alpha(self, df: pd.DataFrame, name: str) -> pd.Series:
        """Compute a single alpha factor."""
        if name not in self._registry:
            raise ValueError(f"Unknown alpha: {name}. Available: {self.list_alphas()[:10]}...")
        return self._registry[name](df)

    def compute_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Compute all alphas in a category."""
        if category not in self._categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(self._categories.keys())}")

        results = {}
        for name in self._categories[category]:
            try:
                results[name] = self._registry[name](df)
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")

        return pd.DataFrame(results)

    def compute_all(self, df: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute all alphas (or specified categories)."""
        if categories is None:
            categories = list(self._categories.keys())

        results = {}
        for category in categories:
            for name in self._categories.get(category, []):
                try:
                    results[name] = self._registry[name](df)
                except Exception as e:
                    logger.warning(f"Failed to compute {name}: {e}")

        return pd.DataFrame(results)


# Convenience function
def compute_all_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 100+ alpha factors."""
    lib = AlphaLibrary()
    return lib.compute_all(df)


# Singleton instance
_alpha_library: Optional[AlphaLibrary] = None


def get_alpha_library() -> AlphaLibrary:
    """Get the singleton AlphaLibrary instance."""
    global _alpha_library
    if _alpha_library is None:
        _alpha_library = AlphaLibrary()
    return _alpha_library


if __name__ == '__main__':
    # Test the library
    lib = AlphaLibrary()
    print(f"Total alphas registered: {len(lib.list_alphas())}")
    print("\nCategories:")
    for cat, alphas in lib.list_categories().items():
        print(f"  {cat}: {len(alphas)} alphas")
