"""
Alpha Library
==============

Collection of classic and custom alphas for systematic research.

Each alpha returns a signal series:
- Positive = long signal
- Negative = short signal (if short-enabled)
- Zero = no signal

Alphas are designed to be:
1. Explainable (can defend in quant interview)
2. Economically motivated (why should this work?)
3. Statistically testable (measurable edge)

Alpha Categories:
- Momentum: trend-following signals
- Mean Reversion: counter-trend signals
- Value: fundamental-like signals
- Quality: low-risk signals
- Volatility: vol-based signals
- Composite: multi-factor signals

Usage:
    from research.alphas import AlphaLibrary, get_alpha_library

    library = get_alpha_library()
    signals = library.compute_alpha('momentum_12_1', ohlcv_df)

    # Or compute all alphas
    all_signals = library.compute_all(ohlcv_df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .features import FeatureExtractor, FEATURE_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class Alpha:
    """
    Definition of a single alpha signal.

    Attributes:
        name: Unique identifier
        description: Human-readable description
        category: Alpha category
        hypothesis: Economic hypothesis for why this should work
        compute_fn: Function that computes the alpha signal
        required_features: List of features needed
        lookback_days: Minimum history required
        long_only: Whether this is long-only or can short
    """
    name: str
    description: str
    category: str
    hypothesis: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    required_features: List[str] = field(default_factory=list)
    lookback_days: int = 20
    long_only: bool = True


# Registry of all alphas
ALPHA_REGISTRY: Dict[str, Alpha] = {}


def register_alpha(
    name: str,
    description: str,
    category: str,
    hypothesis: str,
    required_features: Optional[List[str]] = None,
    lookback_days: int = 20,
    long_only: bool = True,
):
    """Decorator to register an alpha computation function."""
    def decorator(fn: Callable):
        ALPHA_REGISTRY[name] = Alpha(
            name=name,
            description=description,
            category=category,
            hypothesis=hypothesis,
            compute_fn=fn,
            required_features=required_features or [],
            lookback_days=lookback_days,
            long_only=long_only,
        )
        return fn
    return decorator


# ============================================================================
# MOMENTUM ALPHAS
# ============================================================================

@register_alpha(
    name="momentum_12_1",
    description="12-month momentum skipping last month",
    category="momentum",
    hypothesis="Winners tend to keep winning due to slow information diffusion and behavioral biases",
    required_features=["return_60d"],
    lookback_days=252,
    long_only=False,
)
def alpha_momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """Classic 12-1 momentum factor."""
    close = df['close']
    return close.shift(21).pct_change(252 - 21)


@register_alpha(
    name="momentum_3m",
    description="3-month momentum",
    category="momentum",
    hypothesis="Medium-term trends persist due to herding and underreaction",
    required_features=["return_60d"],
    lookback_days=63,
    long_only=False,
)
def alpha_momentum_3m(df: pd.DataFrame) -> pd.Series:
    """3-month momentum."""
    return df['close'].pct_change(63)


@register_alpha(
    name="momentum_breakout",
    description="Donchian channel breakout (20-day high)",
    category="momentum",
    hypothesis="New highs signal strong demand and trend continuation",
    lookback_days=21,
    long_only=True,
)
def alpha_momentum_breakout(df: pd.DataFrame) -> pd.Series:
    """Signal when price makes new 20-day high."""
    high_20 = df['high'].rolling(20).max()
    signal = (df['close'] >= high_20).astype(float)
    return signal


@register_alpha(
    name="momentum_acceleration",
    description="Momentum acceleration (ROC of momentum)",
    category="momentum",
    hypothesis="Accelerating momentum signals strengthening trend",
    lookback_days=42,
    long_only=False,
)
def alpha_momentum_acceleration(df: pd.DataFrame) -> pd.Series:
    """Rate of change of momentum."""
    mom = df['close'].pct_change(20)
    return mom.diff(20)


# ============================================================================
# MEAN REVERSION ALPHAS
# ============================================================================

@register_alpha(
    name="rsi2_oversold",
    description="RSI(2) <= 10 oversold signal",
    category="mean_reversion",
    hypothesis="Short-term oversold conditions often revert as panic sellers are exhausted",
    lookback_days=5,
    long_only=True,
)
def alpha_rsi2_oversold(df: pd.DataFrame) -> pd.Series:
    """Buy when RSI(2) <= 10."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=0.5, adjust=False).mean()
    avg_loss = loss.ewm(alpha=0.5, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return (rsi <= 10).astype(float)


@register_alpha(
    name="ibs_oversold",
    description="Internal Bar Strength <= 0.2",
    category="mean_reversion",
    hypothesis="Closing near daily low suggests panic selling, likely to revert",
    lookback_days=2,
    long_only=True,
)
def alpha_ibs_oversold(df: pd.DataFrame) -> pd.Series:
    """Buy when IBS <= 0.2."""
    ibs = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    return (ibs <= 0.2).astype(float)


@register_alpha(
    name="bollinger_lower",
    description="Price touches lower Bollinger Band",
    category="mean_reversion",
    hypothesis="Price at 2 std below mean often reverts to mean",
    lookback_days=21,
    long_only=True,
)
def alpha_bollinger_lower(df: pd.DataFrame) -> pd.Series:
    """Signal when price at/below lower Bollinger Band."""
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    lower = ma - 2 * std
    return (df['close'] <= lower).astype(float)


@register_alpha(
    name="deviation_from_ma20",
    description="Mean reversion from 20-day MA",
    category="mean_reversion",
    hypothesis="Large deviations from moving average tend to revert",
    lookback_days=21,
    long_only=False,
)
def alpha_deviation_from_ma20(df: pd.DataFrame) -> pd.Series:
    """Negative of deviation from 20-day MA (buy when below)."""
    ma = df['close'].rolling(20).mean()
    deviation = (df['close'] - ma) / ma
    return -deviation  # Negative = long when below MA


@register_alpha(
    name="turtle_soup",
    description="ICT Turtle Soup - failed breakout reversal",
    category="mean_reversion",
    hypothesis="Failed breakouts trap trend followers, leading to sharp reversals",
    lookback_days=21,
    long_only=True,
)
def alpha_turtle_soup(df: pd.DataFrame) -> pd.Series:
    """
    Buy when price breaks below 20-day low then closes back above.
    Classic turtle soup pattern.
    """
    low_20 = df['low'].rolling(20).min().shift(1)
    broke_low = df['low'] < low_20
    closed_above = df['close'] > low_20
    return (broke_low & closed_above).astype(float)


# ============================================================================
# VOLATILITY ALPHAS
# ============================================================================

@register_alpha(
    name="low_volatility",
    description="Low volatility stocks (inverse vol)",
    category="volatility",
    hypothesis="Low vol stocks outperform on risk-adjusted basis (vol anomaly)",
    lookback_days=63,
    long_only=True,
)
def alpha_low_volatility(df: pd.DataFrame) -> pd.Series:
    """Inverse of 60-day volatility (higher = lower vol)."""
    returns = df['close'].pct_change()
    vol = returns.rolling(60).std() * np.sqrt(252)
    return 1 / (vol + 0.01)  # Inverse vol


@register_alpha(
    name="vol_contraction",
    description="Volatility squeeze (low ATR / 20d avg ATR)",
    category="volatility",
    hypothesis="Volatility contraction precedes expansion, often with trend continuation",
    lookback_days=42,
    long_only=True,
)
def alpha_vol_contraction(df: pd.DataFrame) -> pd.Series:
    """Signal when current ATR is low relative to average."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    atr_avg = atr.rolling(20).mean()

    contraction = atr / atr_avg
    return (contraction < 0.8).astype(float)  # Low relative vol


@register_alpha(
    name="vol_breakout",
    description="High volume breakout",
    category="volatility",
    hypothesis="High volume confirms breakout validity",
    lookback_days=21,
    long_only=True,
)
def alpha_vol_breakout(df: pd.DataFrame) -> pd.Series:
    """Signal when price at 20-day high with above-average volume."""
    high_20 = df['high'].rolling(20).max()
    vol_avg = df['volume'].rolling(20).mean()

    price_breakout = df['close'] >= high_20
    volume_confirm = df['volume'] > vol_avg * 1.5

    return (price_breakout & volume_confirm).astype(float)


# ============================================================================
# TREND ALPHAS
# ============================================================================

@register_alpha(
    name="trend_sma_filter",
    description="Above 200-day SMA filter",
    category="trend",
    hypothesis="Bull market stocks (above 200 SMA) have better risk/reward",
    lookback_days=201,
    long_only=True,
)
def alpha_trend_sma_filter(df: pd.DataFrame) -> pd.Series:
    """1 if above 200 SMA, 0 otherwise."""
    sma200 = df['close'].rolling(200).mean()
    return (df['close'] > sma200).astype(float)


@register_alpha(
    name="trend_adx_strong",
    description="Strong trend (ADX > 25)",
    category="trend",
    hypothesis="Strong trends are more likely to continue",
    lookback_days=28,
    long_only=True,
)
def alpha_trend_adx_strong(df: pd.DataFrame) -> pd.Series:
    """1 if ADX > 25 and price trending up."""
    # Compute ADX
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    # Strong uptrend: ADX > 25 and +DI > -DI
    return ((adx > 25) & (plus_di > minus_di)).astype(float)


@register_alpha(
    name="golden_cross",
    description="SMA(50) crosses above SMA(200)",
    category="trend",
    hypothesis="Golden cross signals start of bull trend",
    lookback_days=201,
    long_only=True,
)
def alpha_golden_cross(df: pd.DataFrame) -> pd.Series:
    """1 if 50 SMA > 200 SMA."""
    sma50 = df['close'].rolling(50).mean()
    sma200 = df['close'].rolling(200).mean()
    return (sma50 > sma200).astype(float)


# ============================================================================
# COMPOSITE ALPHAS
# ============================================================================

@register_alpha(
    name="composite_quality_momentum",
    description="Momentum with trend filter",
    category="composite",
    hypothesis="Momentum works best when trend is confirmed",
    lookback_days=252,
    long_only=True,
)
def alpha_quality_momentum(df: pd.DataFrame) -> pd.Series:
    """Momentum * trend filter."""
    momentum = df['close'].pct_change(63)
    sma200 = df['close'].rolling(200).mean()
    trend_filter = (df['close'] > sma200).astype(float)
    return momentum * trend_filter


@register_alpha(
    name="composite_mean_rev_quality",
    description="Mean reversion with quality filter",
    category="composite",
    hypothesis="Mean reversion works best in uptrending stocks",
    lookback_days=201,
    long_only=True,
)
def alpha_mean_rev_quality(df: pd.DataFrame) -> pd.Series:
    """RSI(2) oversold + above 200 SMA."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=0.5, adjust=False).mean()
    avg_loss = loss.ewm(alpha=0.5, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    sma200 = df['close'].rolling(200).mean()
    trend_filter = (df['close'] > sma200).astype(float)

    return ((rsi <= 10) & (trend_filter > 0)).astype(float)


@register_alpha(
    name="composite_vol_adjusted_momentum",
    description="Momentum divided by volatility",
    category="composite",
    hypothesis="Normalize momentum by risk for better Sharpe",
    lookback_days=63,
    long_only=False,
)
def alpha_vol_adjusted_momentum(df: pd.DataFrame) -> pd.Series:
    """Momentum / volatility."""
    momentum = df['close'].pct_change(60)
    vol = df['close'].pct_change().rolling(60).std() * np.sqrt(252)
    return momentum / (vol + 0.01)


# ============================================================================
# ALPHA LIBRARY CLASS
# ============================================================================

class AlphaLibrary:
    """
    Library for computing and managing alphas.

    Provides:
    - Alpha computation with feature dependencies
    - Cross-sectional ranking
    - Alpha combination/blending
    """

    def __init__(
        self,
        alphas: Optional[List[str]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        """
        Initialize alpha library.

        Args:
            alphas: List of alpha names to use (None = all)
            feature_extractor: Optional feature extractor
        """
        self.alphas = alphas or list(ALPHA_REGISTRY.keys())

        # Validate alphas
        for a in self.alphas:
            if a not in ALPHA_REGISTRY:
                raise ValueError(f"Unknown alpha: {a}. Available: {list(ALPHA_REGISTRY.keys())}")

        # Build feature list
        required_features = set()
        for alpha_name in self.alphas:
            alpha = ALPHA_REGISTRY[alpha_name]
            required_features.update(alpha.required_features)

        self.feature_extractor = feature_extractor or FeatureExtractor(
            features=list(required_features) if required_features else None,
            zscore=True,
        )

    def compute_alpha(
        self,
        alpha_name: str,
        df: pd.DataFrame,
        rank: bool = False,
    ) -> pd.Series:
        """
        Compute a single alpha.

        Args:
            alpha_name: Name of alpha
            df: OHLCV DataFrame
            rank: If True, return cross-sectional ranks instead of values

        Returns:
            Series with alpha values
        """
        if alpha_name not in ALPHA_REGISTRY:
            raise ValueError(f"Unknown alpha: {alpha_name}")

        alpha = ALPHA_REGISTRY[alpha_name]

        # Compute per symbol if multi-symbol
        if 'symbol' in df.columns:
            signals = []
            for symbol in df['symbol'].unique():
                sym_df = df[df['symbol'] == symbol].copy()
                sig = alpha.compute_fn(sym_df)
                sig.index = sym_df.index
                signals.append(sig)
            result = pd.concat(signals)
        else:
            result = alpha.compute_fn(df)

        if rank:
            result = result.rank(pct=True)

        return result

    def compute_all(
        self,
        df: pd.DataFrame,
        rank: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all alphas and add as columns.

        Args:
            df: OHLCV DataFrame
            rank: If True, return ranks instead of values

        Returns:
            DataFrame with alpha columns added
        """
        result = df.copy()

        for alpha_name in self.alphas:
            try:
                result[f"alpha_{alpha_name}"] = self.compute_alpha(alpha_name, df, rank)
            except Exception as e:
                logger.warning(f"Failed to compute {alpha_name}: {e}")
                result[f"alpha_{alpha_name}"] = np.nan

        return result

    def get_alpha_info(self) -> pd.DataFrame:
        """Get information about all alphas."""
        info = []
        for name in self.alphas:
            alpha = ALPHA_REGISTRY[name]
            info.append({
                'name': name,
                'description': alpha.description,
                'category': alpha.category,
                'hypothesis': alpha.hypothesis,
                'lookback_days': alpha.lookback_days,
                'long_only': alpha.long_only,
            })
        return pd.DataFrame(info)

    def blend_alphas(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Create composite alpha from weighted blend.

        Args:
            df: OHLCV DataFrame
            weights: Dict of {alpha_name: weight}. None = equal weight.

        Returns:
            Series with blended alpha signal
        """
        weights = weights or {a: 1.0 for a in self.alphas}

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Compute and blend
        blend = pd.Series(0.0, index=df.index)

        for alpha_name, weight in weights.items():
            if alpha_name in self.alphas:
                signal = self.compute_alpha(alpha_name, df, rank=True)
                blend += weight * signal.fillna(0)

        return blend


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_library_cache: Optional[AlphaLibrary] = None


def get_alpha_library() -> AlphaLibrary:
    """Get singleton alpha library instance."""
    global _library_cache
    if _library_cache is None:
        _library_cache = AlphaLibrary()
    return _library_cache


def get_available_alphas() -> List[str]:
    """Get list of all available alphas."""
    return list(ALPHA_REGISTRY.keys())


def get_alphas_by_category(category: str) -> List[str]:
    """Get alphas in a specific category."""
    return [
        name for name, alpha in ALPHA_REGISTRY.items()
        if alpha.category == category
    ]
