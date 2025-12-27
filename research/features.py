"""
Feature Extraction Pipeline
============================

Systematic feature extraction for alpha research.

Features are computed using only past data (no lookahead).
All features are z-scored for cross-sectional comparison.

Feature Categories:
1. Returns: momentum, mean reversion
2. Volatility: realized vol, range, ATR
3. Volume: turnover, relative volume
4. Trend: SMA crossovers, ADX
5. Technical: RSI, MACD, Bollinger %B
6. Regime: rolling correlations, dispersion

Usage:
    from research.features import FeatureExtractor, extract_features

    extractor = FeatureExtractor()
    features_df = extractor.extract(ohlcv_df)

    # Or quick extraction
    features_df = extract_features(ohlcv_df, features=['momentum_20', 'vol_20'])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for a feature."""
    name: str
    description: str
    category: str
    lookback: int
    compute_fn: Callable[[pd.DataFrame, int], pd.Series]


# Registry of all available features
FEATURE_REGISTRY: Dict[str, FeatureConfig] = {}


def register_feature(
    name: str,
    description: str,
    category: str,
    lookback: int,
):
    """Decorator to register a feature computation function."""
    def decorator(fn: Callable):
        FEATURE_REGISTRY[name] = FeatureConfig(
            name=name,
            description=description,
            category=category,
            lookback=lookback,
            compute_fn=fn,
        )
        return fn
    return decorator


# ============================================================================
# RETURNS FEATURES
# ============================================================================

@register_feature(
    name="return_1d",
    description="1-day return",
    category="returns",
    lookback=2,
)
def compute_return_1d(df: pd.DataFrame, lookback: int = 2) -> pd.Series:
    """1-day return."""
    return df['close'].pct_change(1)


@register_feature(
    name="return_5d",
    description="5-day return (weekly momentum)",
    category="returns",
    lookback=6,
)
def compute_return_5d(df: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """5-day return."""
    return df['close'].pct_change(5)


@register_feature(
    name="return_20d",
    description="20-day return (monthly momentum)",
    category="returns",
    lookback=21,
)
def compute_return_20d(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """20-day return."""
    return df['close'].pct_change(20)


@register_feature(
    name="return_60d",
    description="60-day return (quarterly momentum)",
    category="returns",
    lookback=61,
)
def compute_return_60d(df: pd.DataFrame, lookback: int = 61) -> pd.Series:
    """60-day return."""
    return df['close'].pct_change(60)


@register_feature(
    name="momentum_12_1",
    description="12-month momentum skipping last month (classic factor)",
    category="returns",
    lookback=252,
)
def compute_momentum_12_1(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """12-month momentum excluding last month."""
    return df['close'].shift(21).pct_change(252 - 21)


@register_feature(
    name="mean_reversion_5d",
    description="5-day mean reversion (deviation from 5-day mean)",
    category="returns",
    lookback=6,
)
def compute_mean_reversion_5d(df: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """Deviation from 5-day rolling mean."""
    ma = df['close'].rolling(5).mean()
    return (df['close'] - ma) / ma


@register_feature(
    name="mean_reversion_20d",
    description="20-day mean reversion",
    category="returns",
    lookback=21,
)
def compute_mean_reversion_20d(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Deviation from 20-day rolling mean."""
    ma = df['close'].rolling(20).mean()
    return (df['close'] - ma) / ma


# ============================================================================
# VOLATILITY FEATURES
# ============================================================================

@register_feature(
    name="volatility_20d",
    description="20-day realized volatility (annualized)",
    category="volatility",
    lookback=21,
)
def compute_volatility_20d(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """20-day rolling volatility."""
    returns = df['close'].pct_change()
    return returns.rolling(20).std() * np.sqrt(252)


@register_feature(
    name="volatility_60d",
    description="60-day realized volatility",
    category="volatility",
    lookback=61,
)
def compute_volatility_60d(df: pd.DataFrame, lookback: int = 61) -> pd.Series:
    """60-day rolling volatility."""
    returns = df['close'].pct_change()
    return returns.rolling(60).std() * np.sqrt(252)


@register_feature(
    name="atr_14",
    description="14-day Average True Range (normalized)",
    category="volatility",
    lookback=15,
)
def compute_atr_14(df: pd.DataFrame, lookback: int = 15) -> pd.Series:
    """14-day ATR normalized by close."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=14, adjust=False).mean()
    return atr / close


@register_feature(
    name="range_pct",
    description="Daily range as percent of close",
    category="volatility",
    lookback=1,
)
def compute_range_pct(df: pd.DataFrame, lookback: int = 1) -> pd.Series:
    """Daily range percentage."""
    return (df['high'] - df['low']) / df['close']


@register_feature(
    name="vol_of_vol",
    description="Volatility of volatility (vol clustering)",
    category="volatility",
    lookback=42,
)
def compute_vol_of_vol(df: pd.DataFrame, lookback: int = 42) -> pd.Series:
    """Rolling std of rolling volatility."""
    returns = df['close'].pct_change()
    vol = returns.rolling(20).std()
    return vol.rolling(20).std()


# ============================================================================
# VOLUME FEATURES
# ============================================================================

@register_feature(
    name="volume_20d_avg",
    description="Relative volume vs 20-day average",
    category="volume",
    lookback=21,
)
def compute_volume_relative(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Volume relative to 20-day average."""
    vol_ma = df['volume'].rolling(20).mean()
    return df['volume'] / vol_ma


@register_feature(
    name="turnover_20d",
    description="20-day average turnover (dollar volume)",
    category="volume",
    lookback=21,
)
def compute_turnover_20d(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """20-day average dollar volume."""
    dollar_vol = df['close'] * df['volume']
    return dollar_vol.rolling(20).mean()


@register_feature(
    name="volume_trend",
    description="Volume trend (5d/20d ratio)",
    category="volume",
    lookback=21,
)
def compute_volume_trend(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Short-term vs long-term volume ratio."""
    vol_5 = df['volume'].rolling(5).mean()
    vol_20 = df['volume'].rolling(20).mean()
    return vol_5 / vol_20


# ============================================================================
# TREND FEATURES
# ============================================================================

@register_feature(
    name="sma_crossover_5_20",
    description="SMA(5) / SMA(20) ratio",
    category="trend",
    lookback=21,
)
def compute_sma_crossover_5_20(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """SMA 5/20 crossover ratio."""
    sma5 = df['close'].rolling(5).mean()
    sma20 = df['close'].rolling(20).mean()
    return sma5 / sma20 - 1


@register_feature(
    name="sma_crossover_20_50",
    description="SMA(20) / SMA(50) ratio",
    category="trend",
    lookback=51,
)
def compute_sma_crossover_20_50(df: pd.DataFrame, lookback: int = 51) -> pd.Series:
    """SMA 20/50 crossover ratio."""
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    return sma20 / sma50 - 1


@register_feature(
    name="price_vs_sma200",
    description="Price relative to 200-day SMA",
    category="trend",
    lookback=201,
)
def compute_price_vs_sma200(df: pd.DataFrame, lookback: int = 201) -> pd.Series:
    """Price / SMA(200) - 1."""
    sma200 = df['close'].rolling(200).mean()
    return df['close'] / sma200 - 1


@register_feature(
    name="adx_14",
    description="Average Directional Index (trend strength)",
    category="trend",
    lookback=28,
)
def compute_adx_14(df: pd.DataFrame, lookback: int = 28) -> pd.Series:
    """14-day ADX."""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # +DM and -DM
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr)

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    return adx


# ============================================================================
# TECHNICAL FEATURES
# ============================================================================

@register_feature(
    name="rsi_14",
    description="14-day Relative Strength Index",
    category="technical",
    lookback=15,
)
def compute_rsi_14(df: pd.DataFrame, lookback: int = 15) -> pd.Series:
    """14-day RSI."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_feature(
    name="rsi_2",
    description="2-day RSI (short-term overbought/oversold)",
    category="technical",
    lookback=3,
)
def compute_rsi_2(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """2-day RSI for short-term mean reversion."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_feature(
    name="bb_pct_b",
    description="Bollinger Band %B (position within bands)",
    category="technical",
    lookback=21,
)
def compute_bb_pct_b(df: pd.DataFrame, lookback: int = 21) -> pd.Series:
    """Bollinger Band %B."""
    ma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (df['close'] - lower) / (upper - lower + 1e-10)


@register_feature(
    name="macd_signal",
    description="MACD - Signal line",
    category="technical",
    lookback=35,
)
def compute_macd_signal(df: pd.DataFrame, lookback: int = 35) -> pd.Series:
    """MACD histogram."""
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return (macd - signal) / df['close']


# ============================================================================
# REGIME / MARKET FEATURES
# ============================================================================

@register_feature(
    name="drawdown",
    description="Current drawdown from rolling max",
    category="regime",
    lookback=252,
)
def compute_drawdown(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Rolling drawdown."""
    rolling_max = df['close'].rolling(252, min_periods=1).max()
    return (df['close'] - rolling_max) / rolling_max


@register_feature(
    name="days_since_high",
    description="Days since 52-week high",
    category="regime",
    lookback=252,
)
def compute_days_since_high(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """Days since 52-week high."""
    rolling_max = df['close'].rolling(252, min_periods=1).max()
    at_high = (df['close'] == rolling_max).astype(int)
    # Count days since high
    groups = at_high.cumsum()
    return at_high.groupby(groups).cumcount()


# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================

class FeatureExtractor:
    """
    Extracts features from OHLCV data.

    Handles:
    - Per-symbol feature computation
    - Cross-sectional z-scoring
    - NaN handling
    - Feature selection
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        zscore: bool = True,
        dropna: bool = False,
    ):
        """
        Initialize feature extractor.

        Args:
            features: List of feature names to compute (None = all)
            zscore: Whether to z-score features cross-sectionally
            dropna: Whether to drop rows with NaN features
        """
        self.features = features or list(FEATURE_REGISTRY.keys())
        self.zscore = zscore
        self.dropna = dropna

        # Validate features
        for f in self.features:
            if f not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature: {f}. Available: {list(FEATURE_REGISTRY.keys())}")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [timestamp, symbol, open, high, low, close, volume]

        Returns:
            DataFrame with original columns plus feature columns
        """
        if df.empty:
            return df

        result_dfs = []

        # Check if multi-symbol
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()

            for symbol in symbols:
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = self._extract_single(symbol_df)
                result_dfs.append(symbol_df)

            result = pd.concat(result_dfs, ignore_index=True)
        else:
            result = self._extract_single(df.copy())

        # Z-score cross-sectionally (within each date)
        if self.zscore and 'timestamp' in result.columns:
            result = self._zscore_features(result)

        # Drop NaN if requested
        if self.dropna:
            result = result.dropna(subset=self.features)

        return result

    def _extract_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for a single symbol."""
        for feature_name in self.features:
            config = FEATURE_REGISTRY[feature_name]
            try:
                df[feature_name] = config.compute_fn(df, config.lookback)
            except Exception as e:
                logger.warning(f"Failed to compute {feature_name}: {e}")
                df[feature_name] = np.nan

        return df

    def _zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score features cross-sectionally within each date."""
        for feature_name in self.features:
            if feature_name in df.columns:
                # Group by date and z-score
                grouped = df.groupby('timestamp')[feature_name]
                df[f"{feature_name}_zscore"] = grouped.transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )

        return df

    def get_feature_info(self) -> pd.DataFrame:
        """Get information about all features."""
        info = []
        for name in self.features:
            config = FEATURE_REGISTRY[name]
            info.append({
                'name': name,
                'description': config.description,
                'category': config.category,
                'lookback': config.lookback,
            })
        return pd.DataFrame(info)


def extract_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Quick function to extract features.

    Args:
        df: OHLCV DataFrame
        features: Features to compute (None = all)
        zscore: Whether to z-score

    Returns:
        DataFrame with features added
    """
    extractor = FeatureExtractor(features=features, zscore=zscore)
    return extractor.extract(df)


def get_available_features() -> List[str]:
    """Get list of all available features."""
    return list(FEATURE_REGISTRY.keys())


def get_features_by_category(category: str) -> List[str]:
    """Get features in a specific category."""
    return [
        name for name, config in FEATURE_REGISTRY.items()
        if config.category == category
    ]
