"""
Feature Pipeline Module.

Provides a unified interface for extracting all ML features:
- Technical indicators (pandas-ta)
- Anomaly scores (stumpy)
- Custom features (price patterns, volatility metrics)

The pipeline handles:
- Feature normalization and scaling
- Missing value handling
- Feature selection
- Feature caching for efficiency
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any
from enum import Enum

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog
from .technical_features import TechnicalFeatures, TechnicalConfig
from .anomaly_detection import AnomalyDetector, AnomalyConfig


class FeatureCategory(Enum):
    """Categories of features."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    VOLUME = "volume"
    ANOMALY = "anomaly"
    PRICE_PATTERN = "price_pattern"
    ALL = "all"


class ScalingMethod(Enum):
    """Feature scaling methods."""
    NONE = "none"
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"  # 0-1 normalization
    ROBUST = "robust"  # Median-based normalization


@dataclass
class FeatureConfig:
    """Configuration for feature extraction pipeline."""
    # Feature categories to include
    categories: List[FeatureCategory] = field(default_factory=lambda: [FeatureCategory.ALL])

    # Technical indicator config
    technical_config: Optional[TechnicalConfig] = None

    # Anomaly detection config
    anomaly_config: Optional[AnomalyConfig] = None

    # Scaling
    scaling_method: ScalingMethod = ScalingMethod.NONE
    scale_by_symbol: bool = True  # Scale each symbol separately

    # Missing values
    fill_method: str = "ffill"  # forward fill by default
    drop_na_threshold: float = 0.5  # Drop columns with >50% NaN

    # Feature selection
    exclude_features: Set[str] = field(default_factory=set)
    include_only: Optional[Set[str]] = None  # If set, only include these features

    # Performance
    cache_features: bool = True
    shift_features: bool = True  # Shift features to prevent lookahead


class FeaturePipeline:
    """
    Unified feature extraction pipeline.

    Combines technical indicators, anomaly detection, and custom features
    into a single DataFrame for ML model training or signal enhancement.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._technical = TechnicalFeatures(self.config.technical_config)
        self._anomaly = AnomalyDetector(self.config.anomaly_config)
        self._scalers: Dict[str, Any] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}

    def extract(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Extract all configured features from the data.

        Args:
            df: DataFrame with OHLCV columns
            symbol: Optional symbol for caching

        Returns:
            DataFrame with all features added
        """
        if df.empty:
            return df

        # Check cache
        cache_key = f"{symbol}_{len(df)}" if symbol and self.config.cache_features else None
        if cache_key and cache_key in self._feature_cache:
            jlog("feature_cache_hit", level="DEBUG", symbol=symbol)
            return self._feature_cache[cache_key]

        df = df.copy()
        df.columns = df.columns.str.lower()
        original_cols = set(df.columns)

        # Extract features by category
        categories = self.config.categories
        if FeatureCategory.ALL in categories:
            categories = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.VOLUME,
                FeatureCategory.ANOMALY,
                FeatureCategory.PRICE_PATTERN,
            ]

        for category in categories:
            if category == FeatureCategory.MOMENTUM:
                df = self._technical._add_momentum_features(df)
            elif category == FeatureCategory.VOLATILITY:
                df = self._technical._add_volatility_features(df)
            elif category == FeatureCategory.TREND:
                df = self._technical._add_trend_features(df)
            elif category == FeatureCategory.VOLUME:
                df = self._technical._add_volume_features(df)
            elif category == FeatureCategory.ANOMALY:
                df = self._anomaly.detect_all(df)
            elif category == FeatureCategory.PRICE_PATTERN:
                df = self._add_price_patterns(df)

        # Get feature columns
        feature_cols = [c for c in df.columns if c not in original_cols]

        # Shift features to prevent lookahead
        if self.config.shift_features:
            for col in feature_cols:
                df[col] = df[col].shift(1)

        # Apply feature selection
        df = self._apply_feature_selection(df, feature_cols)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Scale features
        if self.config.scaling_method != ScalingMethod.NONE:
            df = self._scale_features(df, feature_cols, symbol)

        # Cache result
        if cache_key:
            self._feature_cache[cache_key] = df

        jlog("features_extracted", level="DEBUG",
             symbol=symbol,
             num_features=len(feature_cols),
             rows=len(df))

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        if 'close' not in df.columns:
            return df

        # Returns at multiple horizons
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period)

        # IBS (Internal Bar Strength)
        if 'high' in df.columns and 'low' in df.columns:
            range_hl = df['high'] - df['low']
            df['ibs'] = (df['close'] - df['low']) / range_hl.replace(0, np.nan)

        # Gap analysis
        if 'open' in df.columns:
            df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gap_filled'] = (
                ((df['gap_pct'] > 0) & (df['low'] <= df['close'].shift(1))) |
                ((df['gap_pct'] < 0) & (df['high'] >= df['close'].shift(1)))
            ).astype(float)

        # Inside bar pattern
        if 'high' in df.columns and 'low' in df.columns:
            df['inside_bar'] = (
                (df['high'] < df['high'].shift(1)) &
                (df['low'] > df['low'].shift(1))
            ).astype(float)

            # Outside bar (engulfing)
            df['outside_bar'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['low'] < df['low'].shift(1))
            ).astype(float)

        # Higher highs / Lower lows
        if 'high' in df.columns and 'low' in df.columns:
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
            df['hh_streak'] = df['higher_high'].groupby(
                (df['higher_high'] != df['higher_high'].shift()).cumsum()
            ).cumsum()
            df['ll_streak'] = df['lower_low'].groupby(
                (df['lower_low'] != df['lower_low'].shift()).cumsum()
            ).cumsum()

        # Candle body metrics
        if 'open' in df.columns:
            body = df['close'] - df['open']
            range_hl = df['high'] - df['low']
            df['body_pct'] = body.abs() / range_hl.replace(0, np.nan)
            df['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / range_hl.replace(0, np.nan)
            df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / range_hl.replace(0, np.nan)
            df['bullish_candle'] = (body > 0).astype(float)

        # Distance from N-day high/low
        for period in [5, 10, 20, 50]:
            df[f'dist_from_{period}d_high'] = (df['close'] - df['high'].rolling(period).max()) / df['close']
            df[f'dist_from_{period}d_low'] = (df['close'] - df['low'].rolling(period).min()) / df['close']

        return df

    def _apply_feature_selection(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Apply feature selection based on config."""
        # Exclude features
        if self.config.exclude_features:
            cols_to_drop = [c for c in feature_cols if c in self.config.exclude_features]
            df = df.drop(columns=cols_to_drop, errors='ignore')

        # Include only specified features
        if self.config.include_only:
            cols_to_keep = [c for c in df.columns if c in self.config.include_only or c in {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}]
            df = df[cols_to_keep]

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Drop columns with too many NaN values
        if self.config.drop_na_threshold > 0:
            nan_ratio = df.isna().sum() / len(df)
            cols_to_drop = nan_ratio[nan_ratio > self.config.drop_na_threshold].index
            if len(cols_to_drop) > 0:
                jlog("dropping_high_nan_columns", level="DEBUG",
                     columns=list(cols_to_drop),
                     threshold=self.config.drop_na_threshold)
                df = df.drop(columns=cols_to_drop)

        # Fill remaining NaN values
        if self.config.fill_method == "ffill":
            df = df.ffill()
        elif self.config.fill_method == "bfill":
            df = df.bfill()
        elif self.config.fill_method == "zero":
            df = df.fillna(0)
        elif self.config.fill_method == "mean":
            df = df.fillna(df.mean())

        return df

    def _scale_features(self, df: pd.DataFrame, feature_cols: List[str], symbol: Optional[str] = None) -> pd.DataFrame:
        """Scale feature columns."""
        if not SKLEARN_AVAILABLE:
            jlog("sklearn_not_available", level="WARNING",
                 message="scikit-learn not installed, skipping scaling")
            return df

        # Get numeric feature columns that exist
        cols_to_scale = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if not cols_to_scale:
            return df

        # Get or create scaler
        scaler_key = symbol if self.config.scale_by_symbol else "global"

        if scaler_key not in self._scalers:
            if self.config.scaling_method == ScalingMethod.STANDARD:
                self._scalers[scaler_key] = StandardScaler()
            elif self.config.scaling_method == ScalingMethod.MINMAX:
                self._scalers[scaler_key] = MinMaxScaler()

        scaler = self._scalers.get(scaler_key)
        if scaler is None:
            return df

        try:
            # Fit and transform
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
        except Exception as e:
            jlog("scaling_error", level="WARNING", error=str(e))

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names."""
        original_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'date'}
        return [c for c in df.columns if c not in original_cols]

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'return_1d') -> pd.Series:
        """
        Get feature importance based on correlation with target.

        Simple correlation-based importance for quick feature selection.
        """
        feature_cols = self.get_feature_names(df)

        if target_col not in df.columns:
            jlog("target_column_missing", level="WARNING", target=target_col)
            return pd.Series()

        correlations = {}
        for col in feature_cols:
            if col != target_col and col in df.columns:
                try:
                    corr = df[col].corr(df[target_col])
                    if pd.notna(corr):
                        correlations[col] = abs(corr)
                except Exception:
                    pass

        return pd.Series(correlations).sort_values(ascending=False)

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._feature_cache.clear()
        jlog("feature_cache_cleared", level="DEBUG")


# Convenience function
def extract_all_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract all features from a DataFrame.

    Args:
        df: DataFrame with OHLCV columns
        config: Optional feature configuration
        symbol: Optional symbol for caching

    Returns:
        DataFrame with all features added
    """
    pipeline = FeaturePipeline(config)
    return pipeline.extract(df, symbol)
