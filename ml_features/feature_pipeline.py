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
    LAG = "lag"  # NEW: Lag features for time series ML
    TIME = "time"  # NEW: Calendar/time features for seasonality
    MICROSTRUCTURE = "microstructure"  # NEW: Amihud, Roll spread, liquidity
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

    # NEW: Lag feature configuration (for tree-based models like XGBoost/LightGBM)
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])
    lag_columns: List[str] = field(default_factory=lambda: ['close', 'volume', 'return_1d'])

    # NEW: Time/calendar feature configuration (for seasonality capture)
    include_day_of_week: bool = True  # Monday=0, Friday=4
    include_month: bool = True  # 1-12
    include_quarter: bool = True  # 1-4
    include_week_of_year: bool = False  # 1-52
    include_trading_day_of_month: bool = True  # For month-end effects


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
                FeatureCategory.LAG,
                FeatureCategory.TIME,
                FeatureCategory.MICROSTRUCTURE,  # NEW: Amihud, Roll spread
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
            elif category == FeatureCategory.LAG:
                df = self._add_lag_features(df)
            elif category == FeatureCategory.TIME:
                df = self._add_time_features(df)
            elif category == FeatureCategory.MICROSTRUCTURE:
                df = self._add_microstructure_features(df)

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

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features for time series ML models.

        Lag features are critical for tree-based models (XGBoost, LightGBM)
        as they cannot implicitly learn temporal dependencies like LSTMs.

        Inspired by: Kaggle Time Series Forecasting (robikscube)
        """
        lag_periods = self.config.lag_periods
        lag_columns = self.config.lag_columns

        for col in lag_columns:
            if col not in df.columns:
                # Try to create return_1d if it doesn't exist
                if col == 'return_1d' and 'close' in df.columns:
                    df['return_1d'] = df['close'].pct_change()
                else:
                    continue

            for lag in lag_periods:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # Also add lag of returns at different horizons
        if 'close' in df.columns:
            returns = df['close'].pct_change()

            # Lagged returns (how did it perform N days ago?)
            for lag in lag_periods:
                df[f'return_lag{lag}'] = returns.shift(lag)

            # Rolling statistics of returns (momentum over lookback)
            for window in [5, 10, 20]:
                df[f'return_mean_{window}d'] = returns.rolling(window).mean()
                df[f'return_std_{window}d'] = returns.rolling(window).std()
                df[f'return_skew_{window}d'] = returns.rolling(window).skew()

        # Volume lag features
        if 'volume' in df.columns:
            vol = df['volume']
            for lag in [1, 2, 5]:
                df[f'volume_lag{lag}'] = vol.shift(lag)

            # Volume relative to recent average
            df['volume_vs_avg_10d'] = vol / vol.rolling(10).mean()
            df['volume_vs_avg_20d'] = vol / vol.rolling(20).mean()

        jlog("lag_features_added", level="DEBUG",
             n_lag_periods=len(lag_periods),
             n_lag_columns=len(lag_columns))

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar/time features for seasonality capture.

        These features help capture patterns like:
        - Monday effect (stocks tend to drop on Mondays)
        - Month-end rebalancing (institutional flows)
        - January effect (small caps outperform)
        - Quarter-end window dressing

        Inspired by: Kaggle Time Series Forecasting (robikscube)
        """
        # Try to get datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            dt = df.index.to_series()
        elif 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            dt = pd.to_datetime(df['date'])
        else:
            jlog("time_features_no_datetime", level="WARNING",
                 message="No datetime column found, skipping time features")
            return df

        # Ensure dt has proper datetime accessor
        if not hasattr(dt, 'dt'):
            dt = pd.to_datetime(dt)

        try:
            # Use dt accessor for Series
            dta = dt.dt if hasattr(dt, 'dt') else dt

            # Day of week (Monday=0, Friday=4)
            if self.config.include_day_of_week:
                df['day_of_week'] = dta.dayofweek.values
                # One-hot encode days
                dayofweek = dta.dayofweek
                for day in range(5):
                    df[f'is_day_{day}'] = (dayofweek == day).astype(float).values
                # Monday effect feature
                df['is_monday'] = (dayofweek == 0).astype(float).values
                df['is_friday'] = (dayofweek == 4).astype(float).values

            # Month (1-12)
            if self.config.include_month:
                month = dta.month
                df['month'] = month.values
                # January effect feature
                df['is_january'] = (month == 1).astype(float).values
                # Sell in May and go away
                df['is_may_to_oct'] = ((month >= 5) & (month <= 10)).astype(float).values

            # Quarter (1-4)
            if self.config.include_quarter:
                df['quarter'] = dta.quarter.values
                # Quarter-end (potential window dressing)
                df['is_quarter_end_month'] = dta.month.isin([3, 6, 9, 12]).astype(float).values

            # Week of year (1-52)
            if self.config.include_week_of_year:
                df['week_of_year'] = dta.isocalendar().week.values

            # Trading day of month (for month-end effects)
            if self.config.include_trading_day_of_month:
                df['day_of_month'] = dta.day.values
                df['is_month_end'] = dta.is_month_end.astype(float).values
                df['is_month_start'] = dta.is_month_start.astype(float).values

            # Sin/cos encoding for cyclical features (better for neural networks)
            if self.config.include_day_of_week:
                dayofweek = dta.dayofweek
                df['day_sin'] = np.sin(2 * np.pi * dayofweek / 5).values
                df['day_cos'] = np.cos(2 * np.pi * dayofweek / 5).values

            if self.config.include_month:
                month = dta.month
                df['month_sin'] = np.sin(2 * np.pi * month / 12).values
                df['month_cos'] = np.cos(2 * np.pi * month / 12).values

            jlog("time_features_added", level="DEBUG",
                 day_of_week=self.config.include_day_of_week,
                 month=self.config.include_month,
                 quarter=self.config.include_quarter)

        except Exception as e:
            jlog("time_features_error", level="WARNING", error=str(e))

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features from EOD data.

        These features capture liquidity, trading friction, and market quality
        without requiring tick-level data. Based on:
        - Amihud (2002): Illiquidity ratio
        - Roll (1984): Bid-ask spread estimator
        - Academic research on liquidity risk

        All features are EOD-compatible (use close, volume, high, low, open).
        """
        if 'close' not in df.columns or 'volume' not in df.columns:
            jlog("microstructure_missing_columns", level="WARNING",
                 message="Need close and volume for microstructure features")
            return df

        # ============================================================
        # 1. AMIHUD ILLIQUIDITY RATIO (Amihud 2002)
        # ============================================================
        # Formula: ILLIQ = mean(|return| / dollar_volume)
        # Interpretation: Higher = less liquid (avoid), Lower = more liquid (prefer)
        # Use: Filter out illiquid stocks that may have large slippage

        returns = df['close'].pct_change()

        # Dollar volume (in millions for numerical stability)
        dollar_volume = df['close'] * df['volume'] / 1e6

        # Amihud ratio: |return| / dollar_volume
        # Add epsilon to avoid division by zero
        epsilon = 1e-9
        amihud_raw = returns.abs() / (dollar_volume + epsilon)

        # Rolling averages at different windows
        for window in [5, 10, 20]:
            df[f'amihud_illiq_{window}d'] = amihud_raw.rolling(window).mean()

        # Amihud percentile rank (0-1, higher = less liquid vs history)
        df['amihud_illiq_pct'] = df['amihud_illiq_20d'].rank(pct=True)

        # ============================================================
        # 2. ROLL SPREAD ESTIMATOR (Roll 1984)
        # ============================================================
        # Estimates effective bid-ask spread from price changes
        # Formula: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))
        # Interpretation: Higher = wider spread (higher friction)

        price_change = df['close'].diff()

        # Rolling covariance of consecutive price changes
        def roll_spread_estimator(series, window=20):
            """Estimate spread from price change serial covariance."""
            result = pd.Series(index=series.index, dtype=float)
            for i in range(window, len(series)):
                window_data = series.iloc[i-window:i]
                cov = window_data.cov(window_data.shift(1))
                # Spread formula: 2 * sqrt(-cov) if cov < 0
                if cov < 0:
                    result.iloc[i] = 2 * np.sqrt(-cov)
                else:
                    result.iloc[i] = 0  # No bounce = near-zero spread
            return result

        df['roll_spread_20d'] = roll_spread_estimator(price_change, window=20)

        # Normalize by price for comparability
        df['roll_spread_pct'] = df['roll_spread_20d'] / df['close']

        # ============================================================
        # 3. KYLE'S LAMBDA (Simplified from EOD data)
        # ============================================================
        # Price impact per unit of volume (simplified version)
        # Full Kyle's lambda requires tick data, this is EOD approximation
        # Formula: lambda ≈ |return| / sqrt(volume)

        df['kyle_lambda_approx'] = returns.abs() / (np.sqrt(df['volume']) + epsilon)
        df['kyle_lambda_20d'] = df['kyle_lambda_approx'].rolling(20).mean()

        # ============================================================
        # 4. VOLUME TURNOVER RATIO
        # ============================================================
        # Relative trading activity (proxy for liquidity)

        for window in [5, 10, 20]:
            df[f'volume_turnover_{window}d'] = df['volume'].rolling(window).sum() / df['volume'].rolling(window * 2).mean()

        # ============================================================
        # 5. PRICE IMPACT RATIO (Simplified Almgren-Chriss)
        # ============================================================
        # How much does price move per unit of volume?

        # Absolute return per dollar volume (normalized impact)
        df['price_impact_ratio'] = (returns.abs() * 10000) / (dollar_volume + epsilon)
        df['price_impact_20d'] = df['price_impact_ratio'].rolling(20).mean()

        # ============================================================
        # 6. HIGH-LOW SPREAD ESTIMATOR (Corwin-Schultz 2012)
        # ============================================================
        # Uses high-low range to estimate bid-ask spread
        # More robust than Roll for EOD data

        if 'high' in df.columns and 'low' in df.columns:
            beta = (np.log(df['high'] / df['low']) ** 2).rolling(2).sum()
            gamma = (np.log(df['high'].rolling(2).max() / df['low'].rolling(2).min())) ** 2

            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
            alpha = alpha - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

            # Clip and transform to spread estimate
            alpha_clipped = alpha.clip(lower=0)
            df['cs_spread'] = (2 * (np.exp(alpha_clipped) - 1)) / (1 + np.exp(alpha_clipped))
            df['cs_spread_20d'] = df['cs_spread'].rolling(20).mean()

        # ============================================================
        # 7. RELATIVE SPREAD (Price-based proxy)
        # ============================================================
        # True range relative to price as spread proxy

        # Defragment DataFrame before adding more columns (prevents PerformanceWarning)
        df = df.copy()

        if 'high' in df.columns and 'low' in df.columns:
            true_range = df['high'] - df['low']
            df['relative_spread'] = true_range / df['close']
            df['relative_spread_20d'] = df['relative_spread'].rolling(20).mean()

        # ============================================================
        # 8. LIQUIDITY SCORE (Composite)
        # ============================================================
        # Combine multiple liquidity measures into single score
        # Higher score = MORE liquid (better for trading)

        # Normalize each component to 0-1 range using percentile rank
        if 'amihud_illiq_20d' in df.columns:
            # Invert Amihud (lower is better)
            amihud_score = 1 - df['amihud_illiq_20d'].rank(pct=True)
        else:
            amihud_score = 0.5

        # Volume-based score (higher volume = more liquid)
        vol_score = df['volume'].rank(pct=True)

        # Roll spread score (lower spread = more liquid)
        if 'roll_spread_pct' in df.columns:
            spread_score = 1 - df['roll_spread_pct'].rank(pct=True)
        else:
            spread_score = 0.5

        # Composite liquidity score (equal weighted)
        df['liquidity_score'] = (amihud_score + vol_score + spread_score) / 3

        # Defragment DataFrame after adding many columns
        df = df.copy()

        jlog("microstructure_features_added", level="DEBUG",
             features_added=[
                 'amihud_illiq_*', 'roll_spread_*', 'kyle_lambda_*',
                 'volume_turnover_*', 'price_impact_*', 'cs_spread_*',
                 'relative_spread_*', 'liquidity_score'
             ])

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
