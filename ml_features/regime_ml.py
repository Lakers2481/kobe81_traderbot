"""
ML-Based Regime Detection Module.

Provides machine learning based market regime detection:
- Trend regimes (bull, bear, sideways)
- Volatility regimes (low, normal, high, crisis)
- Combined regime states

Uses multiple approaches:
- Statistical clustering (KMeans, GMM)
- Hidden Markov Models (when available)
- Rule-based fallback
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog


class RegimeState(Enum):
    """Market regime states."""
    # Trend regimes
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"

    # Volatility regimes
    LOW_VOL = "low_vol"
    NORMAL_VOL = "normal_vol"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"

    # Combined
    UNKNOWN = "unknown"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Feature lookback periods
    return_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 60])

    # Clustering
    n_trend_clusters: int = 5
    n_vol_clusters: int = 4
    use_gmm: bool = True  # Gaussian Mixture Models (softer clustering)

    # Thresholds for rule-based fallback
    bull_threshold: float = 0.10  # 10% return threshold
    bear_threshold: float = -0.10
    high_vol_threshold: float = 0.25  # 25% annualized volatility
    low_vol_threshold: float = 0.10

    # Smoothing
    regime_smoothing: int = 5  # Rolling mode over N days
    min_periods: int = 60  # Minimum data for regime detection


@dataclass
class RegimeResult:
    """Result of regime detection."""
    trend_regime: RegimeState
    vol_regime: RegimeState
    combined_regime: str  # e.g., "bull_low_vol"
    probabilities: Dict[str, float]  # Probability of each regime
    features: Dict[str, float]  # Features used for detection
    confidence: float  # 0-1 confidence in detection


class RegimeDetectorML:
    """
    ML-based market regime detector.

    Uses clustering algorithms to identify market regimes from
    multiple features including returns, volatility, and momentum.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._trend_model = None
        self._vol_model = None
        self._scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'RegimeDetectorML':
        """
        Fit the regime detection models on historical data.

        Args:
            df: DataFrame with OHLCV columns (minimum 60 days recommended)

        Returns:
            self for method chaining
        """
        if not SKLEARN_AVAILABLE:
            jlog("sklearn_not_available", level="WARNING",
                 message="scikit-learn not installed, using rule-based detection")
            return self

        if len(df) < self.config.min_periods:
            jlog("insufficient_data_for_fit", level="WARNING",
                 data_points=len(df), required=self.config.min_periods)
            return self

        # Extract features
        features_df = self._extract_regime_features(df)
        features_df = features_df.dropna()

        if features_df.empty:
            return self

        # Fit trend model
        trend_features = self._get_trend_features(features_df)
        if not trend_features.empty:
            self._fit_trend_model(trend_features)

        # Fit volatility model
        vol_features = self._get_vol_features(features_df)
        if not vol_features.empty:
            self._fit_vol_model(vol_features)

        self._is_fitted = True
        jlog("regime_model_fitted", level="INFO", data_points=len(features_df))

        return self

    def _extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection."""
        df = df.copy()
        df.columns = df.columns.str.lower()

        if 'close' not in df.columns:
            return pd.DataFrame()

        features = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        for period in self.config.return_periods:
            features[f'return_{period}d'] = df['close'].pct_change(periods=period)

        # Realized volatility at multiple horizons
        daily_returns = df['close'].pct_change()
        for period in self.config.volatility_periods:
            features[f'vol_{period}d'] = daily_returns.rolling(period).std() * np.sqrt(252)

        # Trend strength: distance from moving averages
        for period in [20, 50, 200]:
            sma = df['close'].rolling(period).mean()
            features[f'dist_sma_{period}'] = (df['close'] - sma) / sma

        # Momentum
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        features['momentum_60'] = df['close'] / df['close'].shift(60) - 1

        # Higher high / lower low streak
        if 'high' in df.columns and 'low' in df.columns:
            features['hh_count'] = (df['high'] > df['high'].shift(1)).rolling(10).sum()
            features['ll_count'] = (df['low'] < df['low'].shift(1)).rolling(10).sum()

        return features

    def _get_trend_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get trend-related features."""
        trend_cols = [c for c in features_df.columns
                      if 'return_' in c or 'dist_sma' in c or 'momentum' in c]
        return features_df[trend_cols].dropna()

    def _get_vol_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get volatility-related features."""
        vol_cols = [c for c in features_df.columns if 'vol_' in c]
        return features_df[vol_cols].dropna()

    def _fit_trend_model(self, features: pd.DataFrame) -> None:
        """Fit trend regime clustering model."""
        if features.empty:
            return

        scaled = self._scaler.fit_transform(features.values)

        if self.config.use_gmm:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._trend_model = GaussianMixture(
                    n_components=self.config.n_trend_clusters,
                    covariance_type='full',
                    random_state=42,
                    n_init=3
                )
        else:
            self._trend_model = KMeans(
                n_clusters=self.config.n_trend_clusters,
                random_state=42,
                n_init=10
            )

        self._trend_model.fit(scaled)

    def _fit_vol_model(self, features: pd.DataFrame) -> None:
        """Fit volatility regime clustering model."""
        if features.empty:
            return

        scaled = self._scaler.fit_transform(features.values)

        if self.config.use_gmm:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._vol_model = GaussianMixture(
                    n_components=self.config.n_vol_clusters,
                    covariance_type='full',
                    random_state=42,
                    n_init=3
                )
        else:
            self._vol_model = KMeans(
                n_clusters=self.config.n_vol_clusters,
                random_state=42,
                n_init=10
            )

        self._vol_model.fit(scaled)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes for the given data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with regime columns added:
            - trend_regime: Trend regime state
            - vol_regime: Volatility regime state
            - combined_regime: Combined regime string
            - regime_confidence: Detection confidence
        """
        if df.empty:
            return df

        df = df.copy()

        # Extract features
        features_df = self._extract_regime_features(df)

        # Detect regimes
        if self._is_fitted and SKLEARN_AVAILABLE:
            trend_regimes = self._detect_trend_ml(features_df)
            vol_regimes = self._detect_vol_ml(features_df)
        else:
            trend_regimes = self._detect_trend_rules(features_df)
            vol_regimes = self._detect_vol_rules(features_df)

        # Add to dataframe
        df['trend_regime'] = trend_regimes
        df['vol_regime'] = vol_regimes
        df['combined_regime'] = df['trend_regime'] + '_' + df['vol_regime']

        # Calculate confidence
        df['regime_confidence'] = self._calculate_confidence(features_df)

        return df

    def _detect_trend_ml(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect trend regime using ML model."""
        result = pd.Series(RegimeState.NEUTRAL.value, index=features_df.index)

        trend_features = self._get_trend_features(features_df)
        if trend_features.empty or self._trend_model is None:
            return self._detect_trend_rules(features_df)

        try:
            scaled = self._scaler.fit_transform(trend_features.values)

            if self.config.use_gmm:
                labels = self._trend_model.predict(scaled)
                self._trend_model.predict_proba(scaled)
            else:
                labels = self._trend_model.predict(scaled)

            # Map cluster labels to regime states
            # Use mean return of each cluster to determine label
            cluster_returns = {}
            for i in range(self.config.n_trend_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    cluster_returns[i] = trend_features.iloc[mask]['return_20d'].mean()

            sorted_clusters = sorted(cluster_returns.keys(), key=lambda x: cluster_returns.get(x, 0))

            # Assign regime names based on sorted return
            regime_map = {}
            n = len(sorted_clusters)
            for idx, cluster in enumerate(sorted_clusters):
                if idx < n * 0.2:
                    regime_map[cluster] = RegimeState.STRONG_BEAR.value
                elif idx < n * 0.4:
                    regime_map[cluster] = RegimeState.BEAR.value
                elif idx < n * 0.6:
                    regime_map[cluster] = RegimeState.NEUTRAL.value
                elif idx < n * 0.8:
                    regime_map[cluster] = RegimeState.BULL.value
                else:
                    regime_map[cluster] = RegimeState.STRONG_BULL.value

            # Apply mapping
            regimes = pd.Series([regime_map.get(l, RegimeState.NEUTRAL.value) for l in labels],
                               index=trend_features.index)

            result.loc[trend_features.index] = regimes

        except Exception as e:
            jlog("trend_detection_error", level="WARNING", error=str(e))
            return self._detect_trend_rules(features_df)

        return result

    def _detect_vol_ml(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime using ML model."""
        result = pd.Series(RegimeState.NORMAL_VOL.value, index=features_df.index)

        vol_features = self._get_vol_features(features_df)
        if vol_features.empty or self._vol_model is None:
            return self._detect_vol_rules(features_df)

        try:
            scaled = self._scaler.fit_transform(vol_features.values)

            if self.config.use_gmm:
                labels = self._vol_model.predict(scaled)
            else:
                labels = self._vol_model.predict(scaled)

            # Map cluster labels to volatility regimes
            cluster_vol = {}
            for i in range(self.config.n_vol_clusters):
                mask = labels == i
                if mask.sum() > 0:
                    cluster_vol[i] = vol_features.iloc[mask]['vol_20d'].mean()

            sorted_clusters = sorted(cluster_vol.keys(), key=lambda x: cluster_vol.get(x, 0))

            # Assign regime names
            regime_map = {}
            n = len(sorted_clusters)
            for idx, cluster in enumerate(sorted_clusters):
                if idx < n * 0.25:
                    regime_map[cluster] = RegimeState.LOW_VOL.value
                elif idx < n * 0.50:
                    regime_map[cluster] = RegimeState.NORMAL_VOL.value
                elif idx < n * 0.75:
                    regime_map[cluster] = RegimeState.HIGH_VOL.value
                else:
                    regime_map[cluster] = RegimeState.CRISIS.value

            # Apply mapping
            regimes = pd.Series([regime_map.get(l, RegimeState.NORMAL_VOL.value) for l in labels],
                               index=vol_features.index)

            result.loc[vol_features.index] = regimes

        except Exception as e:
            jlog("vol_detection_error", level="WARNING", error=str(e))
            return self._detect_vol_rules(features_df)

        return result

    def _detect_trend_rules(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect trend regime using rule-based approach."""
        result = pd.Series(RegimeState.NEUTRAL.value, index=features_df.index)

        if 'return_60d' not in features_df.columns:
            return result

        returns = features_df['return_60d']

        # Simple thresholds
        result[returns > self.config.bull_threshold * 1.5] = RegimeState.STRONG_BULL.value
        result[(returns > self.config.bull_threshold) & (returns <= self.config.bull_threshold * 1.5)] = RegimeState.BULL.value
        result[returns < self.config.bear_threshold * 1.5] = RegimeState.STRONG_BEAR.value
        result[(returns < self.config.bear_threshold) & (returns >= self.config.bear_threshold * 1.5)] = RegimeState.BEAR.value

        return result

    def _detect_vol_rules(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime using rule-based approach."""
        result = pd.Series(RegimeState.NORMAL_VOL.value, index=features_df.index)

        if 'vol_20d' not in features_df.columns:
            return result

        vol = features_df['vol_20d']

        result[vol < self.config.low_vol_threshold] = RegimeState.LOW_VOL.value
        result[(vol >= self.config.low_vol_threshold) & (vol < self.config.high_vol_threshold)] = RegimeState.NORMAL_VOL.value
        result[(vol >= self.config.high_vol_threshold) & (vol < 0.40)] = RegimeState.HIGH_VOL.value
        result[vol >= 0.40] = RegimeState.CRISIS.value

        return result

    def _calculate_confidence(self, features_df: pd.DataFrame) -> pd.Series:
        """Calculate confidence in regime detection."""
        confidence = pd.Series(0.5, index=features_df.index)

        # Higher confidence when volatility is stable
        if 'vol_20d' in features_df.columns:
            vol_change = features_df['vol_20d'].pct_change(5, fill_method=None).abs()
            confidence += (1 - vol_change.clip(0, 1)) * 0.2

        # Higher confidence when trend is clear
        if 'momentum_60' in features_df.columns:
            momentum_strength = features_df['momentum_60'].abs()
            confidence += momentum_strength.clip(0, 0.3)

        return confidence.clip(0, 1)

    def get_current_regime(self, df: pd.DataFrame) -> RegimeResult:
        """
        Get the current regime (most recent row).

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            RegimeResult with current regime information
        """
        df_with_regime = self.detect(df)

        if df_with_regime.empty:
            return RegimeResult(
                trend_regime=RegimeState.UNKNOWN,
                vol_regime=RegimeState.UNKNOWN,
                combined_regime="unknown",
                probabilities={},
                features={},
                confidence=0.0
            )

        last_row = df_with_regime.iloc[-1]

        trend_str = last_row.get('trend_regime', 'neutral')
        vol_str = last_row.get('vol_regime', 'normal_vol')

        try:
            trend_regime = RegimeState(trend_str)
        except ValueError:
            trend_regime = RegimeState.NEUTRAL

        try:
            vol_regime = RegimeState(vol_str)
        except ValueError:
            vol_regime = RegimeState.NORMAL_VOL

        return RegimeResult(
            trend_regime=trend_regime,
            vol_regime=vol_regime,
            combined_regime=last_row.get('combined_regime', 'neutral_normal_vol'),
            probabilities={},
            features={},
            confidence=last_row.get('regime_confidence', 0.5)
        )


# Convenience function
def detect_regime_ml(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    fit_model: bool = True
) -> RegimeResult:
    """
    Detect current market regime using ML.

    Args:
        df: DataFrame with OHLCV columns
        config: Optional regime configuration
        fit_model: Whether to fit the model first

    Returns:
        RegimeResult with current regime
    """
    detector = RegimeDetectorML(config)

    if fit_model:
        detector.fit(df)

    return detector.get_current_regime(df)
