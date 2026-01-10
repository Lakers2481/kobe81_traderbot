"""
Anomaly Detection Module using stumpy.

Provides matrix profile based anomaly detection for:
- Price anomalies (unusual price movements)
- Volume anomalies (unusual volume spikes)
- Pattern anomalies (unusual sequence patterns)

Matrix profiles are extremely efficient for time series anomaly detection
and can process large datasets in near-linear time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
import pandas as pd

try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    stumpy = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog


class AnomalyType(Enum):
    """Types of detected anomalies."""
    PRICE_SPIKE = "price_spike"
    PRICE_DROP = "price_drop"
    VOLUME_SPIKE = "volume_spike"
    PATTERN_ANOMALY = "pattern_anomaly"
    VOLATILITY_REGIME = "volatility_regime"


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    # Matrix profile settings
    window_size: int = 20  # Subsequence length for matrix profile
    normalize: bool = True  # Normalize before computing matrix profile

    # Anomaly thresholds
    price_zscore_threshold: float = 3.0  # Z-score threshold for price anomalies
    volume_zscore_threshold: float = 3.0  # Z-score threshold for volume anomalies
    mp_percentile_threshold: float = 95.0  # Percentile threshold for matrix profile anomalies

    # Additional settings
    min_periods: int = 50  # Minimum periods required for detection
    ewm_span: int = 20  # Exponential weighted moving span for smoothing


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    timestamp: pd.Timestamp
    anomaly_type: AnomalyType
    score: float  # 0-1 normalized score (higher = more anomalous)
    value: float  # The actual value that triggered the anomaly
    threshold: float  # The threshold that was exceeded
    details: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """
    Anomaly detector using matrix profiles and statistical methods.

    Uses stumpy for efficient matrix profile computation, which identifies
    unusual patterns by finding the nearest neighbor distance for each
    subsequence in the time series.
    """

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self._validate_stumpy()

    def _validate_stumpy(self) -> None:
        """Check that stumpy is available."""
        if not STUMPY_AVAILABLE:
            jlog("stumpy_not_available", level="WARNING",
                 message="stumpy not installed, using fallback anomaly detection")

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all types of anomalies in the data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with anomaly columns added:
            - anomaly_price: Price anomaly score (0-1)
            - anomaly_volume: Volume anomaly score (0-1)
            - anomaly_pattern: Pattern anomaly score (0-1)
            - anomaly_combined: Combined anomaly score (0-1)
            - is_anomaly: Boolean flag for significant anomalies
        """
        if df.empty or len(df) < self.config.min_periods:
            return df

        df = df.copy()
        df.columns = df.columns.str.lower()

        # Detect price anomalies
        df['anomaly_price'] = self._detect_price_anomalies(df)

        # Detect volume anomalies
        if 'volume' in df.columns:
            df['anomaly_volume'] = self._detect_volume_anomalies(df)
        else:
            df['anomaly_volume'] = 0.0

        # Detect pattern anomalies using matrix profile
        df['anomaly_pattern'] = self._detect_pattern_anomalies(df)

        # Combined anomaly score (weighted average)
        df['anomaly_combined'] = (
            0.4 * df['anomaly_price'] +
            0.3 * df['anomaly_volume'] +
            0.3 * df['anomaly_pattern']
        )

        # Binary anomaly flag
        df['is_anomaly'] = df['anomaly_combined'] > 0.7

        return df

    def _detect_price_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect price anomalies using returns z-score.

        Returns a 0-1 normalized anomaly score.
        """
        if 'close' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Calculate returns
        returns = df['close'].pct_change()

        # Rolling z-score of returns
        rolling_mean = returns.rolling(self.config.ewm_span).mean()
        rolling_std = returns.rolling(self.config.ewm_span).std()

        zscore = (returns - rolling_mean) / rolling_std.replace(0, np.nan)

        # Convert z-score to 0-1 anomaly score
        # Using CDF of normal distribution: high z-score = high anomaly
        cdf_values = self._norm_cdf(np.abs(zscore.fillna(0).values))
        anomaly_score = pd.Series(2 * cdf_values - 1, index=df.index)

        return anomaly_score.clip(0, 1)

    def _detect_volume_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect volume anomalies using log volume z-score.

        Returns a 0-1 normalized anomaly score.
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return pd.Series(0.0, index=df.index)

        # Use log volume for better distribution
        log_volume = np.log1p(df['volume'].replace(0, 1))

        # Rolling z-score
        rolling_mean = log_volume.rolling(self.config.ewm_span).mean()
        rolling_std = log_volume.rolling(self.config.ewm_span).std()

        zscore = (log_volume - rolling_mean) / rolling_std.replace(0, np.nan)

        # Convert z-score to 0-1 anomaly score: high z-score = high anomaly
        cdf_values = self._norm_cdf(np.abs(zscore.fillna(0).values))
        anomaly_score = pd.Series(2 * cdf_values - 1, index=df.index)

        # Extra weight for volume spikes
        volume_spike_mask = zscore > self.config.volume_zscore_threshold
        anomaly_score = anomaly_score.where(~volume_spike_mask, 1.0)

        return anomaly_score.clip(0, 1)

    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect pattern anomalies using matrix profile.

        The matrix profile identifies unusual subsequences by computing
        the distance to the nearest neighbor for each subsequence.
        Higher distances indicate more unusual patterns.
        """
        if 'close' not in df.columns:
            return pd.Series(0.0, index=df.index)

        # Prepare the time series
        ts = df['close'].values.astype(np.float64)

        # Need enough data for matrix profile
        if len(ts) < self.config.window_size + 10:
            return pd.Series(0.0, index=df.index)

        if STUMPY_AVAILABLE:
            try:
                # Compute matrix profile
                mp = stumpy.stump(ts, m=self.config.window_size)

                # Matrix profile distances (first column)
                mp_distances = mp[:, 0]

                # Pad to original length (matrix profile is shorter)
                pad_length = len(ts) - len(mp_distances)
                mp_distances = np.concatenate([
                    np.full(pad_length, np.nan),
                    mp_distances
                ])

                # Convert to percentile-based anomaly score
                valid_mask = ~np.isnan(mp_distances)
                if valid_mask.sum() > 0:
                    percentiles = np.zeros_like(mp_distances)
                    percentiles[valid_mask] = self._percentile_rank(mp_distances[valid_mask])
                    anomaly_score = percentiles / 100.0
                else:
                    anomaly_score = np.zeros(len(ts))

            except Exception as e:
                jlog("matrix_profile_error", level="WARNING", error=str(e))
                anomaly_score = self._fallback_pattern_detection(ts)
        else:
            anomaly_score = self._fallback_pattern_detection(ts)

        return pd.Series(anomaly_score, index=df.index).clip(0, 1)

    def _fallback_pattern_detection(self, ts: np.ndarray) -> np.ndarray:
        """Fallback pattern detection without stumpy."""
        # Use rolling statistics as a simple alternative
        ts_series = pd.Series(ts)
        rolling_mean = ts_series.rolling(self.config.window_size).mean()
        rolling_std = ts_series.rolling(self.config.window_size).std()

        # Deviation from rolling mean: high z-score = high anomaly
        deviation = np.abs(ts_series - rolling_mean) / rolling_std.replace(0, np.nan)
        cdf_values = self._norm_cdf(deviation.fillna(0).values)
        anomaly_score = 2 * cdf_values - 1

        return np.clip(anomaly_score, 0, 1)

    def _norm_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF approximation."""
        try:
            from scipy.stats import norm
            return norm.cdf(x)
        except ImportError:
            # Approximation of standard normal CDF
            return 0.5 * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def _percentile_rank(self, values: np.ndarray) -> np.ndarray:
        """Compute percentile rank for each value."""
        sorted_values = np.sort(values)
        ranks = np.searchsorted(sorted_values, values, side='right')
        return ranks / len(values) * 100

    def get_anomalies(self, df: pd.DataFrame, threshold: float = 0.7) -> List[Anomaly]:
        """
        Get list of detected anomalies above threshold.

        Args:
            df: DataFrame with OHLCV columns
            threshold: Minimum combined anomaly score (0-1)

        Returns:
            List of Anomaly objects
        """
        df_with_anomalies = self.detect_all(df)
        anomalies = []

        for idx, row in df_with_anomalies.iterrows():
            if row.get('anomaly_combined', 0) >= threshold:
                # Determine primary anomaly type
                if row.get('anomaly_price', 0) >= 0.7:
                    atype = AnomalyType.PRICE_SPIKE if row.get('close', 0) > row.get('open', 0) else AnomalyType.PRICE_DROP
                elif row.get('anomaly_volume', 0) >= 0.7:
                    atype = AnomalyType.VOLUME_SPIKE
                else:
                    atype = AnomalyType.PATTERN_ANOMALY

                anomaly = Anomaly(
                    timestamp=pd.Timestamp(idx) if not isinstance(idx, pd.Timestamp) else idx,
                    anomaly_type=atype,
                    score=row['anomaly_combined'],
                    value=row.get('close', 0),
                    threshold=threshold,
                    details={
                        'price_score': row.get('anomaly_price', 0),
                        'volume_score': row.get('anomaly_volume', 0),
                        'pattern_score': row.get('anomaly_pattern', 0),
                    }
                )
                anomalies.append(anomaly)

        return anomalies


# Convenience functions
def detect_price_anomalies(df: pd.DataFrame, config: Optional[AnomalyConfig] = None) -> pd.Series:
    """Detect only price anomalies."""
    detector = AnomalyDetector(config)
    return detector._detect_price_anomalies(df)


def detect_volume_anomalies(df: pd.DataFrame, config: Optional[AnomalyConfig] = None) -> pd.Series:
    """Detect only volume anomalies."""
    detector = AnomalyDetector(config)
    return detector._detect_volume_anomalies(df)


def get_anomaly_score(df: pd.DataFrame, config: Optional[AnomalyConfig] = None) -> pd.Series:
    """Get combined anomaly score."""
    detector = AnomalyDetector(config)
    df_result = detector.detect_all(df)
    return df_result['anomaly_combined']
