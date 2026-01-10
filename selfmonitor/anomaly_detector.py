"""
Anomaly Detection for Trading Systems
======================================

Detects unusual patterns in market data, trades, and system behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Type of anomaly detected."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    RETURN_OUTLIER = "return_outlier"
    CORRELATION_BREAK = "correlation_break"
    SPREAD_WIDENING = "spread_widening"
    DATA_GAP = "data_gap"
    SYSTEM_LAG = "system_lag"


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    anomaly_type: AnomalyType
    symbol: str
    detected_at: datetime = field(default_factory=datetime.now)
    severity: float = 0.0  # 0-1 scale
    value: float = 0.0
    threshold: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.anomaly_type.value,
            'symbol': self.symbol,
            'detected_at': self.detected_at.isoformat(),
            'severity': self.severity,
            'value': self.value,
            'threshold': self.threshold,
            'description': self.description,
        }


class AnomalyDetector:
    """
    Detects anomalies in market data and system behavior.

    Uses statistical methods to identify unusual patterns
    that may indicate data issues or market stress.
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        lookback_window: int = 50,
    ):
        self.zscore_threshold = zscore_threshold
        self.lookback_window = lookback_window
        self._alerts: List[AnomalyAlert] = []

        logger.info(f"AnomalyDetector initialized with zscore_threshold={zscore_threshold}")

    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate rolling z-score."""
        rolling_mean = series.rolling(self.lookback_window).mean()
        rolling_std = series.rolling(self.lookback_window).std()
        return (series - rolling_mean) / rolling_std

    def detect_price_anomaly(
        self,
        prices: pd.Series,
        symbol: str = "UNKNOWN",
    ) -> List[AnomalyAlert]:
        """Detect price anomalies."""
        alerts = []

        if len(prices) < self.lookback_window:
            return alerts

        returns = prices.pct_change().dropna()
        zscores = self._calculate_zscore(returns)

        for idx, zscore in zscores.items():
            if abs(zscore) > self.zscore_threshold:
                severity = min(1.0, abs(zscore) / 5.0)
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.RETURN_OUTLIER,
                    symbol=symbol,
                    severity=severity,
                    value=zscore,
                    threshold=self.zscore_threshold,
                    description=f"Return z-score of {zscore:.2f} exceeds threshold",
                )
                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    def detect_volume_anomaly(
        self,
        volume: pd.Series,
        symbol: str = "UNKNOWN",
    ) -> List[AnomalyAlert]:
        """Detect volume anomalies."""
        alerts = []

        if len(volume) < self.lookback_window:
            return alerts

        zscores = self._calculate_zscore(volume)

        for idx, zscore in zscores.items():
            if zscore > self.zscore_threshold * 1.5:  # Higher threshold for volume
                severity = min(1.0, zscore / 6.0)
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    symbol=symbol,
                    severity=severity,
                    value=zscore,
                    threshold=self.zscore_threshold * 1.5,
                    description=f"Volume spike: z-score {zscore:.2f}",
                )
                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    def detect_data_gaps(
        self,
        timestamps: pd.DatetimeIndex,
        expected_freq: str = "D",
    ) -> List[AnomalyAlert]:
        """Detect gaps in data."""
        alerts = []

        if len(timestamps) < 2:
            return alerts

        diffs = timestamps.to_series().diff()
        expected_gap = pd.Timedelta(expected_freq)

        for idx, diff in diffs.items():
            if pd.notna(diff) and diff > expected_gap * 3:
                alert = AnomalyAlert(
                    anomaly_type=AnomalyType.DATA_GAP,
                    symbol="SYSTEM",
                    severity=0.5,
                    value=diff.days,
                    threshold=3,
                    description=f"Data gap of {diff.days} days detected",
                )
                alerts.append(alert)
                self._alerts.append(alert)

        return alerts

    def check_series(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> List[AnomalyAlert]:
        """Check a complete OHLCV series for anomalies."""
        alerts = []

        if 'close' in df.columns:
            alerts.extend(self.detect_price_anomaly(df['close'], symbol))

        if 'volume' in df.columns:
            alerts.extend(self.detect_volume_anomaly(df['volume'], symbol))

        if isinstance(df.index, pd.DatetimeIndex):
            alerts.extend(self.detect_data_gaps(df.index))

        return alerts

    def is_current_anomalous(
        self,
        value: float,
        historical: pd.Series,
    ) -> Tuple[bool, float]:
        """Check if current value is anomalous."""
        if len(historical) < self.lookback_window:
            return False, 0.0

        mean = historical.tail(self.lookback_window).mean()
        std = historical.tail(self.lookback_window).std()

        if std == 0:
            return False, 0.0

        zscore = (value - mean) / std
        is_anomalous = abs(zscore) > self.zscore_threshold

        return is_anomalous, zscore

    def get_recent_alerts(
        self,
        hours: int = 24,
    ) -> List[AnomalyAlert]:
        """Get alerts from recent hours."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        return [a for a in self._alerts if a.detected_at > cutoff]

    def clear_alerts(self):
        """Clear alert history."""
        self._alerts = []


def detect_anomalies(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
) -> List[AnomalyAlert]:
    """Convenience function to detect anomalies."""
    detector = AnomalyDetector()
    return detector.check_series(df, symbol)


def is_anomalous(value: float, historical: pd.Series) -> bool:
    """Check if a value is anomalous."""
    detector = AnomalyDetector()
    result, _ = detector.is_current_anomalous(value, historical)
    return result


# Global instance
_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    """Get or create global detector."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector
