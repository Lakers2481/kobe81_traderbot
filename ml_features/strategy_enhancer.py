"""
Strategy Enhancer Module.

Provides ML enhancement capabilities for existing trading strategies:
- Signal confidence scoring
- Regime-aware filtering
- Anomaly-based signal rejection
- Feature-enriched signal output

This module wraps existing strategies to add ML intelligence
without modifying the original strategy code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Protocol, runtime_checkable

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structured_log import jlog
from .feature_pipeline import FeaturePipeline, FeatureConfig
from .signal_confidence import SignalConfidence, ConfidenceConfig
from .regime_ml import RegimeDetectorML, RegimeConfig, RegimeState
from .anomaly_detection import AnomalyDetector, AnomalyConfig


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol for strategy classes."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price data."""
        ...

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scan for signals across all bars (for backtesting)."""
        ...


@dataclass
class EnhancerConfig:
    """Configuration for strategy enhancement."""
    # Enable/disable features
    enable_confidence: bool = True
    enable_regime_filter: bool = True
    enable_anomaly_filter: bool = True
    enable_feature_enrichment: bool = True

    # Confidence settings
    min_confidence: float = 0.4  # Minimum confidence to keep signal
    confidence_config: Optional[ConfidenceConfig] = None

    # Regime settings
    allowed_trend_regimes: List[str] = field(default_factory=lambda: [
        RegimeState.BULL.value,
        RegimeState.STRONG_BULL.value,
        RegimeState.NEUTRAL.value,
    ])
    allowed_vol_regimes: List[str] = field(default_factory=lambda: [
        RegimeState.LOW_VOL.value,
        RegimeState.NORMAL_VOL.value,
    ])
    regime_config: Optional[RegimeConfig] = None

    # Anomaly settings
    max_anomaly_score: float = 0.7  # Reject signals above this anomaly score
    anomaly_config: Optional[AnomalyConfig] = None

    # Feature settings
    feature_config: Optional[FeatureConfig] = None

    # Output settings
    include_ml_columns: bool = True  # Add ML columns to signal output


class StrategyEnhancer:
    """
    Enhances trading strategies with ML features.

    Wraps any strategy implementing StrategyProtocol to add:
    - Confidence scoring for each signal
    - Regime-based filtering
    - Anomaly rejection
    - Feature enrichment
    """

    def __init__(self, strategy: Any, config: Optional[EnhancerConfig] = None):
        """
        Initialize the enhancer.

        Args:
            strategy: A strategy instance (must have generate_signals method)
            config: Enhancement configuration
        """
        self.strategy = strategy
        self.config = config or EnhancerConfig()

        # Initialize ML components
        self._feature_pipeline = FeaturePipeline(self.config.feature_config)
        self._confidence_scorer = SignalConfidence(self.config.confidence_config)
        self._regime_detector = RegimeDetectorML(self.config.regime_config)
        self._anomaly_detector = AnomalyDetector(self.config.anomaly_config)

        # Track statistics
        self._stats = {
            'total_signals': 0,
            'confidence_filtered': 0,
            'regime_filtered': 0,
            'anomaly_filtered': 0,
            'passed_signals': 0,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML-enhanced signals.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with enhanced signals
        """
        if df.empty:
            return pd.DataFrame()

        # Extract ML features
        df_enhanced = self._prepare_data(df)

        # Generate base signals from strategy
        base_signals = self.strategy.generate_signals(df)

        if base_signals.empty:
            return base_signals

        self._stats['total_signals'] += len(base_signals)

        # Apply ML enhancements
        enhanced_signals = self._enhance_signals(base_signals, df_enhanced)

        self._stats['passed_signals'] += len(enhanced_signals)

        jlog("signals_enhanced", level="DEBUG",
             total=self._stats['total_signals'],
             passed=self._stats['passed_signals'],
             confidence_filtered=self._stats['confidence_filtered'],
             regime_filtered=self._stats['regime_filtered'],
             anomaly_filtered=self._stats['anomaly_filtered'])

        return enhanced_signals

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan for ML-enhanced signals across all bars.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all enhanced signals
        """
        if df.empty:
            return pd.DataFrame()

        # Extract ML features for entire dataset
        df_enhanced = self._prepare_data(df)

        # Generate base signals from strategy
        base_signals = self.strategy.scan_signals_over_time(df)

        if base_signals.empty:
            return base_signals

        self._stats['total_signals'] += len(base_signals)

        # Apply ML enhancements
        enhanced_signals = self._enhance_signals(base_signals, df_enhanced)

        self._stats['passed_signals'] += len(enhanced_signals)

        return enhanced_signals

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with ML features."""
        df_enhanced = df.copy()

        # Add technical features
        if self.config.enable_feature_enrichment:
            df_enhanced = self._feature_pipeline.extract(df_enhanced)

        # Add regime detection
        if self.config.enable_regime_filter:
            self._regime_detector.fit(df_enhanced)
            df_enhanced = self._regime_detector.detect(df_enhanced)

        # Add anomaly detection
        if self.config.enable_anomaly_filter:
            df_enhanced = self._anomaly_detector.detect_all(df_enhanced)

        return df_enhanced

    def _enhance_signals(self, signals: pd.DataFrame, df_enhanced: pd.DataFrame) -> pd.DataFrame:
        """Apply ML enhancements to signals."""
        if signals.empty:
            return signals

        signals = signals.copy()

        # Add confidence scores
        if self.config.enable_confidence:
            signals = self._add_confidence(signals, df_enhanced)
            # Filter by confidence
            pre_count = len(signals)
            signals = signals[signals['confidence'] >= self.config.min_confidence]
            self._stats['confidence_filtered'] += pre_count - len(signals)

        # Filter by regime
        if self.config.enable_regime_filter:
            pre_count = len(signals)
            signals = self._filter_by_regime(signals, df_enhanced)
            self._stats['regime_filtered'] += pre_count - len(signals)

        # Filter by anomaly
        if self.config.enable_anomaly_filter:
            pre_count = len(signals)
            signals = self._filter_by_anomaly(signals, df_enhanced)
            self._stats['anomaly_filtered'] += pre_count - len(signals)

        # Add ML columns if requested
        if self.config.include_ml_columns:
            signals = self._add_ml_columns(signals, df_enhanced)

        return signals

    def _add_confidence(self, signals: pd.DataFrame, df_enhanced: pd.DataFrame) -> pd.DataFrame:
        """Add confidence scores to signals."""
        confidence_scores = []
        confidence_levels = []

        for idx, signal_row in signals.iterrows():
            side = signal_row.get('side', 'long')
            result = self._confidence_scorer.score(signal_row, df_enhanced, side)
            confidence_scores.append(result.score)
            confidence_levels.append(result.level.value)

        signals['confidence'] = confidence_scores
        signals['confidence_level'] = confidence_levels

        return signals

    def _filter_by_regime(self, signals: pd.DataFrame, df_enhanced: pd.DataFrame) -> pd.DataFrame:
        """Filter signals based on market regime."""
        if 'trend_regime' not in df_enhanced.columns or 'vol_regime' not in df_enhanced.columns:
            return signals

        filtered_signals = []

        for idx, signal_row in signals.iterrows():
            # Get regime at signal timestamp
            timestamp = signal_row.get('timestamp')
            if timestamp is not None:
                mask = df_enhanced['timestamp'] <= timestamp if 'timestamp' in df_enhanced.columns else df_enhanced.index <= timestamp
                regime_data = df_enhanced[mask].iloc[-1] if mask.sum() > 0 else None
            else:
                regime_data = df_enhanced.iloc[-1]

            if regime_data is None:
                filtered_signals.append(signal_row)
                continue

            trend_regime = regime_data.get('trend_regime', RegimeState.NEUTRAL.value)
            vol_regime = regime_data.get('vol_regime', RegimeState.NORMAL_VOL.value)

            # Check if regime is allowed
            trend_ok = trend_regime in self.config.allowed_trend_regimes
            vol_ok = vol_regime in self.config.allowed_vol_regimes

            if trend_ok and vol_ok:
                filtered_signals.append(signal_row)

        if not filtered_signals:
            return pd.DataFrame(columns=signals.columns)

        return pd.DataFrame(filtered_signals)

    def _filter_by_anomaly(self, signals: pd.DataFrame, df_enhanced: pd.DataFrame) -> pd.DataFrame:
        """Filter signals based on anomaly detection."""
        if 'anomaly_combined' not in df_enhanced.columns:
            return signals

        filtered_signals = []

        for idx, signal_row in signals.iterrows():
            # Get anomaly score at signal timestamp
            timestamp = signal_row.get('timestamp')
            if timestamp is not None:
                mask = df_enhanced['timestamp'] <= timestamp if 'timestamp' in df_enhanced.columns else df_enhanced.index <= timestamp
                anomaly_data = df_enhanced[mask].iloc[-1] if mask.sum() > 0 else None
            else:
                anomaly_data = df_enhanced.iloc[-1]

            if anomaly_data is None:
                filtered_signals.append(signal_row)
                continue

            anomaly_score = anomaly_data.get('anomaly_combined', 0.0)

            # Reject signals with high anomaly scores
            if anomaly_score < self.config.max_anomaly_score:
                filtered_signals.append(signal_row)

        if not filtered_signals:
            return pd.DataFrame(columns=signals.columns)

        return pd.DataFrame(filtered_signals)

    def _add_ml_columns(self, signals: pd.DataFrame, df_enhanced: pd.DataFrame) -> pd.DataFrame:
        """Add ML-related columns to signal output."""
        if signals.empty:
            return signals

        ml_columns = ['trend_regime', 'vol_regime', 'combined_regime',
                      'anomaly_combined', 'regime_confidence']

        for col in ml_columns:
            if col in df_enhanced.columns:
                values = []
                for _, row in signals.iterrows():
                    val = self._get_value_at_timestamp(df_enhanced, row, col)
                    values.append(val)
                signals[col] = values

        return signals

    def _get_value_at_timestamp(self, df: pd.DataFrame, signal_row: pd.Series, col: str) -> Any:
        """Get value from df at signal timestamp."""
        timestamp = signal_row.get('timestamp')
        if timestamp is not None and 'timestamp' in df.columns:
            mask = df['timestamp'] <= timestamp
            if mask.sum() > 0:
                return df[mask].iloc[-1].get(col, None)
        return df.iloc[-1].get(col, None) if not df.empty else None

    def get_stats(self) -> Dict[str, int]:
        """Get enhancement statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset enhancement statistics."""
        self._stats = {
            'total_signals': 0,
            'confidence_filtered': 0,
            'regime_filtered': 0,
            'anomaly_filtered': 0,
            'passed_signals': 0,
        }


def enhance_strategy(strategy: Any, config: Optional[EnhancerConfig] = None) -> StrategyEnhancer:
    """
    Wrap a strategy with ML enhancements.

    Args:
        strategy: Strategy instance to enhance
        config: Enhancement configuration

    Returns:
        Enhanced strategy wrapper
    """
    return StrategyEnhancer(strategy, config)
