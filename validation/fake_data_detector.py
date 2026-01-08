"""
Fake Data Detector - Catches Hardcoded Placeholder Values

CRITICAL FIX (2026-01-08): This module detects when the system is using
fake/hardcoded values instead of real data. This is essential for a 24/7
autonomous trading robot because:

1. VIX = 20.0 always -> Hardcoded placeholder (should vary 10-80+)
2. ensemble_confidence = 0.5 always -> Hardcoded default (should vary 0-1)
3. regime = "unknown" always -> Not computing regime (should be BULL/BEAR/NEUTRAL)

If fake data is detected, trading should HALT to prevent bad decisions.

Usage:
    from validation.fake_data_detector import detect_fake_data, validate_signals_before_trading

    # Check signals before trading
    alerts = validate_signals_before_trading(signals_df)
    if alerts:
        halt_trading(reason="Fake data detected")
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FakeDataAlert:
    """An alert about detected fake/placeholder data."""
    field: str
    description: str
    expected_behavior: str
    actual_behavior: str
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    sample_values: List[Any]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'description': self.description,
            'expected_behavior': self.expected_behavior,
            'actual_behavior': self.actual_behavior,
            'severity': self.severity,
            'sample_values': self.sample_values[:5],  # Limit sample
            'recommendation': self.recommendation,
        }


# Detection rules for fake data
DETECTION_RULES = [
    {
        'field': 'ensemble_confidence',
        'check_fn': lambda vals: all(v == 0.5 for v in vals if v is not None),
        'description': 'Ensemble confidence is always 0.5 (default placeholder)',
        'expected': 'Values should vary between 0-1 based on model predictions',
        'severity': 'CRITICAL',
        'recommendation': 'Check cognitive/signal_processor.py conf_score extraction',
    },
    {
        'field': 'conf_score',
        'check_fn': lambda vals: all(v == 0.5 for v in vals if v is not None),
        'description': 'Confidence score is always 0.5 (default placeholder)',
        'expected': 'Values should vary between 0-1 based on signal strength',
        'severity': 'CRITICAL',
        'recommendation': 'Check unified_signal_enrichment.py confidence computation',
    },
    {
        'field': 'vix',
        'check_fn': lambda vals: all(v == 20.0 for v in vals if v is not None),
        'description': 'VIX is always 20.0 (hardcoded placeholder)',
        'expected': 'VIX should vary based on market conditions (typically 10-80)',
        'severity': 'CRITICAL',
        'recommendation': 'Check core/vix_monitor.py and ensure FRED API is working',
    },
    {
        'field': 'regime',
        'check_fn': lambda vals: all(str(v).lower() == 'unknown' for v in vals if v is not None),
        'description': 'Regime is always "unknown" (not computing)',
        'expected': 'Regime should be BULL, BEAR, NEUTRAL, or CHOPPY based on HMM',
        'severity': 'CRITICAL',
        'recommendation': 'Check ml_advanced/hmm_regime_detector.py and train HMM model',
    },
    {
        'field': 'cognitive_confidence',
        'check_fn': lambda vals: all(v == 0.0 for v in vals if v is not None),
        'description': 'Cognitive confidence is always 0.0 (not computed)',
        'expected': 'Values should vary based on cognitive deliberation',
        'severity': 'WARNING',
        'recommendation': 'Check cognitive/signal_processor.py cognitive evaluation',
    },
    {
        'field': 'ml_confidence',
        'check_fn': lambda vals: all(v == 0.5 for v in vals if v is not None) and len(vals) > 5,
        'description': 'ML confidence is always 0.5 (model not trained or returning default)',
        'expected': 'ML models should produce varying predictions',
        'severity': 'WARNING',
        'recommendation': 'Retrain ML models or check model loading',
    },
]


def detect_fake_data(signals: List[Dict[str, Any]]) -> List[FakeDataAlert]:
    """
    Scan signals for fake/hardcoded values.

    Args:
        signals: List of signal dictionaries

    Returns:
        List of FakeDataAlert objects for any detected issues
    """
    if not signals:
        return []

    alerts = []

    for rule in DETECTION_RULES:
        field = rule['field']

        # Extract values for this field
        values = [s.get(field) for s in signals if field in s]

        # Skip if no values to check
        if not values or len(values) < 2:
            continue

        # Filter out None and NaN
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]

        if not valid_values:
            continue

        # Check if the rule triggers
        try:
            if rule['check_fn'](valid_values):
                alert = FakeDataAlert(
                    field=field,
                    description=rule['description'],
                    expected_behavior=rule['expected'],
                    actual_behavior=f"All {len(valid_values)} values are identical",
                    severity=rule['severity'],
                    sample_values=valid_values[:5],
                    recommendation=rule['recommendation'],
                )
                alerts.append(alert)
                logger.warning(f"FAKE DATA DETECTED: {field} - {rule['description']}")
        except Exception as e:
            logger.debug(f"Rule check failed for {field}: {e}")

    return alerts


def detect_fake_data_in_df(df: pd.DataFrame) -> List[FakeDataAlert]:
    """
    Detect fake data in a pandas DataFrame.

    Args:
        df: DataFrame with signal data

    Returns:
        List of FakeDataAlert objects
    """
    if df.empty:
        return []

    signals = df.to_dict('records')
    return detect_fake_data(signals)


def validate_signals_before_trading(
    signals: Any,
    halt_on_critical: bool = True,
) -> Optional[List[FakeDataAlert]]:
    """
    Validate signals before trading - HALT if fake data detected.

    This should be called in scan.py and run_paper_trade.py BEFORE
    executing any trades.

    Args:
        signals: List of dicts or DataFrame
        halt_on_critical: If True, raises exception on CRITICAL alerts

    Returns:
        List of alerts, or None if validation passed

    Raises:
        FakeDataError: If halt_on_critical and CRITICAL alerts found
    """
    from core.structured_log import jlog

    # Convert DataFrame to list if needed
    if isinstance(signals, pd.DataFrame):
        if signals.empty:
            return None
        signal_list = signals.to_dict('records')
    elif isinstance(signals, list):
        signal_list = signals
    else:
        logger.warning(f"Unknown signals type: {type(signals)}")
        return None

    # Detect fake data
    alerts = detect_fake_data(signal_list)

    if not alerts:
        jlog('fake_data_check_passed', signals_checked=len(signal_list))
        return None

    # Log all alerts
    critical_count = 0
    warning_count = 0

    for alert in alerts:
        jlog(
            'fake_data_detected',
            field=alert.field,
            severity=alert.severity,
            description=alert.description,
            recommendation=alert.recommendation,
            level='WARN' if alert.severity != 'CRITICAL' else 'ERROR',
        )

        if alert.severity == 'CRITICAL':
            critical_count += 1
        else:
            warning_count += 1

    # Halt on critical alerts
    if halt_on_critical and critical_count > 0:
        error_msg = f"TRADING HALTED: {critical_count} CRITICAL fake data alerts detected"
        logger.error(error_msg)
        jlog('trading_halted', reason='fake_data', critical_alerts=critical_count)
        raise FakeDataError(error_msg, alerts)

    return alerts


class FakeDataError(Exception):
    """Raised when critical fake data is detected."""

    def __init__(self, message: str, alerts: List[FakeDataAlert]):
        super().__init__(message)
        self.alerts = alerts

    def get_summary(self) -> str:
        """Get a summary of all alerts."""
        lines = [str(self)]
        for alert in self.alerts:
            lines.append(f"  - {alert.field}: {alert.description}")
        return "\n".join(lines)


def check_value_variance(values: List[float], min_variance: float = 0.001) -> bool:
    """
    Check if values have sufficient variance (not all identical).

    Args:
        values: List of numeric values
        min_variance: Minimum acceptable variance

    Returns:
        True if variance is sufficient, False if values are too constant
    """
    if len(values) < 2:
        return True

    try:
        variance = np.var(values)
        return variance >= min_variance
    except Exception:
        return True


def get_fake_data_summary(alerts: List[FakeDataAlert]) -> Dict[str, Any]:
    """Get a summary of fake data detection results."""
    return {
        'total_alerts': len(alerts),
        'critical': sum(1 for a in alerts if a.severity == 'CRITICAL'),
        'warnings': sum(1 for a in alerts if a.severity == 'WARNING'),
        'info': sum(1 for a in alerts if a.severity == 'INFO'),
        'affected_fields': [a.field for a in alerts],
        'should_halt': any(a.severity == 'CRITICAL' for a in alerts),
    }
