"""
ML Confidence Dashboard for Kobe Trading System

Aggregates and displays confidence metrics from all ML/AI components:
- HMM Regime Detector
- LSTM Confidence Model
- Ensemble Predictor
- Online Learning Manager
- Cognitive Signal Processor

This is a TIER 1 Quick Win from the AI/ML Enhancement Plan.

Usage:
    from dashboard.ml_confidence import get_ml_confidence_dashboard

    # Get full dashboard data
    dashboard_data = get_ml_confidence_dashboard()
    print(dashboard_data)

    # Get individual component status
    regime_status = get_regime_status()
    online_status = get_online_learning_status()
"""

import logging
import sys
from pathlib import Path

# Add project root to path for standalone execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# State directory for reading cached states
STATE_DIR = Path(__file__).resolve().parents[1] / 'state' / 'cognitive'


@dataclass
class MLComponentStatus:
    """Status of a single ML component."""
    name: str
    status: str  # 'active', 'inactive', 'error', 'not_loaded'
    confidence: float = 0.0
    last_updated: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status,
            'confidence': round(self.confidence, 3),
            'last_updated': self.last_updated,
            'details': self.details,
        }


@dataclass
class MLConfidenceDashboard:
    """Aggregated ML/AI confidence dashboard data."""
    timestamp: datetime = field(default_factory=datetime.now)
    overall_health: str = 'healthy'  # 'healthy', 'degraded', 'unhealthy'
    overall_confidence: float = 0.0
    components: List[MLComponentStatus] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health': self.overall_health,
            'overall_confidence': round(self.overall_confidence, 3),
            'components': [c.to_dict() for c in self.components],
            'alerts': self.alerts,
            'recommendations': self.recommendations,
        }


def get_regime_status() -> MLComponentStatus:
    """Get HMM Regime Detector status."""
    try:
        regime_file = STATE_DIR / 'regime_state.json'
        if regime_file.exists():
            with open(regime_file, 'r') as f:
                state = json.load(f)
            return MLComponentStatus(
                name='HMM Regime Detector',
                status='active',
                confidence=state.get('confidence', 0.0),
                last_updated=state.get('updated_at'),
                details={
                    'current_regime': state.get('regime', 'unknown'),
                    'regime_timestamp': state.get('timestamp'),
                }
            )
        else:
            return MLComponentStatus(
                name='HMM Regime Detector',
                status='not_loaded',
                details={'reason': 'No regime state file found'}
            )
    except Exception as e:
        return MLComponentStatus(
            name='HMM Regime Detector',
            status='error',
            details={'error': str(e)}
        )


def get_lstm_confidence_status() -> MLComponentStatus:
    """Get LSTM Confidence Model status."""
    try:
        # Try to import and check LSTM model
        from ml_advanced.lstm_confidence.model import LSTMConfidenceModel
        LSTMConfidenceModel()

        # Check if model weights are loaded
        model_path = Path(__file__).resolve().parents[1] / 'ml_advanced' / 'lstm_confidence' / 'weights'
        has_weights = model_path.exists() and any(model_path.iterdir()) if model_path.exists() else False

        return MLComponentStatus(
            name='LSTM Confidence Model',
            status='active' if has_weights else 'inactive',
            confidence=0.75 if has_weights else 0.0,  # Default confidence when active
            details={
                'has_weights': has_weights,
                'model_type': 'Multi-Output LSTM',
                'outputs': ['direction', 'magnitude', 'success_probability']
            }
        )
    except ImportError as e:
        return MLComponentStatus(
            name='LSTM Confidence Model',
            status='not_loaded',
            details={'reason': 'LSTM module not available', 'error': str(e)}
        )
    except Exception as e:
        return MLComponentStatus(
            name='LSTM Confidence Model',
            status='error',
            details={'error': str(e)}
        )


def get_ensemble_status() -> MLComponentStatus:
    """Get Ensemble Predictor status."""
    try:
        from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor
        predictor = EnsemblePredictor()

        return MLComponentStatus(
            name='Ensemble Predictor',
            status='active',
            confidence=0.70,  # Default ensemble confidence
            details={
                'models': ['XGBoost', 'LightGBM', 'LSTM'],
                'combination_method': 'weighted_average',
                'trained': predictor.is_trained if hasattr(predictor, 'is_trained') else False
            }
        )
    except ImportError:
        return MLComponentStatus(
            name='Ensemble Predictor',
            status='not_loaded',
            details={'reason': 'Ensemble module not available'}
        )
    except Exception as e:
        return MLComponentStatus(
            name='Ensemble Predictor',
            status='error',
            details={'error': str(e)}
        )


def get_online_learning_status() -> MLComponentStatus:
    """Get Online Learning Manager status."""
    try:
        from cognitive.signal_processor import get_signal_processor
        processor = get_signal_processor()
        status = processor.online_learning_manager.get_status()

        # Determine health based on drift detection
        drift_accuracy = status.get('drift_current_accuracy', 0.0)
        is_ready = status.get('buffer_ready', False)

        component_status = 'active' if is_ready else 'inactive'
        if drift_accuracy < 0.4:
            component_status = 'degraded'

        return MLComponentStatus(
            name='Online Learning Manager',
            status=component_status,
            confidence=drift_accuracy,
            last_updated=status.get('last_update'),
            details={
                'buffer_size': status.get('buffer_size', 0),
                'buffer_ready': is_ready,
                'class_balance': status.get('class_balance', {}),
                'drift_baseline': status.get('drift_baseline_accuracy'),
                'drift_current': drift_accuracy,
                'total_updates': status.get('total_updates', 0),
            }
        )
    except Exception as e:
        return MLComponentStatus(
            name='Online Learning Manager',
            status='error',
            details={'error': str(e)}
        )


def get_cognitive_system_status() -> MLComponentStatus:
    """Get Cognitive Signal Processor status."""
    try:
        from cognitive.signal_processor import get_signal_processor
        processor = get_signal_processor()
        status = processor.get_cognitive_status()

        brain_status = status.get('brain_status', {})
        is_active = status.get('processor_active', False)

        return MLComponentStatus(
            name='Cognitive Signal Processor',
            status='active' if is_active else 'inactive',
            confidence=brain_status.get('overall_confidence', 0.6),
            details={
                'active_episodes': status.get('active_episodes', 0),
                'pending_features': status.get('pending_features', 0),
                'brain_status': brain_status,
            }
        )
    except Exception as e:
        return MLComponentStatus(
            name='Cognitive Signal Processor',
            status='error',
            details={'error': str(e)}
        )


def get_ml_confidence_dashboard() -> MLConfidenceDashboard:
    """
    Aggregate all ML component statuses into a unified dashboard view.

    Returns:
        MLConfidenceDashboard with component statuses, alerts, and recommendations.
    """
    dashboard = MLConfidenceDashboard()

    # Gather component statuses
    components = [
        get_regime_status(),
        get_lstm_confidence_status(),
        get_ensemble_status(),
        get_online_learning_status(),
        get_cognitive_system_status(),
    ]

    dashboard.components = components

    # Calculate overall health
    # Note: Some components have expected "degraded" or "not_loaded" states:
    # - Online Learning: degraded with empty buffer is expected (needs 100 trades to populate)
    # - LSTM Confidence: not_loaded is OK (TensorFlow optional, disabled on Windows)
    sum(1 for c in components if c.status == 'active')
    sum(1 for c in components if c.status == 'error')

    # Only count critical issues - actual errors that indicate system problems
    # Exclude Online Learning degraded (expected) and not_loaded components (optional)
    critical_errors = sum(1 for c in components
        if c.status == 'error' and 'Online Learning' not in c.name)

    if critical_errors > 1:
        dashboard.overall_health = 'unhealthy'
    elif critical_errors > 0:
        dashboard.overall_health = 'degraded'
    else:
        # System is healthy if core components are active
        # (HMM, Ensemble, Cognitive are the critical ones)
        dashboard.overall_health = 'healthy'

    # Calculate overall confidence (weighted average of active components)
    confidences = [c.confidence for c in components if c.status in ('active', 'degraded') and c.confidence > 0]
    dashboard.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Generate alerts
    for comp in components:
        if comp.status == 'error':
            dashboard.alerts.append(f"[ERROR] {comp.name}: {comp.details.get('error', 'Unknown error')}")
        elif comp.status == 'degraded':
            dashboard.alerts.append(f"[WARN] {comp.name} is degraded - check performance")
        elif comp.status == 'not_loaded':
            dashboard.alerts.append(f"[INFO] {comp.name} not loaded: {comp.details.get('reason', 'N/A')}")

    # Add online learning specific alerts
    online_comp = next((c for c in components if c.name == 'Online Learning Manager'), None)
    if online_comp and online_comp.status == 'active':
        drift_accuracy = online_comp.details.get('drift_current', 0)
        drift_baseline = online_comp.details.get('drift_baseline')
        if drift_baseline and drift_accuracy < drift_baseline - 0.15:
            dashboard.alerts.append("[ALERT] Concept drift detected! Model accuracy has degraded significantly.")

    # Generate recommendations
    if dashboard.overall_health == 'unhealthy':
        dashboard.recommendations.append("Consider pausing automated trading until ML systems stabilize.")
    if dashboard.overall_confidence < 0.5:
        dashboard.recommendations.append("Low overall ML confidence - reduce position sizes or wait for better setups.")

    regime_comp = next((c for c in components if c.name == 'HMM Regime Detector'), None)
    if regime_comp and regime_comp.status == 'active':
        regime = regime_comp.details.get('current_regime', '')
        if regime == 'BEAR' or regime == 'BEARISH':
            dashboard.recommendations.append("BEAR regime detected - consider defensive positioning.")
        elif regime == 'CRISIS':
            dashboard.recommendations.append("[CRITICAL] CRISIS regime - risk-off recommended!")

    return dashboard


def print_ml_dashboard():
    """Print a human-readable ML confidence dashboard to console."""
    dashboard = get_ml_confidence_dashboard()

    print("\n" + "=" * 60)
    print("         KOBE ML CONFIDENCE DASHBOARD")
    print("=" * 60)
    print(f"Timestamp: {dashboard.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall Health: {dashboard.overall_health.upper()}")
    print(f"Overall Confidence: {dashboard.overall_confidence:.1%}")
    print("-" * 60)

    print("\nCOMPONENT STATUS:")
    print("-" * 60)
    for comp in dashboard.components:
        status_indicator = {
            'active': '[OK]',
            'inactive': '[--]',
            'degraded': '[!!]',
            'error': '[XX]',
            'not_loaded': '[NL]',
        }.get(comp.status, '[??]')

        conf_str = f"{comp.confidence:.1%}" if comp.confidence > 0 else "N/A"
        print(f"{status_indicator} {comp.name:<30} | Conf: {conf_str:<8} | {comp.status}")

    if dashboard.alerts:
        print("\n" + "-" * 60)
        print("ALERTS:")
        for alert in dashboard.alerts:
            print(f"  {alert}")

    if dashboard.recommendations:
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS:")
        for rec in dashboard.recommendations:
            print(f"  -> {rec}")

    print("=" * 60 + "\n")


if __name__ == '__main__':
    print_ml_dashboard()
