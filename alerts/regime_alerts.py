"""
Regime Transition Alert System for Kobe Trading System

Monitors market regime changes using the HMM Regime Detector and sends
Telegram alerts when transitions occur (e.g., BULL → BEAR).

This is a TIER 1 Quick Win from the AI/ML Enhancement Plan.

Usage:
    from alerts.regime_alerts import check_regime_transition

    # Call daily (e.g., after market close)
    check_regime_transition()
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# State file for tracking previous regime
STATE_DIR = Path(__file__).resolve().parents[1] / 'state' / 'cognitive'
REGIME_STATE_FILE = STATE_DIR / 'regime_state.json'


def _load_previous_regime() -> Optional[Dict[str, Any]]:
    """Load the previous regime state from disk."""
    if REGIME_STATE_FILE.exists():
        try:
            with open(REGIME_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load previous regime state: {e}")
    return None


def _save_current_regime(regime: str, confidence: float, timestamp: str) -> None:
    """Save the current regime state to disk."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        'regime': regime,
        'confidence': confidence,
        'timestamp': timestamp,
        'updated_at': datetime.now().isoformat()
    }
    with open(REGIME_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def check_regime_transition(
    send_alert: bool = True,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Check for market regime transitions and optionally send Telegram alert.

    This function:
    1. Fetches current SPY data
    2. Runs HMM regime detection
    3. Compares with previous regime
    4. Sends alert if regime changed
    5. Saves new regime state

    Args:
        send_alert: Whether to send Telegram alert on transition
        verbose: Print debug info

    Returns:
        Dict with transition info if a change occurred, None otherwise
    """
    # Import here to avoid circular imports
    try:
        from ml_advanced.hmm_regime_detector import AdaptiveRegimeDetector
        from core.regime_filter import fetch_spy_bars
        from alerts.telegram_alerter import TelegramAlerter
    except ImportError as e:
        logger.error(f"Could not import required modules: {e}")
        return None

    # Get current regime
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        spy_bars = fetch_spy_bars(start_date, end_date)
        if spy_bars is None or spy_bars.empty:
            logger.warning("Could not fetch SPY data for regime detection")
            return None

        # Normalize column names (HMM detector expects 'Close', data may have 'close')
        if 'close' in spy_bars.columns and 'Close' not in spy_bars.columns:
            spy_bars = spy_bars.rename(columns={'close': 'Close'})
        if 'open' in spy_bars.columns and 'Open' not in spy_bars.columns:
            spy_bars = spy_bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})

        detector = AdaptiveRegimeDetector(use_hmm=True)
        regime_state = detector.detect_regime(spy_bars, vix_data=None)

        current_regime = regime_state.regime.value
        current_confidence = regime_state.confidence
        current_timestamp = datetime.now().isoformat()

        if verbose:
            print(f"Current regime: {current_regime} (conf={current_confidence:.2f})")

    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return None

    # Load previous regime
    previous = _load_previous_regime()
    previous_regime = previous.get('regime') if previous else None

    # Check for transition
    transition = None
    if previous_regime and previous_regime != current_regime:
        transition = {
            'from_regime': previous_regime,
            'to_regime': current_regime,
            'confidence': current_confidence,
            'timestamp': current_timestamp,
            'previous_timestamp': previous.get('timestamp') if previous else None
        }

        logger.info(f"REGIME TRANSITION: {previous_regime} → {current_regime}")

        if send_alert:
            try:
                alerter = TelegramAlerter()

                # Build alert message
                emoji_map = {
                    'BULLISH': '🟢',
                    'NEUTRAL': '🟡',
                    'BEARISH': '🔴',
                    'CRISIS': '🚨'
                }
                from_emoji = emoji_map.get(previous_regime, '⚪')
                to_emoji = emoji_map.get(current_regime, '⚪')

                message = (
                    f"⚠️ <b>REGIME TRANSITION</b>\n\n"
                    f"{from_emoji} {previous_regime} → {to_emoji} <b>{current_regime}</b>\n\n"
                    f"📊 Confidence: {current_confidence:.0%}\n"
                    f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}\n\n"
                )

                # Add position sizing recommendation
                if current_regime == 'BULLISH':
                    message += "💡 <i>Full position sizes recommended</i>"
                elif current_regime == 'NEUTRAL':
                    message += "💡 <i>Half position sizes recommended</i>"
                elif current_regime == 'BEARISH':
                    message += "💡 <i>Quarter position sizes or stand aside</i>"
                elif current_regime == 'CRISIS':
                    message += "🚨 <i>RISK OFF - Consider closing all positions</i>"

                alerter.send_message(message)
                logger.info("Regime transition alert sent to Telegram")

            except Exception as e:
                logger.warning(f"Could not send Telegram alert: {e}")

    # Save current regime (always, even if no transition)
    _save_current_regime(current_regime, current_confidence, current_timestamp)

    return transition


def get_current_regime() -> Optional[Dict[str, Any]]:
    """
    Get the current stored regime state without checking for transitions.

    Returns:
        Dict with regime, confidence, timestamp, or None if not available
    """
    return _load_previous_regime()


if __name__ == '__main__':
    # Test the regime transition check
    import sys
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print("Checking for regime transition...")
    result = check_regime_transition(send_alert=False, verbose=True)

    if result:
        print("\n[!] TRANSITION DETECTED:")
        print(f"   {result['from_regime']} -> {result['to_regime']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    else:
        current = get_current_regime()
        if current:
            print(f"\n[i] No transition. Current regime: {current['regime']}")
        else:
            print("\n[!] No previous regime state found (first run)")
