"""
Signal generators for testing.

Provides signal generation utilities for:
- Valid signals (pass quality gate)
- Invalid signals (fail quality gate)
- Signal batches (multiple signals)
- Strategy-specific signals
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid


def create_valid_signal(
    symbol: str = "TEST",
    side: str = "BUY",
    entry_price: float = 100.0,
    stop_loss: float = 97.0,
    take_profit: float = 106.0,
    score: int = 75,
    confidence: float = 0.70,
    strategy: str = "dual_strategy",
    timestamp: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a valid signal that passes quality gate.

    Args:
        symbol: Stock symbol
        side: BUY or SELL
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        score: Signal score (0-100)
        confidence: Confidence level (0-1)
        strategy: Strategy name
        timestamp: Signal timestamp
        extra: Additional fields

    Returns:
        Signal dictionary
    """
    if timestamp is None:
        timestamp = datetime.now()

    signal = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "score": score,
        "confidence": confidence,
        "strategy": strategy,
        "reason": f"{strategy} signal triggered",
        "risk_reward": round((take_profit - entry_price) / (entry_price - stop_loss), 2),
    }

    if extra:
        signal.update(extra)

    return signal


def create_invalid_signal(
    reason: str = "low_score",
    symbol: str = "TEST",
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create an invalid signal that fails quality gate.

    Args:
        reason: Why signal is invalid (low_score, low_confidence, poor_rr, etc.)
        symbol: Stock symbol
        timestamp: Signal timestamp

    Returns:
        Signal dictionary that will fail quality checks
    """
    if timestamp is None:
        timestamp = datetime.now()

    base_signal = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": "BUY",
        "entry_price": 100.0,
        "strategy": "test_strategy",
        "reason": f"Test invalid signal: {reason}",
    }

    if reason == "low_score":
        base_signal.update({
            "score": 45,  # Below 70 threshold
            "confidence": 0.70,
            "stop_loss": 97.0,
            "take_profit": 106.0,
            "risk_reward": 2.0,
        })
    elif reason == "low_confidence":
        base_signal.update({
            "score": 75,
            "confidence": 0.45,  # Below 0.60 threshold
            "stop_loss": 97.0,
            "take_profit": 106.0,
            "risk_reward": 2.0,
        })
    elif reason == "poor_rr":
        base_signal.update({
            "score": 75,
            "confidence": 0.70,
            "stop_loss": 99.0,  # Tight stop
            "take_profit": 100.5,  # Small target
            "risk_reward": 0.5,  # Below 1.5:1 threshold
        })
    elif reason == "missing_stop":
        base_signal.update({
            "score": 75,
            "confidence": 0.70,
            "take_profit": 106.0,
            # No stop_loss
        })
    elif reason == "missing_target":
        base_signal.update({
            "score": 75,
            "confidence": 0.70,
            "stop_loss": 97.0,
            # No take_profit
        })
    else:
        # Generic invalid
        base_signal.update({
            "score": 30,
            "confidence": 0.30,
            "risk_reward": 0.5,
        })

    return base_signal


def create_signal_batch(
    count: int = 5,
    symbols: Optional[List[str]] = None,
    valid_ratio: float = 0.6,
    timestamp: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Create a batch of signals with mix of valid and invalid.

    Args:
        count: Number of signals to create
        symbols: List of symbols (cycles through if fewer than count)
        valid_ratio: Ratio of valid signals (0-1)
        timestamp: Base timestamp

    Returns:
        List of signal dictionaries
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    if timestamp is None:
        timestamp = datetime.now()

    signals = []
    valid_count = int(count * valid_ratio)

    invalid_reasons = ["low_score", "low_confidence", "poor_rr"]

    for i in range(count):
        symbol = symbols[i % len(symbols)]

        if i < valid_count:
            signal = create_valid_signal(
                symbol=symbol,
                entry_price=100.0 + i * 10,
                score=70 + (i % 20),
                confidence=0.60 + (i % 30) / 100,
                timestamp=timestamp,
            )
        else:
            reason = invalid_reasons[(i - valid_count) % len(invalid_reasons)]
            signal = create_invalid_signal(
                reason=reason,
                symbol=symbol,
                timestamp=timestamp,
            )

        signals.append(signal)

    return signals


def create_dual_strategy_signal(
    symbol: str = "TEST",
    sub_strategy: str = "ibs_rsi",
    score: int = 75,
    confidence: float = 0.70,
    markov_boost: float = 0.0,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Create a DualStrategyScanner signal with full metadata.

    Args:
        symbol: Stock symbol
        sub_strategy: Which sub-strategy triggered (ibs_rsi or turtle_soup)
        score: Signal score
        confidence: Confidence level
        markov_boost: Markov chain confidence boost
        timestamp: Signal timestamp

    Returns:
        Signal dictionary matching DualStrategyScanner output
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Base prices vary by sub-strategy
    if sub_strategy == "ibs_rsi":
        entry_price = 100.0
        stop_loss = 97.0  # ATR-based stop
        take_profit = 106.0  # 2:1 R:R
        reason = "IBS < 0.08, RSI(2) < 5, Above SMA(200)"
    else:  # turtle_soup
        entry_price = 98.0  # Near sweep level
        stop_loss = 95.0  # Below sweep low
        take_profit = 105.0  # Liquidity target
        reason = "Sweep >= 0.3 ATR below 20-day low, recovered"

    final_confidence = min(1.0, confidence + markov_boost)

    return {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "side": "BUY",
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "score": score,
        "confidence": confidence,
        "conf_score": final_confidence,  # After Markov boost
        "strategy": "dual_strategy",
        "sub_strategy": sub_strategy,
        "reason": reason,
        "risk_reward": round((take_profit - entry_price) / (entry_price - stop_loss), 2),
        "markov_boost": markov_boost,
        "quality_tier": "STANDARD" if score >= 70 else "ELITE" if score >= 85 else "MARGINAL",
    }
