"""
Trade Tweet Generator for Kobe Trading System

Generates Twitter-style trade summaries (max 280 chars) for sharing.
Uses the LLM analyzer when available, falls back to templates.

This is a TIER 1 Quick Win from the AI/ML Enhancement Plan.

Usage:
    from cognitive.tweet_generator import generate_trade_tweet

    tweet = generate_trade_tweet(signal, outcome=None)
    print(tweet)  # "LONG $TSLA @ $454.43 | IBS_RSI signal | Target: $480 | Stop: $419 #swingtrading"
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LLM analyzer
try:
    from cognitive.llm_trade_analyzer import get_trade_analyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def generate_trade_tweet(
    signal: Dict[str, Any],
    outcome: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    max_chars: int = 280
) -> str:
    """
    Generate a Twitter-style trade summary.

    Args:
        signal: Trade signal dict with keys like symbol, side, entry_price, stop_loss, take_profit, strategy, reason
        outcome: Optional outcome dict with pnl, win, holding_period
        use_llm: Whether to use LLM for generation (falls back to template if unavailable)
        max_chars: Maximum character length (default 280 for Twitter)

    Returns:
        Tweet-formatted string
    """
    # Extract signal fields
    symbol = signal.get('symbol', '???')
    side = signal.get('side', 'long').upper()
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    target = signal.get('take_profit', 0)
    strategy = signal.get('strategy', 'Unknown')
    reason = signal.get('reason', '')
    conf_score = signal.get('conf_score', 0)

    # Try LLM first if available and requested
    if use_llm and LLM_AVAILABLE and outcome is None:
        try:
            analyzer = get_trade_analyzer()
            prompt = f"""Generate a concise Twitter-style trade alert (max {max_chars} chars) for:
            Symbol: {symbol}
            Side: {side}
            Entry: ${entry:.2f}
            Stop: ${stop:.2f}
            Target: ${target:.2f}
            Strategy: {strategy}
            Signal: {reason}
            Confidence: {conf_score:.0%}

            Use emojis sparingly. Include hashtags like #swingtrading #stocks.
            Format: [Emoji] [SIDE] $[SYMBOL] @ $[PRICE] | [Brief reason] | Target: $X | Stop: $Y #hashtags
            """
            result = analyzer.analyze_custom(prompt, max_tokens=100)
            if result and len(result) <= max_chars:
                return result
        except Exception as e:
            logger.debug(f"LLM tweet generation failed: {e}")

    # Template-based generation
    if outcome is not None:
        # Post-trade summary
        return _generate_outcome_tweet(signal, outcome, max_chars)
    else:
        # Entry alert
        return _generate_entry_tweet(signal, max_chars)


def _generate_entry_tweet(signal: Dict[str, Any], max_chars: int = 280) -> str:
    """Generate entry alert tweet using template."""
    symbol = signal.get('symbol', '???')
    side = signal.get('side', 'long').upper()
    entry = signal.get('entry_price', 0)
    stop = signal.get('stop_loss', 0)
    target = signal.get('take_profit', 0)
    strategy = signal.get('strategy', 'IBS_RSI')
    conf_score = signal.get('conf_score', 0)

    # Calculate R:R
    if stop and entry and target:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0
    else:
        rr = 0

    # Side emoji
    emoji = "BULL" if side == "LONG" else "BEAR"

    # Build tweet
    tweet = f"{emoji} {side} ${symbol} @ ${entry:.2f}"

    # Add target/stop if space allows
    details = f" | Target: ${target:.2f} | Stop: ${stop:.2f}"
    if len(tweet) + len(details) <= max_chars - 30:
        tweet += details

    # Add R:R if meaningful
    if rr >= 1.5:
        rr_str = f" | R:R {rr:.1f}:1"
        if len(tweet) + len(rr_str) <= max_chars - 20:
            tweet += rr_str

    # Add confidence if high
    if conf_score >= 0.8:
        conf_str = f" | Conf: {conf_score:.0%}"
        if len(tweet) + len(conf_str) <= max_chars - 15:
            tweet += conf_str

    # Add hashtags
    hashtags = " #swingtrading #stocks"
    if strategy == 'IBS_RSI':
        hashtags = " #meanreversion #swingtrading"
    elif strategy == 'TURTLE_SOUP':
        hashtags = " #ICT #liquidity #swingtrading"

    if len(tweet) + len(hashtags) <= max_chars:
        tweet += hashtags

    return tweet[:max_chars]


def _generate_outcome_tweet(
    signal: Dict[str, Any],
    outcome: Dict[str, Any],
    max_chars: int = 280
) -> str:
    """Generate post-trade outcome tweet."""
    symbol = signal.get('symbol', '???')
    side = signal.get('side', 'long').upper()
    entry = signal.get('entry_price', 0)

    pnl = outcome.get('pnl', 0)
    pnl_pct = outcome.get('pnl_pct', 0)
    won = outcome.get('won', pnl > 0)
    holding = outcome.get('holding_period', 0)

    # Result emoji and text
    if won:
        emoji = "WIN"
        result_text = "WIN"
    else:
        emoji = "LOSS"
        result_text = "LOSS"

    # Format P&L
    pnl_sign = "+" if pnl >= 0 else ""
    pnl_str = f"{pnl_sign}${pnl:.2f}"
    if pnl_pct:
        pnl_str = f"{pnl_sign}{pnl_pct:.1f}%"

    # Build tweet
    tweet = f"{emoji} {result_text}: ${symbol} {side} | {pnl_str}"

    # Add holding period
    if holding:
        days_str = f" | Held: {holding}d"
        if len(tweet) + len(days_str) <= max_chars - 25:
            tweet += days_str

    # Add hashtags
    if won:
        hashtags = " #tradingwins #swingtrading"
    else:
        hashtags = " #tradinglessons #riskmanagement"

    if len(tweet) + len(hashtags) <= max_chars:
        tweet += hashtags

    return tweet[:max_chars]


def generate_daily_summary_tweet(
    signals_count: int,
    top_pick: Optional[Dict[str, Any]] = None,
    regime: str = "NEUTRAL"
) -> str:
    """
    Generate a daily market summary tweet.

    Args:
        signals_count: Number of signals generated today
        top_pick: The TOTD signal if available
        regime: Current market regime

    Returns:
        Tweet-formatted daily summary
    """
    date_str = datetime.now().strftime("%m/%d")

    # Regime emoji
    regime_emoji = {
        'BULLISH': 'BULL',
        'NEUTRAL': 'SIDEWAY',
        'BEARISH': 'BEAR',
        'CRISIS': 'WARNING'
    }.get(regime.upper(), 'SIDEWAY')

    tweet = f"Kobe Daily Scan {date_str} | {regime_emoji} {regime} regime | {signals_count} signals"

    if top_pick:
        symbol = top_pick.get('symbol', '???')
        conf = top_pick.get('conf_score', 0)
        tweet += f" | TOTD: ${symbol} ({conf:.0%} conf)"

    hashtags = " #tradingsignals #stockmarket"
    if len(tweet) + len(hashtags) <= 280:
        tweet += hashtags

    return tweet[:280]


if __name__ == '__main__':
    # Test the tweet generator
    test_signal = {
        'symbol': 'TSLA',
        'side': 'long',
        'entry_price': 454.43,
        'stop_loss': 419.35,
        'take_profit': 500.00,
        'strategy': 'IBS_RSI',
        'reason': 'IBS_RSI[ibs=0.06,rsi=0.0]',
        'conf_score': 0.85
    }

    print("Entry Tweet:")
    print(generate_trade_tweet(test_signal, use_llm=False))
    print(f"({len(generate_trade_tweet(test_signal, use_llm=False))} chars)")

    print("\nOutcome Tweet (win):")
    outcome_win = {'pnl': 250.50, 'pnl_pct': 5.5, 'won': True, 'holding_period': 3}
    print(generate_trade_tweet(test_signal, outcome=outcome_win))

    print("\nDaily Summary:")
    print(generate_daily_summary_tweet(5, test_signal, 'BULLISH'))
