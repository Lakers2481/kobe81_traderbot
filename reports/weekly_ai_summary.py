"""
Weekly AI Summary Generator for Kobe Trading System

Generates human-readable weekly performance narratives using Claude.
Summarizes wins, losses, regime changes, and lessons learned.

This is a TIER 1 Quick Win from the AI/ML Enhancement Plan.

Usage:
    from reports.weekly_ai_summary import generate_weekly_summary

    summary = generate_weekly_summary()
    print(summary)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / 'logs'
STATE_DIR = ROOT / 'state'

# Try to import LLM analyzer
try:
    from cognitive.llm_trade_analyzer import get_trade_analyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def _load_signals_for_week() -> List[Dict]:
    """Load signals from the past week from signals.jsonl."""
    signals = []
    signals_file = LOGS_DIR / 'signals.jsonl'

    if not signals_file.exists():
        return signals

    week_ago = datetime.now() - timedelta(days=7)

    try:
        with open(signals_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        signal = json.loads(line)
                        # Check timestamp
                        ts = signal.get('timestamp', '')
                        if ts:
                            sig_date = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            if sig_date.replace(tzinfo=None) >= week_ago:
                                signals.append(signal)
                    except (json.JSONDecodeError, ValueError):
                        continue
    except Exception as e:
        logger.warning(f"Could not load signals: {e}")

    return signals


def _load_trade_outcomes() -> List[Dict]:
    """Load trade outcomes from hash chain or events log."""
    outcomes = []
    events_file = LOGS_DIR / 'events.jsonl'

    if not events_file.exists():
        return outcomes

    week_ago = datetime.now() - timedelta(days=7)

    try:
        with open(events_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        # Look for fill or exit events
                        event_type = event.get('event', '')
                        if event_type in ['fill', 'exit', 'trade_complete']:
                            ts = event.get('ts', '')
                            if ts:
                                event_date = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                if event_date.replace(tzinfo=None) >= week_ago:
                                    outcomes.append(event)
                    except (json.JSONDecodeError, ValueError):
                        continue
    except Exception as e:
        logger.warning(f"Could not load events: {e}")

    return outcomes


def _load_regime_history() -> List[Dict]:
    """Load regime state history."""
    regime_file = STATE_DIR / 'cognitive' / 'regime_state.json'
    if regime_file.exists():
        try:
            with open(regime_file, 'r') as f:
                return [json.load(f)]
        except Exception:
            pass
    return []


def _calculate_weekly_stats(signals: List[Dict], outcomes: List[Dict]) -> Dict[str, Any]:
    """Calculate weekly performance statistics."""
    stats = {
        'total_signals': len(signals),
        'total_trades': len(outcomes),
        'strategies': {},
        'symbols': set(),
        'avg_confidence': 0,
    }

    # Count by strategy
    for sig in signals:
        strategy = sig.get('strategy', 'Unknown')
        stats['strategies'][strategy] = stats['strategies'].get(strategy, 0) + 1
        stats['symbols'].add(sig.get('symbol', ''))
        if sig.get('conf_score'):
            stats['avg_confidence'] += sig.get('conf_score', 0)

    if signals:
        stats['avg_confidence'] /= len(signals)

    stats['symbols'] = list(stats['symbols'])

    # Calculate P&L from outcomes
    total_pnl = 0
    wins = 0
    losses = 0
    for out in outcomes:
        pnl = out.get('pnl', 0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    stats['total_pnl'] = total_pnl
    stats['wins'] = wins
    stats['losses'] = losses
    stats['win_rate'] = wins / (wins + losses) if (wins + losses) > 0 else 0

    return stats


def generate_weekly_summary(use_llm: bool = True) -> str:
    """
    Generate a weekly performance summary.

    Args:
        use_llm: Whether to use Claude for narrative generation

    Returns:
        Human-readable weekly summary
    """
    # Load data
    signals = _load_signals_for_week()
    outcomes = _load_trade_outcomes()
    regimes = _load_regime_history()

    # Calculate stats
    stats = _calculate_weekly_stats(signals, outcomes)

    # Current regime
    current_regime = regimes[-1].get('regime', 'NEUTRAL') if regimes else 'NEUTRAL'

    # Try LLM generation
    if use_llm and LLM_AVAILABLE:
        try:
            analyzer = get_trade_analyzer()

            prompt = f"""Generate a brief weekly trading performance summary (2-3 paragraphs) based on:

            Week Ending: {datetime.now().strftime('%Y-%m-%d')}

            STATISTICS:
            - Total Signals: {stats['total_signals']}
            - Total Trades: {stats['total_trades']}
            - Win Rate: {stats['win_rate']:.1%}
            - Wins: {stats['wins']}, Losses: {stats['losses']}
            - Total P&L: ${stats['total_pnl']:.2f}
            - Strategies Used: {stats['strategies']}
            - Symbols Traded: {', '.join(stats['symbols'][:10])}
            - Average Confidence: {stats['avg_confidence']:.1%}
            - Current Market Regime: {current_regime}

            Write in a professional but conversational tone. Include:
            1. Overall performance assessment
            2. Notable wins or lessons from losses
            3. Market regime context
            4. One actionable insight for next week

            Keep it concise (under 500 words).
            """

            summary = analyzer.analyze_custom(prompt, max_tokens=600)
            if summary:
                return summary

        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")

    # Template fallback
    return _generate_template_summary(stats, current_regime)


def _generate_template_summary(stats: Dict, regime: str) -> str:
    """Generate summary using template (fallback)."""
    date_range = f"{(datetime.now() - timedelta(days=7)).strftime('%m/%d')} - {datetime.now().strftime('%m/%d/%Y')}"

    summary = f"""
KOBE WEEKLY TRADING SUMMARY
Week: {date_range}
{'=' * 40}

PERFORMANCE METRICS:
- Total Signals Generated: {stats['total_signals']}
- Total Trades Executed: {stats['total_trades']}
- Win Rate: {stats['win_rate']:.1%}
- Wins: {stats['wins']} | Losses: {stats['losses']}
- Total P&L: ${stats['total_pnl']:.2f}

STRATEGY BREAKDOWN:
"""
    for strat, count in stats['strategies'].items():
        summary += f"- {strat}: {count} signals\n"

    summary += f"""
MARKET CONTEXT:
- Current Regime: {regime}
- Average Signal Confidence: {stats['avg_confidence']:.1%}
- Symbols Active: {len(stats['symbols'])}

TOP SYMBOLS: {', '.join(stats['symbols'][:5]) if stats['symbols'] else 'None'}

{'=' * 40}
Generated by Kobe AI Trading System
"""
    return summary.strip()


def save_weekly_report(output_path: Optional[Path] = None) -> Path:
    """
    Generate and save the weekly report to a file.

    Args:
        output_path: Optional custom output path

    Returns:
        Path to the saved report
    """
    if output_path is None:
        date_str = datetime.now().strftime('%Y%m%d')
        output_path = LOGS_DIR / f'weekly_summary_{date_str}.txt'

    summary = generate_weekly_summary()

    with open(output_path, 'w') as f:
        f.write(summary)

    logger.info(f"Weekly summary saved to {output_path}")
    return output_path


if __name__ == '__main__':
    print(generate_weekly_summary(use_llm=False))
