#!/usr/bin/env python3
"""
Generate Game Briefings for Kobe Trading System
================================================

CLI for generating PRE_GAME, HALF_TIME, and POST_GAME briefings with
comprehensive LLM/ML/AI analysis.

Usage:
    # Generate morning briefing
    python scripts/generate_briefing.py --phase pregame

    # Generate midday status
    python scripts/generate_briefing.py --phase halftime

    # Generate end-of-day analysis
    python scripts/generate_briefing.py --phase postgame

    # With Telegram notification
    python scripts/generate_briefing.py --phase pregame --telegram

    # Specific date
    python scripts/generate_briefing.py --phase pregame --date 2025-12-27
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Generate briefing based on phase argument."""
    ap = argparse.ArgumentParser(
        description='Generate Kobe trading system briefings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  pregame   - Morning game plan (08:00 ET)
              Regime analysis, Top-3 picks, TOTD, action steps

  halftime  - Midday status check (12:00 ET)
              Position analysis, adjustments, remaining opportunities

  postgame  - End of day analysis (16:00 ET)
              Performance review, lessons, next day setup
"""
    )
    ap.add_argument(
        '--phase',
        choices=['pregame', 'halftime', 'postgame'],
        required=True,
        help='Briefing phase to generate'
    )
    ap.add_argument(
        '--universe',
        type=str,
        default='data/universe/optionable_liquid_900.csv',
        help='Universe file path'
    )
    ap.add_argument(
        '--cap',
        type=int,
        default=300,
        help='Max symbols to scan'
    )
    ap.add_argument(
        '--dotenv',
        type=str,
        default='./.env',
        help='Path to .env file'
    )
    ap.add_argument(
        '--date',
        type=str,
        default=None,
        help='Briefing date (YYYY-MM-DD); default: today'
    )
    ap.add_argument(
        '--telegram',
        action='store_true',
        help='Send summary to Telegram'
    )
    ap.add_argument(
        '--json-only',
        action='store_true',
        help='Output JSON to stdout instead of saving files'
    )
    ap.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load environment
    load_dotenv(args.dotenv)

    try:
        from cognitive.game_briefings import GameBriefingEngine
    except ImportError as e:
        logger.error(f"Could not import game_briefings module: {e}")
        return 1

    # Initialize engine
    engine = GameBriefingEngine(dotenv_path=args.dotenv)

    # Generate briefing based on phase
    phase = args.phase
    logger.info(f"Generating {phase.upper()} briefing...")

    try:
        if phase == 'pregame':
            briefing = engine.generate_pregame(
                universe=args.universe,
                cap=args.cap,
                date=args.date
            )
        elif phase == 'halftime':
            briefing = engine.generate_halftime()
        else:  # postgame
            briefing = engine.generate_postgame()

        # Output handling
        if args.json_only:
            print(json.dumps(briefing.to_dict(), indent=2, default=str))
        else:
            # Save to files
            json_path, md_path = engine.save_briefing(briefing, phase)
            logger.info(f"Saved JSON: {json_path}")
            logger.info(f"Saved Markdown: {md_path}")

            # Print summary to console
            print()
            print("=" * 60)
            print(f" {phase.upper()} BRIEFING - {briefing.context.date}")
            print("=" * 60)
            print()

            if phase == 'pregame':
                print(f"Regime: {briefing.context.regime} ({briefing.context.regime_confidence:.0%})")
                print(f"VIX: {briefing.context.vix_level:.1f}")
                print(f"Mood: {briefing.context.mood_state}")
                print(f"Trading Bias: {briefing.trading_bias}")
                print()

                if briefing.top3_picks:
                    print("TOP-3 PICKS:")
                    for i, pick in enumerate(briefing.top3_picks, 1):
                        print(f"  {i}. {pick.get('symbol')} ({pick.get('strategy')})")
                        print(f"     Entry: ${pick.get('entry_price', 0):.2f} | Stop: ${pick.get('stop_loss', 0):.2f}")
                    print()

                if briefing.totd:
                    print(f"TRADE OF THE DAY: {briefing.totd.get('symbol')}")
                    if briefing.totd_deep_analysis:
                        # Print first 500 chars of analysis
                        analysis = briefing.totd_deep_analysis[:500]
                        if len(briefing.totd_deep_analysis) > 500:
                            analysis += "..."
                        print(analysis)
                    print()

                if briefing.risk_warnings:
                    print("RISK WARNINGS:")
                    for w in briefing.risk_warnings[:3]:
                        print(f"  - {w}")
                    print()

            elif phase == 'halftime':
                print(f"Regime: {briefing.current_regime}")
                if briefing.regime_changed:
                    print(f"  ** REGIME CHANGED from {briefing.morning_regime}! **")
                print(f"VIX: {briefing.current_vix:.1f}")
                print(f"Open Positions: {len(briefing.position_analysis)}")
                print()

                if briefing.position_analysis:
                    print("POSITION STATUS:")
                    for p in briefing.position_analysis:
                        emoji = "🟢" if p.pnl_percent > 0 else "🔴"
                        print(f"  {emoji} {p.symbol}: {p.pnl_percent:+.1f}% | Rec: {p.recommendation}")
                    print()

                if briefing.whats_working:
                    print("WHAT'S WORKING:")
                    for w in briefing.whats_working[:2]:
                        print(f"  + {w[:80]}")
                    print()

                if briefing.whats_not_working:
                    print("WHAT'S NOT WORKING:")
                    for w in briefing.whats_not_working[:2]:
                        print(f"  - {w[:80]}")
                    print()

            elif phase == 'postgame':
                summary = briefing.day_summary
                print(f"Trades: {summary.get('total_trades', 0)}")
                print(f"W/L: {summary.get('wins', 0)}/{summary.get('losses', 0)}")
                print(f"Win Rate: {summary.get('win_rate', 0):.0%}")
                print(f"P&L: ${summary.get('realized_pnl', 0):,.2f}")
                print()

                if briefing.executive_summary:
                    print("EXECUTIVE SUMMARY:")
                    print(briefing.executive_summary[:300])
                    if len(briefing.executive_summary) > 300:
                        print("...")
                    print()

                if briefing.lessons_learned:
                    print("KEY LESSONS:")
                    for l in briefing.lessons_learned[:3]:
                        print(f"  - {l[:80]}")
                    print()

                if briefing.next_day_setup:
                    print("NEXT DAY SETUP:")
                    print(briefing.next_day_setup[:200])
                    if len(briefing.next_day_setup) > 200:
                        print("...")
                    print()

            print("=" * 60)
            print(f"Full report: {md_path}")
            print("=" * 60)

        # Send Telegram if requested
        if args.telegram:
            logger.info("Sending Telegram notification...")
            success = engine.send_telegram_summary(briefing, phase)
            if success:
                logger.info("Telegram notification sent")
            else:
                logger.warning("Telegram notification failed")

        logger.info(f"{phase.upper()} briefing generation complete")
        return 0

    except Exception as e:
        logger.error(f"Briefing generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
