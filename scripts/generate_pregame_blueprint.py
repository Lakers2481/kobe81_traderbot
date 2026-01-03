#!/usr/bin/env python3
"""
Pre-Game Blueprint Generator
=============================

Generates a comprehensive evidence-backed Pre-Game Blueprint for the trading day.

This is like a coach's game plan before a game - complete with:
- Top 5 → Top 3 → Top 2 selection with full reasoning
- Historical patterns (consecutive days, reversal rates)
- News headlines with sentiment scores
- Options expected move (remaining room in weekly range)
- Support/resistance levels with justification
- Entry/Stop/Target with WHY
- AI confidence breakdown
- Bull/bear cases
- What could go wrong

Usage:
    # Standard pre-game blueprint
    python scripts/generate_pregame_blueprint.py

    # With custom parameters
    python scripts/generate_pregame_blueprint.py --cap 500 --top 5 --execute 2

    # For specific positions (analyze existing holdings)
    python scripts/generate_pregame_blueprint.py --positions TSLA PLTR

    # JSON output only
    python scripts/generate_pregame_blueprint.py --format json

Output:
    reports/pregame_YYYYMMDD.json - Structured data
    reports/pregame_YYYYMMDD.md   - Human-readable markdown

Runs at 08:15 ET via scheduler (before market open).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def generate_pregame_blueprint(
    universe_path: str = None,
    cap: int = 900,
    top_n: int = 5,
    execute_n: int = 2,
    positions: List[str] = None,
    dotenv_path: str = "./.env",
    output_format: str = "both",
) -> Dict[str, Any]:
    """
    Generate comprehensive pre-game blueprint.

    Flow:
    1. If positions provided, analyze those directly
    2. Otherwise, run scanner to get Top N signals
    3. For each signal, build complete TradeThesis
    4. Filter Top N → Top (N-2) based on thesis quality
    5. Select Top 2 for execution
    6. Generate full blueprint with:
       - Market context (regime, VIX, sentiment)
       - Weekly budget status
       - All evidence for each trade
       - Full reasoning for selection/rejection

    Args:
        universe_path: Path to universe CSV
        cap: Max symbols to scan
        top_n: Size of watchlist (default 5)
        execute_n: Number to execute (default 2)
        positions: Specific symbols to analyze (overrides scanner)
        dotenv_path: Path to .env file
        output_format: "json", "markdown", or "both"

    Returns:
        Dict with complete blueprint data
    """
    load_dotenv(dotenv_path)

    # Late imports to avoid circular dependencies
    from analysis.historical_patterns import HistoricalPatternAnalyzer
    from analysis.options_expected_move import ExpectedMoveCalculator
    from explainability.trade_thesis_builder import TradeThesisBuilder, TradeThesis

    today = date.today()
    generated_at = datetime.now()

    logger.info(f"Generating Pre-Game Blueprint for {today}")
    divider = "=" * 60
    print(f"\n{divider}")
    print(f"PRE-GAME BLUEPRINT GENERATOR")
    print(f"Date: {today}")
    print(f"{divider}\n")

    # Initialize components
    pattern_analyzer = HistoricalPatternAnalyzer(lookback_years=5)
    em_calculator = ExpectedMoveCalculator()
    thesis_builder = TradeThesisBuilder(dotenv_path)

    # Initialize blueprint structure
    blueprint = {
        "metadata": {
            "generated_at": generated_at.isoformat(),
            "for_date": str(today),
            "generation_method": "deterministic",
        },
        "market_context": {},
        "budget_status": {},
        "positions": {},  # For existing positions analysis
        "selection_funnel": {},
        "trade_theses": [],
        "rejected_candidates": [],
        "execution_plan": {},
        "ai_briefing": {},
    }

    # Mode 1: Analyze specific positions
    if positions:
        logger.info(f"Analyzing existing positions: {positions}")
        blueprint["mode"] = "position_analysis"

        for symbol in positions:
            print(f"\nAnalyzing {symbol}...")

            try:
                # Get historical patterns
                pattern = pattern_analyzer.analyze_consecutive_days(symbol=symbol)
                em = em_calculator.calculate_weekly_expected_move(symbol)
                sr = pattern_analyzer.get_sector_relative_strength(symbol)
                vol = pattern_analyzer.analyze_volume_profile(symbol=symbol)
                levels = pattern_analyzer.analyze_support_resistance(symbol=symbol)

                blueprint["positions"][symbol] = {
                    "consecutive_pattern": pattern.to_dict(),
                    "expected_move": em.to_dict(),
                    "sector_relative_strength": sr.to_dict(),
                    "volume_profile": vol.to_dict(),
                    "support_resistance": [l.to_dict() for l in levels],
                }

                # Print summary
                print(f"  Pattern: {pattern.pattern_type} ({pattern.current_streak} days)")
                print(f"  Reversal Rate: {pattern.historical_reversal_rate:.0%} ({pattern.sample_size} samples)")
                print(f"  Confidence: {pattern.confidence}")
                print(f"  Week Move: {em.move_from_week_open_pct:+.1%}")
                print(f"  Room: {em.remaining_room_direction} ({em.remaining_room_up_pct:.1%} up)")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                blueprint["positions"][symbol] = {"error": str(e)}

        # Save outputs
        _save_blueprint(blueprint, today, output_format)
        return blueprint

    # Mode 2: Run scanner and build theses
    logger.info(f"Running scanner with cap={cap}, top_n={top_n}")
    blueprint["mode"] = "scanner"

    # Get signals from scanner
    signals = _get_scanner_signals(
        universe_path or str(ROOT / "data/universe/optionable_liquid_900.csv"),
        cap,
        top_n,
        dotenv_path,
    )

    if not signals:
        logger.warning("No signals found from scanner")
        blueprint["selection_funnel"] = {
            "scanned": cap,
            "signals_found": 0,
            "top_n_candidates": [],
            "message": "No signals found for today",
        }
        _save_blueprint(blueprint, today, output_format)
        return blueprint

    # Build theses for top N signals
    print(f"\nBuilding trade theses for {len(signals)} signals...")
    theses: List[TradeThesis] = []

    for signal in signals[:top_n]:
        symbol = signal.get("symbol", "")
        print(f"  Building thesis for {symbol}...")

        try:
            thesis = thesis_builder.build_thesis(signal)
            theses.append(thesis)
        except Exception as e:
            logger.error(f"Error building thesis for {symbol}: {e}")

    # Sort by confidence
    theses.sort(key=lambda t: t.ai_confidence, reverse=True)

    # Selection funnel
    top_n_symbols = [t.symbol for t in theses]
    top_3_theses = theses[:3] if len(theses) >= 3 else theses
    top_2_theses = theses[:execute_n] if len(theses) >= execute_n else theses

    # Determine rejected and reasons
    rejected = []
    for thesis in theses[execute_n:]:
        rejected.append({
            "symbol": thesis.symbol,
            "grade": thesis.trade_grade,
            "confidence": thesis.ai_confidence,
            "rejection_reason": _get_rejection_reason(thesis),
        })

    # Build selection funnel
    blueprint["selection_funnel"] = {
        "scanned": cap,
        "signals_found": len(signals),
        "top_n_candidates": top_n_symbols,
        "top_3_selected": [t.symbol for t in top_3_theses],
        "top_2_for_execution": [t.symbol for t in top_2_theses],
        "filter_reasoning": _generate_filter_reasoning(theses, top_n, 3, execute_n),
    }

    # Add theses
    blueprint["trade_theses"] = [
        {"rank": i + 1, "symbol": t.symbol, "thesis": t.to_dict()}
        for i, t in enumerate(top_2_theses)
    ]

    blueprint["rejected_candidates"] = rejected

    # Generate execution plan
    blueprint["execution_plan"] = _generate_execution_plan(top_2_theses)

    # Generate AI briefing
    blueprint["ai_briefing"] = _generate_ai_briefing(top_2_theses, theses)

    # Print summary
    _print_summary(blueprint)

    # Save outputs
    _save_blueprint(blueprint, today, output_format)

    return blueprint


def _get_scanner_signals(
    universe_path: str,
    cap: int,
    top_n: int,
    dotenv_path: str,
) -> List[Dict]:
    """Run scanner and get signals."""
    try:
        from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
        from data.providers.polygon_eod import fetch_daily_bars_polygon
        from data.universe.loader import load_universe
        from datetime import datetime, timedelta

        symbols = load_universe(universe_path, cap=cap)
        logger.info(f"Loaded {len(symbols)} symbols")

        scanner = DualStrategyScanner(DualStrategyParams())

        # Get date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        signals = []
        for i, symbol in enumerate(symbols):
            if i % 100 == 0:
                logger.info(f"Scanning progress: {i}/{len(symbols)}")

            try:
                df = fetch_daily_bars_polygon(symbol, start_date, end_date)
                if df is None or len(df) < 20:
                    continue

                result = scanner.generate_signals(df)
                if result is not None and len(result) > 0:
                    for _, row in result.iterrows():
                        signals.append({
                            "symbol": symbol,
                            "strategy": row.get("strategy", "dual"),
                            "side": row.get("side", "long"),
                            "entry_price": float(row.get("entry_price", df["close"].iloc[-1])),
                            "stop_loss": float(row.get("stop_loss", 0)),
                            "take_profit": float(row.get("take_profit", 0)),
                            "quality_score": float(row.get("quality_score", 50)),
                            "reason": row.get("reason", ""),
                        })
            except Exception as e:
                continue

        # Sort by quality score and return top N
        signals.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        return signals[:top_n]

    except Exception as e:
        logger.error(f"Error running scanner: {e}")
        return []


def _get_rejection_reason(thesis: TradeThesis) -> str:
    """Generate a rejection reason for a thesis."""
    reasons = []

    if thesis.ai_confidence < 60:
        reasons.append(f"Low confidence ({thesis.ai_confidence:.0f}%)")

    if thesis.risk_reward_ratio < 1.5:
        reasons.append(f"Poor R:R ({thesis.risk_reward_ratio:.2f}:1)")

    if thesis.aggregated_sentiment.get("compound", 0) < -0.2:
        reasons.append("Negative sentiment")

    if thesis.expected_move.get("remaining_room_direction") == "EXHAUSTED":
        reasons.append("Expected move exhausted")

    if thesis.sector_relative_strength.get("relative_strength", 0) < -0.05:
        reasons.append("Sector underperformance")

    if not reasons:
        reasons.append("Budget limit (best 2 selected)")

    return "; ".join(reasons)


def _generate_filter_reasoning(
    theses: List,
    top_n: int,
    top_3: int,
    execute_n: int,
) -> Dict[str, str]:
    """Generate reasoning for filter steps."""
    reasoning = {}

    if len(theses) > top_3:
        removed_5_to_3 = theses[top_3:top_n] if len(theses) >= top_n else theses[top_3:]
        reasons = [f"{t.symbol} ({_get_rejection_reason(t)})" for t in removed_5_to_3]
        reasoning["n_to_3"] = f"Removed: {', '.join(reasons)}"

    if len(theses) > execute_n:
        reasoning["3_to_2"] = "Selected highest confidence scores for execution"

    return reasoning


def _generate_execution_plan(theses: List) -> Dict[str, Any]:
    """Generate execution plan for selected trades."""
    plan = {
        "pre_market": [
            "08:15 - Review pre-game blueprint",
            "08:30 - Final news check for selected symbols",
            "09:00 - Verify broker connection and buying power",
        ],
        "market_open": [
            "09:30 - Observe opening range (no trades first 5 min)",
            "09:35 - Check confirmation signals",
        ],
        "execution": [],
        "position_sizing": {},
    }

    for i, thesis in enumerate(theses, 1):
        plan["execution"].append(
            f"09:{35 + i * 5} - Enter {thesis.symbol} if confirmation triggers"
        )
        plan["position_sizing"][thesis.symbol] = {
            "entry_price": thesis.entry_price,
            "stop_loss": thesis.stop_loss,
            "target": thesis.take_profit,
            "risk_pct": 0.02,  # 2% risk per trade
        }

    plan["monitoring"] = [
        "10:30 - Halftime position check",
        "12:00 - Midday review",
        "15:30 - EOD positioning",
    ]

    return plan


def _generate_ai_briefing(top_theses: List, all_theses: List) -> Dict[str, Any]:
    """Generate AI briefing summary."""
    if not top_theses:
        return {
            "executive_summary": "No qualified trades for today.",
            "confidence_level": "N/A",
            "key_risks": [],
        }

    # Build summary
    summaries = []
    for thesis in top_theses:
        pattern = thesis.consecutive_pattern
        if pattern.get("pattern_type") == "consecutive_down":
            summaries.append(
                f"{thesis.symbol}: {pattern.get('current_streak', 0)} days down, "
                f"{pattern.get('historical_reversal_rate', 0):.0%} reversal rate, "
                f"{thesis.trade_grade} grade"
            )
        else:
            summaries.append(
                f"{thesis.symbol}: {thesis.strategy} signal, {thesis.trade_grade} grade"
            )

    avg_confidence = sum(t.ai_confidence for t in top_theses) / len(top_theses)

    if avg_confidence >= 80:
        confidence_level = "HIGH"
    elif avg_confidence >= 70:
        confidence_level = "MEDIUM-HIGH"
    elif avg_confidence >= 60:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Aggregate risks
    all_risks = []
    for thesis in top_theses:
        all_risks.extend(thesis.what_could_go_wrong)

    # Deduplicate and limit
    unique_risks = list(dict.fromkeys(all_risks))[:5]

    return {
        "executive_summary": f"Today we have {len(top_theses)} trades selected. " + " | ".join(summaries),
        "confidence_level": confidence_level,
        "average_confidence": round(avg_confidence, 1),
        "key_risks": unique_risks,
        "total_candidates_analyzed": len(all_theses),
    }


def _print_summary(blueprint: Dict) -> None:
    """Print summary to console."""
    divider = "=" * 60

    print(f"\n{divider}")
    print("PRE-GAME BLUEPRINT SUMMARY")
    print(divider)

    funnel = blueprint.get("selection_funnel", {})
    print(f"\nScanned: {funnel.get('scanned', 0)} symbols")
    print(f"Signals Found: {funnel.get('signals_found', 0)}")
    print(f"Top Candidates: {', '.join(funnel.get('top_n_candidates', []))}")
    print(f"Selected for Execution: {', '.join(funnel.get('top_2_for_execution', []))}")

    print(f"\n{'-' * 40}")
    print("TRADE THESES")
    print("-" * 40)

    for trade in blueprint.get("trade_theses", []):
        thesis = trade.get("thesis", {})
        print(f"\n#{trade.get('rank')} {trade.get('symbol')}")
        print(f"  Grade: {thesis.get('trade_grade')} | Confidence: {thesis.get('ai_confidence'):.0f}%")
        print(f"  Entry: ${thesis.get('entry_price'):.2f} | Stop: ${thesis.get('stop_loss'):.2f} | Target: ${thesis.get('take_profit'):.2f}")
        print(f"  R:R: {thesis.get('risk_reward_ratio'):.2f}:1")

        pattern = thesis.get("consecutive_pattern", {})
        if pattern.get("pattern_type"):
            print(f"  Pattern: {pattern.get('current_streak')} days {pattern.get('pattern_type').replace('consecutive_', '')}")
            print(f"  Reversal Rate: {pattern.get('historical_reversal_rate', 0):.0%} ({pattern.get('sample_size', 0)} samples)")

    briefing = blueprint.get("ai_briefing", {})
    print(f"\n{'-' * 40}")
    print(f"AI CONFIDENCE: {briefing.get('confidence_level', 'N/A')}")
    print(f"Summary: {briefing.get('executive_summary', '')}")

    print(f"\n{divider}\n")


def _save_blueprint(blueprint: Dict, for_date: date, output_format: str) -> None:
    """Save blueprint to files."""
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    date_str = for_date.strftime("%Y%m%d")

    # Save JSON
    if output_format in ("json", "both"):
        json_path = reports_dir / f"pregame_{date_str}.json"
        with open(json_path, "w") as f:
            json.dump(blueprint, f, indent=2, default=str)
        logger.info(f"Saved JSON blueprint to {json_path}")

    # Save Markdown
    if output_format in ("markdown", "both"):
        md_path = reports_dir / f"pregame_{date_str}.md"
        md_content = _generate_markdown(blueprint)
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info(f"Saved Markdown blueprint to {md_path}")


def _generate_markdown(blueprint: Dict) -> str:
    """Generate markdown report from blueprint."""
    lines = []
    metadata = blueprint.get("metadata", {})
    for_date = metadata.get("for_date", "Unknown")

    lines.append(f"# PRE-GAME BLUEPRINT - {for_date}")
    lines.append(f"*Generated: {metadata.get('generated_at', '')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Mode: Position Analysis
    if blueprint.get("mode") == "position_analysis":
        lines.append("## POSITION ANALYSIS")
        lines.append("")

        for symbol, data in blueprint.get("positions", {}).items():
            if "error" in data:
                lines.append(f"### {symbol}")
                lines.append(f"Error: {data['error']}")
                continue

            lines.append(f"### {symbol}")
            lines.append("")

            pattern = data.get("consecutive_pattern", {})
            if pattern:
                lines.append("**Consecutive Pattern:**")
                lines.append(f"- Type: {pattern.get('pattern_type', 'N/A')}")
                lines.append(f"- Streak: {pattern.get('current_streak', 0)} days")
                lines.append(f"- Reversal Rate: {pattern.get('historical_reversal_rate', 0):.0%}")
                lines.append(f"- Sample Size: {pattern.get('sample_size', 0)} instances")
                lines.append(f"- Confidence: **{pattern.get('confidence', 'N/A')}**")
                lines.append("")

                # NEW: Bounce metrics
                lines.append("**Expected Bounce (from historical data):**")
                lines.append(f"- Day 1 Bounce: **{pattern.get('day1_bounce_avg', 0):+.2%}** (min: {pattern.get('day1_bounce_min', 0):+.2%}, max: {pattern.get('day1_bounce_max', 0):+.2%})")
                lines.append(f"- Avg Hold Time: **{pattern.get('avg_bounce_days', 1):.1f} days**")
                lines.append(f"- Total Bounce: **{pattern.get('total_bounce_avg', 0):+.2%}**")
                lines.append("")

                # NEW: Historical instances table
                instances = pattern.get('historical_instances', [])
                if instances:
                    lines.append("**Historical Instances (Verify on Yahoo Finance):**")
                    lines.append("| # | Date | Streak | Day 1 | Days | Total | Drop |")
                    lines.append("|---|------|--------|-------|------|-------|------|")
                    for idx, inst in enumerate(instances, 1):
                        lines.append(
                            f"| {idx} | {inst.get('end_date', 'N/A')} | {inst.get('streak_length', 0)} | "
                            f"{inst.get('day1_return', 0):+.2%} | {inst.get('bounce_days', 0)} | "
                            f"{inst.get('total_bounce', 0):+.2%} | {inst.get('drop_pct', 0):+.1%} |"
                        )
                    lines.append("")

                lines.append(f"- Evidence: {pattern.get('evidence', '')}")
                lines.append("")

            em = data.get("expected_move", {})
            if em:
                lines.append("**Expected Move:**")
                lines.append(f"- Weekly EM: +/-{em.get('weekly_expected_move_pct', 0):.1%} (${em.get('weekly_expected_move_dollars', 0):.2f})")
                lines.append(f"- Week Open: ${em.get('week_open_price', 0):.2f}")
                lines.append(f"- Current: ${em.get('current_price', 0):.2f}")
                lines.append(f"- Move from Open: {em.get('move_from_week_open_pct', 0):+.1%}")
                lines.append(f"- Remaining Room: {em.get('remaining_room_direction', 'N/A')} ({em.get('remaining_room_up_pct', 0):.1%} up)")
                lines.append("")

            # NEW: R:R Gate Analysis
            if pattern and em:
                current_price = em.get('current_price', 0)
                day1_bounce = pattern.get('day1_bounce_avg', 0.02)
                total_bounce = pattern.get('total_bounce_avg', 0.05)

                # Find nearest support for stop
                support_levels = [l for l in data.get("support_resistance", []) if l.get('level_type') == 'support']
                if support_levels:
                    nearest_support = min(support_levels, key=lambda x: abs(x.get('distance_pct', 100)))
                    stop_price = nearest_support.get('price', current_price * 0.97)
                else:
                    stop_price = current_price * 0.97  # 3% default stop

                # Calculate R:R for Day 1 target and Total bounce target
                risk = current_price - stop_price
                day1_target = current_price * (1 + day1_bounce)
                total_target = current_price * (1 + total_bounce)
                day1_reward = day1_target - current_price
                total_reward = total_target - current_price

                day1_rr = day1_reward / risk if risk > 0 else 0
                total_rr = total_reward / risk if risk > 0 else 0

                # Required target for 2.25:1 R:R
                required_reward = risk * 2.25
                required_target = current_price + required_reward
                required_bounce_pct = required_reward / current_price

                lines.append("**Risk:Reward Analysis (2.25:1 minimum required):**")
                lines.append(f"- Entry: ${current_price:.2f}")
                lines.append(f"- Stop: ${stop_price:.2f} (nearest support)")
                lines.append(f"- Risk: ${risk:.2f} ({risk/current_price:.1%})")
                lines.append("")
                lines.append(f"- Day 1 Target: ${day1_target:.2f} ({day1_bounce:+.2%})")
                lines.append(f"  - R:R: {day1_rr:.2f}:1 {'**GO**' if day1_rr >= 2.25 else '**NO-GO** (< 2.25)'}")
                lines.append("")
                lines.append(f"- Total Bounce Target: ${total_target:.2f} ({total_bounce:+.2%})")
                lines.append(f"  - R:R: {total_rr:.2f}:1 {'**GO**' if total_rr >= 2.25 else '**NO-GO** (< 2.25)'}")
                lines.append("")
                lines.append(f"- Required for 2.25:1: ${required_target:.2f} ({required_bounce_pct:+.1%})")
                avg_days = pattern.get('avg_bounce_days', 2)
                if total_bounce >= required_bounce_pct:
                    lines.append(f"- Achievable? **YES** (hold {avg_days:.0f} days)")
                else:
                    lines.append("- Achievable? **NO** - consider wider stop or lower entry")
                lines.append("")

            sr = data.get("sector_relative_strength", {})
            if sr:
                lines.append("**Sector Relative Strength:**")
                lines.append(f"- Sector ETF: {sr.get('sector_etf', 'SPY')}")
                lines.append(f"- Relative Strength: {sr.get('relative_strength', 0):+.1%}")
                lines.append(f"- Beta vs Sector: {sr.get('beta_vs_sector', 1):.2f}")
                lines.append(f"- Interpretation: {sr.get('interpretation', '')}")
                lines.append("")

            levels = data.get("support_resistance", [])
            if levels:
                lines.append("**Support/Resistance Levels:**")
                lines.append("| Price | Type | Strength | Distance | Justification |")
                lines.append("|-------|------|----------|----------|---------------|")
                for level in levels[:6]:
                    lines.append(
                        f"| ${level.get('price', 0):.2f} | {level.get('level_type', '')} | "
                        f"{level.get('strength', 0)} | {level.get('distance_pct', 0):.1f}% | "
                        f"{level.get('justification', '')} |"
                    )
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    # Mode: Scanner
    briefing = blueprint.get("ai_briefing", {})
    lines.append("## EXECUTIVE SUMMARY")
    lines.append("")
    lines.append(f"**Confidence Level:** {briefing.get('confidence_level', 'N/A')}")
    lines.append("")
    lines.append(briefing.get("executive_summary", ""))
    lines.append("")

    # Selection Funnel
    funnel = blueprint.get("selection_funnel", {})
    lines.append("---")
    lines.append("")
    lines.append("## SELECTION FUNNEL")
    lines.append("")
    lines.append(f"- **Scanned:** {funnel.get('scanned', 0)} symbols")
    lines.append(f"- **Signals Found:** {funnel.get('signals_found', 0)}")
    lines.append(f"- **Top Candidates:** {', '.join(funnel.get('top_n_candidates', []))}")
    lines.append(f"- **Selected:** {', '.join(funnel.get('top_2_for_execution', []))}")
    lines.append("")

    reasoning = funnel.get("filter_reasoning", {})
    if reasoning:
        lines.append("**Filter Reasoning:**")
        for step, reason in reasoning.items():
            lines.append(f"- {step}: {reason}")
        lines.append("")

    # Trade Theses
    lines.append("---")
    lines.append("")
    lines.append("## TRADE THESES")
    lines.append("")

    for trade in blueprint.get("trade_theses", []):
        thesis = trade.get("thesis", {})
        lines.append(f"### #{trade.get('rank')}: {trade.get('symbol')}")
        lines.append("")
        lines.append(f"**Grade:** {thesis.get('trade_grade')} | **Confidence:** {thesis.get('ai_confidence'):.0f}% | **Recommendation:** {thesis.get('recommendation')}")
        lines.append("")

        lines.append("**Trade Parameters:**")
        lines.append(f"- Entry: ${thesis.get('entry_price'):.2f}")
        lines.append(f"- Stop: ${thesis.get('stop_loss'):.2f}")
        lines.append(f"- Target: ${thesis.get('take_profit'):.2f}")
        lines.append(f"- R:R: {thesis.get('risk_reward_ratio'):.2f}:1")
        lines.append("")

        lines.append("**Executive Summary:**")
        lines.append(thesis.get("executive_summary", ""))
        lines.append("")

        lines.append("**Entry Justification:**")
        lines.append(thesis.get("entry_justification", ""))
        lines.append("")

        lines.append("**Stop Justification:**")
        lines.append(thesis.get("stop_justification", ""))
        lines.append("")

        lines.append("**Target Justification:**")
        lines.append(thesis.get("target_justification", ""))
        lines.append("")

        lines.append("**Bull Case:**")
        lines.append(thesis.get("bull_case", ""))
        lines.append("")

        lines.append("**Bear Case:**")
        lines.append(thesis.get("bear_case", ""))
        lines.append("")

        risks = thesis.get("what_could_go_wrong", [])
        if risks:
            lines.append("**What Could Go Wrong:**")
            for risk in risks:
                lines.append(f"- {risk}")
            lines.append("")

        breakdown = thesis.get("ai_confidence_breakdown", {})
        if breakdown:
            lines.append("**AI Confidence Breakdown:**")
            lines.append("| Factor | Score |")
            lines.append("|--------|-------|")
            for factor, score in breakdown.items():
                lines.append(f"| {factor.title()} | {score}% |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Rejected Candidates
    rejected = blueprint.get("rejected_candidates", [])
    if rejected:
        lines.append("## REJECTED CANDIDATES")
        lines.append("")
        lines.append("| Symbol | Grade | Confidence | Reason |")
        lines.append("|--------|-------|------------|--------|")
        for r in rejected:
            lines.append(
                f"| {r.get('symbol')} | {r.get('grade')} | "
                f"{r.get('confidence'):.0f}% | {r.get('rejection_reason')} |"
            )
        lines.append("")

    # Execution Plan
    plan = blueprint.get("execution_plan", {})
    if plan:
        lines.append("---")
        lines.append("")
        lines.append("## EXECUTION PLAN")
        lines.append("")

        for phase, steps in plan.items():
            if isinstance(steps, list):
                lines.append(f"**{phase.replace('_', ' ').title()}:**")
                for step in steps:
                    lines.append(f"- {step}")
                lines.append("")

    # Key Risks
    risks = briefing.get("key_risks", [])
    if risks:
        lines.append("---")
        lines.append("")
        lines.append("## KEY RISKS")
        lines.append("")
        for risk in risks:
            lines.append(f"- {risk}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This blueprint was generated using deterministic analysis + historical data.*")
    lines.append("*All trades require manual confirmation before execution.*")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Pre-Game Blueprint for trading day",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=str(ROOT / "data/universe/optionable_liquid_900.csv"),
        help="Path to universe CSV",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=900,
        help="Max symbols to scan (default: 900)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Watchlist size (default: 5)",
    )
    parser.add_argument(
        "--execute",
        type=int,
        default=2,
        help="Number to execute (default: 2)",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        help="Specific symbols to analyze (for existing positions)",
    )
    parser.add_argument(
        "--dotenv",
        type=str,
        default="./.env",
        help="Path to .env file",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    try:
        blueprint = generate_pregame_blueprint(
            universe_path=args.universe,
            cap=args.cap,
            top_n=args.top,
            execute_n=args.execute,
            positions=args.positions,
            dotenv_path=args.dotenv,
            output_format=args.format,
        )

        print("Pre-Game Blueprint generated successfully!")

        # Return success
        return 0

    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 1
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
