"""
Profile Generator Module for Bounce Analysis

Generates PLTR-style markdown reports for each ticker with:
- Results summary table by streak level
- Day distribution charts
- Bounce magnitude & pain metrics
- Recent events list
- Sample quality notes
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bounce.bounce_score import calculate_bounce_score, get_bounce_score_breakdown


def generate_ticker_profile(
    ticker: str,
    per_stock_df: pd.DataFrame,
    events_df: pd.DataFrame,
    years: int,
    output_dir: Path,
    data_health: Optional[Dict] = None,
) -> Path:
    """
    Generate PLTR-style markdown report for a single ticker.

    Sections:
    1. Header (ticker, window, source, coverage)
    2. Results Summary table (streak 1-7)
    3. Day distribution (for best streak)
    4. Bounce magnitude & pain metrics
    5. Last 10 events for streak 5+
    6. Sample quality notes
    7. BounceScore breakdown

    Args:
        ticker: Stock ticker
        per_stock_df: Per-stock summary DataFrame
        events_df: Events DataFrame
        years: Window years (10 or 5)
        output_dir: Output directory for profiles
        data_health: Optional data health info

    Returns:
        Path to generated file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data for this ticker
    ticker_summary = per_stock_df[per_stock_df['ticker'] == ticker].copy()
    ticker_events = events_df[events_df['ticker'] == ticker].copy() if events_df is not None else pd.DataFrame()

    # Sort by streak_n
    ticker_summary = ticker_summary.sort_values('streak_n')

    # Build markdown content
    lines = []

    # Header
    lines.append(f"# {ticker} Bounce Profile ({years}Y)")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Window:** {years} years")
    lines.append("**Recovery:** 7 trading days")
    lines.append("")

    # Data quality
    if len(ticker_summary) > 0:
        quality_flags = ticker_summary['sample_quality_flag'].unique()
        lines.append(f"**Data Quality:** {', '.join(quality_flags)}")

        # Source info from events
        if len(ticker_events) > 0 and 'source_used' in ticker_events.columns:
            sources = ticker_events['source_used'].dropna().unique()
            lines.append(f"**Data Source:** {', '.join(sources)}")
    lines.append("")

    # Results Summary Table
    lines.append("## Results Summary by Streak Level")
    lines.append("")
    lines.append("| Streak | Events | Recovery Rate | Avg Days | Avg Return | Avg Drawdown | BounceScore |")
    lines.append("|--------|--------|---------------|----------|------------|--------------|-------------|")

    best_streak = None
    best_bounce_score = 0

    for _, row in ticker_summary.iterrows():
        streak_n = int(row['streak_n'])
        events = int(row.get('events', 0))

        if events == 0:
            lines.append(f"| {streak_n} | 0 | - | - | - | - | - |")
            continue

        recovery = row.get('recovery_7d_close_rate')
        avg_days = row.get('avg_days_to_recover_7d')
        avg_return = row.get('avg_best_7d_return')
        avg_drawdown = row.get('avg_max_drawdown_7d_pct')

        # Calculate BounceScore
        # Convert percentage to decimal if needed
        avg_return_dec = avg_return / 100 if avg_return and abs(avg_return) > 1 else avg_return
        avg_dd_dec = avg_drawdown / 100 if avg_drawdown and abs(avg_drawdown) > 1 else avg_drawdown

        bounce_score = calculate_bounce_score(
            recovery_rate=recovery if pd.notna(recovery) else 0,
            avg_days=avg_days if pd.notna(avg_days) else 7,
            avg_return=avg_return_dec if pd.notna(avg_return_dec) else 0,
            events=events,
            avg_drawdown=avg_dd_dec if pd.notna(avg_dd_dec) else 0,
        )

        if bounce_score > best_bounce_score and events >= 10:
            best_bounce_score = bounce_score
            best_streak = streak_n

        # Format values
        recovery_str = f"{recovery:.1%}" if pd.notna(recovery) else "-"
        days_str = f"{avg_days:.1f}" if pd.notna(avg_days) else "-"
        return_str = f"{avg_return:.1f}%" if pd.notna(avg_return) else "-"
        dd_str = f"{avg_drawdown:.1f}%" if pd.notna(avg_drawdown) else "-"
        score_str = f"{bounce_score:.0f}"

        lines.append(f"| {streak_n} | {events} | {recovery_str} | {days_str} | {return_str} | {dd_str} | {score_str} |")

    lines.append("")

    # Best streak highlight
    if best_streak is not None:
        lines.append(f"**Best Streak Level:** {best_streak} (BounceScore: {best_bounce_score:.0f})")
        lines.append("")

    # Day Distribution for best streak (or streak 5)
    focus_streak = best_streak if best_streak else 5
    streak_events = ticker_events[ticker_events['streak_n'] == focus_streak] if len(ticker_events) > 0 else pd.DataFrame()

    if len(streak_events) > 0 and 'days_to_recover_close' in streak_events.columns:
        lines.append(f"## Recovery Day Distribution (Streak {focus_streak})")
        lines.append("")

        # Calculate day distribution
        recovered = streak_events[streak_events['recovered_7d_close']]
        total = len(streak_events)

        day_counts = recovered['days_to_recover_close'].value_counts().sort_index()

        lines.append("| Day | Count | % of Total | Cumulative |")
        lines.append("|-----|-------|------------|------------|")

        cumulative = 0
        for day in range(1, 8):
            count = day_counts.get(day, 0)
            pct = count / total * 100 if total > 0 else 0
            cumulative += pct
            lines.append(f"| Day {day} | {count} | {pct:.1f}% | {cumulative:.1f}% |")

        not_recovered = total - len(recovered)
        not_recovered_pct = not_recovered / total * 100 if total > 0 else 0
        lines.append(f"| **Not Recovered** | {not_recovered} | {not_recovered_pct:.1f}% | - |")
        lines.append("")

    # Bounce Magnitude & Pain Metrics
    lines.append("## Bounce Magnitude & Pain Metrics")
    lines.append("")

    if len(ticker_summary) > 0:
        # Get stats for focus streak
        focus_row = ticker_summary[ticker_summary['streak_n'] == focus_streak]
        if len(focus_row) > 0:
            focus_row = focus_row.iloc[0]

            lines.append(f"### Streak {focus_streak} Deep Dive")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            avg_return = focus_row.get('avg_best_7d_return')
            median_return = focus_row.get('median_best_7d_return')
            p95_return = focus_row.get('p95_best_7d_return')
            avg_dd = focus_row.get('avg_max_drawdown_7d_pct')
            median_dd = focus_row.get('median_max_drawdown_7d_pct')

            lines.append(f"| Avg Best 7D Return | {avg_return:.2f}% |" if pd.notna(avg_return) else "| Avg Best 7D Return | - |")
            lines.append(f"| Median Best 7D Return | {median_return:.2f}% |" if pd.notna(median_return) else "| Median Best 7D Return | - |")
            lines.append(f"| P95 Best 7D Return | {p95_return:.2f}% |" if pd.notna(p95_return) else "| P95 Best 7D Return | - |")
            lines.append(f"| Avg Max Drawdown | {avg_dd:.2f}% |" if pd.notna(avg_dd) else "| Avg Max Drawdown | - |")
            lines.append(f"| Median Max Drawdown | {median_dd:.2f}% |" if pd.notna(median_dd) else "| Median Max Drawdown | - |")
            lines.append("")

    # Recent Events (last 10 for streak >= 5)
    high_streak_events = ticker_events[ticker_events['streak_n'] >= 5] if len(ticker_events) > 0 else pd.DataFrame()

    if len(high_streak_events) > 0:
        lines.append("## Recent Events (Streak ≥ 5)")
        lines.append("")

        # Sort by date descending, take last 10
        high_streak_events = high_streak_events.sort_values('event_date', ascending=False).head(10)

        lines.append("| Date | Streak | Close | 7D Return | Recovered | Days |")
        lines.append("|------|--------|-------|-----------|-----------|------|")

        for _, event in high_streak_events.iterrows():
            date_str = str(event['event_date'])[:10]
            streak = int(event['streak_n'])
            close = event.get('event_close', 0)
            return_7d = event.get('forward_7d_best_close_return_pct')
            recovered = event.get('recovered_7d_close', False)
            days = event.get('days_to_recover_close')

            close_str = f"${close:.2f}" if pd.notna(close) else "-"
            return_str = f"{return_7d:.1f}%" if pd.notna(return_7d) else "-"
            recovered_str = "✓" if recovered else "✗"
            days_str = str(int(days)) if pd.notna(days) else "-"

            lines.append(f"| {date_str} | {streak} | {close_str} | {return_str} | {recovered_str} | {days_str} |")

        lines.append("")

    # BounceScore Breakdown for focus streak
    if len(ticker_summary) > 0:
        focus_row = ticker_summary[ticker_summary['streak_n'] == focus_streak]
        if len(focus_row) > 0 and focus_row.iloc[0].get('events', 0) > 0:
            focus_row = focus_row.iloc[0]

            recovery = focus_row.get('recovery_7d_close_rate', 0)
            avg_days = focus_row.get('avg_days_to_recover_7d', 7)
            avg_return = focus_row.get('avg_best_7d_return', 0)
            events = focus_row.get('events', 0)
            avg_drawdown = focus_row.get('avg_max_drawdown_7d_pct', 0)

            # Convert if needed
            avg_return_dec = avg_return / 100 if avg_return and abs(avg_return) > 1 else avg_return
            avg_dd_dec = avg_drawdown / 100 if avg_drawdown and abs(avg_drawdown) > 1 else avg_drawdown

            breakdown = get_bounce_score_breakdown(
                recovery_rate=recovery if pd.notna(recovery) else 0,
                avg_days=avg_days if pd.notna(avg_days) else 7,
                avg_return=avg_return_dec if pd.notna(avg_return_dec) else 0,
                events=events,
                avg_drawdown=avg_dd_dec if pd.notna(avg_dd_dec) else 0,
            )

            lines.append(f"## BounceScore Breakdown (Streak {focus_streak})")
            lines.append("")
            lines.append("| Component | Score | Max | % of Max |")
            lines.append("|-----------|-------|-----|----------|")
            lines.append(f"| Recovery Rate | {breakdown['recovery_component']:.1f} | 40 | {breakdown['recovery_component']/40*100:.0f}% |")
            lines.append(f"| Speed | {breakdown['speed_component']:.1f} | 20 | {breakdown['speed_component']/20*100:.0f}% |")
            lines.append(f"| Opportunity | {breakdown['opportunity_component']:.1f} | 20 | {breakdown['opportunity_component']/20*100:.0f}% |")
            lines.append(f"| Sample Size | {breakdown['sample_component']:.1f} | 10 | {breakdown['sample_component']/10*100:.0f}% |")
            lines.append(f"| Pain Tolerance | {breakdown['pain_component']:.1f} | 10 | {breakdown['pain_component']/10*100:.0f}% |")
            lines.append(f"| **TOTAL** | **{breakdown['total_score']:.0f}** | 100 | {breakdown['total_score']:.0f}% |")
            lines.append("")

    # Sample Quality Notes
    lines.append("## Notes")
    lines.append("")

    quality_issues = []
    for _, row in ticker_summary.iterrows():
        streak_n = int(row['streak_n'])
        events = row.get('events', 0)
        quality = row.get('sample_quality_flag', 'UNKNOWN')

        if quality == 'NO_EVENTS':
            quality_issues.append(f"- Streak {streak_n}: No events found")
        elif quality == 'LOW_SAMPLE':
            quality_issues.append(f"- Streak {streak_n}: Low sample ({events} events)")
        elif quality == 'INSUFFICIENT_HISTORY':
            quality_issues.append(f"- Streak {streak_n}: Insufficient history")

    if quality_issues:
        lines.append("**Sample Quality Issues:**")
        lines.extend(quality_issues)
    else:
        lines.append("**Sample Quality:** All streak levels have adequate samples.")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by Kobe Bounce Analysis System*")

    # Write file
    output_file = output_dir / f"{ticker}_bounce_{years}y.md"
    output_file.write_text("\n".join(lines), encoding='utf-8')

    return output_file


def generate_all_profiles(
    per_stock_df: pd.DataFrame,
    events_df: pd.DataFrame,
    years: int,
    output_dir: Path,
    max_workers: int = 8,
    verbose: bool = True,
) -> List[Path]:
    """
    Generate profiles for ALL tickers in parallel.
    Also generates index.md with links to all profiles.

    Args:
        per_stock_df: Per-stock summary DataFrame
        events_df: Events DataFrame
        years: Window years (10 or 5)
        output_dir: Output directory
        max_workers: Parallel workers
        verbose: Print progress

    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = per_stock_df['ticker'].unique().tolist()

    if verbose:
        print(f"Generating {len(tickers)} ticker profiles ({years}Y)...")

    generated_files = []

    def generate_single(ticker: str) -> Path:
        return generate_ticker_profile(
            ticker=ticker,
            per_stock_df=per_stock_df,
            events_df=events_df,
            years=years,
            output_dir=output_dir,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single, t): t for t in tickers}

        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                path = future.result()
                generated_files.append(path)
            except Exception as e:
                if verbose:
                    print(f"  Error generating {ticker}: {e}")

            completed += 1
            if verbose and completed % 100 == 0:
                print(f"  Generated {completed}/{len(tickers)} profiles...")

    if verbose:
        print(f"  Total: {len(generated_files)} profiles generated")

    # Generate index.md
    index_path = generate_profile_index(
        tickers=tickers,
        per_stock_df=per_stock_df,
        years=years,
        output_dir=output_dir,
    )

    generated_files.append(index_path)

    return generated_files


def generate_profile_index(
    tickers: List[str],
    per_stock_df: pd.DataFrame,
    years: int,
    output_dir: Path,
) -> Path:
    """
    Generate index.md with links to all profiles.

    Args:
        tickers: List of tickers
        per_stock_df: Per-stock summary DataFrame
        years: Window years
        output_dir: Output directory

    Returns:
        Path to index.md
    """
    output_dir = Path(output_dir)

    lines = []
    lines.append(f"# Bounce Profile Index ({years}Y)")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Tickers:** {len(tickers)}")
    lines.append("")

    # Group by first letter
    lines.append("## Quick Navigation")
    lines.append("")

    letters = sorted(set(t[0].upper() for t in tickers if t))
    nav_links = " | ".join([f"[{l}](#{l.lower()})" for l in letters])
    lines.append(nav_links)
    lines.append("")

    # Build table with best BounceScore for each ticker
    ticker_scores = []

    for ticker in tickers:
        ticker_rows = per_stock_df[per_stock_df['ticker'] == ticker]

        best_score = 0
        best_streak = None

        for _, row in ticker_rows.iterrows():
            events = row.get('events', 0)
            if events < 5:
                continue

            recovery = row.get('recovery_7d_close_rate', 0)
            avg_days = row.get('avg_days_to_recover_7d', 7)
            avg_return = row.get('avg_best_7d_return', 0)
            avg_dd = row.get('avg_max_drawdown_7d_pct', 0)

            # Convert
            avg_return_dec = avg_return / 100 if avg_return and abs(avg_return) > 1 else avg_return
            avg_dd_dec = avg_dd / 100 if avg_dd and abs(avg_dd) > 1 else avg_dd

            score = calculate_bounce_score(
                recovery_rate=recovery if pd.notna(recovery) else 0,
                avg_days=avg_days if pd.notna(avg_days) else 7,
                avg_return=avg_return_dec if pd.notna(avg_return_dec) else 0,
                events=events,
                avg_drawdown=avg_dd_dec if pd.notna(avg_dd_dec) else 0,
            )

            if score > best_score:
                best_score = score
                best_streak = int(row['streak_n'])

        ticker_scores.append({
            "ticker": ticker,
            "best_score": best_score,
            "best_streak": best_streak,
        })

    # Sort by score descending for top performers
    ticker_scores_sorted = sorted(ticker_scores, key=lambda x: x['best_score'], reverse=True)

    # Top 20 performers
    lines.append("## Top 20 Bounce Performers")
    lines.append("")
    lines.append("| Rank | Ticker | Best BounceScore | Best Streak | Profile |")
    lines.append("|------|--------|------------------|-------------|---------|")

    for i, item in enumerate(ticker_scores_sorted[:20], 1):
        ticker = item['ticker']
        score = item['best_score']
        streak = item['best_streak']
        streak_str = str(streak) if streak else "-"
        lines.append(f"| {i} | {ticker} | {score:.0f} | {streak_str} | [{ticker}]({ticker}_bounce_{years}y.md) |")

    lines.append("")

    # Alphabetical index
    lines.append("## All Tickers (Alphabetical)")
    lines.append("")

    current_letter = None
    for ticker in sorted(tickers):
        first_letter = ticker[0].upper()

        if first_letter != current_letter:
            current_letter = first_letter
            lines.append(f"### {current_letter}")
            lines.append("")

        # Find score for this ticker
        ticker_data = next((t for t in ticker_scores if t['ticker'] == ticker), None)
        score = ticker_data['best_score'] if ticker_data else 0

        score_badge = ""
        if score >= 80:
            score_badge = " 🟢"
        elif score >= 60:
            score_badge = " 🟡"
        elif score >= 40:
            score_badge = " 🟠"
        else:
            score_badge = " ⚪"

        lines.append(f"- [{ticker}]({ticker}_bounce_{years}y.md){score_badge} (Score: {score:.0f})")

    lines.append("")
    lines.append("---")
    lines.append("*Legend: 🟢 Score ≥80 | 🟡 Score ≥60 | 🟠 Score ≥40 | ⚪ Score <40*")

    # Write file
    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding='utf-8')

    return index_path


def generate_summary_report(
    overall_df: pd.DataFrame,
    per_stock_df: pd.DataFrame,
    quality_report: pd.DataFrame,
    years: int,
    output_path: Path,
) -> Path:
    """
    Generate main summary markdown report.

    Args:
        overall_df: Overall summary DataFrame
        per_stock_df: Per-stock summary DataFrame
        quality_report: Data quality report
        years: Window years
        output_path: Output file path

    Returns:
        Path to generated file
    """
    lines = []

    # Header
    lines.append(f"# Week Down Then Bounce Summary ({years}Y)")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Window:** {years} years of historical data")
    lines.append("**Recovery:** 7 trading day forward window")
    lines.append("")

    # Data Health Section
    if quality_report is not None and len(quality_report) > 0:
        total = len(quality_report)
        loaded = quality_report['bars_count'].gt(0).sum() if 'bars_count' in quality_report.columns else 0
        failed = total - loaded

        # Source breakdown
        polygon_only = 0
        fallback_used = 0

        if 'source_used' in quality_report.columns:
            polygon_only = (quality_report['source_used'] == 'polygon').sum()
            fallback_used = quality_report['source_used'].isin(['yfinance', 'stooq']).sum()

        # Validation
        validated = quality_report['validated'].sum() if 'validated' in quality_report.columns else 0
        flagged = quality_report['mismatch_flagged'].sum() if 'mismatch_flagged' in quality_report.columns else 0
        rejected = quality_report['rejected'].sum() if 'rejected' in quality_report.columns else 0

        # Coverage
        if 'bars_count' in quality_report.columns:
            bars = quality_report['bars_count']
            years_coverage = bars / 252
            min_years = years_coverage.min()
            median_years = years_coverage.median()
            max_years = years_coverage.max()
        else:
            min_years = median_years = max_years = 0

        lines.append("## DATA HEALTH")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Tickers Attempted | {total} |")
        lines.append(f"| Tickers Loaded | {loaded} |")
        lines.append(f"| Tickers Failed | {failed} |")
        lines.append(f"| Polygon Only | {polygon_only} |")
        lines.append(f"| Fallback Used | {fallback_used} |")
        lines.append(f"| Validated Sample | {validated} |")
        lines.append(f"| Mismatch Flagged | {flagged} |")
        lines.append(f"| Rejected (Bad Data) | {rejected} |")
        lines.append(f"| Coverage (Years) | {min_years:.1f} / {median_years:.1f} / {max_years:.1f} (min/med/max) |")
        lines.append("")

    # Overall Results
    lines.append("## OVERALL RESULTS BY STREAK LEVEL")
    lines.append("")

    if overall_df is not None and len(overall_df) > 0:
        lines.append("| Streak | Events | Recovery Rate | Avg Days | Day 1 | Day 2 | Day 3 | Not Recovered |")
        lines.append("|--------|--------|---------------|----------|-------|-------|-------|---------------|")

        for _, row in overall_df.iterrows():
            streak = int(row['streak_n'])
            events = int(row.get('valid_events', row.get('events', 0)))
            recovery = row.get('recovery_7d_close_rate')
            avg_days = row.get('avg_days_to_recover_7d')
            day1 = row.get('day1_pct', 0)
            day2 = row.get('day2_pct', 0)
            day3 = row.get('day3_pct', 0)
            not_rec = row.get('not_recovered_pct', 0)

            recovery_str = f"{recovery:.1%}" if pd.notna(recovery) else "-"
            days_str = f"{avg_days:.1f}" if pd.notna(avg_days) else "-"

            lines.append(f"| {streak} | {events:,} | {recovery_str} | {days_str} | {day1:.1f}% | {day2:.1f}% | {day3:.1f}% | {not_rec:.1f}% |")

        lines.append("")

    # Bounce Magnitude
    lines.append("## BOUNCE MAGNITUDE")
    lines.append("")

    if overall_df is not None and len(overall_df) > 0:
        lines.append("| Streak | Avg Return | Median Return | P95 Return | Avg Drawdown | Median Drawdown |")
        lines.append("|--------|------------|---------------|------------|--------------|-----------------|")

        for _, row in overall_df.iterrows():
            streak = int(row['streak_n'])
            avg_ret = row.get('avg_best_7d_return')
            med_ret = row.get('median_best_7d_return')
            p95_ret = row.get('p95_best_7d_return')
            avg_dd = row.get('avg_max_drawdown_7d_pct')
            med_dd = row.get('median_max_drawdown_7d_pct')

            avg_ret_str = f"{avg_ret:.2f}%" if pd.notna(avg_ret) else "-"
            med_ret_str = f"{med_ret:.2f}%" if pd.notna(med_ret) else "-"
            p95_ret_str = f"{p95_ret:.2f}%" if pd.notna(p95_ret) else "-"
            avg_dd_str = f"{avg_dd:.2f}%" if pd.notna(avg_dd) else "-"
            med_dd_str = f"{med_dd:.2f}%" if pd.notna(med_dd) else "-"

            lines.append(f"| {streak} | {avg_ret_str} | {med_ret_str} | {p95_ret_str} | {avg_dd_str} | {med_dd_str} |")

        lines.append("")

    # Per-Stock Summary
    lines.append("## PER-STOCK STATISTICS")
    lines.append("")

    if per_stock_df is not None and len(per_stock_df) > 0:
        # Count quality flags
        good_count = (per_stock_df['sample_quality_flag'] == 'GOOD').sum()
        low_sample = (per_stock_df['sample_quality_flag'] == 'LOW_SAMPLE').sum()
        no_events = (per_stock_df['sample_quality_flag'] == 'NO_EVENTS').sum()
        insufficient = (per_stock_df['sample_quality_flag'] == 'INSUFFICIENT_HISTORY').sum()

        lines.append("| Quality Flag | Count |")
        lines.append("|--------------|-------|")
        lines.append(f"| GOOD | {good_count} |")
        lines.append(f"| LOW_SAMPLE | {low_sample} |")
        lines.append(f"| NO_EVENTS | {no_events} |")
        lines.append(f"| INSUFFICIENT_HISTORY | {insufficient} |")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by Kobe Bounce Analysis System*")

    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding='utf-8')

    return output_path
