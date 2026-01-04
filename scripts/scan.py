#!/usr/bin/env python3
"""
Kobe Daily Stock Scanner

Scans the universe for trading signals using the Dual Strategy System (v2.2):
- IBS+RSI Mean Reversion (high frequency) — ~59.9% WR, ~1.46 PF
- Turtle Soup Liquidity Sweep (high conviction) — ~61.0% WR, ~1.37 PF

Features:
- Loads universe from data/universe/optionable_liquid_900.csv (900 symbols)
- Fetches latest EOD data via Polygon or multi-source fallback
- Runs DualStrategyScanner for combined signals
- Outputs signals to stdout and logs/signals.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from data.providers.multi_source import fetch_daily_bars_multi
from data.universe.loader import load_universe
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
from config.settings_loader import get_selection_config
from core.regime_filter import get_regime_filter_config, filter_signals_by_regime, fetch_spy_bars
from core.earnings_filter import filter_signals_by_earnings
from ml_meta.features import compute_features_frame
from ml_meta.model import load_model, predict_proba, FEATURE_COLS
from altdata.sentiment import load_daily_cache, normalize_sentiment_to_conf

# Portfolio-aware filtering (Phase 7 - Scheduler v2)
try:
    from portfolio.heat_monitor import get_heat_monitor, HeatLevel
    from risk.advanced.correlation_limits import EnhancedCorrelationLimits, SECTOR_MAP
    PORTFOLIO_FILTERS_AVAILABLE = True
except ImportError:
    PORTFOLIO_FILTERS_AVAILABLE = False

# LLM Trade Analyzer (human-like reasoning)
try:
    from cognitive.llm_trade_analyzer import get_trade_analyzer, DailyInsightReport
    LLM_ANALYZER_AVAILABLE = True
except ImportError:
    LLM_ANALYZER_AVAILABLE = False

# Alpaca live data (for real-time prices during market hours)
try:
    from data.providers.alpaca_live import (
        is_market_open,
        get_market_clock,
        fetch_multi_quotes,
        get_current_price,
    )
    ALPACA_LIVE_AVAILABLE = True
except ImportError:
    ALPACA_LIVE_AVAILABLE = False

# HMM Regime Detector (ML-powered market regime classification)
try:
    from ml_advanced.hmm_regime_detector import AdaptiveRegimeDetector, MarketRegime
    HMM_REGIME_AVAILABLE = True
except ImportError:
    HMM_REGIME_AVAILABLE = False

# VIX Monitor (pause trading when VIX > 30)
try:
    from core.vix_monitor import get_vix_monitor, VIXConfig
    VIX_MONITOR_AVAILABLE = True
except ImportError:
    VIX_MONITOR_AVAILABLE = False


def get_last_trading_day(reference_date: datetime = None) -> tuple[str, bool, str]:
    """
    Get the last trading day and determine scan mode.

    Returns:
        tuple: (date_str, use_preview_mode, mode_reason)
        - date_str: YYYY-MM-DD of the trading day to use
        - use_preview_mode: True if we should use preview (current bar values)
        - mode_reason: Human-readable explanation

    WEEKEND LOGIC:
    - Saturday/Sunday: Use Friday's close + PREVIEW mode (signals trigger Monday)
    - Monday-Friday: Use today's date + NORMAL mode (fresh data)

    WHY PREVIEW ON WEEKENDS:
    - Normal mode uses .shift(1) for lookahead safety (checks PREVIOUS bar)
    - On weekends, Friday is the last bar - but shift(1) would check Thursday
    - Preview mode uses CURRENT bar (Friday's values) so we see what triggers Monday
    """
    if reference_date is None:
        reference_date = datetime.now()

    weekday = reference_date.weekday()  # Monday=0, Sunday=6
    is_weekend = weekday >= 5  # Saturday=5, Sunday=6

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar('NYSE')

        # Look back up to 10 days to find last trading day
        start_check = reference_date - timedelta(days=10)
        schedule = nyse.schedule(
            start_date=start_check.strftime('%Y-%m-%d'),
            end_date=reference_date.strftime('%Y-%m-%d')
        )

        if len(schedule) > 0:
            last_trading = schedule.index[-1]
            last_trading_str = last_trading.strftime('%Y-%m-%d')
            ref_str = reference_date.strftime('%Y-%m-%d')

            if is_weekend:
                # Weekend: use last trading day (Friday) + preview mode
                return last_trading_str, True, f"WEEKEND: Using {last_trading_str} (Friday) + PREVIEW mode"
            elif ref_str == last_trading_str:
                # Weekday and today is a trading day: use today + normal mode
                return ref_str, False, f"WEEKDAY: Using today ({ref_str}) + NORMAL mode (fresh data)"
            else:
                # Weekday but today is a holiday: use last trading day + preview
                return last_trading_str, True, f"HOLIDAY: Using {last_trading_str} + PREVIEW mode"
        else:
            # Fallback
            return _fallback_trading_day(reference_date, is_weekend)

    except ImportError:
        return _fallback_trading_day(reference_date, is_weekend)


def _fallback_trading_day(reference_date: datetime, is_weekend: bool) -> tuple[str, bool, str]:
    """Fallback when pandas_market_calendars not available."""
    if is_weekend:
        # Go back to Friday
        days_back = (reference_date.weekday() - 4) % 7
        if days_back == 0:
            days_back = 7 if reference_date.weekday() != 4 else 0
        last_friday = reference_date - timedelta(days=days_back)
        return last_friday.strftime('%Y-%m-%d'), True, f"WEEKEND: Using Friday ({last_friday.strftime('%Y-%m-%d')}) + PREVIEW"
    else:
        return reference_date.strftime('%Y-%m-%d'), False, "WEEKDAY: Using today + NORMAL mode"

# Quality Gate System (v2.0 - reduces ~50/week to ~5/week)
try:
    from risk.signal_quality_gate import filter_to_best_signals, get_quality_gate
    QUALITY_GATE_AVAILABLE = True
except ImportError:
    QUALITY_GATE_AVAILABLE = False

# Signal Adjudicator (4-factor ranking - complements quality gate)
try:
    from cognitive.signal_adjudicator import adjudicate_signals
    ADJUDICATOR_AVAILABLE = True
except ImportError:
    ADJUDICATOR_AVAILABLE = False

# Cognitive system (optional)
try:
    from cognitive.signal_processor import get_signal_processor
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_UNIVERSE = ROOT / "data" / "universe" / "optionable_liquid_900.csv"
SIGNALS_LOG = ROOT / "logs" / "signals.jsonl"
CACHE_DIR = ROOT / "data" / "cache"
LOOKBACK_DAYS = 400  # Need 280+ trading days for SMA(200) + buffer (~400 calendar days)


# -----------------------------------------------------------------------------
# Top-N Selection with Portfolio Gates (Phase 5 - Codex #4)
# -----------------------------------------------------------------------------
def select_top_n_with_gates(
    signals: pd.DataFrame,
    config: dict,
    current_positions: List[dict] = None,
    price_data: pd.DataFrame = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Select top N signals respecting portfolio gates.

    Phase 5 Implementation (Codex #4):
    - Respects max correlation with existing positions
    - Respects sector concentration limits
    - Respects single position size limits
    - Maintains strategy mix (ICT/IBS diversification)

    Args:
        signals: DataFrame of candidate signals with 'conf_score' column
        config: Selection configuration from settings
        current_positions: List of current position dicts
        price_data: Historical price data for correlation calculation
        verbose: Print selection details

    Returns:
        DataFrame of selected signals (up to top_n.n)
    """
    if signals.empty:
        return signals

    top_n_cfg = config.get('top_n', {})
    if not top_n_cfg.get('enabled', False):
        # Fallback to TOTD mode (single best signal)
        if verbose:
            print("  [INFO] Top-N disabled, using TOTD mode")
        return signals.sort_values('conf_score', ascending=False).head(1)

    n = int(top_n_cfg.get('n', 3))
    max_correlation = float(top_n_cfg.get('max_correlation', 0.70))
    max_sector_pct = float(top_n_cfg.get('max_sector_pct', 0.40))
    float(top_n_cfg.get('max_single_name_pct', 0.25))
    min_diversification = int(top_n_cfg.get('min_diversification', 2))

    # Sort by confidence (highest first)
    ranked = signals.sort_values('conf_score', ascending=False).copy()

    selected = []
    selected_symbols = set()
    selected_sectors = {}
    selected_strategies = {}

    # Get sector map
    try:
        sector_map = SECTOR_MAP if PORTFOLIO_FILTERS_AVAILABLE else {}
    except Exception:
        sector_map = {}

    for _, signal in ranked.iterrows():
        if len(selected) >= n:
            break

        symbol = str(signal.get('symbol', ''))
        strategy = str(signal.get('strategy', 'unknown'))
        sector = sector_map.get(symbol, 'Unknown')

        # Skip if already selected
        if symbol in selected_symbols:
            continue

        # Check sector concentration
        sector_count = selected_sectors.get(sector, 0)
        total_selected = len(selected) + 1
        if total_selected > 1:
            sector_pct = (sector_count + 1) / total_selected
            if sector_pct > max_sector_pct and sector != 'Unknown':
                if verbose:
                    print(f"    Skip {symbol}: sector {sector} would exceed {max_sector_pct*100:.0f}% limit")
                continue

        # Check strategy diversification (try to mix ICT and IBS)
        priority = top_n_cfg.get('priority', {})
        strategy_mix_weight = float(priority.get('strategy_mix', 0.30))
        if strategy_mix_weight > 0 and len(selected) > 0:
            strat_count = selected_strategies.get(strategy.lower(), 0)
            if strat_count >= 2 and total_selected <= n:
                # Allow but deprioritize (we're iterating by conf_score anyway)
                pass

        # Check correlation with existing positions (if available)
        if current_positions and PORTFOLIO_FILTERS_AVAILABLE and price_data is not None:
            try:
                from risk.advanced.correlation_limits import EnhancedCorrelationLimits
                corr_checker = EnhancedCorrelationLimits(max_correlation=max_correlation)
                positions_dict = {
                    p.get('symbol', ''): {'value': abs(float(p.get('market_value', 0)))}
                    for p in current_positions if p.get('symbol')
                }

                # Build returns data
                returns_data = {}
                for sym in [symbol] + list(positions_dict.keys()):
                    sym_data = price_data[price_data['symbol'] == sym]
                    if len(sym_data) >= 20:
                        sym_data = sym_data.sort_values('timestamp')
                        returns = sym_data['close'].pct_change().dropna().values
                        if len(returns) >= 20:
                            returns_data[sym] = returns

                if returns_data:
                    check = corr_checker.check_entry(
                        symbol=symbol,
                        proposed_value=1000,  # Notional for checking
                        current_positions=positions_dict,
                        returns_data=returns_data,
                    )
                    if not check.can_enter:
                        if verbose:
                            print(f"    Skip {symbol}: {check.reason}")
                        continue
            except Exception as e:
                if verbose:
                    print(f"    [WARN] Correlation check failed for {symbol}: {e}")

        # Passed all checks - add to selected
        selected.append(signal)
        selected_symbols.add(symbol)
        selected_sectors[sector] = selected_sectors.get(sector, 0) + 1
        selected_strategies[strategy.lower()] = selected_strategies.get(strategy.lower(), 0) + 1

    if verbose:
        print(f"  Top-N selection: {len(selected)}/{n} signals selected")
        if selected_strategies:
            print(f"    Strategy mix: {dict(selected_strategies)}")
        if selected_sectors:
            print(f"    Sector diversity: {len([s for s in selected_sectors.keys() if s != 'Unknown'])} unique sectors")

    if not selected:
        return pd.DataFrame()

    result = pd.DataFrame(selected)

    # Verify diversification
    unique_sectors = len([s for s in selected_sectors.keys() if s != 'Unknown'])
    if unique_sectors < min_diversification and len(result) >= min_diversification:
        if verbose:
            print(f"    [WARN] Low diversification: {unique_sectors} sectors (min: {min_diversification})")

    return result


# -----------------------------------------------------------------------------
# Portfolio-aware filtering (Scheduler v2 - Phase 7)
# -----------------------------------------------------------------------------
def apply_portfolio_filters(
    signals: pd.DataFrame,
    current_positions: List[Dict],
    price_data: pd.DataFrame,
    equity: float = 10000.0,
    max_correlation: float = 0.70,
    max_sector_pct: float = 0.40,
    max_single_position_pct: float = 0.20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filter signals by portfolio constraints.

    Per Scheduler v2 Plan Phase 7:
    1. Correlation cap (max 0.70 correlation with existing positions)
    2. Sector cap (max 40% in any sector)
    3. Exposure cap (max 20% single position)
    4. Don't add if heat = HOT/OVERHEATED

    Args:
        signals: DataFrame of candidate signals
        current_positions: List of current position dicts {symbol, market_value, ...}
        price_data: DataFrame with OHLCV data for correlation calculation
        equity: Total account equity for exposure calculations
        max_correlation: Maximum allowed correlation with existing positions
        max_sector_pct: Maximum sector concentration allowed
        max_single_position_pct: Maximum single position as % of equity
        verbose: Print filtering details

    Returns:
        Filtered DataFrame of signals that pass portfolio constraints
    """
    if not PORTFOLIO_FILTERS_AVAILABLE:
        if verbose:
            print("  [INFO] Portfolio filters not available (modules not imported)")
        return signals

    if signals.empty:
        return signals

    # If no existing positions, all signals pass portfolio filters
    if not current_positions:
        if verbose:
            print("  [INFO] No existing positions - all signals pass portfolio filter")
        return signals

    filtered_signals = []
    rejected_reasons = []

    # Initialize heat monitor and correlation checker
    heat_monitor = get_heat_monitor()
    corr_checker = EnhancedCorrelationLimits(
        max_correlation=max_correlation,
        max_sector_weight=max_sector_pct,
        max_sector_positions=3,  # Max 3 positions per sector
    )

    # Calculate current heat status
    heat_status = heat_monitor.calculate_heat(
        positions=current_positions,
        equity=equity,
    )

    # Check 4: Don't add if heat = HOT/OVERHEATED
    if heat_status.heat_level in [HeatLevel.HOT, HeatLevel.OVERHEATED]:
        if verbose:
            print(f"  [WARN] Portfolio is {heat_status.heat_level.value} (score: {heat_status.heat_score:.0f}) - blocking all new entries")
        return pd.DataFrame()

    # Build returns data for correlation calculation
    returns_data = {}
    if price_data is not None and not price_data.empty:
        # DETERMINISM FIX: Use sorted() instead of set() for deterministic iteration order
        symbols_for_correlation = sorted(set(
            list([p.get('symbol', '') for p in current_positions]) +
            list(signals['symbol'].unique() if 'symbol' in signals.columns else [])
        ))
        for symbol in symbols_for_correlation:
            try:
                sym_data = price_data[price_data['symbol'] == symbol].copy()
                if len(sym_data) >= 60:
                    sym_data = sym_data.sort_values('timestamp')
                    returns = sym_data['close'].pct_change().dropna().values
                    if len(returns) >= 20:
                        returns_data[symbol] = returns
            except Exception:
                pass

    # Build current positions dict for correlation check
    positions_dict = {
        p.get('symbol', ''): {'value': abs(float(p.get('market_value', 0)))}
        for p in current_positions if p.get('symbol')
    }

    # Sector counts for current positions
    current_sectors = {}
    for pos in current_positions:
        sym = pos.get('symbol', '')
        sector = SECTOR_MAP.get(sym, 'Unknown')
        current_sectors[sector] = current_sectors.get(sector, 0) + 1

    for _, row in signals.iterrows():
        symbol = row.get('symbol', '')
        float(row.get('entry_price', 0))

        # Skip if symbol already in portfolio
        if symbol in positions_dict:
            rejected_reasons.append((symbol, 'already_in_portfolio'))
            continue

        # Calculate proposed position value (default 2% risk sizing)
        proposed_value = min(equity * max_single_position_pct, 1500)

        # Check 1 & 2: Correlation and sector using EnhancedCorrelationLimits
        if positions_dict and returns_data:
            try:
                check_result = corr_checker.check_entry(
                    symbol=symbol,
                    proposed_value=proposed_value,
                    current_positions=positions_dict,
                    returns_data=returns_data,
                )

                if not check_result.can_enter:
                    rejected_reasons.append((symbol, check_result.reason))
                    continue

                # Log warnings but allow entry
                if check_result.warnings:
                    for warn in check_result.warnings:
                        if verbose:
                            print(f"  [WARN] {symbol}: {warn}")

            except Exception as e:
                if verbose:
                    print(f"  [WARN] Correlation check failed for {symbol}: {e}")

        # Check 3: Single position exposure cap (already handled in proposed_value calc)
        # Position would be sized to max_single_position_pct

        # Passed all checks
        filtered_signals.append(row)

    if verbose:
        passed = len(filtered_signals)
        total = len(signals)
        rejected = total - passed
        print(f"  Portfolio filter: {passed}/{total} signals passed ({rejected} rejected)")
        if rejected_reasons and len(rejected_reasons) <= 10:
            for sym, reason in rejected_reasons[:5]:
                print(f"    - {sym}: {reason}")

    if not filtered_signals:
        return pd.DataFrame()

    return pd.DataFrame(filtered_signals)


# -----------------------------------------------------------------------------
# Scanner functions
# -----------------------------------------------------------------------------
def fetch_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch EOD data for a single symbol."""
    try:
        df = fetch_daily_bars_multi(
            symbol=symbol,
            start=start_date,
            end=end_date,
            cache_dir=cache_dir,
        )
        return df
    except Exception as e:
        print(f"  [WARN] Failed to fetch {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def run_strategies(
    data: pd.DataFrame,
    strategies: List[str],
    apply_filters: bool,
    spy_bars: Optional[pd.DataFrame],
    preview_mode: bool = False,
) -> pd.DataFrame:
    """Run Dual Strategy Scanner and return combined signals."""
    try:
        sel_cfg = get_selection_config()
        params = DualStrategyParams(min_price=float(sel_cfg.get('min_price', 10.0)))
        scanner = DualStrategyScanner(params, preview_mode=preview_mode)

        # Generate signals (IBS+RSI + Turtle Soup combined)
        signals = scanner.generate_signals(data)

        if signals.empty:
            return pd.DataFrame()

        # Apply regime/earnings filters if requested
        if apply_filters and spy_bars is not None and not spy_bars.empty:
            try:
                signals = filter_signals_by_regime(signals, spy_bars, get_regime_filter_config())
            except Exception:
                pass
        if apply_filters and not signals.empty:
            try:
                recs = signals.to_dict('records')
                signals = pd.DataFrame(filter_signals_by_earnings(recs))
            except Exception:
                pass

        return signals

    except Exception as e:
        print(f"  [WARN] Dual Strategy error: {e}", file=sys.stderr)
        return pd.DataFrame()


def log_signals(signals: pd.DataFrame, scan_id: str) -> None:
    """Append signals to JSONL log file."""
    SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)

    with SIGNALS_LOG.open("a", encoding="utf-8") as f:
        for _, row in signals.iterrows():
            record = {
                "ts": datetime.utcnow().isoformat(),
                "scan_id": scan_id,
                "event": "signal",
                **{k: v for k, v in row.items() if pd.notna(v)},
            }
            # Convert Timestamp to string
            for k, v in record.items():
                if isinstance(v, pd.Timestamp):
                    record[k] = v.isoformat()
            f.write(json.dumps(record, default=str) + "\n")


def compute_conf_score(row: pd.Series) -> float:
    """
    Compute confidence score for a signal row.

    Returns existing conf_score if present, otherwise computes from strategy score.
    - Turtle Soup: score typically 100-300, normalize to 0-1
    - IBS_RSI: score typically 0-25, normalize to 0-1
    """
    if 'conf_score' in row.index and pd.notna(row.get('conf_score')):
        try:
            return float(row['conf_score'])
        except Exception:
            pass
    score = float(row.get('score', 0.0))
    if score > 50:  # Turtle Soup
        return min(score / 300.0, 1.0)
    else:  # IBS_RSI
        return min(score / 25.0, 1.0)


def format_signal_row(row: pd.Series) -> str:
    """Format a single signal for display."""
    parts = [
        f"{row.get('strategy', '?'):>5}",
        f"{row.get('symbol', '?'):<6}",
        f"{row.get('side', '?'):<6}",
        f"@ ${row.get('entry_price', 0):>8.2f}",
        f"stop ${row.get('stop_loss', 0):>8.2f}",
    ]
    reason = row.get("reason", "")
    if reason:
        parts.append(f"| {reason}")
    return " ".join(parts)


def print_signals_table(signals: pd.DataFrame) -> None:
    """Print signals in a formatted table."""
    if signals.empty:
        print("\n  No signals generated.")
        return

    print("\n  SIGNALS")
    print("  " + "-" * 76)
    print(f"  {'STRAT':>5} {'SYMBOL':<6} {'SIDE':<6} {'ENTRY':>12} {'STOP':>12} | REASON")
    print("  " + "-" * 76)

    for _, row in signals.iterrows():
        print("  " + format_signal_row(row))

    print("  " + "-" * 76)
    print(f"  Total: {len(signals)} signal(s)")

    # Summary by strategy
    if "strategy" in signals.columns:
        by_strat = signals.groupby("strategy").size()
        print(f"  By strategy: {dict(by_strat)}")

    # Summary by side
    if "side" in signals.columns:
        by_side = signals.groupby("side").size()
        print(f"  By side: {dict(by_side)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kobe Daily Stock Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scan.py                        # Scan all strategies
  python scripts/scan.py --strategy ibs_rsi     # Only IBS+RSI signals
  python scripts/scan.py --strategy turtle_soup # Only ICT signals
  python scripts/scan.py --cap 50               # Scan first 50 symbols
  python scripts/scan.py --json                 # Output as JSON
        """,
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="./.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--universe",
        type=str,
        default=str(DEFAULT_UNIVERSE),
        help="Path to universe CSV file",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        choices=["dual", "ibs_rsi", "turtle_soup", "all"],
        default="all",
        help="Strategy to run: dual (IBS+RSI + TurtleSoup), ibs_rsi, turtle_soup, all (default: all)",
    )
    ap.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Limit number of symbols to scan",
    )
    ap.add_argument("--top3", action="store_true", help="Select Top-3 picks and write logs/daily_picks.csv")
    ap.add_argument(
        "--top3-mix",
        type=str,
        choices=["ict2_ibs1", "pure"],
        default="ict2_ibs1",
        help="Top-3 selection rule: ict2_ibs1 (default) enforces 2x ICT + 1x IBS; pure takes the highest 3 by confidence",
    )
    ap.add_argument("--min-price", type=float, default=None, help="Override min price for selection")
    ap.add_argument("--no-filters", action="store_true", help="Disable regime/earnings filters")
    ap.add_argument("--date", type=str, default=None, help="Use YYYY-MM-DD as end date (default: last business day)")
    ap.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD), default: lookback from today",
    )
    ap.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), default: today",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output signals as JSON",
    )
    ap.add_argument("--ml", action="store_true", default=True, help="Score signals with ML meta-models (ON by default)")
    ap.add_argument("--no-ml", action="store_true", help="Disable ML scoring")
    ap.add_argument("--min-conf", type=float, default=0.55, help="Min confidence [0-1] to approve TOTD when --ml is on")
    ap.add_argument("--min-adv-usd", type=float, default=5000000.0, help="Minimum 60-day ADV in USD to consider for Top-3/TOTD")
    ap.add_argument("--ensure-top3", action="store_true", help="Guarantee 3 picks; fill from highest-confidence leftovers")
    ap.add_argument("--out-picks", type=str, default=str(ROOT / 'logs' / 'daily_picks.csv'), help="Output CSV for Top-3 picks")
    ap.add_argument("--out-totd", type=str, default=str(ROOT / 'logs' / 'trade_of_day.csv'), help="Output CSV for Trade of the Day")
    # (ml arg already defined above)
    ap.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing to signals.jsonl",
    )
    ap.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Disable v2.0 quality gate filtering (keeps ~50 signals/week instead of ~5)",
    )
    ap.add_argument(
        "--quality-max-signals",
        type=int,
        default=3,
        help="Max signals per day when quality gate is enabled (default: 3)",
    )
    ap.add_argument(
        "--cognitive",
        action="store_true",
        default=True,
        help="Enable cognitive brain evaluation (ON by default)",
    )
    ap.add_argument(
        "--no-cognitive",
        action="store_true",
        help="Disable cognitive brain evaluation",
    )
    ap.add_argument(
        "--cognitive-min-conf",
        type=float,
        default=0.45,
        help="Minimum cognitive confidence to approve signal (default: 0.45, lowered from 0.5 because ML ensemble predicts honestly around 45-50%%)",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Enforce 100% deterministic output order (stable sorts, seeded RNG)",
    )
    ap.add_argument(
        "--narrative",
        action="store_true",
        help="Generate Claude LLM narrative analysis for picks (human-like reasoning)",
    )
    ap.add_argument(
        "--out-insights",
        type=str,
        default=str(ROOT / "logs" / "daily_insights.json"),
        help="Output JSON file for daily insights with LLM narratives",
    )
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: use current bar values (for weekend analysis). Shows what would trigger on next trading day.",
    )
    # === Phase 2: Feature Toggle Flags (ML/AI Components) ===
    ap.add_argument(
        "--calibration",
        action="store_true",
        help="Enable probability calibration (isotonic/platt) for ML confidence scores",
    )
    ap.add_argument(
        "--conformal",
        action="store_true",
        help="Enable conformal prediction for uncertainty-aware position sizing",
    )
    ap.add_argument(
        "--exec-bandit",
        action="store_true",
        help="Enable execution bandit for adaptive order routing (Thompson/UCB/epsilon-greedy)",
    )
    ap.add_argument(
        "--intraday-trigger",
        action="store_true",
        help="Enable intraday entry trigger (VWAP reclaim/first-hour confirmation before entry)",
    )
    ap.add_argument(
        "--portfolio-filter",
        action="store_true",
        help="Enable portfolio-aware filtering (correlation cap, sector cap, heat check)",
    )
    ap.add_argument(
        "--positions-file",
        type=str,
        default=str(ROOT / "state" / "positions.json"),
        help="Path to current positions JSON file for portfolio filtering",
    )
    ap.add_argument(
        "--equity",
        type=float,
        default=10000.0,
        help="Account equity for portfolio filter calculations (default: 10000)",
    )
    ap.add_argument(
        "--live-data",
        action="store_true",
        help="Use Alpaca live data for current prices (paper trading mode). Historical data still from Polygon.",
    )
    args = ap.parse_args()

    # Handle --no-* override flags for ML and cognitive defaults
    if args.no_ml:
        args.ml = False
    if args.no_cognitive:
        args.cognitive = False

    # Load environment
    dotenv_path = Path(args.dotenv)
    if dotenv_path.exists():
        loaded = load_env(dotenv_path)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv_path}")
    else:
        if args.verbose:
            print(f"Warning: dotenv file not found: {dotenv_path}", file=sys.stderr)

    # === DETERMINISM MODE: Enforce reproducible results ===
    if args.deterministic:
        import random
        import numpy as np
        import hashlib
        # CRITICAL: Seed BOTH random modules for full determinism
        random.seed(42)     # Python built-in random (used by execution_bandit, curiosity_engine)
        np.random.seed(42)  # NumPy random (used by ML models, Thompson sampling)
        if args.verbose:
            print("[DETERMINISM] Mode ENABLED - seeded random + np.random + stable sorts")

    # CRITICAL: Validate strategy imports at startup
    # This ensures we NEVER use deprecated standalone strategies
    try:
        from strategies.registry import validate_strategy_import
        validate_strategy_import()
        if args.verbose:
            print("[STRATEGY] Using canonical DualStrategyScanner (v2.2 verified)")
    except ImportError:
        pass  # Registry not available

    # === Apply CLI feature flags to config (runtime overrides) ===
    if args.calibration or args.conformal or args.exec_bandit or args.intraday_trigger:
        try:
            from config.settings_loader import load_settings, _settings_cache
            config = load_settings()

            # Override calibration setting
            if args.calibration:
                if 'ml' not in config:
                    config['ml'] = {}
                if 'calibration' not in config['ml']:
                    config['ml']['calibration'] = {}
                config['ml']['calibration']['enabled'] = True
                if args.verbose:
                    print("[CLI] --calibration: Probability calibration ENABLED")

            # Override conformal setting
            if args.conformal:
                if 'ml' not in config:
                    config['ml'] = {}
                if 'conformal' not in config['ml']:
                    config['ml']['conformal'] = {}
                config['ml']['conformal']['enabled'] = True
                if args.verbose:
                    print("[CLI] --conformal: Conformal prediction ENABLED")

            # Override execution bandit setting
            if args.exec_bandit:
                if 'execution' not in config:
                    config['execution'] = {}
                config['execution']['bandit_enabled'] = True
                if args.verbose:
                    print("[CLI] --exec-bandit: Execution bandit ENABLED")

            # Override intraday trigger setting
            if args.intraday_trigger:
                if 'execution' not in config:
                    config['execution'] = {}
                if 'intraday_trigger' not in config['execution']:
                    config['execution']['intraday_trigger'] = {}
                config['execution']['intraday_trigger']['enabled'] = True
                if args.verbose:
                    print("[CLI] --intraday-trigger: Intraday entry trigger ENABLED")

            # Update the settings cache so other modules see these overrides
            _settings_cache.clear()
            _settings_cache.update(config)

        except Exception as e:
            if args.verbose:
                print(f"[WARN] Could not apply CLI feature flags: {e}")

    # Check Polygon API key
    if not os.getenv("POLYGON_API_KEY"):
        print("Error: POLYGON_API_KEY not set. Please provide via --dotenv.", file=sys.stderr)
        return 1

    # VIX Pause Check: Block signal generation when VIX > 30
    if VIX_MONITOR_AVAILABLE:
        try:
            vix_monitor = get_vix_monitor()
            should_pause, vix_level, reason = vix_monitor.should_pause_trading()

            if should_pause:
                print(f"\n*** TRADING PAUSED: {reason} ***")
                print(f"*** VIX = {vix_level:.1f} (threshold: {vix_monitor.config.pause_threshold}) ***")
                print("*** Scan aborted. No signals will be generated. ***\n")
                return 0  # Clean exit, not an error

            if args.verbose:
                print(f"[VIX] Level: {vix_level:.1f} - {reason}")

        except Exception as e:
            # VIX fetch failed - continue with warning
            if args.verbose:
                print(f"[VIX] Warning: Could not check VIX level: {e}")

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"Error: Universe file not found: {universe_path}", file=sys.stderr)
        return 1

    symbols = load_universe(universe_path, cap=args.cap)
    if not symbols:
        print(f"Error: No symbols loaded from {universe_path}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(symbols)} symbols from {universe_path}")

    # Check live data mode
    market_is_open = False
    live_data_enabled = args.live_data and ALPACA_LIVE_AVAILABLE
    if live_data_enabled:
        clock = get_market_clock()
        if clock:
            market_is_open = clock.get("is_open", False)
            print(f"\n[LIVE DATA] Market open: {market_is_open}")
            if market_is_open:
                print("[LIVE DATA] Will use Alpaca for current prices")
            else:
                print("[LIVE DATA] Market closed - using cached EOD data")
        else:
            print("[LIVE DATA] Could not get market clock - using cached data")
            live_data_enabled = False
    elif args.live_data and not ALPACA_LIVE_AVAILABLE:
        print("[WARN] --live-data requested but Alpaca module not available")

    # Determine date range
    # WEEKEND-SAFE: Auto-detect weekends/holidays and use appropriate mode
    if args.date or args.end:
        # User specified date - use as-is, check if it's a trading day
        end_date = args.date or args.end
        use_preview = args.preview  # Only use preview if explicitly requested
        mode_reason = f"USER-SPECIFIED: Using {end_date}"
        if args.preview:
            mode_reason += " + PREVIEW mode (user requested)"
    else:
        # Auto-detect: determine best date and mode based on day of week
        end_date, auto_preview, mode_reason = get_last_trading_day(datetime.now())
        use_preview = args.preview or auto_preview

    # Print mode explanation
    print(f"\n*** {mode_reason} ***")

    if args.start:
        start_date = args.start
    else:
        end_dt = datetime.fromisoformat(end_date)
        start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
        start_date = start_dt.date().isoformat()

    if args.verbose:
        print(f"Date range: {start_date} to {end_date}")

    # Determine strategies to run (dual strategy handles both IBS_RSI and TurtleSoup)
    strategies = ["dual"]  # Always use dual strategy scanner
    if args.verbose:
        print("Running Dual Strategy Scanner (IBS+RSI + Turtle Soup)")

    # Scan ID for logging
    scan_id = f"SCAN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Fetch data and run strategies
    print(f"\nKobe Scanner - {scan_id}")
    if use_preview:
        print("*** PREVIEW MODE: Using current bar values (signals trigger NEXT trading day) ***")
    print(f"Scanning {len(symbols)} symbols with Dual Strategy (IBS+RSI + Turtle Soup)...")
    print("-" * 60)

    all_data: List[pd.DataFrame] = []
    success_count = 0
    fail_count = 0

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Parallel fetching for 5-10x speedup (respects Polygon rate limits)
    max_workers = min(10, len(symbols))  # Max 10 concurrent requests

    def fetch_wrapper(symbol: str) -> tuple:
        """Wrapper to return (symbol, dataframe) tuple."""
        df = fetch_symbol_data(symbol, start_date, end_date, CACHE_DIR)
        return (symbol, df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_wrapper, sym): sym for sym in symbols}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            symbol, df = future.result()

            if not df.empty and len(df) > 0:
                all_data.append(df)
                success_count += 1
                if args.verbose:
                    print(f"  [{completed}/{len(symbols)}] {symbol}: {len(df)} bars", flush=True)
            else:
                fail_count += 1
                if args.verbose:
                    print(f"  [{completed}/{len(symbols)}] {symbol}: SKIP (no data)", flush=True)

            # Progress update every 50 symbols (non-verbose mode)
            if not args.verbose and completed % 50 == 0:
                print(f"  Progress: {completed}/{len(symbols)} symbols...", flush=True)

    print(f"\nFetched: {success_count} symbols, skipped: {fail_count}")

    if not all_data:
        print("Error: No data fetched for any symbols.", file=sys.stderr)
        return 1

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total bars: {len(combined):,}")

    # Run strategies
    print("\nRunning strategies...")
    # Load SPY for regime filter if enabled
    spy_bars = None
    hmm_regime_state = None
    apply_filters = not args.no_filters
    if apply_filters:
        try:
            spy_bars = fetch_spy_bars(start_date, end_date, cache_dir=CACHE_DIR)
        except Exception:
            spy_bars = None

    # HMM Regime Detection (ML-enhanced)
    if HMM_REGIME_AVAILABLE and args.ml and spy_bars is not None and not spy_bars.empty:
        try:
            regime_detector = AdaptiveRegimeDetector(use_hmm=True)
            # Note: VIX data is optional, we'll pass None if not available
            hmm_regime_state = regime_detector.detect_regime(spy_bars, vix_data=None)
            if args.verbose:
                print(f"  HMM Regime: {hmm_regime_state.regime.value} (conf={hmm_regime_state.confidence:.2f})")
        except Exception as e:
            if args.verbose:
                print(f"  HMM regime detection failed: {e}", file=sys.stderr)

    # Selection config overrides
    sel_cfg = get_selection_config()
    if args.min_price is not None and args.min_price > 0:
        sel_cfg['min_price'] = float(args.min_price)

    signals = run_strategies(combined, strategies, apply_filters=apply_filters, spy_bars=spy_bars, preview_mode=use_preview)

    # === QUALITY GATE (v2.0) ===
    # Reduces ~50 signals/week to ~5/week with higher win rate
    if QUALITY_GATE_AVAILABLE and not args.no_quality_gate and not signals.empty:
        try:
            pre_count = len(signals)
            signals = filter_to_best_signals(
                signals=signals,
                price_data=combined,
                spy_data=spy_bars,
                max_signals=args.quality_max_signals,
            )
            post_count = len(signals)
            if args.verbose:
                print(f"Quality gate: {pre_count} -> {post_count} signal(s) (filtered {pre_count - post_count})")
            else:
                print(f"Quality gate: filtered to {post_count} high-quality signal(s)")
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] Quality gate failed: {e}", file=sys.stderr)
    elif not QUALITY_GATE_AVAILABLE and not args.no_quality_gate:
        if args.verbose:
            print("  [INFO] Quality gate not available (module not found)")

    # === SIGNAL ADJUDICATOR (4-factor ranking) ===
    # Ranks quality-passed signals by: signal strength, pattern confluence,
    # volatility contraction, and sector strength
    if ADJUDICATOR_AVAILABLE and not args.no_quality_gate and not signals.empty:
        try:
            signals = adjudicate_signals(
                signals=signals,
                price_data=combined,
                spy_data=spy_bars,
                max_signals=20,  # Keep top 20 for further processing
            )
            if args.verbose and 'adjudication_score' in signals.columns:
                top_score = signals['adjudication_score'].iloc[0] if len(signals) > 0 else 0
                print(f"Adjudicator: ranked {len(signals)} signals (top score: {top_score:.1f})")
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] Signal adjudication failed: {e}", file=sys.stderr)

    # Optional ML scoring
    if args.ml and not signals.empty:
        try:
            feats = compute_features_frame(combined)
            feats['timestamp'] = pd.to_datetime(feats['timestamp']).dt.normalize()
            sigs = signals.copy()
            sigs['timestamp'] = pd.to_datetime(sigs['timestamp']).dt.normalize()
            sigs = pd.merge(sigs, feats, on=['symbol','timestamp'], how='left')
            for col in FEATURE_COLS:
                if col not in sigs.columns:
                    sigs[col] = 0.0
            m_don = load_model('ibs_rsi')
            m_ict = load_model('turtle_soup')
            conf_vals = []
            for _, r in sigs.iterrows():
                strat = str(r.get('strategy','')).lower()
                row = r.reindex(FEATURE_COLS).astype(float).to_frame().T
                if strat in ('ibs_rsi','ibs') and m_don is not None:
                    conf_vals.append(float(predict_proba(m_don, row)[0]))
                elif strat in ('turtle_soup',) and m_ict is not None:
                    conf_vals.append(float(predict_proba(m_ict, row)[0]))
                else:
                    conf_vals.append(float(r.get('conf_score', 0.0)) if 'conf_score' in r else 0.0)
            sigs['conf_score'] = conf_vals

            # Blend sentiment if available for end_date
            try:
                end_day = pd.to_datetime(end_date).date().isoformat()
                sent = load_daily_cache(end_day)
                if not sent.empty and 'date' in sent.columns:
                    sent['date'] = pd.to_datetime(sent['date']).dt.normalize()
                    sigs = pd.merge(
                        sigs,
                        sent.rename(columns={'date': 'timestamp'})[['timestamp','symbol','sent_mean']],
                        on=['timestamp','symbol'], how='left'
                    )
                    # FIX: Use median of available sentiment for missing values (not 0.0)
                    # This prevents penalizing stocks with missing sentiment data
                    available_sent = sigs['sent_mean'].dropna()
                    fill_value = float(available_sent.median()) if len(available_sent) > 0 else 0.5
                    sigs['sent_mean'] = sigs['sent_mean'].astype(float).fillna(fill_value)
                    sent_conf = sigs['sent_mean'].apply(normalize_sentiment_to_conf)
                    # Blend: 0.8 ML probability + 0.2 sentiment
                    sigs['conf_score'] = 0.8 * sigs['conf_score'].astype(float) + 0.2 * sent_conf.astype(float)
            except Exception:
                pass

            signals = sigs
        except Exception as e:
            if args.verbose:
                print(f"  [WARN] ML scoring failed: {e}")

    # Optional cognitive brain evaluation
    cognitive_evaluated = []
    if args.cognitive and COGNITIVE_AVAILABLE and not signals.empty:
        print("\nRunning cognitive brain evaluation...")
        try:
            processor = get_signal_processor()
            processor.min_confidence = args.cognitive_min_conf

            # Build fast confidences from ML scores
            # Evaluate through cognitive system
            approved_df, cognitive_evaluated = processor.evaluate_signals(
                signals=signals,
                market_data=combined,
                spy_data=spy_bars,
            )

            if not approved_df.empty:
                # Merge cognitive results back
                signals = approved_df
                print(f"  Cognitive: {len(cognitive_evaluated)} evaluated -> {len(approved_df)} approved")

                # Show cognitive reasoning for each approved signal
                if args.verbose:
                    for ev in cognitive_evaluated:
                        if ev.approved:
                            sym = ev.original_signal.get('symbol', '?')
                            print(f"    {sym}: conf={ev.cognitive_confidence:.2f}, mode={ev.decision_mode}")
                            if ev.concerns:
                                print(f"      Concerns: {ev.concerns[:2]}")
            else:
                print("  Cognitive: All signals rejected (below confidence threshold)")
                signals = pd.DataFrame()

        except Exception as e:
            print(f"  [WARN] Cognitive evaluation failed: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif args.cognitive and not COGNITIVE_AVAILABLE:
        print("  [WARN] Cognitive system not available (import failed)")

    # Output results
    if args.json:
        if not signals.empty:
            # Convert to JSON-serializable format
            output = []
            for _, row in signals.iterrows():
                rec = {k: v for k, v in row.items() if pd.notna(v)}
                for k, v in rec.items():
                    if isinstance(v, pd.Timestamp):
                        rec[k] = v.isoformat()
                output.append(rec)
            print(json.dumps(output, indent=2, default=str))
        else:
            print("[]")
    else:
        if not args.top3:
            print_signals_table(signals)
        else:
            # Rank by score (dual strategy already has score column)
            picks = []
            if not signals.empty:
                df = signals.copy()
                # Enforce min_price
                if 'entry_price' in df.columns:
                    df = df[df['entry_price'] >= float(sel_cfg.get('min_price', 10.0))]
                # ADV $ filter using combined data last 60 days
                try:
                    bars = combined.copy()
                    bars['usd_vol'] = (bars['close'] * bars['volume']).astype(float)
                    adv = bars.groupby('symbol')['usd_vol'].rolling(60, min_periods=10).mean().reset_index(level=0, drop=True)
                    bars['adv_usd60'] = adv
                    bars_last = bars.sort_values('timestamp').groupby('symbol').tail(1)[['symbol','adv_usd60']]
                    df = pd.merge(df, bars_last, on='symbol', how='left')
                    df = df[df['adv_usd60'] >= float(args.min_adv_usd)]
                except Exception:
                    pass

                # Portfolio-aware filtering (Scheduler v2 - Phase 7)
                if args.portfolio_filter and PORTFOLIO_FILTERS_AVAILABLE and not df.empty:
                    try:
                        # Load current positions
                        current_positions = []
                        positions_path = Path(args.positions_file)
                        if positions_path.exists():
                            with open(positions_path, 'r') as f:
                                pos_data = json.load(f)
                                if isinstance(pos_data, list):
                                    current_positions = pos_data
                                elif isinstance(pos_data, dict):
                                    current_positions = pos_data.get('positions', [])

                        pre_count = len(df)
                        df = apply_portfolio_filters(
                            signals=df,
                            current_positions=current_positions,
                            price_data=combined,
                            equity=args.equity,
                            verbose=args.verbose,
                        )
                        post_count = len(df)

                        if args.verbose:
                            print(f"Portfolio filter: {pre_count} -> {post_count} signal(s)")
                        elif pre_count != post_count:
                            print(f"Portfolio filter: filtered to {post_count} signal(s) (from {pre_count})")

                    except Exception as e:
                        if args.verbose:
                            print(f"  [WARN] Portfolio filter failed: {e}")

                # Compute conf_score early for selection
                def base_conf_row(row: pd.Series) -> float:
                    if 'conf_score' in row and pd.notna(row['conf_score']):
                        try:
                            return float(row['conf_score'])
                        except Exception:
                            pass
                    score = float(row.get('score', 0.0))
                    # Heuristic normalization consistent with earlier
                    if score > 50:  # Turtle Soup (typically 100-300)
                        return min(score / 300.0, 1.0)
                    else:  # IBS_RSI (typically 0-25)
                        return min(score / 25.0, 1.0)
                df['conf_score'] = df.apply(base_conf_row, axis=1)

                # Apply Historical Edge Boost (symbol-specific) BEFORE selection
                try:
                    from config.settings_loader import get_setting
                    hedg_enabled = bool(get_setting("historical_edge.enabled", True))
                    cap_pp = float(get_setting("historical_edge.cap_pp", 15.0))
                except Exception:
                    hedg_enabled = True
                    cap_pp = 15.0

                if hedg_enabled and LLM_ANALYZER_AVAILABLE and not df.empty:
                    try:
                        analyzer = get_trade_analyzer()
                        def apply_boost(row: pd.Series) -> float:
                            base = float(row.get('conf_score', 0.0))
                            sym = str(row.get('symbol', ''))
                            strat = str(row.get('strategy', ''))
                            if not sym or not strat:
                                return base
                            # Boost is in percentage points (pp). Convert to fraction of 1.0.
                            boost_pp = analyzer.get_symbol_boost(strategy=strat, symbol=sym, cap_pp=cap_pp)
                            boost_frac = float(boost_pp) / 100.0
                            # Additive on [0,1] score; clamp to [0,1]
                            return max(0.0, min(1.0, base + boost_frac))
                        df['conf_score'] = df.apply(apply_boost, axis=1)
                    except Exception:
                        pass

                # Selection: enforce mix or pure top-3
                if args.top3_mix == 'ict2_ibs1':
                    ict_df = df[df['strategy'].astype(str).str.lower().isin(['turtle_soup'])].copy()
                    ibs_df = df[df['strategy'].astype(str).str.lower().isin(['ibs_rsi','ibs'])].copy()
                    # DETERMINISM FIX: Use stable sort with tie-breakers (timestamp, symbol)
                    ict_df = ict_df.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')
                    ibs_df = ibs_df.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')
                    parts = []
                    if not ict_df.empty:
                        parts.append(ict_df.head(2))
                    if not ibs_df.empty:
                        parts.append(ibs_df.head(1))
                    out_sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                    # Fill shortfalls with next-best overall
                    if len(out_sel) < 3:
                        remaining = df.copy()
                        if not out_sel.empty:
                            keycols = ['timestamp','symbol','side']
                            remaining = remaining.merge(out_sel[keycols], on=keycols, how='left', indicator=True)
                            remaining = remaining[remaining['_merge'] == 'left_only'].drop(columns=['_merge'])
                        # DETERMINISM FIX: Use stable sort with tie-breakers
                        remaining = remaining.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')
                        need = 3 - len(out_sel)
                        if need > 0 and not remaining.empty:
                            out_sel = pd.concat([out_sel, remaining.head(need)], ignore_index=True)
                    picks = [out_sel]
                else:
                    # DETERMINISM FIX: Use stable sort with tie-breakers
                    picks = [df.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort').head(3)]
            out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
            if out.empty:
                print_signals_table(signals)
            else:
                # Compute confidence score and write picks + trade of the day
                out = out.copy()
                # Ensure we have conf_score column populated
                if 'conf_score' not in out.columns:
                    out['conf_score'] = out.apply(base_conf_row, axis=1)

                # Ensure Top-3 by filling from highest-confidence leftovers if requested
                if args.ensure_top3 and len(out) < 3:
                    left = df.copy()
                    # Remove already picked rows
                    if not out.empty:
                        keycols = ['timestamp','symbol','side']
                        left = left.merge(out[keycols], on=keycols, how='left', indicator=True)
                        left = left[left['_merge'] == 'left_only'].drop(columns=['_merge'])
                    # Attach confidence to leftovers (prefer ML, else heuristic)
                    if 'conf_score' not in left.columns:
                        def base_conf(r):
                            score = float(r.get('score', 0.0))
                            if score > 50:  # TurtleSoup
                                return min(score / 300.0, 1.0)
                            else:  # IBS_RSI
                                return min(score / 25.0, 1.0)
                        left['conf_score'] = left.apply(base_conf, axis=1)
                    # DETERMINISM FIX: Use stable sort with tie-breakers
                    left = left.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')
                    need = 3 - len(out)
                    if need > 0 and not left.empty:
                        out = pd.concat([out, left.head(need)], ignore_index=True)

                # Update with live prices if enabled and market is open
                if live_data_enabled and market_is_open and not out.empty:
                    print("\n[LIVE DATA] Fetching real-time prices from Alpaca...")
                    pick_symbols = out['symbol'].unique().tolist()
                    live_quotes = fetch_multi_quotes(pick_symbols)

                    if live_quotes:
                        # Store original (EOD) entry price for comparison
                        out['eod_entry_price'] = out['entry_price'].copy()

                        def update_with_live_price(row):
                            sym = row['symbol']
                            if sym in live_quotes:
                                quote = live_quotes[sym]
                                side = str(row.get('side', '')).lower()
                                # For long: use ask (what we'd pay)
                                # For short: use bid (what we'd receive)
                                if side == 'long' and quote.get('ask_price'):
                                    return quote['ask_price']
                                elif side == 'short' and quote.get('bid_price'):
                                    return quote['bid_price']
                            return row['entry_price']

                        out['entry_price'] = out.apply(update_with_live_price, axis=1)

                        # Print comparison
                        print("\n  Symbol  | EOD Price | Live Price | Change")
                        print("  " + "-" * 45)
                        for _, row in out.iterrows():
                            sym = row['symbol']
                            eod = row['eod_entry_price']
                            live = row['entry_price']
                            change = ((live - eod) / eod * 100) if eod > 0 else 0
                            direction = "+" if change >= 0 else ""
                            print(f"  {sym:<7} | ${eod:>8.2f} | ${live:>9.2f} | {direction}{change:.2f}%")
                    else:
                        print("  [WARN] Could not fetch live quotes")

                # Write Top 3 picks
                picks_path = Path(args.out_picks)
                picks_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(picks_path, index=False)

                # Choose Trade of the Day (highest confidence)
                # DETERMINISM FIX: Use stable sort with tie-breakers
                totd = out.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort').head(1)
                approve_totd = True
                if args.ml and not totd.empty:
                    try:
                        approve_totd = float(totd.iloc[0]['conf_score']) >= float(args.min_conf)
                    except Exception:
                        approve_totd = True
                totd_path = Path(args.out_totd)
                if approve_totd and not totd.empty:
                    totd.to_csv(totd_path, index=False)
                else:
                    # Write empty placeholder to indicate skip
                    pd.DataFrame(columns=out.columns).to_csv(totd_path, index=False)

                # Show signal date prominently to prevent stale signal confusion
                signal_date = end_date  # The date these signals are generated for
                print(f"\n{'='*60}")
                print(f"TOP 3 PICKS - SIGNAL DATE: {signal_date}")
                print(f"{'='*60}")
                print("  NOTE: These signals are valid for the NEXT trading day")
                print("-" * 60)
                print(out[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                print(f"\nWrote: {picks_path}")
                print("\nTRADE OF THE DAY")
                print("-" * 60)
                if approve_totd and not totd.empty:
                    print(totd[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                    print(f"\nWrote: {totd_path}")
                else:
                    print("No TOTD due to low confidence or no picks.")

                # === LLM NARRATIVE ANALYSIS ===
                # Generate human-like reasoning for picks using Claude
                if args.narrative and LLM_ANALYZER_AVAILABLE and not out.empty:
                    print("\n" + "=" * 60)
                    print("GENERATING LLM ANALYSIS (Claude Human-Like Reasoning)")
                    print("=" * 60)

                    try:
                        analyzer = get_trade_analyzer()

                        # Build market context
                        regime = "NEUTRAL"
                        regime_conf = 0.5
                        vix = 20.0
                        sentiment_data = {}

                        # Get regime from HMM detector (ML) or SPY SMA200 (fallback)
                        if hmm_regime_state is not None:
                            # Use ML-powered HMM regime detection
                            regime = hmm_regime_state.regime.value
                            regime_conf = hmm_regime_state.confidence
                        elif spy_bars is not None and not spy_bars.empty:
                            try:
                                spy_close = spy_bars['close'].iloc[-1]
                                spy_sma200 = spy_bars['close'].rolling(200).mean().iloc[-1]
                                if spy_close > spy_sma200 * 1.02:
                                    regime = "BULL"
                                    regime_conf = min(0.7 + (spy_close / spy_sma200 - 1) * 2, 0.95)
                                elif spy_close < spy_sma200 * 0.98:
                                    regime = "BEAR"
                                    regime_conf = min(0.7 + (1 - spy_close / spy_sma200) * 2, 0.95)
                                else:
                                    regime = "NEUTRAL"
                                    regime_conf = 0.6
                            except Exception:
                                pass

                        # Get sentiment if available
                        try:
                            end_day = pd.to_datetime(end_date).date().isoformat()
                            sent_df = load_daily_cache(end_day)
                            if not sent_df.empty and 'sent_mean' in sent_df.columns:
                                sentiment_data = {
                                    'compound': float(sent_df['sent_mean'].mean()),
                                    'positive': 0.0,
                                    'negative': 0.0,
                                }
                        except Exception:
                            sentiment_data = {'compound': 0.0}

                        market_context = {
                            'regime': regime,
                            'regime_confidence': regime_conf,
                            'vix': vix,
                            'sentiment': sentiment_data,
                            'spy_position': f"{'above' if regime == 'BULL' else 'below' if regime == 'BEAR' else 'near'} SMA(200)",
                        }

                        # Get news articles if available
                        news_articles = []
                        try:
                            from altdata.news_processor import get_news_processor
                            news_proc = get_news_processor()
                            symbols = out['symbol'].tolist() if 'symbol' in out.columns else []
                            articles = news_proc.fetch_news(symbols=symbols, limit=10)
                            news_articles = [a.to_dict() for a in articles] if articles else []
                        except Exception:
                            pass

                        # Generate full insight report
                        totd_dict = totd.iloc[0].to_dict() if not totd.empty and approve_totd else None
                        report = analyzer.generate_daily_insight_report(
                            picks=out,
                            totd=totd_dict,
                            market_context=market_context,
                            news_articles=news_articles,
                            sentiment=sentiment_data,
                            all_signals=signals,
                        )

                        # Save insights to JSON
                        insights_path = Path(args.out_insights)
                        insights_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(insights_path, 'w', encoding='utf-8') as f:
                            json.dump(report.to_dict(), f, indent=2, default=str)
                        print(f"\nInsights saved: {insights_path}")

                        # Print narrative output
                        print("\n" + "-" * 60)
                        print("MARKET SUMMARY")
                        print("-" * 60)
                        print(report.market_summary)

                        print("\n" + "-" * 60)
                        print("REGIME ASSESSMENT")
                        print("-" * 60)
                        print(report.regime_assessment)

                        if report.top3_narratives:
                            print("\n" + "-" * 60)
                            print("TOP 3 PICKS - REASONING")
                            print("-" * 60)
                            for i, narr in enumerate(report.top3_narratives, 1):
                                print(f"\n#{i} {narr.symbol} ({narr.strategy}) [{narr.confidence_rating}]")
                                print(f"   {narr.narrative}")
                                if narr.conviction_reasons:
                                    print("   Conviction:")
                                    for reason in narr.conviction_reasons[:2]:
                                        print(f"     - {reason}")
                                if narr.risk_factors:
                                    print("   Risks:")
                                    for risk in narr.risk_factors[:2]:
                                        print(f"     - {risk}")

                        # Generate comprehensive TOTD report if we have a TOTD
                        if totd_dict:
                            comp_report = analyzer.generate_comprehensive_totd_report(
                                totd=totd_dict,
                                market_context=market_context,
                                price_data=None,  # Will fetch automatically
                            )

                            print("\n" + "=" * 70)
                            print(f"COMPREHENSIVE TRADE OF THE DAY - {comp_report.symbol}")
                            print("=" * 70)

                            # Core metrics
                            print(f"\nStrategy: {comp_report.strategy} | Entry: ${comp_report.entry_price:.2f} | Stop: ${comp_report.stop_loss:.2f}")
                            print(f"Risk: ${comp_report.risk_per_share:.2f} ({comp_report.stop_distance_pct:.1f}%) | R:R: {comp_report.risk_reward_ratio:.2f}:1")
                            print(f"Overall Confidence: {comp_report.overall_confidence:.0f}% | Method: {comp_report.generation_method}")

                            # Confidence breakdown
                            print("\n--- CONFIDENCE BREAKDOWN ---")
                            for k, v in comp_report.confidence_breakdown.items():
                                # Normalize values for display
                                if k == 'symbol_boost':
                                    # Symbol boost is percentage points (pp), capped at +/-15
                                    v_pp = max(-15.0, min(15.0, float(v)))
                                    filled = int(min(1.0, abs(v_pp) / 15.0) * 20)
                                    bar = "#" * filled + "-" * (20 - filled)
                                    sign = "+" if v_pp > 0 else ("-" if v_pp < 0 else "")
                                    print(f"  {k:18}: [{bar}] {sign}{abs(v_pp):.0f} pp")
                                else:
                                    v_pct = max(0.0, min(100.0, float(v)))
                                    filled = int(v_pct / 5.0)
                                    bar = "#" * filled + "-" * (20 - filled)
                                    print(f"  {k:18}: [{bar}] {v_pct:.0f}%")

                            # Symbol-specific historical stats (REAL BACKTEST DATA)
                            h = comp_report.historical
                            if h and h.symbol_stats:
                                s = h.symbol_stats
                                print("\n--- SYMBOL-SPECIFIC HISTORICAL PERFORMANCE (10yr backtest) ---")
                                print(f"  {s.symbol} with {comp_report.strategy}:")
                                print(f"    Total Trades: {s.total_trades} | Wins: {s.wins} | Losses: {s.losses}")
                                print(f"    Win Rate: {s.win_rate:.1f}% (vs {h.win_rate:.1f}% overall)")
                                if s.confidence_boost >= 0:
                                    print(f"    Confidence Boost: +{s.confidence_boost*100:.1f}% (symbol outperforms)")
                                else:
                                    print(f"    Confidence Boost: {s.confidence_boost*100:.1f}% (symbol underperforms)")
                                print(f"    Avg Win: +{s.avg_win_pct:.2f}% | Avg Loss: -{s.avg_loss_pct:.2f}%")
                                print(f"    Profit Factor: {s.profit_factor:.2f} | Total P&L: {s.total_pnl_pct:+.1f}%")
                                print(f"    Best Trade: +{s.best_trade_pct:.1f}% | Worst: {s.worst_trade_pct:.1f}%")
                                if s.recent_trades:
                                    print(f"    Recent {s.symbol} trades:")
                                    for t in s.recent_trades[-3:]:
                                        print(f"      - {t.get('buy_date', 'N/A')[:10]}: {t.get('pnl_pct', 0):+.2f}%")
                                print(f"    Data Source: {h.data_source.upper()}")

                            # All analysis sections
                            print("\n--- EXECUTIVE SUMMARY ---")
                            print(comp_report.executive_summary)

                            print("\n--- WHY THIS TRADE? ---")
                            print(comp_report.why_this_trade)

                            print("\n--- HISTORICAL EDGE ANALYSIS ---")
                            print(comp_report.historical_edge_analysis)

                            print("\n--- TECHNICAL ANALYSIS ---")
                            print(comp_report.technical_analysis)

                            print("\n--- NEWS IMPACT ---")
                            print(comp_report.news_impact_analysis)

                            print("\n--- RISK ANALYSIS ---")
                            print(comp_report.risk_analysis)

                            print("\n--- EXECUTION PLAN ---")
                            print(comp_report.execution_plan)

                            print("\n--- POSITION SIZING ---")
                            print(comp_report.position_sizing)

                            print("\n--- RISK WARNINGS ---")
                            for w in comp_report.risk_warnings:
                                print(f"  [!] {w}")

                            print("\n--- KEY LEVELS TO WATCH ---")
                            for l in comp_report.key_levels_to_watch:
                                print(f"  > {l}")

                            # Save comprehensive report
                            comp_path = Path("logs/comprehensive_totd.json")
                            with open(comp_path, 'w', encoding='utf-8') as f:
                                json.dump(comp_report.to_dict(), f, indent=2, default=str)
                            print(f"\n[Comprehensive TOTD saved to {comp_path}]")

                        elif report.totd_deep_analysis:
                            print("\n" + "-" * 60)
                            print("TRADE OF THE DAY - DEEP ANALYSIS")
                            print("-" * 60)
                            analysis = report.totd_deep_analysis
                            if len(analysis) > 600:
                                print(analysis[:600] + "...")
                            else:
                                print(analysis)

                        if report.key_findings:
                            print("\n" + "-" * 60)
                            print("KEY FINDINGS")
                            print("-" * 60)
                            for finding in report.key_findings[:5]:
                                print(f"  - {finding}")

                        if report.sentiment_interpretation:
                            print("\n" + "-" * 60)
                            print("SENTIMENT INTERPRETATION")
                            print("-" * 60)
                            print(report.sentiment_interpretation)

                        if report.risk_warnings:
                            print("\n" + "-" * 60)
                            print("RISK WARNINGS")
                            print("-" * 60)
                            for warn in report.risk_warnings:
                                print(f"  [!] {warn}")

                        if report.opportunities:
                            print("\n" + "-" * 60)
                            print("OPPORTUNITIES")
                            print("-" * 60)
                            for opp in report.opportunities:
                                print(f"  [+] {opp}")

                        print("\n" + "-" * 60)
                        gen_method = report.generation_method
                        print(f"Analysis generated via: {gen_method.upper()} ({report.llm_model})")
                        print("-" * 60)

                    except Exception as e:
                        print(f"\n  [WARN] LLM analysis failed: {e}", file=sys.stderr)
                        if args.verbose:
                            import traceback
                            traceback.print_exc()

                elif args.narrative and not LLM_ANALYZER_AVAILABLE:
                    print("\n  [WARN] LLM analyzer not available (import failed)")

    # Log signals (ensure conf_score is included for telemetry)
    if not args.no_log and not signals.empty:
        # Add conf_score if not already present (for confidence telemetry)
        if 'conf_score' not in signals.columns:
            signals = signals.copy()
            signals['conf_score'] = signals.apply(compute_conf_score, axis=1)
        log_signals(signals, scan_id)
        print(f"\nSignals logged to: {SIGNALS_LOG}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Scan complete: {len(signals)} signal(s) generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
