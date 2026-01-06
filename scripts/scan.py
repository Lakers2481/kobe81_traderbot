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

# Markov Chain Predictor (next-day direction scoring)
try:
    from ml_advanced.markov_chain import MarkovAssetScorer, MarkovPredictor
    MARKOV_AVAILABLE = True
except ImportError:
    MARKOV_AVAILABLE = False

# Options Signal Generator (calls/puts from equity signals)
try:
    from scanner.options_signals import generate_options_signals, OPTIONS_AVAILABLE
except ImportError:
    OPTIONS_AVAILABLE = False

# Crypto Signal Generator (BTC, ETH, etc.)
try:
    from scanner.crypto_signals import (
        generate_crypto_signals,
        scan_crypto,
        fetch_crypto_universe_data,
        DEFAULT_CRYPTO_UNIVERSE,
        CRYPTO_DATA_AVAILABLE,
    )
except ImportError:
    CRYPTO_DATA_AVAILABLE = False
    DEFAULT_CRYPTO_UNIVERSE = []

# Unified Signal Enrichment Pipeline (ALL COMPONENTS WIRED)
try:
    from pipelines.unified_signal_enrichment import (
        run_full_enrichment,
        get_unified_pipeline,
        ComponentRegistry,
    )
    UNIFIED_PIPELINE_AVAILABLE = True
except ImportError:
    UNIFIED_PIPELINE_AVAILABLE = False


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

    PRE-MARKET LOGIC (FIX 2026-01-05):
    - Before 9:30 AM ET: Use previous trading day + PREVIEW mode
    - This ensures we have data when running scans before market open

    WHY PREVIEW ON WEEKENDS:
    - Normal mode uses .shift(1) for lookahead safety (checks PREVIOUS bar)
    - On weekends, Friday is the last bar - but shift(1) would check Thursday
    - Preview mode uses CURRENT bar (Friday's values) so we see what triggers Monday
    """
    if reference_date is None:
        reference_date = datetime.now()

    weekday = reference_date.weekday()  # Monday=0, Sunday=6
    is_weekend = weekday >= 5  # Saturday=5, Sunday=6

    # Check if before market open (9:30 AM ET)
    # Note: This assumes the system is running in ET timezone
    is_pre_market = reference_date.hour < 9 or (reference_date.hour == 9 and reference_date.minute < 30)

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
                # Today is a trading day
                if is_pre_market:
                    # Before market open: use previous trading day
                    # Find the trading day before today
                    if len(schedule) > 1:
                        prev_trading = schedule.index[-2]
                        prev_trading_str = prev_trading.strftime('%Y-%m-%d')
                        return prev_trading_str, True, f"PRE-MARKET: Using {prev_trading_str} + PREVIEW mode (market opens at 9:30 AM)"
                    else:
                        return last_trading_str, True, f"PRE-MARKET: Using {last_trading_str} + PREVIEW mode"
                else:
                    # Market is open or has closed: use today + normal mode
                    return ref_str, False, f"WEEKDAY: Using today ({ref_str}) + NORMAL mode (fresh data)"
            else:
                # Weekday but today is a holiday: use last trading day + preview
                return last_trading_str, True, f"HOLIDAY: Using {last_trading_str} + PREVIEW mode"
        else:
            # Fallback
            return _fallback_trading_day(reference_date, is_weekend, is_pre_market)

    except ImportError:
        return _fallback_trading_day(reference_date, is_weekend, is_pre_market)


def _fallback_trading_day(reference_date: datetime, is_weekend: bool, is_pre_market: bool = False) -> tuple[str, bool, str]:
    """Fallback when pandas_market_calendars not available."""
    if is_weekend:
        # Go back to Friday
        days_back = (reference_date.weekday() - 4) % 7
        if days_back == 0:
            days_back = 7 if reference_date.weekday() != 4 else 0
        last_friday = reference_date - timedelta(days=days_back)
        return last_friday.strftime('%Y-%m-%d'), True, f"WEEKEND: Using Friday ({last_friday.strftime('%Y-%m-%d')}) + PREVIEW"
    elif is_pre_market:
        # Before market open: go back 1 day (may not be accurate for holidays)
        prev_day = reference_date - timedelta(days=1)
        # Skip weekend
        if prev_day.weekday() >= 5:
            days_back = prev_day.weekday() - 4
            prev_day = prev_day - timedelta(days=days_back)
        return prev_day.strftime('%Y-%m-%d'), True, f"PRE-MARKET: Using {prev_day.strftime('%Y-%m-%d')} + PREVIEW mode"
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
        # Crypto has no min_price filter (BTC is $40k+)
        has_crypto = 'asset_class' in data.columns and (data['asset_class'] == 'CRYPTO').any()
        min_price = 0.0 if has_crypto else float(sel_cfg.get('min_price', 10.0))
        params = DualStrategyParams(min_price=min_price)
        scanner = DualStrategyScanner(params, preview_mode=preview_mode)

        # Generate signals (IBS+RSI + Turtle Soup combined)
        signals = scanner.generate_signals(data)

        if signals.empty:
            return pd.DataFrame()

        # === ASSET-AWARE: Propagate asset_class to signals ===
        # Build lookup from source data
        if 'asset_class' in data.columns:
            asset_lookup = data.groupby('symbol')['asset_class'].first().to_dict()
            signals['asset_class'] = signals['symbol'].map(asset_lookup).fillna('EQUITY')
        else:
            signals['asset_class'] = 'EQUITY'

        # === ASSET-AWARE: Apply regime/earnings filters ONLY to equities ===
        # Crypto doesn't have SPY regime correlation or earnings
        if apply_filters and spy_bars is not None and not spy_bars.empty and not signals.empty:
            # Separate by asset class
            equity_mask = signals['asset_class'] == 'EQUITY'
            equity_signals = signals[equity_mask].copy()
            crypto_signals = signals[~equity_mask].copy()

            # Apply regime filter only to equities
            if not equity_signals.empty:
                try:
                    equity_signals = filter_signals_by_regime(
                        equity_signals, spy_bars, get_regime_filter_config()
                    )
                except Exception:
                    pass

            # Apply earnings filter only to equities
            if not equity_signals.empty:
                try:
                    recs = equity_signals.to_dict('records')
                    equity_signals = pd.DataFrame(filter_signals_by_earnings(recs))
                except Exception:
                    pass

            # Recombine: filtered equities + unfiltered crypto
            all_parts = []
            if not equity_signals.empty:
                all_parts.append(equity_signals)
            if not crypto_signals.empty:
                all_parts.append(crypto_signals)
            signals = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

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
KOBE STANDARD PIPELINE: 900 -> 5 -> 2
=====================================
This is the ONLY way to trade. No exceptions.

  Step 1: Scan 900 stocks (full universe)
  Step 2: Filter to Top 5 (STUDY - follow, analyze, test, understand)
  Step 3: Trade Top 2 (EXECUTE - best 2 out of the 5)

CANONICAL COMMAND:
  python scripts/scan.py --cap 900 --deterministic --top5

OUTPUT FILES:
  logs/daily_top5.csv  - Top 5 to STUDY (follow, analyze, test)
  logs/tradeable.csv   - Top 2 to TRADE (what we actually execute)

Examples:
  python scripts/scan.py --cap 900 --deterministic --top5     # Full pipeline
  python scripts/scan.py --strategy ibs_rsi                   # Only IBS+RSI signals
  python scripts/scan.py --markov --markov-prefilter 100      # With Markov pre-filter
  python scripts/scan.py --json                               # Output as JSON
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
    ap.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel data fetch workers (default: 5, use 1 for debugging)",
    )
    ap.add_argument("--top3", action="store_true", help="Select Top-3 picks and write logs/daily_picks.csv")
    ap.add_argument(
        "--top3-mix",
        type=str,
        choices=["ict2_ibs1", "pure"],
        default="ict2_ibs1",
        help="Top-3 selection rule: ict2_ibs1 (default) enforces 2x ICT + 1x IBS; pure takes the highest 3 by confidence",
    )
    # === 900 -> 5 -> 2 PIPELINE (Kobe Standard Flow) ===
    ap.add_argument(
        "--top5",
        action="store_true",
        help="KOBE STANDARD: Filter to Top-5 for STUDY (follow, analyze, test) then trade Top-2",
    )
    ap.add_argument(
        "--trade-top-n",
        type=int,
        default=2,
        help="How many of the Top-5 to actually TRADE (default: 2 = KOBE STANDARD)",
    )
    ap.add_argument(
        "--out-top5",
        type=str,
        default=str(ROOT / 'logs' / 'daily_top5.csv'),
        help="Output CSV for Top-5 candidates",
    )
    ap.add_argument(
        "--out-tradeable",
        type=str,
        default=str(ROOT / 'logs' / 'tradeable.csv'),
        help="Output CSV for the signals to actually trade (top N of Top-3)",
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
        help="Enforce 100%% deterministic output order (stable sorts, seeded RNG)",
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
    # === Markov Chain Integration ===
    ap.add_argument(
        "--markov",
        action="store_true",
        help="Enable Markov chain scoring for signal confidence boost",
    )
    ap.add_argument(
        "--markov-prefilter",
        type=int,
        default=0,
        metavar="N",
        help="Pre-filter universe to top N stocks by Markov pi(Up) before scanning (0=disabled)",
    )
    ap.add_argument(
        "--markov-min-pi-up",
        type=float,
        default=0.35,
        help="Minimum stationary pi(Up) for Markov pre-filter (default: 0.35)",
    )
    # === Options Integration (calls/puts from equity signals) ===
    ap.add_argument(
        "--options",
        action="store_true",
        help="Generate options signals (calls/puts) from equity signals",
    )
    ap.add_argument(
        "--options-delta",
        type=float,
        default=0.30,
        help="Target delta for options strikes (default: 0.30 = 30-delta)",
    )
    ap.add_argument(
        "--options-dte",
        type=int,
        default=21,
        help="Target days to expiration for options (default: 21)",
    )
    ap.add_argument(
        "--options-max",
        type=int,
        default=3,
        help="Maximum options signals to generate (default: 3)",
    )
    # === Crypto Integration (BTC, ETH, etc.) ===
    ap.add_argument(
        "--crypto",
        action="store_true",
        help="Include crypto signals (BTC, ETH, SOL, etc.) in scan output",
    )
    ap.add_argument(
        "--crypto-max",
        type=int,
        default=3,
        help="Maximum crypto signals to generate (default: 3)",
    )
    # === UNIFIED ENRICHMENT PIPELINE (ALL 250+ COMPONENTS) ===
    ap.add_argument(
        "--unified",
        action="store_true",
        default=True,
        help="Enable unified enrichment pipeline (ALL AI/ML/Data components wired). ON by default.",
    )
    ap.add_argument(
        "--no-unified",
        action="store_true",
        help="Disable unified enrichment pipeline (use legacy flow)",
    )
    ap.add_argument(
        "--out-unified",
        type=str,
        default=str(ROOT / 'logs' / 'unified_signals.csv'),
        help="Output CSV for unified enriched signals",
    )
    ap.add_argument(
        "--out-thesis",
        type=str,
        default=str(ROOT / 'logs' / 'trade_thesis'),
        help="Output directory for TOP 2 trade thesis markdown files",
    )
    ap.add_argument(
        "--out-top2",
        type=str,
        default=str(ROOT / 'logs' / 'top2_trade.csv'),
        help="Output CSV for TOP 2 trades with full analysis",
    )
    args = ap.parse_args()

    # Handle --no-* override flags for ML and cognitive defaults
    if args.no_ml:
        args.ml = False
    if args.no_cognitive:
        args.cognitive = False
    if args.no_unified:
        args.unified = False

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
    # Use --workers arg (default 5), capped at symbol count
    max_workers = min(args.workers, len(symbols)) if hasattr(args, 'workers') else min(5, len(symbols))

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

    # Combine all equity data
    combined = pd.concat(all_data, ignore_index=True)

    # Add asset_class column to track source (EQUITY vs CRYPTO)
    combined['asset_class'] = 'EQUITY'
    print(f"Equity bars: {len(combined):,}")

    # === CRYPTO DATA FETCHING (EARLY - before AI pipeline) ===
    # Crypto goes through the SAME pipeline as equities for unified ranking
    if args.crypto and CRYPTO_DATA_AVAILABLE and DEFAULT_CRYPTO_UNIVERSE:
        try:
            print(f"\nFetching crypto data ({len(DEFAULT_CRYPTO_UNIVERSE[:8])} pairs)...")
            crypto_bars = fetch_crypto_universe_data(
                symbols=DEFAULT_CRYPTO_UNIVERSE[:8],
                start=start_date,
                end=end_date,
                cache_dir=CACHE_DIR,
            )
            if not crypto_bars.empty:
                # Add asset_class column for crypto
                crypto_bars['asset_class'] = 'CRYPTO'
                # Combine with equity data
                combined = pd.concat([combined, crypto_bars], ignore_index=True)
                print(f"Added {len(crypto_bars):,} crypto bars")
            else:
                print("  No crypto data fetched")
        except Exception as e:
            print(f"  [WARN] Crypto fetch failed: {e}")
    elif args.crypto and not CRYPTO_DATA_AVAILABLE:
        print("[WARN] --crypto requested but crypto provider not available")

    print(f"Total bars (equity + crypto): {len(combined):,}")

    # === MARKOV PRE-FILTER (optional) ===
    # Rank stocks by stationary pi(Up) and filter to top N before detailed scan
    markov_scores = {}  # Cache for later use in signal scoring
    if MARKOV_AVAILABLE and args.markov_prefilter > 0:
        try:
            print(f"\n[MARKOV] Pre-filtering to top {args.markov_prefilter} by pi(Up)...")
            scorer = MarkovAssetScorer(
                n_states=3,
                lookback_days=252,
                classification_method="threshold",
            )

            # Build returns dict from combined data
            returns_dict = {}
            for sym in combined['symbol'].unique():
                sym_data = combined[combined['symbol'] == sym].sort_values('timestamp')
                if len(sym_data) >= 60:  # Need at least 60 days
                    closes = sym_data['close'].values
                    returns = pd.Series(closes).pct_change().dropna()
                    returns_dict[sym] = returns

            if returns_dict:
                all_syms = list(returns_dict.keys())
                scored_df = scorer.score_universe(all_syms, returns_dict)

                if not scored_df.empty:
                    # Store scores for later signal scoring
                    for _, row in scored_df.iterrows():
                        markov_scores[row['symbol']] = {
                            'pi_up': row['pi_up'],
                            'p_up_today': row['p_up_today'],
                            'current_state': row['current_state'],
                            'composite_score': row['composite_score'],
                        }

                    # Filter to top N
                    top_symbols = scorer.filter_top_n(
                        scored_df,
                        n=args.markov_prefilter,
                        min_score=args.markov_min_pi_up,
                    )

                    if top_symbols:
                        pre_count = len(combined['symbol'].unique())
                        combined = combined[combined['symbol'].isin(top_symbols)]
                        post_count = len(combined['symbol'].unique())
                        print(f"[MARKOV] Pre-filtered: {pre_count} -> {post_count} symbols (top pi_up >= {args.markov_min_pi_up:.2f})")

                        # Show top 5 by pi(Up)
                        if args.verbose:
                            top5 = scored_df.head(5)
                            for _, r in top5.iterrows():
                                print(f"  {r['symbol']}: pi(Up)={r['pi_up']:.3f}, P(Up|now)={r['p_up_today']:.3f}")
                    else:
                        print("[MARKOV] No symbols passed Markov filter - using all")
                else:
                    print("[MARKOV] Scoring failed - using all symbols")
            else:
                print("[MARKOV] Insufficient data for Markov scoring - using all symbols")
        except Exception as e:
            print(f"[MARKOV] Pre-filter failed: {e} - using all symbols")
    elif args.markov_prefilter > 0 and not MARKOV_AVAILABLE:
        print("[WARN] --markov-prefilter requested but Markov module not available")

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

    # === PROPAGATE ASSET_CLASS TO SIGNALS ===
    # Ensure signals have asset_class column from source data
    if not signals.empty and 'asset_class' not in signals.columns:
        # Build asset_class lookup from combined data
        asset_class_map = {}
        if 'asset_class' in combined.columns:
            for sym in combined['symbol'].unique():
                sym_class = combined[combined['symbol'] == sym]['asset_class'].iloc[0]
                asset_class_map[sym] = sym_class

        # Apply to signals
        if asset_class_map:
            signals['asset_class'] = signals['symbol'].map(asset_class_map).fillna('EQUITY')
        else:
            signals['asset_class'] = 'EQUITY'

        # Report asset class mix
        if args.verbose and 'asset_class' in signals.columns:
            by_class = signals.groupby('asset_class').size()
            print(f"  Signals by asset class: {dict(by_class)}")

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

    # === OPTIONS GENERATION (AFTER QUALITY GATE) ===
    # Generate CALL and PUT options from enriched equity signals
    # Options inherit adjusted conf_score from parent (see options_signals.py)
    # This ensures options go through remaining AI stages (Markov, ML, Cognitive)
    all_bars = combined  # Save reference for options generation
    if args.options and OPTIONS_AVAILABLE and not signals.empty:
        try:
            # Only generate options from EQUITY signals (not crypto)
            equity_signals = signals[signals.get('asset_class', 'EQUITY') == 'EQUITY'].copy()

            if not equity_signals.empty:
                print(f"\n[OPTIONS] Generating calls/puts from {len(equity_signals)} equity signal(s)...")
                options_df = generate_options_signals(
                    equity_signals=equity_signals,
                    price_data=all_bars,
                    max_signals=args.options_max * 2,  # calls + puts
                    target_delta=args.options_delta,
                    target_dte=args.options_dte,
                )

                if not options_df.empty:
                    # Add asset_class if not present
                    if 'asset_class' not in options_df.columns:
                        options_df['asset_class'] = 'OPTIONS'

                    # Add to signal pool (will go through Markov/ML/Cognitive)
                    signals = pd.concat([signals, options_df], ignore_index=True)

                    calls = len(options_df[options_df.get('option_type', '') == 'CALL'])
                    puts = len(options_df[options_df.get('option_type', '') == 'PUT'])
                    print(f"[OPTIONS] Generated {len(options_df)} options: {calls} CALLs, {puts} PUTs")

                    if args.verbose:
                        for _, opt in options_df.iterrows():
                            adj_conf = opt.get('conf_score', 0)
                            print(f"  {opt['symbol']} {opt.get('option_type', '?')} @ ${opt.get('strike', 0):.2f}: conf={adj_conf:.4f}")
                else:
                    print("[OPTIONS] No options generated (check volatility/premium constraints)")

        except Exception as e:
            print(f"[OPTIONS] Generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif args.options and not OPTIONS_AVAILABLE:
        print("[WARN] --options requested but options module not available")

    # === MARKOV CHAIN SCORING ===
    # Add Markov metrics to signals for confidence adjustment
    if MARKOV_AVAILABLE and args.markov and not signals.empty:
        try:
            # If we don't have cached scores, compute them now
            if not markov_scores:
                scorer = MarkovAssetScorer(n_states=3, lookback_days=252)
                returns_dict = {}
                for sym in signals['symbol'].unique():
                    sym_data = combined[combined['symbol'] == sym].sort_values('timestamp')
                    if len(sym_data) >= 60:
                        closes = sym_data['close'].values
                        returns = pd.Series(closes).pct_change().dropna()
                        returns_dict[sym] = returns
                if returns_dict:
                    scored_df = scorer.score_universe(list(returns_dict.keys()), returns_dict)
                    for _, row in scored_df.iterrows():
                        markov_scores[row['symbol']] = {
                            'pi_up': row['pi_up'],
                            'p_up_today': row['p_up_today'],
                            'current_state': row['current_state'],
                            'composite_score': row['composite_score'],
                        }

            # Add Markov columns to signals
            markov_pi_up = []
            markov_p_up_today = []
            markov_agrees = []

            for _, sig in signals.iterrows():
                sym = sig['symbol']
                side = sig.get('side', 'long')

                if sym in markov_scores:
                    m = markov_scores[sym]
                    markov_pi_up.append(m['pi_up'])
                    markov_p_up_today.append(m['p_up_today'])
                    # Markov agrees if: long signal + high P(Up), or short signal + low P(Up)
                    if side == 'long':
                        agrees = m['p_up_today'] >= 0.40  # Above 40% = Markov agrees with long
                    else:
                        agrees = m['p_up_today'] <= 0.30  # Below 30% = Markov agrees with short
                    markov_agrees.append(agrees)
                else:
                    markov_pi_up.append(0.33)  # Neutral
                    markov_p_up_today.append(0.33)
                    markov_agrees.append(False)

            signals['markov_pi_up'] = markov_pi_up
            signals['markov_p_up_today'] = markov_p_up_today
            signals['markov_agrees'] = markov_agrees

            # Report Markov agreement
            agree_count = sum(markov_agrees)
            if args.verbose:
                print(f"[MARKOV] {agree_count}/{len(signals)} signals have Markov agreement")
                for _, sig in signals.iterrows():
                    sym = sig['symbol']
                    if sym in markov_scores:
                        m = markov_scores[sym]
                        agree_str = "YES" if sig['markov_agrees'] else "NO"
                        print(f"  {sym}: pi(Up)={m['pi_up']:.3f}, P(Up|now)={m['p_up_today']:.3f}, agrees={agree_str}")

        except Exception as e:
            if args.verbose:
                print(f"[MARKOV] Signal scoring failed: {e}")
    elif args.markov and not MARKOV_AVAILABLE:
        print("[WARN] --markov requested but Markov module not available")

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

    # === UNIFIED ENRICHMENT PIPELINE (ALL 250+ COMPONENTS) ===
    # This runs ALL AI/ML/Data components through 18 stages to produce
    # fully-enriched signals and comprehensive trade theses for TOP 2
    enriched_signals = []
    top2_theses = []

    if args.unified and UNIFIED_PIPELINE_AVAILABLE and not signals.empty:
        print("\n" + "=" * 70)
        print("UNIFIED ENRICHMENT PIPELINE - ALL COMPONENTS WIRED")
        print("=" * 70)

        try:
            # Run the full enrichment pipeline
            enriched_signals, top2_theses = run_full_enrichment(
                signals=signals,
                price_data=combined,
                spy_data=spy_bars,
                verbose=args.verbose,
            )

            print(f"\n[UNIFIED PIPELINE] Enriched {len(enriched_signals)} signals through 18 stages")
            print(f"[UNIFIED PIPELINE] Generated {len(top2_theses)} comprehensive trade theses")

            # Save enriched signals to CSV
            if enriched_signals:
                unified_path = Path(args.out_unified)
                unified_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert to DataFrame
                enriched_df = pd.DataFrame([s.to_dict() for s in enriched_signals])
                enriched_df.to_csv(unified_path, index=False)
                print(f"[UNIFIED PIPELINE] Wrote enriched signals: {unified_path}")

                # Get TOP 5 for study
                top5_unified = enriched_df.head(5)
                top5_path = Path(args.out_unified).parent / 'top5_unified.csv'
                top5_unified.to_csv(top5_path, index=False)
                print(f"[UNIFIED PIPELINE] Wrote TOP 5 for study: {top5_path}")

                # Get TOP 2 for trading
                top2_df = enriched_df.head(2)
                top2_path = Path(args.out_top2)
                top2_path.parent.mkdir(parents=True, exist_ok=True)
                top2_df.to_csv(top2_path, index=False)
                print(f"[UNIFIED PIPELINE] Wrote TOP 2 for trading: {top2_path}")

            # Save trade theses as markdown
            if top2_theses:
                thesis_dir = Path(args.out_thesis)
                thesis_dir.mkdir(parents=True, exist_ok=True)

                for i, thesis in enumerate(top2_theses, 1):
                    thesis_path = thesis_dir / f"thesis_{thesis.signal.symbol}_{end_date}.md"
                    with open(thesis_path, 'w', encoding='utf-8') as f:
                        f.write(thesis.to_markdown())
                    print(f"[UNIFIED PIPELINE] Wrote thesis #{i}: {thesis_path}")

            # Display TOP 2 with comprehensive analysis
            print("\n" + "=" * 70)
            print(f"TOP 2 TRADES OF THE DAY - FULL ANALYSIS | {end_date}")
            print("=" * 70)

            for i, thesis in enumerate(top2_theses, 1):
                s = thesis.signal
                print(f"\n{'─' * 70}")
                print(f"TRADE #{i}: {s.symbol} ({s.asset_class}) - {s.strategy.upper()}")
                print(f"{'─' * 70}")
                print(f"  Side: {s.side.upper()} | Entry: ${s.entry_price:.2f} | Stop: ${s.stop_loss:.2f}")
                print(f"  Final Confidence: {s.final_conf_score:.2%} | Rank: #{s.final_rank}")

                print(f"\n  CONFIDENCE BREAKDOWN:")
                print(f"    ML Meta (XGB/LGBM): {s.ml_meta_conf:.2%}")
                print(f"    LSTM Success Prob:  {s.lstm_success:.2%}")
                print(f"    Ensemble Conf:      {s.ensemble_conf:.2%}")
                print(f"    Markov pi(Up):      {s.markov_pi_up:.2%} {'✓' if s.markov_agrees else '✗'}")
                print(f"    Conviction Score:   {s.conviction_score}/100 ({s.conviction_tier})")
                print(f"    Cognitive Approved: {'YES' if s.cognitive_approved else 'NO'} ({s.cognitive_confidence:.0%})")

                print(f"\n  HISTORICAL PATTERN:")
                print(f"    Consecutive Days:   {s.streak_length}")
                print(f"    Historical Samples: {s.streak_samples}")
                print(f"    Win Rate:           {s.streak_win_rate:.0%}")
                print(f"    Avg Bounce:         {s.streak_avg_bounce:+.1%}")
                print(f"    Auto-Pass Eligible: {'YES' if s.qualifies_auto_pass else 'NO'}")

                print(f"\n  ALT DATA:")
                print(f"    News Sentiment: {s.news_sentiment:+.2f} ({s.news_article_count} articles)")
                print(f"    Insider Signal: {s.insider_signal.upper()}")
                print(f"    Congress:       {s.congress_signal.upper()} ({s.congress_buys} buys / {s.congress_sells} sells)")
                print(f"    Options Flow:   {s.options_flow_signal.upper()}")

                print(f"\n  RISK ANALYSIS:")
                print(f"    Kelly Optimal:  {s.kelly_optimal_pct:.1%}")
                print(f"    VaR Contrib:    ${s.var_contribution:,.0f}")
                print(f"    Correlation:    {s.correlation_with_portfolio:.2f}")
                print(f"    Expected Move:  {s.expected_move_weekly:.1%} (weekly)")

                print(f"\n  VERDICT: {thesis.verdict}")
                print(f"  CONVICTION: {thesis.conviction_level}")

        except Exception as e:
            print(f"[UNIFIED PIPELINE] Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    elif args.unified and not UNIFIED_PIPELINE_AVAILABLE:
        print("[WARN] Unified pipeline requested but module not available")

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

                # === MARKOV CHAIN CONFIDENCE BOOST (Medallion-inspired) ===
                # Boost conf_score when Markov chain agrees with signal direction
                if MARKOV_AVAILABLE and args.markov and 'markov_agrees' in df.columns:
                    try:
                        def apply_markov_boost(row: pd.Series) -> float:
                            base = float(row.get('conf_score', 0.0))
                            agrees = bool(row.get('markov_agrees', False))
                            p_up = float(row.get('markov_p_up_today', 0.33))
                            side = str(row.get('side', 'long'))

                            if not agrees:
                                return base  # No boost if Markov disagrees

                            # Base boost for agreement: +5%
                            boost = 0.05

                            # Additional boost for strong Markov signal
                            if side == 'long' and p_up >= 0.55:
                                boost += 0.05  # +5% more for strong bullish Markov
                            elif side == 'short' and p_up <= 0.25:
                                boost += 0.05  # +5% more for strong bearish Markov

                            return max(0.0, min(1.0, base + boost))

                        df['conf_score'] = df.apply(apply_markov_boost, axis=1)
                        if args.verbose:
                            boosted = df[df['markov_agrees'] == True]
                            if len(boosted) > 0:
                                print(f"[MARKOV BOOST] Applied to {len(boosted)} signals with Markov agreement")
                    except Exception as e:
                        if args.verbose:
                            print(f"[MARKOV BOOST] Failed: {e}")

                # === KOBE STANDARD: 900 -> 5 -> 2 PIPELINE ===
                # Step 1: Filter to Top-5 candidates for STUDY (follow, analyze, test)
                top5_df = None
                if args.top5 and not df.empty:
                    # Sort by conf_score and take top 5
                    df = df.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort')
                    top5_df = df.head(5).copy()

                    # Write Top-5 to file
                    top5_path = Path(args.out_top5)
                    top5_path.parent.mkdir(parents=True, exist_ok=True)
                    top5_df.to_csv(top5_path, index=False)

                    if args.verbose:
                        print(f"\n[KOBE PIPELINE] 900 -> 5: Filtered to top {len(top5_df)} candidates")
                        print(f"  Wrote: {top5_path}")

                    # Continue with only the top 5 for further selection
                    df = top5_df.copy()

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

                # === KOBE STANDARD: 900 -> 5 -> 2 PIPELINE ===
                # Select top N (default 2) to actually TRADE from Top 5
                # If --top5 is used, tradeable comes from top5_df, else from out
                source_df = top5_df if (args.top5 and top5_df is not None and not top5_df.empty) else out
                trade_n = min(args.trade_top_n, len(source_df))
                tradeable = source_df.sort_values(['conf_score', 'timestamp', 'symbol'], ascending=[False, True, True], kind='mergesort').head(trade_n)

                # Write tradeable signals to file
                tradeable_path = Path(args.out_tradeable)
                tradeable_path.parent.mkdir(parents=True, exist_ok=True)
                tradeable.to_csv(tradeable_path, index=False)

                if args.verbose:
                    print(f"\n[KOBE PIPELINE] 5 -> {trade_n}: Selected top {len(tradeable)} for trading")
                    print(f"  Wrote: {tradeable_path}")

                # Show signal date prominently to prevent stale signal confusion
                signal_date = end_date  # The date these signals are generated for

                # === KOBE STANDARD PIPELINE OUTPUT: 900 -> 5 -> 2 ===
                print(f"\n{'='*70}")
                print(f"KOBE PIPELINE: 900 -> 5 (STUDY) -> 2 (TRADE) | DATE: {signal_date}")
                print(f"{'='*70}")
                print("  NOTE: These signals are valid for the NEXT trading day")

                # Show Top-5 for STUDY (follow, analyze, test, understand)
                if args.top5 and top5_df is not None and not top5_df.empty:
                    print(f"\n[STEP 1] TOP 5 TO STUDY - Follow, Analyze, Test ({len(top5_df)} signals)")
                    print("-" * 70)
                    cols = ['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']
                    avail_cols = [c for c in cols if c in top5_df.columns]
                    print(top5_df[avail_cols].to_string(index=False))
                    print(f"  Wrote: {args.out_top5}")
                    print("  PURPOSE: Study these 5 stocks - follow the algo, analyze patterns, test ideas")

                # Show Top-2 to TRADE (what we actually execute)
                print(f"\n[STEP 2] TOP 2 TO TRADE - Execute These ({len(tradeable)} signals)")
                print("-" * 70)
                print(tradeable[['strategy','symbol','side','entry_price','stop_loss','take_profit','conf_score']].to_string(index=False))
                print(f"  Wrote: {tradeable_path}")
                print("  PURPOSE: These are the ONLY 2 trades to execute. Best of the Top 5.")

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

    # === UNIFIED MULTI-ASSET RANKING ===
    # signals now contains ALL asset classes: EQUITY, CRYPTO, OPTIONS
    # All have been through the same AI pipeline with proper conf_scores
    print("\n" + "=" * 70)
    print("UNIFIED MULTI-ASSET RANKING")
    print("=" * 70)

    if not signals.empty:
        unified_signals = signals.copy()

        # Add trade_type column based on asset_class
        if 'trade_type' not in unified_signals.columns:
            def get_trade_type(row):
                asset = str(row.get('asset_class', 'EQUITY'))
                if asset == 'EQUITY':
                    return 'shares'
                elif asset == 'OPTIONS':
                    return str(row.get('option_type', 'option')).lower()
                elif asset == 'CRYPTO':
                    return 'crypto'
                return 'unknown'
            unified_signals['trade_type'] = unified_signals.apply(get_trade_type, axis=1)

        # Ensure conf_score is numeric
        unified_signals['conf_score'] = pd.to_numeric(unified_signals['conf_score'], errors='coerce').fillna(0.5)

        # Sort by conf_score (descending) for unified ranking
        # STRICT RANKING: conf_score is the ONLY primary sort key
        # Ties broken by symbol (alphabetical) for determinism
        unified_signals = unified_signals.sort_values(
            ['conf_score', 'symbol'],
            ascending=[False, True],
            kind='mergesort'
        ).reset_index(drop=True)

        # Add unified rank
        unified_signals['unified_rank'] = range(1, len(unified_signals) + 1)

        # === RANKING VALIDATION ===
        # Verify that ranking is strictly by conf_score (no errors in the math)
        if len(unified_signals) > 1:
            prev_conf = float(unified_signals.iloc[0]['conf_score'])
            validation_errors = []
            for i in range(1, len(unified_signals)):
                curr_conf = float(unified_signals.iloc[i]['conf_score'])
                if curr_conf > prev_conf + 0.0001:  # Allow tiny floating point tolerance
                    validation_errors.append(
                        f"Rank {i}: {unified_signals.iloc[i]['symbol']} (conf={curr_conf:.4f}) > "
                        f"Rank {i-1}: {unified_signals.iloc[i-1]['symbol']} (conf={prev_conf:.4f})"
                    )
                prev_conf = curr_conf

            if validation_errors:
                print("\n[RANKING ERROR] Math validation FAILED:")
                for err in validation_errors[:5]:
                    print(f"  {err}")
                print("  This is a bug - ranking should be strictly by conf_score!")
            elif args.verbose:
                print(f"[RANKING VALID] All {len(unified_signals)} signals correctly ranked by conf_score")

        # === TOP 5 STUDY SIGNALS (across ALL asset classes) ===
        top5_unified = unified_signals.head(5).copy()

        # === TOP 2 TRADE SIGNALS (across ALL asset classes) ===
        top2_unified = unified_signals.head(2).copy()

        # Count by asset class
        equity_count = len(unified_signals[unified_signals['asset_class'] == 'EQUITY'])
        options_count = len(unified_signals[unified_signals['asset_class'] == 'OPTIONS'])
        crypto_count = len(unified_signals[unified_signals['asset_class'] == 'CRYPTO'])

        print(f"\nTotal Signal Pool: {len(unified_signals)} signals")
        print(f"  EQUITIES: {equity_count} | OPTIONS: {options_count} | CRYPTO: {crypto_count}")

        # Show TOP 5 (unified across all asset classes)
        print(f"\n{'='*70}")
        print("TOP 5 TO STUDY (Unified Across All Asset Classes)")
        print("="*70)
        display_cols = ['unified_rank', 'asset_class', 'trade_type', 'symbol', 'side', 'entry_price', 'conf_score']
        available_cols = [c for c in display_cols if c in top5_unified.columns]
        if available_cols:
            print(top5_unified[available_cols].to_string(index=False))
        else:
            print(top5_unified.head().to_string(index=False))

        # Show TOP 2 (what we actually trade)
        print(f"\n{'='*70}")
        print("TOP 2 TO TRADE (Execute These - Any Asset Class)")
        print("="*70)
        if not top2_unified.empty:
            for idx, row in top2_unified.iterrows():
                asset = row.get('asset_class', 'UNKNOWN')
                trade_type = row.get('trade_type', 'unknown')
                symbol = row.get('symbol', '???')
                side = row.get('side', 'long')
                price = row.get('entry_price', 0)
                conf = row.get('conf_score', 0)
                rank = row.get('unified_rank', '?')

                if asset == 'EQUITY':
                    print(f"  #{rank} [{asset}] BUY {symbol} shares @ ${price:.2f} (conf: {conf:.2f})")
                elif asset == 'OPTIONS':
                    opt_type = row.get('option_type', trade_type.upper())
                    strike = row.get('strike', 0)
                    exp = row.get('expiration', 'N/A')
                    print(f"  #{rank} [{asset}] BUY {symbol} {opt_type} ${strike:.0f} exp {exp} @ ${price:.2f} (conf: {conf:.2f})")
                elif asset == 'CRYPTO':
                    print(f"  #{rank} [{asset}] BUY {symbol} @ ${price:.2f} (conf: {conf:.2f})")
                else:
                    print(f"  #{rank} [{asset}] {side.upper()} {symbol} @ ${price:.2f} (conf: {conf:.2f})")

        # Write unified outputs
        unified_path = ROOT / 'logs' / 'unified_signals.csv'
        unified_path.parent.mkdir(parents=True, exist_ok=True)
        unified_signals.to_csv(unified_path, index=False)

        top5_path = ROOT / 'logs' / 'top5_unified.csv'
        top5_unified.to_csv(top5_path, index=False)

        top2_path = ROOT / 'logs' / 'top2_trade.csv'
        top2_unified.to_csv(top2_path, index=False)

        print(f"\n{'='*70}")
        print("OUTPUT FILES")
        print("="*70)
        print(f"  All signals:  {unified_path}")
        print(f"  Top 5 study:  {top5_path}")
        print(f"  Top 2 trade:  {top2_path}")

        # Final summary
        print(f"\n{'='*70}")
        print("KOBE PIPELINE: 900 STOCKS + OPTIONS + CRYPTO → TOP 5 → TOP 2")
        print("="*70)

        # Show asset mix in top 2
        top2_equities = len(top2_unified[top2_unified['asset_class'] == 'EQUITY'])
        top2_options = len(top2_unified[top2_unified['asset_class'] == 'OPTIONS'])
        top2_crypto = len(top2_unified[top2_unified['asset_class'] == 'CRYPTO'])

        mix_parts = []
        if top2_equities > 0:
            mix_parts.append(f"{top2_equities} shares")
        if top2_options > 0:
            mix_parts.append(f"{top2_options} options")
        if top2_crypto > 0:
            mix_parts.append(f"{top2_crypto} crypto")

        print(f"  TODAY'S MIX: {' + '.join(mix_parts) if mix_parts else 'No signals'}")
        print(f"  TRADE COUNT: {len(top2_unified)}")

    else:
        print("\n  No signals generated across any asset class")

    return 0


if __name__ == "__main__":
    sys.exit(main())
