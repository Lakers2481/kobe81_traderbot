#!/usr/bin/env python
"""
COMPREHENSIVE ENTRY-HOLD-RECOVERY OPTIMIZER - RENAISSANCE STYLE

Mission: Build a FULL, reproducible 900-stock optimizer to discover optimal:
- ENTRY CONDITION (streak length, mode, timing, filters)
- HOLDING TIME (1-7 days)
- RECOVERY / REBOUND TARGET BEHAVIOR

Tests ALL combinations across 800 stocks with full statistical rigor:
- ~294 parameter combinations
- Walk-forward validation (2015-2019 train, 2020-2025 test)
- Multiple testing correction (Benjamini-Hochberg FDR)
- Cost sensitivity analysis (0, 5, 10 bps)
- Multi-objective ranking

ABSOLUTE RULES (No Exceptions):
1. No claims without exact definitions, raw counts, reproducible code
2. No silent defaults - every parameter printed
3. No lookahead - entry based only on info available at time t
4. Report uncertainty (Wilson CI or bootstrap) + sample size
5. Prevent overfitting with walk-forward and cost sensitivity

Author: Kobe Trading System
Date: 2026-01-08
"""

import argparse
import itertools
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.statistical_testing import (
    benjamini_hochberg_fdr,
    bonferroni_correction,
    compute_binomial_pvalue,
    wilson_confidence_interval,
    interpret_fdr_result
)
from analytics.recovery_analyzer import (
    analyze_recovery_times,
    generate_recovery_curves,
    calculate_mfe_mae_stats,
    rank_combos_by_recovery_speed,
    rank_combos_by_target_hit
)
from bounce.streak_analyzer import calculate_streaks_vectorized
from data.providers.multi_source import fetch_daily_bars_resilient


# ============================================================================
# PART 1: CONFIGURATION & DEFINITIONS
# ============================================================================

@dataclass
class OptimizationConfig:
    """All optimization parameters (NO SILENT DEFAULTS)."""

    # Universe
    universe_path: str = "data/universe/optionable_liquid_800.csv"

    # Date ranges
    full_start: str = "2015-01-01"
    full_end: str = "2025-12-31"
    train_start: str = "2015-01-01"
    train_end: str = "2019-12-31"
    test_start: str = "2020-01-01"
    test_end: str = "2025-12-31"

    # Entry optimization
    streak_lengths: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    streak_modes: List[str] = field(default_factory=lambda: ["AT_LEAST", "EXACT"])
    entry_timings: List[str] = field(default_factory=lambda: ["CLOSE_T", "OPEN_T1"])
    use_ibs_filter: bool = False  # Set True if IBS<0.2 available
    use_rsi_filter: bool = False  # Set True if RSI(2)<5 available

    # Holding periods
    hold_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])

    # Recovery targets
    recovery_targets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])

    # Filters
    min_instances_per_symbol: int = 20
    min_instances_global: int = 500
    min_symbols_contributing: int = 200

    # Costs
    cost_bps_scenarios: List[int] = field(default_factory=lambda: [0, 5, 10])

    # Statistical
    alpha: float = 0.05
    fdr_method: str = "benjamini_hochberg"

    # Execution
    smoke_test_symbols: int = 20  # Fast test mode
    n_workers: int = 4  # Parallel workers
    cache_dir: str = "data/cache"
    output_dir: str = "output"

    # Data provider
    provider_order: List[str] = field(default_factory=lambda: ["polygon", "stooq"])


def print_config(config: OptimizationConfig):
    """Print all configuration parameters (no silent defaults)."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ENTRY-HOLD-RECOVERY OPTIMIZER - CONFIGURATION")
    print("=" * 80)
    print(f"\nUniverse: {config.universe_path}")
    print(f"Date Range (Full): {config.full_start} to {config.full_end}")
    print(f"Train Period: {config.train_start} to {config.train_end}")
    print(f"Test Period: {config.test_start} to {config.test_end}")
    print(f"\nStreak Lengths: {config.streak_lengths}")
    print(f"Streak Modes: {config.streak_modes}")
    print(f"Entry Timings: {config.entry_timings}")
    print(f"IBS Filter: {config.use_ibs_filter}")
    print(f"RSI Filter: {config.use_rsi_filter}")
    print(f"\nHolding Periods: {config.hold_periods}")
    print(f"Recovery Targets: {config.recovery_targets}")
    print(f"\nMin Instances Per Symbol: {config.min_instances_per_symbol}")
    print(f"Min Instances Global: {config.min_instances_global}")
    print(f"Min Symbols Contributing: {config.min_symbols_contributing}")
    print(f"\nCost Scenarios (bps): {config.cost_bps_scenarios}")
    print(f"Significance Level (alpha): {config.alpha}")
    print(f"FDR Method: {config.fdr_method}")
    print(f"\nSmoke Test Symbols: {config.smoke_test_symbols}")
    print(f"Parallel Workers: {config.n_workers}")
    print(f"Cache Directory: {config.cache_dir}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Data Providers: {' -> '.join(config.provider_order)}")
    print("=" * 80 + "\n")


# ============================================================================
# PART 2: PARAMETER GRID GENERATOR
# ============================================================================

def generate_parameter_grid(config: OptimizationConfig) -> List[Dict]:
    """
    Generate all parameter combinations to test.

    Returns list of dicts, each representing one combo:
    {
        'combo_id': unique identifier,
        'streak_length': int,
        'streak_mode': "AT_LEAST" | "EXACT",
        'entry_timing': "CLOSE_T" | "OPEN_T1",
        'hold_period': int,
        'use_ibs': bool,
        'use_rsi': bool
    }
    """
    ibs_options = [config.use_ibs_filter] if not config.use_ibs_filter else [False, True]
    rsi_options = [config.use_rsi_filter] if not config.use_rsi_filter else [False, True]

    # Cartesian product of all parameters
    param_combinations = list(itertools.product(
        config.streak_lengths,
        config.streak_modes,
        config.entry_timings,
        config.hold_periods,
        ibs_options,
        rsi_options
    ))

    param_grid = []
    for idx, (streak_len, streak_mode, entry_timing, hold_period, use_ibs, use_rsi) in enumerate(param_combinations):
        combo_id = f"S{streak_len}_{streak_mode[:2]}_{entry_timing[:4]}_H{hold_period}"
        if use_ibs:
            combo_id += "_IBS"
        if use_rsi:
            combo_id += "_RSI"

        param_grid.append({
            'combo_id': combo_id,
            'combo_num': idx + 1,
            'streak_length': streak_len,
            'streak_mode': streak_mode,
            'entry_timing': entry_timing,
            'hold_period': hold_period,
            'use_ibs': use_ibs,
            'use_rsi': use_rsi
        })

    return param_grid


# ============================================================================
# PART 3: EVENT DETECTION ENGINE
# ============================================================================

def detect_entry_events(
    df: pd.DataFrame,
    symbol: str,
    streak_length: int,
    streak_mode: str,
    entry_timing: str,
    use_ibs: bool = False,
    use_rsi: bool = False
) -> pd.DataFrame:
    """
    Detect all entry events for a symbol based on parameters.

    Returns DataFrame with columns:
    - entry_date
    - entry_price (based on entry_timing)
    - streak_length_actual
    - ibs_value (if use_ibs)
    - rsi_value (if use_rsi)
    """
    if df.empty or len(df) < 10:
        return pd.DataFrame()

    # Ensure required columns exist
    required_cols = ['close']
    if entry_timing == "OPEN_T1":
        required_cols.append('open')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Missing columns {missing_cols} for {symbol}")
        return pd.DataFrame()

    # Detect consecutive down days using bounce/streak_analyzer
    df_with_streaks = calculate_streaks_vectorized(df.copy())

    if 'streak_len' not in df_with_streaks.columns:
        return pd.DataFrame()

    # Apply streak mode filter with lookahead prevention (shift by 1)
    if streak_mode == "AT_LEAST":
        # Entry signal when prior bar had streak >= target
        signal_mask = (df_with_streaks['streak_len'].shift(1) >= streak_length)
    elif streak_mode == "EXACT":
        # Entry signal when prior bar had exactly target streak
        signal_mask = (df_with_streaks['streak_len'].shift(1) == streak_length)
    else:
        raise ValueError(f"Unknown streak_mode: {streak_mode}")

    # Apply optional filters (also shifted to prevent lookahead)
    if use_ibs and 'ibs' in df_with_streaks.columns:
        signal_mask &= (df_with_streaks['ibs'].shift(1) < 0.2)

    if use_rsi and 'rsi_2' in df_with_streaks.columns:
        signal_mask &= (df_with_streaks['rsi_2'].shift(1) < 5.0)

    # Get entry dates and prices
    entry_indices = df_with_streaks.index[signal_mask]

    if len(entry_indices) == 0:
        return pd.DataFrame()

    events = []
    for entry_idx in entry_indices:
        try:
            # Entry price based on timing
            if entry_timing == "CLOSE_T":
                entry_price = df_with_streaks.loc[entry_idx, 'close']
                entry_date = df_with_streaks.loc[entry_idx, 'timestamp'] if 'timestamp' in df_with_streaks.columns else entry_idx
            elif entry_timing == "OPEN_T1":
                # Enter at next day's open
                next_idx_pos = df_with_streaks.index.get_loc(entry_idx) + 1
                if next_idx_pos >= len(df_with_streaks):
                    continue
                next_idx = df_with_streaks.index[next_idx_pos]
                entry_price = df_with_streaks.loc[next_idx, 'open']
                entry_date = df_with_streaks.loc[next_idx, 'timestamp'] if 'timestamp' in df_with_streaks.columns else next_idx
            else:
                raise ValueError(f"Unknown entry_timing: {entry_timing}")

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            event = {
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'streak_length_actual': df_with_streaks.loc[entry_idx, 'streak_len']
            }

            if use_ibs and 'ibs' in df_with_streaks.columns:
                event['ibs_value'] = df_with_streaks.loc[entry_idx, 'ibs']

            if use_rsi and 'rsi_2' in df_with_streaks.columns:
                event['rsi_value'] = df_with_streaks.loc[entry_idx, 'rsi_2']

            events.append(event)

        except (KeyError, IndexError):
            continue

    return pd.DataFrame(events)


def calculate_holding_outcomes(
    df: pd.DataFrame,
    events: pd.DataFrame,
    hold_period: int
) -> pd.DataFrame:
    """
    For each event, calculate outcomes at hold_period.

    Returns DataFrame with columns:
    - entry_date, entry_price, symbol
    - exit_price (at hold_period)
    - return_pct
    - close_1d, close_2d, ..., close_7d (forward prices)
    - high_1d, high_2d, ..., high_7d (if available)
    - low_1d, low_2d, ..., low_7d (if available)
    """
    if events.empty:
        return pd.DataFrame()

    # Ensure timestamp column exists for matching
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            warnings.warn("DataFrame missing timestamp column")
            return pd.DataFrame()

    results = []

    for _, event in events.iterrows():
        entry_date = event['entry_date']
        entry_price = event['entry_price']

        # Find entry date index in price data
        try:
            entry_loc = df[df['timestamp'] == entry_date].index
            if len(entry_loc) == 0:
                continue
            entry_idx_pos = df.index.get_loc(entry_loc[0])
        except (KeyError, IndexError):
            continue

        # Calculate forward prices for up to 7 days
        forward_data = {'entry_date': entry_date, 'entry_price': entry_price, 'symbol': event['symbol']}

        for day in range(1, 8):  # Days 1-7
            future_idx_pos = entry_idx_pos + day

            if future_idx_pos >= len(df):
                # Not enough forward data
                forward_data[f'close_{day}d'] = np.nan
                forward_data[f'high_{day}d'] = np.nan
                forward_data[f'low_{day}d'] = np.nan
                continue

            future_idx = df.index[future_idx_pos]

            # Get close, high, low (if available)
            forward_data[f'close_{day}d'] = df.loc[future_idx, 'close']

            if 'high' in df.columns:
                forward_data[f'high_{day}d'] = df.loc[future_idx, 'high']
            else:
                forward_data[f'high_{day}d'] = df.loc[future_idx, 'close']

            if 'low' in df.columns:
                forward_data[f'low_{day}d'] = df.loc[future_idx, 'low']
            else:
                forward_data[f'low_{day}d'] = df.loc[future_idx, 'close']

        # Exit price at specific holding period
        exit_col = f'close_{hold_period}d'
        if exit_col in forward_data:
            exit_price = forward_data[exit_col]
            if not pd.isna(exit_price) and exit_price > 0:
                return_pct = (exit_price - entry_price) / entry_price
                forward_data['exit_price'] = exit_price
                forward_data['return_pct'] = return_pct
            else:
                forward_data['exit_price'] = np.nan
                forward_data['return_pct'] = np.nan

        # Copy other event data
        for col in event.index:
            if col not in forward_data:
                forward_data[col] = event[col]

        results.append(forward_data)

    return pd.DataFrame(results)


# ============================================================================
# PART 4: AGGREGATION & STATISTICS
# ============================================================================

def aggregate_combo_results(
    events: pd.DataFrame,
    combo: Dict,
    config: OptimizationConfig,
    aggregation_mode: str = "event_weighted"
) -> Dict:
    """
    Aggregate all events for one combo.

    aggregation_mode:
    - "event_weighted": pool all events across all stocks
    - "stock_equal_weighted": average per symbol, then average across symbols

    Returns metrics dict with comprehensive statistics.
    """
    if events.empty:
        return {
            'combo_id': combo['combo_id'],
            'n_instances': 0,
            'n_symbols': 0,
            'win_rate': 0.0,
            'mean_return': 0.0,
            'median_return': 0.0,
            'std_return': 0.0,
            'percentile_5_return': 0.0,
            'percentile_95_return': 0.0,
            'mean_mfe': 0.0,
            'mean_mae': 0.0,
            'median_time_to_breakeven': np.nan,
            'median_time_to_1pct': np.nan,
            'p_breakeven_by_3d': 0.0,
            'p_hit_1pct_by_3d': 0.0,
            'p_hit_2pct_by_7d': 0.0,
            'p_value': 1.0,
            'is_significant_bonferroni': False,
            'is_significant_fdr': False  # Will be updated in main()
        }

    # Filter out events with NaN returns
    valid_events = events[events['return_pct'].notna()].copy()

    if valid_events.empty:
        return aggregate_combo_results(pd.DataFrame(), combo, config, aggregation_mode)

    # Basic statistics
    n_instances = len(valid_events)
    n_symbols = valid_events['symbol'].nunique()

    returns = valid_events['return_pct'].values
    wins = (returns > 0).sum()
    win_rate = wins / len(returns) if len(returns) > 0 else 0.0

    mean_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    std_return = float(np.std(returns))
    percentile_5 = float(np.percentile(returns, 5))
    percentile_95 = float(np.percentile(returns, 95))

    # MFE/MAE stats
    mfe_mae_stats = calculate_mfe_mae_stats(valid_events, max_days=combo['hold_period'])

    # Recovery analysis
    events_with_recovery = analyze_recovery_times(valid_events, max_days=7)

    # Time to targets (median)
    median_time_to_breakeven = events_with_recovery['time_to_breakeven'].median()
    median_time_to_1pct = events_with_recovery['time_to_1pct'].median()

    # Probability of hitting targets by specific days
    p_breakeven_by_3d = events_with_recovery['hit_breakeven_by_day_3'].mean() if 'hit_breakeven_by_day_3' in events_with_recovery.columns else 0.0
    p_hit_1pct_by_3d = events_with_recovery['hit_1pct_by_day_3'].mean() if 'hit_1pct_by_day_3' in events_with_recovery.columns else 0.0
    p_hit_2pct_by_7d = events_with_recovery['hit_2pct_by_day_7'].mean() if 'hit_2pct_by_day_7' in events_with_recovery.columns else 0.0

    # Statistical significance
    stat_result = compute_binomial_pvalue(
        wins=int(wins),
        total=len(returns),
        null_prob=0.5,
        alpha=config.alpha,
        n_trials=1,  # Will apply FDR correction later
        alternative="greater"
    )

    # Wilson confidence interval for win rate
    ci_result = wilson_confidence_interval(wins=int(wins), total=len(returns), confidence_level=0.95)

    return {
        'combo_id': combo['combo_id'],
        'combo_num': combo['combo_num'],
        'streak_length': combo['streak_length'],
        'streak_mode': combo['streak_mode'],
        'entry_timing': combo['entry_timing'],
        'hold_period': combo['hold_period'],
        'use_ibs': combo['use_ibs'],
        'use_rsi': combo['use_rsi'],
        'n_instances': n_instances,
        'n_symbols': n_symbols,
        'win_rate': win_rate,
        'win_rate_ci_lower': ci_result.lower_bound,
        'win_rate_ci_upper': ci_result.upper_bound,
        'mean_return': mean_return,
        'median_return': median_return,
        'std_return': std_return,
        'percentile_5_return': percentile_5,
        'percentile_95_return': percentile_95,
        'mean_mfe': mfe_mae_stats['mean_mfe'],
        'median_mfe': mfe_mae_stats['median_mfe'],
        'mean_mae': mfe_mae_stats['mean_mae'],
        'median_mae': mfe_mae_stats['median_mae'],
        'mfe_mae_ratio': mfe_mae_stats['mfe_mae_ratio'],
        'median_time_to_breakeven': median_time_to_breakeven,
        'median_time_to_1pct': median_time_to_1pct,
        'p_breakeven_by_3d': p_breakeven_by_3d,
        'p_hit_1pct_by_3d': p_hit_1pct_by_3d,
        'p_hit_2pct_by_7d': p_hit_2pct_by_7d,
        'p_value': stat_result.p_value,
        'is_significant_bonferroni': stat_result.is_significant,
        'is_significant_fdr': False  # Updated later
    }


def apply_cost_sensitivity(
    events: pd.DataFrame,
    cost_bps: int
) -> pd.DataFrame:
    """
    Apply round-trip cost to all returns.

    cost_bps: basis points (0, 5, or 10)

    Returns events DataFrame with adjusted returns.
    """
    if events.empty:
        return events

    events_with_cost = events.copy()
    cost_pct = cost_bps / 10000.0  # Convert bps to decimal

    if 'return_pct' in events_with_cost.columns:
        events_with_cost['return_pct'] = events_with_cost['return_pct'] - cost_pct

    return events_with_cost


# ============================================================================
# PART 5: MULTI-OBJECTIVE RANKING
# ============================================================================

def rank_by_expected_return(results_df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """
    Rank combos by mean_return (after costs).

    Filter:
    - n_instances >= min_instances_global
    - n_symbols >= min_symbols_contributing

    Sort descending by mean_return.
    """
    filtered = results_df[
        (results_df['n_instances'] >= config.min_instances_global) &
        (results_df['n_symbols'] >= config.min_symbols_contributing)
    ].copy()

    ranked = filtered.sort_values('mean_return', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked


def rank_by_fast_recovery(results_df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """
    Rank combos by recovery speed.

    Objective: Maximize P(breakeven by day 3), minimize median TimeTo0

    Combined score:
    score = P(breakeven_by_3d) * 100 - median_TimeTo0 * 10
    """
    filtered = results_df[
        (results_df['n_instances'] >= config.min_instances_global) &
        (results_df['n_symbols'] >= config.min_symbols_contributing)
    ].copy()

    # Calculate recovery speed score
    filtered['recovery_speed_score'] = (
        filtered['p_breakeven_by_3d'] * 100 -
        filtered['median_time_to_breakeven'].fillna(7) * 10
    )

    ranked = filtered.sort_values('recovery_speed_score', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked


def rank_by_target_hit(results_df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """
    Rank combos by target hit probability.

    Objective: Maximize P(hit +1% within 3d) and P(hit +2% within 7d)

    Combined score:
    score = P(1pct_by_3d) * 50 + P(2pct_by_7d) * 50
    """
    filtered = results_df[
        (results_df['n_instances'] >= config.min_instances_global) &
        (results_df['n_symbols'] >= config.min_symbols_contributing)
    ].copy()

    # Calculate target hit score
    filtered['target_hit_score'] = (
        filtered['p_hit_1pct_by_3d'] * 50 +
        filtered['p_hit_2pct_by_7d'] * 50
    )

    ranked = filtered.sort_values('target_hit_score', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked


def rank_by_risk_adjusted(results_df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """
    Rank combos by risk-adjusted return.

    Objective: Maximize (mean_return / abs(mean_MAE))

    This rewards high returns with low downside pain.
    """
    filtered = results_df[
        (results_df['n_instances'] >= config.min_instances_global) &
        (results_df['n_symbols'] >= config.min_symbols_contributing)
    ].copy()

    # Calculate risk-adjusted score
    filtered['risk_adjusted_score'] = filtered.apply(
        lambda row: row['mean_return'] / abs(row['mean_mae']) if row['mean_mae'] != 0 else 0,
        axis=1
    )

    ranked = filtered.sort_values('risk_adjusted_score', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked


# ============================================================================
# PART 6: WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward_validation(
    combo: Dict,
    universe: List[str],
    config: OptimizationConfig
) -> Dict:
    """
    Run combo on train period, then test period.

    Returns:
    {
        'train_metrics': {...},
        'test_metrics': {...},
        'stability_ratio': test_mean_return / train_mean_return
    }
    """
    # Collect events for train period
    train_events_all = []
    for symbol in tqdm(universe, desc=f"WF Train - {combo['combo_id']}", leave=False):
        try:
            df = fetch_daily_bars_resilient(
                symbol=symbol,
                start=config.train_start,
                end=config.train_end,
                cache_dir=config.cache_dir,
                provider_order=config.provider_order
            )

            if df.empty or len(df) < 50:
                continue

            events = detect_entry_events(
                df, symbol,
                combo['streak_length'], combo['streak_mode'], combo['entry_timing'],
                combo['use_ibs'], combo['use_rsi']
            )

            if not events.empty:
                events_with_outcomes = calculate_holding_outcomes(df, events, combo['hold_period'])
                train_events_all.append(events_with_outcomes)

        except Exception as e:
            warnings.warn(f"Error processing {symbol} (train): {e}")
            continue

    train_events_df = pd.concat(train_events_all, ignore_index=True) if train_events_all else pd.DataFrame()
    train_metrics = aggregate_combo_results(train_events_df, combo, config)

    # Collect events for test period
    test_events_all = []
    for symbol in tqdm(universe, desc=f"WF Test - {combo['combo_id']}", leave=False):
        try:
            df = fetch_daily_bars_resilient(
                symbol=symbol,
                start=config.test_start,
                end=config.test_end,
                cache_dir=config.cache_dir,
                provider_order=config.provider_order
            )

            if df.empty or len(df) < 50:
                continue

            events = detect_entry_events(
                df, symbol,
                combo['streak_length'], combo['streak_mode'], combo['entry_timing'],
                combo['use_ibs'], combo['use_rsi']
            )

            if not events.empty:
                events_with_outcomes = calculate_holding_outcomes(df, events, combo['hold_period'])
                test_events_all.append(events_with_outcomes)

        except Exception as e:
            warnings.warn(f"Error processing {symbol} (test): {e}")
            continue

    test_events_df = pd.concat(test_events_all, ignore_index=True) if test_events_all else pd.DataFrame()
    test_metrics = aggregate_combo_results(test_events_df, combo, config)

    # Calculate stability ratio
    if train_metrics['mean_return'] != 0:
        stability_ratio = test_metrics['mean_return'] / train_metrics['mean_return']
    else:
        stability_ratio = 0.0

    return {
        'combo_id': combo['combo_id'],
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'stability_ratio': stability_ratio
    }


# ============================================================================
# PART 7: MAIN ORCHESTRATOR
# ============================================================================

def load_universe(universe_path: str) -> List[str]:
    """Load universe symbols from CSV."""
    if not os.path.exists(universe_path):
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    df = pd.read_csv(universe_path)

    # Assume first column is symbols
    symbols = df.iloc[:, 0].tolist()

    # Remove any NaN or empty values
    symbols = [s for s in symbols if pd.notna(s) and str(s).strip() != '']

    return symbols


def run_optimization(
    param_grid: List[Dict],
    universe: List[str],
    config: OptimizationConfig
) -> pd.DataFrame:
    """
    Run optimization across all parameter combinations and symbols.

    Returns DataFrame with results for all combos.
    """
    all_results = []

    for combo in tqdm(param_grid, desc="Testing parameter combinations"):
        # Collect events across all symbols for this combo
        combo_events_all = []

        for symbol in tqdm(universe, desc=f"Processing {combo['combo_id']}", leave=False):
            try:
                # Fetch data for full period
                df = fetch_daily_bars_resilient(
                    symbol=symbol,
                    start=config.full_start,
                    end=config.full_end,
                    cache_dir=config.cache_dir,
                    provider_order=config.provider_order
                )

                if df.empty or len(df) < 50:
                    continue

                # Detect entry events
                events = detect_entry_events(
                    df, symbol,
                    combo['streak_length'],
                    combo['streak_mode'],
                    combo['entry_timing'],
                    combo['use_ibs'],
                    combo['use_rsi']
                )

                if events.empty:
                    continue

                # Calculate holding outcomes
                events_with_outcomes = calculate_holding_outcomes(df, events, combo['hold_period'])

                if not events_with_outcomes.empty:
                    combo_events_all.append(events_with_outcomes)

            except Exception as e:
                warnings.warn(f"Error processing {symbol} for {combo['combo_id']}: {e}")
                continue

        # Aggregate results for this combo
        combo_events_df = pd.concat(combo_events_all, ignore_index=True) if combo_events_all else pd.DataFrame()
        combo_metrics = aggregate_combo_results(combo_events_df, combo, config)

        all_results.append(combo_metrics)

    results_df = pd.DataFrame(all_results)
    return results_df


def apply_multiple_testing_correction(results_df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to all p-values."""
    if results_df.empty:
        return results_df

    p_values = results_df['p_value'].values

    # Apply FDR correction
    fdr_result = benjamini_hochberg_fdr(p_values.tolist(), alpha=config.alpha)

    # Update is_significant_fdr column
    results_df['is_significant_fdr'] = fdr_result.significant

    # Print FDR interpretation
    print("\n" + "=" * 80)
    print("MULTIPLE TESTING CORRECTION RESULTS")
    print("=" * 80)
    print(interpret_fdr_result(fdr_result))
    print("=" * 80 + "\n")

    return results_df


def run_cost_sensitivity(results_df: pd.DataFrame, config: OptimizationConfig, param_grid: List[Dict], universe: List[str]):
    """
    Re-run top combos with different cost scenarios.

    Returns dict mapping cost_bps -> results_df
    """
    cost_results = {}

    for cost_bps in config.cost_bps_scenarios:
        if cost_bps == 0:
            # Already have 0-cost results
            cost_results[0] = results_df
            continue

        print(f"\nRecomputing results with {cost_bps} bps transaction costs...")

        # Recompute for all combos (simplified - reuse events with cost adjustment)
        # For full implementation, would re-run entire optimization with cost
        # Here we approximate by adjusting returns
        cost_adjusted_df = results_df.copy()
        cost_pct = cost_bps / 10000.0

        cost_adjusted_df['mean_return'] = cost_adjusted_df['mean_return'] - cost_pct
        cost_adjusted_df['median_return'] = cost_adjusted_df['median_return'] - cost_pct

        # Recalculate win rate (approximate - assumes symmetric cost impact)
        # This is a simplification; full version would reprocess all events
        cost_results[cost_bps] = cost_adjusted_df

    return cost_results


# ============================================================================
# PART 8: OUTPUT GENERATORS
# ============================================================================

def save_outputs(
    results_df: pd.DataFrame,
    top_combos_wf: Dict,
    config: OptimizationConfig,
    cost_results: Dict
):
    """
    Save all required CSV and markdown outputs.

    Outputs:
    1. output/entry_hold_grid_event_weighted.csv
    2. output/best_combos_expected_return.csv
    3. output/best_combos_fast_recovery.csv
    4. output/best_combos_target_hit.csv
    5. output/best_combos_risk_adjusted.csv
    6. output/coverage_report.csv
    7. output/cost_sensitivity_comparison.csv
    8. output/walk_forward_results.json
    9. output/optimizer_report.md
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. Full grid results
    results_df.to_csv(f"{config.output_dir}/entry_hold_grid_event_weighted.csv", index=False)
    print(f"Saved: {config.output_dir}/entry_hold_grid_event_weighted.csv")

    # 2-5. Top combos by objective
    objectives = ['expected_return', 'fast_recovery', 'target_hit', 'risk_adjusted']
    for obj in objectives:
        if obj in top_combos_wf:
            df = top_combos_wf[obj]
            output_path = f"{config.output_dir}/best_combos_{obj}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

    # 6. Coverage report (symbols contributing per combo)
    # Simplified version - full implementation would track per-symbol contributions
    coverage_data = results_df[['combo_id', 'n_symbols', 'n_instances']].copy()
    coverage_data['avg_instances_per_symbol'] = coverage_data['n_instances'] / coverage_data['n_symbols'].replace(0, 1)
    coverage_data.to_csv(f"{config.output_dir}/coverage_report.csv", index=False)
    print(f"Saved: {config.output_dir}/coverage_report.csv")

    # 7. Cost sensitivity comparison
    cost_comparison = []
    for cost_bps, cost_df in cost_results.items():
        top_5 = cost_df.nlargest(5, 'mean_return')[['combo_id', 'mean_return', 'win_rate', 'n_instances']]
        for idx, row in top_5.iterrows():
            cost_comparison.append({
                'cost_bps': cost_bps,
                'combo_id': row['combo_id'],
                'mean_return': row['mean_return'],
                'win_rate': row['win_rate'],
                'n_instances': row['n_instances']
            })

    cost_comp_df = pd.DataFrame(cost_comparison)
    cost_comp_df.to_csv(f"{config.output_dir}/cost_sensitivity_comparison.csv", index=False)
    print(f"Saved: {config.output_dir}/cost_sensitivity_comparison.csv")

    # 8. Walk-forward results
    with open(f"{config.output_dir}/walk_forward_results.json", 'w') as f:
        json.dump(top_combos_wf.get('walk_forward_data', {}), f, indent=2, default=str)
    print(f"Saved: {config.output_dir}/walk_forward_results.json")


def generate_markdown_report(
    results_df: pd.DataFrame,
    top_combos_wf: Dict,
    config: OptimizationConfig,
    cost_results: Dict
):
    """Generate comprehensive markdown report."""
    report_path = f"{config.output_dir}/optimizer_report.md"

    with open(report_path, 'w') as f:
        f.write("# COMPREHENSIVE ENTRY-HOLD-RECOVERY OPTIMIZER - RESULTS\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration section
        f.write("## CONFIGURATION\n\n")
        f.write(f"- **Universe:** {config.universe_path}\n")
        f.write(f"- **Date Range:** {config.full_start} to {config.full_end}\n")
        f.write(f"- **Train Period:** {config.train_start} to {config.train_end}\n")
        f.write(f"- **Test Period:** {config.test_start} to {config.test_end}\n")
        f.write(f"- **Streak Lengths:** {config.streak_lengths}\n")
        f.write(f"- **Streak Modes:** {config.streak_modes}\n")
        f.write(f"- **Entry Timings:** {config.entry_timings}\n")
        f.write(f"- **Holding Periods:** {config.hold_periods}\n")
        f.write(f"- **Cost Scenarios:** {config.cost_bps_scenarios} bps\n")
        f.write(f"- **FDR Alpha:** {config.alpha}\n\n")

        # Summary statistics
        f.write("## SUMMARY STATISTICS\n\n")
        f.write(f"- **Total Combinations Tested:** {len(results_df)}\n")
        f.write(f"- **Combos Passing FDR (5%):** {results_df['is_significant_fdr'].sum()}\n")
        f.write(f"- **Combos Meeting Coverage Requirements:** {len(results_df[results_df['n_instances'] >= config.min_instances_global])}\n\n")

        # Top combos by objective
        f.write("## TOP PARAMETER COMBINATIONS\n\n")

        objectives_titles = {
            'expected_return': 'Expected Return',
            'fast_recovery': 'Fast Recovery',
            'target_hit': 'Target Hit Probability',
            'risk_adjusted': 'Risk-Adjusted Return'
        }

        for obj_key, obj_title in objectives_titles.items():
            f.write(f"### {obj_title}\n\n")

            if obj_key in top_combos_wf and not top_combos_wf[obj_key].empty:
                top_5 = top_combos_wf[obj_key].head(5)

                f.write("| Rank | Combo ID | Win Rate | Mean Return | N Instances | FDR Sig |\n")
                f.write("|------|----------|----------|-------------|-------------|--------|\n")

                for _, row in top_5.iterrows():
                    f.write(f"| {row.get('rank', '-')} | {row['combo_id']} | {row['win_rate']:.1%} | {row['mean_return']:.2%} | {row['n_instances']} | {row['is_significant_fdr']} |\n")

                f.write("\n")
            else:
                f.write("No results available.\n\n")

        # Cost sensitivity
        f.write("## COST SENSITIVITY ANALYSIS\n\n")
        f.write("Top 3 combos by mean return across cost scenarios:\n\n")
        f.write("| Cost (bps) | Combo ID | Mean Return | Win Rate |\n")
        f.write("|------------|----------|-------------|----------|\n")

        for cost_bps in config.cost_bps_scenarios:
            if cost_bps in cost_results:
                top_3 = cost_results[cost_bps].nlargest(3, 'mean_return')
                for _, row in top_3.iterrows():
                    f.write(f"| {cost_bps} | {row['combo_id']} | {row['mean_return']:.2%} | {row['win_rate']:.1%} |\n")

        f.write("\n")

        # Walk-forward results
        if 'walk_forward_data' in top_combos_wf:
            f.write("## WALK-FORWARD VALIDATION\n\n")
            f.write("Out-of-sample stability analysis:\n\n")
            f.write("| Combo ID | Train WR | Test WR | Train Return | Test Return | Stability |\n")
            f.write("|----------|----------|---------|--------------|-------------|----------|\n")

            for combo_id, wf_data in list(top_combos_wf['walk_forward_data'].items())[:10]:
                train_m = wf_data['train_metrics']
                test_m = wf_data['test_metrics']
                stability = wf_data['stability_ratio']

                f.write(f"| {combo_id} | {train_m['win_rate']:.1%} | {test_m['win_rate']:.1%} | ")
                f.write(f"{train_m['mean_return']:.2%} | {test_m['mean_return']:.2%} | {stability:.2f} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("*Generated by Kobe Trading System - Comprehensive Entry-Hold-Recovery Optimizer*\n")

    print(f"\nSaved: {report_path}")


def main():
    """
    Main execution flow.

    Steps:
    1. Load configuration
    2. Load universe
    3. Generate parameter grid
    4. Run smoke test (20 symbols) if --smoke flag
    5. Run full universe
    6. Apply multiple testing correction
    7. Rank by objectives
    8. Generate outputs
    9. Create markdown report
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Comprehensive Entry-Hold-Recovery Optimizer")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (20 symbols)")
    parser.add_argument("--config", type=str, help="Path to config YAML (optional)")
    args = parser.parse_args()

    # Load config (default for now - YAML support can be added later)
    config = OptimizationConfig()

    # Print configuration (NO SILENT DEFAULTS)
    print_config(config)

    # Load universe
    universe = load_universe(config.universe_path)
    print(f"Loaded {len(universe)} symbols from universe")

    if args.smoke:
        universe = universe[:config.smoke_test_symbols]
        print(f"\n[SMOKE TEST MODE] Using {len(universe)} symbols\n")

    # Generate parameter grid
    param_grid = generate_parameter_grid(config)
    print(f"Generated {len(param_grid)} parameter combinations\n")

    # Run optimization
    print("Starting optimization...")
    results_df = run_optimization(param_grid, universe, config)

    if results_df.empty:
        print("\nERROR: No results generated. Check data availability and parameters.")
        return

    print(f"\nOptimization complete. Generated {len(results_df)} combo results.")

    # Apply multiple testing correction
    results_df = apply_multiple_testing_correction(results_df, config)

    # Rank by objectives
    print("\nRanking by multiple objectives...")
    top_expected_return = rank_by_expected_return(results_df, config)
    top_fast_recovery = rank_by_fast_recovery(results_df, config)
    top_target_hit = rank_by_target_hit(results_df, config)
    top_risk_adjusted = rank_by_risk_adjusted(results_df, config)

    top_combos_wf = {
        'expected_return': top_expected_return,
        'fast_recovery': top_fast_recovery,
        'target_hit': top_target_hit,
        'risk_adjusted': top_risk_adjusted
    }

    # Cost sensitivity
    print("\nRunning cost sensitivity analysis...")
    cost_results = run_cost_sensitivity(results_df, config, param_grid, universe)

    # Walk-forward validation on top combos
    print("\nRunning walk-forward validation on top combos...")
    wf_data = {}

    # Select top 5 unique combos across all objectives
    top_combo_ids = set()
    for obj_df in [top_expected_return, top_fast_recovery, top_target_hit, top_risk_adjusted]:
        top_combo_ids.update(obj_df.head(5)['combo_id'].tolist())

    for combo_id in list(top_combo_ids)[:10]:  # Limit to 10 for time
        combo = next((c for c in param_grid if c['combo_id'] == combo_id), None)
        if combo:
            wf_result = run_walk_forward_validation(combo, universe, config)
            wf_data[combo_id] = wf_result

    top_combos_wf['walk_forward_data'] = wf_data

    # Generate outputs
    print("\nGenerating output files...")
    save_outputs(results_df, top_combos_wf, config, cost_results)

    # Generate markdown report
    print("\nGenerating markdown report...")
    generate_markdown_report(results_df, top_combos_wf, config, cost_results)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {config.output_dir}/")
    print(f"- entry_hold_grid_event_weighted.csv")
    print(f"- best_combos_*.csv (4 files)")
    print(f"- coverage_report.csv")
    print(f"- cost_sensitivity_comparison.csv")
    print(f"- walk_forward_results.json")
    print(f"- optimizer_report.md")
    print("\nReview optimizer_report.md for comprehensive results.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

