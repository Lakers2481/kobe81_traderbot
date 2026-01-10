"""
Recovery Pattern Analyzer for Trading Strategies

Analyzes recovery behavior after entry events:
- Time to breakeven (0% return)
- Time to profit targets (+0.5%, +1.0%, +2.0%)
- Recovery probability curves by holding period
- Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) analysis

Supports multi-objective ranking by recovery speed, target hit probability, and risk-adjusted returns.

Author: Kobe Trading System
Date: 2026-01-08
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def analyze_recovery_times(
    events: pd.DataFrame,
    price_col: str = 'entry_price',
    max_days: int = 7
) -> pd.DataFrame:
    """
    Calculate recovery times for each entry event.

    For each event, computes:
    - time_to_breakeven: Days to reach 0% return (entry price)
    - time_to_0.5pct: Days to reach +0.5% profit
    - time_to_1pct: Days to reach +1.0% profit
    - time_to_2pct: Days to reach +2.0% profit

    Args:
        events: DataFrame with columns:
            - entry_date: Date of entry
            - entry_price: Entry price
            - forward_prices: Dict or array of future prices (day 1..max_days)
            OR individual columns: close_1d, close_2d, ..., close_7d
        price_col: Column name for entry price (default 'entry_price')
        max_days: Maximum forward days to analyze (default 7)

    Returns:
        DataFrame with original events + recovery time columns:
            - time_to_breakeven (int or NaN)
            - time_to_0.5pct (int or NaN)
            - time_to_1pct (int or NaN)
            - time_to_2pct (int or NaN)
            - hit_breakeven_by_day_{k} (bool) for k=1..max_days
            - hit_0.5pct_by_day_{k} (bool) for k=1..max_days
            - hit_1pct_by_day_{k} (bool) for k=1..max_days
            - hit_2pct_by_day_{k} (bool) for k=1..max_days

    Example:
        >>> events = pd.DataFrame({
        ...     'entry_date': ['2020-01-01', '2020-01-02'],
        ...     'entry_price': [100.0, 50.0],
        ...     'close_1d': [101.0, 49.5],
        ...     'close_2d': [102.0, 50.5],
        ...     'close_3d': [99.0, 51.0]
        ... })
        >>> result = analyze_recovery_times(events, max_days=3)
        >>> print(result[['entry_date', 'time_to_1pct']])
    """
    if events.empty:
        raise ValueError("events DataFrame is empty")

    if price_col not in events.columns:
        raise ValueError(f"'{price_col}' not found in events DataFrame")

    result = events.copy()

    # Target thresholds (as decimal returns)
    targets = {
        'breakeven': 0.0,
        '0.5pct': 0.005,
        '1pct': 0.01,
        '2pct': 0.02
    }

    # Initialize recovery time columns
    for target_name in targets.keys():
        result[f'time_to_{target_name}'] = np.nan

    # Initialize hit-by-day columns
    for day in range(1, max_days + 1):
        for target_name in targets.keys():
            result[f'hit_{target_name}_by_day_{day}'] = False

    # Process each event
    for idx in result.index:
        entry_price = result.loc[idx, price_col]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Extract forward prices for this event
        forward_prices = []
        for day in range(1, max_days + 1):
            col_name = f'close_{day}d'
            if col_name in result.columns:
                price = result.loc[idx, col_name]
                forward_prices.append(price if not pd.isna(price) else None)
            else:
                forward_prices.append(None)

        # Calculate returns for each forward day
        forward_returns = []
        for price in forward_prices:
            if price is not None and price > 0:
                ret = (price - entry_price) / entry_price
                forward_returns.append(ret)
            else:
                forward_returns.append(None)

        # Find time to each target
        for target_name, target_threshold in targets.items():
            time_to_target = None

            for day, ret in enumerate(forward_returns, start=1):
                if ret is not None:
                    # Check if target hit by this day
                    if ret >= target_threshold:
                        result.loc[idx, f'hit_{target_name}_by_day_{day}'] = True

                        # Record first time target was hit
                        if time_to_target is None:
                            time_to_target = day

                    # Propagate hit status to future days if already hit
                    if time_to_target is not None:
                        result.loc[idx, f'hit_{target_name}_by_day_{day}'] = True

            # Record time to target
            if time_to_target is not None:
                result.loc[idx, f'time_to_{target_name}'] = time_to_target

    return result


def generate_recovery_curves(
    events: pd.DataFrame,
    combo_id: Optional[str] = None,
    max_days: int = 7
) -> pd.DataFrame:
    """
    Generate recovery probability curves across holding periods.

    Computes P(target hit by day k) for k=1..max_days and multiple targets.

    Args:
        events: DataFrame from analyze_recovery_times() with hit_*_by_day_* columns
        combo_id: Optional identifier for this parameter combination
        max_days: Maximum holding period (default 7)

    Returns:
        DataFrame with columns:
            - combo_id (if provided)
            - day_k: Day number (1..max_days)
            - p_breakeven_by_k: Probability of breakeven by day k
            - p_0.5pct_by_k: Probability of +0.5% by day k
            - p_1pct_by_k: Probability of +1.0% by day k
            - p_2pct_by_k: Probability of +2.0% by day k
            - n_events: Number of events contributing to this day's calculation

    Example:
        >>> events_with_recovery = analyze_recovery_times(events)
        >>> curves = generate_recovery_curves(events_with_recovery, combo_id='combo_123')
        >>> print(curves[['day_k', 'p_1pct_by_k']])
    """
    if events.empty:
        raise ValueError("events DataFrame is empty")

    targets = ['breakeven', '0.5pct', '1pct', '2pct']

    # Check required columns exist
    required_cols = []
    for day in range(1, max_days + 1):
        for target in targets:
            required_cols.append(f'hit_{target}_by_day_{day}')

    missing_cols = [col for col in required_cols if col not in events.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns. Run analyze_recovery_times() first. "
            f"Missing: {missing_cols[:5]}..."
        )

    # Build curves
    curves_data = []

    for day in range(1, max_days + 1):
        row = {'day_k': day}

        if combo_id is not None:
            row['combo_id'] = combo_id

        # Count events with data for this day
        n_events = 0
        for target in targets:
            col_name = f'hit_{target}_by_day_{day}'
            # Count non-NaN values
            n_events = max(n_events, events[col_name].notna().sum())

        row['n_events'] = n_events

        # Calculate probabilities for each target
        for target in targets:
            col_name = f'hit_{target}_by_day_{day}'

            if n_events > 0:
                # Probability = (number of True) / (number of non-NaN)
                hits = events[col_name].sum()
                valid_count = events[col_name].notna().sum()
                prob = hits / valid_count if valid_count > 0 else 0.0
            else:
                prob = 0.0

            row[f'p_{target}_by_k'] = prob

        curves_data.append(row)

    curves_df = pd.DataFrame(curves_data)

    return curves_df


def calculate_mfe_mae_stats(
    events: pd.DataFrame,
    max_days: int = 7
) -> Dict[str, float]:
    """
    Calculate Maximum Favorable/Adverse Excursion statistics.

    Args:
        events: DataFrame with forward price columns (high_1d, low_1d, etc.)
            OR close_1d, close_2d, etc. as fallback
        max_days: Maximum holding period to analyze

    Returns:
        Dictionary with MFE/MAE statistics:
            - mean_mfe: Average best price within holding period (%)
            - median_mfe: Median best price (%)
            - mean_mae: Average worst price within holding period (%)
            - median_mae: Median worst price (%)
            - mfe_mae_ratio: mean_mfe / abs(mean_mae)

    Example:
        >>> stats = calculate_mfe_mae_stats(events, max_days=7)
        >>> print(f"Avg MFE: {stats['mean_mfe']:.2%}")
        >>> print(f"Avg MAE: {stats['mean_mae']:.2%}")
    """
    if events.empty:
        return {
            'mean_mfe': 0.0,
            'median_mfe': 0.0,
            'mean_mae': 0.0,
            'median_mae': 0.0,
            'mfe_mae_ratio': 0.0
        }

    mfe_list = []
    mae_list = []

    for idx in events.index:
        entry_price = events.loc[idx, 'entry_price'] if 'entry_price' in events.columns else None

        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            continue

        # Collect high/low prices if available
        highs = []
        lows = []

        for day in range(1, max_days + 1):
            high_col = f'high_{day}d'
            low_col = f'low_{day}d'
            close_col = f'close_{day}d'

            # Use high/low if available, otherwise use close as both
            if high_col in events.columns:
                high_price = events.loc[idx, high_col]
                if not pd.isna(high_price) and high_price > 0:
                    highs.append(high_price)
            elif close_col in events.columns:
                close_price = events.loc[idx, close_col]
                if not pd.isna(close_price) and close_price > 0:
                    highs.append(close_price)

            if low_col in events.columns:
                low_price = events.loc[idx, low_col]
                if not pd.isna(low_price) and low_price > 0:
                    lows.append(low_price)
            elif close_col in events.columns:
                close_price = events.loc[idx, close_col]
                if not pd.isna(close_price) and close_price > 0:
                    lows.append(close_price)

        # Calculate MFE (best high) and MAE (worst low)
        if highs:
            best_high = max(highs)
            mfe = (best_high - entry_price) / entry_price
            mfe_list.append(mfe)

        if lows:
            worst_low = min(lows)
            mae = (worst_low - entry_price) / entry_price
            mae_list.append(mae)

    # Calculate statistics
    mean_mfe = float(np.mean(mfe_list)) if mfe_list else 0.0
    median_mfe = float(np.median(mfe_list)) if mfe_list else 0.0
    mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
    median_mae = float(np.median(mae_list)) if mae_list else 0.0

    # MFE/MAE ratio (risk-adjusted excursion)
    mfe_mae_ratio = mean_mfe / abs(mean_mae) if mean_mae != 0 else 0.0

    return {
        'mean_mfe': mean_mfe,
        'median_mfe': median_mfe,
        'mean_mae': mean_mae,
        'median_mae': median_mae,
        'mfe_mae_ratio': mfe_mae_ratio
    }


def rank_combos_by_recovery_speed(
    curves: pd.DataFrame,
    target_days: int = 3,
    target_level: str = 'breakeven'
) -> pd.DataFrame:
    """
    Rank parameter combinations by recovery speed.

    Args:
        curves: DataFrame from generate_recovery_curves() for multiple combos
        target_days: Number of days to reach target (default 3)
        target_level: Which target to optimize ('breakeven', '0.5pct', '1pct', '2pct')

    Returns:
        DataFrame with combos ranked by recovery probability, sorted descending

    Example:
        >>> ranked = rank_combos_by_recovery_speed(all_curves, target_days=3, target_level='1pct')
        >>> print(ranked.head())
    """
    if 'combo_id' not in curves.columns:
        raise ValueError("curves DataFrame must have 'combo_id' column")

    # Filter to target day
    target_data = curves[curves['day_k'] == target_days].copy()

    if target_data.empty:
        raise ValueError(f"No data for day_k={target_days}")

    # Get probability column
    prob_col = f'p_{target_level}_by_k'

    if prob_col not in target_data.columns:
        raise ValueError(f"Column '{prob_col}' not found. Valid targets: breakeven, 0.5pct, 1pct, 2pct")

    # Sort by probability descending
    ranked = target_data.sort_values(prob_col, ascending=False).reset_index(drop=True)

    # Add rank column
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked[['rank', 'combo_id', prob_col, 'n_events']]


def rank_combos_by_target_hit(
    curves: pd.DataFrame,
    targets_config: Optional[List[Tuple[str, int]]] = None
) -> pd.DataFrame:
    """
    Rank parameter combinations by multi-target hit probability.

    Default scoring: P(+1% by day 3) * 50 + P(+2% by day 7) * 50

    Args:
        curves: DataFrame from generate_recovery_curves() for multiple combos
        targets_config: List of (target_level, day) tuples with weights
            Default: [('1pct', 3, 50), ('2pct', 7, 50)]

    Returns:
        DataFrame with combos ranked by combined score, sorted descending

    Example:
        >>> ranked = rank_combos_by_target_hit(all_curves)
        >>> print(ranked.head())
    """
    if 'combo_id' not in curves.columns:
        raise ValueError("curves DataFrame must have 'combo_id' column")

    # Default config: 50% weight on +1% by day 3, 50% weight on +2% by day 7
    if targets_config is None:
        targets_config = [
            ('1pct', 3, 50.0),
            ('2pct', 7, 50.0)
        ]

    # Calculate scores for each combo
    combo_scores = []

    for combo_id in curves['combo_id'].unique():
        combo_curves = curves[curves['combo_id'] == combo_id]

        score = 0.0
        score_components = {}

        for target_level, target_day, weight in targets_config:
            prob_col = f'p_{target_level}_by_k'
            day_data = combo_curves[combo_curves['day_k'] == target_day]

            if not day_data.empty and prob_col in day_data.columns:
                prob = day_data[prob_col].iloc[0]
                contribution = prob * weight
                score += contribution
                score_components[f'{target_level}_by_{target_day}d'] = prob
            else:
                score_components[f'{target_level}_by_{target_day}d'] = 0.0

        combo_scores.append({
            'combo_id': combo_id,
            'target_hit_score': score,
            **score_components
        })

    ranked = pd.DataFrame(combo_scores).sort_values('target_hit_score', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'rank', range(1, len(ranked) + 1))

    return ranked
