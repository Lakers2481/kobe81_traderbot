"""
QUANT AUDIT: MARKOV 5-DOWN-DAY CLAIM
=====================================
No-BS verification of the Renaissance Markov claim.

EXACT RULE:
- Up day = daily return >= 0
- Down day = daily return < 0
- Return = (Close_t - Close_{t-1}) / Close_{t-1}

TESTS:
- SPY: 2010-2022 (video window) and 2015-2025 (Claude window)
- 10 symbols aggregate
- Both Close and Close
- Sensitivity: streak lengths 3,4,5,6,7
- Split test: first half vs second half
- 95% Wilson confidence intervals
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from datetime import datetime

def wilson_confidence_interval(successes, trials, alpha=0.05):
    """
    Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation for small samples.
    """
    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    z = stats.norm.ppf(1 - alpha/2)

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

    return max(0, center - margin), min(1, center + margin)

def test_consecutive_down_pattern(
    df: pd.DataFrame,
    price_col: str,
    n_days: int = 5
) -> dict:
    """
    Test N consecutive down days -> next day probability.

    Args:
        df: DataFrame with OHLC data
        price_col: 'Close' or 'Close'
        n_days: Streak length

    Returns:
        dict with all statistics
    """
    # Calculate returns
    df = df.copy()
    df['return'] = df[price_col].pct_change()
    df['is_down'] = df['return'] < 0

    # Find pattern matches
    matches = []
    for i in range(n_days, len(df) - 1):
        # Check if previous N days were ALL down
        if all(df['is_down'].iloc[i-n_days+j] for j in range(n_days)):
            next_return = df['return'].iloc[i]
            next_up = next_return >= 0
            matches.append({
                'date': df.index[i-1],
                'next_date': df.index[i],
                'next_return': next_return,
                'next_up': next_up
            })

    # Calculate statistics
    if not matches:
        return {
            'instances': 0,
            'next_up': 0,
            'prob_up': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'mean_next_ret': 0.0,
            'uncond_mean_ret': df['return'].mean(),
            'lift': 0.0,
            'matches': []
        }

    matches_df = pd.DataFrame(matches)
    instances = len(matches_df)
    next_up = matches_df['next_up'].sum()
    prob_up = next_up / instances

    # Wilson confidence interval
    ci_lower, ci_upper = wilson_confidence_interval(next_up, instances)

    # Mean returns
    mean_next_ret = matches_df['next_return'].mean()
    uncond_mean_ret = df['return'].mean()
    lift = mean_next_ret / uncond_mean_ret if uncond_mean_ret != 0 else 0

    return {
        'instances': instances,
        'next_up': int(next_up),
        'prob_up': prob_up,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_next_ret': mean_next_ret,
        'uncond_mean_ret': uncond_mean_ret,
        'lift': lift,
        'matches': matches
    }

def run_full_audit():
    """Run complete audit with all tests."""

    print("=" * 100)
    print("QUANT AUDIT: MARKOV CONSECUTIVE DOWN-DAY PATTERN")
    print("=" * 100)
    print()

    # PART A: SPY with multiple windows and price series
    print("PART A: SPY ANALYSIS")
    print("-" * 100)

    windows = [
        ('2010-01-01', '2022-12-31', 'Video Window'),
        ('2015-01-01', '2025-12-31', 'Claude Window'),
    ]

    price_series = ['Close', 'Close']
    streak_lengths = [3, 4, 5, 6, 7]

    results_a = []

    for start, end, window_name in windows:
        # Download data
        spy = yf.Ticker('SPY')
        df = spy.history(start=start, end=end)

        # Check available columns and map
        available_cols = df.columns.tolist()
        print(f"DEBUG: Available columns for {window_name}: {available_cols}")

        for price_col in price_series:
            # Skip if column doesn't exist
            if price_col not in available_cols:
                print(f"WARNING: {price_col} not in data, skipping")
                continue
            for streak in streak_lengths:
                result = test_consecutive_down_pattern(df, price_col, streak)

                results_a.append({
                    'Symbol': 'SPY',
                    'Window': window_name,
                    'Start': start,
                    'End': end,
                    'PriceSeries': price_col,
                    'Streak': streak,
                    'Instances': result['instances'],
                    'NextUp': result['next_up'],
                    'P(Up)': result['prob_up'],
                    'CI_Lower': result['ci_lower'],
                    'CI_Upper': result['ci_upper'],
                    'MeanNextRet': result['mean_next_ret'],
                    'UncondMeanRet': result['uncond_mean_ret'],
                    'Lift': result['lift']
                })

    # Print Part A results
    df_a = pd.DataFrame(results_a)
    print(df_a.to_string(index=False))
    print()

    # PART B: 10 Symbols
    print("=" * 100)
    print("PART B: 10-SYMBOL AGGREGATE ANALYSIS")
    print("-" * 100)

    symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    start = '2015-01-01'
    end = '2025-12-31'
    price_col = 'Close'  # Use adjusted for consistency
    streak = 5  # Focus on 5-day streak

    results_b = []
    total_instances = 0
    total_next_up = 0

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)

            if df.empty:
                print(f"WARNING: No data for {symbol}")
                continue

            result = test_consecutive_down_pattern(df, price_col, streak)

            results_b.append({
                'Symbol': symbol,
                'Instances': result['instances'],
                'NextUp': result['next_up'],
                'P(Up)': result['prob_up'],
                'CI_Lower': result['ci_lower'],
                'CI_Upper': result['ci_upper'],
                'MeanNextRet': result['mean_next_ret'],
                'UncondMeanRet': result['uncond_mean_ret'],
                'Lift': result['lift']
            })

            total_instances += result['instances']
            total_next_up += result['next_up']

        except Exception as e:
            print(f"ERROR {symbol}: {e}")
            continue

    # Print per-symbol results
    df_b = pd.DataFrame(results_b)
    print("PER-SYMBOL RESULTS:")
    print(df_b.to_string(index=False))
    print()

    # Aggregate results
    if total_instances > 0:
        agg_prob = total_next_up / total_instances
        agg_ci_lower, agg_ci_upper = wilson_confidence_interval(total_next_up, total_instances)

        print("AGGREGATE RESULTS (10 symbols combined):")
        print(f"  Total Instances: {total_instances}")
        print(f"  Total Next Up: {total_next_up}")
        print(f"  Aggregate P(Up): {agg_prob:.3f} ({agg_prob:.1%})")
        print(f"  95% CI: [{agg_ci_lower:.3f}, {agg_ci_upper:.3f}]")
        print(f"  Claimed: 0.66 (66%)")
        print(f"  Difference: {abs(agg_prob - 0.66):.3f}")
        print()

        # Verdict
        if 0.60 <= agg_prob <= 0.72:  # Within reasonable range
            print("VERDICT: REPRODUCED")
            print(f"  The claim of ~66% is supported by data (measured: {agg_prob:.1%})")
        else:
            print("VERDICT: NOT REPRODUCED")
            print(f"  The claim of 66% is NOT supported (measured: {agg_prob:.1%})")
    else:
        print("VERDICT: NOT REPRODUCED (no instances found)")

    print()

    # PART A SPLIT TEST
    print("=" * 100)
    print("PART A.2: SPLIT TEST (First Half vs Second Half)")
    print("-" * 100)

    for start, end, window_name in windows:
        spy = yf.Ticker('SPY')
        df = spy.history(start=start, end=end)

        mid_idx = len(df) // 2
        df_first = df.iloc[:mid_idx]
        df_second = df.iloc[mid_idx:]

        result_first = test_consecutive_down_pattern(df_first, 'Close', 5)
        result_second = test_consecutive_down_pattern(df_second, 'Close', 5)

        print(f"\n{window_name} ({start} to {end}):")
        print(f"  First Half:  {result_first['instances']} instances, P(Up)={result_first['prob_up']:.3f}")
        print(f"  Second Half: {result_second['instances']} instances, P(Up)={result_second['prob_up']:.3f}")

        if result_first['instances'] > 0 and result_second['instances'] > 0:
            diff = abs(result_first['prob_up'] - result_second['prob_up'])
            print(f"  Difference: {diff:.3f}")
            if diff > 0.15:
                print(f"  WARNING: Large difference suggests instability")

    print()
    print("=" * 100)
    print("END OF AUDIT")
    print("=" * 100)

if __name__ == '__main__':
    run_full_audit()
