"""
VERIFY RENAISSANCE MARKOV CLAIM
================================
Test the claim: "After 5 consecutive down days, there's a 66% probability of an up day"

This uses REAL historical data to verify the claim.

Usage:
    python tools/verify_5_down_day_claim.py

Data Source:
    - Primary: Polygon.io (if API key available)
    - Fallback: Yahoo Finance (free, no API key)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_consecutive_down_pattern(symbol: str, n_days: int, start: str, end: str):
    """
    Test: N consecutive down days -> up day probability

    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        n_days: Number of consecutive down days (e.g., 5)
        start: Start date 'YYYY-MM-DD'
        end: End date 'YYYY-MM-DD'

    Returns:
        dict with results
    """

    # Try to load data using yfinance directly
    df = None
    data_source = "Yahoo Finance (yfinance)"

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)

        if df.empty:
            return {
                'error': f'No data returned for {symbol}',
                'sample_size': 0,
                'claim_verified': False
            }

        # Rename columns to match expected format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

    except Exception as e:
        print(f"FAIL Yahoo Finance failed: {e}")
        return {
            'error': 'Could not load data from any source',
            'sample_size': 0,
            'claim_verified': False
        }

    # Calculate daily returns
    df['return'] = df['close'].pct_change()
    df['is_down'] = df['return'] < 0

    # Find all instances of N consecutive down days
    matches = []
    for i in range(n_days, len(df) - 1):  # -1 to ensure we have next day
        # Check if previous N days were ALL down
        consecutive_down = all(df['is_down'].iloc[i-n_days+j] for j in range(n_days))

        if consecutive_down:
            # Record what happened on the NEXT day
            next_day_return = df['return'].iloc[i]
            next_day_up = next_day_return > 0

            matches.append({
                'date': df.index[i-1].strftime('%Y-%m-%d'),  # Last down day
                'next_day': df.index[i].strftime('%Y-%m-%d'),
                'next_day_up': next_day_up,
                'next_day_return': next_day_return
            })

    # Calculate statistics
    if not matches:
        return {
            'symbol': symbol,
            'n_days': n_days,
            'sample_size': 0,
            'up_probability': 0.0,
            'claim_verified': False,
            'error': f'No instances of {n_days} consecutive down days found',
            'data_source': data_source
        }

    matches_df = pd.DataFrame(matches)
    up_count = matches_df['next_day_up'].sum()
    total_count = len(matches_df)
    up_prob = up_count / total_count

    # Calculate confidence interval (95%)
    from scipy import stats
    ci = stats.binom.interval(0.95, total_count, up_prob)
    ci_lower = ci[0] / total_count
    ci_upper = ci[1] / total_count

    return {
        'symbol': symbol,
        'n_days': n_days,
        'period': f'{start} to {end}',
        'data_source': data_source,
        'sample_size': total_count,
        'up_count': int(up_count),
        'down_count': int(total_count - up_count),
        'up_probability': up_prob,
        'confidence_interval_95': (ci_lower, ci_upper),
        'claim_66pct': 0.66,
        'difference': abs(up_prob - 0.66),
        'claim_verified': abs(up_prob - 0.66) < 0.10,  # Within 10% margin
        'matches': matches_df.to_dict('records')
    }

if __name__ == '__main__':
    print("="*80)
    print("TESTING RENAISSANCE MARKOV CLAIM")
    print("Claim: After 5 consecutive down days -> 66% probability of up day")
    print("="*80)
    print()

    # Test on SPY (S&P 500) - 10 years of data
    symbol = 'SPY'
    n_days = 5
    start = '2015-01-01'
    end = '2025-12-31'

    print(f"Loading historical data for {symbol}...")
    print(f"Period: {start} to {end}")
    print()

    result = test_consecutive_down_pattern(symbol, n_days, start, end)

    if 'error' in result and result['sample_size'] == 0:
        print(f"FAIL ERROR: {result['error']}")
        sys.exit(1)

    # Display results
    print("RESULTS:")
    print(f"  Data Source: {result['data_source']}")
    print(f"  Pattern: {result['n_days']} consecutive down days")
    print(f"  Sample Size: {result['sample_size']} instances found")
    print(f"  Next Day Up: {result['up_count']} ({result['up_count']/result['sample_size']*100:.1f}%)")
    print(f"  Next Day Down: {result['down_count']} ({result['down_count']/result['sample_size']*100:.1f}%)")
    print()
    print(f"  Measured Probability: {result['up_probability']:.1%}")
    print(f"  95% Confidence Interval: ({result['confidence_interval_95'][0]:.1%}, {result['confidence_interval_95'][1]:.1%})")
    print(f"  Claimed Probability: 66%")
    print(f"  Difference: {result['difference']:.1%}")
    print()

    # Verdict
    if result['claim_verified']:
        print("PASS CLAIM VERIFIED (within 10% margin)")
        verdict = 0
    else:
        print("FAIL CLAIM REJECTED (outside 10% margin)")
        print(f"  The actual probability is {result['up_probability']:.1%}, not 66%")
        verdict = 1

    print("="*80)
    print()

    # Show some examples
    if result['sample_size'] > 0:
        print("EXAMPLE INSTANCES (first 10):")
        for i, match in enumerate(result['matches'][:10], 1):
            direction = "UP PASS" if match['next_day_up'] else "DOWN FAIL"
            print(f"  {i}. {match['date']} -> {match['next_day']}: {direction} ({match['next_day_return']:+.2%})")
        print("="*80)

    sys.exit(verdict)
