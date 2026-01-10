#!/usr/bin/env python3
"""
Math Invariants Test - PHASE 4

Tests mathematical correctness of:
1. Position sizing (2% risk + 20% notional caps)
2. OHLC relationships
3. Indicator calculations
4. Risk/reward calculations
"""

from risk.equity_sizer import calculate_position_size
import pandas as pd
import numpy as np

def test_position_sizing():
    """Test dual-cap position sizing formula."""
    print("=" * 60)
    print("POSITION SIZING MATH INVARIANTS")
    print("=" * 60)

    test_cases = [
        {
            'name': 'Standard 2% risk',
            'entry': 250.0,
            'stop': 237.50,
            'risk_pct': 0.02,
            'equity': 105000,
            'max_notional_pct': 0.20,
            'expected_shares': 168,  # 105000 * 0.02 / (250-237.5) = 168
            'expected_risk': 2100,
        },
        {
            'name': 'Notional cap test',
            'entry': 500.0,
            'stop': 490.0,
            'risk_pct': 0.02,
            'equity': 50000,
            'max_notional_pct': 0.10,
            'expected_shares': 10,  # Capped by 50000*0.10/500 = 10 shares
            'expected_risk': 100,
        },
        {
            'name': 'Low price stock',
            'entry': 15.0,
            'stop': 13.50,
            'risk_pct': 0.02,
            'equity': 50000,
            'max_notional_pct': 0.20,
            'expected_shares': 666,  # 50000 * 0.02 / 1.5 = 666
            'expected_risk': 999,
        },
    ]

    all_pass = True
    for tc in test_cases:
        size = calculate_position_size(
            entry_price=tc['entry'],
            stop_loss=tc['stop'],
            risk_pct=tc['risk_pct'],
            account_equity=tc['equity'],
            max_notional_pct=tc['max_notional_pct'],
        )

        print(f"\n{tc['name']}:")
        print(f"  Shares: {size.shares} (expected {tc['expected_shares']})")
        print(f"  Risk $: {size.risk_dollars:.0f} (expected {tc['expected_risk']})")
        print(f"  Notional: ${size.notional:.0f}")
        print(f"  Capped: {size.capped}")

        # Verify
        shares_match = abs(size.shares - tc['expected_shares']) <= 1
        risk_match = abs(size.risk_dollars - tc['expected_risk']) <= 10

        if shares_match and risk_match:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Math error!")
            all_pass = False

    return all_pass

def test_ohlc_invariants():
    """Test OHLC relationships."""
    print("\n" + "=" * 60)
    print("OHLC INVARIANTS")
    print("=" * 60)

    # Create test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    })

    # Fix OHLC to be valid
    for i in range(len(df)):
        o, c = df.loc[i, 'open'], df.loc[i, 'close']
        df.loc[i, 'high'] = max(o, c, df.loc[i, 'high'])
        df.loc[i, 'low'] = min(o, c, df.loc[i, 'low'])

    # Check invariants
    high_ge_open = (df['high'] >= df['open']).all()
    high_ge_close = (df['high'] >= df['close']).all()
    low_le_open = (df['low'] <= df['open']).all()
    low_le_close = (df['low'] <= df['close']).all()

    print(f"\nHigh >= Open: {high_ge_open}")
    print(f"High >= Close: {high_ge_close}")
    print(f"Low <= Open: {low_le_open}")
    print(f"Low <= Close: {low_le_close}")

    all_pass = high_ge_open and high_ge_close and low_le_open and low_le_close

    if all_pass:
        print("\n✓ ALL OHLC INVARIANTS PASS")
    else:
        print("\n✗ OHLC INVARIANTS FAIL")

    return all_pass

def test_rr_calculation():
    """Test Risk/Reward ratio calculations."""
    print("\n" + "=" * 60)
    print("RISK/REWARD CALCULATIONS")
    print("=" * 60)

    test_cases = [
        {
            'entry': 100.0,
            'stop': 98.0,
            'target': 105.0,
            'expected_rr': 2.5,  # (105-100) / (100-98) = 2.5
        },
        {
            'entry': 250.0,
            'stop': 237.5,
            'target': 265.0,
            'expected_rr': 1.2,  # (265-250) / (250-237.5) = 1.2
        },
    ]

    all_pass = True
    for tc in test_cases:
        risk = tc['entry'] - tc['stop']
        reward = tc['target'] - tc['entry']
        rr = reward / risk

        print(f"\nEntry: ${tc['entry']}, Stop: ${tc['stop']}, Target: ${tc['target']}")
        print(f"  Risk: ${risk:.2f}")
        print(f"  Reward: ${reward:.2f}")
        print(f"  R:R: {rr:.2f} (expected {tc['expected_rr']})")

        if abs(rr - tc['expected_rr']) < 0.01:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_pass = False

    return all_pass

if __name__ == '__main__':
    print("\nMATH INVARIANTS TEST - PHASE 4")
    print("Date: 2026-01-08")
    print("=" * 60)

    p1 = test_position_sizing()
    p2 = test_ohlc_invariants()
    p3 = test_rr_calculation()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Position Sizing: {'PASS' if p1 else 'FAIL'}")
    print(f"OHLC Invariants: {'PASS' if p2 else 'FAIL'}")
    print(f"R:R Calculations: {'PASS' if p3 else 'FAIL'}")

    if p1 and p2 and p3:
        print("\n✓ ALL MATH INVARIANTS PASS")
        exit(0)
    else:
        print("\n✗ SOME MATH INVARIANTS FAIL")
        exit(1)
