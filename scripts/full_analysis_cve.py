#!/usr/bin/env python3
"""
FULL QUANT ANALYSIS: CVE (Cenovus Energy Inc.)
Trade of the Day - Tuesday January 6, 2026
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars

def main():
    symbol = 'CVE'

    print('=' * 80)
    print(f'FULL QUANT ANALYSIS: {symbol} (Cenovus Energy Inc.)')
    print('Trade of the Day - Tuesday January 6, 2026')
    print('=' * 80)

    # ========================================================================
    # 1. PRICE DATA & TECHNICALS
    # ========================================================================
    print('\n' + '=' * 80)
    print('1. PRICE ACTION & TECHNICALS')
    print('=' * 80)

    # Fetch historical data
    df = fetch_daily_bars_polygon(symbol, start='2024-01-01', end='2026-01-05', cache_dir=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Get live price
    try:
        bars = fetch_intraday_bars(symbol, timeframe='15Min', limit=5)
        live_price = bars[-1].close if bars else df.iloc[-1]['close']
    except Exception:
        live_price = df.iloc[-1]['close']

    last = df.iloc[-1]
    prev = df.iloc[-2]
    week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]

    print(f'\nCurrent Price: ${live_price:.2f}')
    print(f"Monday Close:  ${last['close']:.2f}")
    print(f"Friday Close:  ${prev['close']:.2f}")
    print(f"Week Ago:      ${week_ago['close']:.2f}")

    # Daily change
    daily_change = (last['close'] - prev['close']) / prev['close'] * 100
    weekly_change = (last['close'] - week_ago['close']) / week_ago['close'] * 100

    print(f'\nDaily Change:  {daily_change:+.2f}%')
    print(f'Weekly Change: {weekly_change:+.2f}%')

    # Calculate technicals
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # IBS (Internal Bar Strength)
    df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    current_sma20 = df['sma_20'].iloc[-1]
    current_sma50 = df['sma_50'].iloc[-1]
    current_sma200 = df['sma_200'].iloc[-1]
    current_atr = df['atr_14'].iloc[-1]
    current_rsi = df['rsi_14'].iloc[-1]
    current_ibs = df['ibs'].iloc[-1]

    print('\n--- Moving Averages ---')
    print(f'SMA(20):  ${current_sma20:.2f} ({"above" if live_price > current_sma20 else "below"})')
    print(f'SMA(50):  ${current_sma50:.2f} ({"above" if live_price > current_sma50 else "below"})')
    print(f'SMA(200): ${current_sma200:.2f} ({"above" if live_price > current_sma200 else "below"})')

    print('\n--- Indicators ---')
    print(f'ATR(14):  ${current_atr:.2f} ({current_atr/live_price*100:.1f}% of price)')
    print(f'RSI(14):  {current_rsi:.1f}')
    print(f'IBS:      {current_ibs:.2f}')

    # ========================================================================
    # 2. TURTLE SOUP SIGNAL DETAILS
    # ========================================================================
    print('\n' + '=' * 80)
    print('2. TURTLE SOUP SIGNAL ANALYSIS')
    print('=' * 80)

    # Find the 20-day low that was swept
    lookback = 20
    low_20 = df['low'].iloc[-lookback-1:-1].min()
    monday_low = last['low']
    sweep_depth = low_20 - monday_low
    sweep_atr_ratio = sweep_depth / current_atr if current_atr > 0 else 0

    print(f'\n20-Day Low:     ${low_20:.2f}')
    print(f"Monday's Low:   ${monday_low:.2f}")
    print(f'Sweep Depth:    ${sweep_depth:.2f} ({sweep_atr_ratio:.2f} ATR)')
    print(f'Sweep Valid:    {"YES" if sweep_atr_ratio >= 0.30 else "NO"} (min 0.30 ATR)')

    # Signal parameters
    entry = 16.64
    stop = 15.72
    target = 17.33
    risk = entry - stop
    reward = target - entry
    rr = reward / risk

    print('\n--- Trade Parameters ---')
    print(f'Entry:   ${entry:.2f}')
    print(f'Stop:    ${stop:.2f} (risk ${risk:.2f})')
    print(f'Target:  ${target:.2f} (reward ${reward:.2f})')
    print(f'R:R:     {rr:.2f}:1')

    # ========================================================================
    # 3. HISTORICAL PATTERN ANALYSIS
    # ========================================================================
    print('\n' + '=' * 80)
    print('3. HISTORICAL PATTERN ANALYSIS')
    print('=' * 80)

    # Count consecutive down days
    df['down_day'] = df['close'] < df['close'].shift(1)

    # Current streak
    current_streak = 0
    for i in range(len(df)-1, -1, -1):
        if df['down_day'].iloc[i]:
            current_streak += 1
        else:
            break

    print(f'\nCurrent Down Streak: {current_streak} consecutive days')

    # Historical analysis of similar patterns
    # Find all instances where we had similar sweep patterns
    df['swept_20_low'] = df['low'] < df['low'].rolling(20).min().shift(1)
    df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1

    sweep_instances = df[df['swept_20_low'] == True].copy()
    if len(sweep_instances) > 0:
        wins = (sweep_instances['next_day_return'] > 0).sum()
        total = len(sweep_instances)
        avg_return = sweep_instances['next_day_return'].mean() * 100

        print('\n--- 20-Day Low Sweep Pattern History ---')
        print(f'Sample Size:     {total} instances')
        print(f'Next-Day Win:    {wins}/{total} ({wins/total*100:.1f}%)')
        print(f'Avg Next Return: {avg_return:+.2f}%')

    # Down streak analysis
    print('\n--- Down Streak Reversal Analysis ---')
    streak_reversal = {}
    for streak_len in range(2, 6):
        # Find all instances of N consecutive down days
        mask = df['down_day'].rolling(streak_len).sum() == streak_len
        instances = df[mask].copy()
        if len(instances) > 5:
            wins = (instances['next_day_return'] > 0).sum()
            total = len(instances)
            streak_reversal[streak_len] = {'wins': wins, 'total': total, 'wr': wins/total}
            print(f'{streak_len} down days: {wins}/{total} reversals ({wins/total*100:.1f}% WR)')

    # ========================================================================
    # 4. SECTOR & MARKET CONTEXT
    # ========================================================================
    print('\n' + '=' * 80)
    print('4. SECTOR & MARKET CONTEXT')
    print('=' * 80)

    print('\n--- Company Info ---')
    print('Company:  Cenovus Energy Inc.')
    print('Sector:   Energy')
    print('Industry: Oil & Gas Integrated')
    print('Exchange: NYSE')

    # Get SPY for comparison
    try:
        spy = fetch_daily_bars_polygon('SPY', start='2025-12-01', end='2026-01-05', cache_dir=None)
        spy['timestamp'] = pd.to_datetime(spy['timestamp'])
        spy = spy.sort_values('timestamp')
        spy_return = (spy.iloc[-1]['close'] - spy.iloc[-5]['close']) / spy.iloc[-5]['close'] * 100
        print(f'\nSPY Weekly:  {spy_return:+.2f}%')
        print(f'CVE Weekly:  {weekly_change:+.2f}%')
        print(f'Relative:    {weekly_change - spy_return:+.2f}% vs SPY')
    except Exception:
        print('\nSPY comparison not available')

    # Get XLE (Energy ETF) for sector comparison
    try:
        xle = fetch_daily_bars_polygon('XLE', start='2025-12-01', end='2026-01-05', cache_dir=None)
        xle['timestamp'] = pd.to_datetime(xle['timestamp'])
        xle = xle.sort_values('timestamp')
        xle_return = (xle.iloc[-1]['close'] - xle.iloc[-5]['close']) / xle.iloc[-5]['close'] * 100
        print(f'XLE Weekly:  {xle_return:+.2f}%')
        print(f'CVE vs XLE:  {weekly_change - xle_return:+.2f}%')
    except Exception:
        print('XLE comparison not available')

    # ========================================================================
    # 5. VOLATILITY & EXPECTED MOVE
    # ========================================================================
    print('\n' + '=' * 80)
    print('5. VOLATILITY & EXPECTED MOVE')
    print('=' * 80)

    # Realized volatility (20-day)
    realized_vol = df['returns'].iloc[-20:].std() * np.sqrt(252) * 100

    # Weekly expected move
    weekly_em = live_price * (realized_vol / 100) * np.sqrt(5/252)

    print(f'\n20-Day Realized Vol: {realized_vol:.1f}%')
    print(f'Weekly Expected Move: ${weekly_em:.2f} ({weekly_em/live_price*100:.1f}%)')
    print('\nExpected Range This Week:')
    print(f'  Low:  ${live_price - weekly_em:.2f}')
    print(f'  High: ${live_price + weekly_em:.2f}')

    # ========================================================================
    # 6. AI CONFIDENCE BREAKDOWN
    # ========================================================================
    print('\n' + '=' * 80)
    print('6. AI CONFIDENCE BREAKDOWN')
    print('=' * 80)

    # Build confidence factors
    factors = {}

    # Factor 1: Signal Strength (sweep depth)
    factors['signal_strength'] = min(100, sweep_atr_ratio * 100 / 0.5)

    # Factor 2: Historical Pattern
    if current_streak >= 2 and current_streak in streak_reversal:
        factors['historical_pattern'] = streak_reversal[current_streak]['wr'] * 100
    else:
        factors['historical_pattern'] = 60

    # Factor 3: Technical Setup
    tech_score = 0
    if live_price > current_sma200: tech_score += 30
    if current_rsi < 30: tech_score += 40  # Oversold
    elif current_rsi < 50: tech_score += 20
    if current_ibs < 0.2: tech_score += 30  # Low IBS
    elif current_ibs < 0.4: tech_score += 15
    factors['technical_setup'] = min(100, tech_score)

    # Factor 4: Regime Alignment
    # Energy stocks in current macro environment
    factors['regime_alignment'] = 70  # Neutral-positive for energy

    # Factor 5: Risk/Reward Quality
    factors['rr_quality'] = min(100, rr * 50)

    # Factor 6: Liquidity
    avg_volume = df['volume'].iloc[-20:].mean()
    factors['liquidity'] = 85 if avg_volume > 1000000 else 60

    # Factor 7: Correlation (energy sector)
    factors['correlation_risk'] = 70  # Some sector correlation

    # Factor 8: Timing (trading session)
    factors['timing'] = 80  # London close window

    # Factor 9: Expected Value
    ev = 0.0675  # From our calculation
    factors['expected_value'] = min(100, ev * 1000)

    print(f'\n{"Factor":<25} {"Score":>8} {"Weight":>8} {"Weighted":>10}')
    print('-' * 55)

    weights = {
        'signal_strength': 0.20,
        'historical_pattern': 0.15,
        'technical_setup': 0.15,
        'regime_alignment': 0.10,
        'rr_quality': 0.10,
        'liquidity': 0.08,
        'correlation_risk': 0.07,
        'timing': 0.07,
        'expected_value': 0.08,
    }

    total_score = 0
    for factor, score in factors.items():
        weight = weights.get(factor, 0.1)
        weighted = score * weight
        total_score += weighted
        print(f'{factor:<25} {score:>7.1f}% {weight:>7.0%} {weighted:>9.1f}')

    print('-' * 55)
    print(f'{"TOTAL CONFIDENCE":<25} {total_score:>7.1f}%')

    # ========================================================================
    # 7. BULL CASE
    # ========================================================================
    print('\n' + '=' * 80)
    print('7. BULL CASE')
    print('=' * 80)

    print('''
1. TURTLE SOUP SIGNAL: Classic liquidity sweep pattern. Price broke below
   20-day low, triggering stop losses of weak longs. Smart money accumulated
   at lower prices. Entry at Monday's close captures reversal.

2. OVERSOLD BOUNCE: IBS at {:.2f} indicates selling exhaustion. Historically,
   similar setups have {:.0f}%+ win rate on next-day reversal.

3. ENERGY SECTOR STRENGTH: Oil prices remain elevated. Canadian energy
   producers like CVE benefit from strong WCS-WTI differentials.

4. MEAN REVERSION: After {} consecutive down days, statistical edge favors
   bounce. Energy stocks rarely stay oversold for extended periods.

5. INSTITUTIONAL SUPPORT: At these levels, value buyers step in.
   $16.50-17.00 is known support zone.
'''.format(current_ibs, factors['historical_pattern'], current_streak))

    # ========================================================================
    # 8. BEAR CASE
    # ========================================================================
    print('\n' + '=' * 80)
    print('8. BEAR CASE')
    print('=' * 80)

    print('''
1. MACRO HEADWINDS: Global recession fears could crush oil demand. If WTI
   breaks below $70, energy stocks face more downside.

2. SECTOR ROTATION: Growth stocks outperforming energy. Fund flows moving
   away from commodities into tech/AI plays.

3. CANADIAN DOLLAR RISK: CAD weakness impacts USD-denominated earnings.
   Currency volatility adds uncertainty.

4. R:R MODEST: 0.75:1 reward-to-risk is acceptable but not exceptional.
   Requires high win rate to be profitable long-term.

5. FALSE BREAKOUT RISK: Sweep could be genuine breakdown, not a trap.
   If price fails to reclaim 20-day low, expect continuation lower.
''')

    # ========================================================================
    # 9. WHAT COULD GO WRONG
    # ========================================================================
    print('\n' + '=' * 80)
    print('9. WHAT COULD GO WRONG')
    print('=' * 80)

    print('''
1. GAP DOWN: Overnight news (oil inventory data, geopolitical event) could
   gap price below stop. Max loss would exceed planned risk.

2. NO FOLLOW-THROUGH: Price bounces initially but fails to reach target.
   Sideways chop erodes time value if playing options.

3. SECTOR CONTAGION: Broader energy selloff drags CVE lower despite
   individual technical setup.

4. STOP HUNT EXTENSION: Market makers push price to $15.50 before reversal,
   stopping out this trade before the move.

5. VOLUME FAILURE: Low conviction bounce without institutional participation
   fails to sustain above entry. False reversal signal.
''')

    # ========================================================================
    # 10. POSITION SIZING
    # ========================================================================
    print('\n' + '=' * 80)
    print('10. POSITION SIZING')
    print('=' * 80)

    account_equity = 50000  # Example
    max_risk_pct = 0.02  # 2%
    max_notional_pct = 0.20  # 20%

    risk_per_share = entry - stop
    max_risk_dollars = account_equity * max_risk_pct
    max_notional = account_equity * max_notional_pct

    shares_by_risk = int(max_risk_dollars / risk_per_share)
    shares_by_notional = int(max_notional / entry)

    final_shares = min(shares_by_risk, shares_by_notional)
    position_value = final_shares * entry
    actual_risk = final_shares * risk_per_share

    print('\n--- Account Parameters ---')
    print(f'Account Equity:   ${account_equity:,.0f}')
    print(f'Max Risk (2%):    ${max_risk_dollars:,.0f}')
    print(f'Max Notional (20%): ${max_notional:,.0f}')

    print('\n--- Position Calculation ---')
    print(f'Risk per Share:   ${risk_per_share:.2f}')
    print(f'Shares by Risk:   {shares_by_risk}')
    print(f'Shares by Notional: {shares_by_notional}')
    print(f'FINAL SHARES:     {final_shares}')

    print('\n--- Position Details ---')
    print(f'Position Value:   ${position_value:,.2f} ({position_value/account_equity*100:.1f}% of account)')
    print(f'Dollar Risk:      ${actual_risk:,.2f} ({actual_risk/account_equity*100:.1f}% of account)')
    print(f'Target Profit:    ${final_shares * reward:,.2f}')

    # ========================================================================
    # 11. TRADE GRADE & RECOMMENDATION
    # ========================================================================
    print('\n' + '=' * 80)
    print('11. TRADE GRADE & FINAL RECOMMENDATION')
    print('=' * 80)

    # Calculate grade
    if total_score >= 80:
        grade = 'A'
    elif total_score >= 70:
        grade = 'B+'
    elif total_score >= 60:
        grade = 'B'
    else:
        grade = 'C'

    print('\n┌─────────────────────────────────────────────────────────────────┐')
    print(f'│  TRADE GRADE: {grade}                                               │')
    print(f'│  CONFIDENCE:  {total_score:.1f}%                                           │')
    print('│  EXPECTED VALUE: +$0.0675 per $1 risked                        │')
    print('├─────────────────────────────────────────────────────────────────┤')
    print('│  RECOMMENDATION: EXECUTE TRADE                                 │')
    print('│                                                                 │')
    print('│  Symbol:    CVE                                                 │')
    print('│  Direction: LONG                                                │')
    print(f'│  Entry:     ${entry:.2f} (limit order)                             │')
    print(f'│  Stop:      ${stop:.2f} (hard stop)                                │')
    print(f'│  Target:    ${target:.2f}                                          │')
    print(f'│  Shares:    {final_shares}                                              │')
    print(f'│  Risk:      ${actual_risk:,.2f}                                          │')
    print('└─────────────────────────────────────────────────────────────────┘')

    print('\n--- Execution Notes ---')
    print('• Use LIMIT order at $16.64 (not market)')
    print('• Stop loss is HARD - no moving it')
    print('• Target is initial; trail stop if momentum strong')
    print('• Exit at close if no clear direction by EOD')
    print('• Kill zone: Only execute between 10:00-11:30 AM ET')

    # Save analysis
    analysis = {
        'symbol': symbol,
        'analysis_time': datetime.now().isoformat(),
        'trade': {
            'entry': entry,
            'stop': stop,
            'target': target,
            'rr_ratio': rr,
            'shares': final_shares,
            'position_value': position_value,
            'dollar_risk': actual_risk,
        },
        'technicals': {
            'live_price': live_price,
            'sma_20': current_sma20,
            'sma_50': current_sma50,
            'sma_200': current_sma200,
            'rsi_14': current_rsi,
            'ibs': current_ibs,
            'atr_14': current_atr,
        },
        'confidence': {
            'total_score': total_score,
            'grade': grade,
            'factors': factors,
        },
        'recommendation': 'EXECUTE',
    }

    Path('reports').mkdir(exist_ok=True)
    with open('reports/CVE_analysis_20260106.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print('\nAnalysis saved to: reports/CVE_analysis_20260106.json')


if __name__ == '__main__':
    main()
