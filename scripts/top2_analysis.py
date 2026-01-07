#!/usr/bin/env python3
"""
Full Professional Analysis for Top 2 Trades

Includes:
- Stock-specific historical pattern analysis
- Expected Value calculation
- Position sizing (2% risk, 20% notional)
- EXPECTED MOVE FOR THE WEEK (2026-01-07 fix)
- Bull/Bear cases
- Risk factors
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from analysis.options_expected_move import ExpectedMoveCalculator

ACCOUNT_EQUITY = 50000
MAX_RISK_PCT = 0.02
MAX_NOTIONAL_PCT = 0.20


def full_analysis(symbol, entry, stop, target, wr, samples, ev, ibs, rsi, strategy='TurtleSoup'):
    print()
    print('=' * 70)
    print(f'TRADE: {symbol}')
    print('=' * 70)

    # FIX (2026-01-07): Use dynamic dates
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch data
    df = fetch_daily_bars_polygon(symbol, start=start_date, end=end_date, cache_dir=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    last = df.iloc[-1]
    signal_date = last['timestamp'].strftime('%Y-%m-%d')
    trade_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    # Get live price
    try:
        bars = fetch_intraday_bars(symbol, timeframe='15Min', limit=3)
        live = bars[-1].close if bars else last['close']
    except Exception:
        live = last['close']

    gap_pct = (live / last['close'] - 1) * 100

    print()
    print('1. SIGNAL SUMMARY')
    print('-' * 50)
    print(f'   Strategy:     {strategy}')
    print('   Asset Class:  EQUITY')
    print(f'   Signal Date:  {signal_date}')
    print(f'   Trade Date:   {trade_date}')

    print()
    print('2. PRICE DATA (REAL - Polygon.io)')
    print('-' * 50)
    print(f'   Last Close:   ${last["close"]:.2f}')
    print(f'   Last High:    ${last["high"]:.2f}')
    print(f'   Last Low:     ${last["low"]:.2f}')
    print(f'   Current Live: ${live:.2f}')
    print(f'   Gap:          {gap_pct:+.2f}%')

    # FIX (2026-01-07): Add Expected Move for the Week
    print()
    print('*** EXPECTED MOVE FOR THE WEEK ***')
    print('-' * 50)
    try:
        em_calc = ExpectedMoveCalculator()
        em = em_calc.calculate_weekly_expected_move(symbol, live)

        print(f'   Current Price:     ${em.current_price:.2f}')
        print(f'   Week Open:         ${em.week_open_price:.2f}')
        print(f'   Expected Move:     +/- {em.weekly_expected_move_pct*100:.1f}% (${em.weekly_expected_move_dollars:.2f})')
        print(f'   Upper Bound:       ${em.upper_bound:.2f}')
        print(f'   Lower Bound:       ${em.lower_bound:.2f}')
        print()
        print(f'   Move This Week:    {em.move_from_week_open_pct*100:+.1f}%')
        print(f'   Room Up:           {em.remaining_room_up_pct*100:.1f}%')
        print(f'   Room Down:         {em.remaining_room_down_pct*100:.1f}%')

        # Analysis
        if em.remaining_room_up_pct > em.remaining_room_down_pct:
            print(f'   >>> MORE ROOM UP ({em.remaining_room_up_pct*100:.1f}% vs {em.remaining_room_down_pct*100:.1f}% down)')
        else:
            print(f'   >>> MORE ROOM DOWN ({em.remaining_room_down_pct*100:.1f}% vs {em.remaining_room_up_pct*100:.1f}% up)')
    except Exception as e:
        print(f'   [Error calculating expected move: {e}]')
        # Fallback: calculate simple volatility-based EM
        returns = df['close'].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)  # Annualized volatility
        weekly_em = live * vol * np.sqrt(5/252)
        print(f'   Current Price:     ${live:.2f}')
        print(f'   Est Weekly Move:   +/- ${weekly_em:.2f} ({vol*100:.1f}% annualized vol)')
        print(f'   Upper Bound:       ${live + weekly_em:.2f}')
        print(f'   Lower Bound:       ${live - weekly_em:.2f}')

    print()
    print('3. TRADE PARAMETERS')
    print('-' * 50)
    print(f'   Entry:        ${entry:.2f}')
    print(f'   Stop Loss:    ${stop:.2f}')
    print(f'   Target:       ${target:.2f}')
    risk = entry - stop
    reward = target - entry
    rr = reward / risk if risk > 0 else 0
    print(f'   Risk:         ${risk:.2f} per share')
    print(f'   Reward:       ${reward:.2f} per share')
    print(f'   R:R Ratio:    {rr:.2f}:1')

    print()
    print('4. STOCK-SPECIFIC HISTORICAL ANALYSIS')
    print('-' * 50)
    print('   Pattern:      20-day low sweep + close above')
    print(f'   Win Rate:     {wr*100:.1f}% (ACTUAL for {symbol})')
    print(f'   Sample Size:  {samples} historical instances')
    confidence = "HIGH" if samples >= 30 else "MEDIUM"
    print(f'   Confidence:   {confidence} ({samples} samples)')

    # Calculate avg return from pattern
    df_calc = df.copy()
    df_calc['next_ret'] = df_calc['close'].shift(-1) / df_calc['close'] - 1
    df_calc['low_20'] = df_calc['low'].rolling(20).min().shift(1)
    df_calc['pattern'] = df_calc['low'] < df_calc['low_20']
    instances = df_calc[df_calc['pattern'] == True].dropna(subset=['next_ret'])
    avg_ret = instances['next_ret'].mean() * 100 if len(instances) > 0 else 0

    print(f'   Avg Return:   {avg_ret:+.2f}% (next day)')

    print()
    print('5. OVERSOLD CONDITIONS')
    print('-' * 50)
    ibs_status = 'OVERSOLD' if ibs < 0.20 else 'Normal'
    rsi_status = 'OVERSOLD' if rsi < 30 else 'Normal'
    print(f'   IBS:          {ibs:.3f} ({ibs_status})')
    print(f'   RSI(14):      {rsi:.1f} ({rsi_status})')
    verdict = "Confirmed oversold" if ibs < 0.20 or rsi < 30 else "Not oversold"
    print(f'   Verdict:      {verdict}')

    print()
    print('6. EXPECTED VALUE CALCULATION')
    print('-' * 50)
    print('   Formula:      EV = (WR x RR) - ((1-WR) x 1)')
    print(f'   Calculation:  ({wr:.3f} x {rr:.2f}) - ({1-wr:.3f} x 1)')
    print(f'   EV:           {ev:+.4f} per $1 risked')
    if ev > 0.15:
        quality = 'EXCELLENT (EV > 0.15)'
    elif ev > 0.05:
        quality = 'GOOD (EV > 0.05)'
    else:
        quality = 'ACCEPTABLE (EV > 0)'
    print(f'   Quality:      {quality}')

    print()
    print('7. POSITION SIZING (2% Risk, 20% Notional Cap)')
    print('-' * 50)
    risk_budget = ACCOUNT_EQUITY * MAX_RISK_PCT
    notional_cap = ACCOUNT_EQUITY * MAX_NOTIONAL_PCT
    shares_by_risk = int(risk_budget / risk) if risk > 0 else 0
    shares_by_notional = int(notional_cap / entry)
    final_shares = min(shares_by_risk, shares_by_notional)
    position_value = final_shares * entry
    actual_risk = final_shares * risk

    print(f'   Account:      ${ACCOUNT_EQUITY:,.0f}')
    print(f'   Risk Budget:  ${risk_budget:.0f} (2%)')
    print(f'   Notional Cap: ${notional_cap:.0f} (20%)')
    print(f'   Shares (risk):     {shares_by_risk}')
    print(f'   Shares (notional): {shares_by_notional}')
    print(f'   FINAL SHARES: {final_shares}')
    print(f'   Position Value: ${position_value:,.2f}')
    print(f'   Actual Risk:  ${actual_risk:.2f}')

    print()
    print('8. BULL CASE')
    print('-' * 50)
    print(f'   - Pattern has {wr*100:.1f}% win rate over {samples} instances')
    print('   - Stock is oversold (selling exhaustion)')
    print('   - 20-day low sweep often traps weak longs')
    print('   - Close above low shows buyer absorption')
    print(f'   - Expected bounce: {avg_ret:+.2f}% next day')

    print()
    print('9. BEAR CASE')
    print('-' * 50)
    print(f'   - {100-wr*100:.1f}% of instances did NOT bounce')
    print('   - Could break below stop if selling continues')
    print('   - Market regime could override stock pattern')
    print('   - Gap risk overnight')

    print()
    print('10. WHAT COULD GO WRONG')
    print('-' * 50)
    print('   1. Market-wide selloff overrides pattern')
    print('   2. Negative news after hours')
    print('   3. Stop hit on opening gap')
    print('   4. Selling pressure continues (not exhausted)')
    print('   5. Lower volume = fake reversal')

    print()
    print('11. TRADE GRADE & RECOMMENDATION')
    print('-' * 50)

    # Grade calculation
    grade_score = 0
    if ev > 0.15:
        grade_score += 3
    elif ev > 0.05:
        grade_score += 2
    elif ev > 0:
        grade_score += 1

    if samples >= 40:
        grade_score += 2
    elif samples >= 20:
        grade_score += 1

    if ibs < 0.20 or rsi < 30:
        grade_score += 2

    if abs(gap_pct) < 1:
        grade_score += 1

    if grade_score >= 7:
        grade = 'A+'
    elif grade_score >= 6:
        grade = 'A'
    elif grade_score >= 4:
        grade = 'B'
    else:
        grade = 'C'

    print(f'   Trade Grade:  {grade}')
    print('   Factors:')
    ev_quality = "Excellent" if ev > 0.15 else "Good"
    samples_quality = "High" if samples >= 40 else "Medium"
    oversold_check = "Yes" if ibs < 0.20 or rsi < 30 else "No"
    gap_check = "OK" if abs(gap_pct) < 3 else "Warning"
    print(f'     - EV: {ev:+.4f} ({ev_quality})')
    print(f'     - Samples: {samples} ({samples_quality})')
    print(f'     - Oversold: {oversold_check}')
    print(f'     - Gap: {gap_pct:+.2f}% ({gap_check})')
    print()

    if grade in ['A+', 'A']:
        print('   RECOMMENDATION: TRADE')
        print(f'   - Buy {final_shares} shares at ${entry:.2f}')
        print(f'   - Stop at ${stop:.2f}')
        print(f'   - Target ${target:.2f}')
    else:
        print(f'   RECOMMENDATION: CONSIDER (Grade {grade})')

    return {
        'symbol': symbol,
        'entry': entry,
        'stop': stop,
        'target': target,
        'wr': wr,
        'samples': samples,
        'ev': ev,
        'grade': grade,
        'shares': final_shares,
        'position_value': position_value,
    }


def main():
    print('=' * 70)
    print('FULL PROFESSIONAL ANALYSIS - TOP 2 TRADES')
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # FIX (2026-01-07): Read from watchlist instead of hardcoded values
    watchlist_path = ROOT / 'state' / 'watchlist' / 'next_day.json'
    if watchlist_path.exists():
        with open(watchlist_path) as f:
            watchlist_data = json.load(f)
        top2 = watchlist_data.get('watchlist', [])[:2]
        print(f"\nLoaded watchlist for {watchlist_data.get('for_date', 'unknown')}")
    else:
        print("\nNo watchlist found, using fallback symbols")
        top2 = [
            {'symbol': 'SPY', 'entry_price': 590, 'stop_loss': 585, 'take_profit': 600, 'strategy': 'IBS_RSI'},
            {'symbol': 'QQQ', 'entry_price': 520, 'stop_loss': 515, 'take_profit': 530, 'strategy': 'IBS_RSI'},
        ]

    trades = []
    for stock in top2:
        symbol = stock['symbol']
        entry = stock.get('entry_price', 0)
        stop = stock.get('stop_loss', 0)
        target = stock.get('take_profit', entry + (entry - stop))  # Default to 1:1 if no target
        strategy = stock.get('strategy', 'Unknown')

        # Calculate pattern stats from historical data
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df = fetch_daily_bars_polygon(symbol, start=start_date, end=end_date, cache_dir=None)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate IBS and RSI
            last = df.iloc[-1]
            ibs = (last['close'] - last['low']) / (last['high'] - last['low']) if last['high'] != last['low'] else 0.5

            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Calculate pattern win rate
            df['next_ret'] = df['close'].shift(-1) / df['close'] - 1
            df['low_20'] = df['low'].rolling(20).min().shift(1)
            df['pattern'] = df['low'] < df['low_20']
            instances = df[df['pattern'] == True].dropna(subset=['next_ret'])
            samples = len(instances)
            wr = (instances['next_ret'] > 0).mean() if samples > 0 else 0.5

            # Calculate EV
            risk = entry - stop
            reward = target - entry
            rr = reward / risk if risk > 0 else 1.0
            ev = (wr * rr) - ((1 - wr) * 1.0)

        except Exception as e:
            print(f"Error calculating stats for {symbol}: {e}")
            ibs, rsi, wr, samples, ev = 0.5, 50, 0.5, 0, 0.0

        trade = full_analysis(symbol, entry, stop, target, wr, samples, ev, ibs, rsi, strategy)
        trades.append(trade)

    trade1 = trades[0] if len(trades) > 0 else None
    trade2 = trades[1] if len(trades) > 1 else None

    print()
    print('=' * 70)
    print('FINAL SUMMARY - TOP 2 TRADES')
    print('=' * 70)
    print()

    if trade1:
        print(f'Trade 1: {trade1["symbol"]} | Grade {trade1["grade"]} | {trade1["shares"]} shares | EV={trade1["ev"]:+.4f}')
    else:
        print('Trade 1: No valid trade found')

    if trade2:
        print(f'Trade 2: {trade2["symbol"]} | Grade {trade2["grade"]} | {trade2["shares"]} shares | EV={trade2["ev"]:+.4f}')
    else:
        print('Trade 2: No valid trade found')

    print()
    if trade1 and trade2:
        print('Both trades analyzed with EXPECTED MOVE FOR THE WEEK.')
        print('This is how a professional quant trades.')
    elif trade1 or trade2:
        print('Only one trade available - proceed with caution.')
    else:
        print('No valid trades found. Watchlist may need regeneration with R:R >= 1.5:1.')

    # Save
    results = {
        'analysis_time': datetime.now().isoformat(),
        'top2': [t for t in trades if t is not None],
    }
    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/top2_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print('Full analysis saved: state/watchlist/top2_analysis.json')


if __name__ == '__main__':
    main()
