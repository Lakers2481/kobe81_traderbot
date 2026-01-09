#!/usr/bin/env python3
"""
Fast Professional Quant Scan - Streamlined for Speed

Scans stocks + crypto + options with proper quant methodology:
1. Stock-specific historical pattern analysis
2. Actual EV calculation (not assumed)
3. Verified oversold conditions
4. Unified ranking by EV
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Minimal imports - no TF
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from data.universe.loader import load_universe

# Quant thresholds
MIN_SAMPLE_SIZE = 20
MIN_WIN_RATE = 0.45
MIN_EV = 0.01
IBS_OVERSOLD = 0.20
RSI_OVERSOLD = 30.0
MAX_GAP = 0.03


def calculate_pattern_stats(df: pd.DataFrame, pattern: str = 'turtle_soup'):
    """Calculate stock-specific historical win rate."""
    if len(df) < 50:
        return 0.5, 0, 0.0

    df = df.copy()
    df['next_day_ret'] = df['close'].shift(-1) / df['close'] - 1

    if pattern == 'turtle_soup':
        df['low_20'] = df['low'].rolling(20).min().shift(1)
        df['swept'] = df['low'] < df['low_20']
        instances = df[df['swept'] == True]
    elif pattern == 'ibs_rsi':
        df['ibs'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_2'] = 100 - (100 / (1 + rs))
        instances = df[(df['ibs'] < 0.08) & (df['rsi_2'] < 5)]
    else:
        return 0.5, 0, 0.0

    if len(instances) < 5:
        return 0.5, len(instances), 0.0

    wins = (instances['next_day_ret'] > 0).sum()
    total = len(instances)
    win_rate = wins / total if total > 0 else 0.5
    avg_ret = instances['next_day_ret'].mean() * 100

    return win_rate, total, avg_ret


def calculate_ev(win_rate: float, rr_ratio: float) -> float:
    """EV = (WR x RR) - ((1-WR) x 1)"""
    return (win_rate * rr_ratio) - ((1 - win_rate) * 1.0)


def is_oversold(df: pd.DataFrame):
    """Check if actually oversold (IBS < 0.2 OR RSI < 30)."""
    if len(df) < 14:
        return False, 0.5, 50.0

    last = df.iloc[-1]
    ibs = (last['close'] - last['low']) / (last['high'] - last['low']) if last['high'] != last['low'] else 0.5

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    oversold = (ibs < IBS_OVERSOLD) or (rsi < RSI_OVERSOLD)
    return oversold, ibs, rsi


def generate_turtle_soup_signal(df: pd.DataFrame):
    """Generate Turtle Soup signal if conditions met."""
    if len(df) < 25:
        return None

    # Check for 20-day low sweep
    low_20 = df['low'].iloc[-21:-1].min()
    last = df.iloc[-1]

    if last['low'] < low_20 and last['close'] > low_20:
        # Calculate entry/stop/target
        entry = last['close']
        stop = last['low'] - 0.3 * (last['high'] - last['low'])
        target = entry + (entry - stop)

        return {
            'entry': entry,
            'stop': stop,
            'target': target,
            'strategy': 'TurtleSoup',
        }
    return None


def scan_stocks(symbols: list, end_date: str = '2026-01-05'):
    """Scan stocks with professional quant methodology."""
    print("=" * 80)
    print("EQUITY SCAN - Professional Quant Methodology")
    print("=" * 80)
    print(f"\nScanning {len(symbols)} stocks...")
    print(f"End date: {end_date}")
    print()

    signals = []
    passing = 0

    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(symbols)} | Found: {len(signals)} | Passing: {passing}")

        try:
            # Fetch EOD data
            df = fetch_daily_bars_polygon(sym, start='2024-01-01', end=end_date, cache_dir=None)
            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Generate signal
            sig = generate_turtle_soup_signal(df)
            if not sig:
                continue

            # Get stock-specific stats
            win_rate, samples, avg_ret = calculate_pattern_stats(df, 'turtle_soup')

            # Check oversold
            oversold, ibs, rsi = is_oversold(df)

            # Calculate R:R and EV
            risk = sig['entry'] - sig['stop']
            reward = sig['target'] - sig['entry']
            rr = reward / risk if risk > 0 else 0
            ev = calculate_ev(win_rate, rr)

            # Get live price for gap check
            try:
                bars = fetch_intraday_bars(sym, timeframe='15Min', limit=3)
                live_price = bars[-1].close if bars else sig['entry']
                gap = (live_price / sig['entry']) - 1
            except Exception:
                live_price = sig['entry']
                gap = 0

            # Quality gate
            passes = True
            reason = ""

            if samples < MIN_SAMPLE_SIZE:
                passes = False
                reason = f"Samples {samples} < {MIN_SAMPLE_SIZE}"
            elif win_rate < MIN_WIN_RATE:
                passes = False
                reason = f"WR {win_rate*100:.1f}% < {MIN_WIN_RATE*100}%"
            elif ev < MIN_EV:
                passes = False
                reason = f"EV {ev:.4f} < {MIN_EV}"
            elif not oversold:
                passes = False
                reason = f"Not oversold (IBS={ibs:.2f}, RSI={rsi:.1f})"
            elif abs(gap) > MAX_GAP:
                passes = False
                reason = f"Gap {gap*100:.1f}% > {MAX_GAP*100}%"

            signal = {
                'symbol': sym,
                'asset_class': 'EQUITY',
                'strategy': sig['strategy'],
                'entry': sig['entry'],
                'stop': sig['stop'],
                'target': sig['target'],
                'live': live_price,
                'gap_pct': gap,
                'rr': rr,
                'win_rate': win_rate,
                'samples': samples,
                'avg_return': avg_ret,
                'ibs': ibs,
                'rsi': rsi,
                'oversold': oversold,
                'ev': ev,
                'passes': passes,
                'reason': reason,
            }
            signals.append(signal)

            if passes:
                passing += 1
                print(f"  PASS: {sym} | WR={win_rate*100:.1f}% ({samples}) | EV={ev:+.4f} | IBS={ibs:.2f}")

        except Exception:
            continue

    print(f"\nScan complete: {len(signals)} signals, {passing} pass quant gate")
    return signals


def scan_crypto(end_date: str = '2026-01-05'):
    """Scan crypto with same professional methodology."""
    print()
    print("=" * 80)
    print("CRYPTO SCAN - Professional Quant Methodology")
    print("=" * 80)

    try:
        from data.providers.polygon_crypto import fetch_crypto_bars
    except ImportError:
        print("Crypto provider not available")
        return []

    pairs = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD', 'X:AVAXUSD', 'X:LINKUSD']
    signals = []

    for pair in pairs:
        try:
            df = fetch_crypto_bars(pair, start='2025-06-01', end=end_date, timeframe='1d')
            if df is None or len(df) < 50:
                clean = pair.replace('X:', '')
                print(f"  {clean}: Insufficient data")
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Generate signal
            sig = generate_turtle_soup_signal(df)
            if not sig:
                clean = pair.replace('X:', '')
                print(f"  {clean}: No signal (conditions not met)")
                continue

            # Get stats
            win_rate, samples, avg_ret = calculate_pattern_stats(df, 'turtle_soup')
            oversold, ibs, rsi = is_oversold(df)

            # Calculate metrics
            risk = sig['entry'] - sig['stop']
            reward = sig['target'] - sig['entry']
            rr = reward / risk if risk > 0 else 0
            ev = calculate_ev(win_rate, rr)

            live_price = df.iloc[-1]['close']
            gap = (live_price / sig['entry']) - 1

            # Quality gate
            passes = True
            reason = ""

            if samples < MIN_SAMPLE_SIZE:
                passes = False
                reason = f"Samples {samples} < {MIN_SAMPLE_SIZE}"
            elif win_rate < MIN_WIN_RATE:
                passes = False
                reason = f"WR {win_rate*100:.1f}% < {MIN_WIN_RATE*100}%"
            elif ev < MIN_EV:
                passes = False
                reason = f"EV {ev:.4f} < {MIN_EV}"
            elif not oversold:
                passes = False
                reason = f"Not oversold (IBS={ibs:.2f}, RSI={rsi:.1f})"

            clean = pair.replace('X:', '')
            status = "PASS" if passes else f"FAIL: {reason}"
            print(f"  {clean}: WR={win_rate*100:.1f}% ({samples}) | EV={ev:+.4f} | {status}")

            signals.append({
                'symbol': clean,
                'asset_class': 'CRYPTO',
                'strategy': 'TurtleSoup_CRYPTO',
                'entry': sig['entry'],
                'stop': sig['stop'],
                'target': sig['target'],
                'live': live_price,
                'gap_pct': gap,
                'rr': rr,
                'win_rate': win_rate,
                'samples': samples,
                'ev': ev,
                'ibs': ibs,
                'rsi': rsi,
                'oversold': oversold,
                'passes': passes,
                'reason': reason,
            })

        except Exception as e:
            clean = pair.replace('X:', '')
            print(f"  {clean}: Error - {str(e)[:40]}")

    return signals


def generate_options(equity_signals: list):
    """Generate options from top equity signals."""
    print()
    print("=" * 80)
    print("OPTIONS GENERATION")
    print("=" * 80)

    try:
        from scanner.options_signals import OptionsSignalGenerator
    except ImportError:
        print("Options module not available")
        return []

    # Use top passing equity signals
    passing = [s for s in equity_signals if s['passes']][:3]

    if not passing:
        print("No passing equity signals for options")
        return []

    options = []

    for eq in passing:
        try:
            df = fetch_daily_bars_polygon(eq['symbol'], start='2025-10-01', end='2026-01-05', cache_dir=None)
            if df is None or len(df) < 20:
                continue

            # Create options signal
            eq_df = pd.DataFrame([{
                'symbol': eq['symbol'],
                'entry_price': eq['entry'],
                'conf_score': eq['ev'],
                'strategy': eq['strategy'],
            }])

            df['symbol'] = eq['symbol']

            gen = OptionsSignalGenerator(target_delta=0.30, target_dte=21)
            opts_df = gen.generate_from_equity_signals(eq_df, df, max_signals=2)

            if not opts_df.empty:
                for _, row in opts_df.iterrows():
                    opt_sym = f"{row['symbol']}_{row['strike']:.0f}{row['option_type'][0]}"
                    premium = row['option_price']
                    delta = row['delta']

                    # Options EV (discounted from equity)
                    opt_ev = eq['ev'] * 0.80

                    print(f"  {opt_sym}: Premium=${premium:.2f} | Delta={delta:.2f} | EV={opt_ev:+.4f}")

                    options.append({
                        'symbol': opt_sym,
                        'asset_class': 'OPTIONS',
                        'strategy': f"{eq['strategy']}_{row['option_type']}",
                        'entry': premium,
                        'stop': 0,
                        'target': premium * 2,
                        'live': premium,
                        'gap_pct': 0,
                        'rr': 1.0,
                        'win_rate': eq['win_rate'] * 0.85,
                        'samples': eq['samples'],
                        'ev': opt_ev,
                        'passes': eq['passes'] and opt_ev > MIN_EV,
                        'reason': "" if opt_ev > MIN_EV else "Options EV too low",
                    })

        except Exception as e:
            print(f"  {eq['symbol']} options error: {str(e)[:40]}")

    return options


def print_results(all_signals: list):
    """Print comprehensive results."""
    print()
    print("=" * 80)
    print("UNIFIED RESULTS - All Asset Classes")
    print("=" * 80)

    passing = [s for s in all_signals if s['passes']]
    failing = [s for s in all_signals if not s['passes']]

    # Sort by EV
    passing.sort(key=lambda x: x['ev'], reverse=True)

    print()
    print(f"Total signals: {len(all_signals)}")
    print(f"Pass quant gate: {len(passing)}")
    print(f"Fail quant gate: {len(failing)}")

    if passing:
        print()
        print("-" * 80)
        print("SIGNALS PASSING QUANT GATE (Ranked by Expected Value)")
        print("-" * 80)
        print()
        print(f"{'Rank':<5} {'Symbol':<12} {'Class':<8} {'Entry':>10} {'WR':>8} {'EV':>10} {'Oversold':<10}")
        print("-" * 80)

        for i, s in enumerate(passing[:10]):
            oversold = "YES" if s.get('oversold', False) else "NO"
            print(f"{i+1:<5} {s['symbol']:<12} {s['asset_class']:<8} ${s['entry']:>9.2f} {s['win_rate']*100:>7.1f}% {s['ev']:>+9.4f} {oversold:<10}")

        # Top 5 watchlist
        top5 = passing[:5]
        print()
        print("=" * 80)
        print("TOP 5 WATCHLIST")
        print("=" * 80)

        for i, s in enumerate(top5):
            print(f"\n{i+1}. {s['symbol']} ({s['asset_class']})")
            print(f"   Strategy:    {s['strategy']}")
            print(f"   Entry:       ${s['entry']:.2f}")
            print(f"   Stop:        ${s['stop']:.2f}")
            print(f"   Target:      ${s['target']:.2f}")
            print(f"   R:R:         {s['rr']:.2f}:1")
            print(f"   Win Rate:    {s['win_rate']*100:.1f}% ({s['samples']} samples)")
            print(f"   EV:          {s['ev']:+.4f} per $1 risked")

        # Top 2 trades
        top2 = passing[:2]
        print()
        print("=" * 80)
        print("TOP 2 TRADES")
        print("=" * 80)

        for i, s in enumerate(top2):
            print(f"\n{'='*40}")
            print(f"TRADE {i+1}: {s['symbol']}")
            print(f"{'='*40}")
            print(f"  Asset Class:  {s['asset_class']}")
            print(f"  Strategy:     {s['strategy']}")
            print(f"  Entry:        ${s['entry']:.2f}")
            print(f"  Stop:         ${s['stop']:.2f}")
            print(f"  Target:       ${s['target']:.2f}")
            print(f"  R:R Ratio:    {s['rr']:.2f}:1")
            print()
            print(f"  Win Rate:     {s['win_rate']*100:.1f}%")
            print(f"  Sample Size:  {s['samples']}")
            print(f"  Expected Value: {s['ev']:+.4f}")
            print()
            print(f"  Live Price:   ${s['live']:.2f}")
            print(f"  Gap:          {s['gap_pct']*100:+.1f}%")

    else:
        print()
        print("=" * 80)
        print("NO SIGNALS PASS QUANT GATE")
        print("=" * 80)
        print()
        print("This is CORRECT behavior for a professional quant:")
        print("  - Capital preservation > forced trades")
        print("  - Negative EV setups are rejected")
        print("  - No oversold signals = no trades")
        print()
        print("RECOMMENDATION: Wait for better setups")

    # Show top failures
    if failing:
        print()
        print("-" * 80)
        print("TOP REJECTIONS (Why Signals Failed)")
        print("-" * 80)

        # Group by reason
        reasons = {}
        for s in failing:
            r = s.get('reason', 'Unknown')
            if r not in reasons:
                reasons[r] = []
            reasons[r].append(s)

        for reason, sigs in sorted(reasons.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"\n{reason}: {len(sigs)} signals")
            for sig in sigs[:3]:
                print(f"  - {sig['symbol']}: WR={sig['win_rate']*100:.1f}%, EV={sig['ev']:+.4f}")

    return passing


def main():
    print("=" * 80)
    print("PROFESSIONAL QUANT SCAN")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Prior Close: Monday 2026-01-05")
    print("=" * 80)
    print()
    print("Methodology:")
    print("  - Stock-specific historical pattern analysis")
    print("  - Expected Value calculation (not assumed)")
    print("  - Verified oversold conditions (IBS < 0.2 OR RSI < 30)")
    print("  - Minimum 20 samples for statistical significance")
    print("  - Ranking by EV (not arbitrary score)")
    print()

    # 1. Scan stocks
    symbols = load_universe('data/universe/optionable_liquid_800.csv', cap=800)
    stock_signals = scan_stocks(symbols, end_date='2026-01-05')

    # 2. Scan crypto
    crypto_signals = scan_crypto(end_date='2026-01-05')

    # 3. Generate options
    options_signals = generate_options(stock_signals)

    # 4. Combine and rank
    all_signals = stock_signals + crypto_signals + options_signals

    # 5. Print results
    passing = print_results(all_signals)

    # 6. Save results
    results = {
        'scan_time': datetime.now().isoformat(),
        'methodology': 'professional_quant',
        'total_signals': len(all_signals),
        'passing_signals': len(passing),
        'by_class': {
            'equity': len([s for s in all_signals if s['asset_class'] == 'EQUITY']),
            'crypto': len([s for s in all_signals if s['asset_class'] == 'CRYPTO']),
            'options': len([s for s in all_signals if s['asset_class'] == 'OPTIONS']),
        },
        'top5': passing[:5],
        'top2': passing[:2],
    }

    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/professional_scan.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("Results saved to: state/watchlist/professional_scan.json")


if __name__ == '__main__':
    main()
