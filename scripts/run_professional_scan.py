#!/usr/bin/env python3
"""
Run Professional Quant Scan - The CORRECT Way

This script implements institutional-grade scanning:
1. Scan 800 stocks with stock-specific historical analysis
2. Scan crypto with same methodology
3. Generate options from validated signals
4. Rank ALL by Expected Value
5. Output Top 5 watchlist and Top 2 trades

A professional quant aims for:
- Positive Expected Value (EV > 0)
- Statistical significance (20+ samples)
- Verified oversold conditions
- Capital preservation first
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

# Data providers
from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.providers.alpaca_intraday import fetch_intraday_bars
from data.universe.loader import load_universe

# Strategy scanner
from strategies.dual_strategy.combined import DualStrategyScanner, DualStrategyParams

# Our professional scanner
from scanner.professional_quant_scanner import (
    ProfessionalQuantScanner,
    QuantSignal,
    run_professional_scan,
)


def scan_equities(universe_file: str, cap: int, end_date: str, verbose: bool = True):
    """Scan equity universe with professional methodology."""
    if verbose:
        print("=" * 80)
        print("EQUITY SCAN (Professional Quant Methodology)")
        print("=" * 80)
        print()

    # Load universe
    symbols = load_universe(universe_file, cap=cap)
    if verbose:
        print(f"Universe: {len(symbols)} stocks")
        print(f"End date: {end_date}")
        print()

    # Initialize scanner
    params = DualStrategyParams()
    scanner = DualStrategyScanner(params)

    # Run professional scan
    ranked, all_signals = run_professional_scan(
        symbols=symbols,
        fetch_eod_func=fetch_daily_bars_polygon,
        fetch_intraday_func=fetch_intraday_bars,
        scanner_func=scanner,
        end_date=end_date,
        verbose=verbose,
    )

    return ranked, all_signals


def scan_crypto(end_date: str, verbose: bool = True):
    """Scan crypto with same professional methodology."""
    if verbose:
        print()
        print("=" * 80)
        print("CRYPTO SCAN (Professional Quant Methodology)")
        print("=" * 80)
        print()

    try:
        from data.providers.polygon_crypto import fetch_crypto_bars
    except ImportError:
        if verbose:
            print("Crypto provider not available")
        return [], []

    crypto_pairs = ['X:BTCUSD', 'X:ETHUSD', 'X:SOLUSD', 'X:AVAXUSD', 'X:LINKUSD',
                    'X:DOGEUSD', 'X:MATICUSD', 'X:ADAUSD']

    params = DualStrategyParams(min_price=0.0)
    scanner = DualStrategyScanner(params)
    quant = ProfessionalQuantScanner()

    all_signals = []

    for pair in crypto_pairs:
        try:
            df = fetch_crypto_bars(pair, start='2024-06-01', end=end_date, timeframe='1d')
            if df is None or len(df) < 50:
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            sig = scanner.generate_signals(df)
            if sig.empty:
                clean_sym = pair.replace('X:', '')
                if verbose:
                    print(f"  {clean_sym}: No signal (conditions not met)")
                continue

            r = sig.iloc[0]
            entry = float(r.get('entry_price', 0))
            stop = float(r.get('stop_loss', 0))
            target = float(r.get('take_profit', 0)) if pd.notna(r.get('take_profit')) else entry + (entry - stop)
            strategy = r.get('strategy', 'Unknown') + '_CRYPTO'

            if entry <= 0 or stop <= 0:
                continue

            live_price = df.iloc[-1]['close']

            quant_signal = quant.analyze_signal(
                symbol=pair.replace('X:', ''),
                df=df,
                entry=entry,
                stop=stop,
                target=target,
                strategy=strategy,
                asset_class='CRYPTO',
                live_price=live_price,
            )

            if quant_signal:
                all_signals.append(quant_signal)
                status = "PASS" if quant_signal.passes_quant_gate else f"FAIL: {quant_signal.rejection_reason}"
                if verbose:
                    print(f"  {quant_signal.symbol}: EV={quant_signal.expected_value:+.4f} | {status}")

        except Exception as e:
            clean_sym = pair.replace('X:', '')
            if verbose:
                print(f"  {clean_sym}: Error - {str(e)[:40]}")

    ranked = quant.rank_signals(all_signals)
    return ranked, all_signals


def generate_options(equity_signals: list, verbose: bool = True):
    """Generate options from validated equity signals."""
    if verbose:
        print()
        print("=" * 80)
        print("OPTIONS GENERATION (From Validated Equities)")
        print("=" * 80)
        print()

    try:
        from scanner.options_signals import OptionsSignalGenerator
    except ImportError:
        if verbose:
            print("Options module not available")
        return []

    # Only use top equity signals that pass quant gate
    valid_equities = [s for s in equity_signals if s.passes_quant_gate][:5]

    if not valid_equities:
        if verbose:
            print("No validated equity signals for options generation")
        return []

    options_signals = []

    for eq in valid_equities:
        try:
            # Fetch price data for volatility
            df = fetch_daily_bars_polygon(eq.symbol, start='2025-06-01', end='2026-01-05', cache_dir=None)
            if df is None or len(df) < 30:
                continue

            # Create equity DataFrame for options generator
            eq_df = pd.DataFrame([{
                'symbol': eq.symbol,
                'entry_price': eq.entry_price,
                'conf_score': eq.expected_value,  # Use EV as confidence
                'strategy': eq.strategy,
            }])

            df['symbol'] = eq.symbol

            gen = OptionsSignalGenerator(target_delta=0.30, target_dte=21)
            opts_df = gen.generate_from_equity_signals(eq_df, df, max_signals=2)

            if not opts_df.empty:
                for _, row in opts_df.iterrows():
                    opt_symbol = f"{row['symbol']}_{row['strike']:.0f}{row['option_type'][0]}"
                    opt_premium = row['option_price']
                    opt_delta = row['delta']

                    # Create QuantSignal for options
                    opt_signal = QuantSignal(
                        symbol=opt_symbol,
                        asset_class='OPTIONS',
                        strategy=f"{eq.strategy}_{row['option_type']}",
                        entry_price=opt_premium,
                        stop_loss=0,  # Max loss is premium
                        take_profit=opt_premium * 2,  # 100% target
                        risk=opt_premium,
                        reward=opt_premium,
                        rr_ratio=1.0,
                        pattern_win_rate=eq.pattern_win_rate * 0.85,  # Discount for options
                        pattern_sample_size=eq.pattern_sample_size,
                        pattern_avg_return=eq.pattern_avg_return,
                        ibs=eq.ibs,
                        rsi=eq.rsi,
                        is_oversold=eq.is_oversold,
                        expected_value=eq.expected_value * 0.80,  # Options have more risk
                        passes_quant_gate=eq.passes_quant_gate and eq.expected_value > 0.02,
                        rejection_reason="" if eq.passes_quant_gate else "Parent equity failed",
                        live_price=opt_premium,
                        gap_pct=0,
                    )
                    options_signals.append(opt_signal)

                    if verbose:
                        print(f"  {opt_symbol}: Premium=${opt_premium:.2f} | Delta={opt_delta:.2f} | EV={opt_signal.expected_value:+.4f}")

        except Exception as e:
            if verbose:
                print(f"  {eq.symbol} options error: {str(e)[:40]}")

    return options_signals


def print_results(all_signals: list, verbose: bool = True):
    """Print comprehensive results."""
    print()
    print("=" * 80)
    print("UNIFIED RESULTS (All Asset Classes)")
    print("=" * 80)

    # Separate by pass/fail
    passing = [s for s in all_signals if s.passes_quant_gate]
    failing = [s for s in all_signals if not s.passes_quant_gate]

    # Sort passing by EV
    passing.sort(key=lambda x: x.expected_value, reverse=True)

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
            oversold = "YES" if s.is_oversold else "NO"
            print(f"{i+1:<5} {s.symbol:<12} {s.asset_class:<8} ${s.entry_price:>9.2f} {s.pattern_win_rate*100:>7.1f}% {s.expected_value:>+9.4f} {oversold:<10}")

        # Top 5 watchlist
        top5 = passing[:5]
        print()
        print("=" * 80)
        print("TOP 5 WATCHLIST")
        print("=" * 80)

        for i, s in enumerate(top5):
            print(f"\n{i+1}. {s.symbol} ({s.asset_class})")
            print(f"   Strategy:    {s.strategy}")
            print(f"   Entry:       ${s.entry_price:.2f}")
            print(f"   Stop:        ${s.stop_loss:.2f}")
            print(f"   Target:      ${s.take_profit:.2f}")
            print(f"   R:R:         {s.rr_ratio:.2f}:1")
            print(f"   Win Rate:    {s.pattern_win_rate*100:.1f}% ({s.pattern_sample_size} samples)")
            print(f"   EV:          {s.expected_value:+.4f} per $1 risked")
            print(f"   IBS/RSI:     {s.ibs:.2f} / {s.rsi:.1f}")
            print(f"   Oversold:    {'YES' if s.is_oversold else 'NO'}")

        # Top 2 to trade
        top2 = passing[:2]
        print()
        print("=" * 80)
        print("TOP 2 TRADES")
        print("=" * 80)

        if top2:
            for i, s in enumerate(top2):
                print(f"\n{'='*40}")
                print(f"TRADE {i+1}: {s.symbol}")
                print(f"{'='*40}")
                print(f"  Asset Class:  {s.asset_class}")
                print(f"  Strategy:     {s.strategy}")
                print(f"  Entry:        ${s.entry_price:.2f}")
                print(f"  Stop:         ${s.stop_loss:.2f}")
                print(f"  Target:       ${s.take_profit:.2f}")
                print(f"  R:R Ratio:    {s.rr_ratio:.2f}:1")
                print()
                print(f"  Win Rate:     {s.pattern_win_rate*100:.1f}%")
                print(f"  Sample Size:  {s.pattern_sample_size}")
                print(f"  Avg Return:   {s.pattern_avg_return:+.2f}%")
                print(f"  Expected Value: {s.expected_value:+.4f}")
                print()
                print(f"  IBS:          {s.ibs:.2f} ({'OVERSOLD' if s.ibs < 0.2 else 'normal'})")
                print(f"  RSI(14):      {s.rsi:.1f} ({'OVERSOLD' if s.rsi < 30 else 'normal'})")
                print(f"  Live Price:   ${s.live_price:.2f}")
                print(f"  Gap:          {s.gap_pct*100:+.1f}%")
        else:
            print("\nNO TRADES PASS QUANT GATE")

    else:
        print()
        print("=" * 80)
        print("NO SIGNALS PASS QUANT GATE")
        print("=" * 80)
        print()
        print("This is CORRECT behavior for a professional quant:")
        print("  - Capital preservation > forced trades")
        print("  - Negative EV setups are rejected")
        print("  - Wait for better opportunities")
        print()
        print("RECOMMENDATION: Wait for EOD scan at 3:30 PM ET")

    # Show why signals failed (top 5 failures)
    if failing and verbose:
        print()
        print("-" * 80)
        print("TOP FAILURES (Why Signals Were Rejected)")
        print("-" * 80)

        # Group by rejection reason
        reasons = {}
        for s in failing:
            reason = s.rejection_reason or "Unknown"
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(s)

        for reason, signals in sorted(reasons.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"\n{reason}: {len(signals)} signals")
            for s in signals[:3]:
                print(f"  - {s.symbol}: WR={s.pattern_win_rate*100:.1f}%, EV={s.expected_value:+.4f}")

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

    end_date = '2026-01-05'

    # 1. Scan equities
    equity_ranked, equity_all = scan_equities(
        universe_file='data/universe/optionable_liquid_800.csv',
        cap=900,
        end_date=end_date,
        verbose=True,
    )

    # 2. Scan crypto
    crypto_ranked, crypto_all = scan_crypto(end_date=end_date, verbose=True)

    # 3. Generate options from validated equities
    options_signals = generate_options(equity_ranked, verbose=True)

    # 4. Combine all signals
    all_signals = equity_all + crypto_all + options_signals

    # 5. Print results
    passing = print_results(all_signals, verbose=True)

    # 6. Save results
    results = {
        'scan_time': datetime.now().isoformat(),
        'methodology': 'professional_quant',
        'end_date': end_date,
        'total_signals': len(all_signals),
        'passing_signals': len(passing),
        'by_class': {
            'equity': len([s for s in all_signals if s.asset_class == 'EQUITY']),
            'crypto': len([s for s in all_signals if s.asset_class == 'CRYPTO']),
            'options': len([s for s in all_signals if s.asset_class == 'OPTIONS']),
        },
        'top5': [s.to_dict() for s in passing[:5]],
        'top2': [s.to_dict() for s in passing[:2]],
        'all_passing': [s.to_dict() for s in passing],
    }

    Path('state/watchlist').mkdir(parents=True, exist_ok=True)
    with open('state/watchlist/professional_scan.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("Results saved to: state/watchlist/professional_scan.json")


if __name__ == '__main__':
    main()
