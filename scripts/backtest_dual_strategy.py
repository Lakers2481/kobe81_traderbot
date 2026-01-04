#!/usr/bin/env python3
"""
Backtest Dual Strategy System - Quant Interview Ready

Combined: IBS+RSI (high frequency) + Turtle Soup (high conviction)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from data.providers.polygon_eod import fetch_daily_bars_polygon
from data.universe.loader import load_universe
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams


def simulate_trades(signals: pd.DataFrame, combined: pd.DataFrame, params: DualStrategyParams) -> List[Dict]:
    """Simulate trades with proper exits for each strategy."""
    trades = []

    for _, sig in signals.iterrows():
        sym = sig['symbol']
        strategy = sig['strategy']
        sym_data = combined[combined['symbol'] == sym].sort_values('timestamp')

        entry_ts = sig['timestamp']
        mask = sym_data['timestamp'] > entry_ts
        if not mask.any():
            continue

        entry_idx = mask.idxmax()
        entry_row = sym_data.loc[entry_idx]
        entry_price = float(entry_row['open'])

        # Use signal's time_stop_bars (strategy-specific)
        time_stop = int(sig.get('time_stop_bars', params.time_stop_bars))

        # For Turtle Soup, recalculate stop/TP based on actual entry price
        if strategy == 'TurtleSoup':
            atr_val = float(sig.get('atr', 1.0))
            signal_low = float(sig['entry_price']) - atr_val  # Approximate signal bar low
            stop_price = signal_low - params.ts_stop_buffer_mult * atr_val
            risk = entry_price - stop_price
            tp_price = entry_price + params.ts_r_multiple * risk if risk > 0 else None
        else:
            stop_price = sig['stop_loss']
            tp_price = sig['take_profit']

        exit_price = None
        exit_reason = None
        bars_held = 0

        future = sym_data.loc[entry_idx:].iloc[1:]
        for _, bar in future.iterrows():
            bars_held += 1
            close = float(bar['close'])
            low = float(bar['low'])
            high = float(bar['high'])
            bar_ibs = bar.get('ibs')
            bar_rsi = bar.get('rsi2')

            # Stop loss (both strategies)
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = 'stop_loss'
                break

            # Strategy-specific exits
            if strategy == 'IBS_RSI':
                # IBS exit
                if bar_ibs is not None and float(bar_ibs) > params.ibs_exit:
                    exit_price = close
                    exit_reason = 'ibs_exit'
                    break
                # RSI exit
                if bar_rsi is not None and float(bar_rsi) > params.rsi_exit:
                    exit_price = close
                    exit_reason = 'rsi_exit'
                    break
            else:  # TurtleSoup
                # Take profit
                if tp_price and high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break

            # Time stop (uses signal's strategy-specific value)
            if bars_held >= time_stop:
                exit_price = close
                exit_reason = 'time_stop'
                break

        if exit_price is None:
            if len(future) > 0:
                exit_price = float(future.iloc[-1]['close'])
                exit_reason = 'end_of_data'
            else:
                continue

        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100

        trades.append({
            'symbol': sym,
            'strategy': strategy,
            'entry_date': entry_ts,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'win': pnl > 0,
        })

    return trades


def run_backtest(symbols: List[str], start: str, end: str, params: DualStrategyParams, max_symbols: int) -> Dict:
    """Run combined strategy backtest."""
    scanner = DualStrategyScanner(params)
    cache_dir = Path("data/cache/polygon")

    print(f"Fetching data for {min(len(symbols), max_symbols)} symbols...")
    all_data = []
    for i, sym in enumerate(symbols[:max_symbols]):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(len(symbols), max_symbols)}...")
        try:
            df = fetch_daily_bars_polygon(sym, start, end, cache_dir=cache_dir)
            if df is not None and len(df) > 220:
                if 'symbol' not in df.columns:
                    df['symbol'] = sym
                all_data.append(df)
        except Exception:
            pass

    if not all_data:
        return {'error': 'No data'}

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Got {len(all_data)} symbols, {len(combined)} bars")

    # Compute indicators
    combined = scanner._compute_indicators(combined)

    # Generate signals
    print("Generating signals...")
    signals = scanner.scan_signals_over_time(combined.copy())
    print(f"Generated {len(signals)} total signals")

    if signals.empty:
        return {'signals': 0, 'trades': 0}

    # Count by strategy
    ibs_rsi_signals = len(signals[signals['strategy'] == 'IBS_RSI'])
    ts_signals = len(signals[signals['strategy'] == 'TurtleSoup'])
    print(f"  IBS+RSI: {ibs_rsi_signals}")
    print(f"  Turtle Soup: {ts_signals}")

    # Simulate trades
    print("Simulating trades...")
    trades = simulate_trades(signals, combined, params)

    if not trades:
        return {'signals': len(signals), 'trades': 0}

    trades_df = pd.DataFrame(trades)

    # Calculate metrics for each strategy
    results = {'symbols_tested': len(all_data)}

    for strat_name in ['IBS_RSI', 'TurtleSoup', 'Combined']:
        if strat_name == 'Combined':
            strat_trades = trades_df
            strat_signals = signals
        else:
            strat_trades = trades_df[trades_df['strategy'] == strat_name]
            strat_signals = signals[signals['strategy'] == strat_name]

        if strat_trades.empty:
            results[strat_name] = {'trades': 0, 'win_rate': 0, 'profit_factor': 0}
            continue

        wins = strat_trades['win'].sum()
        losses = len(strat_trades) - wins
        win_rate = wins / len(strat_trades) * 100

        gp = strat_trades[strat_trades['pnl'] > 0]['pnl'].sum()
        gl = abs(strat_trades[strat_trades['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else float('inf')

        avg_win = strat_trades[strat_trades['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = strat_trades[strat_trades['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0

        # Signals per day
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        days = (end_dt - start_dt).days * 252 / 365
        spd = len(strat_signals) / days if days > 0 else 0

        results[strat_name] = {
            'signals': len(strat_signals),
            'trades': len(strat_trades),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'avg_bars_held': round(strat_trades['bars_held'].mean(), 1),
            'signals_per_day': round(spd, 2),
            'exit_reasons': strat_trades['exit_reason'].value_counts().to_dict(),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', default='data/universe/optionable_liquid_900.csv')
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default='2025-12-26')
    parser.add_argument('--cap', type=int, default=200)
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    params = DualStrategyParams()

    print(f"\n{'='*70}")
    print("DUAL STRATEGY SYSTEM BACKTEST")
    print(f"{'='*70}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Universe: {len(symbols)} symbols (testing {args.cap})")
    print()
    print("Strategy 1: IBS+RSI Mean Reversion")
    print(f"  Entry: IBS<{params.ibs_entry}, RSI(2)<{params.rsi_entry}, >SMA200")
    print(f"  Exit: IBS>{params.ibs_exit}, RSI>{params.rsi_exit}, ATR*{params.ibs_rsi_stop_mult} stop")
    print()
    print("Strategy 2: Turtle Soup (Strong Sweep)")
    print(f"  Entry: Sweep>{params.ts_min_sweep_strength}ATR below {params.ts_lookback}-day low, revert")
    print(f"  Exit: {params.ts_r_multiple}R profit, ATR*{params.ts_stop_buffer_mult} stop")
    print(f"{'='*70}\n")

    results = run_backtest(symbols, args.start, args.end, params, args.cap)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return 1

    # Print results for each strategy
    for strat_name in ['IBS_RSI', 'TurtleSoup', 'Combined']:
        strat = results.get(strat_name, {})
        if not strat or strat.get('trades', 0) == 0:
            continue

        print(f"\n{'='*70}")
        print(f"{strat_name.upper()} RESULTS")
        print(f"{'='*70}")
        print(f"Signals: {strat.get('signals', 0)}")
        print(f"Trades:  {strat.get('trades', 0)}")
        print(f"Wins/Losses: {strat.get('wins', 0)}/{strat.get('losses', 0)}")
        print(f"Win Rate: {strat.get('win_rate', 0)}%")
        print(f"Profit Factor: {strat.get('profit_factor', 0)}")
        print(f"Avg Win: +{strat.get('avg_win_pct', 0)}%")
        print(f"Avg Loss: {strat.get('avg_loss_pct', 0)}%")
        print(f"Avg Bars: {strat.get('avg_bars_held', 0)}")
        print(f"Signals/Day: {strat.get('signals_per_day', 0)}")

    # Scale projections for 900 stocks
    scale = 900 / results['symbols_tested'] if results['symbols_tested'] > 0 else 1

    print(f"\n{'='*70}")
    print("QUANT INTERVIEW CRITERIA (900-stock projection)")
    print(f"{'='*70}")

    for strat_name in ['IBS_RSI', 'TurtleSoup', 'Combined']:
        strat = results.get(strat_name, {})
        if not strat or strat.get('trades', 0) == 0:
            continue

        wr = strat.get('win_rate', 0)
        pf = strat.get('profit_factor', 0)
        spd = strat.get('signals_per_day', 0) * scale

        wr_pass = wr >= 60
        pf_pass = pf >= 1.3
        spd_pass = spd >= 1.0

        print(f"\n{strat_name}:")
        print(f"  Win Rate >= 60%:    {'PASS' if wr_pass else 'FAIL'} ({wr}%)")
        print(f"  Profit Factor >= 1.3: {'PASS' if pf_pass else 'FAIL'} ({pf})")
        print(f"  Signals/Day >= 1:   {'PASS' if spd_pass else 'FAIL'} ({spd:.1f})")

        if wr_pass and pf_pass and spd_pass:
            print("  >>> ALL CRITERIA PASSED <<<")

    combined = results.get('Combined', {})
    if combined.get('trades', 0) > 0:
        wr = combined.get('win_rate', 0)
        pf = combined.get('profit_factor', 0)
        spd = combined.get('signals_per_day', 0) * scale

        if wr >= 60 and pf >= 1.3 and spd >= 1.0:
            print(f"\n{'*'*70}")
            print("*** DUAL STRATEGY SYSTEM - QUANT INTERVIEW READY ***")
            print(f"{'*'*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
