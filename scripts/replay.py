#!/usr/bin/env python3
"""
Historical Signal Replay

- Replay signals from a specific date range
- Show what trades would have been taken
- Compare with actual trades if available

Usage:
    python replay.py --start 2023-01-01 --end 2023-12-31 --strategy ibs_rsi
    python replay.py --start 2023-06-01 --end 2023-06-30 --strategy turtle_soup --actual-trades outputs/trade_list.csv
    python replay.py --start 2023-01-01 --end 2023-03-31 --dotenv ./.env
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.env_loader import load_env
from strategies.registry import get_production_scanner


def create_strategy(strategy_name: str = None):
    """Create strategy instance. Always returns DualStrategyScanner."""
    # Deprecated: strategy_name is ignored. Always use DualStrategyScanner.
    return get_production_scanner()


def load_actual_trades(path: Path) -> pd.DataFrame:
    """Load actual trades from CSV for comparison."""
    if not path.exists():
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])

    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Failed to load actual trades: {e}")
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])


def generate_signals_for_period(
    strategy,
    symbols: List[str],
    start: str,
    end: str,
    fetch_bars,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Generate all signals for given symbols and date range.

    Returns DataFrame with columns:
    - timestamp, symbol, side, entry_price, stop_loss, reason, etc.
    """
    all_signals = []

    progress_step = max(1, len(symbols) // 20)

    for i, sym in enumerate(symbols):
        if show_progress and i % progress_step == 0:
            pct = int(100 * i / len(symbols))
            print(f"\r[PROGRESS] Generating signals: {pct}%", end='', flush=True)

        try:
            df = fetch_bars(sym)
            if df is None or df.empty:
                continue

            # Filter to date range
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

            if len(df) < 210:  # Need enough data for indicators
                continue

            # Add symbol column if missing
            if 'symbol' not in df.columns:
                df['symbol'] = sym

            # Generate signals using scan method
            signals = strategy.scan_signals_over_time(df)

            if not signals.empty:
                # Filter to date range again (signals might be from lookback period)
                signals['timestamp'] = pd.to_datetime(signals['timestamp'])
                signals = signals[(signals['timestamp'] >= start) & (signals['timestamp'] <= end)]
                all_signals.append(signals)

        except Exception:
            if show_progress:
                pass  # Silently skip errors during progress
            continue

    if show_progress:
        print("\r[PROGRESS] Generating signals: 100%")

    if all_signals:
        result = pd.concat(all_signals, ignore_index=True)
        return result.sort_values('timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'side', 'entry_price', 'stop_loss',
            'take_profit', 'reason'
        ])


def simulate_trades(
    signals: pd.DataFrame,
    fetch_bars,
    initial_cash: float = 100000.0,
    position_size_pct: float = 0.007,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Simulate trades from signals with next-bar entry and exit logic.

    Returns DataFrame with simulated trades including entry/exit details.
    """
    if signals.empty:
        return pd.DataFrame(columns=[
            'signal_timestamp', 'symbol', 'side', 'entry_timestamp', 'entry_price',
            'exit_timestamp', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason'
        ])

    trades = []
    cash = initial_cash

    # Group by symbol and process chronologically
    for sym, sym_signals in signals.groupby('symbol'):
        sym_signals = sym_signals.sort_values('timestamp').reset_index(drop=True)

        try:
            bars = fetch_bars(sym)
            if bars is None or bars.empty:
                continue

            bars['timestamp'] = pd.to_datetime(bars['timestamp'])
            bars = bars.sort_values('timestamp').reset_index(drop=True)
        except Exception:
            continue

        open_trade = None

        for _, sig in sym_signals.iterrows():
            sig_ts = pd.to_datetime(sig['timestamp'])

            # Skip if already in a trade for this symbol
            if open_trade is not None:
                continue

            # Find next bar after signal for entry
            next_bars = bars[bars['timestamp'] > sig_ts]
            if next_bars.empty:
                continue

            entry_idx = next_bars.index[0]
            entry_bar = bars.loc[entry_idx]
            entry_ts = entry_bar['timestamp']

            # Calculate entry price with slippage
            side = str(sig.get('side', 'long')).lower()
            entry_open = float(entry_bar['open'])
            slippage_mult = 1 + (slippage_bps / 10000) * (1 if side == 'long' else -1)
            entry_price = entry_open * slippage_mult

            # Position sizing
            notional = cash * position_size_pct
            qty = int(max(1, notional // entry_price))

            if qty <= 0:
                continue

            stop_loss = sig.get('stop_loss')
            open_trade = {
                'signal_timestamp': sig_ts,
                'symbol': sym,
                'side': side,
                'entry_idx': entry_idx,
                'entry_timestamp': entry_ts,
                'entry_price': entry_price,
                'qty': qty,
                'stop_loss': float(stop_loss) if pd.notna(stop_loss) else None,
            }

            # Simulate exit (ATR stop or 5-bar time stop)
            time_stop_bars = 5
            exit_reason = None
            exit_price = None
            exit_ts = None

            for j in range(entry_idx + 1, min(entry_idx + 1 + time_stop_bars, len(bars))):
                bar = bars.loc[j]

                # Check stop loss
                if open_trade['stop_loss'] is not None:
                    if side == 'long' and float(bar['low']) <= open_trade['stop_loss']:
                        exit_price = open_trade['stop_loss']
                        exit_ts = bar['timestamp']
                        exit_reason = 'stop_loss'
                        break
                    elif side == 'short' and float(bar['high']) >= open_trade['stop_loss']:
                        exit_price = open_trade['stop_loss']
                        exit_ts = bar['timestamp']
                        exit_reason = 'stop_loss'
                        break

                # Time stop at final bar
                if j == entry_idx + time_stop_bars - 1:
                    exit_price = float(bar['close'])
                    exit_ts = bar['timestamp']
                    exit_reason = 'time_stop'
                    break

            # If no exit yet, exit at last available bar
            if exit_price is None:
                last_idx = min(entry_idx + time_stop_bars, len(bars) - 1)
                bar = bars.loc[last_idx]
                exit_price = float(bar['close'])
                exit_ts = bar['timestamp']
                exit_reason = 'time_stop'

            # Apply exit slippage
            exit_slippage = 1 - (slippage_bps / 10000) * (1 if side == 'long' else -1)
            exit_price = exit_price * exit_slippage

            # Calculate PnL
            if side == 'long':
                pnl = (exit_price - open_trade['entry_price']) * open_trade['qty']
                pnl_pct = (exit_price - open_trade['entry_price']) / open_trade['entry_price']
            else:
                pnl = (open_trade['entry_price'] - exit_price) * open_trade['qty']
                pnl_pct = (open_trade['entry_price'] - exit_price) / open_trade['entry_price']

            trades.append({
                'signal_timestamp': open_trade['signal_timestamp'],
                'symbol': sym,
                'side': side,
                'qty': open_trade['qty'],
                'entry_timestamp': open_trade['entry_timestamp'],
                'entry_price': round(open_trade['entry_price'], 2),
                'stop_loss': open_trade['stop_loss'],
                'exit_timestamp': exit_ts,
                'exit_price': round(exit_price, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct * 100, 2),
                'exit_reason': exit_reason,
            })

            # Update cash
            cash += pnl
            open_trade = None

    if trades:
        return pd.DataFrame(trades).sort_values('signal_timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            'signal_timestamp', 'symbol', 'side', 'qty', 'entry_timestamp',
            'entry_price', 'stop_loss', 'exit_timestamp', 'exit_price',
            'pnl', 'pnl_pct', 'exit_reason'
        ])


def compare_with_actual(
    simulated: pd.DataFrame,
    actual: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare simulated signals/trades with actual trades.

    Returns comparison statistics.
    """
    if simulated.empty:
        return {
            'simulated_count': 0,
            'actual_count': len(actual),
            'matched': 0,
            'missed': len(actual),
            'extra': 0,
        }

    if actual.empty:
        return {
            'simulated_count': len(simulated),
            'actual_count': 0,
            'matched': 0,
            'missed': 0,
            'extra': len(simulated),
        }

    # Normalize timestamps to dates for comparison
    sim_signals = set()
    for _, row in simulated.iterrows():
        ts = pd.to_datetime(row.get('signal_timestamp', row.get('timestamp')))
        sym = row['symbol']
        side = row['side'].upper() if 'side' in row else 'BUY'
        # Use date only for matching (signals at close lead to next-day trades)
        sim_signals.add((ts.date(), sym, side))

    actual_signals = set()
    for _, row in actual.iterrows():
        ts = pd.to_datetime(row['timestamp'])
        sym = row['symbol']
        side = row['side'].upper()
        if side == 'BUY':
            # Actual BUY corresponds to simulated long signal day before
            actual_signals.add((ts.date(), sym, 'LONG'))

    matched = sim_signals.intersection(actual_signals)
    missed = actual_signals - sim_signals
    extra = sim_signals - actual_signals

    return {
        'simulated_count': len(simulated),
        'actual_count': len(actual),
        'matched': len(matched),
        'missed': len(missed),
        'extra': len(extra),
        'match_rate': len(matched) / len(actual_signals) * 100 if actual_signals else 0,
    }


def format_signals_table(signals: pd.DataFrame, limit: int = 50) -> str:
    """Format signals as text table."""
    if signals.empty:
        return "[INFO] No signals generated for this period"

    lines = []
    lines.append("")
    lines.append("REPLAYED SIGNALS")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Date':<12} | {'Symbol':<8} | {'Side':<6} | {'Entry':>10} | {'Stop':>10} | {'Reason':<30}"
    lines.append(header)
    lines.append("-" * len(header))

    # Limit display
    display_df = signals.head(limit)

    for _, row in display_df.iterrows():
        ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')
        sym = str(row['symbol'])[:8]
        side = str(row['side'])[:6].upper()
        entry = f"${row['entry_price']:,.2f}" if pd.notna(row.get('entry_price')) else 'N/A'
        stop = f"${row['stop_loss']:,.2f}" if pd.notna(row.get('stop_loss')) else 'N/A'
        reason = str(row.get('reason', ''))[:30]

        line = f"{ts:<12} | {sym:<8} | {side:<6} | {entry:>10} | {stop:>10} | {reason:<30}"
        lines.append(line)

    if len(signals) > limit:
        lines.append(f"\n... and {len(signals) - limit} more signals")

    lines.append("")
    lines.append(f"Total Signals: {len(signals)}")
    lines.append("")

    return '\n'.join(lines)


def format_trades_table(trades: pd.DataFrame, limit: int = 50) -> str:
    """Format simulated trades as text table."""
    if trades.empty:
        return "[INFO] No simulated trades for this period"

    lines = []
    lines.append("")
    lines.append("SIMULATED TRADES")
    lines.append("=" * 120)
    lines.append("")

    # Header
    header = (f"{'Signal Date':<12} | {'Symbol':<8} | {'Side':<5} | {'Entry':>10} | "
              f"{'Exit':>10} | {'PnL':>12} | {'PnL%':>8} | {'Exit Reason':<12}")
    lines.append(header)
    lines.append("-" * len(header))

    # Limit display
    display_df = trades.head(limit)

    for _, row in display_df.iterrows():
        sig_ts = pd.to_datetime(row['signal_timestamp']).strftime('%Y-%m-%d')
        sym = str(row['symbol'])[:8]
        side = str(row['side'])[:5].upper()
        entry = f"${row['entry_price']:,.2f}"
        exit_px = f"${row['exit_price']:,.2f}"
        pnl = row['pnl']
        pnl_str = f"${pnl:>+,.2f}"
        pnl_pct = f"{row['pnl_pct']:>+.2f}%"
        exit_reason = str(row['exit_reason'])[:12]

        line = (f"{sig_ts:<12} | {sym:<8} | {side:<5} | {entry:>10} | "
                f"{exit_px:>10} | {pnl_str:>12} | {pnl_pct:>8} | {exit_reason:<12}")
        lines.append(line)

    if len(trades) > limit:
        lines.append(f"\n... and {len(trades) - limit} more trades")

    # Summary stats
    lines.append("")
    lines.append("-" * 60)
    total_pnl = trades['pnl'].sum()
    win_count = (trades['pnl'] > 0).sum()
    loss_count = (trades['pnl'] <= 0).sum()
    win_rate = win_count / len(trades) * 100 if len(trades) > 0 else 0

    lines.append(f"Total Trades: {len(trades)}")
    lines.append(f"Total PnL: ${total_pnl:,.2f}")
    lines.append(f"Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)")

    if len(trades) > 0:
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if win_count > 0 else 0
        avg_loss = trades[trades['pnl'] <= 0]['pnl'].mean() if loss_count > 0 else 0
        lines.append(f"Avg Win: ${avg_win:,.2f}")
        lines.append(f"Avg Loss: ${avg_loss:,.2f}")

    lines.append("")

    return '\n'.join(lines)


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format comparison results as text table."""
    lines = []
    lines.append("")
    lines.append("COMPARISON WITH ACTUAL TRADES")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"{'Simulated Signals:':<25} {comparison['simulated_count']:>10}")
    lines.append(f"{'Actual Trades:':<25} {comparison['actual_count']:>10}")
    lines.append(f"{'Matched:':<25} {comparison['matched']:>10}")
    lines.append(f"{'Missed (in actual):':<25} {comparison['missed']:>10}")
    lines.append(f"{'Extra (not in actual):':<25} {comparison['extra']:>10}")
    lines.append(f"{'Match Rate:':<25} {comparison.get('match_rate', 0):>9.1f}%")
    lines.append("")

    return '\n'.join(lines)


def save_results(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    comparison: Optional[Dict[str, Any]],
    outdir: Path,
    strategy: str,
    start: str,
    end: str,
) -> None:
    """Save replay results to files."""
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = f"{strategy}_{start}_{end}"

    # Save signals
    signals.to_csv(outdir / f'{prefix}_signals.csv', index=False)

    # Save simulated trades
    trades.to_csv(outdir / f'{prefix}_simulated_trades.csv', index=False)

    # Save comparison if available
    if comparison:
        import json
        with open(outdir / f'{prefix}_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"[INFO] Results saved to {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description='Historical signal replay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay.py --start 2023-01-01 --end 2023-12-31 --strategy rsi2
  python replay.py --start 2023-06-01 --end 2023-06-30 --strategy ibs --actual-trades trades.csv
  python replay.py --start 2023-01-01 --end 2023-03-31 --strategy rsi2 --symbols AAPL,MSFT,NVDA
        """
    )
    ap.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    ap.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    ap.add_argument('--strategy', type=str, default='rsi2', choices=['rsi2', 'ibs'],
                    help='Strategy to replay')
    ap.add_argument('--universe', type=str, default='data/universe.csv',
                    help='Path to universe CSV file')
    ap.add_argument('--symbols', type=str, default=None,
                    help='Comma-separated list of symbols (overrides --universe)')
    ap.add_argument('--actual-trades', type=str, default=None,
                    help='Path to actual trades CSV for comparison')
    ap.add_argument('--cache', type=str, default='data/cache',
                    help='Cache directory for price data')
    ap.add_argument('--cap', type=int, default=100,
                    help='Max symbols from universe')
    ap.add_argument('--outdir', type=str, default='outputs/replay',
                    help='Output directory for results')
    ap.add_argument('--limit', type=int, default=50,
                    help='Max rows to display in tables')
    ap.add_argument('--simulate-trades', action='store_true',
                    help='Simulate full trades (not just signals)')
    ap.add_argument('--dotenv', type=str, default=None, help='Path to .env file')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress output')
    args = ap.parse_args()

    # Load environment variables
    if args.dotenv:
        dotenv_path = Path(args.dotenv)
        if dotenv_path.exists():
            loaded = load_env(dotenv_path)
            if not args.quiet:
                print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv_path}")

    # Determine symbols to use
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        print(f"[INFO] Using {len(symbols)} symbols from command line")
    else:
        from data.universe.loader import load_universe
        universe_path = Path(args.universe)
        if universe_path.exists():
            symbols = load_universe(universe_path, cap=args.cap)
            print(f"[INFO] Loaded {len(symbols)} symbols from universe")
        else:
            # Default symbols if no universe
            symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'INTC', 'SPY']
            print(f"[INFO] Using default {len(symbols)} symbols (universe file not found)")

    # Setup data fetcher
    cache_dir = Path(args.cache)

    # Fetch with lookback for indicator calculation
    # We need data before start date for SMA200 etc.
    lookback_start = (datetime.strptime(args.start, '%Y-%m-%d') -
                     pd.Timedelta(days=300)).strftime('%Y-%m-%d')

    try:
        from data.providers.polygon_eod import fetch_daily_bars_polygon

        def fetch_bars(sym: str) -> pd.DataFrame:
            return fetch_daily_bars_polygon(sym, lookback_start, args.end, cache_dir=cache_dir)

        print("[INFO] Using Polygon data provider")
    except ImportError:
        print("[WARN] Polygon provider not available, using synthetic data")

        def fetch_bars(sym: str) -> pd.DataFrame:
            np.random.seed(abs(hash(sym)) % 2**32)
            days = 400
            dates = pd.date_range(end=args.end, periods=days, freq='B')
            rets = np.random.normal(0.0004, 0.01, days)
            close = 100 * np.cumprod(1 + rets)
            return pd.DataFrame({
                'timestamp': dates,
                'symbol': sym,
                'open': close * (1 + np.random.uniform(-0.002, 0.002, days)),
                'high': close * (1 + np.random.uniform(0, 0.01, days)),
                'low': close * (1 - np.random.uniform(0, 0.01, days)),
                'close': close,
                'volume': np.random.randint(1_000_000, 5_000_000, days),
            })

    # Create strategy
    print(f"[INFO] Replaying {args.strategy.upper()} strategy from {args.start} to {args.end}")
    strategy = create_strategy(args.strategy)

    # Generate signals
    signals = generate_signals_for_period(
        strategy, symbols, args.start, args.end,
        fetch_bars, show_progress=not args.quiet
    )

    # Display signals
    print(format_signals_table(signals, args.limit))

    # Simulate trades if requested
    trades = pd.DataFrame()
    if args.simulate_trades or args.actual_trades:
        print("[INFO] Simulating trades...")
        trades = simulate_trades(signals, fetch_bars)
        print(format_trades_table(trades, args.limit))

    # Compare with actual trades if provided
    comparison = None
    if args.actual_trades:
        actual_path = Path(args.actual_trades)
        if actual_path.exists():
            print(f"[INFO] Loading actual trades from {actual_path}")
            actual = load_actual_trades(actual_path)

            # Filter actual trades to date range
            if not actual.empty:
                actual['timestamp'] = pd.to_datetime(actual['timestamp'])
                actual = actual[
                    (actual['timestamp'] >= args.start) &
                    (actual['timestamp'] <= args.end)
                ]

            comparison = compare_with_actual(signals, actual)
            print(format_comparison_table(comparison))
        else:
            print(f"[WARN] Actual trades file not found: {actual_path}")

    # Summary
    print("=" * 60)
    print("REPLAY SUMMARY")
    print("=" * 60)
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {len(symbols)}")
    print(f"Signals Generated: {len(signals)}")
    if not trades.empty:
        print(f"Trades Simulated: {len(trades)}")
        print(f"Total PnL: ${trades['pnl'].sum():,.2f}")
    print("=" * 60)

    # Save results
    save_results(signals, trades, comparison, Path(args.outdir), args.strategy, args.start, args.end)


if __name__ == '__main__':
    main()
