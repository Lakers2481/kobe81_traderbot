#!/usr/bin/env python3
"""
Monte Carlo Simulation for Trade Performance

- Bootstrap trade returns to generate scenarios
- Calculate confidence intervals for future performance
- Output percentile outcomes (5th, 50th, 95th)

Usage:
    python simulate.py --trades outputs/trade_list.csv --simulations 10000 --horizon 252
    python simulate.py --trades outputs/trade_list.csv --dotenv path/to/.env
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.env_loader import load_env


def load_trades(path: Path) -> pd.DataFrame:
    """
    Load trades from CSV file.
    Expected columns: timestamp, symbol, side, qty, price
    Returns DataFrame with trade data or empty DataFrame if file missing.
    """
    if not path.exists():
        print(f"[WARN] Trade file not found: {path}")
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])

    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        required = {'timestamp', 'symbol', 'side', 'qty', 'price'}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            print(f"[WARN] Missing columns in trade file: {missing}")
            return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] Failed to load trades: {e}")
        return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'qty', 'price'])


def compute_trade_returns(trades: pd.DataFrame) -> np.ndarray:
    """
    Compute per-trade returns from BUY/SELL pairs (FIFO matching).
    Returns array of trade PnL percentages.
    """
    from collections import defaultdict, deque

    if trades.empty:
        return np.array([])

    buys: Dict[str, deque] = defaultdict(deque)
    returns: List[float] = []

    for _, tr in trades.iterrows():
        symbol = str(tr['symbol'])
        side = str(tr['side']).upper()
        qty = int(tr['qty'])
        price = float(tr['price'])

        if side == 'BUY':
            buys[symbol].append((qty, price))
        elif side == 'SELL':
            remaining = qty
            while remaining > 0 and buys[symbol]:
                bqty, bpx = buys[symbol][0]
                used = min(remaining, bqty)

                # Calculate return as percentage
                if bpx > 0:
                    ret = (price - bpx) / bpx
                    returns.append(ret)

                remaining -= used
                bqty -= used
                if bqty == 0:
                    buys[symbol].popleft()
                else:
                    buys[symbol][0] = (bqty, bpx)

    return np.array(returns)


def run_monte_carlo(
    trade_returns: np.ndarray,
    n_simulations: int = 10000,
    horizon: int = 252,
    initial_equity: float = 100000.0,
    position_size: float = 0.007,
    trades_per_period: Optional[float] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation by bootstrapping trade returns.

    Args:
        trade_returns: Array of per-trade returns (as decimals)
        n_simulations: Number of simulation paths
        horizon: Number of trading days to simulate
        initial_equity: Starting equity
        position_size: Fraction of equity per trade
        trades_per_period: Expected trades per day (estimated from data if None)
        show_progress: Show progress indicator

    Returns:
        Dictionary with simulation results and statistics
    """
    if len(trade_returns) == 0:
        return {
            'n_simulations': n_simulations,
            'horizon': horizon,
            'initial_equity': initial_equity,
            'final_equities': np.array([initial_equity] * n_simulations),
            'percentiles': {
                '5th': initial_equity,
                '25th': initial_equity,
                '50th': initial_equity,
                '75th': initial_equity,
                '95th': initial_equity,
            },
            'mean': initial_equity,
            'std': 0.0,
            'prob_profit': 0.0,
            'max_drawdown_mean': 0.0,
            'sharpe_mean': 0.0,
            'error': 'No trade returns available for simulation',
        }

    # Estimate trades per day if not provided
    if trades_per_period is None:
        # Assume historical data represents actual trading frequency
        # Default to ~1 trade per 5 days for mean-reversion strategies
        trades_per_period = max(0.2, len(trade_returns) / 252)

    final_equities = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    sharpe_ratios = np.zeros(n_simulations)

    progress_step = max(1, n_simulations // 20)

    for sim in range(n_simulations):
        if show_progress and sim % progress_step == 0:
            pct = int(100 * sim / n_simulations)
            print(f"\r[PROGRESS] Simulation: {pct}%", end='', flush=True)

        equity = initial_equity
        peak = equity
        max_dd = 0.0
        daily_returns: List[float] = []

        for day in range(horizon):
            # Sample number of trades for this day (Poisson)
            n_trades = np.random.poisson(trades_per_period)
            daily_pnl = 0.0

            for _ in range(n_trades):
                # Bootstrap: sample a random trade return
                ret = np.random.choice(trade_returns)
                trade_pnl = equity * position_size * ret
                daily_pnl += trade_pnl

            equity += daily_pnl
            equity = max(0, equity)  # Cannot go below zero

            # Track drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            # Track daily return
            if equity > 0 and (equity - daily_pnl) > 0:
                daily_ret = daily_pnl / (equity - daily_pnl) if (equity - daily_pnl) > 0 else 0
                daily_returns.append(daily_ret)

        final_equities[sim] = equity
        max_drawdowns[sim] = max_dd

        # Calculate Sharpe for this path
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratios[sim] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratios[sim] = 0.0

    if show_progress:
        print("\r[PROGRESS] Simulation: 100%")

    # Calculate percentiles
    percentiles = {
        '5th': float(np.percentile(final_equities, 5)),
        '25th': float(np.percentile(final_equities, 25)),
        '50th': float(np.percentile(final_equities, 50)),
        '75th': float(np.percentile(final_equities, 75)),
        '95th': float(np.percentile(final_equities, 95)),
    }

    return {
        'n_simulations': n_simulations,
        'horizon': horizon,
        'initial_equity': initial_equity,
        'n_trades_input': len(trade_returns),
        'avg_trade_return': float(np.mean(trade_returns)),
        'trade_return_std': float(np.std(trade_returns)),
        'win_rate': float(np.mean(trade_returns > 0)),
        'trades_per_day': trades_per_period,
        'final_equities': final_equities,
        'percentiles': percentiles,
        'mean': float(np.mean(final_equities)),
        'std': float(np.std(final_equities)),
        'min': float(np.min(final_equities)),
        'max': float(np.max(final_equities)),
        'prob_profit': float(np.mean(final_equities > initial_equity)),
        'prob_loss_50pct': float(np.mean(final_equities < initial_equity * 0.5)),
        'max_drawdown_mean': float(np.mean(max_drawdowns)),
        'max_drawdown_95th': float(np.percentile(max_drawdowns, 95)),
        'sharpe_mean': float(np.mean(sharpe_ratios)),
        'sharpe_median': float(np.median(sharpe_ratios)),
    }


def format_results_table(results: Dict[str, Any]) -> str:
    """Format simulation results as a text table."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("MONTE CARLO SIMULATION RESULTS")
    lines.append("=" * 60)
    lines.append("")

    # Input summary
    lines.append("INPUT PARAMETERS")
    lines.append("-" * 40)
    lines.append(f"{'Simulations:':<25} {results['n_simulations']:>15,}")
    lines.append(f"{'Horizon (days):':<25} {results['horizon']:>15,}")
    lines.append(f"{'Initial Equity:':<25} ${results['initial_equity']:>14,.2f}")
    lines.append(f"{'Input Trades:':<25} {results.get('n_trades_input', 0):>15,}")
    lines.append(f"{'Avg Trade Return:':<25} {results.get('avg_trade_return', 0)*100:>14.2f}%")
    lines.append(f"{'Trade Return Std:':<25} {results.get('trade_return_std', 0)*100:>14.2f}%")
    lines.append(f"{'Historical Win Rate:':<25} {results.get('win_rate', 0)*100:>14.1f}%")
    lines.append(f"{'Trades/Day (est):':<25} {results.get('trades_per_day', 0):>15.2f}")
    lines.append("")

    # Percentile outcomes
    lines.append("FINAL EQUITY DISTRIBUTION")
    lines.append("-" * 40)
    pct = results['percentiles']
    lines.append(f"{'5th Percentile:':<25} ${pct['5th']:>14,.2f}")
    lines.append(f"{'25th Percentile:':<25} ${pct['25th']:>14,.2f}")
    lines.append(f"{'50th Percentile (Median):':<25} ${pct['50th']:>14,.2f}")
    lines.append(f"{'75th Percentile:':<25} ${pct['75th']:>14,.2f}")
    lines.append(f"{'95th Percentile:':<25} ${pct['95th']:>14,.2f}")
    lines.append("")

    # Summary statistics
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"{'Mean Final Equity:':<25} ${results['mean']:>14,.2f}")
    lines.append(f"{'Std Dev:':<25} ${results['std']:>14,.2f}")
    lines.append(f"{'Minimum:':<25} ${results['min']:>14,.2f}")
    lines.append(f"{'Maximum:':<25} ${results['max']:>14,.2f}")
    lines.append("")

    # Risk metrics
    lines.append("RISK METRICS")
    lines.append("-" * 40)
    lines.append(f"{'Probability of Profit:':<25} {results['prob_profit']*100:>14.1f}%")
    lines.append(f"{'Probability of 50% Loss:':<25} {results.get('prob_loss_50pct', 0)*100:>14.1f}%")
    lines.append(f"{'Mean Max Drawdown:':<25} {results['max_drawdown_mean']*100:>14.1f}%")
    lines.append(f"{'95th Pctile Max DD:':<25} {results.get('max_drawdown_95th', 0)*100:>14.1f}%")
    lines.append(f"{'Mean Sharpe Ratio:':<25} {results['sharpe_mean']:>15.2f}")
    lines.append(f"{'Median Sharpe Ratio:':<25} {results.get('sharpe_median', 0):>15.2f}")
    lines.append("")

    # Confidence intervals
    lines.append("CONFIDENCE INTERVALS")
    lines.append("-" * 40)
    ci_90_low = pct['5th']
    ci_90_high = pct['95th']
    lines.append(f"{'90% CI:':<25} ${ci_90_low:>10,.2f} - ${ci_90_high:>10,.2f}")
    ci_50_low = pct['25th']
    ci_50_high = pct['75th']
    lines.append(f"{'50% CI:':<25} ${ci_50_low:>10,.2f} - ${ci_50_high:>10,.2f}")

    # Expected return
    expected_return = (results['mean'] - results['initial_equity']) / results['initial_equity'] * 100
    median_return = (pct['50th'] - results['initial_equity']) / results['initial_equity'] * 100
    lines.append("")
    lines.append(f"{'Expected Return:':<25} {expected_return:>14.1f}%")
    lines.append(f"{'Median Return:':<25} {median_return:>14.1f}%")

    lines.append("")
    lines.append("=" * 60)

    if 'error' in results:
        lines.append(f"[WARNING] {results['error']}")
        lines.append("=" * 60)

    return '\n'.join(lines)


def save_results(results: Dict[str, Any], outdir: Path) -> None:
    """Save simulation results to files."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Save summary as CSV
    summary_data = {
        'metric': [
            'simulations', 'horizon', 'initial_equity', 'input_trades',
            'avg_trade_return', 'win_rate', 'trades_per_day',
            'pct_5th', 'pct_25th', 'pct_50th', 'pct_75th', 'pct_95th',
            'mean', 'std', 'min', 'max',
            'prob_profit', 'prob_loss_50pct',
            'max_drawdown_mean', 'max_drawdown_95th',
            'sharpe_mean', 'sharpe_median',
        ],
        'value': [
            results['n_simulations'], results['horizon'], results['initial_equity'],
            results.get('n_trades_input', 0),
            results.get('avg_trade_return', 0), results.get('win_rate', 0),
            results.get('trades_per_day', 0),
            results['percentiles']['5th'], results['percentiles']['25th'],
            results['percentiles']['50th'], results['percentiles']['75th'],
            results['percentiles']['95th'],
            results['mean'], results['std'], results['min'], results['max'],
            results['prob_profit'], results.get('prob_loss_50pct', 0),
            results['max_drawdown_mean'], results.get('max_drawdown_95th', 0),
            results['sharpe_mean'], results.get('sharpe_median', 0),
        ],
    }
    pd.DataFrame(summary_data).to_csv(outdir / 'monte_carlo_summary.csv', index=False)

    # Save distribution of final equities
    dist_df = pd.DataFrame({'final_equity': results['final_equities']})
    dist_df.to_csv(outdir / 'monte_carlo_distribution.csv', index=False)

    print(f"[INFO] Results saved to {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description='Monte Carlo simulation on trade history',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate.py --trades outputs/trade_list.csv
  python simulate.py --trades outputs/trade_list.csv --simulations 50000 --horizon 504
  python simulate.py --trades outputs/trade_list.csv --initial-equity 50000 --position-size 0.01
        """
    )
    ap.add_argument('--trades', type=str, required=True, help='Path to trade list CSV')
    ap.add_argument('--simulations', type=int, default=10000, help='Number of Monte Carlo simulations (default: 10000)')
    ap.add_argument('--horizon', type=int, default=252, help='Simulation horizon in trading days (default: 252 = 1 year)')
    ap.add_argument('--initial-equity', type=float, default=100000.0, help='Initial equity (default: 100000)')
    ap.add_argument('--position-size', type=float, default=0.007, help='Position size as fraction of equity (default: 0.007)')
    ap.add_argument('--trades-per-day', type=float, default=None, help='Expected trades per day (estimated from data if not set)')
    ap.add_argument('--outdir', type=str, default='outputs/monte_carlo', help='Output directory for results')
    ap.add_argument('--dotenv', type=str, default=None, help='Path to .env file')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress output')
    args = ap.parse_args()

    # Load environment variables if specified
    if args.dotenv:
        dotenv_path = Path(args.dotenv)
        if dotenv_path.exists():
            loaded = load_env(dotenv_path)
            if not args.quiet:
                print(f"[INFO] Loaded {len(loaded)} env vars from {dotenv_path}")
        else:
            print(f"[WARN] .env file not found: {dotenv_path}")

    # Load trades
    trades_path = Path(args.trades)
    print(f"[INFO] Loading trades from {trades_path}")
    trades = load_trades(trades_path)

    if trades.empty:
        print("[ERROR] No trades loaded. Exiting.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(trades)} trade records")

    # Compute trade returns
    print("[INFO] Computing trade returns from BUY/SELL pairs...")
    trade_returns = compute_trade_returns(trades)

    if len(trade_returns) == 0:
        print("[WARN] No completed trades found (need BUY/SELL pairs)")
    else:
        print(f"[INFO] Found {len(trade_returns)} completed trades")
        print(f"[INFO] Average return: {np.mean(trade_returns)*100:.2f}%, Win rate: {np.mean(trade_returns > 0)*100:.1f}%")

    # Run Monte Carlo simulation
    print(f"\n[INFO] Running {args.simulations:,} simulations over {args.horizon} day horizon...")
    results = run_monte_carlo(
        trade_returns,
        n_simulations=args.simulations,
        horizon=args.horizon,
        initial_equity=args.initial_equity,
        position_size=args.position_size,
        trades_per_period=args.trades_per_day,
        show_progress=not args.quiet,
    )

    # Display results
    print(format_results_table(results))

    # Save results
    save_results(results, Path(args.outdir))


if __name__ == '__main__':
    main()
