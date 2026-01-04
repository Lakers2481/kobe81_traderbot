#!/usr/bin/env python3
"""
Monte Carlo Robustness Testing for Kobe81 Trading Bot.
Reorders trades and performs block bootstrap to assess strategy robustness.
Outputs distribution plots and robustness scores.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path as _P
from typing import Dict, List, Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from config.env_loader import load_env


def main():
    ap = argparse.ArgumentParser(description='Monte Carlo robustness testing')
    ap.add_argument('--trades', type=str, required=True, help='Path to trade_list.csv')
    ap.add_argument('--equity', type=str, help='Path to equity_curve.csv (optional)')
    ap.add_argument('--iterations', type=int, default=1000, help='Number of Monte Carlo iterations')
    ap.add_argument('--block-size', type=int, default=5, help='Block size for bootstrap')
    ap.add_argument('--outdir', type=str, default='monte_carlo_outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
    args = ap.parse_args()

    dotenv = _P(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = _P(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load trades
    trades_df = pd.read_csv(args.trades)
    print(f"Loaded {len(trades_df)} trades from {args.trades}")

    # Compute per-trade PnL
    pnls = compute_trade_pnls(trades_df)
    if len(pnls) < 10:
        print("Too few trades for meaningful Monte Carlo analysis")
        return

    # Run Monte Carlo simulations
    print(f"Running {args.iterations} Monte Carlo iterations...")
    mc_results = run_monte_carlo(pnls, args.iterations, args.block_size)

    # Compute robustness metrics
    robustness = compute_robustness_score(mc_results, pnls)

    # Save results
    mc_df = pd.DataFrame(mc_results)
    mc_df.to_csv(outdir / 'monte_carlo_distributions.csv', index=False)

    with open(outdir / 'robustness_score.json', 'w') as f:
        json.dump(robustness, f, indent=2)

    # Generate HTML report
    _generate_html_report(mc_results, robustness, outdir)

    print(f"\nMonte Carlo analysis complete. Results in {outdir}/")
    print(f"Robustness Score: {robustness['overall_score']:.2f}/100")
    print(f"Original Sharpe: {robustness['original']['sharpe']:.2f}")
    print(f"MC Sharpe (5th-95th): {robustness['mc_percentiles']['sharpe_5th']:.2f} - {robustness['mc_percentiles']['sharpe_95th']:.2f}")


def compute_trade_pnls(trades_df: pd.DataFrame) -> List[float]:
    """
    Compute per-trade PnL from trade list.
    Assumes BUY/SELL pairs in chronological order.
    """
    pnls = []
    buys = {}  # symbol -> [(qty, price), ...]

    for _, row in trades_df.iterrows():
        sym = row['symbol']
        side = str(row['side']).upper()
        qty = int(row['qty'])
        price = float(row['price'])

        if side == 'BUY':
            if sym not in buys:
                buys[sym] = []
            buys[sym].append((qty, price))
        elif side == 'SELL':
            if sym in buys and buys[sym]:
                # Match FIFO
                remaining = qty
                while remaining > 0 and buys[sym]:
                    bqty, bpx = buys[sym][0]
                    used = min(remaining, bqty)
                    pnl = (price - bpx) * used
                    pnls.append(pnl)
                    remaining -= used
                    bqty -= used
                    if bqty == 0:
                        buys[sym].pop(0)
                    else:
                        buys[sym][0] = (bqty, bpx)

    return pnls


def run_monte_carlo(pnls: List[float], iterations: int, block_size: int) -> List[Dict[str, float]]:
    """
    Run Monte Carlo simulations with trade reordering and block bootstrap.
    """
    results = []
    original = pnls.copy()

    for i in range(iterations):
        if i % 2 == 0:
            # Random shuffle
            shuffled = original.copy()
            random.shuffle(shuffled)
        else:
            # Block bootstrap
            shuffled = block_bootstrap(original, block_size)

        # Compute metrics for shuffled trades
        metrics = compute_metrics(shuffled)
        results.append(metrics)

    return results


def block_bootstrap(pnls: List[float], block_size: int) -> List[float]:
    """
    Block bootstrap sampling to preserve some autocorrelation.
    """
    n = len(pnls)
    if n <= block_size:
        return pnls.copy()

    blocks = [pnls[i:i+block_size] for i in range(0, n, block_size)]
    n_blocks = len(blocks)

    # Sample blocks with replacement
    sampled_blocks = [blocks[random.randint(0, n_blocks-1)] for _ in range(n_blocks)]

    # Flatten
    result = []
    for block in sampled_blocks:
        result.extend(block)

    return result[:n]  # Trim to original length


def compute_metrics(pnls: List[float]) -> Dict[str, float]:
    """Compute trading metrics from PnL series."""
    if not pnls:
        return {'sharpe': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0}

    # Win rate
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else 0.0

    # Profit factor
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = sum(p for p in pnls if p < 0)
    profit_factor = abs(gross_win / gross_loss) if gross_loss < 0 else (float('inf') if gross_win > 0 else 0.0)
    if profit_factor == float('inf'):
        profit_factor = 100.0  # Cap for JSON serialization

    # Total PnL
    total_pnl = sum(pnls)

    # Sharpe (using daily returns approximation)
    returns = np.array(pnls) / 100000  # Normalize by starting capital
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Max drawdown
    cumsum = np.cumsum(pnls)
    cummax = np.maximum.accumulate(cumsum)
    dd = cumsum - cummax
    max_drawdown = np.min(dd) if len(dd) > 0 else 0.0

    return {
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_pnl': float(total_pnl),
    }


def compute_robustness_score(mc_results: List[Dict[str, float]], original_pnls: List[float]) -> Dict[str, Any]:
    """
    Compute overall robustness score based on Monte Carlo results.
    Score 0-100 based on how original compares to MC distribution.
    """
    original_metrics = compute_metrics(original_pnls)

    # Extract distributions
    sharpes = [r['sharpe'] for r in mc_results]
    max_dds = [r['max_drawdown'] for r in mc_results]
    win_rates = [r['win_rate'] for r in mc_results]
    [r['profit_factor'] for r in mc_results]

    # Percentiles
    mc_percentiles = {
        'sharpe_5th': np.percentile(sharpes, 5),
        'sharpe_50th': np.percentile(sharpes, 50),
        'sharpe_95th': np.percentile(sharpes, 95),
        'max_dd_5th': np.percentile(max_dds, 5),
        'max_dd_50th': np.percentile(max_dds, 50),
        'max_dd_95th': np.percentile(max_dds, 95),
        'win_rate_5th': np.percentile(win_rates, 5),
        'win_rate_50th': np.percentile(win_rates, 50),
        'win_rate_95th': np.percentile(win_rates, 95),
    }

    # Robustness score components
    # 1. Sharpe stability: how close is original to median?
    sharpe_stability = 1 - abs(original_metrics['sharpe'] - mc_percentiles['sharpe_50th']) / (max(abs(original_metrics['sharpe']), 0.1))
    sharpe_stability = max(0, min(1, sharpe_stability))

    # 2. Drawdown stability
    dd_stability = 1 - abs(original_metrics['max_drawdown'] - mc_percentiles['max_dd_50th']) / (max(abs(original_metrics['max_drawdown']), 100))
    dd_stability = max(0, min(1, dd_stability))

    # 3. Win rate stability
    wr_stability = 1 - abs(original_metrics['win_rate'] - mc_percentiles['win_rate_50th']) / max(original_metrics['win_rate'], 0.1)
    wr_stability = max(0, min(1, wr_stability))

    # 4. Positive expectancy in MC
    pos_expectancy = sum(1 for r in mc_results if r['total_pnl'] > 0) / len(mc_results)

    # Overall score (0-100)
    overall_score = (sharpe_stability * 25 + dd_stability * 25 + wr_stability * 25 + pos_expectancy * 25)

    return {
        'overall_score': round(overall_score, 1),
        'components': {
            'sharpe_stability': round(sharpe_stability * 100, 1),
            'dd_stability': round(dd_stability * 100, 1),
            'win_rate_stability': round(wr_stability * 100, 1),
            'positive_expectancy_pct': round(pos_expectancy * 100, 1),
        },
        'original': original_metrics,
        'mc_percentiles': mc_percentiles,
        'n_iterations': len(mc_results),
    }


def _generate_html_report(
    mc_results: List[Dict[str, float]],
    robustness: Dict[str, Any],
    outdir: _P,
) -> None:
    """Generate HTML Monte Carlo report."""
    html = [
        '<html><head><meta charset="utf-8"><title>Monte Carlo Robustness Report</title>',
        '<style>body{font-family:Arial;margin:20px} .score{font-size:48px;font-weight:bold;color:#28a745} .warning{color:#dc3545} table{border-collapse:collapse;margin:10px 0} th,td{border:1px solid #ddd;padding:6px} th{background:#f3f3f3}</style>',
        '</head><body>',
        '<h1>Monte Carlo Robustness Report</h1>',
        f'<p class="score">Robustness Score: {robustness["overall_score"]}/100</p>',
        '<h2>Score Components</h2>',
        '<ul>',
        f'<li>Sharpe Stability: {robustness["components"]["sharpe_stability"]}%</li>',
        f'<li>Drawdown Stability: {robustness["components"]["dd_stability"]}%</li>',
        f'<li>Win Rate Stability: {robustness["components"]["win_rate_stability"]}%</li>',
        f'<li>Positive Expectancy: {robustness["components"]["positive_expectancy_pct"]}%</li>',
        '</ul>',
        '<h2>Original vs Monte Carlo</h2>',
        '<table>',
        '<tr><th>Metric</th><th>Original</th><th>MC 5th</th><th>MC Median</th><th>MC 95th</th></tr>',
        f'<tr><td>Sharpe</td><td>{robustness["original"]["sharpe"]:.2f}</td><td>{robustness["mc_percentiles"]["sharpe_5th"]:.2f}</td><td>{robustness["mc_percentiles"]["sharpe_50th"]:.2f}</td><td>{robustness["mc_percentiles"]["sharpe_95th"]:.2f}</td></tr>',
        f'<tr><td>Max Drawdown</td><td>${robustness["original"]["max_drawdown"]:,.0f}</td><td>${robustness["mc_percentiles"]["max_dd_5th"]:,.0f}</td><td>${robustness["mc_percentiles"]["max_dd_50th"]:,.0f}</td><td>${robustness["mc_percentiles"]["max_dd_95th"]:,.0f}</td></tr>',
        f'<tr><td>Win Rate</td><td>{robustness["original"]["win_rate"]:.1%}</td><td>{robustness["mc_percentiles"]["win_rate_5th"]:.1%}</td><td>{robustness["mc_percentiles"]["win_rate_50th"]:.1%}</td><td>{robustness["mc_percentiles"]["win_rate_95th"]:.1%}</td></tr>',
        '</table>',
        f'<p><em>Based on {robustness["n_iterations"]} Monte Carlo iterations</em></p>',
        '</body></html>',
    ]
    (outdir / 'monte_carlo_report.html').write_text('\n'.join(html), encoding='utf-8')


if __name__ == '__main__':
    main()
