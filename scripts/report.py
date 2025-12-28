#!/usr/bin/env python3
"""
Report Generator for Kobe Trading System.

Generates HTML, CSV, or JSON performance reports including:
- Trade list with P&L
- Equity curve data
- Strategy breakdown
- Performance summary

Usage:
    python report.py --format html --output report.html
    python report.py --format csv --output trades.csv --dotenv
    python report.py --format json --output report.json --strategy ibs_rsi
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Individual trade record with P&L."""
    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    pnl: Optional[float] = None
    cumulative_pnl: Optional[float] = None
    source: Optional[str] = None


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade: float


@dataclass
class ReportData:
    """Container for all report data."""
    generated_at: str
    period_start: str
    period_end: str
    trades: List[TradeRecord]
    equity_curve: List[Dict[str, Any]]
    strategy_breakdown: List[StrategyStats]
    summary: Dict[str, Any]


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).parent.parent / '.env'
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val


def load_trades_from_csv(path: Path) -> pd.DataFrame:
    """Load trades from CSV file."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.lower().str.strip()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def load_trades_from_jsonl(path: Path) -> pd.DataFrame:
    """Load trades from JSONL file."""
    if not path.exists():
        return pd.DataFrame()

    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.columns = df.columns.str.lower().str.strip()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def load_equity_curve(path: Path) -> pd.DataFrame:
    """Load equity curve from CSV file."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df.columns = df.columns.str.lower().str.strip()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    return df


def load_summary_json(path: Path) -> Dict[str, Any]:
    """Load summary JSON file."""
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def discover_files(base_dir: Path, strategy: Optional[str] = None) -> Dict[str, List[Path]]:
    """Discover all relevant files in directory structure."""
    result = {
        'trades': [],
        'equity': [],
        'summary': [],
    }

    search_dir = base_dir / strategy if strategy else base_dir

    if search_dir.exists():
        result['trades'].extend(search_dir.glob('**/trade_list.csv'))
        result['equity'].extend(search_dir.glob('**/equity_curve.csv'))
        result['summary'].extend(search_dir.glob('**/summary.json'))

    # Also check state directory
    state_dir = base_dir.parent / 'state'
    if state_dir.exists():
        result['trades'].extend(state_dir.glob('trades.jsonl'))

    return result


def compute_trade_pnl(trades_df: pd.DataFrame) -> List[TradeRecord]:
    """Compute P&L for each trade using FIFO matching."""
    if trades_df.empty:
        return []

    required_cols = {'symbol', 'side', 'qty', 'price'}
    if not required_cols.issubset(set(trades_df.columns)):
        return []

    # FIFO matching per symbol
    buys: Dict[str, deque] = defaultdict(deque)
    records: List[TradeRecord] = []
    cumulative_pnl = 0.0

    for _, row in trades_df.sort_values('timestamp').iterrows():
        symbol = str(row['symbol'])
        side = str(row['side']).upper()
        qty = int(row['qty'])
        price = float(row['price'])
        ts = str(row['timestamp'])
        source = str(row.get('source_file', ''))

        if side == 'BUY':
            buys[symbol].append((qty, price))
            records.append(TradeRecord(
                timestamp=ts,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                pnl=None,  # P&L computed on SELL
                cumulative_pnl=cumulative_pnl,
                source=source,
            ))
        elif side == 'SELL':
            remaining = qty
            trade_pnl = 0.0

            while remaining > 0 and buys[symbol]:
                buy_qty, buy_price = buys[symbol][0]
                matched = min(remaining, buy_qty)
                trade_pnl += (price - buy_price) * matched
                remaining -= matched
                buy_qty -= matched

                if buy_qty == 0:
                    buys[symbol].popleft()
                else:
                    buys[symbol][0] = (buy_qty, buy_price)

            cumulative_pnl += trade_pnl
            records.append(TradeRecord(
                timestamp=ts,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                pnl=trade_pnl,
                cumulative_pnl=cumulative_pnl,
                source=source,
            ))

    return records


def compute_strategy_stats(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    strategy_name: str
) -> StrategyStats:
    """Compute statistics for a strategy."""
    stats = StrategyStats(
        name=strategy_name,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        net_profit=0.0,
        profit_factor=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        avg_trade=0.0,
    )

    if trades_df.empty:
        return stats

    # Compute P&L per trade
    trade_records = compute_trade_pnl(trades_df)
    pnls = [t.pnl for t in trade_records if t.pnl is not None]

    if not pnls:
        return stats

    pnl_array = np.array(pnls)
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]

    stats.total_trades = len(pnls)
    stats.winning_trades = len(wins)
    stats.losing_trades = len(losses)
    stats.gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    stats.gross_loss = float(losses.sum()) if len(losses) > 0 else 0.0
    stats.net_profit = float(pnl_array.sum())

    if stats.total_trades > 0:
        stats.win_rate = stats.winning_trades / stats.total_trades
        stats.avg_trade = stats.net_profit / stats.total_trades

    if stats.gross_loss < 0:
        stats.profit_factor = abs(stats.gross_profit / stats.gross_loss)
    elif stats.gross_profit > 0:
        stats.profit_factor = float('inf')

    # Equity metrics
    if not equity_df.empty and 'equity' in equity_df.columns:
        equity = equity_df['equity']
        returns = equity.pct_change().dropna()

        if len(returns) > 1:
            std = returns.std()
            if std > 0:
                stats.sharpe_ratio = float(returns.mean() / std * np.sqrt(252))

            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax
            stats.max_drawdown = float(drawdown.min())

    return stats


def generate_report_data(
    base_dir: Path,
    strategy: Optional[str] = None
) -> ReportData:
    """Generate comprehensive report data."""
    files = discover_files(base_dir, strategy)

    # Load all trades
    all_trades = []
    strategy_trades: Dict[str, pd.DataFrame] = defaultdict(list)

    for tf in files['trades']:
        if tf.suffix == '.jsonl':
            df = load_trades_from_jsonl(tf)
        else:
            df = load_trades_from_csv(tf)

        if not df.empty:
            df['source_file'] = str(tf)

            # Determine strategy from path
            parts = tf.parts
            for i, p in enumerate(parts):
                if p in ('ibs_rsi', 'TURTLE_SOUP', 'and', 'connors_ibs_rsi'):
                    df['strategy'] = p
                    break
            else:
                df['strategy'] = 'unknown'

            all_trades.append(df)

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        trades_df = pd.DataFrame()

    # Load all equity curves
    all_equity = []
    for ef in files['equity']:
        df = load_equity_curve(ef)
        if not df.empty:
            all_equity.append(df)

    if all_equity:
        # Merge equity curves
        merged = pd.concat([df[['equity']] for df in all_equity], axis=1)
        equity_df = pd.DataFrame({'equity': merged.mean(axis=1)})
    else:
        equity_df = pd.DataFrame()

    # Load summaries
    summaries = []
    for sf in files['summary']:
        summary = load_summary_json(sf)
        if summary:
            summary['source'] = str(sf)
            summaries.append(summary)

    # Compute trade records with P&L
    trade_records = compute_trade_pnl(trades_df) if not trades_df.empty else []

    # Compute strategy breakdown
    strategy_breakdown = []
    if not trades_df.empty and 'strategy' in trades_df.columns:
        for strat_name in trades_df['strategy'].unique():
            strat_trades = trades_df[trades_df['strategy'] == strat_name]
            stats = compute_strategy_stats(strat_trades, equity_df, strat_name)
            strategy_breakdown.append(stats)
    elif not trades_df.empty:
        stats = compute_strategy_stats(trades_df, equity_df, strategy or 'combined')
        strategy_breakdown.append(stats)

    # Convert equity curve to list of dicts
    equity_curve = []
    if not equity_df.empty:
        for ts, row in equity_df.iterrows():
            equity_curve.append({
                'timestamp': str(ts),
                'equity': float(row['equity']),
                'returns': float(row.get('returns', 0.0)) if 'returns' in row else None,
            })

    # Compute summary
    summary = {
        'total_trades': len(trade_records),
        'strategies': len(strategy_breakdown),
    }

    if trade_records:
        pnls = [t.pnl for t in trade_records if t.pnl is not None]
        summary['total_pnl'] = sum(pnls)
        summary['winning_trades'] = len([p for p in pnls if p > 0])
        summary['losing_trades'] = len([p for p in pnls if p < 0])
        if pnls:
            summary['win_rate'] = summary['winning_trades'] / len(pnls)

    if equity_curve:
        summary['initial_equity'] = equity_curve[0]['equity']
        summary['final_equity'] = equity_curve[-1]['equity']
        summary['total_return'] = (summary['final_equity'] / summary['initial_equity']) - 1

    # Determine period
    period_start = ""
    period_end = ""
    if trade_records:
        timestamps = [t.timestamp for t in trade_records]
        period_start = min(timestamps)
        period_end = max(timestamps)

    return ReportData(
        generated_at=datetime.now().isoformat(),
        period_start=period_start,
        period_end=period_end,
        trades=trade_records,
        equity_curve=equity_curve,
        strategy_breakdown=strategy_breakdown,
        summary=summary,
    )


def generate_html_report(data: ReportData) -> str:
    """Generate HTML report."""
    html_parts = []

    # Header
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kobe Trading System - Performance Report</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #1a1a1a; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .meta { color: #666; font-size: 0.9em; margin-bottom: 20px; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card .label { color: #666; font-size: 0.85em; text-transform: uppercase; }
        .summary-card .value { font-size: 1.8em; font-weight: bold; color: #1a1a1a; margin-top: 5px; }
        .summary-card .value.positive { color: #28a745; }
        .summary-card .value.negative { color: #dc3545; }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #333; }
        tr:hover { background: #f8f9fa; }
        .pnl-positive { color: #28a745; font-weight: 500; }
        .pnl-negative { color: #dc3545; font-weight: 500; }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .equity-chart {
            width: 100%;
            height: 300px;
            background: linear-gradient(to bottom, #f8f9fa, white);
            border: 1px solid #eee;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
""")

    # Title and meta
    html_parts.append(f"""
        <h1>Performance Report</h1>
        <div class="meta">
            Generated: {data.generated_at}<br>
            Period: {data.period_start} to {data.period_end}
        </div>
""")

    # Summary cards
    html_parts.append('<div class="summary-grid">')

    summary = data.summary
    total_pnl = summary.get('total_pnl', 0)
    pnl_class = 'positive' if total_pnl >= 0 else 'negative'
    html_parts.append(f"""
        <div class="summary-card">
            <div class="label">Total P&L</div>
            <div class="value {pnl_class}">${total_pnl:,.2f}</div>
        </div>
""")

    html_parts.append(f"""
        <div class="summary-card">
            <div class="label">Total Trades</div>
            <div class="value">{summary.get('total_trades', 0)}</div>
        </div>
""")

    win_rate = summary.get('win_rate', 0) * 100
    html_parts.append(f"""
        <div class="summary-card">
            <div class="label">Win Rate</div>
            <div class="value">{win_rate:.1f}%</div>
        </div>
""")

    total_return = summary.get('total_return', 0) * 100
    ret_class = 'positive' if total_return >= 0 else 'negative'
    html_parts.append(f"""
        <div class="summary-card">
            <div class="label">Total Return</div>
            <div class="value {ret_class}">{total_return:.2f}%</div>
        </div>
""")

    html_parts.append('</div>')

    # Strategy breakdown
    if data.strategy_breakdown:
        html_parts.append('<h2>Strategy Breakdown</h2>')
        html_parts.append('<table>')
        html_parts.append("""
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Net P&L</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                </tr>
            </thead>
            <tbody>
""")

        for stats in data.strategy_breakdown:
            pnl_class = 'pnl-positive' if stats.net_profit >= 0 else 'pnl-negative'
            pf_display = f"{stats.profit_factor:.2f}" if stats.profit_factor < float('inf') else "Inf"
            html_parts.append(f"""
                <tr>
                    <td><strong>{stats.name}</strong></td>
                    <td>{stats.total_trades}</td>
                    <td>{stats.win_rate:.1%}</td>
                    <td>{pf_display}</td>
                    <td class="{pnl_class}">${stats.net_profit:,.2f}</td>
                    <td>{stats.sharpe_ratio:.2f}</td>
                    <td>{stats.max_drawdown:.1%}</td>
                </tr>
""")

        html_parts.append('</tbody></table>')

    # Equity curve placeholder
    if data.equity_curve:
        html_parts.append('<h2>Equity Curve</h2>')
        html_parts.append('<div class="chart-container">')
        html_parts.append('<div class="equity-chart">')
        html_parts.append(f'Equity curve data available ({len(data.equity_curve)} data points)')
        html_parts.append('</div>')

        # Show equity data as mini table
        html_parts.append('<table style="margin-top:20px">')
        html_parts.append('<thead><tr><th>Date</th><th>Equity</th><th>Daily Return</th></tr></thead><tbody>')

        # Show first and last 5 rows
        display_rows = data.equity_curve[:5] + [{'timestamp': '...', 'equity': '...', 'returns': '...'}] + data.equity_curve[-5:]
        for row in display_rows:
            if row['timestamp'] == '...':
                html_parts.append('<tr><td>...</td><td>...</td><td>...</td></tr>')
            else:
                ret_val = row.get('returns')
                ret_str = f"{ret_val:.4%}" if ret_val is not None else "N/A"
                html_parts.append(f'<tr><td>{row["timestamp"]}</td><td>${row["equity"]:,.2f}</td><td>{ret_str}</td></tr>')

        html_parts.append('</tbody></table>')
        html_parts.append('</div>')

    # Trade list
    if data.trades:
        html_parts.append('<h2>Trade List</h2>')
        html_parts.append('<table>')
        html_parts.append("""
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Price</th>
                    <th>P&L</th>
                    <th>Cumulative P&L</th>
                </tr>
            </thead>
            <tbody>
""")

        # Show max 100 trades
        display_trades = data.trades[:100]
        for trade in display_trades:
            pnl_str = ""
            pnl_class = ""
            if trade.pnl is not None:
                pnl_class = 'pnl-positive' if trade.pnl >= 0 else 'pnl-negative'
                pnl_str = f"${trade.pnl:,.2f}"

            cum_pnl_str = f"${trade.cumulative_pnl:,.2f}" if trade.cumulative_pnl is not None else ""

            html_parts.append(f"""
                <tr>
                    <td>{trade.timestamp}</td>
                    <td><strong>{trade.symbol}</strong></td>
                    <td>{trade.side}</td>
                    <td>{trade.qty}</td>
                    <td>${trade.price:,.2f}</td>
                    <td class="{pnl_class}">{pnl_str}</td>
                    <td>{cum_pnl_str}</td>
                </tr>
""")

        if len(data.trades) > 100:
            html_parts.append(f'<tr><td colspan="7" style="text-align:center;color:#666">... and {len(data.trades) - 100} more trades</td></tr>')

        html_parts.append('</tbody></table>')

    # Footer
    html_parts.append(f"""
        <footer>
            Report generated by Kobe Trading System<br>
            Total trades analyzed: {len(data.trades)} | Strategies: {len(data.strategy_breakdown)}
        </footer>
    </div>
</body>
</html>
""")

    return ''.join(html_parts)


def generate_csv_report(data: ReportData, output_type: str = 'trades') -> str:
    """Generate CSV report."""
    output = []

    if output_type == 'trades':
        writer = csv.writer(output := [])
        output = []

        # Use StringIO for CSV
        import io
        string_io = io.StringIO()
        writer = csv.writer(string_io)

        writer.writerow(['timestamp', 'symbol', 'side', 'qty', 'price', 'pnl', 'cumulative_pnl', 'source'])

        for trade in data.trades:
            writer.writerow([
                trade.timestamp,
                trade.symbol,
                trade.side,
                trade.qty,
                trade.price,
                trade.pnl if trade.pnl is not None else '',
                trade.cumulative_pnl if trade.cumulative_pnl is not None else '',
                trade.source or '',
            ])

        return string_io.getvalue()

    elif output_type == 'equity':
        import io
        string_io = io.StringIO()
        writer = csv.writer(string_io)

        writer.writerow(['timestamp', 'equity', 'returns'])

        for row in data.equity_curve:
            writer.writerow([
                row['timestamp'],
                row['equity'],
                row.get('returns', ''),
            ])

        return string_io.getvalue()

    elif output_type == 'summary':
        import io
        string_io = io.StringIO()
        writer = csv.writer(string_io)

        writer.writerow(['strategy', 'total_trades', 'winning_trades', 'losing_trades',
                        'win_rate', 'gross_profit', 'gross_loss', 'net_profit',
                        'profit_factor', 'sharpe_ratio', 'max_drawdown', 'avg_trade'])

        for stats in data.strategy_breakdown:
            writer.writerow([
                stats.name,
                stats.total_trades,
                stats.winning_trades,
                stats.losing_trades,
                stats.win_rate,
                stats.gross_profit,
                stats.gross_loss,
                stats.net_profit,
                stats.profit_factor if stats.profit_factor < float('inf') else 'Inf',
                stats.sharpe_ratio,
                stats.max_drawdown,
                stats.avg_trade,
            ])

        return string_io.getvalue()

    return ""


def generate_json_report(data: ReportData) -> str:
    """Generate JSON report."""
    output = {
        'generated_at': data.generated_at,
        'period': {
            'start': data.period_start,
            'end': data.period_end,
        },
        'summary': data.summary,
        'strategy_breakdown': [asdict(s) for s in data.strategy_breakdown],
        'trades': [asdict(t) for t in data.trades],
        'equity_curve': data.equity_curve,
    }

    return json.dumps(output, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='Generate performance reports for Kobe Trading System'
    )
    parser.add_argument(
        '--wfdir',
        type=str,
        default='wf_outputs',
        help='Directory containing WF outputs (default: wf_outputs)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Filter by strategy name (e.g., ibs_rsi, TURTLE_SOUP)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'csv', 'json'],
        default='html',
        help='Output format (default: html)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file path (default: stdout for csv/json, report.html for html)'
    )
    parser.add_argument(
        '--csv-type',
        type=str,
        choices=['trades', 'equity', 'summary'],
        default='trades',
        help='Type of CSV to generate (default: trades)'
    )
    parser.add_argument(
        '--dotenv',
        action='store_true',
        help='Load environment variables from .env file'
    )

    args = parser.parse_args()

    if args.dotenv:
        load_env()

    # Determine base directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    wf_dir = Path(args.wfdir)
    if not wf_dir.is_absolute():
        wf_dir = project_dir / wf_dir

    if not wf_dir.exists():
        print(f"Error: Directory not found: {wf_dir}")
        sys.exit(1)

    # Generate report data
    print(f"Generating report from {wf_dir}...", file=sys.stderr)
    data = generate_report_data(wf_dir, args.strategy)

    # Generate output
    if args.format == 'html':
        output = generate_html_report(data)
        default_output = 'report.html'
    elif args.format == 'csv':
        output = generate_csv_report(data, args.csv_type)
        default_output = f'{args.csv_type}.csv'
    else:  # json
        output = generate_json_report(data)
        default_output = 'report.json'

    # Write output
    output_path = args.output or default_output

    if args.output:
        output_file = Path(args.output)
        if not output_file.is_absolute():
            output_file = Path.cwd() / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output, encoding='utf-8')
        print(f"Report written to: {output_file}", file=sys.stderr)
    else:
        if args.format == 'html':
            output_file = Path(output_path)
            output_file.write_text(output, encoding='utf-8')
            print(f"Report written to: {output_file}", file=sys.stderr)
        else:
            print(output)


if __name__ == '__main__':
    main()

