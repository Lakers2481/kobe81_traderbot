#!/usr/bin/env python3
"""
Synthetic Options Backtest Runner.

Runs backtests using synthetic (Black-Scholes) options pricing.
No real options data required - uses realized volatility from underlying.

Supports:
- Long calls/puts (limited risk)
- Short puts (cash-secured)
- Covered calls (with underlying)
- Delta-targeted strikes
- 2% risk per trade enforcement

Usage:
    # Windows:
    python scripts\\run_backtest_options_synth.py ^
        --dataset-id stooq_1d_2015_2025_abc123 ^
        --signals signals.csv ^
        --equity 100000 ^
        --risk-pct 0.02

    # Linux/macOS:
    python scripts/run_backtest_options_synth.py \
        --dataset-id stooq_1d_2015_2025_abc123 \
        --signals signals.csv \
        --equity 100000 \
        --risk-pct 0.02

    # Generate sample signals (for testing):
    python scripts/run_backtest_options_synth.py --generate-sample-signals

    # Quick test with synthetic data:
    python scripts/run_backtest_options_synth.py --demo

Signal CSV format:
    timestamp,symbol,side,option_type,delta,dte,reason
    2023-01-15,AAPL,long,CALL,0.30,30,RSI oversold bounce
    2023-01-20,MSFT,long,PUT,0.30,30,Breaking support
    2023-02-01,GOOGL,short,PUT,0.25,45,Cash-secured wheel
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_signals(
    symbols: list = None,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31',
    signals_per_month: int = 2,
) -> pd.DataFrame:
    """Generate sample signals for testing."""
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    signals = []
    current = start

    while current <= end:
        for _ in range(signals_per_month):
            symbol = np.random.choice(symbols)
            side = np.random.choice(['long', 'long', 'long', 'short'])  # Bias toward longs
            option_type = np.random.choice(['CALL', 'PUT'])
            delta = np.random.choice([0.25, 0.30, 0.35, 0.40, 0.50])
            dte = np.random.choice([21, 30, 45])

            # Random day within month
            day_offset = np.random.randint(0, 28)
            sig_date = current + timedelta(days=day_offset)

            if sig_date > end:
                break

            signals.append({
                'timestamp': sig_date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'side': side,
                'option_type': option_type,
                'delta': delta,
                'dte': dte,
                'reason': f'Sample signal ({option_type} {side})',
            })

        current += pd.DateOffset(months=1)

    return pd.DataFrame(signals)


def generate_demo_ohlcv(
    symbols: list = None,
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-31',
) -> pd.DataFrame:
    """Generate demo OHLCV data for testing."""
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    dates = pd.date_range(start_date, end_date, freq='B')  # Business days

    all_data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)

        # Random starting price
        base_price = np.random.uniform(100, 500)

        prices = [base_price]
        for _ in range(len(dates) - 1):
            # Random walk with drift
            change = np.random.normal(0.0005, 0.02)  # Small drift, 2% daily vol
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        for i, date in enumerate(dates):
            close = prices[i]
            daily_range = close * np.random.uniform(0.01, 0.03)
            high = close + daily_range * np.random.uniform(0.3, 0.7)
            low = close - daily_range * np.random.uniform(0.3, 0.7)
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
            volume = int(np.random.uniform(1e6, 1e7))

            all_data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume,
            })

    return pd.DataFrame(all_data)


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic options backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with frozen dataset and signals file
    python scripts/run_backtest_options_synth.py \\
        --dataset-id stooq_1d_2015_2025_abc123 \\
        --signals my_signals.csv

    # Run demo with synthetic data
    python scripts/run_backtest_options_synth.py --demo

    # Generate sample signals CSV
    python scripts/run_backtest_options_synth.py --generate-sample-signals

Signal CSV format:
    timestamp,symbol,side,option_type,delta,dte,reason
    2023-01-15,AAPL,long,CALL,0.30,30,RSI oversold
    2023-01-20,MSFT,long,PUT,0.30,30,Breaking support
"""
    )

    parser.add_argument(
        '--dataset-id',
        help='Frozen dataset ID from data lake',
    )
    parser.add_argument(
        '--signals',
        help='Path to signals CSV file',
    )
    parser.add_argument(
        '--ohlcv',
        help='Path to OHLCV CSV file (alternative to --dataset-id)',
    )
    parser.add_argument(
        '--equity',
        type=float,
        default=100_000,
        help='Initial equity (default: 100000)',
    )
    parser.add_argument(
        '--risk-pct',
        type=float,
        default=0.02,
        help='Risk per trade (default: 0.02 = 2%%)',
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=0.65,
        help='Commission per contract (default: 0.65)',
    )
    parser.add_argument(
        '--spread-pct',
        type=float,
        default=0.02,
        help='Bid-ask spread percentage (default: 0.02 = 2%%)',
    )
    parser.add_argument(
        '--vol-lookback',
        type=int,
        default=20,
        help='Volatility lookback days (default: 20)',
    )
    parser.add_argument(
        '--vol-method',
        choices=['close_to_close', 'yang_zhang', 'parkinson', 'garman_klass'],
        default='yang_zhang',
        help='Volatility calculation method (default: yang_zhang)',
    )
    parser.add_argument(
        '--default-dte',
        type=int,
        default=30,
        help='Default days to expiry (default: 30)',
    )
    parser.add_argument(
        '--output',
        default='backtest_outputs/options_synth',
        help='Output directory (default: backtest_outputs/options_synth)',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with synthetic data',
    )
    parser.add_argument(
        '--generate-sample-signals',
        action='store_true',
        help='Generate sample signals CSV and exit',
    )
    parser.add_argument(
        '--lake-dir',
        default='data/lake',
        help='Data lake directory (default: data/lake)',
    )
    parser.add_argument(
        '--manifest-dir',
        default='data/manifests',
        help='Manifest directory (default: data/manifests)',
    )

    args = parser.parse_args()

    # Generate sample signals if requested
    if args.generate_sample_signals:
        signals_df = generate_sample_signals()
        output_path = Path('sample_options_signals.csv')
        signals_df.to_csv(output_path, index=False)
        print(f"Generated {len(signals_df)} sample signals to: {output_path}")
        print("\nSample signals:")
        print(signals_df.head(10).to_string())
        sys.exit(0)

    # Demo mode
    if args.demo:
        print("=" * 60)
        print("SYNTHETIC OPTIONS BACKTEST - DEMO MODE")
        print("=" * 60)

        print("\nGenerating demo OHLCV data...")
        ohlcv_df = generate_demo_ohlcv()
        print(f"  Generated {len(ohlcv_df):,} rows for {ohlcv_df['symbol'].nunique()} symbols")

        print("\nGenerating demo signals...")
        signals_df = generate_sample_signals(
            symbols=ohlcv_df['symbol'].unique().tolist(),
            start_date='2023-01-01',
            end_date='2024-06-30',
        )
        print(f"  Generated {len(signals_df)} signals")

    else:
        # Load OHLCV data
        if args.dataset_id:
            print(f"\nLoading frozen dataset: {args.dataset_id}")
            from data.lake import LakeReader
            reader = LakeReader(
                lake_dir=Path(args.lake_dir),
                manifest_dir=Path(args.manifest_dir),
            )
            ohlcv_df = reader.load_dataset(args.dataset_id)
            print(f"  Loaded {len(ohlcv_df):,} rows")

        elif args.ohlcv:
            print(f"\nLoading OHLCV from: {args.ohlcv}")
            ohlcv_df = pd.read_csv(args.ohlcv)
            ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'])
            print(f"  Loaded {len(ohlcv_df):,} rows")

        else:
            print("ERROR: Must specify --dataset-id, --ohlcv, or --demo")
            sys.exit(1)

        # Load signals
        if args.signals:
            print(f"\nLoading signals from: {args.signals}")
            signals_df = pd.read_csv(args.signals)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            print(f"  Loaded {len(signals_df)} signals")
        else:
            print("ERROR: Must specify --signals or --demo")
            sys.exit(1)

    # Import backtester
    from options import run_options_backtest

    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    print(f"Initial equity:  ${args.equity:,.0f}")
    print(f"Risk per trade:  {args.risk_pct * 100:.1f}%")
    print(f"Commission:      ${args.commission}/contract")
    print(f"Spread:          {args.spread_pct * 100:.1f}%")
    print(f"Vol method:      {args.vol_method}")
    print(f"Default DTE:     {args.default_dte} days")
    print("=" * 60)

    # Run backtest
    result = run_options_backtest(
        ohlcv_df=ohlcv_df,
        signals_df=signals_df,
        initial_equity=args.equity,
        risk_pct=args.risk_pct,
        commission_per_contract=args.commission,
        spread_pct=args.spread_pct,
        vol_lookback=args.vol_lookback,
        vol_method=args.vol_method,
        default_dte=args.default_dte,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total trades:      {result.total_trades}")
    print(f"Winning trades:    {result.winning_trades}")
    print(f"Losing trades:     {result.losing_trades}")
    print(f"Win rate:          {result.win_rate * 100:.1f}%")
    print(f"Total P&L:         ${result.total_pnl:,.2f}")
    print(f"Avg trade P&L:     ${result.avg_trade_pnl:,.2f}")
    print(f"Avg days held:     {result.avg_days_held:.1f}")
    print(f"Max drawdown:      {result.max_drawdown * 100:.1f}%")
    print(f"Sharpe ratio:      {result.sharpe_ratio:.2f}")
    print("-" * 60)
    print("Transaction Costs:")
    print(f"  Total costs:     ${result.total_transaction_costs:,.2f}")
    print(f"  Premium paid:    ${result.total_premium_paid:,.2f}")
    print(f"  Premium recv:    ${result.total_premium_collected:,.2f}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary JSON
    summary_path = output_dir / f'summary_{timestamp_str}.json'
    with open(summary_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # Save trade list
    if result.trade_list:
        trades_df = pd.DataFrame(result.trade_list)
        trades_path = output_dir / f'trades_{timestamp_str}.csv'
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade list saved to: {trades_path}")

    # Save equity curve
    if result.equity_curve:
        equity_df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'equity'])
        equity_path = output_dir / f'equity_curve_{timestamp_str}.csv'
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve saved to: {equity_path}")

    print("\nBacktest complete!")


if __name__ == '__main__':
    main()
