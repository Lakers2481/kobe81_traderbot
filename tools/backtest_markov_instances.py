"""
BACKTEST MARKOV INSTANCES
=========================

Simple backtest: Buy next day open, sell next day close.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.providers.yfinance_eod import YFinanceEODProvider


def backtest_instances(instances_csv: Path) -> pd.DataFrame:
    """Backtest all Markov instances."""
    print(f"\n{'='*80}")
    print("BACKTEST MARKOV INSTANCES")
    print(f"{'='*80}\n")

    # Load instances
    instances_df = pd.read_csv(instances_csv)
    instances_df['date'] = pd.to_datetime(instances_df['date'])

    print(f"Loaded {len(instances_df)} instances from {instances_csv.name}")
    print(f"Symbols: {instances_df['symbol'].nunique()}")
    print(f"Date range: {instances_df['date'].min()} to {instances_df['date'].max()}\n")

    provider = YFinanceEODProvider(warn_unofficial=False)
    all_trades = []

    symbols = instances_df['symbol'].unique()

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing {symbol}...", end=' ')

        # Get all instances for this symbol
        symbol_instances = instances_df[instances_df['symbol'] == symbol]

        try:
            # Fetch data
            df = provider.fetch_symbol(symbol, '2015-01-01', '2024-12-31')
            if df is None or len(df) == 0:
                print("SKIP (no data)")
                continue

            df = df.set_index('timestamp')
            print(f"OK ({len(symbol_instances)} instances)")

            # Process each instance
            for _, row in symbol_instances.iterrows():
                pattern_date = row['date']

                if pattern_date not in df.index:
                    continue

                pattern_loc = df.index.get_loc(pattern_date)

                # Entry: Next day open (day after pattern)
                if pattern_loc + 1 >= len(df):
                    continue

                entry_date = df.index[pattern_loc + 1]
                entry_price = df['open'].iloc[pattern_loc + 1]

                # Exit: Same day close
                exit_date = entry_date
                exit_price = df['close'].iloc[pattern_loc + 1]

                # Calculate P&L
                pnl = exit_price - entry_price
                pnl_pct = pnl / entry_price

                all_trades.append({
                    'symbol': symbol,
                    'pattern_date': pattern_date,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'is_win': 1 if pnl > 0 else 0,
                })

        except Exception as e:
            print(f"FAIL ({e})")

    # Convert to DataFrame
    trades_df = pd.DataFrame(all_trades)

    # Calculate stats
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        total_wins = trades_df['is_win'].sum()
        total_losses = total_trades - total_wins
        win_rate = total_wins / total_trades

        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if total_wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if total_losses > 0 else 0

        total_pnl = trades_df['pnl'].sum()

        print(f"\n{'-'*80}")
        print("BACKTEST RESULTS")
        print(f"{'-'*80}")
        print(f"Total trades: {total_trades}")
        print(f"Wins: {total_wins} ({win_rate:.1%})")
        print(f"Losses: {total_losses} ({1-win_rate:.1%})")
        print(f"Profit factor: {profit_factor:.2f}")
        print(f"Gross profit: ${gross_profit:.2f}")
        print(f"Gross loss: ${gross_loss:.2f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Avg win: ${avg_win:.2f}")
        print(f"Avg loss: ${avg_loss:.2f}")
        print(f"{'-'*80}\n")

        # Save trades
        output_path = project_root / "data" / "verification" / "fast_backtest_trades.csv"
        trades_df.to_csv(output_path, index=False)
        print(f"[SAVED] Trades saved to {output_path}\n")

        # Compare to claim
        print(f"{'='*80}")
        print("CLAIM VERIFICATION")
        print(f"{'='*80}")
        print(f"Claimed WR: 59.9%")
        print(f"Actual WR (50 symbols): {win_rate:.1%}")
        wr_diff = abs(0.599 - win_rate)
        print(f"Difference: {wr_diff:.1%}")

        print(f"\nClaimed PF: 1.24")
        print(f"Actual PF (50 symbols): {profit_factor:.2f}")
        pf_diff = abs(1.24 - profit_factor)
        print(f"Difference: {pf_diff:.2f}")

        if wr_diff < 0.05 and pf_diff < 0.20:
            print("\nResult: VERIFIED")
        elif wr_diff < 0.10 and pf_diff < 0.40:
            print("\nResult: PARTIALLY VERIFIED")
        else:
            print("\nResult: NOT VERIFIED")
        print(f"{'='*80}\n")

    return trades_df


def main():
    """Run backtest."""
    instances_csv = project_root / "data" / "verification" / "fast_markov_instances.csv"

    if not instances_csv.exists():
        print(f"Error: {instances_csv} not found")
        print("Run verify_data_math_fast.py first")
        sys.exit(1)

    backtest_instances(instances_csv)


if __name__ == "__main__":
    main()
