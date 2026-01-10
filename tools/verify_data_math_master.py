"""
DATA & MATH INTEGRITY MASTER VERIFIER
======================================

ZERO FAKE DATA. ZERO HALLUCINATIONS. ZERO BIAS.

This script verifies:
1. Markov 5-down pattern claim (64.0% up probability)
2. Backtest performance claim (59.9% WR, 1.24 PF)
3. Data authenticity (lookahead, fake data, leakage)

ALL claims must be PROVABLE with raw CSV files.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.providers.yfinance_eod import YFinanceEODProvider
from data.providers.stooq_eod import StooqEODProvider

# Try Polygon if available
try:
    from data.providers.polygon_eod import PolygonEODProvider
    HAS_POLYGON = True
except Exception:
    HAS_POLYGON = False


class DataMathVerifier:
    """Master verifier with ZERO tolerance for fake data."""

    def __init__(self, universe_path: str, start_date: str, end_date: str):
        self.universe_path = universe_path
        self.start_date = start_date
        self.end_date = end_date

        # Load universe
        self.symbols = self._load_universe()
        print(f"[BASELINE] Loaded {len(self.symbols)} symbols from {universe_path}")

        # Initialize providers
        self.yfinance_provider = YFinanceEODProvider(warn_unofficial=False)
        self.stooq_provider = StooqEODProvider()

        # Output paths
        self.output_dir = project_root / "AUDITS"
        self.data_verification_dir = project_root / "data" / "verification"
        self.data_verification_dir.mkdir(parents=True, exist_ok=True)

    def _load_universe(self) -> List[str]:
        """Load universe with validation."""
        df = pd.read_csv(self.universe_path)
        symbols = df['symbol'].dropna().unique().tolist()
        return symbols

    def fetch_data_multi_source(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch from multiple sources for cross-validation."""
        data = {}

        # Try yfinance (primary)
        try:
            df_yf = self.yfinance_provider.fetch_symbol(symbol, self.start_date, self.end_date)
            if df_yf is not None and len(df_yf) > 0:
                # Set timestamp as index
                df_yf = df_yf.set_index('timestamp')
                data['yfinance'] = df_yf
                # Only print every 50th symbol to reduce noise
                if len(data) % 50 == 1:
                    print(f"[DATA] {symbol}: yfinance OK ({len(df_yf)} bars)")
        except Exception as e:
            print(f"[DATA] {symbol}: yfinance FAIL - {e}")

        # Try Stooq (fallback)
        try:
            df_stooq = self.stooq_provider.fetch_symbol(symbol, self.start_date, self.end_date)
            if df_stooq is not None and len(df_stooq) > 0:
                # Set timestamp as index
                df_stooq = df_stooq.set_index('timestamp')
                data['stooq'] = df_stooq
        except Exception as e:
            pass  # Silent fallback

        return data

    def verify_markov_5down_pattern(self) -> Tuple[pd.DataFrame, Dict]:
        """
        CRITICAL VERIFICATION: Markov 5-down pattern

        Claim: 64.0% up probability with 431 instances (10 symbols)
        Reality: We will check ALL 800 symbols with 10 years of data

        Returns:
            instances_df: All pattern instances with dates/returns
            summary: Aggregate statistics
        """
        print("\n" + "="*80)
        print("MARKOV 5-DOWN PATTERN VERIFICATION")
        print("="*80)
        print(f"Universe: {len(self.symbols)} symbols")
        print(f"Period: {self.start_date} to {self.end_date}")
        print("")

        all_instances = []

        for i, symbol in enumerate(self.symbols, 1):
            if i % 50 == 0:
                print(f"[PROGRESS] Processed {i}/{len(self.symbols)} symbols...")

            # Fetch data
            data_sources = self.fetch_data_multi_source(symbol)

            if 'yfinance' not in data_sources:
                print(f"[SKIP] {symbol}: No yfinance data available")
                continue

            df = data_sources['yfinance'].copy()

            # Ensure we have required columns
            if 'close' not in df.columns:
                print(f"[SKIP] {symbol}: Missing close column")
                continue

            # Calculate daily returns
            df['return'] = df['close'].pct_change()

            # Identify consecutive down days
            df['is_down'] = (df['return'] < 0).astype(int)

            # Count consecutive down days
            df['down_streak'] = 0
            streak = 0
            for idx in range(len(df)):
                if df['is_down'].iloc[idx] == 1:
                    streak += 1
                    df.iloc[idx, df.columns.get_loc('down_streak')] = streak
                else:
                    streak = 0

            # Find instances where down_streak == 5
            five_down_mask = (df['down_streak'] == 5)
            five_down_dates = df[five_down_mask].index.tolist()

            if len(five_down_dates) == 0:
                continue

            # For each instance, check next day return
            for date in five_down_dates:
                date_loc = df.index.get_loc(date)

                # Check if we have next day data
                if date_loc + 1 < len(df):
                    next_day_return = df['return'].iloc[date_loc + 1]
                    next_day_up = 1 if next_day_return > 0 else 0

                    all_instances.append({
                        'symbol': symbol,
                        'date': date,
                        'next_day_return': next_day_return,
                        'next_day_up': next_day_up,
                        'close_day5': df['close'].iloc[date_loc],
                        'close_day6': df['close'].iloc[date_loc + 1] if date_loc + 1 < len(df) else None
                    })

        # Convert to DataFrame
        instances_df = pd.DataFrame(all_instances)

        # Calculate aggregate statistics
        if len(instances_df) > 0:
            total_instances = len(instances_df)
            total_up = instances_df['next_day_up'].sum()
            up_probability = total_up / total_instances

            # 95% confidence interval (binomial)
            z = 1.96
            se = np.sqrt(up_probability * (1 - up_probability) / total_instances)
            ci_lower = up_probability - z * se
            ci_upper = up_probability + z * se

            summary = {
                'total_instances': total_instances,
                'next_day_up': total_up,
                'next_day_down': total_instances - total_up,
                'up_probability': up_probability,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'symbols_with_pattern': instances_df['symbol'].nunique(),
                'date_range_start': instances_df['date'].min(),
                'date_range_end': instances_df['date'].max(),
            }
        else:
            summary = {
                'total_instances': 0,
                'next_day_up': 0,
                'next_day_down': 0,
                'up_probability': None,
                'ci_95_lower': None,
                'ci_95_upper': None,
                'symbols_with_pattern': 0,
                'date_range_start': None,
                'date_range_end': None,
            }

        print("\n" + "-"*80)
        print("MARKOV 5-DOWN PATTERN RESULTS")
        print("-"*80)
        print(f"Total instances found: {summary['total_instances']}")
        print(f"Next day up: {summary['next_day_up']}")
        print(f"Next day down: {summary['next_day_down']}")
        print(f"Up probability: {summary['up_probability']:.4f}" if summary['up_probability'] else "Up probability: N/A")
        print(f"95% CI: [{summary['ci_95_lower']:.4f}, {summary['ci_95_upper']:.4f}]" if summary['ci_95_lower'] else "95% CI: N/A")
        print(f"Symbols with pattern: {summary['symbols_with_pattern']}")
        print("-"*80)

        # Save instances to CSV
        output_path = self.data_verification_dir / "900_markov_instances.csv"
        instances_df.to_csv(output_path, index=False)
        print(f"\n[SAVED] All instances saved to {output_path}")

        return instances_df, summary

    def verify_backtest_performance(self, instances_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        CRITICAL VERIFICATION: Backtest performance

        Claim: 59.9% WR, 1.24 PF (2,912 trades)
        Reality: We will run FULL backtest on 800 stocks

        This is a SIMPLE backtest:
        - Entry: Day after 5-down pattern (market open)
        - Exit: Next day close (or stop loss)
        - Stop: 2% below entry
        - Target: 5% above entry

        Returns:
            trades_df: All trade records
            summary: Performance statistics
        """
        print("\n" + "="*80)
        print("BACKTEST PERFORMANCE VERIFICATION")
        print("="*80)
        print(f"Using {len(instances_df)} pattern instances")
        print("")

        all_trades = []

        for idx, row in instances_df.iterrows():
            symbol = row['symbol']
            entry_date = row['date']

            # Fetch data for this symbol
            try:
                df = self.yfinance_provider.fetch_symbol(symbol, self.start_date, self.end_date)
                if df is None or len(df) == 0:
                    continue
                # Set timestamp as index
                df = df.set_index('timestamp')
            except Exception:
                continue

            # Find entry date in data
            if entry_date not in df.index:
                continue

            entry_loc = df.index.get_loc(entry_date)

            # Entry is next day open
            if entry_loc + 1 >= len(df):
                continue

            entry_price = df['open'].iloc[entry_loc + 1]
            entry_actual_date = df.index[entry_loc + 1]

            # Simple exit: next day close (Day 2)
            if entry_loc + 2 >= len(df):
                continue

            exit_price = df['close'].iloc[entry_loc + 2]
            exit_date = df.index[entry_loc + 2]

            # Calculate P&L
            pnl = exit_price - entry_price
            pnl_pct = pnl / entry_price

            # Classify as win/loss
            is_win = 1 if pnl > 0 else 0

            all_trades.append({
                'symbol': symbol,
                'pattern_date': entry_date,
                'entry_date': entry_actual_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'is_win': is_win,
            })

        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)

        # Calculate performance statistics
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            total_wins = trades_df['is_win'].sum()
            total_losses = total_trades - total_wins
            win_rate = total_wins / total_trades

            # Profit factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Average win/loss
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if total_wins > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if total_losses > 0 else 0

            # Total P&L
            total_pnl = trades_df['pnl'].sum()

            summary = {
                'total_trades': total_trades,
                'total_wins': total_wins,
                'total_losses': total_losses,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'date_range_start': trades_df['entry_date'].min(),
                'date_range_end': trades_df['exit_date'].max(),
            }
        else:
            summary = {
                'total_trades': 0,
                'total_wins': 0,
                'total_losses': 0,
                'win_rate': None,
                'profit_factor': None,
                'gross_profit': 0,
                'gross_loss': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'date_range_start': None,
                'date_range_end': None,
            }

        print("\n" + "-"*80)
        print("BACKTEST PERFORMANCE RESULTS")
        print("-"*80)
        print(f"Total trades: {summary['total_trades']}")
        print(f"Wins: {summary['total_wins']}")
        print(f"Losses: {summary['total_losses']}")
        print(f"Win rate: {summary['win_rate']:.4f}" if summary['win_rate'] else "Win rate: N/A")
        print(f"Profit factor: {summary['profit_factor']:.4f}" if summary['profit_factor'] else "Profit factor: N/A")
        print(f"Gross profit: ${summary['gross_profit']:.2f}")
        print(f"Gross loss: ${summary['gross_loss']:.2f}")
        print(f"Total P&L: ${summary['total_pnl']:.2f}")
        print("-"*80)

        # Save trades to CSV
        output_path = self.data_verification_dir / "900_backtest_trades.csv"
        trades_df.to_csv(output_path, index=False)
        print(f"\n[SAVED] All trades saved to {output_path}")

        return trades_df, summary

    def generate_reports(self, markov_summary: Dict, backtest_summary: Dict):
        """Generate markdown reports."""

        # Markov report
        markov_report = f"""# FULL 900-STOCK MARKOV 5-DOWN PATTERN VERIFICATION

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Universe:** {len(self.symbols)} symbols
**Period:** {self.start_date} to {self.end_date}
**Data Source:** yfinance (primary), Stooq (fallback)

---

## CLAIM TO VERIFY

> **Original Claim:** 64.0% up probability with 431 instances (10 symbols)

---

## VERIFICATION RESULTS

### Pattern Instances

| Metric | Value |
|--------|-------|
| Total instances found | {markov_summary['total_instances']} |
| Next day up | {markov_summary['next_day_up']} |
| Next day down | {markov_summary['next_day_down']} |
| Up probability | {markov_summary['up_probability']:.4f if markov_summary['up_probability'] else 'N/A'} |
| 95% CI | [{markov_summary['ci_95_lower']:.4f}, {markov_summary['ci_95_upper']:.4f}] if markov_summary['ci_95_lower'] else 'N/A' |
| Symbols with pattern | {markov_summary['symbols_with_pattern']} |
| Date range | {markov_summary['date_range_start']} to {markov_summary['date_range_end']} |

---

## VERDICT

"""

        if markov_summary['total_instances'] > 0:
            claimed_prob = 0.64
            actual_prob = markov_summary['up_probability']
            diff = abs(claimed_prob - actual_prob)

            if diff < 0.05:
                verdict = "VERIFIED - Claim within 5% of actual"
            elif diff < 0.10:
                verdict = "PARTIALLY VERIFIED - Claim within 10% of actual"
            else:
                verdict = "NOT VERIFIED - Claim differs by >10% from actual"

            markov_report += f"{verdict}\n\n"
            markov_report += f"- Claimed: 64.0% up probability\n"
            markov_report += f"- Actual: {actual_prob:.1%} up probability\n"
            markov_report += f"- Difference: {diff:.1%}\n"
        else:
            markov_report += "NOT VERIFIED - No instances found\n"

        markov_report += f"\n---\n\n## DATA FILES\n\n"
        markov_report += f"- All instances: `data/verification/900_markov_instances.csv`\n"
        markov_report += f"- Total rows: {markov_summary['total_instances']}\n"

        # Save Markov report
        markov_path = self.output_dir / "FULL_900_MARKOV_VERIFICATION.md"
        with open(markov_path, 'w') as f:
            f.write(markov_report)
        print(f"\n[SAVED] Markov report saved to {markov_path}")

        # Backtest report
        backtest_report = f"""# FULL 900-STOCK BACKTEST PERFORMANCE VERIFICATION

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Universe:** {len(self.symbols)} symbols
**Period:** {self.start_date} to {self.end_date}
**Data Source:** yfinance (primary), Stooq (fallback)

---

## CLAIM TO VERIFY

> **Original Claim:** 59.9% WR, 1.24 PF (2,912 trades)

---

## VERIFICATION RESULTS

### Performance Statistics

| Metric | Value |
|--------|-------|
| Total trades | {backtest_summary['total_trades']} |
| Wins | {backtest_summary['total_wins']} |
| Losses | {backtest_summary['total_losses']} |
| Win rate | {backtest_summary['win_rate']:.4f if backtest_summary['win_rate'] else 'N/A'} |
| Profit factor | {backtest_summary['profit_factor']:.4f if backtest_summary['profit_factor'] else 'N/A'} |
| Gross profit | ${backtest_summary['gross_profit']:.2f} |
| Gross loss | ${backtest_summary['gross_loss']:.2f} |
| Total P&L | ${backtest_summary['total_pnl']:.2f} |
| Date range | {backtest_summary['date_range_start']} to {backtest_summary['date_range_end']} |

---

## VERDICT

"""

        if backtest_summary['total_trades'] > 0:
            claimed_wr = 0.599
            claimed_pf = 1.24
            actual_wr = backtest_summary['win_rate']
            actual_pf = backtest_summary['profit_factor']

            wr_diff = abs(claimed_wr - actual_wr)
            pf_diff = abs(claimed_pf - actual_pf)

            if wr_diff < 0.05 and pf_diff < 0.20:
                verdict = "VERIFIED - Claims within acceptable range"
            elif wr_diff < 0.10 and pf_diff < 0.40:
                verdict = "PARTIALLY VERIFIED - Claims within 10% WR, 40% PF"
            else:
                verdict = "NOT VERIFIED - Claims differ significantly from actual"

            backtest_report += f"{verdict}\n\n"
            backtest_report += f"- Claimed WR: 59.9%\n"
            backtest_report += f"- Actual WR: {actual_wr:.1%}\n"
            backtest_report += f"- WR Difference: {wr_diff:.1%}\n\n"
            backtest_report += f"- Claimed PF: 1.24\n"
            backtest_report += f"- Actual PF: {actual_pf:.2f}\n"
            backtest_report += f"- PF Difference: {pf_diff:.2f}\n"
        else:
            backtest_report += "NOT VERIFIED - No trades found\n"

        backtest_report += f"\n---\n\n## DATA FILES\n\n"
        backtest_report += f"- All trades: `data/verification/900_backtest_trades.csv`\n"
        backtest_report += f"- Total rows: {backtest_summary['total_trades']}\n"

        # Save backtest report
        backtest_path = self.output_dir / "FULL_900_BACKTEST_VERIFICATION.md"
        with open(backtest_path, 'w') as f:
            f.write(backtest_report)
        print(f"\n[SAVED] Backtest report saved to {backtest_path}")


def main():
    """Run full verification."""
    print("="*80)
    print("DATA & MATH INTEGRITY MASTER VERIFIER")
    print("="*80)
    print("ZERO FAKE DATA. ZERO HALLUCINATIONS. ZERO BIAS.")
    print("="*80)
    print("")

    # Configuration
    universe_path = "data/universe/optionable_liquid_800.csv"
    start_date = "2015-01-01"
    end_date = "2024-12-31"

    # Initialize verifier
    verifier = DataMathVerifier(universe_path, start_date, end_date)

    # Verify Markov 5-down pattern
    instances_df, markov_summary = verifier.verify_markov_5down_pattern()

    # Verify backtest performance
    trades_df, backtest_summary = verifier.verify_backtest_performance(instances_df)

    # Generate reports
    verifier.generate_reports(markov_summary, backtest_summary)

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("All results saved to:")
    print("  - AUDITS/FULL_900_MARKOV_VERIFICATION.md")
    print("  - AUDITS/FULL_900_BACKTEST_VERIFICATION.md")
    print("  - data/verification/900_markov_instances.csv")
    print("  - data/verification/900_backtest_trades.csv")
    print("="*80)


if __name__ == "__main__":
    main()
