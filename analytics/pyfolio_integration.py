"""
Pyfolio Tearsheet Generation for Kobe Trading System.

Converts Kobe backtest outputs to pyfolio format and generates
professional tearsheets for portfolio analysis.

Usage:
    from analytics.pyfolio_integration import generate_simple_tearsheet

    # Generate tearsheet from backtest results
    generate_simple_tearsheet(
        backtest_dir=Path("wf_outputs/ibs_rsi/split_01"),
        output_dir=Path("reports/tearsheets"),
        benchmark_symbol="SPY"
    )

Features:
- Automatic conversion from Kobe equity_curve.csv to pyfolio returns
- Transaction conversion from trade_list.csv
- Position reconstruction for full tearsheets
- Benchmark comparison (SPY default)
- PNG and PDF output

Author: Claude Opus 4.5
Date: 2026-01-07
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import warnings

import numpy as np
import pandas as pd

# Suppress pyfolio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyfolio")

# NumPy 2.0 compatibility fix for pyfolio/empyrical
if not hasattr(np, "NINF"):
    np.NINF = -np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf

try:
    import pyfolio as pf
    HAS_PYFOLIO = True
except ImportError:
    HAS_PYFOLIO = False
    pf = None

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_returns_from_backtest(backtest_dir: Path) -> pd.Series:
    """
    Load returns series from Kobe backtest equity_curve.csv.

    Args:
        backtest_dir: Path to backtest output (contains equity_curve.csv)

    Returns:
        pd.Series with DatetimeIndex (UTC) and daily returns
    """
    equity_path = backtest_dir / "equity_curve.csv"

    if not equity_path.exists():
        raise FileNotFoundError(f"equity_curve.csv not found in {backtest_dir}")

    df = pd.read_csv(equity_path, parse_dates=["timestamp"], index_col="timestamp")

    # Get returns column
    if "returns" in df.columns:
        returns = df["returns"].copy()
    elif "equity" in df.columns:
        # Calculate returns from equity if not present
        returns = df["equity"].pct_change().fillna(0.0)
    else:
        raise ValueError("equity_curve.csv must have 'returns' or 'equity' column")

    # Ensure UTC timezone (pyfolio requirement)
    if returns.index.tz is None:
        returns.index = returns.index.tz_localize("UTC")
    else:
        returns.index = returns.index.tz_convert("UTC")

    # Remove any NaN
    returns = returns.fillna(0.0)

    logger.info(f"Loaded {len(returns)} returns from {equity_path}")
    return returns


def load_transactions_from_backtest(backtest_dir: Path) -> pd.DataFrame:
    """
    Load transactions from Kobe trade_list.csv and convert to pyfolio format.

    Pyfolio expects:
        - DatetimeIndex (UTC)
        - Columns: amount (signed qty), price, symbol

    Args:
        backtest_dir: Path to backtest output

    Returns:
        pd.DataFrame in pyfolio transactions format
    """
    trade_path = backtest_dir / "trade_list.csv"

    if not trade_path.exists():
        logger.warning(f"trade_list.csv not found in {backtest_dir}")
        return pd.DataFrame(columns=["amount", "price", "symbol"])

    df = pd.read_csv(trade_path, parse_dates=["timestamp"])

    if df.empty:
        return pd.DataFrame(columns=["amount", "price", "symbol"])

    # Convert side to signed amount
    df["amount"] = df.apply(
        lambda row: row["qty"] if row["side"].upper() == "BUY" else -row["qty"],
        axis=1
    )

    # Set index and select columns
    txn = df[["amount", "price", "symbol"]].copy()
    txn.index = pd.to_datetime(df["timestamp"])

    # Ensure UTC
    if txn.index.tz is None:
        txn.index = txn.index.tz_localize("UTC")
    else:
        txn.index = txn.index.tz_convert("UTC")

    logger.info(f"Loaded {len(txn)} transactions from {trade_path}")
    return txn


def fetch_benchmark_returns(
    start_date: str,
    end_date: str,
    symbol: str = "SPY"
) -> pd.Series:
    """
    Fetch benchmark returns from Polygon for same date range.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        symbol: Benchmark ticker (default SPY)

    Returns:
        pd.Series of daily returns
    """
    try:
        from data.providers.polygon_eod import PolygonEODProvider

        provider = PolygonEODProvider()
        df = provider.get_bars(symbol, start_date, end_date)

        if df is None or df.empty:
            logger.warning(f"Could not fetch {symbol} benchmark data")
            return pd.Series(dtype=float)

        # Ensure datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        df = df.sort_index()

        # Calculate returns
        returns = df["close"].pct_change().fillna(0.0)

        # Ensure UTC
        if returns.index.tz is None:
            returns.index = returns.index.tz_localize("UTC")

        logger.info(f"Fetched {len(returns)} benchmark returns for {symbol}")
        return returns

    except Exception as e:
        logger.warning(f"Could not fetch benchmark: {e}")
        return pd.Series(dtype=float)


def reconstruct_positions(
    transactions: pd.DataFrame,
    equity_df: pd.DataFrame,
    price_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Reconstruct daily positions from transactions.

    Args:
        transactions: Pyfolio transactions DataFrame
        equity_df: Kobe equity curve with daily equity values
        price_data: Optional dict of {symbol: OHLCV DataFrame} for marking to market

    Returns:
        pd.DataFrame with columns = symbols + 'cash', index = dates
    """
    if transactions.empty:
        return pd.DataFrame()

    # Get all trading days from equity curve
    dates = pd.to_datetime(equity_df.index)
    if dates.tz is None:
        dates = dates.tz_localize("UTC")

    # Track position quantities over time
    from collections import defaultdict
    positions_dict = defaultdict(lambda: pd.Series(0.0, index=dates))

    # Group transactions by symbol
    symbols = transactions["symbol"].unique()

    for date in dates:
        # Get all transactions up to this date
        mask = transactions.index <= date
        day_txns = transactions[mask]

        for symbol in symbols:
            sym_txns = day_txns[day_txns["symbol"] == symbol]
            qty = sym_txns["amount"].sum()

            if qty != 0 and price_data and symbol in price_data:
                # Mark to market
                prices = price_data[symbol]
                if hasattr(prices.index, "tz") and prices.index.tz is None:
                    prices.index = prices.index.tz_localize("UTC")

                if date in prices.index:
                    close_price = prices.loc[date, "close"]
                    positions_dict[symbol][date] = qty * close_price
                else:
                    # Use last known price
                    prior = prices[prices.index <= date]
                    if not prior.empty:
                        positions_dict[symbol][date] = qty * prior.iloc[-1]["close"]
            elif qty != 0:
                # No price data - use average transaction price
                avg_price = sym_txns["price"].mean()
                positions_dict[symbol][date] = qty * avg_price

    # Convert to DataFrame
    pos_df = pd.DataFrame(dict(positions_dict), index=dates)

    # Add cash column (equity - sum of positions)
    equity_series = equity_df["equity"] if "equity" in equity_df.columns else equity_df.iloc[:, 0]
    equity_series.index = dates
    pos_df["cash"] = equity_series - pos_df.sum(axis=1)

    return pos_df


# =============================================================================
# TEARSHEET GENERATION
# =============================================================================

def generate_simple_tearsheet(
    backtest_dir: Path,
    output_dir: Optional[Path] = None,
    benchmark_symbol: str = "SPY",
    save_png: bool = True,
    save_pdf: bool = True,
) -> Optional[Path]:
    """
    Generate a simple pyfolio tearsheet from Kobe backtest results.

    This version only requires equity_curve.csv (no positions/transactions).

    Args:
        backtest_dir: Path to backtest output directory
        output_dir: Where to save tearsheet (default: same as backtest_dir)
        benchmark_symbol: Benchmark ticker to compare against
        save_png: Save as PNG
        save_pdf: Save as PDF

    Returns:
        Path to saved tearsheet (PNG), or None if failed
    """
    if not HAS_PYFOLIO:
        logger.error("pyfolio not installed: pip install pyfolio-reloaded")
        return None

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    backtest_dir = Path(backtest_dir)

    # Load returns
    try:
        returns = load_returns_from_backtest(backtest_dir)
    except Exception as e:
        logger.error(f"Failed to load returns: {e}")
        return None

    if returns.empty:
        logger.error("No returns data found")
        return None

    # Fetch benchmark
    start_date = returns.index.min().strftime("%Y-%m-%d")
    end_date = returns.index.max().strftime("%Y-%m-%d")
    benchmark = fetch_benchmark_returns(start_date, end_date, benchmark_symbol)

    # Align benchmark to strategy dates
    if not benchmark.empty:
        benchmark = benchmark.reindex(returns.index).fillna(0.0)
    else:
        benchmark = None

    # Generate tearsheet
    try:
        fig = plt.figure(figsize=(14, 10))

        if benchmark is not None:
            pf.create_simple_tear_sheet(returns, benchmark_rets=benchmark)
        else:
            pf.create_simple_tear_sheet(returns)

        # Save to file
        if output_dir is None:
            output_dir = backtest_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = None

        if save_png:
            png_path = output_dir / "pyfolio_simple_tearsheet.png"
            plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved tearsheet: {png_path}")
            output_path = png_path

        if save_pdf:
            pdf_path = output_dir / "pyfolio_simple_tearsheet.pdf"
            plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved tearsheet: {pdf_path}")

        plt.close("all")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate tearsheet: {e}")
        plt.close("all")
        return None


def generate_full_tearsheet(
    backtest_dir: Path,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Optional[Path] = None,
    live_start_date: Optional[str] = None,
    benchmark_symbol: str = "SPY",
    save_png: bool = True,
    save_pdf: bool = True,
) -> Optional[Path]:
    """
    Generate comprehensive pyfolio tearsheet with positions and transactions.

    Args:
        backtest_dir: Path to backtest output
        price_data: Dict of {symbol: OHLCV DataFrame} for position reconstruction
        output_dir: Where to save output
        live_start_date: Out-of-sample split date (e.g., '2024-01-01')
        benchmark_symbol: Benchmark ticker
        save_png: Save as PNG
        save_pdf: Save as PDF

    Returns:
        Path to saved tearsheet, or None if failed
    """
    if not HAS_PYFOLIO:
        logger.error("pyfolio not installed: pip install pyfolio-reloaded")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    backtest_dir = Path(backtest_dir)

    # Load data
    try:
        returns = load_returns_from_backtest(backtest_dir)
        transactions = load_transactions_from_backtest(backtest_dir)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    # Load equity curve for position reconstruction
    equity_path = backtest_dir / "equity_curve.csv"
    equity_df = pd.read_csv(equity_path, parse_dates=["timestamp"], index_col="timestamp")

    # Reconstruct positions if we have transactions
    positions = None
    if not transactions.empty:
        positions = reconstruct_positions(transactions, equity_df, price_data)

    # Fetch benchmark
    start_date = returns.index.min().strftime("%Y-%m-%d")
    end_date = returns.index.max().strftime("%Y-%m-%d")
    benchmark = fetch_benchmark_returns(start_date, end_date, benchmark_symbol)

    if not benchmark.empty:
        benchmark = benchmark.reindex(returns.index).fillna(0.0)
    else:
        benchmark = None

    # Generate full tearsheet
    try:
        fig = plt.figure(figsize=(14, 20))

        kwargs = {"returns": returns}
        if benchmark is not None:
            kwargs["benchmark_rets"] = benchmark
        if positions is not None and not positions.empty:
            kwargs["positions"] = positions
        if not transactions.empty:
            kwargs["transactions"] = transactions
        if live_start_date:
            kwargs["live_start_date"] = live_start_date

        # Try full tearsheet, fall back to simple if it fails
        try:
            pf.create_full_tear_sheet(**kwargs, round_trips=True)
        except Exception as e:
            logger.warning(f"Full tearsheet failed, using simple: {e}")
            pf.create_simple_tear_sheet(returns, benchmark_rets=benchmark)

        # Save
        if output_dir is None:
            output_dir = backtest_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = None

        if save_png:
            png_path = output_dir / "pyfolio_full_tearsheet.png"
            plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved full tearsheet: {png_path}")
            output_path = png_path

        if save_pdf:
            pdf_path = output_dir / "pyfolio_full_tearsheet.pdf"
            plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved full tearsheet: {pdf_path}")

        plt.close("all")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate full tearsheet: {e}")
        plt.close("all")
        return None


def generate_wf_tearsheets(
    wf_dir: Path,
    output_dir: Optional[Path] = None,
    benchmark_symbol: str = "SPY",
    max_splits: int = 10,
) -> int:
    """
    Generate tearsheets for all walk-forward splits.

    Args:
        wf_dir: Walk-forward strategy directory (contains split_* folders)
        output_dir: Where to save (default: in each split folder)
        benchmark_symbol: Benchmark ticker
        max_splits: Maximum splits to process

    Returns:
        Number of tearsheets generated
    """
    wf_dir = Path(wf_dir)
    splits = sorted(wf_dir.glob("split_*"))[:max_splits]

    if not splits:
        logger.warning(f"No split directories found in {wf_dir}")
        return 0

    generated = 0

    for split_dir in splits:
        logger.info(f"Processing {split_dir.name}...")

        out_dir = Path(output_dir) / split_dir.name if output_dir else split_dir

        result = generate_simple_tearsheet(
            split_dir,
            output_dir=out_dir,
            benchmark_symbol=benchmark_symbol,
        )

        if result:
            generated += 1

    logger.info(f"Generated {generated}/{len(splits)} tearsheets")
    return generated


# =============================================================================
# METRICS EXTRACTION
# =============================================================================

def extract_pyfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Extract key metrics using pyfolio/empyrical.

    Args:
        returns: Daily returns series

    Returns:
        Dict of metric name -> value
    """
    if not HAS_PYFOLIO:
        return {}

    import empyrical as ep

    metrics = {}

    try:
        metrics["total_return"] = ep.cum_returns_final(returns)
        metrics["cagr"] = ep.cagr(returns)
        metrics["sharpe_ratio"] = ep.sharpe_ratio(returns)
        metrics["sortino_ratio"] = ep.sortino_ratio(returns)
        metrics["calmar_ratio"] = ep.calmar_ratio(returns)
        metrics["max_drawdown"] = ep.max_drawdown(returns)
        metrics["annual_volatility"] = ep.annual_volatility(returns)
        metrics["tail_ratio"] = ep.tail_ratio(returns)
        metrics["stability"] = ep.stability_of_timeseries(returns)
    except Exception as e:
        logger.warning(f"Failed to calculate some metrics: {e}")

    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pyfolio tearsheets")
    parser.add_argument("--backtest-dir", type=str, help="Single backtest directory")
    parser.add_argument("--wf-dir", type=str, help="Walk-forward strategy directory")
    parser.add_argument("--all-splits", action="store_true", help="Process all WF splits")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark symbol")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")
    parser.add_argument("--full", action="store_true", help="Generate full tearsheet")

    args = parser.parse_args()

    if args.backtest_dir:
        backtest_path = Path(args.backtest_dir)
        output_path = Path(args.output_dir) if args.output_dir else backtest_path

        print(f"Generating tearsheet for: {backtest_path}")

        if args.full:
            result = generate_full_tearsheet(
                backtest_path,
                output_dir=output_path,
                benchmark_symbol=args.benchmark
            )
        else:
            result = generate_simple_tearsheet(
                backtest_path,
                output_dir=output_path,
                benchmark_symbol=args.benchmark
            )

        if result:
            print(f"Saved: {result}")
        else:
            print("Failed to generate tearsheet")

    elif args.wf_dir and args.all_splits:
        wf_path = Path(args.wf_dir)
        output_path = Path(args.output_dir) if args.output_dir else None

        count = generate_wf_tearsheets(
            wf_path,
            output_dir=output_path,
            benchmark_symbol=args.benchmark
        )

        print(f"Generated {count} tearsheets")

    else:
        parser.print_help()
