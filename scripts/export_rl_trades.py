"""
Export RL Agent Trade History - Production Grade

Exports RL agent trade history to standardized format for benchmarking.
Renaissance Technologies quality standard.

Features:
- Schema validation (ensures all required fields present)
- Data type validation (prices are floats, timestamps are dates)
- Regime annotation (adds Bull/Bear/Neutral from HMM)
- Transaction cost adjustment (applies realistic slippage/fees)
- Reproducibility (deterministic output, version tracking)

Usage:
    python scripts/export_rl_trades.py --agent PPO --input logs/rl_ppo_trades.csv --output exports/ppo_validated.csv

Author: Kobe Trading System
Date: 2026-01-08
Version: 1.0
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required schema for benchmark-ready trade history
REQUIRED_COLUMNS = [
    'timestamp',
    'symbol',
    'side',          # 'long' or 'short'
    'entry_price',
    'exit_price',
    'quantity',
    'pnl',           # Absolute P&L
    'pnl_pct',       # Percentage P&L
]

OPTIONAL_COLUMNS = [
    'entry_timestamp',
    'exit_timestamp',
    'holding_period_days',
    'mfe',           # Max Favorable Excursion
    'mae',           # Max Adverse Excursion
    'regime',        # Bull/Bear/Neutral
    'strategy',
    'stop_loss',
    'take_profit',
]

# Transaction costs
DEFAULT_SLIPPAGE_BPS = 2.5  # 2.5 basis points slippage
DEFAULT_COMMISSION_BPS = 5.0  # 5 bps commission (Alpaca)
TOTAL_COST_BPS = DEFAULT_SLIPPAGE_BPS + DEFAULT_COMMISSION_BPS  # 7.5 bps


def validate_trade_schema(df: pd.DataFrame, strict: bool = True) -> bool:
    """
    Validate trade history schema.

    Args:
        df: Trade history DataFrame
        strict: If True, raise exception on validation failure

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails and strict=True
    """
    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        msg = "timestamp must be datetime64"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    for col in ['entry_price', 'exit_price', 'pnl', 'pnl_pct']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            msg = f"{col} must be numeric"
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False

    # Check side values
    valid_sides = {'long', 'short', 'LONG', 'SHORT', 'BUY', 'SELL'}
    invalid_sides = set(df['side'].unique()) - valid_sides
    if invalid_sides:
        msg = f"Invalid side values: {invalid_sides}. Must be long/short"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    logger.info("Schema validation passed")
    return True


def normalize_trade_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trade history to standard format.

    Args:
        df: Raw trade history

    Returns:
        Normalized DataFrame
    """
    df = df.copy()

    # Normalize side column
    df['side'] = df['side'].str.lower()
    df['side'] = df['side'].replace({'buy': 'long', 'sell': 'short'})

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add holding period if not present
    if 'holding_period_days' not in df.columns and 'entry_timestamp' in df.columns and 'exit_timestamp' in df.columns:
        df['holding_period_days'] = (
            pd.to_datetime(df['exit_timestamp']) - pd.to_datetime(df['entry_timestamp'])
        ).dt.total_seconds() / (24 * 3600)

    # Ensure numeric columns are float
    for col in ['entry_price', 'exit_price', 'pnl', 'pnl_pct']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    logger.info("Trade history normalized")
    return df


def apply_transaction_costs(
    df: pd.DataFrame,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
) -> pd.DataFrame:
    """
    Apply realistic transaction costs to P&L.

    Args:
        df: Trade history
        slippage_bps: Slippage in basis points (default: 2.5 bps)
        commission_bps: Commission in basis points (default: 5 bps)

    Returns:
        DataFrame with adjusted P&L
    """
    df = df.copy()

    total_cost_pct = (slippage_bps + commission_bps) / 10000  # Convert bps to decimal

    # Apply cost per trade (round-trip: entry + exit)
    cost_per_trade_pct = 2 * total_cost_pct  # 2x for round-trip

    # Adjust P&L percentage
    df['pnl_pct_gross'] = df['pnl_pct']  # Save gross P&L
    df['pnl_pct'] = df['pnl_pct'] - (cost_per_trade_pct * 100)  # Subtract costs

    # Adjust absolute P&L (based on entry price and quantity)
    if 'quantity' in df.columns and 'entry_price' in df.columns:
        trade_value = df['quantity'] * df['entry_price']
        cost_dollars = trade_value * cost_per_trade_pct
        df['pnl_gross'] = df['pnl']  # Save gross P&L
        df['pnl'] = df['pnl'] - cost_dollars
    else:
        # Approximate based on percentage
        df['pnl_gross'] = df['pnl']
        df['pnl'] = df['pnl'] * (1 - cost_per_trade_pct)

    logger.info(f"Applied transaction costs: {slippage_bps} bps slippage + {commission_bps} bps commission")
    return df


def annotate_with_regime(
    df: pd.DataFrame,
    regime_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Annotate trades with market regime.

    Args:
        df: Trade history
        regime_data: DataFrame with timestamp, regime columns (optional)

    Returns:
        DataFrame with regime column added
    """
    df = df.copy()

    if regime_data is None:
        # Try to load from default location
        regime_path = Path("state/regime/hmm_regime_history.csv")
        if regime_path.exists():
            regime_data = pd.read_csv(regime_path, parse_dates=['timestamp'])
            logger.info(f"Loaded regime data from {regime_path}")
        else:
            logger.warning("No regime data available - skipping regime annotation")
            df['regime'] = 'UNKNOWN'
            return df

    # Merge trades with regime data
    df_with_regime = df.merge(
        regime_data[['timestamp', 'regime']],
        on='timestamp',
        how='left'
    )

    # Fill missing regimes with forward fill
    df_with_regime['regime'] = df_with_regime['regime'].fillna(method='ffill')
    df_with_regime['regime'] = df_with_regime['regime'].fillna('UNKNOWN')

    logger.info("Annotated trades with market regime")
    return df_with_regime


def export_trade_history(
    input_path: str,
    output_path: str,
    agent_name: str = "PPO",
    apply_costs: bool = True,
    add_regime: bool = True,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    regime_data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Export RL agent trade history to benchmark-ready format.

    Args:
        input_path: Path to raw trade history CSV
        output_path: Path to save validated trade history
        agent_name: Name of RL agent (for metadata)
        apply_costs: Whether to apply transaction costs
        add_regime: Whether to annotate with market regime
        slippage_bps: Slippage in basis points
        commission_bps: Commission in basis points
        regime_data_path: Path to regime data CSV (optional)

    Returns:
        Validated and exported DataFrame
    """
    logger.info(f"Exporting RL trade history: {agent_name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Load raw trade history
    df = pd.read_csv(input_path, parse_dates=['timestamp'])
    logger.info(f"Loaded {len(df)} trades from {input_path}")

    # Normalize
    df = normalize_trade_history(df)

    # Apply transaction costs
    if apply_costs:
        df = apply_transaction_costs(df, slippage_bps, commission_bps)

    # Annotate with regime
    if add_regime:
        if regime_data_path:
            regime_data = pd.read_csv(regime_data_path, parse_dates=['timestamp'])
        else:
            regime_data = None

        df = annotate_with_regime(df, regime_data)

    # Validate schema
    validate_trade_schema(df, strict=True)

    # Add metadata columns
    df['agent_name'] = agent_name
    df['export_timestamp'] = datetime.now().isoformat()
    df['export_version'] = '1.0'

    # Save to output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(df)} validated trades to {output_path}")

    # Print summary statistics
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).sum() / len(df) if len(df) > 0 else 0.0
    avg_pnl_pct = df['pnl_pct'].mean()

    logger.info("=" * 60)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Agent: {agent_name}")
    logger.info(f"Total Trades: {len(df)}")
    logger.info(f"Total P&L: ${total_pnl:,.2f}")
    logger.info(f"Win Rate: {win_rate:.1%}")
    logger.info(f"Avg P&L%: {avg_pnl_pct:.2f}%")
    logger.info(f"Transaction Costs Applied: {apply_costs}")
    logger.info(f"Regime Annotation: {add_regime}")
    logger.info("=" * 60)

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export RL agent trade history")
    parser.add_argument('--input', type=str, required=True, help="Path to raw trade history CSV")
    parser.add_argument('--output', type=str, required=True, help="Path to save validated trade history")
    parser.add_argument('--agent', type=str, default="PPO", help="Agent name (default: PPO)")
    parser.add_argument('--no-costs', action='store_true', help="Skip transaction cost adjustment")
    parser.add_argument('--no-regime', action='store_true', help="Skip regime annotation")
    parser.add_argument('--slippage-bps', type=float, default=DEFAULT_SLIPPAGE_BPS, help="Slippage in basis points")
    parser.add_argument('--commission-bps', type=float, default=DEFAULT_COMMISSION_BPS, help="Commission in basis points")
    parser.add_argument('--regime-data', type=str, help="Path to regime data CSV (optional)")

    args = parser.parse_args()

    export_trade_history(
        input_path=args.input,
        output_path=args.output,
        agent_name=args.agent,
        apply_costs=not args.no_costs,
        add_regime=not args.no_regime,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        regime_data_path=args.regime_data,
    )


if __name__ == "__main__":
    main()
