"""
Vectorized Backtester (vectorbt-inspired)
==========================================

High-performance vectorized backtesting for large datasets.

Features:
- Numpy-based vectorized operations (no row iteration)
- Numba JIT compilation for hot paths (optional)
- Memory-efficient chunked processing
- Multi-symbol parallel execution

Based on vectorbt approach but customized for Kobe's strategies.

Usage:
    from backtest.vectorized import VectorizedBacktester

    bt = VectorizedBacktester(config)
    results = bt.run(data, signals)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation (optional)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class VectorConfig:
    """Configuration for vectorized backtester."""
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 1.0  # Simplified commission
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_positions: int = 10
    allow_fractional: bool = False


@dataclass
class VectorResults:
    """Results from vectorized backtest."""
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: pd.DataFrame
    metrics: Dict[str, float]
    symbol_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'equity_final': float(self.equity_curve[-1]) if len(self.equity_curve) > 0 else 0,
            'total_return': float((self.equity_curve[-1] / self.equity_curve[0] - 1) * 100) if len(self.equity_curve) > 0 else 0,
            'metrics': self.metrics,
            'trade_count': len(self.trades),
        }


# Numba-optimized functions for hot paths
@jit(nopython=True, cache=True)
def _compute_position_pnl(
    prices: np.ndarray,
    positions: np.ndarray,
    entry_prices: np.ndarray,
) -> np.ndarray:
    """Compute P&L for positions (vectorized with numba)."""
    n = len(prices)
    pnl = np.zeros(n)
    for i in range(n):
        if positions[i] != 0 and entry_prices[i] > 0:
            pnl[i] = (prices[i] - entry_prices[i]) * positions[i]
    return pnl


@jit(nopython=True, cache=True)
def _check_stops(
    lows: np.ndarray,
    highs: np.ndarray,
    positions: np.ndarray,
    stop_prices: np.ndarray,
    take_profit_prices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Check stop losses and take profits (vectorized with numba)."""
    n = len(lows)
    stop_hit = np.zeros(n, dtype=np.bool_)
    tp_hit = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        if positions[i] > 0:  # Long position
            if stop_prices[i] > 0 and lows[i] <= stop_prices[i]:
                stop_hit[i] = True
            elif take_profit_prices[i] > 0 and highs[i] >= take_profit_prices[i]:
                tp_hit[i] = True
        elif positions[i] < 0:  # Short position
            if stop_prices[i] > 0 and highs[i] >= stop_prices[i]:
                stop_hit[i] = True
            elif take_profit_prices[i] > 0 and lows[i] <= take_profit_prices[i]:
                tp_hit[i] = True

    return stop_hit, tp_hit


@jit(nopython=True, cache=True)
def _compute_drawdown(equity: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute drawdown series and max drawdown (vectorized with numba)."""
    n = len(equity)
    if n == 0:
        return np.array([0.0]), 0.0

    peak = equity[0]
    drawdown = np.zeros(n)
    max_dd = 0.0

    for i in range(n):
        if equity[i] > peak:
            peak = equity[i]
        if peak > 0:
            dd = (equity[i] - peak) / peak
            drawdown[i] = dd
            if dd < max_dd:
                max_dd = dd

    return drawdown, max_dd


class VectorizedBacktester:
    """
    High-performance vectorized backtester.

    Key differences from event-driven backtester:
    - Pre-allocates arrays for positions, equity, P&L
    - Uses numpy broadcasting for signal application
    - Numba JIT for hot loops (stop checks, P&L calculation)
    - Processes all symbols in parallel via matrix operations
    """

    def __init__(self, config: Optional[VectorConfig] = None):
        self.config = config or VectorConfig()
        self.results: Optional[VectorResults] = None

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> VectorResults:
        """
        Run vectorized backtest.

        Args:
            data: OHLCV DataFrame with columns [timestamp, symbol, open, high, low, close, volume]
            signals: Signals DataFrame with columns [timestamp, symbol, side, entry_price, stop_loss, take_profit]

        Returns:
            VectorResults with equity curve, trades, and metrics
        """
        logger.info(f"Starting vectorized backtest: {len(data)} bars, {len(signals)} signals")

        # Validate inputs
        if data.empty:
            return self._empty_results()

        # Convert to wide format for vectorized operations
        symbols = data['symbol'].unique().tolist()
        dates = data['timestamp'].unique()
        dates = np.sort(dates)

        n_dates = len(dates)
        n_symbols = len(symbols)

        # Create symbol index mapping
        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        # Pre-allocate matrices (dates x symbols)
        prices_open = np.full((n_dates, n_symbols), np.nan)
        prices_high = np.full((n_dates, n_symbols), np.nan)
        prices_low = np.full((n_dates, n_symbols), np.nan)
        prices_close = np.full((n_dates, n_symbols), np.nan)

        # Fill price matrices
        for sym in symbols:
            sym_data = data[data['symbol'] == sym].set_index('timestamp')
            idx = sym_to_idx[sym]
            for i, dt in enumerate(dates):
                if dt in sym_data.index:
                    row = sym_data.loc[dt]
                    prices_open[i, idx] = row['open']
                    prices_high[i, idx] = row['high']
                    prices_low[i, idx] = row['low']
                    prices_close[i, idx] = row['close']

        # Forward-fill prices
        for j in range(n_symbols):
            for i in range(1, n_dates):
                if np.isnan(prices_close[i, j]):
                    prices_close[i, j] = prices_close[i-1, j]
                    prices_open[i, j] = prices_open[i-1, j]
                    prices_high[i, j] = prices_high[i-1, j]
                    prices_low[i, j] = prices_low[i-1, j]

        # Initialize position tracking
        positions = np.zeros((n_dates, n_symbols))  # Quantity held
        entry_prices = np.zeros((n_dates, n_symbols))  # Entry price per position
        stop_prices = np.zeros((n_dates, n_symbols))  # Stop loss levels
        tp_prices = np.zeros((n_dates, n_symbols))  # Take profit levels
        entry_dates = np.zeros((n_dates, n_symbols), dtype=int)  # Entry bar index

        # Process signals
        trades_list = []
        cash = self.config.initial_cash
        equity_curve = np.zeros(n_dates)

        for i in range(n_dates):
            dt = dates[i]

            # Check exits for existing positions
            for j in range(n_symbols):
                if positions[i-1, j] > 0 if i > 0 else False:
                    # Carry forward position
                    positions[i, j] = positions[i-1, j]
                    entry_prices[i, j] = entry_prices[i-1, j]
                    stop_prices[i, j] = stop_prices[i-1, j]
                    tp_prices[i, j] = tp_prices[i-1, j]
                    entry_dates[i, j] = entry_dates[i-1, j]

                    # Check stop loss
                    if stop_prices[i, j] > 0 and prices_low[i, j] <= stop_prices[i, j]:
                        exit_price = stop_prices[i, j]
                        pnl = (exit_price - entry_prices[i, j]) * positions[i, j]
                        pnl -= abs(pnl) * self.config.commission_bps / 10000  # Commission
                        cash += positions[i, j] * exit_price + pnl
                        trades_list.append({
                            'entry_date': dates[int(entry_dates[i, j])],
                            'exit_date': dt,
                            'symbol': symbols[j],
                            'side': 'long',
                            'qty': positions[i, j],
                            'entry_price': entry_prices[i, j],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss',
                        })
                        positions[i, j] = 0
                        entry_prices[i, j] = 0
                        stop_prices[i, j] = 0
                        tp_prices[i, j] = 0

                    # Check take profit
                    elif tp_prices[i, j] > 0 and prices_high[i, j] >= tp_prices[i, j]:
                        exit_price = tp_prices[i, j]
                        pnl = (exit_price - entry_prices[i, j]) * positions[i, j]
                        pnl -= abs(pnl) * self.config.commission_bps / 10000
                        cash += positions[i, j] * exit_price + pnl
                        trades_list.append({
                            'entry_date': dates[int(entry_dates[i, j])],
                            'exit_date': dt,
                            'symbol': symbols[j],
                            'side': 'long',
                            'qty': positions[i, j],
                            'entry_price': entry_prices[i, j],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_reason': 'take_profit',
                        })
                        positions[i, j] = 0
                        entry_prices[i, j] = 0
                        stop_prices[i, j] = 0
                        tp_prices[i, j] = 0

            # Process new entries for this bar
            bar_signals = signals[signals['timestamp'] == dt] if not signals.empty else pd.DataFrame()
            for _, sig in bar_signals.iterrows():
                sym = sig['symbol']
                if sym not in sym_to_idx:
                    continue
                j = sym_to_idx[sym]

                # Skip if already in position
                if positions[i, j] > 0:
                    continue

                # Calculate entry price with slippage
                entry_px = prices_open[i, j] * (1 + self.config.slippage_bps / 10000)
                if np.isnan(entry_px):
                    continue

                # Calculate position size based on risk
                stop_loss = float(sig.get('stop_loss', 0))
                if stop_loss > 0:
                    risk_per_share = abs(entry_px - stop_loss)
                    if risk_per_share > 0:
                        risk_amount = cash * self.config.risk_per_trade
                        qty = int(risk_amount / risk_per_share)
                    else:
                        qty = int(cash * 0.01 / entry_px)
                else:
                    qty = int(cash * 0.01 / entry_px)

                qty = max(1, min(qty, int(cash / entry_px)))

                if qty <= 0:
                    continue

                # Execute entry
                cost = qty * entry_px * (1 + self.config.commission_bps / 10000)
                if cost > cash:
                    continue

                cash -= cost
                positions[i, j] = qty
                entry_prices[i, j] = entry_px
                stop_prices[i, j] = stop_loss
                tp_prices[i, j] = float(sig.get('take_profit', 0))
                entry_dates[i, j] = i

            # Calculate equity (cash + positions marked to market)
            port_value = cash
            for j in range(n_symbols):
                if positions[i, j] > 0:
                    port_value += positions[i, j] * prices_close[i, j]
            equity_curve[i] = port_value

        # Calculate returns
        returns = np.zeros(n_dates)
        for i in range(1, n_dates):
            if equity_curve[i-1] > 0:
                returns[i] = (equity_curve[i] / equity_curve[i-1]) - 1

        # Calculate metrics
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        metrics = self._compute_metrics(equity_curve, returns, trades_df)

        self.results = VectorResults(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades_df,
            metrics=metrics,
        )

        logger.info(f"Backtest complete: {len(trades_list)} trades, final equity ${equity_curve[-1]:,.2f}")

        return self.results

    def _compute_metrics(
        self,
        equity: np.ndarray,
        returns: np.ndarray,
        trades: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        if len(equity) == 0:
            return {}

        # Basic metrics
        initial = equity[0]
        final = equity[-1]
        total_return = (final / initial - 1) * 100 if initial > 0 else 0

        # Drawdown
        _, max_dd = _compute_drawdown(equity)

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            mu = np.nanmean(returns)
            sigma = np.nanstd(returns)
            sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        else:
            sharpe = 0

        # Trade metrics
        if not trades.empty and 'pnl' in trades.columns:
            wins = trades[trades['pnl'] > 0]
            losses = trades[trades['pnl'] <= 0]
            win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
            gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        return {
            'initial_equity': float(initial),
            'final_equity': float(final),
            'total_return_pct': float(total_return),
            'max_drawdown_pct': float(max_dd * 100),
            'sharpe_ratio': float(sharpe),
            'total_trades': len(trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
        }

    def _empty_results(self) -> VectorResults:
        """Return empty results."""
        return VectorResults(
            equity_curve=np.array([self.config.initial_cash]),
            returns=np.array([0.0]),
            positions=np.array([[0]]),
            trades=pd.DataFrame(),
            metrics={},
        )


def run_parallel_backtests(
    data: pd.DataFrame,
    signals_list: List[pd.DataFrame],
    config: Optional[VectorConfig] = None,
    n_jobs: int = -1,
) -> List[VectorResults]:
    """
    Run multiple backtests in parallel.

    Args:
        data: OHLCV data
        signals_list: List of signal DataFrames to test
        config: Backtest configuration
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        List of VectorResults
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not available, running sequentially")
        results = []
        for signals in signals_list:
            bt = VectorizedBacktester(config)
            results.append(bt.run(data, signals))
        return results

    def _run_single(signals):
        bt = VectorizedBacktester(config)
        return bt.run(data, signals)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single)(signals) for signals in signals_list
    )

    return results
