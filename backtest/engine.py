from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings_loader import get_sizing_config, is_sizing_enabled

@dataclass
class CommissionConfig:
    """Commission model configuration."""
    enabled: bool = False
    per_share: float = 0.0  # $ per share
    min_per_order: float = 0.0  # Minimum $ per order
    bps: float = 0.0  # Basis points of notional
    sec_fee_per_dollar: float = 0.0000278  # SEC fee on sales
    taf_fee_per_share: float = 0.000166  # FINRA TAF on sales


@dataclass
class BacktestConfig:
    """Configuration for backtest execution including dates, capital, and costs."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_cash: float = 100_000.0
    slippage_bps: float = 5.0  # 5 bps default
    commissions: Optional[CommissionConfig] = None

@dataclass
class Trade:
    """Record of an executed trade with timestamp, symbol, side, quantity, and price."""
    timestamp: datetime
    symbol: str
    side: str
    qty: int
    price: float

@dataclass
class Position:
    """Represents an open position with symbol, quantity, and average cost basis."""
    symbol: str
    qty: int = 0
    avg_cost: float = 0.0

class Backtester:
    """Event-driven backtester with FIFO P&L, slippage, and commission modeling."""
    def __init__(self, cfg: BacktestConfig, get_signals: Callable[[pd.DataFrame], pd.DataFrame], fetch_bars: Callable[[str], pd.DataFrame]):
        self.cfg = cfg
        self.get_signals = get_signals
        self.fetch_bars = fetch_bars
        self.cash = cfg.initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.initial_cash = cfg.initial_cash
        self.total_commissions = 0.0  # Track total fees paid

    def _compute_commission(self, qty: int, price: float, is_sell: bool) -> float:
        """Compute commission for a trade. Returns total fee in dollars."""
        comm_cfg = self.cfg.commissions
        if comm_cfg is None or not comm_cfg.enabled:
            return 0.0

        notional = qty * price
        fee = 0.0

        # Per-share commission (e.g., IBKR Pro: $0.005/share)
        if comm_cfg.per_share > 0:
            fee = max(qty * comm_cfg.per_share, comm_cfg.min_per_order)

        # Basis points of notional (alternative model)
        if comm_cfg.bps > 0:
            fee = max(fee, notional * comm_cfg.bps / 10000.0)

        # SEC fee on sales only (~$27.80 per $1M sold)
        if is_sell and comm_cfg.sec_fee_per_dollar > 0:
            fee += notional * comm_cfg.sec_fee_per_dollar

        # FINRA TAF on sales only (~$0.000166/share sold)
        if is_sell and comm_cfg.taf_fee_per_share > 0:
            fee += qty * comm_cfg.taf_fee_per_share

        return fee

    def run(self, symbols: List[str], outdir: Optional[str] = None) -> Dict[str, Any]:
        """Execute backtest across symbols, simulating trades with FIFO P&L accounting.

        Args:
            symbols: List of ticker symbols to backtest.
            outdir: Optional directory to write trade_list.csv and equity_curve.csv.

        Returns:
            Dict with keys: trades, pnl, equity (DataFrame), metrics (win_rate, profit_factor, etc.)
        """
        # Load data per symbol
        by_sym: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            df = self.fetch_bars(s)
            if df is None or df.empty:
                continue
            if 'symbol' not in df:
                df = df.copy(); df['symbol'] = s
            # Normalize timestamps to tz-naive
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            by_sym[s] = df.sort_values('timestamp').reset_index(drop=True)
        if not by_sym:
            return {"trades": [], "pnl": 0.0, "equity": pd.DataFrame(), "metrics": {}}

        # Merge for signal generation (strategies may need cross-symbol frame)
        merged = pd.concat(by_sym.values(), ignore_index=True)
        merged['timestamp'] = pd.to_datetime(merged['timestamp'], utc=True).dt.tz_localize(None)
        merged = merged.sort_values(['symbol','timestamp'])
        signals = self.get_signals(merged)
        # Normalize signal timestamps
        if not signals.empty and 'timestamp' in signals.columns:
            signals['timestamp'] = pd.to_datetime(signals['timestamp'], utc=True).dt.tz_localize(None)

        # Group signals by symbol and simulate with exits (ATR stop + 5-bar time stop)
        for sym, sym_df in by_sym.items():
            sym_sigs = signals[signals['symbol'] == sym].sort_values('timestamp') if not signals.empty else pd.DataFrame(columns=signals.columns if isinstance(signals, pd.DataFrame) else [])
            if sym_sigs.empty:
                continue
            self._simulate_symbol(sym_df, sym_sigs)

        # Build daily equity curve (mark-to-market by close)
        all_data = pd.concat(list(by_sym.values()), ignore_index=True)
        equity_df = self._compute_equity_series(all_data)
        metrics = self._compute_metrics(equity_df, self.trades)

        # Optional outputs
        if outdir:
            self._write_outputs(outdir, equity_df, metrics)

        final_equity = float(equity_df['equity'].iloc[-1]) if not equity_df.empty else self.cash
        pnl = final_equity - self.initial_cash
        return {"trades": self.trades, "equity": equity_df, "metrics": metrics, "cash": self.cash, "pnl": pnl}

    def _execute(self, sym: str, side: str, qty: int, price: float, ts: datetime):
        if side == 'long':
            cost = qty * price
            commission = self._compute_commission(qty, price, is_sell=False)
            total_cost = cost + commission
            if total_cost > self.cash:
                return
            self.cash -= total_cost
            self.total_commissions += commission
            pos = self.positions.get(sym, Position(symbol=sym))
            new_qty = pos.qty + qty
            pos.avg_cost = (pos.avg_cost * pos.qty + cost) / new_qty if new_qty else pos.avg_cost
            pos.qty = new_qty
            self.positions[sym] = pos
            self.trades.append(Trade(ts, sym, 'BUY', qty, price))
        else:  # short entry not implemented in v1 (use long puts instead)
            pass

    def _exit(self, sym: str, qty: int, price: float, ts: datetime):
        # Sell long position
        pos = self.positions.get(sym)
        if not pos or pos.qty < qty:
            return
        proceeds = qty * price
        commission = self._compute_commission(qty, price, is_sell=True)
        self.cash += proceeds - commission
        self.total_commissions += commission
        pos.qty -= qty
        if pos.qty == 0:
            pos.avg_cost = 0.0
        self.positions[sym] = pos
        self.trades.append(Trade(ts, sym, 'SELL', qty, price))

    def _simulate_symbol(self, df: pd.DataFrame, sigs: pd.DataFrame):
        """
        For a single symbol:
        - Fill entries at next bar open after signal timestamp
        - Exit via ATR stop (static from signal) or time stop of 5 bars (close)
        - One open position per symbol at a time
        """
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Precompute ATR(14) for trailing stops if needed
        def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
            high = data['high']
            low = data['low']
            close = data['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            return tr.rolling(window=period, min_periods=period).mean()
        df['__atr14__'] = _atr(df, 14)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        open_trade = None  # dict with keys: entry_idx, qty, stop, side
        # Pre-index by timestamp for quick lookup of entry index
        ts_index = pd.Series(df.index.values, index=df['timestamp'])
        # Iterate over signals chronologically
        for _, sig in sigs.sort_values('timestamp').iterrows():
            # Robust next-bar lookup using searchsorted to avoid tz/naive mismatches
            try:
                sig_ts = pd.to_datetime(sig['timestamp'])
            except Exception:
                continue
            ts_series = pd.to_datetime(df['timestamp'])
            # Find insertion point to the right (strictly greater than signal ts)
            entry_idx = int(ts_series.searchsorted(sig_ts, side='right'))
            if entry_idx >= len(df):
                # No next bar available to fill
                continue
            if open_trade is not None:
                # Skip new signal until current position is closed
                continue
            # Determine entry fill price at next open with slippage
            entry_open = float(df.loc[entry_idx, 'open'])
            side = str(sig.get('side', 'long'))
            px = entry_open * (1 + (self.cfg.slippage_bps/1e4) * (1 if side=='long' else -1))

            # Position sizing (config-gated volatility targeting)
            stop_price = float(sig.get('stop_loss')) if sig.get('stop_loss') is not None else None
            trail_mult = float(sig.get('trail_atr_mult')) if sig.get('trail_atr_mult') is not None else None

            if is_sizing_enabled() and stop_price is not None:
                # Volatility-targeted sizing: qty = (risk_pct * equity) / (entry - stop)
                sizing_cfg = get_sizing_config()
                risk_pct = sizing_cfg.get('risk_per_trade_pct', 0.005)
                risk_amount = max(0.0, self.cash) * risk_pct
                risk_per_share = abs(px - stop_price)
                if risk_per_share > 0:
                    qty = int(max(1, risk_amount / risk_per_share))
                else:
                    # Fallback if stop is at entry
                    notional = max(0.0, self.cash) * 0.007
                    qty = int(max(1, notional // px))
            else:
                # Default sizing: ~0.7% of current cash
                notional = max(0.0, self.cash) * 0.007
                qty = int(max(1, notional // px))

            if qty <= 0:
                continue
            # Book entry
            self._execute(sig['symbol'], side, qty, px, df.loc[entry_idx, 'timestamp'].to_pydatetime())
            stop_price = float(sig.get('stop_loss')) if sig.get('stop_loss') is not None else None
            open_trade = {
                'entry_idx': entry_idx,
                'qty': qty,
                'stop': stop_price,
                'side': side,
            }
            # Walk forward up to time stop (allow strategy override per signal)
            time_stop = int(sig.get('time_stop_bars', 5))  # bars
            for i in range(entry_idx + 1, min(entry_idx + 1 + time_stop, len(df))):
                bar = df.loc[i]
                # ATR stop check (long only in v1)
                do_exit = False
                exit_px = float(bar['close'])
                if open_trade['side'] == 'long' and open_trade['stop'] is not None:
                    if float(bar['low']) <= float(open_trade['stop']):
                        do_exit = True
                        exit_px = float(open_trade['stop'])
                # Take-profit check (long only)
                tp = sig.get('take_profit')
                if not do_exit and tp is not None:
                    try:
                        tp_val = float(tp)
                        # If intraday high touches or exceeds TP, fill at TP
                        if float(bar['high']) >= tp_val:
                            do_exit = True
                            exit_px = tp_val
                    except Exception:
                        pass
                # Trailing stop update (long only, ATR-based)
                if not do_exit and open_trade['side'] == 'long' and trail_mult is not None and not pd.isna(df.loc[i, '__atr14__']):
                    atrv = float(df.loc[i, '__atr14__'])
                    if atrv > 0:
                        trail_stop = float(bar['close']) - trail_mult * atrv
                        if open_trade['stop'] is not None:
                            open_trade['stop'] = max(open_trade['stop'], trail_stop)
                        else:
                            open_trade['stop'] = trail_stop
                # Time stop at final bar in window
                if i == entry_idx + time_stop - 1:
                    do_exit = True
                    exit_px = float(bar['close'])
                if do_exit:
                    self._exit(sig['symbol'], open_trade['qty'], exit_px, bar['timestamp'].to_pydatetime())
                    open_trade = None
                    break
            # If loop ended without exit and trade still open, close at last bar processed
            if open_trade is not None:
                last_i = min(entry_idx + time_stop - 1, len(df)-1)
                bar = df.loc[last_i]
                self._exit(sig['symbol'], open_trade['qty'], float(bar['close']), bar['timestamp'].to_pydatetime())
                open_trade = None

    def _compute_equity_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct the daily equity curve by replaying trades chronologically.
        - Start from initial cash (not the final cash), then apply BUY/SELL cash flows
        - Track position quantities per symbol
        - Mark to market at each day's close
        Commissions are applied if enabled in the current config.
        """
        # Wide close price matrix (dates x symbols), forward-filled
        closes = (
            data.pivot_table(index='timestamp', columns='symbol', values='close')
            .sort_index()
            .ffill()
        )

        # Nothing to do without prices
        if closes.empty:
            return pd.DataFrame(columns=["equity", "returns"])\
                .assign(equity=pd.Series(dtype=float), returns=pd.Series(dtype=float))

        # Ensure trade timestamps align with price index for matching
        trade_rows = [
            Trade(
                timestamp=pd.to_datetime(t.timestamp, utc=True).tz_localize(None),
                symbol=t.symbol,
                side=t.side,
                qty=t.qty,
                price=t.price,
            )
            for t in self.trades
        ]

        # Track cash and positions over time by replaying trades
        cash = float(self.initial_cash)
        pos_qty: Dict[str, int] = {}
        eq_rows: List[Dict[str, Any]] = []

        for ts in closes.index:
            # Apply any trades that occurred at this timestamp
            for tr in (tr for tr in trade_rows if tr.timestamp == ts):
                is_sell = tr.side.upper() == 'SELL'
                # Commission model if enabled
                fee = self._compute_commission(tr.qty, tr.price, is_sell=is_sell)
                if is_sell:
                    # Reduce position and add proceeds minus fees
                    prev = pos_qty.get(tr.symbol, 0)
                    pos_qty[tr.symbol] = max(0, prev - tr.qty)
                    cash += tr.qty * tr.price - fee
                else:  # BUY
                    prev = pos_qty.get(tr.symbol, 0)
                    pos_qty[tr.symbol] = prev + tr.qty
                    cash -= tr.qty * tr.price + fee

            # Compute mark-to-market equity at close
            port_val = cash
            row = closes.loc[ts]
            for sym, qty in pos_qty.items():
                if qty <= 0:
                    continue
                px = row.get(sym)
                if pd.notna(px):
                    port_val += qty * float(px)
            eq_rows.append({"timestamp": ts, "equity": port_val})

        equity_df = pd.DataFrame(eq_rows).set_index("timestamp")
        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0.0)
        return equity_df

    def _compute_metrics(self, equity: pd.DataFrame, trades: List[Trade]) -> Dict[str, Any]:
        # Win rate & profit factor from realized BUY/SELL pairs
        wins = 0
        losses = 0
        gross_win = 0.0
        gross_loss = 0.0
        # Pair trades FIFO per symbol
        from collections import defaultdict, deque
        buys: Dict[str, deque] = defaultdict(deque)
        for tr in trades:
            if tr.side.upper() == 'BUY':
                buys[tr.symbol].append((tr.qty, tr.price))
            elif tr.side.upper() == 'SELL':
                # Match against prior buys
                remaining = tr.qty
                while remaining > 0 and buys[tr.symbol]:
                    bqty, bpx = buys[tr.symbol][0]
                    used = min(remaining, bqty)
                    pnl = (tr.price - bpx) * used
                    if pnl >= 0:
                        gross_win += pnl
                        wins += 1
                    else:
                        gross_loss += pnl  # negative
                        losses += 1
                    remaining -= used
                    bqty -= used
                    if bqty == 0:
                        buys[tr.symbol].popleft()
                    else:
                        buys[tr.symbol][0] = (bqty, bpx)
        total_pairs = wins + losses
        wr = (wins / total_pairs) if total_pairs else 0.0
        pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float('inf') if gross_win > 0 else 0.0)
        # Sharpe and MaxDD from equity curve
        rets = equity['returns'] if not equity.empty else pd.Series([0.0])
        mu = rets.mean()
        sigma = rets.std(ddof=1) if len(rets) > 1 else 0.0
        sharpe = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0
        cummax = equity['equity'].cummax() if not equity.empty else pd.Series([0.0])
        dd_series = (equity['equity'] / cummax - 1.0) if not equity.empty else pd.Series([0.0])
        maxdd = dd_series.min() if len(dd_series) else 0.0

        # Compute gross and net PnL
        final_equity = float(equity['equity'].iloc[-1]) if not equity.empty else self.cash
        net_pnl = final_equity - self.initial_cash
        gross_pnl = net_pnl + self.total_commissions  # Add back fees to get gross

        return {
            "trades": len(trades),
            "win_rate": wr,
            "profit_factor": pf,
            "sharpe": sharpe,
            "max_drawdown": float(maxdd),
            "final_equity": final_equity,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "total_fees": self.total_commissions,
        }

    def _write_outputs(self, outdir: str, equity: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        from pathlib import Path
        import json
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        # Trades CSV
        import csv
        trades_path = out / 'trade_list.csv'
        with open(trades_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','symbol','side','qty','price'])
            for t in self.trades:
                w.writerow([t.timestamp, t.symbol, t.side, t.qty, t.price])
        # Equity CSV
        equity.to_csv(out / 'equity_curve.csv')
        # Summary JSON
        with open(out / 'summary.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
