"""
Daily P&L Tracker - Real-Time P&L Decomposition

Tracks P&L in real-time and decomposes it by:
- Strategy (which strategy generated the P&L)
- Position (which position contributed)
- Source (realized vs unrealized, slippage, commissions)

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class PositionPnL:
    """P&L for a single position."""
    symbol: str
    strategy: str
    entry_date: date
    entry_price: float
    current_price: float
    shares: int
    side: str                     # "long" or "short"
    realized_pnl: float           # Closed P&L
    unrealized_pnl: float         # Open P&L
    slippage_cost: float          # Entry slippage
    commission_cost: float        # Commissions
    gross_pnl: float              # Before costs
    net_pnl: float                # After costs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "entry_date": self.entry_date.isoformat(),
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "shares": self.shares,
            "side": self.side,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "slippage_cost": self.slippage_cost,
            "commission_cost": self.commission_cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
        }


@dataclass
class DailyPnL:
    """Daily P&L summary."""
    date: date
    gross_pnl: float              # Before costs
    net_pnl: float                # After costs
    realized_pnl: float           # Closed positions
    unrealized_pnl: float         # Open positions
    slippage_cost: float          # Slippage total
    commission_cost: float        # Commissions total
    by_strategy: Dict[str, float] # P&L by strategy
    by_sector: Dict[str, float]   # P&L by sector
    by_position: List[PositionPnL]  # Per-position breakdown
    trades_opened: int
    trades_closed: int
    winners: int
    losers: int
    win_rate: float
    largest_winner: float
    largest_loser: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "slippage_cost": self.slippage_cost,
            "commission_cost": self.commission_cost,
            "by_strategy": self.by_strategy,
            "by_sector": self.by_sector,
            "by_position": [p.to_dict() for p in self.by_position],
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "winners": self.winners,
            "losers": self.losers,
            "win_rate": self.win_rate,
            "largest_winner": self.largest_winner,
            "largest_loser": self.largest_loser,
        }

    def to_summary(self) -> str:
        """Generate plain English summary."""
        direction = "up" if self.net_pnl >= 0 else "down"
        emoji = "+" if self.net_pnl >= 0 else ""

        lines = [
            f"**Daily P&L: {emoji}${self.net_pnl:,.2f}** ({direction})",
            "",
            f"Gross P&L: ${self.gross_pnl:,.2f}",
            f"Costs: -${self.slippage_cost + self.commission_cost:,.2f}",
            f"  Slippage: ${self.slippage_cost:,.2f}",
            f"  Commissions: ${self.commission_cost:,.2f}",
            "",
        ]

        # Strategy breakdown
        if self.by_strategy:
            lines.append("**By Strategy:**")
            for strategy, pnl in sorted(self.by_strategy.items(), key=lambda x: x[1], reverse=True):
                emoji = "+" if pnl >= 0 else ""
                lines.append(f"  {strategy}: {emoji}${pnl:,.2f}")
            lines.append("")

        # Win/loss stats
        lines.extend([
            f"**Stats:** {self.trades_closed} trades, {self.win_rate:.0%} win rate",
            f"  Largest winner: +${self.largest_winner:,.2f}",
            f"  Largest loser: -${abs(self.largest_loser):,.2f}",
        ])

        return "\n".join(lines)


class DailyPnLTracker:
    """
    Track daily P&L in real-time.

    Features:
    - Real-time P&L updates
    - Strategy-level decomposition
    - Cost attribution
    - Historical tracking
    """

    HISTORY_FILE = Path("state/pnl/daily_history.json")
    POSITIONS_FILE = Path("state/pnl/positions.json")

    # Default commission per share (can be updated)
    COMMISSION_PER_SHARE = 0.005  # $0.005 per share

    # Sector mapping for common symbols
    SECTOR_MAP = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
        "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
        "XOM": "Energy", "CVX": "Energy", "OXY": "Energy",
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    }

    def __init__(self):
        """Initialize P&L tracker."""
        self._positions: Dict[str, Dict] = {}  # symbol -> position data
        self._daily_trades: List[Dict] = []
        self._history: Dict[str, Dict] = {}  # date -> DailyPnL

        # Ensure directories exist
        self.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

    def _load_state(self) -> None:
        """Load positions and history."""
        if self.POSITIONS_FILE.exists():
            try:
                with open(self.POSITIONS_FILE, "r") as f:
                    self._positions = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load positions: {e}")

        if self.HISTORY_FILE.exists():
            try:
                with open(self.HISTORY_FILE, "r") as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_state(self) -> None:
        """Save positions and history."""
        try:
            with open(self.POSITIONS_FILE, "w") as f:
                json.dump(self._positions, f, indent=2)

            with open(self.HISTORY_FILE, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.SECTOR_MAP.get(symbol, "Other")

    def record_entry(
        self,
        symbol: str,
        strategy: str,
        side: str,
        shares: int,
        fill_price: float,
        expected_price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a position entry.

        Args:
            symbol: Stock symbol
            strategy: Strategy that generated the signal
            side: "long" or "short"
            shares: Number of shares
            fill_price: Actual fill price
            expected_price: Expected price (for slippage calc)
            timestamp: Entry timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()

        slippage = abs(fill_price - expected_price) * shares
        commission = shares * self.COMMISSION_PER_SHARE

        self._positions[symbol] = {
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "shares": shares,
            "entry_price": fill_price,
            "entry_date": timestamp.date().isoformat(),
            "entry_slippage": slippage,
            "entry_commission": commission,
        }

        self._daily_trades.append({
            "type": "entry",
            "symbol": symbol,
            "strategy": strategy,
            "timestamp": timestamp.isoformat(),
            "slippage": slippage,
            "commission": commission,
        })

        self._save_state()
        logger.info(f"Recorded entry: {symbol} {side} {shares} @ ${fill_price:.2f}")

    def record_exit(
        self,
        symbol: str,
        fill_price: float,
        expected_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Record a position exit.

        Args:
            symbol: Stock symbol
            fill_price: Actual fill price
            expected_price: Expected price (for slippage calc)
            timestamp: Exit timestamp

        Returns:
            Net P&L for the closed position
        """
        if symbol not in self._positions:
            logger.warning(f"No position found for {symbol}")
            return None

        if timestamp is None:
            timestamp = datetime.now()

        position = self._positions[symbol]
        shares = position["shares"]
        entry_price = position["entry_price"]
        side = position["side"]

        # Calculate P&L
        if side == "long":
            gross_pnl = (fill_price - entry_price) * shares
        else:  # short
            gross_pnl = (entry_price - fill_price) * shares

        exit_slippage = abs(fill_price - expected_price) * shares
        exit_commission = shares * self.COMMISSION_PER_SHARE

        total_slippage = position["entry_slippage"] + exit_slippage
        total_commission = position["entry_commission"] + exit_commission
        net_pnl = gross_pnl - total_slippage - total_commission

        # Record trade
        self._daily_trades.append({
            "type": "exit",
            "symbol": symbol,
            "strategy": position["strategy"],
            "timestamp": timestamp.isoformat(),
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "slippage": exit_slippage,
            "commission": exit_commission,
            "total_slippage": total_slippage,
            "total_commission": total_commission,
            "is_winner": net_pnl > 0,
        })

        # Remove position
        del self._positions[symbol]
        self._save_state()

        logger.info(f"Recorded exit: {symbol} @ ${fill_price:.2f}, P&L: ${net_pnl:,.2f}")
        return net_pnl

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for unrealized P&L calculation.

        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol]["current_price"] = price

        self._save_state()

    def get_daily_pnl(self, for_date: Optional[date] = None) -> DailyPnL:
        """
        Get P&L breakdown for a specific date.

        Args:
            for_date: Date to get P&L for (defaults to today)

        Returns:
            DailyPnL object
        """
        if for_date is None:
            for_date = date.today()

        date_str = for_date.isoformat()

        # Filter trades for this date
        day_trades = [
            t for t in self._daily_trades
            if t["timestamp"].startswith(date_str)
        ]

        # Calculate metrics
        realized_pnl = sum(t.get("net_pnl", 0) for t in day_trades if t["type"] == "exit")
        slippage_cost = sum(t.get("slippage", 0) for t in day_trades)
        commission_cost = sum(t.get("commission", 0) for t in day_trades)

        # Unrealized P&L from open positions
        unrealized_pnl = 0.0
        position_pnls = []
        by_strategy: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}

        for symbol, pos in self._positions.items():
            current = pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            shares = pos["shares"]
            side = pos["side"]

            if side == "long":
                unrealized = (current - entry) * shares
            else:
                unrealized = (entry - current) * shares

            unrealized_pnl += unrealized

            # Track by strategy
            strategy = pos.get("strategy", "unknown")
            by_strategy[strategy] = by_strategy.get(strategy, 0) + unrealized

            # Track by sector
            sector = self._get_sector(symbol)
            by_sector[sector] = by_sector.get(sector, 0) + unrealized

            position_pnls.append(PositionPnL(
                symbol=symbol,
                strategy=strategy,
                entry_date=date.fromisoformat(pos["entry_date"]),
                entry_price=entry,
                current_price=current,
                shares=shares,
                side=side,
                realized_pnl=0,
                unrealized_pnl=unrealized,
                slippage_cost=pos.get("entry_slippage", 0),
                commission_cost=pos.get("entry_commission", 0),
                gross_pnl=unrealized,
                net_pnl=unrealized - pos.get("entry_slippage", 0) - pos.get("entry_commission", 0),
            ))

        # Add realized P&L to strategy breakdown
        for trade in day_trades:
            if trade["type"] == "exit":
                strategy = trade.get("strategy", "unknown")
                by_strategy[strategy] = by_strategy.get(strategy, 0) + trade.get("net_pnl", 0)

        # Calculate stats
        exits = [t for t in day_trades if t["type"] == "exit"]
        winners = [t for t in exits if t.get("is_winner", False)]
        losers = [t for t in exits if not t.get("is_winner", True)]

        pnls = [t.get("net_pnl", 0) for t in exits]
        largest_winner = max(pnls) if pnls else 0
        largest_loser = min(pnls) if pnls else 0

        gross_pnl = realized_pnl + unrealized_pnl + slippage_cost + commission_cost
        net_pnl = realized_pnl + unrealized_pnl

        return DailyPnL(
            date=for_date,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            slippage_cost=slippage_cost,
            commission_cost=commission_cost,
            by_strategy=by_strategy,
            by_sector=by_sector,
            by_position=position_pnls,
            trades_opened=len([t for t in day_trades if t["type"] == "entry"]),
            trades_closed=len(exits),
            winners=len(winners),
            losers=len(losers),
            win_rate=len(winners) / len(exits) if exits else 0.0,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for dashboard."""
        daily = self.get_daily_pnl()

        return {
            "net_pnl": daily.net_pnl,
            "realized_pnl": daily.realized_pnl,
            "unrealized_pnl": daily.unrealized_pnl,
            "total_costs": daily.slippage_cost + daily.commission_cost,
            "open_positions": len(self._positions),
            "trades_today": daily.trades_closed,
            "win_rate": daily.win_rate,
        }


# Singleton
_tracker: Optional[DailyPnLTracker] = None


def get_daily_pnl_tracker() -> DailyPnLTracker:
    """Get or create singleton tracker."""
    global _tracker
    if _tracker is None:
        _tracker = DailyPnLTracker()
    return _tracker


if __name__ == "__main__":
    # Demo
    tracker = DailyPnLTracker()

    print("=== Daily P&L Tracker Demo ===\n")

    # Simulate trades
    tracker.record_entry("AAPL", "IBS_RSI", "long", 100, 175.10, 175.00)
    tracker.record_entry("MSFT", "TurtleSoup", "long", 50, 380.25, 380.00)

    # Update prices
    tracker.update_prices({"AAPL": 177.00, "MSFT": 378.00})

    # Get daily P&L
    daily = tracker.get_daily_pnl()
    print(daily.to_summary())

    # Simulate exit
    pnl = tracker.record_exit("AAPL", 177.00, 177.05)
    print(f"\nAPPL Exit P&L: ${pnl:,.2f}")

    # Updated daily
    print("\n--- After Exit ---")
    daily = tracker.get_daily_pnl()
    print(daily.to_summary())
