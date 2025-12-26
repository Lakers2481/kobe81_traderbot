"""
Tax Lot Accounting - IRS-Compliant Cost Basis Tracking.

Provides tax lot management for trading:
- Multiple cost basis methods (FIFO, LIFO, Specific ID)
- Wash sale rule detection (61-day window)
- Short-term vs long-term holding period tracking
- Tax lot ledger with full audit trail
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CostBasisMethod(Enum):
    """Cost basis calculation methods."""
    FIFO = "FIFO"           # First In, First Out
    LIFO = "LIFO"           # Last In, First Out
    SPECIFIC_ID = "SPECIFIC_ID"  # Specific lot identification
    AVERAGE = "AVERAGE"     # Average cost (mutual funds only)


@dataclass
class TaxLot:
    """Individual tax lot record."""
    lot_id: str
    symbol: str
    quantity: int
    cost_basis: float        # Total cost basis
    cost_per_share: float    # Per-share cost
    purchase_date: datetime
    purchase_order_id: str = ""

    # Sale information (if closed)
    is_open: bool = True
    sale_date: Optional[datetime] = None
    sale_price: Optional[float] = None
    sale_proceeds: Optional[float] = None
    sale_order_id: str = ""

    # Computed fields
    holding_days: int = 0
    is_long_term: bool = False  # > 365 days
    realized_gain_loss: float = 0.0
    wash_sale_disallowed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lot_id": self.lot_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "cost_basis": round(self.cost_basis, 2),
            "cost_per_share": round(self.cost_per_share, 4),
            "purchase_date": self.purchase_date.isoformat(),
            "purchase_order_id": self.purchase_order_id,
            "is_open": self.is_open,
            "sale_date": self.sale_date.isoformat() if self.sale_date else None,
            "sale_price": round(self.sale_price, 4) if self.sale_price else None,
            "sale_proceeds": round(self.sale_proceeds, 2) if self.sale_proceeds else None,
            "sale_order_id": self.sale_order_id,
            "holding_days": self.holding_days,
            "is_long_term": self.is_long_term,
            "realized_gain_loss": round(self.realized_gain_loss, 2),
            "wash_sale_disallowed": round(self.wash_sale_disallowed, 2),
        }


@dataclass
class WashSaleResult:
    """Result of wash sale check."""
    is_wash_sale: bool
    disallowed_loss: float = 0.0
    adjustment_lot_id: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_wash_sale": self.is_wash_sale,
            "disallowed_loss": round(self.disallowed_loss, 2),
            "adjustment_lot_id": self.adjustment_lot_id,
            "reason": self.reason,
        }


class TaxLotAccountant:
    """
    Tax Lot Accountant for IRS-compliant tracking.

    Manages tax lots, tracks cost basis, and detects wash sales.
    """

    WASH_SALE_WINDOW_DAYS = 61  # 30 days before + sale day + 30 days after

    def __init__(
        self,
        cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO,
        ledger_path: Optional[Path] = None
    ):
        """
        Initialize tax lot accountant.

        Args:
            cost_basis_method: Default cost basis method
            ledger_path: Path to persist ledger (optional)
        """
        self.cost_basis_method = cost_basis_method
        self.ledger_path = ledger_path
        self.lots: Dict[str, List[TaxLot]] = {}  # symbol -> list of lots
        self._lot_counter = 0

        if ledger_path and ledger_path.exists():
            self._load_ledger()

    def _generate_lot_id(self) -> str:
        """Generate unique lot ID."""
        self._lot_counter += 1
        return f"LOT_{self._lot_counter:06d}"

    def record_purchase(
        self,
        symbol: str,
        quantity: int,
        price_per_share: float,
        purchase_date: Optional[datetime] = None,
        order_id: str = ""
    ) -> TaxLot:
        """
        Record a purchase and create tax lot.

        Args:
            symbol: Stock symbol
            quantity: Shares purchased
            price_per_share: Price per share
            purchase_date: Date of purchase (defaults to now)
            order_id: Order ID for audit trail

        Returns:
            Created TaxLot
        """
        if symbol not in self.lots:
            self.lots[symbol] = []

        lot = TaxLot(
            lot_id=self._generate_lot_id(),
            symbol=symbol,
            quantity=quantity,
            cost_basis=quantity * price_per_share,
            cost_per_share=price_per_share,
            purchase_date=purchase_date or datetime.now(),
            purchase_order_id=order_id,
        )

        self.lots[symbol].append(lot)
        self._save_ledger()

        logger.info(f"Tax lot created: {lot.lot_id} - {symbol} x{quantity} @ ${price_per_share:.2f}")
        return lot

    def record_sale(
        self,
        symbol: str,
        quantity: int,
        sale_price: float,
        sale_date: Optional[datetime] = None,
        order_id: str = "",
        specific_lot_id: Optional[str] = None
    ) -> List[TaxLot]:
        """
        Record a sale and close tax lots.

        Args:
            symbol: Stock symbol
            quantity: Shares sold
            sale_price: Price per share
            sale_date: Date of sale (defaults to now)
            order_id: Order ID for audit trail
            specific_lot_id: Specific lot to sell (for SPECIFIC_ID method)

        Returns:
            List of closed TaxLots
        """
        if symbol not in self.lots:
            logger.warning(f"No lots found for {symbol}")
            return []

        sale_date = sale_date or datetime.now()
        remaining = quantity
        closed_lots = []

        # Get lots to close based on method
        lots_to_close = self._select_lots(symbol, quantity, specific_lot_id)

        for lot in lots_to_close:
            if remaining <= 0:
                break

            shares_from_lot = min(lot.quantity, remaining)
            remaining -= shares_from_lot

            # Calculate holding period
            lot.holding_days = (sale_date - lot.purchase_date).days
            lot.is_long_term = lot.holding_days > 365

            # Calculate gain/loss
            lot.sale_date = sale_date
            lot.sale_price = sale_price
            lot.sale_proceeds = shares_from_lot * sale_price
            lot.sale_order_id = order_id
            lot.realized_gain_loss = lot.sale_proceeds - (shares_from_lot * lot.cost_per_share)

            # Check for wash sale (if loss)
            if lot.realized_gain_loss < 0:
                wash_result = self._check_wash_sale(symbol, sale_date, lot)
                if wash_result.is_wash_sale:
                    lot.wash_sale_disallowed = wash_result.disallowed_loss

            # Handle partial lot sale
            if shares_from_lot < lot.quantity:
                # Split the lot
                new_lot = TaxLot(
                    lot_id=self._generate_lot_id(),
                    symbol=symbol,
                    quantity=lot.quantity - shares_from_lot,
                    cost_basis=(lot.quantity - shares_from_lot) * lot.cost_per_share,
                    cost_per_share=lot.cost_per_share,
                    purchase_date=lot.purchase_date,
                    purchase_order_id=lot.purchase_order_id,
                )
                self.lots[symbol].append(new_lot)
                lot.quantity = shares_from_lot
                lot.cost_basis = shares_from_lot * lot.cost_per_share

            lot.is_open = False
            closed_lots.append(lot)

        self._save_ledger()
        return closed_lots

    def _select_lots(
        self,
        symbol: str,
        quantity: int,
        specific_lot_id: Optional[str] = None
    ) -> List[TaxLot]:
        """Select lots to close based on cost basis method."""
        open_lots = [l for l in self.lots.get(symbol, []) if l.is_open]

        if not open_lots:
            return []

        if specific_lot_id:
            return [l for l in open_lots if l.lot_id == specific_lot_id]

        if self.cost_basis_method == CostBasisMethod.FIFO:
            return sorted(open_lots, key=lambda l: l.purchase_date)

        elif self.cost_basis_method == CostBasisMethod.LIFO:
            return sorted(open_lots, key=lambda l: l.purchase_date, reverse=True)

        else:
            return open_lots

    def _check_wash_sale(
        self,
        symbol: str,
        sale_date: datetime,
        sold_lot: TaxLot
    ) -> WashSaleResult:
        """
        Check if sale triggers wash sale rule.

        Wash sale occurs when:
        - You sell at a loss
        - You buy substantially identical stock within 30 days before or after
        """
        # Only applies to losses
        if sold_lot.realized_gain_loss >= 0:
            return WashSaleResult(is_wash_sale=False)

        window_start = sale_date - timedelta(days=30)
        window_end = sale_date + timedelta(days=30)

        # Check for purchases in wash sale window
        for lot in self.lots.get(symbol, []):
            if lot.lot_id == sold_lot.lot_id:
                continue

            if window_start <= lot.purchase_date <= window_end:
                # Wash sale detected
                disallowed = abs(sold_lot.realized_gain_loss)
                return WashSaleResult(
                    is_wash_sale=True,
                    disallowed_loss=disallowed,
                    adjustment_lot_id=lot.lot_id,
                    reason=f"Purchase of {symbol} on {lot.purchase_date.strftime('%Y-%m-%d')} "
                           f"within 30-day window of loss sale"
                )

        return WashSaleResult(is_wash_sale=False)

    def get_open_lots(self, symbol: Optional[str] = None) -> List[TaxLot]:
        """Get all open lots, optionally filtered by symbol."""
        if symbol:
            return [l for l in self.lots.get(symbol, []) if l.is_open]

        all_open = []
        for lots in self.lots.values():
            all_open.extend([l for l in lots if l.is_open])
        return all_open

    def get_closed_lots(
        self,
        symbol: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[TaxLot]:
        """Get closed lots, optionally filtered by symbol and year."""
        closed = []

        symbols = [symbol] if symbol else list(self.lots.keys())

        for sym in symbols:
            for lot in self.lots.get(sym, []):
                if not lot.is_open:
                    if year is None or (lot.sale_date and lot.sale_date.year == year):
                        closed.append(lot)

        return closed

    def get_realized_gains_summary(self, year: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of realized gains/losses."""
        closed = self.get_closed_lots(year=year)

        short_term_gains = sum(l.realized_gain_loss for l in closed if not l.is_long_term and l.realized_gain_loss > 0)
        short_term_losses = sum(l.realized_gain_loss for l in closed if not l.is_long_term and l.realized_gain_loss < 0)
        long_term_gains = sum(l.realized_gain_loss for l in closed if l.is_long_term and l.realized_gain_loss > 0)
        long_term_losses = sum(l.realized_gain_loss for l in closed if l.is_long_term and l.realized_gain_loss < 0)
        wash_sale_disallowed = sum(l.wash_sale_disallowed for l in closed)

        return {
            "year": year or "all",
            "total_lots_closed": len(closed),
            "short_term": {
                "gains": round(short_term_gains, 2),
                "losses": round(short_term_losses, 2),
                "net": round(short_term_gains + short_term_losses, 2),
            },
            "long_term": {
                "gains": round(long_term_gains, 2),
                "losses": round(long_term_losses, 2),
                "net": round(long_term_gains + long_term_losses, 2),
            },
            "total_net": round(short_term_gains + short_term_losses + long_term_gains + long_term_losses, 2),
            "wash_sale_disallowed": round(wash_sale_disallowed, 2),
            "adjusted_net": round(
                short_term_gains + short_term_losses + long_term_gains + long_term_losses + wash_sale_disallowed,
                2
            ),
        }

    def get_cost_basis(self, symbol: str) -> float:
        """Get total cost basis for open lots of a symbol."""
        open_lots = self.get_open_lots(symbol)
        return sum(l.cost_basis for l in open_lots)

    def _save_ledger(self):
        """Save ledger to file."""
        if not self.ledger_path:
            return

        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "cost_basis_method": self.cost_basis_method.value,
                "lot_counter": self._lot_counter,
                "lots": {
                    symbol: [lot.to_dict() for lot in lots]
                    for symbol, lots in self.lots.items()
                }
            }
            with open(self.ledger_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ledger: {e}")

    def _load_ledger(self):
        """Load ledger from file."""
        try:
            with open(self.ledger_path) as f:
                data = json.load(f)

            self.cost_basis_method = CostBasisMethod(data.get("cost_basis_method", "FIFO"))
            self._lot_counter = data.get("lot_counter", 0)

            for symbol, lot_dicts in data.get("lots", {}).items():
                self.lots[symbol] = []
                for d in lot_dicts:
                    lot = TaxLot(
                        lot_id=d["lot_id"],
                        symbol=d["symbol"],
                        quantity=d["quantity"],
                        cost_basis=d["cost_basis"],
                        cost_per_share=d["cost_per_share"],
                        purchase_date=datetime.fromisoformat(d["purchase_date"]),
                        purchase_order_id=d.get("purchase_order_id", ""),
                        is_open=d.get("is_open", True),
                        sale_date=datetime.fromisoformat(d["sale_date"]) if d.get("sale_date") else None,
                        sale_price=d.get("sale_price"),
                        sale_proceeds=d.get("sale_proceeds"),
                        sale_order_id=d.get("sale_order_id", ""),
                        holding_days=d.get("holding_days", 0),
                        is_long_term=d.get("is_long_term", False),
                        realized_gain_loss=d.get("realized_gain_loss", 0),
                        wash_sale_disallowed=d.get("wash_sale_disallowed", 0),
                    )
                    self.lots[symbol].append(lot)

            logger.info(f"Loaded {sum(len(l) for l in self.lots.values())} tax lots from ledger")

        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")


# Singleton instance
_tax_accountant: Optional[TaxLotAccountant] = None


def get_tax_accountant(ledger_path: Optional[Path] = None) -> TaxLotAccountant:
    """Get or create the global tax accountant instance."""
    global _tax_accountant
    if _tax_accountant is None:
        default_path = Path(__file__).parent.parent / "state" / "tax_ledger.json"
        _tax_accountant = TaxLotAccountant(ledger_path=ledger_path or default_path)
    return _tax_accountant
