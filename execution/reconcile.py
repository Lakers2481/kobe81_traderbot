"""
Automated reconciliation engine for KOBE81.

Compares broker state with OMS state to detect discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import json

ET = ZoneInfo("America/New_York")


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies."""
    MISSING_IN_BROKER = auto()    # OMS has position, broker doesn't
    MISSING_IN_OMS = auto()       # Broker has position, OMS doesn't
    QUANTITY_MISMATCH = auto()    # Both have position but qty differs
    PRICE_MISMATCH = auto()       # Significant avg price difference
    PARTIAL_FILL = auto()         # Order partially filled
    ORPHAN_ORDER = auto()         # Order in broker with no OMS record
    UNKNOWN_FILL = auto()         # Fill appeared without corresponding order


class DiscrepancySeverity(Enum):
    """Severity levels for discrepancies."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class Discrepancy:
    """A detected reconciliation discrepancy."""
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: str
    broker_value: Any
    oms_value: Any
    description: str
    suggested_action: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(ET).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.discrepancy_type.name,
            "severity": self.severity.name,
            "symbol": self.symbol,
            "broker_value": self.broker_value,
            "oms_value": self.oms_value,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "timestamp": self.timestamp,
        }


@dataclass
class BrokerPosition:
    """Position data from broker."""
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    side: str  # "long" or "short"


@dataclass
class BrokerOrder:
    """Order data from broker."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int
    limit_price: Optional[float]
    avg_fill_price: Optional[float]
    status: str
    created_at: datetime
    filled_at: Optional[datetime]


@dataclass
class OMSPosition:
    """Position data from OMS."""
    symbol: str
    qty: int
    avg_price: float
    side: str


@dataclass
class OMSOrder:
    """Order data from OMS."""
    order_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int
    limit_price: float
    status: str
    created_at: datetime


@dataclass
class ReconciliationReport:
    """Full reconciliation report."""
    run_id: str
    timestamp: str
    broker_positions: int
    oms_positions: int
    broker_orders_checked: int
    oms_orders_checked: int
    discrepancies: List[Discrepancy]
    is_clean: bool
    summary: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "broker_positions": self.broker_positions,
            "oms_positions": self.oms_positions,
            "broker_orders_checked": self.broker_orders_checked,
            "oms_orders_checked": self.oms_orders_checked,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "is_clean": self.is_clean,
            "summary": self.summary,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class Reconciler:
    """
    Automated reconciliation engine.

    Compares broker positions/orders with OMS records to detect
    discrepancies that need attention.
    """

    def __init__(
        self,
        price_tolerance_pct: float = 1.0,
        check_partial_fills: bool = True,
        check_orphans: bool = True,
    ):
        self.price_tolerance_pct = price_tolerance_pct
        self.check_partial_fills = check_partial_fills
        self.check_orphans = check_orphans
        self._run_counter = 0

    def fetch_broker_positions(self) -> List[BrokerPosition]:
        """
        Fetch current positions from broker.

        Returns list of BrokerPosition objects.
        """
        try:
            from execution.broker_alpaca import get_positions

            raw_positions = get_positions()
            positions = []

            for pos in raw_positions:
                positions.append(BrokerPosition(
                    symbol=pos.get("symbol", ""),
                    qty=int(pos.get("qty", 0)),
                    avg_price=float(pos.get("avg_entry_price", 0)),
                    current_price=float(pos.get("current_price", 0)),
                    market_value=float(pos.get("market_value", 0)),
                    unrealized_pnl=float(pos.get("unrealized_pl", 0)),
                    side="long" if int(pos.get("qty", 0)) > 0 else "short",
                ))

            return positions

        except ImportError:
            return []
        except Exception:
            return []

    def fetch_broker_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[BrokerOrder]:
        """
        Fetch orders from broker.

        Args:
            status: Order status filter ("open", "closed", "all")
            limit: Maximum orders to fetch
            after: Only fetch orders after this time

        Returns list of BrokerOrder objects.
        """
        try:
            from execution.broker_alpaca import get_orders

            raw_orders = get_orders(status=status, limit=limit, after=after)
            orders = []

            for order in raw_orders:
                created_at = order.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

                filled_at = order.get("filled_at")
                if filled_at and isinstance(filled_at, str):
                    filled_at = datetime.fromisoformat(filled_at.replace("Z", "+00:00"))

                orders.append(BrokerOrder(
                    order_id=order.get("id", ""),
                    client_order_id=order.get("client_order_id", ""),
                    symbol=order.get("symbol", ""),
                    side=order.get("side", ""),
                    qty=int(order.get("qty", 0)),
                    filled_qty=int(order.get("filled_qty", 0)),
                    limit_price=float(order.get("limit_price", 0)) if order.get("limit_price") else None,
                    avg_fill_price=float(order.get("filled_avg_price", 0)) if order.get("filled_avg_price") else None,
                    status=order.get("status", ""),
                    created_at=created_at,
                    filled_at=filled_at,
                ))

            return orders

        except ImportError:
            return []
        except Exception:
            return []

    def fetch_oms_positions(self) -> List[OMSPosition]:
        """
        Fetch positions from OMS.

        Returns list of OMSPosition objects.
        """
        try:
            from oms.order_state import load_positions

            raw_positions = load_positions()
            positions = []

            for pos in raw_positions:
                positions.append(OMSPosition(
                    symbol=pos.get("symbol", ""),
                    qty=int(pos.get("qty", 0)),
                    avg_price=float(pos.get("avg_price", 0)),
                    side="long" if int(pos.get("qty", 0)) > 0 else "short",
                ))

            return positions

        except ImportError:
            return []
        except Exception:
            return []

    def fetch_oms_orders(
        self,
        after: Optional[datetime] = None,
    ) -> List[OMSOrder]:
        """
        Fetch orders from OMS.

        Returns list of OMSOrder objects.
        """
        try:
            from oms.order_state import load_orders

            raw_orders = load_orders(after=after)
            orders = []

            for order in raw_orders:
                created_at = order.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)

                orders.append(OMSOrder(
                    order_id=order.get("order_id", ""),
                    symbol=order.get("symbol", ""),
                    side=order.get("side", ""),
                    qty=int(order.get("qty", 0)),
                    filled_qty=int(order.get("filled_qty", 0)),
                    limit_price=float(order.get("limit_price", 0)),
                    status=order.get("status", ""),
                    created_at=created_at,
                ))

            return orders

        except ImportError:
            return []
        except Exception:
            return []

    def detect_position_discrepancies(
        self,
        broker_positions: List[BrokerPosition],
        oms_positions: List[OMSPosition],
    ) -> List[Discrepancy]:
        """Detect discrepancies between broker and OMS positions."""
        discrepancies = []

        # Build lookup maps
        broker_by_symbol = {p.symbol.upper(): p for p in broker_positions}
        oms_by_symbol = {p.symbol.upper(): p for p in oms_positions}

        # Check for positions in OMS but not in broker
        for symbol, oms_pos in oms_by_symbol.items():
            if symbol not in broker_by_symbol:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_IN_BROKER,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    broker_value=None,
                    oms_value={"qty": oms_pos.qty, "avg_price": oms_pos.avg_price},
                    description=f"OMS shows {oms_pos.qty} shares but broker has no position",
                    suggested_action="Investigate: position may have been closed externally",
                ))

        # Check for positions in broker but not in OMS
        for symbol, broker_pos in broker_by_symbol.items():
            if symbol not in oms_by_symbol:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.MISSING_IN_OMS,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    broker_value={"qty": broker_pos.qty, "avg_price": broker_pos.avg_price},
                    oms_value=None,
                    description=f"Broker shows {broker_pos.qty} shares but OMS has no record",
                    suggested_action="Investigate: position may have been opened externally",
                ))

        # Check for quantity and price mismatches
        for symbol in set(broker_by_symbol.keys()) & set(oms_by_symbol.keys()):
            broker_pos = broker_by_symbol[symbol]
            oms_pos = oms_by_symbol[symbol]

            # Quantity mismatch
            if broker_pos.qty != oms_pos.qty:
                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    broker_value=broker_pos.qty,
                    oms_value=oms_pos.qty,
                    description=f"Quantity mismatch: broker={broker_pos.qty}, OMS={oms_pos.qty}",
                    suggested_action="Reconcile position quantities",
                ))

            # Price mismatch (significant difference)
            if broker_pos.avg_price > 0 and oms_pos.avg_price > 0:
                price_diff_pct = abs(broker_pos.avg_price - oms_pos.avg_price) / broker_pos.avg_price * 100
                if price_diff_pct > self.price_tolerance_pct:
                    discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.PRICE_MISMATCH,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=symbol,
                        broker_value=broker_pos.avg_price,
                        oms_value=oms_pos.avg_price,
                        description=f"Avg price differs by {price_diff_pct:.2f}%: broker=${broker_pos.avg_price:.2f}, OMS=${oms_pos.avg_price:.2f}",
                        suggested_action="Review fill prices and update OMS if needed",
                    ))

        return discrepancies

    def detect_partial_fills(
        self,
        broker_orders: List[BrokerOrder],
    ) -> List[Discrepancy]:
        """Detect orders with partial fills that need attention."""
        if not self.check_partial_fills:
            return []

        discrepancies = []

        for order in broker_orders:
            if order.status in ("partially_filled", "pending_new"):
                if order.filled_qty > 0 and order.filled_qty < order.qty:
                    discrepancies.append(Discrepancy(
                        discrepancy_type=DiscrepancyType.PARTIAL_FILL,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=order.symbol,
                        broker_value={
                            "order_id": order.order_id,
                            "requested_qty": order.qty,
                            "filled_qty": order.filled_qty,
                        },
                        oms_value=None,
                        description=f"Partial fill: {order.filled_qty}/{order.qty} shares filled",
                        suggested_action="Monitor for complete fill or cancel remaining",
                    ))

        return discrepancies

    def detect_orphans(
        self,
        broker_orders: List[BrokerOrder],
        oms_orders: List[OMSOrder],
    ) -> List[Discrepancy]:
        """Detect orders in broker with no OMS record."""
        if not self.check_orphans:
            return []

        discrepancies = []

        # Build OMS order ID set
        oms_order_ids = {o.order_id for o in oms_orders}

        for broker_order in broker_orders:
            # Check if client_order_id matches any OMS order
            if broker_order.client_order_id not in oms_order_ids:
                # Only flag filled orders as critical
                severity = (
                    DiscrepancySeverity.CRITICAL
                    if broker_order.status == "filled"
                    else DiscrepancySeverity.WARNING
                )

                discrepancies.append(Discrepancy(
                    discrepancy_type=DiscrepancyType.ORPHAN_ORDER,
                    severity=severity,
                    symbol=broker_order.symbol,
                    broker_value={
                        "order_id": broker_order.order_id,
                        "client_order_id": broker_order.client_order_id,
                        "status": broker_order.status,
                        "qty": broker_order.qty,
                    },
                    oms_value=None,
                    description=f"Order {broker_order.order_id} not found in OMS",
                    suggested_action="Investigate order origin and update OMS",
                ))

        return discrepancies

    def run_reconciliation(
        self,
        check_date: Optional[date] = None,
    ) -> ReconciliationReport:
        """
        Run full reconciliation.

        Returns ReconciliationReport with all findings.
        """
        self._run_counter += 1
        run_id = f"reconcile_{datetime.now(ET).strftime('%Y%m%d_%H%M%S')}_{self._run_counter}"

        # Fetch data from both sides
        broker_positions = self.fetch_broker_positions()
        oms_positions = self.fetch_oms_positions()
        broker_orders = self.fetch_broker_orders(status="all", limit=100)
        oms_orders = self.fetch_oms_orders()

        # Run all discrepancy checks
        all_discrepancies = []

        # Position discrepancies
        position_discrepancies = self.detect_position_discrepancies(
            broker_positions, oms_positions
        )
        all_discrepancies.extend(position_discrepancies)

        # Partial fills
        partial_fill_discrepancies = self.detect_partial_fills(broker_orders)
        all_discrepancies.extend(partial_fill_discrepancies)

        # Orphan orders
        orphan_discrepancies = self.detect_orphans(broker_orders, oms_orders)
        all_discrepancies.extend(orphan_discrepancies)

        # Build summary
        summary = {
            "total_discrepancies": len(all_discrepancies),
            "critical": sum(1 for d in all_discrepancies if d.severity == DiscrepancySeverity.CRITICAL),
            "warning": sum(1 for d in all_discrepancies if d.severity == DiscrepancySeverity.WARNING),
            "info": sum(1 for d in all_discrepancies if d.severity == DiscrepancySeverity.INFO),
            "position_issues": sum(1 for d in all_discrepancies if d.discrepancy_type in (
                DiscrepancyType.MISSING_IN_BROKER,
                DiscrepancyType.MISSING_IN_OMS,
                DiscrepancyType.QUANTITY_MISMATCH,
            )),
            "order_issues": sum(1 for d in all_discrepancies if d.discrepancy_type in (
                DiscrepancyType.PARTIAL_FILL,
                DiscrepancyType.ORPHAN_ORDER,
            )),
        }

        is_clean = summary["critical"] == 0

        return ReconciliationReport(
            run_id=run_id,
            timestamp=datetime.now(ET).isoformat(),
            broker_positions=len(broker_positions),
            oms_positions=len(oms_positions),
            broker_orders_checked=len(broker_orders),
            oms_orders_checked=len(oms_orders),
            discrepancies=all_discrepancies,
            is_clean=is_clean,
            summary=summary,
        )

    def quick_check(self) -> Tuple[bool, str]:
        """
        Quick reconciliation check.

        Returns (is_clean, summary_message).
        """
        report = self.run_reconciliation()

        if report.is_clean:
            return True, f"Reconciliation clean: {report.broker_positions} positions match"

        critical = report.summary.get("critical", 0)
        return False, f"Reconciliation issues: {critical} critical, {report.summary['total_discrepancies']} total"


# Convenience function for scripts
def run_full_reconciliation(
    output_path: Optional[Path] = None,
) -> ReconciliationReport:
    """
    Run full reconciliation and optionally save report.

    Args:
        output_path: Optional path to save JSON report

    Returns:
        ReconciliationReport
    """
    reconciler = Reconciler()
    report = reconciler.run_reconciliation()

    if output_path:
        report.save(output_path)

    return report
