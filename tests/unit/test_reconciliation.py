"""
Unit tests for reconciliation engine.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import Mock, patch

from execution.reconcile import (
    Reconciler,
    ReconciliationReport,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    BrokerPosition,
    BrokerOrder,
    OMSPosition,
    OMSOrder,
)

ET = ZoneInfo("America/New_York")


class TestBrokerPosition:
    """Tests for broker position dataclass."""

    def test_basic_position(self):
        pos = BrokerPosition(
            symbol="AAPL",
            qty=100,
            avg_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            side="long",
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 100
        assert pos.unrealized_pnl == 500.0


class TestOMSPosition:
    """Tests for OMS position dataclass."""

    def test_basic_position(self):
        pos = OMSPosition(
            symbol="AAPL",
            qty=100,
            avg_price=150.0,
            side="long",
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 100


class TestDiscrepancy:
    """Tests for discrepancy detection."""

    def test_quantity_mismatch(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            severity=DiscrepancySeverity.CRITICAL,
            symbol="AAPL",
            broker_value=100,
            oms_value=90,
            description="Quantity mismatch: broker=100, OMS=90",
            suggested_action="Reconcile position quantities",
        )
        assert disc.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        assert disc.severity == DiscrepancySeverity.CRITICAL

    def test_to_dict(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.MISSING_IN_OMS,
            severity=DiscrepancySeverity.CRITICAL,
            symbol="MSFT",
            broker_value=50,
            oms_value=None,
            description="Position exists at broker but not in OMS",
            suggested_action="Investigate order origin",
        )
        d = disc.to_dict()
        assert d["type"] == "MISSING_IN_OMS"
        assert d["symbol"] == "MSFT"


class TestReconciler:
    """Tests for reconciliation engine."""

    def test_initialization(self):
        reconciler = Reconciler()
        assert reconciler is not None

    def test_detect_quantity_mismatch(self):
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100,
                avg_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
        ]
        oms_positions = [
            OMSPosition(symbol="AAPL", qty=90, avg_price=150.0, side="long"),
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        assert len(discrepancies) >= 1
        qty_mismatches = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        ]
        assert len(qty_mismatches) == 1
        assert qty_mismatches[0].symbol == "AAPL"

    def test_detect_missing_in_oms(self):
        """Position in broker but not in OMS (formerly ORPHAN_BROKER)."""
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100,
                avg_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
            BrokerPosition(
                symbol="MSFT",
                qty=50,
                avg_price=300.0,
                current_price=310.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
        ]
        oms_positions = [
            OMSPosition(symbol="AAPL", qty=100, avg_price=150.0, side="long"),
            # MSFT missing in OMS
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        missing_in_oms = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.MISSING_IN_OMS
        ]
        assert len(missing_in_oms) == 1
        assert missing_in_oms[0].symbol == "MSFT"

    def test_detect_missing_in_broker(self):
        """Position in OMS but not in broker (formerly ORPHAN_OMS)."""
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100,
                avg_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
        ]
        oms_positions = [
            OMSPosition(symbol="AAPL", qty=100, avg_price=150.0, side="long"),
            OMSPosition(symbol="GOOG", qty=25, avg_price=100.0, side="long"),
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        missing_in_broker = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.MISSING_IN_BROKER
        ]
        assert len(missing_in_broker) == 1
        assert missing_in_broker[0].symbol == "GOOG"

    def test_detect_price_mismatch(self):
        reconciler = Reconciler(price_tolerance_pct=1.0)  # 1% tolerance

        broker_positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100,
                avg_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
        ]
        oms_positions = [
            OMSPosition(symbol="AAPL", qty=100, avg_price=145.0, side="long"),
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        price_mismatches = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.PRICE_MISMATCH
        ]
        assert len(price_mismatches) == 1

    def test_no_discrepancies(self):
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition(
                symbol="AAPL",
                qty=100,
                avg_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                unrealized_pnl=500.0,
                side="long",
            ),
        ]
        oms_positions = [
            OMSPosition(symbol="AAPL", qty=100, avg_price=150.0, side="long"),
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        assert len(discrepancies) == 0

    def test_detect_partial_fills(self):
        reconciler = Reconciler()

        broker_orders = [
            BrokerOrder(
                order_id="broker-order1",
                client_order_id="order1",
                symbol="AAPL",
                side="buy",
                qty=100,
                filled_qty=75,  # Partial fill
                limit_price=150.0,
                avg_fill_price=149.50,
                status="partially_filled",
                created_at=datetime.now(ET),
                filled_at=None,
            ),
        ]

        discrepancies = reconciler.detect_partial_fills(broker_orders)

        partial_fills = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.PARTIAL_FILL
        ]
        assert len(partial_fills) == 1


class TestReconciliationReport:
    """Tests for reconciliation report."""

    def test_report_creation(self):
        report = ReconciliationReport(
            run_id="reconcile_test_001",
            timestamp=datetime.now(ET).isoformat(),
            broker_positions=1,
            oms_positions=1,
            broker_orders_checked=5,
            oms_orders_checked=5,
            discrepancies=[],
            is_clean=True,
            summary={"total_discrepancies": 0, "critical": 0, "warning": 0, "info": 0},
        )
        assert report.is_clean is True
        assert len(report.discrepancies) == 0

    def test_report_with_discrepancies(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            severity=DiscrepancySeverity.CRITICAL,
            symbol="AAPL",
            broker_value=100,
            oms_value=90,
            description="Mismatch",
            suggested_action="Reconcile",
        )
        report = ReconciliationReport(
            run_id="reconcile_test_002",
            timestamp=datetime.now(ET).isoformat(),
            broker_positions=1,
            oms_positions=1,
            broker_orders_checked=5,
            oms_orders_checked=5,
            discrepancies=[disc],
            is_clean=False,
            summary={"total_discrepancies": 1, "critical": 1, "warning": 0, "info": 0},
        )
        assert report.is_clean is False
        assert len(report.discrepancies) == 1

    def test_report_to_dict(self):
        report = ReconciliationReport(
            run_id="reconcile_test_003",
            timestamp=datetime.now(ET).isoformat(),
            broker_positions=2,
            oms_positions=2,
            broker_orders_checked=10,
            oms_orders_checked=10,
            discrepancies=[],
            is_clean=True,
            summary={"total_discrepancies": 0, "critical": 0, "warning": 0, "info": 0},
        )
        d = report.to_dict()
        assert d["is_clean"] is True
        assert d["broker_positions"] == 2
        assert "timestamp" in d
