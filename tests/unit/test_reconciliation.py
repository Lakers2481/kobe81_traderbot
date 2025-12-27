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
            side="long",
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
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
            side="long",
            avg_entry_price=150.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 100


class TestDiscrepancy:
    """Tests for discrepancy detection."""

    def test_quantity_mismatch(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            symbol="AAPL",
            broker_value=100,
            local_value=90,
            message="Quantity mismatch: broker=100, local=90",
            severity="high",
        )
        assert disc.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        assert disc.severity == "high"

    def test_to_dict(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.ORPHAN_BROKER,
            symbol="MSFT",
            broker_value=50,
            local_value=None,
            message="Position exists at broker but not locally",
            severity="high",
        )
        d = disc.to_dict()
        assert d["type"] == "ORPHAN_BROKER"
        assert d["symbol"] == "MSFT"


class TestReconciler:
    """Tests for reconciliation engine."""

    def test_initialization(self):
        reconciler = Reconciler()
        assert reconciler is not None

    def test_detect_quantity_mismatch(self):
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition("AAPL", 100, "long", 150.0, 155.0, 15500.0, 500.0),
        ]
        oms_positions = [
            OMSPosition("AAPL", 90, "long", 150.0),  # 10 share difference
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

    def test_detect_orphan_broker(self):
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition("AAPL", 100, "long", 150.0, 155.0, 15500.0, 500.0),
            BrokerPosition("MSFT", 50, "long", 300.0, 310.0, 15500.0, 500.0),
        ]
        oms_positions = [
            OMSPosition("AAPL", 100, "long", 150.0),
            # MSFT missing locally
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        orphans = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.ORPHAN_BROKER
        ]
        assert len(orphans) == 1
        assert orphans[0].symbol == "MSFT"

    def test_detect_orphan_local(self):
        reconciler = Reconciler()

        broker_positions = [
            BrokerPosition("AAPL", 100, "long", 150.0, 155.0, 15500.0, 500.0),
        ]
        oms_positions = [
            OMSPosition("AAPL", 100, "long", 150.0),
            OMSPosition("GOOG", 25, "long", 100.0),  # Not at broker
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        orphans = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.ORPHAN_OMS
        ]
        assert len(orphans) == 1
        assert orphans[0].symbol == "GOOG"

    def test_detect_price_mismatch(self):
        reconciler = Reconciler(price_tolerance_pct=1.0)  # 1% tolerance

        broker_positions = [
            BrokerPosition("AAPL", 100, "long", 150.0, 155.0, 15500.0, 500.0),
        ]
        oms_positions = [
            OMSPosition("AAPL", 100, "long", 145.0),  # 3.3% difference
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
            BrokerPosition("AAPL", 100, "long", 150.0, 155.0, 15500.0, 500.0),
        ]
        oms_positions = [
            OMSPosition("AAPL", 100, "long", 150.0),
        ]

        discrepancies = reconciler.detect_position_discrepancies(
            broker_positions, oms_positions
        )

        assert len(discrepancies) == 0

    def test_detect_partial_fills(self):
        reconciler = Reconciler()

        broker_orders = [
            BrokerOrder(
                order_id="order1",
                symbol="AAPL",
                side="buy",
                qty=100,
                filled_qty=75,  # Partial fill
                status="partially_filled",
                order_type="limit",
                limit_price=150.0,
                submitted_at=datetime.now(ET),
            ),
        ]
        oms_orders = [
            OMSOrder(
                order_id="order1",
                symbol="AAPL",
                side="buy",
                qty=100,
                status="submitted",
            ),
        ]

        discrepancies = reconciler.detect_order_discrepancies(broker_orders, oms_orders)

        partial_fills = [
            d for d in discrepancies if d.discrepancy_type == DiscrepancyType.PARTIAL_FILL
        ]
        assert len(partial_fills) == 1


class TestReconciliationReport:
    """Tests for reconciliation report."""

    def test_report_creation(self):
        report = ReconciliationReport(
            timestamp=datetime.now(ET),
            broker_positions=1,
            local_positions=1,
            broker_orders=5,
            local_orders=5,
            discrepancies=[],
            status="CLEAN",
        )
        assert report.status == "CLEAN"
        assert len(report.discrepancies) == 0

    def test_report_with_discrepancies(self):
        disc = Discrepancy(
            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
            symbol="AAPL",
            broker_value=100,
            local_value=90,
            message="Mismatch",
            severity="high",
        )
        report = ReconciliationReport(
            timestamp=datetime.now(ET),
            broker_positions=1,
            local_positions=1,
            broker_orders=5,
            local_orders=5,
            discrepancies=[disc],
            status="DISCREPANCIES_FOUND",
        )
        assert report.status == "DISCREPANCIES_FOUND"
        assert len(report.discrepancies) == 1

    def test_report_to_dict(self):
        report = ReconciliationReport(
            timestamp=datetime.now(ET),
            broker_positions=2,
            local_positions=2,
            broker_orders=10,
            local_orders=10,
            discrepancies=[],
            status="CLEAN",
        )
        d = report.to_dict()
        assert d["status"] == "CLEAN"
        assert d["broker_positions"] == 2
        assert "timestamp" in d
