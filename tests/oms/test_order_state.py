import pytest
from datetime import datetime
from oms.order_state import OrderRecord, OrderStatus

class TestOrderRecord:
    @pytest.fixture
    def sample_order_record(self):
        return OrderRecord(
            decision_id="DEC-123",
            signal_id="SIG-456",
            symbol="AAPL",
            side="BUY",
            qty=100,
            limit_price=150.0,
            tif="IOC",
            order_type="IOC_LIMIT",
            idempotency_key="IDEMP-789",
            created_at=datetime.now(),
        )

    def test_order_record_initialization(self, sample_order_record):
        order = sample_order_record
        assert order.decision_id == "DEC-123"
        assert order.symbol == "AAPL"
        assert order.status == OrderStatus.PENDING
        assert order.execution_id is None
        assert order.entry_price_decision is None
        assert order.strategy_used is None

    def test_order_record_initialization_with_new_fields(self):
        now = datetime.now()
        order = OrderRecord(
            decision_id="DEC-444",
            signal_id="SIG-555",
            symbol="MSFT",
            side="SELL",
            qty=50,
            limit_price=200.0,
            tif="GTC",
            order_type="LIMIT",
            idempotency_key="IDEMP-666",
            created_at=now,
            execution_id="EXEC-001",
            entry_price_decision=200.5,
            strategy_used="momentum",
            status=OrderStatus.APPROVED,
            broker_order_id="ALPACA-ABC",
            last_update=now,
            notes="Approved by risk",
            fill_price=200.2,
            filled_qty=50,
        )
        assert order.decision_id == "DEC-444"
        assert order.symbol == "MSFT"
        assert order.status == OrderStatus.APPROVED
        assert order.execution_id == "EXEC-001"
        assert order.entry_price_decision == 200.5
        assert order.strategy_used == "momentum"
        assert order.broker_order_id == "ALPACA-ABC"
        assert order.fill_price == 200.2
        assert order.filled_qty == 50
        assert order.notes == "Approved by risk"

    def test_update_status_only_status(self, sample_order_record):
        order = sample_order_record
        old_last_update = order.last_update
        order.update_status(OrderStatus.SUBMITTED)
        assert order.status == OrderStatus.SUBMITTED
        assert order.last_update > old_last_update if old_last_update else order.last_update is not None
        assert order.notes is None
        assert order.filled_qty is None
        assert order.fill_price is None

    def test_update_status_with_message(self, sample_order_record):
        order = sample_order_record
        order.update_status(OrderStatus.REJECTED, message="Policy gate blocked")
        assert order.status == OrderStatus.REJECTED
        assert order.notes == "Policy gate blocked"

    def test_update_status_with_fill_details(self, sample_order_record):
        order = sample_order_record
        order.update_status(OrderStatus.FILLED, filled_qty=100, fill_price=150.15)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.fill_price == 150.15

    def test_update_status_all_fields(self, sample_order_record):
        order = sample_order_record
        order.update_status(
            OrderStatus.CANCELLED,
            message="Kill switch engaged",
            filled_qty=0,
            fill_price=0.0
        )
        assert order.status == OrderStatus.CANCELLED
        assert order.notes == "Kill switch engaged"
        assert order.filled_qty == 0
        assert order.fill_price == 0.0

    def test_order_status_enum_values(self):
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.FAILED.value == "FAILED" # Check new FAILED status
