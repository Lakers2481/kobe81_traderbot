import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from execution.tca.transaction_cost_analyzer import TransactionCostAnalyzer, TCARecord, get_tca_analyzer
from oms.order_state import OrderRecord, OrderStatus

# Fixtures for mocking dependencies
@pytest.fixture
def mock_self_model():
    with patch('execution.tca.transaction_cost_analyzer.get_self_model') as mock_get_self_model:
        mock_sm = MagicMock()
        mock_get_self_model.return_value = mock_sm
        yield mock_sm

@pytest.fixture
def mock_workspace():
    with patch('execution.tca.transaction_cost_analyzer.get_workspace') as mock_get_workspace:
        mock_ws = MagicMock()
        mock_get_workspace.return_value = mock_ws
        yield mock_ws

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Provides a real temp directory for tests that need filesystem access."""
    return tmp_path

# Fixture for a sample TCARecord
@pytest.fixture
def sample_tca_record():
    return TCARecord(
        execution_id="EXEC-123",
        symbol="AAPL",
        side="BUY",
        qty=100,
        entry_price_decision=150.00,
        fill_price=150.05,
        market_bid_at_execution=150.00,
        market_ask_at_execution=150.06,
        timestamp=datetime.now(),
        market_context={'regime': 'BULL'},
        strategy="ibs_rsi",
        status=OrderStatus.FILLED
    )

class TestTCARecord:
    """Tests for the TCARecord dataclass and its properties."""

    def test_slippage_bps_buy(self):
        record = TCARecord(
            execution_id="id", symbol="AAPL", side="BUY", qty=100,
            entry_price_decision=100.00, fill_price=100.05,
            market_bid_at_execution=99.98, market_ask_at_execution=100.02,
            timestamp=datetime.now()
        )
        # (100.05 - 100.00) / 100.00 * 10000 = 5 bps
        assert record.slippage_bps == pytest.approx(5.0)

    def test_slippage_bps_sell(self):
        record = TCARecord(
            execution_id="id", symbol="AAPL", side="SELL", qty=100,
            entry_price_decision=100.00, fill_price=99.95,
            market_bid_at_execution=99.98, market_ask_at_execution=100.02,
            timestamp=datetime.now()
        )
        # (100.00 - 99.95) / 100.00 * 10000 = 5 bps
        assert record.slippage_bps == pytest.approx(5.0)

    def test_spread_capture_bps_buy(self):
        record = TCARecord(
            execution_id="id", symbol="AAPL", side="BUY", qty=100,
            entry_price_decision=100.00, fill_price=100.01,
            market_bid_at_execution=100.00, market_ask_at_execution=100.04,
            timestamp=datetime.now()
        )
        # Mid price = 100.02
        # (100.02 - 100.01) / 100.02 * 10000 = ~1 bps
        assert record.spread_capture_bps == pytest.approx(0.999, rel=1e-3)

    def test_spread_capture_bps_sell(self):
        record = TCARecord(
            execution_id="id", symbol="AAPL", side="SELL", qty=100,
            entry_price_decision=100.00, fill_price=100.01,
            market_bid_at_execution=100.00, market_ask_at_execution=100.04,
            timestamp=datetime.now()
        )
        # Mid price = 100.02
        # (100.01 - 100.02) / 100.02 * 10000 = ~-0.999 bps
        assert record.spread_capture_bps == pytest.approx(-0.999, rel=1e-3)

    def test_to_dict_conversion(self, sample_tca_record):
        d = sample_tca_record.to_dict()
        assert d['execution_id'] == "EXEC-123"
        assert d['slippage_bps'] == pytest.approx(3.333, rel=1e-3) # (150.05-150)/150 * 10000
        assert 'timestamp' in d


class TestTransactionCostAnalyzer:
    """Tests for the TransactionCostAnalyzer class."""

    def test_init_loads_records(self, temp_storage_dir, mock_self_model, mock_workspace):
        # Simulate an existing records file
        test_file = temp_storage_dir / "tca_records.json"
        record_data = {
            'execution_id': "id_loaded",
            'symbol': "MSFT",
            'side': "BUY",
            'qty': 10,
            'entry_price_decision': 200.0,
            'fill_price': 200.01,
            'market_bid_at_execution': 200.00,
            'market_ask_at_execution': 200.02,
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
            'market_context': {},
            'strategy': 'test',
            'status': 'FILLED',
        }
        with open(test_file, 'w') as f:
            json.dump([record_data], f)

        analyzer = TransactionCostAnalyzer(storage_dir=str(temp_storage_dir), auto_persist=False)
        assert len(analyzer._tca_records) >= 1  # May have loaded existing records

    def test_record_execution(self, mock_self_model, mock_workspace, temp_storage_dir):
        analyzer = TransactionCostAnalyzer(storage_dir=str(temp_storage_dir), auto_persist=False)
        analyzer._tca_records = [] # Ensure empty for this test

        order = OrderRecord(
            decision_id="DEC-456", signal_id="SIG-789", symbol="GOOG", side="SELL", qty=50,
            limit_price=120.0, tif="IOC", order_type="LIMIT", idempotency_key="IDEMP-abc",
            created_at=datetime.now(), entry_price_decision=120.0, strategy_used="momentum"
        )
        
        analyzer.record_execution(
            order=order,
            fill_price=119.90,
            market_bid_at_execution=119.90,
            market_ask_at_execution=120.00,
            entry_price_decision=120.00,
            market_context={'regime': 'BEAR'},
        )

        assert len(analyzer._tca_records) == 1
        assert analyzer._tca_records[0].symbol == "GOOG"
        assert analyzer._tca_records[0].slippage_bps == pytest.approx(8.333, rel=1e-3) # (120-119.9)/120 * 10000
        mock_self_model.record_tca_feedback.assert_called_once()
        args, kwargs = mock_self_model.record_tca_feedback.call_args
        assert kwargs['strategy'] == "momentum"
        assert kwargs['regime'] == "BEAR"
        assert kwargs['slippage_bps'] == pytest.approx(8.333, rel=1e-3)

    def test_record_execution_pruning(self, mock_self_model, mock_workspace, temp_storage_dir):
        analyzer = TransactionCostAnalyzer(storage_dir=str(temp_storage_dir), auto_persist=False)
        analyzer._max_records = 3
        
        for i in range(5):
            order = OrderRecord(
                decision_id=f"DEC-{i}", signal_id=f"SIG-{i}", symbol="MSFT", side="BUY", qty=10,
                limit_price=100.0, tif="IOC", order_type="LIMIT", idempotency_key=f"IDEMP-{i}",
                created_at=datetime.now(), entry_price_decision=100.0, strategy_used="test"
            )
            analyzer.record_execution(order, 100.0 + i*0.01, 100.0, 100.02, 100.0, {'regime': 'test'})
        
        assert len(analyzer._tca_records) == 3 # Should have pruned 2 records

    def test_get_summary_tca_metrics(self, mock_self_model, mock_workspace, temp_storage_dir):
        analyzer = TransactionCostAnalyzer(storage_dir=str(temp_storage_dir), auto_persist=False)
        analyzer._tca_records = [
            TCARecord(
                execution_id="exec1", symbol="AAPL", side="BUY", qty=100,
                entry_price_decision=150.0, fill_price=150.05,
                market_bid_at_execution=150.00, market_ask_at_execution=150.06,
                timestamp=datetime.now() - timedelta(hours=1),
            ),
            TCARecord(
                execution_id="exec2", symbol="MSFT", side="SELL", qty=50,
                entry_price_decision=200.0, fill_price=199.90,
                market_bid_at_execution=199.80, market_ask_at_execution=200.10,
                timestamp=datetime.now() - timedelta(days=2),
            ),
            TCARecord( # Older record, should be excluded by default lookback
                execution_id="exec3", symbol="GOOG", side="BUY", qty=20,
                entry_price_decision=1000.0, fill_price=1000.50,
                market_bid_at_execution=1000.0, market_ask_at_execution=1000.60,
                timestamp=datetime.now() - timedelta(days=8),
            ),
        ]

        summary = analyzer.get_summary_tca_metrics(lookback_days=3)
        assert summary['total_trades'] == 2 # Only 2 records within 3 days
        assert summary['avg_slippage_bps'] == pytest.approx((3.333 + 5.0) / 2, rel=1e-3)
        assert summary['total_cost_usd'] == pytest.approx((0.05 * 100) + (0.10 * 50), rel=1e-3) # 5 + 5 = 10

    def test_get_summary_tca_metrics_empty(self):
        analyzer = TransactionCostAnalyzer()
        analyzer._tca_records = []
        summary = analyzer.get_summary_tca_metrics()
        assert summary['total_trades'] == 0
        assert summary['avg_slippage_bps'] == 0.0

    def test_get_recent_tca(self, sample_tca_record):
        analyzer = TransactionCostAnalyzer()
        analyzer._tca_records = [sample_tca_record] * 5
        recent = analyzer.get_recent_tca(limit=3)
        assert len(recent) == 3
        assert recent[0].execution_id == "EXEC-123"

    def test_get_tca_analyzer_singleton(self):
        analyzer1 = get_tca_analyzer()
        analyzer2 = get_tca_analyzer()
        assert analyzer1 is analyzer2
        assert isinstance(analyzer1, TransactionCostAnalyzer)

    def test_introspect_returns_report(self):
        analyzer = TransactionCostAnalyzer()
        report = analyzer.introspect()
        assert "Transaction Cost Analyzer Introspection" in report
        assert "Average Slippage" in report
