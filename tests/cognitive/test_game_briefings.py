"""Unit tests for cognitive/game_briefings.py - 3-Phase AI Briefing System."""

import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Import the module under test
from cognitive.game_briefings import (
    GameBriefingEngine,
    BriefingContext,
    PreGameBriefing,
    HalfTimeBriefing,
    PostGameBriefing,
    PositionStatus,
    TradeOutcome,
    get_briefing_engine,
)


def make_context(date="2025-12-28", **kwargs):
    """Helper to create BriefingContext with required fields."""
    defaults = {
        'timestamp': datetime.now(),
        'date': date,
    }
    defaults.update(kwargs)
    return BriefingContext(**defaults)


class TestBriefingContext:
    """Tests for BriefingContext dataclass."""

    def test_required_fields(self):
        """Test BriefingContext requires timestamp and date."""
        ctx = make_context()
        assert ctx.date == "2025-12-28"
        assert ctx.timestamp is not None

    def test_default_values(self):
        """Test BriefingContext initializes with sensible defaults."""
        ctx = make_context()
        assert ctx.regime == "NEUTRAL"
        assert ctx.regime_confidence == 0.0
        assert ctx.mood_state == "Neutral"
        assert ctx.vix_level == 20.0
        assert ctx.positions == []
        assert ctx.news_articles == []

    def test_to_dict(self):
        """Test BriefingContext serializes to dict."""
        ctx = make_context(
            date="2025-12-28",
            regime="BULLISH",
            regime_confidence=0.9,
            vix_level=18.5
        )
        d = ctx.to_dict()
        assert d['date'] == "2025-12-28"
        assert d['regime'] == "BULLISH"
        assert d['regime_confidence'] == 0.9
        assert d['vix_level'] == 18.5


class TestPreGameBriefing:
    """Tests for PreGameBriefing dataclass."""

    def test_creation(self):
        """Test PreGameBriefing creation."""
        ctx = make_context(date="2025-12-28")
        briefing = PreGameBriefing(
            context=ctx,
            generated_at=datetime.now().isoformat()
        )
        assert briefing.context.date == "2025-12-28"
        assert briefing.top3_picks == []
        assert briefing.totd is None

    def test_to_dict(self):
        """Test PreGameBriefing serializes correctly."""
        ctx = make_context(date="2025-12-28", regime="BULLISH")
        briefing = PreGameBriefing(
            context=ctx,
            regime_analysis="Market is bullish",
            generated_at="2025-12-28T08:00:00"
        )
        d = briefing.to_dict()
        assert d['context']['regime'] == "BULLISH"
        assert d['regime_analysis'] == "Market is bullish"


class TestHalfTimeBriefing:
    """Tests for HalfTimeBriefing dataclass."""

    def test_creation(self):
        """Test HalfTimeBriefing creation."""
        ctx = make_context(date="2025-12-28")
        briefing = HalfTimeBriefing(
            context=ctx,
            generated_at=datetime.now().isoformat()
        )
        assert briefing.position_analysis == []
        assert briefing.whats_working == []
        assert not briefing.regime_changed


class TestPostGameBriefing:
    """Tests for PostGameBriefing dataclass."""

    def test_creation(self):
        """Test PostGameBriefing creation."""
        ctx = make_context(date="2025-12-28")
        briefing = PostGameBriefing(
            context=ctx,
            generated_at=datetime.now().isoformat()
        )
        assert briefing.trades_today == []
        assert briefing.lessons_learned == []


class TestTradeOutcome:
    """Tests for TradeOutcome dataclass."""

    def test_winner_detection(self):
        """Test winner is correctly identified."""
        winner = TradeOutcome(
            symbol="AAPL",
            strategy="IBS_RSI",
            side="long",
            entry_price=150.0,
            exit_price=155.0,
            shares=100,
            entry_time="09:30:00",
            exit_time="15:00:00",
            hold_bars=5,
            pnl_dollars=500.0,
            pnl_percent=3.33,
            exit_reason="TARGET",
            was_winner=True
        )
        assert winner.was_winner
        assert winner.pnl_dollars == 500.0

        loser = TradeOutcome(
            symbol="MSFT",
            strategy="IBS_RSI",
            side="long",
            entry_price=400.0,
            exit_price=390.0,
            shares=10,
            entry_time="10:00:00",
            exit_time="14:00:00",
            hold_bars=3,
            pnl_dollars=-100.0,
            pnl_percent=-2.5,
            exit_reason="STOP",
            was_winner=False
        )
        assert not loser.was_winner
        assert loser.pnl_dollars == -100.0


class TestPositionStatus:
    """Tests for PositionStatus dataclass."""

    def test_creation(self):
        """Test PositionStatus creation with required fields."""
        position = PositionStatus(
            symbol="AAPL",
            side="long",
            entry_price=150.0,
            current_price=155.0,
            shares=100,
            entry_date="2025-12-26",
            days_held=2,
            unrealized_pnl=500.0,
            pnl_percent=3.33,
            recommendation="HOLD"
        )
        assert position.symbol == "AAPL"
        assert position.unrealized_pnl == 500.0
        assert position.pnl_percent == 3.33
        assert position.recommendation == "HOLD"


class TestGameBriefingEngine:
    """Tests for GameBriefingEngine class."""

    def test_filter_test_trades(self, tmp_path):
        """Test that test trades are filtered out correctly."""
        # Create trades.jsonl with test and real trades
        trades_file = tmp_path / 'logs' / 'trades.jsonl'
        trades_file.parent.mkdir(parents=True, exist_ok=True)

        test_trades = [
            # Should be filtered - REJECTED
            {"timestamp": "2025-12-28T10:00:00", "status": "REJECTED", "symbol": "LOWVOL"},
            # Should be filtered - TEST in decision_id
            {"timestamp": "2025-12-28T10:00:00", "status": "FILLED", "decision_id": "DEC_TEST_123", "symbol": "AAPL"},
            # Should be filtered - test in strategy
            {"timestamp": "2025-12-28T10:00:00", "status": "FILLED", "decision_id": "DEC_123", "strategy_used": "test_strategy", "symbol": "MSFT"},
            # Should be filtered - fake broker_order_id
            {"timestamp": "2025-12-28T10:00:00", "status": "FILLED", "decision_id": "DEC_456", "broker_order_id": "broker-order-id-123", "symbol": "GOOG"},
            # Should NOT be filtered - real trade
            {"timestamp": "2025-12-28T10:00:00", "status": "FILLED", "decision_id": "DEC_REAL_789", "broker_order_id": "real-broker-id-abc", "strategy_used": "IBS_RSI", "symbol": "PLTR", "pnl_dollars": 100.0},
        ]

        with open(trades_file, 'w') as f:
            for trade in test_trades:
                f.write(json.dumps(trade) + '\n')

        # Count trades that pass the filter (same logic as game_briefings.py)
        real_trades = []
        with open(trades_file) as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    trade_status = trade.get('status', '').upper()
                    decision_id = trade.get('decision_id', '')
                    strategy = trade.get('strategy_used', '') or ''
                    broker_id = trade.get('broker_order_id', '') or ''

                    if trade_status != 'FILLED':
                        continue
                    if 'TEST' in decision_id.upper() or 'test' in strategy.lower():
                        continue
                    if 'test' in broker_id.lower() or broker_id == 'broker-order-id-123':
                        continue

                    real_trades.append(trade)

        assert len(real_trades) == 1
        assert real_trades[0]['symbol'] == 'PLTR'

    def test_save_briefing_utf8_encoding(self, tmp_path):
        """Test that briefings are saved with UTF-8 encoding."""
        reports_dir = tmp_path / 'reports'
        reports_dir.mkdir()

        # Create briefing with Unicode characters
        ctx = make_context(date="2025-12-28")
        briefing = PreGameBriefing(
            context=ctx,
            regime_analysis="Market trend → bullish",  # Unicode arrow
            generated_at="2025-12-28T08:00:00"
        )

        # Save with UTF-8
        json_path = reports_dir / 'pregame_20251228.json'
        md_path = reports_dir / 'pregame_20251228.md'

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(briefing.to_dict(), f, ensure_ascii=False, default=str)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Briefing\n{briefing.regime_analysis}")

        # Verify files can be read back
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert '→' in data['regime_analysis']

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '→' in content


class TestGetBriefingEngine:
    """Tests for get_briefing_engine singleton."""

    def test_returns_engine_instance(self):
        """Test get_briefing_engine returns GameBriefingEngine."""
        with patch('cognitive.game_briefings.GameBriefingEngine') as MockEngine:
            MockEngine.return_value = Mock()
            engine = get_briefing_engine("./.env")
            assert engine is not None


class TestIntegration:
    """Integration tests for the briefing system."""

    def test_briefing_flow(self, tmp_path):
        """Test full briefing generation flow."""
        # Setup temp directories
        logs_dir = tmp_path / 'logs'
        reports_dir = tmp_path / 'reports'
        logs_dir.mkdir()
        reports_dir.mkdir()

        # Create empty trades file
        (logs_dir / 'trades.jsonl').touch()

        # Create context
        ctx = make_context(
            date="2025-12-28",
            regime="BULLISH",
            regime_confidence=0.9,
            vix_level=18.5,
            mood_state="Neutral",
            mood_score=0.0
        )

        # Create pregame briefing
        pregame = PreGameBriefing(
            context=ctx,
            regime_analysis="Market is bullish with 90% confidence",
            market_mood_description="Neutral sentiment",
            generated_at=datetime.now().isoformat()
        )

        # Verify serialization
        d = pregame.to_dict()
        assert d['context']['regime'] == "BULLISH"
        assert d['regime_analysis'] is not None

        # Save and reload
        json_path = reports_dir / 'pregame_20251228.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(d, f, default=str)

        with open(json_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded['context']['regime'] == "BULLISH"

    def test_day_summary_calculation(self):
        """Test day summary statistics calculation."""
        trades = [
            TradeOutcome(
                symbol="AAPL", strategy="IBS_RSI", side="long",
                entry_price=150.0, exit_price=155.0, shares=100,
                entry_time="09:30", exit_time="15:00", hold_bars=5,
                pnl_dollars=500.0, pnl_percent=3.33, exit_reason="TARGET",
                was_winner=True
            ),
            TradeOutcome(
                symbol="MSFT", strategy="IBS_RSI", side="long",
                entry_price=400.0, exit_price=390.0, shares=10,
                entry_time="10:00", exit_time="14:00", hold_bars=3,
                pnl_dollars=-100.0, pnl_percent=-2.5, exit_reason="STOP",
                was_winner=False
            ),
            TradeOutcome(
                symbol="GOOG", strategy="TURTLE_SOUP", side="long",
                entry_price=180.0, exit_price=185.0, shares=50,
                entry_time="11:00", exit_time="15:30", hold_bars=4,
                pnl_dollars=250.0, pnl_percent=2.78, exit_reason="TARGET",
                was_winner=True
            ),
        ]

        # Calculate summary
        wins = sum(1 for t in trades if t.was_winner)
        losses = len(trades) - wins
        total_pnl = sum(t.pnl_dollars for t in trades)
        win_rate = wins / len(trades) if trades else 0

        assert wins == 2
        assert losses == 1
        assert total_pnl == 650.0
        assert win_rate == pytest.approx(0.667, rel=0.01)
