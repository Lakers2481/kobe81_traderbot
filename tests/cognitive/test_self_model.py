"""
Comprehensive Unit Tests for SelfModel
========================================

Tests the AI's self-awareness system that tracks its own
capabilities, limitations, and calibration.

Run: python -m pytest tests/cognitive/test_self_model.py -v
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path


class TestSelfModelInitialization:
    """Tests for SelfModel initialization."""

    def test_default_initialization(self):
        """Test initialization with default settings."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            assert model._performance == {}
            assert model._limitations == {}
            assert model._calibration == {}
            assert not model.auto_persist

    def test_initialization_creates_state_dir(self):
        """Test that state directory is created if it doesn't exist."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, 'new_state_dir')
            SelfModel(state_dir=state_dir, auto_persist=False)

            assert os.path.exists(state_dir)


class TestCapabilityEnum:
    """Tests for the Capability enumeration."""

    def test_capability_values(self):
        """Test all capability levels are accessible."""
        from cognitive.self_model import Capability

        assert Capability.EXCELLENT.value == 'excellent'
        assert Capability.GOOD.value == 'good'
        assert Capability.ADEQUATE.value == 'adequate'
        assert Capability.WEAK.value == 'weak'
        assert Capability.POOR.value == 'poor'
        assert Capability.UNKNOWN.value == 'unknown'


class TestStrategyPerformance:
    """Tests for the StrategyPerformance dataclass."""

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        from cognitive.self_model import StrategyPerformance

        perf = StrategyPerformance(
            strategy='ibs_rsi',
            regime='BULL',
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
        )

        assert perf.win_rate == 0.65

    def test_win_rate_zero_trades(self):
        """Test win rate with no trades."""
        from cognitive.self_model import StrategyPerformance

        perf = StrategyPerformance(strategy='test', regime='test')
        assert perf.win_rate == 0.0

    def test_capability_classification(self):
        """Test capability is assigned correctly based on win rate."""
        from cognitive.self_model import StrategyPerformance, Capability

        # Test EXCELLENT (>65%)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=20, winning_trades=15, losing_trades=5
        )
        assert perf.capability == Capability.EXCELLENT

        # Test GOOD (55-65%)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=20, winning_trades=12, losing_trades=8
        )
        assert perf.capability == Capability.GOOD

        # Test ADEQUATE (45-55%)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=20, winning_trades=10, losing_trades=10
        )
        assert perf.capability == Capability.ADEQUATE

        # Test WEAK (35-45%)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=20, winning_trades=8, losing_trades=12
        )
        assert perf.capability == Capability.WEAK

        # Test POOR (<35%)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=20, winning_trades=5, losing_trades=15
        )
        assert perf.capability == Capability.POOR

        # Test UNKNOWN (too few trades)
        perf = StrategyPerformance(
            strategy='test', regime='test',
            total_trades=5, winning_trades=4, losing_trades=1
        )
        assert perf.capability == Capability.UNKNOWN

    def test_to_dict(self):
        """Test StrategyPerformance serialization."""
        from cognitive.self_model import StrategyPerformance

        perf = StrategyPerformance(
            strategy='turtle_soup',
            regime='BEAR',
            total_trades=50,
            winning_trades=35,
            total_pnl=2500.0,
        )

        d = perf.to_dict()

        assert d['strategy'] == 'turtle_soup'
        assert d['regime'] == 'BEAR'
        assert d['win_rate'] == 0.70
        assert d['capability'] == 'excellent'


class TestLimitation:
    """Tests for the Limitation dataclass."""

    def test_limitation_creation(self):
        """Test creating a Limitation object."""
        from cognitive.self_model import Limitation

        lim = Limitation(
            context='high_volatility',
            description='Poor exit timing',
            severity='moderate',
        )

        assert lim.context == 'high_volatility'
        assert lim.description == 'Poor exit timing'
        assert lim.severity == 'moderate'
        assert lim.occurrences == 1

    def test_limitation_to_dict(self):
        """Test Limitation serialization."""
        from cognitive.self_model import Limitation

        lim = Limitation(
            context='earnings',
            description='Unexpected gap risk',
            severity='severe',
            mitigation='Check earnings calendar',
        )

        d = lim.to_dict()

        assert d['context'] == 'earnings'
        assert d['severity'] == 'severe'
        assert d['mitigation'] == 'Check earnings calendar'


class TestConfidenceCalibration:
    """Tests for the ConfidenceCalibration dataclass."""

    def test_actual_accuracy(self):
        """Test actual accuracy calculation."""
        from cognitive.self_model import ConfidenceCalibration

        cal = ConfidenceCalibration(
            confidence_bucket='70-80%',
            predictions=100,
            correct=75,
        )

        assert cal.actual_accuracy == 0.75

    def test_calibration_error(self):
        """Test calibration error calculation."""
        from cognitive.self_model import ConfidenceCalibration

        # Well-calibrated: 70-80% bucket, 75% actual
        cal = ConfidenceCalibration(
            confidence_bucket='70-80%',
            predictions=100,
            correct=75,
        )
        assert cal.calibration_error == 0.0  # Expected 75%, actual 75%

        # Overconfident: 70-80% bucket, 50% actual
        cal = ConfidenceCalibration(
            confidence_bucket='70-80%',
            predictions=100,
            correct=50,
        )
        assert cal.calibration_error == 0.25  # Expected 75%, actual 50%


class TestRecordTradeOutcome:
    """Tests for recording trade outcomes."""

    def test_record_winning_trade(self):
        """Test recording a winning trade."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            model.record_trade_outcome(
                strategy='ibs_rsi',
                regime='BULL',
                won=True,
                pnl=150.0,
                r_multiple=1.5,
            )

            perf = model.get_performance('ibs_rsi', 'BULL')
            assert perf is not None
            assert perf.total_trades == 1
            assert perf.winning_trades == 1
            assert perf.total_pnl == 150.0

    def test_record_losing_trade(self):
        """Test recording a losing trade."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            model.record_trade_outcome(
                strategy='turtle_soup',
                regime='CHOPPY',
                won=False,
                pnl=-75.0,
            )

            perf = model.get_performance('turtle_soup', 'CHOPPY')
            assert perf.total_trades == 1
            assert perf.losing_trades == 1
            assert perf.total_pnl == -75.0

    def test_record_multiple_trades(self):
        """Test recording multiple trades updates statistics."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Record 10 winning trades
            for i in range(10):
                model.record_trade_outcome(
                    strategy='test',
                    regime='BULL',
                    won=True,
                    pnl=100.0,
                )

            # Record 5 losing trades
            for i in range(5):
                model.record_trade_outcome(
                    strategy='test',
                    regime='BULL',
                    won=False,
                    pnl=-50.0,
                )

            perf = model.get_performance('test', 'BULL')
            assert perf.total_trades == 15
            assert perf.winning_trades == 10
            assert perf.losing_trades == 5
            assert abs(perf.win_rate - 0.6667) < 0.01


class TestRecordLimitation:
    """Tests for recording limitations."""

    def test_record_new_limitation(self):
        """Test recording a new limitation."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            model.record_limitation(
                context='high_vix',
                description='Poor performance when VIX > 30',
                severity='moderate',
            )

            lims = model.known_limitations()
            assert len(lims) == 1
            assert lims[0].context == 'high_vix'

    def test_record_duplicate_limitation(self):
        """Test recording the same limitation multiple times."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            model.record_limitation(
                context='earnings',
                description='Gap risk',
                severity='severe',
            )
            model.record_limitation(
                context='earnings',
                description='Gap risk',
                severity='severe',
            )

            lims = model.known_limitations()
            # Should be one limitation with occurrence count of 2
            matching = [l for l in lims if l.context == 'earnings']
            assert len(matching) == 1
            assert matching[0].occurrences == 2


class TestCalibrationTracking:
    """Tests for confidence calibration tracking."""

    def test_record_prediction(self):
        """Test recording a prediction for calibration."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            model.record_prediction(confidence=0.75, correct=True)
            model.record_prediction(confidence=0.75, correct=False)

            # Should have created a calibration bucket
            assert '70-80%' in model._calibration

    def test_is_well_calibrated(self):
        """Test calibration assessment."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # With no data, should be considered calibrated
            assert model.is_well_calibrated()

    def test_get_calibration_error(self):
        """Test getting the calibration error."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # With no data, should return 0
            assert model.get_calibration_error() == 0.0


class TestShouldStandDown:
    """Tests for stand-down recommendations."""

    def test_stand_down_poor_capability(self):
        """Test stand-down is recommended for poor capability."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Record many losing trades to get POOR capability
            for i in range(20):
                model.record_trade_outcome(
                    strategy='bad_strategy',
                    regime='CHOPPY',
                    won=i < 4,  # Only 4 wins out of 20 = 20% win rate
                    pnl=100 if i < 4 else -50,
                )

            should_stand, reason = model.should_stand_down('bad_strategy', 'CHOPPY')
            assert should_stand
            assert 'POOR' in reason

    def test_no_stand_down_good_capability(self):
        """Test no stand-down for good capability."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Record many winning trades
            for i in range(20):
                model.record_trade_outcome(
                    strategy='good_strategy',
                    regime='BULL',
                    won=i < 14,  # 14 wins out of 20 = 70% win rate
                    pnl=100 if i < 14 else -50,
                )

            should_stand, reason = model.should_stand_down('good_strategy', 'BULL')
            assert not should_stand


class TestStrengthsAndWeaknesses:
    """Tests for identifying strengths and weaknesses."""

    def test_get_strengths(self):
        """Test getting list of strengths."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Create a strength
            for i in range(15):
                model.record_trade_outcome(
                    strategy='winner',
                    regime='BULL',
                    won=i < 12,  # 80% win rate
                    pnl=100,
                )

            strengths = model.get_strengths()
            assert len(strengths) > 0
            assert any('winner' in s for s in strengths)

    def test_get_weaknesses(self):
        """Test getting list of weaknesses."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Create a weakness
            for i in range(15):
                model.record_trade_outcome(
                    strategy='loser',
                    regime='BEAR',
                    won=i < 4,  # ~27% win rate
                    pnl=-50,
                )

            weaknesses = model.get_weaknesses()
            assert len(weaknesses) > 0
            assert any('loser' in w for w in weaknesses)


class TestSelfDescription:
    """Tests for self-description generation."""

    def test_get_self_description(self):
        """Test generating a self-description."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            description = model.get_self_description()

            assert "trading agent" in description.lower()
            assert isinstance(description, str)

    def test_self_description_with_data(self):
        """Test self-description after recording trades."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Record some trades
            for i in range(15):
                model.record_trade_outcome(
                    strategy='test',
                    regime='BULL',
                    won=i < 10,
                    pnl=100,
                )

            description = model.get_self_description()

            assert "15 trades" in description


class TestPersistence:
    """Tests for state persistence."""

    def test_save_and_load_state(self):
        """Test saving and loading state."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model and add data
            model1 = SelfModel(state_dir=tmpdir, auto_persist=True)
            model1.record_trade_outcome(
                strategy='persist_test',
                regime='BULL',
                won=True,
                pnl=100.0,
            )

            # Create new model instance, should load saved state
            model2 = SelfModel(state_dir=tmpdir, auto_persist=False)

            # Check that the data was loaded
            perf = model2.get_performance('persist_test', 'BULL')
            assert perf is not None
            assert perf.total_trades == 1


class TestGetStatus:
    """Tests for the get_status method."""

    def test_get_status(self):
        """Test getting status dictionary."""
        from cognitive.self_model import SelfModel

        with tempfile.TemporaryDirectory() as tmpdir:
            model = SelfModel(state_dir=tmpdir, auto_persist=False)

            status = model.get_status()

            assert 'performance_records' in status
            assert 'known_limitations' in status
            assert 'calibration_error' in status
            assert 'is_well_calibrated' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
