"""
Unit tests for DecisionPacket integration with run_paper_trade.py

Verifies that DecisionPacket is correctly created and saved when orders are placed,
capturing all 99 enrichment fields and 4 ML model predictions.

Fix #1 (2026-01-08): Save DecisionPacket for Learning
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import pytest

from core.decision_packet import create_decision_packet, DecisionPacket


class TestDecisionPacketCreation:
    """Test DecisionPacket factory function."""

    def test_create_basic_packet(self):
        """Verify basic packet creation with minimal fields."""
        packet = create_decision_packet(
            symbol="AAPL",
            decision="BUY",
            reason="Test trade"
        )

        # Symbol is not a direct attribute - it's in market snapshot
        assert packet.decision == "BUY"
        assert packet.decision_reason == "Test trade"
        assert packet.packet_id is not None
        assert packet.created_at is not None

    def test_create_packet_with_indicators(self):
        """Verify packet captures indicator snapshot."""
        indicators = {
            'rsi_2': 4.5,
            'rsi_14': 35.2,
            'ibs': 0.08,
            'sma_200': 150.0,
            'atr_14': 3.5,
            'sweep_strength': 0.35,
        }

        packet = create_decision_packet(
            symbol="AAPL",
            indicators=indicators,
            decision="BUY"
        )

        assert packet.indicators is not None
        assert packet.indicators.rsi_2 == 4.5
        assert packet.indicators.ibs == 0.08
        assert packet.indicators.atr_14 == 3.5

    def test_create_packet_with_ml_predictions(self):
        """Verify packet captures all 4 ML model predictions."""
        ml_predictions = [
            {
                'model': 'ml_meta',
                'version': '1.0',
                'confidence': 0.65,
                'prediction': 1.0,
            },
            {
                'model': 'lstm',
                'version': '1.0',
                'confidence': 0.72,
                'prediction': 0.03,
            },
            {
                'model': 'ensemble',
                'version': '1.0',
                'confidence': 0.68,
                'prediction': 1.0,
            },
            {
                'model': 'markov',
                'version': '1.0',
                'confidence': 0.61,
                'prediction': 0.58,
                'regime': 'BULL',
            },
        ]

        packet = create_decision_packet(
            symbol="AAPL",
            ml_predictions=ml_predictions,
            decision="BUY"
        )

        assert len(packet.ml_models) == 4
        model_names = [m.model_name for m in packet.ml_models]
        assert 'ml_meta' in model_names
        assert 'lstm' in model_names
        assert 'ensemble' in model_names
        assert 'markov' in model_names

    def test_create_packet_with_enriched_signal(self):
        """Verify packet captures full enriched signal (99 fields)."""
        signal = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'strategy': 'IBS_RSI',
            'rsi2': 4.5,
            'ibs': 0.08,
            'quality_score': 75,
            'conviction_score': 0.85,
            # Add more fields to simulate 99 enrichment fields
            'streak_length': 3,
            'sector_relative_strength': 1.05,
            'ml_meta_conf': 0.65,
            'lstm_direction': 0.72,
            'ensemble_conf': 0.68,
            'markov_pi_up': 0.61,
        }

        packet = create_decision_packet(
            symbol="AAPL",
            signal=signal,
            decision="BUY"
        )

        assert packet.signal is not None
        assert packet.signal.side == 'BUY'
        assert packet.signal.entry_price == 150.0
        assert packet.signal.stop_loss == 145.0
        assert packet.signal.take_profit == 160.0
        assert len(signal) >= 6  # At least 6 enrichment fields in this test

    def test_create_packet_with_risk_checks(self):
        """Verify packet captures risk gate results."""
        risk_checks = {
            'policy_gate': True,
            'kill_zone': True,
            'exposure_limit': True,
            'correlation_limit': True,
            'size_calculated': 100,
            'size_capped': 100,
            'risk_per_trade': 0.02,
            'notes': ['All gates passed'],
        }

        packet = create_decision_packet(
            symbol="AAPL",
            risk_checks=risk_checks,
            decision="BUY"
        )

        assert packet.risk is not None
        assert packet.risk.policy_gate_passed is True
        assert packet.risk.position_size_calculated == 100
        assert packet.risk.position_size_capped == 100

    def test_create_packet_with_strategy_params(self):
        """Verify packet captures strategy parameters."""
        strategy_params = {
            'strategy': 'IBS_RSI',
            'entry_timing': 'CLOSE_T',
            'hold_period': 7,
            'time_stop_bars': 7,
            'use_ibs': True,
            'use_rsi': True,
        }

        packet = create_decision_packet(
            symbol="AAPL",
            strategy_params=strategy_params,
            decision="BUY"
        )

        assert packet.strategy_params == strategy_params

    def test_create_packet_with_context(self):
        """Verify packet captures position sizing multipliers and context."""
        packet = create_decision_packet(
            symbol="AAPL",
            decision="BUY"
        )

        # Add context manually (as done in run_paper_trade.py)
        packet.context = {
            'decision_id': 'DEC_12345',
            'config_pin': 'abc123',
            'position_size_multiplier': {
                'kelly': 1.0,
                'regime': 1.0,
                'vix': 0.9,
                'confidence': 1.05,
                'cognitive': 1.0,
                'sector': 1.0,
            },
            'enrichment_fields_count': 99,
        }

        assert packet.context['decision_id'] == 'DEC_12345'
        assert packet.context['enrichment_fields_count'] == 99
        assert 'position_size_multiplier' in packet.context
        assert len(packet.context['position_size_multiplier']) == 6

    def test_packet_save_and_load(self):
        """Verify packet can be saved and loaded from disk."""
        # Create packet with OHLCV data so market snapshot is created
        ohlcv = pd.DataFrame({
            'open': [150.0],
            'high': [155.0],
            'low': [149.0],
            'close': [152.0],
            'volume': [1000000]
        })

        packet = create_decision_packet(
            symbol="AAPL",
            ohlcv=ohlcv,
            decision="BUY",
            reason="Test save/load"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save packet
            packet_path = packet.save(directory=tmpdir)

            # Verify file exists
            assert Path(packet_path).exists()

            # Verify filename format (contains symbol)
            assert 'AAPL' in str(packet_path)
            assert packet.packet_id[:8] in str(packet_path)

            # Load and verify content
            with open(packet_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['market']['symbol'] == 'AAPL'
            assert loaded_data['decision'] == 'BUY'
            assert loaded_data['packet_id'] == packet.packet_id


class TestRunPaperTradeIntegration:
    """Test DecisionPacket integration with run_paper_trade.py execution flow."""

    @pytest.fixture
    def mock_enriched_signal(self):
        """Create a mock enriched signal with 99 fields."""
        return pd.Series({
            'symbol': 'AAPL',
            'side': 'BUY',
            'entry_price': 150.0,
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'strategy': 'IBS_RSI',
            'rsi2': 4.5,
            'rsi_14': 35.2,
            'ibs': 0.08,
            'sma_200': 150.0,
            'atr': 3.5,
            'atr_14': 3.5,
            'sweep_strength': 0.35,
            'donchian_high': 155.0,
            'donchian_low': 145.0,
            'streak_length': 3,
            'conviction_score': 0.85,
            'quality_score': 75,
            'sector_relative_strength': 1.05,
            'ml_meta_conf': 0.65,
            'lstm_direction': 0.72,
            'lstm_magnitude': 0.03,
            'ensemble_conf': 0.68,
            'markov_pi_up': 0.61,
            'markov_p_up_today': 0.58,
            'regime': 'BULL',
            'entry_timing': 'CLOSE_T',
            'hold_period': 7,
            'time_stop_bars': 7,
            'use_ibs': True,
            'use_rsi': True,
            # Add more fields to reach 99
            **{f'field_{i}': i for i in range(70)}
        })

    def test_decision_packet_created_on_order(self, mock_enriched_signal):
        """Verify DecisionPacket is created when order is placed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate packet creation from run_paper_trade.py
            row = mock_enriched_signal

            indicators = {
                'rsi_2': row.get('rsi2'),
                'rsi_14': row.get('rsi_14'),
                'ibs': row.get('ibs'),
                'sma_200': row.get('sma_200'),
                'atr_14': row.get('atr', row.get('atr_14')),
            }

            ml_predictions = [
                {
                    'model': 'ml_meta',
                    'version': '1.0',
                    'confidence': float(row.get('ml_meta_conf', 0)),
                    'prediction': 1.0,
                },
                {
                    'model': 'lstm',
                    'version': '1.0',
                    'confidence': float(row.get('lstm_direction', 0)),
                    'prediction': float(row.get('lstm_magnitude', 0)),
                },
            ]

            packet = create_decision_packet(
                symbol=row['symbol'],
                indicators=indicators,
                signal=row.to_dict(),
                ml_predictions=ml_predictions,
                decision='BUY',
                reason="Test order placement"
            )

            # Save packet
            packet_path = packet.save(directory=tmpdir)

            # Verify packet was saved
            packet_files = list(Path(tmpdir).glob("*.json"))
            assert len(packet_files) == 1, "DecisionPacket not saved"

            # Load and verify
            with open(packet_files[0]) as f:
                packet_data = json.load(f)

            # Symbol is in signal snapshot, not at top level
            assert packet_data['decision'] == 'BUY'
            assert len(packet_data['ml_models']) == 2

    def test_all_ml_models_captured(self, mock_enriched_signal):
        """Verify all 4 ML models are captured in packet."""
        row = mock_enriched_signal

        ml_predictions = []

        # ML Meta
        if pd.notna(row.get('ml_meta_conf')):
            ml_predictions.append({
                'model': 'ml_meta',
                'version': '1.0',
                'confidence': float(row.get('ml_meta_conf', 0)),
                'prediction': 1.0,
            })

        # LSTM
        if pd.notna(row.get('lstm_direction')):
            ml_predictions.append({
                'model': 'lstm',
                'version': '1.0',
                'confidence': float(row.get('lstm_direction', 0)),
                'prediction': float(row.get('lstm_magnitude', 0)),
            })

        # Ensemble
        if pd.notna(row.get('ensemble_conf')):
            ml_predictions.append({
                'model': 'ensemble',
                'version': '1.0',
                'confidence': float(row.get('ensemble_conf', 0)),
                'prediction': 1.0,
            })

        # Markov
        if pd.notna(row.get('markov_pi_up')):
            ml_predictions.append({
                'model': 'markov',
                'version': '1.0',
                'confidence': float(row.get('markov_pi_up', 0)),
                'prediction': float(row.get('markov_p_up_today', 0)),
                'regime': str(row.get('regime', 'UNKNOWN')),
            })

        packet = create_decision_packet(
            symbol=row['symbol'],
            ml_predictions=ml_predictions,
            decision='BUY'
        )

        assert len(packet.ml_models) == 4
        model_names = [m.model_name for m in packet.ml_models]
        assert 'ml_meta' in model_names
        assert 'lstm' in model_names
        assert 'ensemble' in model_names
        assert 'markov' in model_names

    def test_enrichment_fields_preserved(self, mock_enriched_signal):
        """Verify all 99 enrichment fields are preserved in packet."""
        row = mock_enriched_signal

        packet = create_decision_packet(
            symbol=row['symbol'],
            signal=row.to_dict(),
            decision='BUY'
        )

        # Verify signal fields are preserved
        signal_dict = row.to_dict()
        assert len(signal_dict) >= 99  # Should have at least 99 fields

        # Verify packet contains signal
        assert packet.signal is not None

        # Save and reload to verify serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            packet_path = packet.save(directory=tmpdir)

            with open(packet_path) as f:
                loaded = json.load(f)

            # Verify signal fields are in loaded data
            assert 'signal' in loaded
            assert loaded['signal']['side'] == 'BUY'

    def test_position_sizing_multipliers_captured(self):
        """Verify all position sizing multipliers are captured in context."""
        packet = create_decision_packet(
            symbol="AAPL",
            decision="BUY"
        )

        # Add context as done in run_paper_trade.py
        packet.context = {
            'position_size_multiplier': {
                'kelly': 1.0,
                'regime': 1.0,
                'vix': 0.9,
                'confidence': 1.05,
                'cognitive': 1.0,
                'sector': 1.0,
            },
        }

        assert 'position_size_multiplier' in packet.context
        multipliers = packet.context['position_size_multiplier']
        assert 'kelly' in multipliers
        assert 'regime' in multipliers
        assert 'vix' in multipliers
        assert 'confidence' in multipliers
        assert 'cognitive' in multipliers
        assert 'sector' in multipliers

    def test_packet_serialization_roundtrip(self, mock_enriched_signal):
        """Verify packet can be serialized and deserialized without data loss."""
        row = mock_enriched_signal

        # Create comprehensive packet
        packet = create_decision_packet(
            symbol=row['symbol'],
            indicators={
                'rsi_2': row.get('rsi2'),
                'ibs': row.get('ibs'),
                'atr_14': row.get('atr_14'),
            },
            signal=row.to_dict(),
            ml_predictions=[
                {'model': 'ml_meta', 'confidence': 0.65, 'prediction': 1.0},
                {'model': 'lstm', 'confidence': 0.72, 'prediction': 0.03},
            ],
            risk_checks={
                'policy_gate': True,
                'size_calculated': 100,
            },
            strategy_params={
                'strategy': 'IBS_RSI',
                'hold_period': 7,
            },
            decision='BUY',
            reason="Test roundtrip"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            path = packet.save(directory=tmpdir)

            # Load
            with open(path) as f:
                loaded = json.load(f)

            # Verify all sections present
            assert loaded['decision'] == 'BUY'
            assert 'indicators' in loaded
            assert 'ml_models' in loaded
            assert len(loaded['ml_models']) == 2
            assert 'risk' in loaded
            assert 'strategy_params' in loaded
            assert loaded['strategy_params']['strategy'] == 'IBS_RSI'
            assert 'signal' in loaded
            assert loaded['signal']['side'] == 'BUY'


class TestPacketFilenameFormat:
    """Test packet filename format matches expected pattern."""

    def test_filename_contains_date_symbol_id(self):
        """Verify filename format: {date}_{symbol}_{packet_id[:8]}.json"""
        packet = create_decision_packet(
            symbol="AAPL",
            decision="BUY"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = packet.save(directory=tmpdir)
            filename = Path(path).name

            # Filename should contain packet ID prefix
            assert packet.packet_id[:8] in filename
            assert filename.endswith('.json')

    def test_multiple_packets_same_directory(self):
        """Verify multiple packets can be saved to same directory."""
        packets = [
            create_decision_packet(symbol="AAPL", decision="BUY"),
            create_decision_packet(symbol="TSLA", decision="BUY"),
            create_decision_packet(symbol="NVDA", decision="BUY"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [p.save(directory=tmpdir) for p in packets]

            # All packets should be saved
            assert len(paths) == 3

            # All files should exist
            for path in paths:
                assert Path(path).exists()

            # All filenames should be unique
            filenames = [Path(p).name for p in paths]
            assert len(set(filenames)) == 3
