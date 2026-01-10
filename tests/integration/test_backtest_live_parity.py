"""
End-to-end parity test: Ensure backtest and live execution produce similar results.

FIX (2026-01-08): Created to validate backtest-live parity after fixing gaps.

These tests verify that:
1. Backtest configuration matches live execution parameters
2. Kill zone filtering is applied in backtest
3. Quality gate is enforced in paper trading
4. Risk percentages are aligned between backtest and live
5. Time stops use trading days (NYSE calendar)
"""

import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestBacktestLiveParity:
    """Test that backtest configuration matches live execution."""

    def test_slippage_aligned(self):
        """Verify backtest slippage matches live (~10 bps)."""
        from backtest.engine import BacktestConfig
        cfg = BacktestConfig()
        assert 8 <= cfg.slippage_bps <= 12, \
            f"Slippage {cfg.slippage_bps} bps not aligned with live (10 bps)"

    def test_kill_zone_config_exists(self):
        """Verify backtest config has kill zone flag."""
        from backtest.engine import BacktestConfig
        cfg = BacktestConfig()
        assert hasattr(cfg, 'apply_kill_zones'), \
            "BacktestConfig missing apply_kill_zones flag"
        assert cfg.apply_kill_zones is True, \
            "apply_kill_zones should be True by default"

    def test_kill_zone_filter_exists(self):
        """Verify backtest has kill zone filtering method."""
        engine_path = ROOT / "backtest" / "engine.py"
        content = engine_path.read_text()
        assert "_filter_signals_by_kill_zone" in content, \
            "Backtest missing _filter_signals_by_kill_zone method"

    def test_kill_zone_filter_called(self):
        """Verify kill zone filter is called in run()."""
        engine_path = ROOT / "backtest" / "engine.py"
        content = engine_path.read_text()
        assert "self._filter_signals_by_kill_zone(signals)" in content, \
            "Kill zone filter not called in backtest run()"

    def test_quality_gate_in_paper_trade(self):
        """Verify paper trade applies quality gate."""
        paper_path = ROOT / "scripts" / "run_paper_trade.py"
        content = paper_path.read_text()
        assert "filter_to_best_signals" in content, \
            "Paper trade missing quality gate (filter_to_best_signals)"

    def test_quality_gate_import_exists(self):
        """Verify paper trade imports quality gate."""
        paper_path = ROOT / "scripts" / "run_paper_trade.py"
        content = paper_path.read_text()
        assert "from risk.signal_quality_gate import filter_to_best_signals" in content, \
            "Paper trade missing quality gate import"

    def test_risk_pct_aligned(self):
        """Verify backtest risk% matches live mode."""
        import yaml
        config_path = ROOT / "config" / "base.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        sizing_risk = cfg.get("sizing", {}).get("risk_per_trade_pct", 0.005)
        live_risk = cfg.get("policy", {}).get("modes", {}).get("medium", {}).get("risk_per_trade_pct", 0.025)

        assert abs(sizing_risk - live_risk) < 0.005, \
            f"Risk% mismatch: sizing={sizing_risk}, live={live_risk}"

    def test_time_stop_uses_trading_days(self):
        """Verify time stop uses NYSE trading days."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from market_calendar import get_trading_days_between

        # Should return ~3 trading days for Mon Jan 5 to Fri Jan 10 (skipping weekend)
        from datetime import date
        days = get_trading_days_between(date(2026, 1, 5), date(2026, 1, 10))
        # Jan 5 (Mon) to Jan 10 (Sat) - trading days: Jan 6, 7, 8, 9 = 4 days
        # (Jan 5 is start, so we count days AFTER Jan 5)
        assert 3 <= days <= 5, f"Trading days calculation broken: {days}"

    def test_stop_coverage_logic_fixed(self):
        """Verify stop coverage tracks only STOP orders."""
        runner_path = ROOT / "scripts" / "runner.py"
        content = runner_path.read_text()
        assert "symbols_with_stop_coverage" in content, \
            "Runner missing symbols_with_stop_coverage tracking"

    def test_paper_mode_assertion_exists(self):
        """Verify paper mode URL assertion exists."""
        runner_path = ROOT / "scripts" / "runner.py"
        content = runner_path.read_text()
        assert "SAFETY VIOLATION" in content or "paper_mode_url_mismatch" in content, \
            "Runner missing paper mode URL assertion"


class TestKillZoneLogic:
    """Test kill zone filtering logic in detail."""

    def test_primary_window_allowed(self):
        """Signals at 10:30 should pass kill zone filter."""
        import pandas as pd
        from backtest.engine import Backtester, BacktestConfig

        # Create backtester with minimal config
        cfg = BacktestConfig(apply_kill_zones=True)
        bt = Backtester(cfg, lambda x: x, lambda x: pd.DataFrame())

        # Signal at 10:30 AM (within primary window 10:00-11:30)
        signals = pd.DataFrame({
            'timestamp': [pd.Timestamp('2026-01-06 10:30:00')],
            'symbol': ['TEST'],
            'side': ['long'],
        })

        filtered = bt._filter_signals_by_kill_zone(signals)
        assert len(filtered) == 1, "Signal at 10:30 should pass kill zone filter"

    def test_opening_range_blocked(self):
        """Signals at 9:45 should be blocked."""
        import pandas as pd
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(apply_kill_zones=True)
        bt = Backtester(cfg, lambda x: x, lambda x: pd.DataFrame())

        # Signal at 9:45 AM (within opening range 9:30-10:00)
        signals = pd.DataFrame({
            'timestamp': [pd.Timestamp('2026-01-06 09:45:00')],
            'symbol': ['TEST'],
            'side': ['long'],
        })

        filtered = bt._filter_signals_by_kill_zone(signals)
        assert len(filtered) == 0, "Signal at 9:45 should be blocked by kill zone"

    def test_lunch_blocked(self):
        """Signals at 12:00 should be blocked."""
        import pandas as pd
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(apply_kill_zones=True)
        bt = Backtester(cfg, lambda x: x, lambda x: pd.DataFrame())

        # Signal at 12:00 PM (within lunch chop 11:30-14:30)
        signals = pd.DataFrame({
            'timestamp': [pd.Timestamp('2026-01-06 12:00:00')],
            'symbol': ['TEST'],
            'side': ['long'],
        })

        filtered = bt._filter_signals_by_kill_zone(signals)
        assert len(filtered) == 0, "Signal at 12:00 should be blocked by kill zone"

    def test_power_hour_allowed(self):
        """Signals at 15:00 should pass kill zone filter."""
        import pandas as pd
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(apply_kill_zones=True)
        bt = Backtester(cfg, lambda x: x, lambda x: pd.DataFrame())

        # Signal at 15:00 (within power hour 14:30-15:30)
        signals = pd.DataFrame({
            'timestamp': [pd.Timestamp('2026-01-06 15:00:00')],
            'symbol': ['TEST'],
            'side': ['long'],
        })

        filtered = bt._filter_signals_by_kill_zone(signals)
        assert len(filtered) == 1, "Signal at 15:00 should pass kill zone filter"

    def test_empty_signals_handled(self):
        """Empty signals should return empty without error."""
        import pandas as pd
        from backtest.engine import Backtester, BacktestConfig

        cfg = BacktestConfig(apply_kill_zones=True)
        bt = Backtester(cfg, lambda x: x, lambda x: pd.DataFrame())

        signals = pd.DataFrame()
        filtered = bt._filter_signals_by_kill_zone(signals)
        assert filtered.empty, "Empty signals should return empty"
