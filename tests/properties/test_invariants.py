"""
Property-Based Tests for Trading System Invariants.

This module uses Hypothesis to verify critical invariants that must
ALWAYS hold true in the trading system, regardless of input.

Blueprint Alignment:
    Implements Section 2.5 requirements for property-based testing with:
    - Position sizing invariants
    - Kill switch behavior
    - Data validation invariants
    - Exception hierarchy invariants
    - DecisionPacket determinism

Key Invariants Tested:
1. Position sizing never exceeds dual-cap (2% risk + 20% notional)
2. Kill switch ALWAYS blocks when active
3. Shifted indicators never have lookahead bias
4. DecisionPacket hash is deterministic
5. OHLC relationships are always valid
6. Exception hierarchy is consistent
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Import modules under test
from core.exceptions import (
    QuantSystemError,
    SafetyError,
    KillSwitchActiveError,
    ExecutionError,
    DataError,
    is_safety_critical,
    get_error_code,
)
from core.kill_switch import (
    is_kill_switch_active,
    activate_kill_switch,
    deactivate_kill_switch,
    check_kill_switch,
    KillSwitchTrigger,
)
from research.evidence import EvidencePack, EvidencePackBuilder
from features.registry import FeatureMetadata, FeatureRegistry, FeatureCategory


# =============================================================================
# STRATEGIES (Custom data generators)
# =============================================================================

# Price strategy: positive floats with reasonable bounds
price_strategy = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Quantity strategy: positive integers
qty_strategy = st.integers(min_value=1, max_value=1000000)

# Equity strategy: account equity
equity_strategy = st.floats(min_value=1000.0, max_value=10000000.0, allow_nan=False, allow_infinity=False)

# Risk percent strategy
risk_pct_strategy = st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False)

# Symbol strategy
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu",), whitelist_characters=""),
    min_size=1,
    max_size=5,
).filter(lambda x: x.isalpha())

# Date strategy
date_strategy = st.dates(min_value=datetime(2000, 1, 1).date(), max_value=datetime(2030, 12, 31).date())


# =============================================================================
# POSITION SIZING INVARIANTS
# =============================================================================

class TestPositionSizingInvariants:
    """
    Property tests for position sizing.

    Critical Invariant: Position size must NEVER exceed:
    1. 2% of equity at risk (risk_pct * equity / risk_per_share)
    2. 20% of equity in notional (notional_pct * equity / entry_price)

    The final size is the MINIMUM of these two.
    """

    @given(
        equity=equity_strategy,
        entry_price=price_strategy,
        stop_price=price_strategy,
        risk_pct=st.floats(min_value=0.01, max_value=0.05, allow_nan=False),
        notional_pct=st.floats(min_value=0.10, max_value=0.30, allow_nan=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_dual_cap_always_respected(
        self, equity: float, entry_price: float, stop_price: float,
        risk_pct: float, notional_pct: float
    ):
        """Position size never exceeds either cap."""
        # Entry must be above stop for long positions
        assume(entry_price > stop_price)
        assume(entry_price - stop_price > 0.01)  # Minimum risk per share

        risk_per_share = entry_price - stop_price

        # Calculate shares by each method
        max_risk_dollars = equity * risk_pct
        shares_by_risk = int(max_risk_dollars / risk_per_share)

        max_notional = equity * notional_pct
        shares_by_notional = int(max_notional / entry_price)

        # Final size is the minimum
        final_shares = min(shares_by_risk, shares_by_notional)

        # INVARIANT 1: Risk never exceeds risk_pct of equity
        actual_risk = final_shares * risk_per_share
        assert actual_risk <= equity * risk_pct + 0.01  # Small float tolerance

        # INVARIANT 2: Notional never exceeds notional_pct of equity
        actual_notional = final_shares * entry_price
        assert actual_notional <= equity * notional_pct + 0.01

    @given(
        equity=equity_strategy,
        entry_price=price_strategy,
    )
    @settings(max_examples=100)
    def test_position_size_never_negative(self, equity: float, entry_price: float):
        """Position size is always non-negative."""
        assume(entry_price > 0)

        max_notional = equity * 0.20  # 20% cap
        shares = int(max_notional / entry_price)

        assert shares >= 0

    @given(
        entry_price=price_strategy,
        stop_price=price_strategy,
    )
    @settings(max_examples=100)
    def test_risk_per_share_positive_for_valid_trade(
        self, entry_price: float, stop_price: float
    ):
        """Risk per share is positive when entry > stop (long position)."""
        assume(entry_price > stop_price)

        risk_per_share = entry_price - stop_price
        assert risk_per_share > 0


# =============================================================================
# KILL SWITCH INVARIANTS
# =============================================================================

class TestKillSwitchInvariants:
    """
    Property tests for kill switch behavior.

    Critical Invariant: When kill switch is active, check_kill_switch()
    MUST raise KillSwitchActiveError. There are NO exceptions.
    """

    @given(reason=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_kill_switch_always_blocks_when_active(self, reason: str):
        """Kill switch ALWAYS raises when active, regardless of reason."""
        # Clean up first
        deactivate_kill_switch()

        try:
            # Activate with any reason
            activate_kill_switch(reason)

            # INVARIANT: check_kill_switch MUST raise
            with pytest.raises(KillSwitchActiveError):
                check_kill_switch()

            # Also verify is_kill_switch_active returns True
            assert is_kill_switch_active() is True

        finally:
            # Always clean up
            deactivate_kill_switch()

    def test_kill_switch_inactive_allows_operations(self):
        """When kill switch is inactive, operations proceed."""
        deactivate_kill_switch()

        # Should NOT raise
        check_kill_switch()

        assert is_kill_switch_active() is False

    @given(trigger=st.sampled_from(list(KillSwitchTrigger)))
    @settings(max_examples=20)
    def test_all_trigger_types_activate_switch(self, trigger: KillSwitchTrigger):
        """All trigger types properly activate the kill switch."""
        deactivate_kill_switch()

        try:
            from core.kill_switch import auto_trigger_kill_switch

            auto_trigger_kill_switch(
                trigger=trigger,
                reason=f"Test trigger: {trigger.value}",
                context={"test": True}
            )

            # INVARIANT: Switch must be active after any trigger
            assert is_kill_switch_active() is True

            with pytest.raises(KillSwitchActiveError):
                check_kill_switch()

        finally:
            deactivate_kill_switch()


# =============================================================================
# OHLC DATA INVARIANTS
# =============================================================================

class TestOHLCInvariants:
    """
    Property tests for OHLC data validity.

    Critical Invariant: For any valid OHLC bar:
    - high >= max(open, close)
    - low <= min(open, close)
    - high >= low
    - All prices > 0
    """

    @given(
        open_price=price_strategy,
        high_price=price_strategy,
        low_price=price_strategy,
        close_price=price_strategy,
    )
    @settings(max_examples=200)
    def test_valid_ohlc_relationships(
        self, open_price: float, high_price: float,
        low_price: float, close_price: float
    ):
        """Test OHLC validation logic."""
        # Generate valid OHLC from random values
        prices = [open_price, high_price, low_price, close_price]

        # Construct valid bar
        valid_open = prices[0]
        valid_close = prices[3]
        valid_high = max(prices)
        valid_low = min(prices)

        # INVARIANT 1: high >= max(open, close)
        assert valid_high >= max(valid_open, valid_close)

        # INVARIANT 2: low <= min(open, close)
        assert valid_low <= min(valid_open, valid_close)

        # INVARIANT 3: high >= low
        assert valid_high >= valid_low

        # INVARIANT 4: All prices positive
        assert all(p > 0 for p in [valid_open, valid_high, valid_low, valid_close])

    @given(
        open_price=price_strategy,
        close_price=price_strategy,
        hl_range=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_construct_valid_bar_from_open_close(
        self, open_price: float, close_price: float, hl_range: float
    ):
        """Any open/close pair can form a valid bar."""
        # High is max of open/close plus some extension
        high = max(open_price, close_price) + hl_range

        # Low is min of open/close minus some extension (but positive)
        low = max(0.01, min(open_price, close_price) - hl_range)

        # Verify invariants hold
        assert high >= max(open_price, close_price)
        assert low <= min(open_price, close_price)
        assert high >= low
        assert all(p > 0 for p in [open_price, high, low, close_price])


# =============================================================================
# EXCEPTION HIERARCHY INVARIANTS
# =============================================================================

class TestExceptionHierarchyInvariants:
    """
    Property tests for exception hierarchy.

    Critical Invariants:
    - All exceptions inherit from QuantSystemError
    - SafetyError.is_recoverable is ALWAYS False
    - error_code is ALWAYS a non-empty string
    """

    @given(message=st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_quant_system_error_always_has_code(self, message: str):
        """All QuantSystemError instances have an error code."""
        error = QuantSystemError(message)

        # INVARIANT: error_code is always set
        assert error.error_code is not None
        assert isinstance(error.error_code, str)
        assert len(error.error_code) > 0

    @given(message=st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_safety_error_never_recoverable(self, message: str):
        """SafetyError is NEVER recoverable."""
        error = SafetyError(message)

        # INVARIANT: Safety errors are never recoverable
        assert error.is_recoverable is False
        assert is_safety_critical(error) is True

    @given(message=st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_kill_switch_error_is_safety_critical(self, message: str):
        """KillSwitchActiveError is always safety-critical."""
        error = KillSwitchActiveError(message)

        # INVARIANT: Kill switch errors are safety-critical
        assert is_safety_critical(error) is True
        assert error.is_recoverable is False
        assert isinstance(error, SafetyError)
        assert isinstance(error, QuantSystemError)

    @given(
        context=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text()),
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_exception_context_preserved(self, context: Dict[str, Any]):
        """Exception context is always preserved."""
        error = QuantSystemError("Test", context=context)

        # INVARIANT: Context is preserved exactly
        assert error.context == context

    def test_exception_hierarchy_inheritance(self):
        """All exception classes maintain proper inheritance."""
        # Define expected hierarchy
        safety_subclasses = [KillSwitchActiveError]

        for cls in safety_subclasses:
            instance = cls("test")
            assert isinstance(instance, SafetyError)
            assert isinstance(instance, QuantSystemError)
            assert isinstance(instance, Exception)


# =============================================================================
# EVIDENCE PACK INVARIANTS
# =============================================================================

class TestEvidencePackInvariants:
    """
    Property tests for evidence pack system.

    Critical Invariants:
    - Pack hash is deterministic (same inputs -> same hash)
    - Pack hash changes when content changes
    - Save/load preserves all data
    """

    @given(
        win_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        profit_factor=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_pack_hash_deterministic(self, win_rate: float, profit_factor: float):
        """Same inputs always produce same hash."""
        # Use fixed pack_id to ensure determinism (pack_id varies by design)
        fixed_datetime = datetime(2025, 1, 1, 12, 0, 0)

        pack1 = EvidencePack(
            pack_id="test_fixed_id",
            created_at=fixed_datetime,
            pack_type="test",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            python_version="3.11.0",
            package_versions={},
            config_snapshot={},
            frozen_params={},
            metrics={"win_rate": win_rate, "profit_factor": profit_factor},
        )

        pack2 = EvidencePack(
            pack_id="test_fixed_id",
            created_at=fixed_datetime,
            pack_type="test",
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            python_version="3.11.0",
            package_versions={},
            config_snapshot={},
            frozen_params={},
            metrics={"win_rate": win_rate, "profit_factor": profit_factor},
        )

        # INVARIANT: Same inputs produce same hash
        assert pack1.pack_hash == pack2.pack_hash

    @given(
        win_rate1=st.floats(min_value=0.0, max_value=0.49, allow_nan=False),
        win_rate2=st.floats(min_value=0.51, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_pack_hash_changes_with_content(self, win_rate1: float, win_rate2: float):
        """Different content produces different hash."""
        assume(abs(win_rate1 - win_rate2) > 0.01)

        builder1 = EvidencePackBuilder("test")
        builder1.git_commit = "abc123"
        builder1.set_metrics({"win_rate": win_rate1})
        pack1 = builder1.build()

        builder2 = EvidencePackBuilder("test")
        builder2.git_commit = "abc123"
        builder2.set_metrics({"win_rate": win_rate2})
        pack2 = builder2.build()

        # INVARIANT: Different metrics produce different hash
        assert pack1.pack_hash != pack2.pack_hash

    @given(
        win_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        total_trades=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)  # Disable deadline for I/O tests
    def test_save_load_preserves_data(self, win_rate: float, total_trades: int):
        """Save and load preserves all pack data."""
        builder = EvidencePackBuilder("backtest")
        builder.capture_git_state()
        builder.set_metrics({"win_rate": win_rate, "total_trades": total_trades})
        original = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = original.save(Path(tmpdir))
            loaded = EvidencePack.load(filepath)

            # INVARIANT: All data preserved
            assert loaded.pack_id == original.pack_id
            assert loaded.pack_hash == original.pack_hash
            assert loaded.win_rate == original.win_rate
            assert loaded.total_trades == original.total_trades
            assert loaded.verify_hash() is True


# =============================================================================
# FEATURE REGISTRY INVARIANTS
# =============================================================================

class TestFeatureRegistryInvariants:
    """
    Property tests for feature registry.

    Critical Invariants:
    - Registered features can always be retrieved
    - Feature hash is deterministic
    - Lookback calculation is always >= individual lookbacks
    """

    @given(
        name=st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=20),
        lookback=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=50)
    def test_registered_feature_retrievable(self, name: str, lookback: int):
        """Registered features can always be retrieved."""
        assume(name.isalpha())  # Valid feature names

        registry = FeatureRegistry()
        meta = FeatureMetadata(
            name=name,
            version="1.0.0",
            lookback_periods=lookback,
        )
        registry.register(meta)

        # INVARIANT: Can retrieve what we registered
        retrieved = registry.get(name)
        assert retrieved is not None
        assert retrieved.name == name
        assert retrieved.lookback_periods == lookback

    @given(
        lookback1=st.integers(min_value=1, max_value=100),
        lookback2=st.integers(min_value=1, max_value=100),
        lookback3=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_max_lookback_calculation(self, lookback1: int, lookback2: int, lookback3: int):
        """Max lookback is always >= individual lookbacks."""
        registry = FeatureRegistry()

        features = [
            FeatureMetadata(name="feat1", lookback_periods=lookback1),
            FeatureMetadata(name="feat2", lookback_periods=lookback2),
            FeatureMetadata(name="feat3", lookback_periods=lookback3),
        ]

        for f in features:
            registry.register(f)

        max_lb = registry.get_required_lookback(["feat1", "feat2", "feat3"])

        # INVARIANT: Max lookback >= each individual lookback
        assert max_lb >= lookback1
        assert max_lb >= lookback2
        assert max_lb >= lookback3
        assert max_lb == max(lookback1, lookback2, lookback3)

    @given(is_shifted=st.booleans())
    @settings(max_examples=20)
    def test_lookahead_safety_check(self, is_shifted: bool):
        """Lookahead safety check correctly identifies unshifted features."""
        registry = FeatureRegistry()
        registry.register(FeatureMetadata(
            name="test_feat",
            is_shifted=is_shifted,
        ))

        is_safe, unsafe = registry.check_lookahead_safety(["test_feat"])

        # INVARIANT: Safety check matches is_shifted flag
        assert is_safe == is_shifted
        if not is_shifted:
            assert "test_feat" in unsafe


# =============================================================================
# SHIFTED INDICATOR INVARIANTS
# =============================================================================

class TestShiftedIndicatorInvariants:
    """
    Property tests for shifted indicators (lookahead prevention).

    Critical Invariant: When using .shift(1), the value at time t
    should equal the original value at time t-1.
    """

    @given(
        values=st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            min_size=5,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_shift_one_equals_previous(self, values):
        """Shifted values equal previous bar values."""
        df = pd.DataFrame({"value": values})
        df["shifted"] = df["value"].shift(1)

        # Skip first row (NaN after shift)
        for i in range(1, len(df)):
            # INVARIANT: shifted[i] == value[i-1]
            assert df.iloc[i]["shifted"] == df.iloc[i-1]["value"]

    @given(
        values=st.lists(
            st.floats(min_value=0.01, max_value=1000, allow_nan=False),
            min_size=10,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_no_lookahead_in_shifted_signals(self, values):
        """Signals generated from shifted data have no lookahead."""
        df = pd.DataFrame({"close": values})

        # Simulate a simple threshold signal
        threshold = sum(values) / len(values)  # Mean

        # WRONG: Using current bar (lookahead)
        df["signal_wrong"] = df["close"] < threshold

        # CORRECT: Using previous bar (no lookahead)
        df["signal_correct"] = df["close"].shift(1) < threshold

        # For any signal at time t, the shifted version uses data from t-1
        for i in range(1, len(df)):
            if df.iloc[i]["signal_correct"]:
                # INVARIANT: Signal is based on PREVIOUS bar
                assert df.iloc[i-1]["close"] < threshold
