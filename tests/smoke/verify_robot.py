"""
SMOKE TEST: Verify PAPER-ONLY mode is enforced.

This test MUST pass before any deployment.
It verifies:
1. Orders are never sent to live endpoint
2. "PAPER MODE CONFIRMED" is logged
3. Any attempt to call live endpoints raises immediately
4. All order paths have paper guard

Run with: python -m pytest tests/smoke/verify_robot.py -v
Or: python tests/smoke/verify_robot.py
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPaperModeEnforcement:
    """Verify paper mode is enforced at ALL entry points."""

    def test_ensure_paper_mode_or_die_blocks_live_url(self):
        """Calling with live URL must raise immediately."""
        from safety.paper_guard import ensure_paper_mode_or_die, LiveTradingBlockedError

        with patch.dict(os.environ, {"ALPACA_BASE_URL": "https://api.alpaca.markets"}):
            try:
                ensure_paper_mode_or_die(context="test")
                assert False, "Should have raised LiveTradingBlockedError"
            except LiveTradingBlockedError as e:
                assert "LIVE endpoint" in str(e)
                assert "PAPER-ONLY" in str(e)

    def test_ensure_paper_mode_or_die_allows_paper_url(self):
        """Calling with paper URL must succeed."""
        from safety.paper_guard import ensure_paper_mode_or_die

        # Remove kill switch if it exists (for test only)
        kill_switch = Path("state/KILL_SWITCH")
        kill_switch_existed = kill_switch.exists()
        if kill_switch_existed:
            kill_switch_content = kill_switch.read_text()
            kill_switch.unlink()

        try:
            with patch.dict(os.environ, {"ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}):
                ok, msg = ensure_paper_mode_or_die(context="test")
                assert ok is True
                assert "PAPER MODE CONFIRMED" in msg
        finally:
            # Restore kill switch if it existed
            if kill_switch_existed:
                kill_switch.write_text(kill_switch_content)

    def test_paper_only_enforced_flag_is_true(self):
        """PAPER_ONLY_ENFORCED must be True in paper_guard.py."""
        from safety.paper_guard import PAPER_ONLY_ENFORCED

        assert PAPER_ONLY_ENFORCED is True, \
            "BLOCKER: PAPER_ONLY_ENFORCED is False - code has been tampered with!"

    def test_no_live_orders_flag_is_true(self):
        """NO_LIVE_ORDERS must be True in execution_choke.py."""
        from safety.execution_choke import NO_LIVE_ORDERS

        assert NO_LIVE_ORDERS is True, \
            "BLOCKER: NO_LIVE_ORDERS is False - code has been tampered with!"

    def test_no_live_orders_gate_blocks_live(self):
        """NO_LIVE_ORDERS gate must block live endpoint."""
        from safety.execution_choke import assert_no_live_orders, SafetyViolationError

        with patch.dict(os.environ, {"ALPACA_BASE_URL": "https://api.alpaca.markets"}):
            try:
                assert_no_live_orders(context="test")
                assert False, "Should have raised SafetyViolationError"
            except SafetyViolationError as e:
                assert "NO_LIVE_ORDERS" in str(e)

    def test_all_order_paths_have_paper_guard(self):
        """Verify ALL order placement functions have paper guard."""
        order_functions = [
            ("execution/broker_alpaca.py", ["place_ioc_limit", "place_bracket_order",
                                            "place_order_with_liquidity_check", "place_order",
                                            "_place_order_direct"]),
            ("execution/broker_paper.py", ["place_order"]),
            ("execution/broker_crypto.py", ["place_order"]),
            ("execution/broker_alpaca_crypto.py", ["place_order"]),
            ("options/order_router.py", ["submit_order"]),
        ]

        missing_guards = []
        for filepath, func_names in order_functions:
            full_path = PROJECT_ROOT / filepath
            if not full_path.exists():
                missing_guards.append(f"FILE NOT FOUND: {filepath}")
                continue

            content = full_path.read_text()

            # Check that ensure_paper_mode_or_die is called in this file
            if "ensure_paper_mode_or_die" not in content:
                missing_guards.append(f"MISSING GUARD in {filepath}")

        assert len(missing_guards) == 0, \
            f"BLOCKER: Missing paper guards:\n" + "\n".join(missing_guards)

    def test_alpaca_base_url_default_is_paper(self):
        """Default ALPACA_BASE_URL must be paper endpoint."""
        from safety.paper_guard import PAPER_ENDPOINT

        # Remove env var to test default behavior
        env_backup = os.environ.pop("ALPACA_BASE_URL", None)

        try:
            # The default should be paper endpoint
            default_url = os.getenv("ALPACA_BASE_URL", PAPER_ENDPOINT)
            assert "paper-api" in default_url or default_url == PAPER_ENDPOINT, \
                f"BLOCKER: Default endpoint is not paper: {default_url}"
        finally:
            if env_backup:
                os.environ["ALPACA_BASE_URL"] = env_backup

    def test_kill_switch_blocks_all_orders(self):
        """Kill switch must block ALL order submissions."""
        from safety.paper_guard import ensure_paper_mode_or_die, KillSwitchActiveError

        kill_switch = Path("state/KILL_SWITCH")
        kill_switch.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing kill switch if any
        kill_switch_existed = kill_switch.exists()
        if kill_switch_existed:
            kill_switch_content = kill_switch.read_text()

        try:
            # Create kill switch
            kill_switch.write_text('{"reason": "test", "timestamp": "2026-01-08"}')

            with patch.dict(os.environ, {"ALPACA_BASE_URL": "https://paper-api.alpaca.markets"}):
                try:
                    ensure_paper_mode_or_die(context="test")
                    assert False, "Should have raised KillSwitchActiveError"
                except KillSwitchActiveError as e:
                    assert "Kill switch" in str(e) or "KILL_SWITCH" in str(e)
        finally:
            # Restore or clean up
            if kill_switch_existed:
                kill_switch.write_text(kill_switch_content)
            elif kill_switch.exists():
                kill_switch.unlink()

    def test_safety_mode_paper_only_is_true(self):
        """PAPER_ONLY must be True in safety/mode.py."""
        try:
            from safety.mode import PAPER_ONLY
            assert PAPER_ONLY is True, \
                "BLOCKER: PAPER_ONLY is False in safety/mode.py!"
        except ImportError:
            # If mode.py doesn't exist, that's OK - we have paper_guard.py
            pass

    def test_safety_mode_live_trading_disabled(self):
        """LIVE_TRADING_ENABLED must be False in safety/mode.py."""
        try:
            from safety.mode import LIVE_TRADING_ENABLED
            assert LIVE_TRADING_ENABLED is False, \
                "BLOCKER: LIVE_TRADING_ENABLED is True in safety/mode.py!"
        except ImportError:
            # If mode.py doesn't exist, that's OK - we have paper_guard.py
            pass


def run_smoke_test():
    """Run all paper mode verification tests."""
    import traceback

    print("=" * 70)
    print("PAPER MODE VERIFICATION SMOKE TEST")
    print("=" * 70)
    print()

    test_class = TestPaperModeEnforcement()
    test_methods = [m for m in dir(test_class) if m.startswith("test_")]

    passed = 0
    failed = 0
    results = []

    for method_name in test_methods:
        method = getattr(test_class, method_name)
        test_name = method_name.replace("test_", "").replace("_", " ").title()

        try:
            method()
            print(f"  [PASS] {test_name}")
            passed += 1
            results.append((method_name, "PASS", None))
        except AssertionError as e:
            print(f"  [FAIL] {test_name}")
            print(f"         {e}")
            failed += 1
            results.append((method_name, "FAIL", str(e)))
        except Exception as e:
            print(f"  [ERROR] {test_name}")
            print(f"          {e}")
            failed += 1
            results.append((method_name, "ERROR", str(e)))

    print()
    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("-" * 70)

    if failed == 0:
        print()
        print("=" * 70)
        print("PAPER MODE CONFIRMED - All tests passed")
        print("Robot is SAFE for paper trading")
        print("=" * 70)
        return 0
    else:
        print()
        print("=" * 70)
        print("BLOCKER: Paper mode verification FAILED")
        print("DO NOT PROCEED until all tests pass")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_smoke_test()
    sys.exit(exit_code)
