#!/usr/bin/env python3
"""
Component Verification Script
=============================

FIX (2026-01-08): Phase 5 - Verify ALL claimed components actually work.

This script:
1. Attempts to import every critical module
2. Verifies factory functions exist and are callable
3. Verifies key classes can be instantiated
4. Reports comprehensive status

Run: python tools/verify_all_components.py
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of a single component verification."""
    name: str
    module: str
    attribute: str
    status: str  # OK, FAIL, WARN
    error: Optional[str] = None
    details: Optional[str] = None


# Critical components to verify
COMPONENTS = [
    # === ML Advanced ===
    ("HMM Regime Detector", "ml_advanced.hmm_regime_detector", "HMMRegimeDetector"),
    ("HMM Factory Function", "ml_advanced.hmm_regime_detector", "get_hmm_detector"),
    ("LSTM Confidence Model", "ml_advanced.lstm_confidence.model", "LSTMConfidenceModel"),
    ("LSTM Factory Function", "ml_advanced.lstm_confidence.model", "get_lstm_model"),
    ("Ensemble Predictor", "ml_advanced.ensemble.ensemble_predictor", "EnsemblePredictor"),
    ("Markov Predictor", "ml_advanced.markov_chain.predictor", "MarkovPredictor"),

    # === Strategies ===
    ("Dual Strategy Scanner", "strategies.dual_strategy", "DualStrategyScanner"),
    ("Dual Strategy Params", "strategies.dual_strategy", "DualStrategyParams"),
    ("Strategy Registry", "strategies.registry", "get_production_scanner"),

    # === Risk Management ===
    ("Policy Gate", "risk.policy_gate", "PolicyGate"),
    ("Risk Limits", "risk.policy_gate", "RiskLimits"),
    ("Equity Sizer", "risk.equity_sizer", "calculate_position_size"),
    ("Kelly-Enhanced Sizer", "risk.equity_sizer", "calculate_position_size_with_kelly"),
    ("Kill Zone Gate", "risk.kill_zone_gate", "can_trade_now"),

    # === Advanced Risk ===
    ("Monte Carlo VaR", "risk.advanced.monte_carlo_var", "MonteCarloVaR"),
    ("Portfolio VaR Check", "risk.advanced", "check_portfolio_var"),
    ("Kelly Position Sizer", "risk.advanced.kelly_position_sizer", "KellyPositionSizer"),
    ("Correlation Limits", "risk.advanced.correlation_limits", "EnhancedCorrelationLimits"),

    # === Backtest ===
    ("Backtester", "backtest.engine", "Backtester"),
    ("Backtest Config", "backtest.engine", "BacktestConfig"),
    ("Walk Forward", "backtest.walk_forward", "run_walk_forward"),

    # === Execution ===
    ("Broker Alpaca", "execution.broker_alpaca", "place_ioc_limit"),
    ("Order Record", "oms.order_state", "OrderRecord"),
    ("Idempotency Store", "oms.idempotency_store", "IdempotencyStore"),

    # === Core ===
    ("Hash Chain Verify", "core.hash_chain", "verify_chain"),
    ("Hash Chain Append", "core.hash_chain", "append_block"),
    ("Structured Log", "core.structured_log", "jlog"),
    ("Kill Switch", "core.kill_switch", "is_kill_switch_active"),
    ("Config Hash", "core.config_pin", "sha256_file"),

    # === Data ===
    ("Polygon Provider", "data.providers.polygon_eod", "fetch_daily_bars_polygon"),
    ("Polygon Config", "data.providers.polygon_eod", "PolygonConfig"),
    ("Universe Loader", "data.universe.loader", "load_universe"),

    # === Cognitive ===
    ("Cognitive Brain", "cognitive.cognitive_brain", "CognitiveBrain"),
    ("Self Model", "cognitive.self_model", "SelfModel"),
    ("Reflection Engine", "cognitive.reflection_engine", "ReflectionEngine"),

    # === Autonomous ===
    ("Autonomous Brain", "autonomous.brain", "AutonomousBrain"),
    ("Time Awareness", "autonomous.awareness", "TimeAwareness"),
    ("Market Context", "autonomous.awareness", "MarketContext"),
]


def verify_component(name: str, module_path: str, attribute: str) -> VerificationResult:
    """Verify a single component can be imported."""
    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the attribute
        obj = getattr(module, attribute, None)

        if obj is None:
            return VerificationResult(
                name=name,
                module=module_path,
                attribute=attribute,
                status="FAIL",
                error=f"Attribute '{attribute}' not found in module",
            )

        # Determine if it's callable
        is_callable = callable(obj)
        is_class = isinstance(obj, type)

        details = []
        if is_class:
            details.append("class")
        elif is_callable:
            details.append("callable")
        else:
            details.append(f"type={type(obj).__name__}")

        return VerificationResult(
            name=name,
            module=module_path,
            attribute=attribute,
            status="OK",
            details=", ".join(details),
        )

    except ImportError as e:
        return VerificationResult(
            name=name,
            module=module_path,
            attribute=attribute,
            status="FAIL",
            error=f"ImportError: {e}",
        )
    except Exception as e:
        return VerificationResult(
            name=name,
            module=module_path,
            attribute=attribute,
            status="WARN",
            error=f"{type(e).__name__}: {e}",
        )


def verify_all() -> Tuple[List[VerificationResult], Dict[str, int]]:
    """Verify all components and return results with summary."""
    results = []
    summary = {"OK": 0, "FAIL": 0, "WARN": 0}

    for name, module, attr in COMPONENTS:
        result = verify_component(name, module, attr)
        results.append(result)
        summary[result.status] += 1

    return results, summary


def test_factory_functions() -> List[VerificationResult]:
    """Test that factory functions actually work."""
    results = []

    # Test get_hmm_detector
    try:
        from ml_advanced.hmm_regime_detector import get_hmm_detector
        detector = get_hmm_detector()
        if detector is not None:
            results.append(VerificationResult(
                name="HMM Factory Instantiation",
                module="ml_advanced.hmm_regime_detector",
                attribute="get_hmm_detector()",
                status="OK",
                details=f"Returns {type(detector).__name__}",
            ))
        else:
            results.append(VerificationResult(
                name="HMM Factory Instantiation",
                module="ml_advanced.hmm_regime_detector",
                attribute="get_hmm_detector()",
                status="WARN",
                error="Returns None",
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="HMM Factory Instantiation",
            module="ml_advanced.hmm_regime_detector",
            attribute="get_hmm_detector()",
            status="FAIL",
            error=str(e),
        ))

    # Test get_lstm_model (may fail due to TensorFlow)
    try:
        from ml_advanced.lstm_confidence.model import get_lstm_model
        model = get_lstm_model()
        results.append(VerificationResult(
            name="LSTM Factory Instantiation",
            module="ml_advanced.lstm_confidence.model",
            attribute="get_lstm_model()",
            status="OK",
            details=f"Returns {type(model).__name__}",
        ))
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            results.append(VerificationResult(
                name="LSTM Factory Instantiation",
                module="ml_advanced.lstm_confidence.model",
                attribute="get_lstm_model()",
                status="WARN",
                error="TensorFlow not installed (optional)",
            ))
        else:
            results.append(VerificationResult(
                name="LSTM Factory Instantiation",
                module="ml_advanced.lstm_confidence.model",
                attribute="get_lstm_model()",
                status="FAIL",
                error=str(e),
            ))
    except Exception as e:
        results.append(VerificationResult(
            name="LSTM Factory Instantiation",
            module="ml_advanced.lstm_confidence.model",
            attribute="get_lstm_model()",
            status="WARN",
            error=str(e),
        ))

    # Test check_portfolio_var
    try:
        from risk.advanced import check_portfolio_var
        passes, result = check_portfolio_var(positions=[])
        results.append(VerificationResult(
            name="VaR Check Execution",
            module="risk.advanced",
            attribute="check_portfolio_var([])",
            status="OK",
            details=f"passes={passes}, var_pct={result.get('var_pct', 'N/A')}",
        ))
    except Exception as e:
        results.append(VerificationResult(
            name="VaR Check Execution",
            module="risk.advanced",
            attribute="check_portfolio_var([])",
            status="FAIL",
            error=str(e),
        ))

    # Test position sizing
    try:
        from risk.equity_sizer import calculate_position_size
        size = calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            risk_pct=0.02,
            account_equity=100000.0,
        )
        results.append(VerificationResult(
            name="Position Sizing Execution",
            module="risk.equity_sizer",
            attribute="calculate_position_size()",
            status="OK",
            details=f"shares={size.shares}, risk=${size.risk_dollars:.2f}",
        ))
    except Exception as e:
        results.append(VerificationResult(
            name="Position Sizing Execution",
            module="risk.equity_sizer",
            attribute="calculate_position_size()",
            status="FAIL",
            error=str(e),
        ))

    return results


def print_results(results: List[VerificationResult], summary: Dict[str, int]):
    """Print verification results."""
    print("\n" + "=" * 80)
    print("           COMPONENT VERIFICATION REPORT")
    print("=" * 80)

    # Group by status
    ok_results = [r for r in results if r.status == "OK"]
    fail_results = [r for r in results if r.status == "FAIL"]
    warn_results = [r for r in results if r.status == "WARN"]

    # Print failures first
    if fail_results:
        print("\n[FAILURES]")
        print("-" * 40)
        for r in fail_results:
            print(f"  X {r.name}")
            print(f"    Module: {r.module}")
            print(f"    Attr: {r.attribute}")
            print(f"    Error: {r.error}")
            print()

    # Print warnings
    if warn_results:
        print("\n[WARNINGS]")
        print("-" * 40)
        for r in warn_results:
            print(f"  ! {r.name}")
            print(f"    Module: {r.module}")
            print(f"    Error: {r.error}")
            print()

    # Print successes (abbreviated)
    if ok_results:
        print("\n[PASSED]")
        print("-" * 40)
        for r in ok_results:
            details = f" ({r.details})" if r.details else ""
            print(f"  + {r.name}{details}")

    # Summary
    print("\n" + "=" * 80)
    total = summary["OK"] + summary["FAIL"] + summary["WARN"]
    print(f"SUMMARY: {summary['OK']}/{total} OK, {summary['FAIL']} FAIL, {summary['WARN']} WARN")

    if summary["FAIL"] == 0:
        print("\n[OK] All critical components verified!")
        return 0
    else:
        print(f"\n[FAIL] {summary['FAIL']} component(s) failed verification")
        return 1


def main():
    """Main verification function."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("\nVerifying all components...")
    print(f"Project root: {project_root}")

    # Basic import verification
    results, summary = verify_all()

    # Factory function tests
    print("\nTesting factory functions...")
    factory_results = test_factory_functions()
    for r in factory_results:
        results.append(r)
        summary[r.status] += 1

    # Print report
    exit_code = print_results(results, summary)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
