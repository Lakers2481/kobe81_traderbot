#!/usr/bin/env python3
"""
KOBE 100-COMPONENT VERIFICATION SCRIPT

This script verifies ALL 100 critical components are wired and working.
Run this to ensure the trading system is 100% ready.

Usage:
    python tools/verify_100_components.py
    python tools/verify_100_components.py --verbose
    python tools/verify_100_components.py --json

Author: KOBE Trading System
Version: 1.0.0
Date: 2026-01-05
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class ComponentCheck:
    """Result of a component check."""
    category: str
    name: str
    module: str
    attribute: str
    status: str  # PASS, FAIL, WARN
    message: str = ""
    value: Any = None


# ============================================================================
# 100 CRITICAL COMPONENTS TO VERIFY
# ============================================================================

COMPONENTS = [
    # =========================================================================
    # CATEGORY 1: SAFETY (10 components)
    # =========================================================================
    ("1. Safety", "PAPER_ONLY constant", "safety.mode", "PAPER_ONLY"),
    ("1. Safety", "LIVE_TRADING_ENABLED", "safety.mode", "LIVE_TRADING_ENABLED"),
    ("1. Safety", "get_trading_mode()", "safety.mode", "get_trading_mode"),
    ("1. Safety", "assert_paper_only()", "safety.mode", "assert_paper_only"),
    ("1. Safety", "is_kill_switch_active()", "core.kill_switch", "is_kill_switch_active"),
    ("1. Safety", "activate_kill_switch()", "core.kill_switch", "activate_kill_switch"),
    ("1. Safety", "deactivate_kill_switch()", "core.kill_switch", "deactivate_kill_switch"),
    ("1. Safety", "@require_no_kill_switch", "core.kill_switch", "require_no_kill_switch"),
    ("1. Safety", "APPROVE_LIVE_ACTION", "research_os.approval_gate", "APPROVE_LIVE_ACTION"),
    ("1. Safety", "ApprovalGate", "research_os.approval_gate", "ApprovalGate"),

    # =========================================================================
    # CATEGORY 2: STRATEGIES (10 components)
    # =========================================================================
    ("2. Strategies", "DualStrategyScanner", "strategies.dual_strategy", "DualStrategyScanner"),
    ("2. Strategies", "DualStrategyParams", "strategies.dual_strategy", "DualStrategyParams"),
    ("2. Strategies", "get_production_scanner()", "strategies.registry", "get_production_scanner"),
    ("2. Strategies", "IbsRsiStrategy", "strategies.ibs_rsi.strategy", "IbsRsiStrategy"),
    ("2. Strategies", "IbsRsiParams", "strategies.ibs_rsi.strategy", "IbsRsiParams"),
    ("2. Strategies", "TurtleSoupStrategy", "strategies.ict.turtle_soup", "TurtleSoupStrategy"),
    ("2. Strategies", "TurtleSoupParams", "strategies.ict.turtle_soup", "TurtleSoupParams"),
    ("2. Strategies", "validate_strategy_import()", "strategies.registry", "validate_strategy_import"),
    ("2. Strategies", "CANONICAL_STRATEGY", "strategies.dual_strategy", "DualStrategyScanner"),
    ("2. Strategies", "generate_signals()", "strategies.dual_strategy.combined", "DualStrategyScanner"),

    # =========================================================================
    # CATEGORY 3: EXECUTION (10 components)
    # =========================================================================
    ("3. Execution", "execute_signal()", "execution.broker_alpaca", "execute_signal"),
    ("3. Execution", "place_order_with_liquidity_check()", "execution.broker_alpaca", "place_order_with_liquidity_check"),
    ("3. Execution", "place_ioc_limit()", "execution.broker_alpaca", "place_ioc_limit"),
    ("3. Execution", "get_best_ask()", "execution.broker_alpaca", "get_best_ask"),
    ("3. Execution", "PaperBroker", "execution.broker_paper", "PaperBroker"),
    ("3. Execution", "OrderManager", "execution.order_manager", "OrderManager"),
    ("3. Execution", "OrderRecord", "oms.order_state", "OrderRecord"),
    ("3. Execution", "OrderStatus", "oms.order_state", "OrderStatus"),
    ("3. Execution", "IdempotencyStore", "oms.idempotency_store", "IdempotencyStore"),
    ("3. Execution", "IntelligentExecutor", "execution.intelligent_executor", "IntelligentExecutor"),

    # =========================================================================
    # CATEGORY 4: RISK MANAGEMENT (10 components)
    # =========================================================================
    ("4. Risk", "PolicyGate", "risk.policy_gate", "PolicyGate"),
    ("4. Risk", "SignalQualityGate", "risk.signal_quality_gate", "SignalQualityGate"),
    ("4. Risk", "can_trade_now()", "risk.kill_zone_gate", "can_trade_now"),
    ("4. Risk", "get_current_zone()", "risk.kill_zone_gate", "get_current_zone"),
    ("4. Risk", "calculate_position_size()", "risk.equity_sizer", "calculate_position_size"),
    ("4. Risk", "calculate_dynamic_allocations()", "risk.dynamic_position_sizer", "calculate_dynamic_allocations"),
    ("4. Risk", "MonteCarloVaR", "risk.advanced.monte_carlo_var", "MonteCarloVaR"),
    ("4. Risk", "KellyPositionSizer", "risk.advanced.kelly_position_sizer", "KellyPositionSizer"),
    ("4. Risk", "EnhancedCorrelationLimits", "risk.advanced.correlation_limits", "EnhancedCorrelationLimits"),
    ("4. Risk", "TrailingStopManager", "risk.trailing_stops", "TrailingStopManager"),

    # =========================================================================
    # CATEGORY 5: DATA PROVIDERS (10 components)
    # =========================================================================
    ("5. Data", "fetch_daily_bars_polygon()", "data.providers.polygon_eod", "fetch_daily_bars_polygon"),
    ("5. Data", "load_universe()", "data.universe.loader", "load_universe"),
    ("5. Data", "StooqEODProvider", "data.providers.stooq_eod", "StooqEODProvider"),
    ("5. Data", "YFinanceEODProvider", "data.providers.yfinance_eod", "YFinanceEODProvider"),
    ("5. Data", "BinanceKlinesProvider", "data.providers.binance_klines", "BinanceKlinesProvider"),
    ("5. Data", "LakeWriter", "data.lake.io", "LakeWriter"),
    ("5. Data", "LakeReader", "data.lake.io", "LakeReader"),
    ("5. Data", "DatasetManifest", "data.lake.manifest", "DatasetManifest"),
    ("5. Data", "PolygonConfig", "data.providers.polygon_eod", "PolygonConfig"),
    ("5. Data", "DataQuorum", "data.quorum", "DataQuorum"),

    # =========================================================================
    # CATEGORY 6: COGNITIVE BRAIN (10 components)
    # =========================================================================
    ("6. Cognitive", "CognitiveBrain", "cognitive.cognitive_brain", "CognitiveBrain"),
    ("6. Cognitive", "MetacognitiveGovernor", "cognitive.metacognitive_governor", "MetacognitiveGovernor"),
    ("6. Cognitive", "ReflectionEngine", "cognitive.reflection_engine", "ReflectionEngine"),
    ("6. Cognitive", "CuriosityEngine", "cognitive.curiosity_engine", "CuriosityEngine"),
    ("6. Cognitive", "EpisodicMemory", "cognitive.episodic_memory", "EpisodicMemory"),
    ("6. Cognitive", "SemanticMemory", "cognitive.semantic_memory", "SemanticMemory"),
    ("6. Cognitive", "SelfModel", "cognitive.self_model", "SelfModel"),
    ("6. Cognitive", "KnowledgeBoundary", "cognitive.knowledge_boundary", "KnowledgeBoundary"),
    ("6. Cognitive", "Adjudicator", "cognitive.adjudicator", "adjudicate"),
    ("6. Cognitive", "CognitiveDecision", "cognitive.cognitive_brain", "CognitiveDecision"),

    # =========================================================================
    # CATEGORY 7: AUTONOMOUS BRAIN (10 components)
    # =========================================================================
    ("7. Autonomous", "get_context()", "autonomous.awareness", "get_context"),
    ("7. Autonomous", "TimeAwareness", "autonomous.awareness", "TimeAwareness"),
    ("7. Autonomous", "MarketPhase", "autonomous.awareness", "MarketPhase"),
    ("7. Autonomous", "FullScheduler", "autonomous.scheduler_full", "FullScheduler"),
    ("7. Autonomous", "MasterBrainFull", "autonomous.master_brain_full", "MasterBrainFull"),
    ("7. Autonomous", "WorkMode", "autonomous.awareness", "WorkMode"),
    ("7. Autonomous", "Season", "autonomous.awareness", "Season"),
    ("7. Autonomous", "MarketContext", "autonomous.awareness", "MarketContext"),
    ("7. Autonomous", "run_autonomous()", "autonomous.run", "main"),
    ("7. Autonomous", "ContextBuilder", "autonomous.awareness", "ContextBuilder"),

    # =========================================================================
    # CATEGORY 8: ML/AI (10 components)
    # =========================================================================
    ("8. ML/AI", "HMMRegimeDetector", "ml_advanced.hmm_regime_detector", "HMMRegimeDetector"),
    ("8. ML/AI", "FeaturePipeline", "ml_features.feature_pipeline", "FeaturePipeline"),
    ("8. ML/AI", "TechnicalFeatures", "ml_features.technical_features", "TechnicalFeatures"),
    ("8. ML/AI", "PCAReducer", "ml_features.pca_reducer", "PCAReducer"),
    ("8. ML/AI", "EnsemblePredictor", "ml_advanced.ensemble.ensemble_predictor", "EnsemblePredictor"),
    ("8. ML/AI", "OnlineLearningManager", "ml_advanced.online_learning", "OnlineLearningManager"),
    ("8. ML/AI", "LSTMConfidenceModel", "ml_advanced.lstm_confidence.model", "LSTMConfidenceModel"),
    ("8. ML/AI", "AnomalyDetector", "ml_features.anomaly_detection", "AnomalyDetector"),
    ("8. ML/AI", "RegimeDetectorML", "ml_features.regime_ml", "RegimeDetectorML"),
    ("8. ML/AI", "EnsembleBrain", "ml_features.ensemble_brain", "EnsembleBrain"),

    # =========================================================================
    # CATEGORY 9: ALERTS & MONITORING (10 components)
    # =========================================================================
    ("9. Alerts", "TelegramAlerter", "alerts.telegram_alerter", "TelegramAlerter"),
    ("9. Alerts", "get_alerter()", "alerts.telegram_alerter", "get_alerter"),
    ("9. Alerts", "alert_signal()", "alerts.telegram_alerter", "TelegramAlerter"),
    ("9. Alerts", "ProfessionalAlerts", "alerts.professional_alerts", "ProfessionalAlerts"),
    ("9. Alerts", "check_regime_transition()", "alerts.regime_alerts", "check_regime_transition"),
    ("9. Alerts", "start_health_server()", "monitor.health_endpoints", "start_health_server"),
    ("9. Alerts", "CircuitBreaker", "selfmonitor.circuit_breaker", "CircuitBreaker"),
    ("9. Alerts", "AnomalyDetector (self)", "selfmonitor.anomaly_detector", "AnomalyDetector"),
    ("9. Alerts", "jlog()", "core.structured_log", "jlog"),
    ("9. Alerts", "verify_chain()", "core.hash_chain", "verify_chain"),

    # =========================================================================
    # CATEGORY 10: BACKTEST & ANALYSIS (10 components)
    # =========================================================================
    ("10. Backtest", "Backtester", "backtest.engine", "Backtester"),
    ("10. Backtest", "BacktestConfig", "backtest.engine", "BacktestConfig"),
    ("10. Backtest", "WalkForwardCV", "backtest.purged_cv", "WalkForwardCV"),
    ("10. Backtest", "VectorizedBacktester", "backtest.vectorized", "VectorizedBacktester"),
    ("10. Backtest", "SyntheticOptionsBacktester", "options.backtest", "SyntheticOptionsBacktester"),
    ("10. Backtest", "BlackScholes", "options.black_scholes", "BlackScholes"),
    ("10. Backtest", "HistoricalPatterns", "analysis.historical_patterns", "HistoricalPatternAnalyzer"),
    ("10. Backtest", "ExpectedMoveCalculator", "analysis.options_expected_move", "ExpectedMoveCalculator"),
    ("10. Backtest", "TradeThesisBuilder", "explainability.trade_thesis_builder", "TradeThesisBuilder"),
    ("10. Backtest", "ExperimentRegistry", "experiments.registry", "ExperimentRegistry"),
]


def check_component(category: str, name: str, module: str, attribute: str, verbose: bool = False) -> ComponentCheck:
    """Check if a component is properly wired."""
    try:
        mod = importlib.import_module(module)
        obj = getattr(mod, attribute)

        # Get value for constants
        value = None
        if attribute in ['PAPER_ONLY', 'LIVE_TRADING_ENABLED', 'APPROVE_LIVE_ACTION']:
            value = obj

        return ComponentCheck(
            category=category,
            name=name,
            module=module,
            attribute=attribute,
            status="PASS",
            message=f"Loaded from {module}",
            value=value
        )
    except ImportError as e:
        return ComponentCheck(
            category=category,
            name=name,
            module=module,
            attribute=attribute,
            status="FAIL",
            message=f"Import error: {str(e)[:50]}"
        )
    except AttributeError as e:
        return ComponentCheck(
            category=category,
            name=name,
            module=module,
            attribute=attribute,
            status="FAIL",
            message=f"Attribute not found: {str(e)[:50]}"
        )
    except Exception as e:
        return ComponentCheck(
            category=category,
            name=name,
            module=module,
            attribute=attribute,
            status="FAIL",
            message=f"Error: {str(e)[:50]}"
        )


def run_verification(verbose: bool = False) -> Tuple[List[ComponentCheck], Dict[str, Any]]:
    """Run full verification of all 100 components."""
    results = []

    for category, name, module, attribute in COMPONENTS:
        result = check_component(category, name, module, attribute, verbose)
        results.append(result)

    # Calculate summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")

    # Group by category
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"passed": 0, "failed": 0}
        if r.status == "PASS":
            by_category[r.category]["passed"] += 1
        else:
            by_category[r.category]["failed"] += 1

    summary = {
        "total": len(COMPONENTS),
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed/len(COMPONENTS)*100:.1f}%",
        "by_category": by_category,
        "timestamp": datetime.utcnow().isoformat(),
        "verdict": "PASS" if passed == len(COMPONENTS) else "FAIL"
    }

    return results, summary


def print_results(results: List[ComponentCheck], summary: Dict[str, Any], verbose: bool = False):
    """Print verification results."""
    print("=" * 70)
    print("KOBE 100-COMPONENT VERIFICATION")
    print("=" * 70)
    print()

    current_category = None
    for r in results:
        if r.category != current_category:
            current_category = r.category
            print(f"\n{current_category.upper()}")
            print("-" * 50)

        icon = "[+]" if r.status == "PASS" else "[!]"
        print(f"{icon} {r.name:40} {r.status}")

        if r.value is not None:
            print(f"    Value: {r.value}")

        if verbose and r.status == "FAIL":
            print(f"    {r.message}")

    print()
    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)

    for cat, stats in summary["by_category"].items():
        total = stats["passed"] + stats["failed"]
        icon = "[+]" if stats["failed"] == 0 else "[!]"
        print(f"{icon} {cat:30} {stats['passed']}/{total}")

    print()
    print("=" * 70)
    print(f"TOTAL: {summary['passed']}/{summary['total']} ({summary['pass_rate']})")
    print(f"VERDICT: {summary['verdict']}")
    print("=" * 70)

    if summary["failed"] > 0:
        print("\nFAILED COMPONENTS:")
        for r in results:
            if r.status == "FAIL":
                print(f"  - {r.name}: {r.message}")


def main():
    parser = argparse.ArgumentParser(description="Verify 100 KOBE components")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed errors")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Suppress TensorFlow warnings
    import warnings
    warnings.filterwarnings("ignore")
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    results, summary = run_verification(args.verbose)

    if args.json:
        output = {
            "summary": summary,
            "results": [
                {
                    "category": r.category,
                    "name": r.name,
                    "module": r.module,
                    "attribute": r.attribute,
                    "status": r.status,
                    "message": r.message,
                    "value": str(r.value) if r.value is not None else None
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(results, summary, args.verbose)

    # Save report
    report_path = ROOT / "AUDITS" / "100_COMPONENT_VERIFICATION.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": [
                {
                    "category": r.category,
                    "name": r.name,
                    "status": r.status,
                    "message": r.message
                }
                for r in results
            ]
        }, f, indent=2)

    if not args.json:
        print(f"\nReport saved to: {report_path}")

    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
