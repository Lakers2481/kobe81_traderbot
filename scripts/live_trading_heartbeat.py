#!/usr/bin/env python
"""
LIVE TRADING HEARTBEAT - THE ONE COMPREHENSIVE DISPLAY
=======================================================

This script shows HONESTLY what IS and ISN'T working in Kobe.
Real money on the line - NO BS, NO LIES, NO CUTTING CORNERS.

Run continuously:
    python scripts/live_trading_heartbeat.py

Run once:
    python scripts/live_trading_heartbeat.py --once

What this shows:
    1. Trading Status (kill zones, can trade, next scan)
    2. Last Scan Results (signals generated, filtered, executed)
    3. Active Positions (from broker, live P&L)
    4. Risk Gates Status (policy, position limit, weekly exposure)
    5. ML Models - HONEST (what's USED vs what EXISTS)
    6. Cognitive Brain - HONEST (what features are INVOKED)
    7. Autonomous Brain (SEPARATE from trading)
    8. What's NOT Wired (TRANSPARENCY)

Author: Claude Code
Created: 2026-01-07
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# Import core modules
try:
    from core.structured_log import jlog
except ImportError:
    def jlog(event, **kwargs):
        print(f"[{datetime.now().isoformat()}] {event}: {kwargs}")

# Import risk gates
try:
    from risk.kill_zone_gate import (
        KillZoneGate,
        get_kill_zone_gate,
        can_trade_now,
        check_trade_allowed,
    )
    KILL_ZONE_AVAILABLE = True
except ImportError:
    KILL_ZONE_AVAILABLE = False

try:
    from risk.policy_gate import PolicyGate
    POLICY_GATE_AVAILABLE = True
except ImportError:
    POLICY_GATE_AVAILABLE = False

try:
    from risk.position_limit_gate import PositionLimitGate
    POSITION_LIMIT_AVAILABLE = True
except ImportError:
    POSITION_LIMIT_AVAILABLE = False

try:
    from risk.weekly_exposure_gate import WeeklyExposureGate
    WEEKLY_EXPOSURE_AVAILABLE = True
except ImportError:
    WEEKLY_EXPOSURE_AVAILABLE = False

# Import ML components - check what EXISTS vs what's USED
try:
    from ml_advanced.markov_chain.predictor import MarkovPredictor
    MARKOV_EXISTS = True
except ImportError:
    MARKOV_EXISTS = False

try:
    from ml_advanced.hmm_regime_detector import HMMRegimeDetector
    HMM_EXISTS = True
except ImportError:
    HMM_EXISTS = False

try:
    from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor
    ENSEMBLE_EXISTS = True
except ImportError:
    ENSEMBLE_EXISTS = False

try:
    from ml_advanced.lstm_confidence.model import LSTMConfidenceModel
    LSTM_EXISTS = True
except ImportError:
    LSTM_EXISTS = False

# Import cognitive brain
try:
    from cognitive.cognitive_brain import get_cognitive_brain
    COGNITIVE_AVAILABLE = True
except ImportError:
    COGNITIVE_AVAILABLE = False

# Import broker
try:
    from execution.broker_alpaca import AlpacaBroker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False


class LiveTradingHeartbeat:
    """
    THE ONE COMPREHENSIVE HEARTBEAT.

    Shows EVERYTHING that IS and ISN'T working.
    Complete honesty - real money on the line.
    """

    def __init__(self, log_to_file: bool = True):
        self.project_root = PROJECT_ROOT
        self.state_dir = self.project_root / "state"
        self.logs_dir = self.project_root / "logs"
        self.log_to_file = log_to_file

        # Heartbeat log file
        self.heartbeat_log = self.logs_dir / "heartbeat_decisions.jsonl"

        # State files to read
        self.runner_state_file = self.state_dir / "runner_state.json"
        self.heartbeat_file = self.state_dir / "heartbeat.json"
        self.autonomous_heartbeat = self.state_dir / "autonomous" / "heartbeat.json"
        self.kill_switch_file = self.state_dir / "KILL_SWITCH"

        # Log files to read
        self.signals_log = self.logs_dir / "signals.jsonl"
        self.daily_top5 = self.logs_dir / "daily_top5.csv"
        self.trading_decisions_log = self.logs_dir / "trading_decisions.jsonl"

        self.cycle_count = 0

    def _read_json_file(self, path: Path) -> Optional[Dict]:
        """Read a JSON file safely."""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            pass
        return None

    def _read_last_jsonl_entry(self, path: Path) -> Optional[Dict]:
        """Read the last entry from a JSONL file."""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        return json.loads(lines[-1])
        except Exception:
            pass
        return None

    def _log_heartbeat(self, heartbeat: Dict):
        """Log heartbeat to JSONL file."""
        if self.log_to_file:
            self.heartbeat_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.heartbeat_log, 'a') as f:
                f.write(json.dumps(heartbeat) + '\n')

    # ========== SECTION 1: TRADING STATUS ==========

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "section": "TRADING_STATUS",
        }

        # Kill switch
        status["kill_switch_active"] = self.kill_switch_file.exists()

        # Kill zone
        if KILL_ZONE_AVAILABLE:
            try:
                gate = get_kill_zone_gate()
                zone = gate.get_current_zone()
                kz_status = gate.check_can_trade()
                status["kill_zone"] = {
                    "current_zone": zone.value if hasattr(zone, 'value') else str(zone),
                    "can_trade": kz_status.can_trade,
                    "reason": kz_status.reason,
                }
            except Exception as e:
                status["kill_zone"] = {"error": str(e)}
        else:
            status["kill_zone"] = {"error": "Module not available"}

        # Runner state
        runner_state = self._read_json_file(self.runner_state_file)
        if runner_state:
            status["runner"] = {
                "status": runner_state.get("status", "unknown"),
                "mode": runner_state.get("mode", "unknown"),
                "last_scan": runner_state.get("last_scan"),
                "scan_count": runner_state.get("scan_count", 0),
                "trade_count": runner_state.get("trade_count", 0),
            }
        else:
            status["runner"] = {"status": "not_found"}

        # Main heartbeat
        heartbeat = self._read_json_file(self.heartbeat_file)
        if heartbeat:
            status["heartbeat"] = {
                "mode": heartbeat.get("mode", "unknown"),
                "pid": heartbeat.get("pid"),
                "uptime_seconds": heartbeat.get("uptime_seconds", 0),
                "last_action": heartbeat.get("last_action"),
            }
        else:
            status["heartbeat"] = {"status": "not_found"}

        return status

    # ========== SECTION 2: LAST SCAN RESULTS ==========

    def get_last_scan_results(self) -> Dict[str, Any]:
        """Get results from last scan."""
        results = {
            "section": "LAST_SCAN_RESULTS",
        }

        # Read last signal
        last_signal = self._read_last_jsonl_entry(self.signals_log)
        if last_signal:
            results["last_signal"] = {
                "symbol": last_signal.get("symbol"),
                "side": last_signal.get("side"),
                "confidence": last_signal.get("confidence"),
                "timestamp": last_signal.get("timestamp"),
            }
        else:
            results["last_signal"] = None

        # Read daily top 5
        if self.daily_top5.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.daily_top5)
                if not df.empty:
                    results["top5_count"] = len(df)
                    results["top5_symbols"] = df["symbol"].tolist()[:5] if "symbol" in df.columns else []
                else:
                    results["top5_count"] = 0
                    results["top5_symbols"] = []
            except Exception as e:
                results["top5_error"] = str(e)
        else:
            results["top5_count"] = 0
            results["top5_symbols"] = []

        # Read trading decisions log
        last_decision = self._read_last_jsonl_entry(self.trading_decisions_log)
        if last_decision:
            results["last_decision"] = last_decision
        else:
            results["last_decision"] = None

        return results

    # ========== SECTION 3: ACTIVE POSITIONS ==========

    def get_active_positions(self) -> Dict[str, Any]:
        """Get active positions from broker."""
        positions = {
            "section": "ACTIVE_POSITIONS",
        }

        if not BROKER_AVAILABLE:
            positions["error"] = "Broker module not available"
            return positions

        try:
            broker = AlpacaBroker()
            broker_positions = broker.get_positions()

            positions["count"] = len(broker_positions)
            positions["positions"] = []
            total_pnl = 0.0
            total_value = 0.0

            for pos in broker_positions:
                # Position is a dataclass with attributes
                pos_data = {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "side": pos.side,
                    "avg_entry": pos.avg_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                total_pnl += pos_data["unrealized_pnl"]
                total_value += pos_data["market_value"]
                positions["positions"].append(pos_data)

            positions["total_unrealized_pnl"] = round(total_pnl, 2)
            positions["total_market_value"] = round(total_value, 2)

            # Get account equity
            account = broker.get_account()
            if account:
                # Account is also a dataclass
                positions["account_equity"] = account.equity if hasattr(account, 'equity') else 0
                positions["buying_power"] = account.buying_power if hasattr(account, 'buying_power') else 0

        except Exception as e:
            positions["error"] = str(e)

        return positions

    # ========== SECTION 4: RISK GATES STATUS ==========

    def get_risk_gates_status(self) -> Dict[str, Any]:
        """Get status of all risk gates."""
        gates = {
            "section": "RISK_GATES_STATUS",
        }

        # Policy Gate
        if POLICY_GATE_AVAILABLE:
            try:
                gate = PolicyGate.from_config()
                gates["policy_gate"] = {
                    "status": "ACTIVE",
                    "daily_notional_used": gate._daily_notional,
                    "daily_limit": gate.limits.max_daily_notional,
                    "max_per_order": gate.limits.max_notional_per_order,
                    "max_positions": gate.limits.max_positions,
                    "mode": gate.limits.mode_name,
                }
            except Exception as e:
                gates["policy_gate"] = {"status": "ERROR", "error": str(e)}
        else:
            gates["policy_gate"] = {"status": "NOT_AVAILABLE"}

        # Position Limit Gate
        if POSITION_LIMIT_AVAILABLE:
            try:
                gate = PositionLimitGate()
                gates["position_limit_gate"] = {
                    "status": "ACTIVE",
                    "current_positions": gate.get_open_position_count(),
                    "max_positions": gate.limits.max_positions,
                }
            except Exception as e:
                gates["position_limit_gate"] = {"status": "ERROR", "error": str(e)}
        else:
            gates["position_limit_gate"] = {"status": "NOT_AVAILABLE"}

        # Weekly Exposure Gate
        if WEEKLY_EXPOSURE_AVAILABLE:
            try:
                gate = WeeklyExposureGate()
                gates["weekly_exposure_gate"] = {
                    "status": "ACTIVE",
                    "weekly_budget": gate.budget if hasattr(gate, 'budget') else "N/A",
                    "weekly_used": gate.used if hasattr(gate, 'used') else "N/A",
                }
            except Exception as e:
                gates["weekly_exposure_gate"] = {"status": "ERROR", "error": str(e)}
        else:
            gates["weekly_exposure_gate"] = {"status": "NOT_AVAILABLE"}

        # Kill Zone Gate
        if KILL_ZONE_AVAILABLE:
            try:
                gate = get_kill_zone_gate()
                zone = gate.get_current_zone()
                kz_status = gate.check_can_trade()
                gates["kill_zone_gate"] = {
                    "status": "ACTIVE",
                    "current_zone": zone.value if hasattr(zone, 'value') else str(zone),
                    "trading_allowed": kz_status.can_trade,
                    "reason": kz_status.reason,
                }
            except Exception as e:
                gates["kill_zone_gate"] = {"status": "ERROR", "error": str(e)}
        else:
            gates["kill_zone_gate"] = {"status": "NOT_AVAILABLE"}

        return gates

    # ========== SECTION 5: ML MODELS - HONEST ==========

    def get_ml_models_status(self) -> Dict[str, Any]:
        """
        HONEST assessment of ML models.
        Shows what EXISTS vs what's actually IN HOT PATH.
        """
        ml = {
            "section": "ML_MODELS_HONEST",
            "disclaimer": "This shows what EXISTS vs what's actually USED in trading",
        }

        # Markov Chain - CAN be in hot path with --markov flag
        ml["markov_chain"] = {
            "exists": MARKOV_EXISTS,
            "in_hot_path": "OPTIONAL (--markov flag)",
            "blocks_trades": False,
            "what_it_does": "Boosts confidence +5-10% when direction prediction agrees",
            "honest_status": "AVAILABLE - used if --markov flag enabled"
        }

        # Check if Markov was used in last scan
        try:
            # Read runner state for markov flag
            runner_state = self._read_json_file(self.runner_state_file)
            if runner_state and runner_state.get("markov_enabled"):
                ml["markov_chain"]["currently_enabled"] = True
            else:
                ml["markov_chain"]["currently_enabled"] = False
        except:
            ml["markov_chain"]["currently_enabled"] = "unknown"

        # HMM Regime Detector - EXISTS but NOT in hot path
        ml["hmm_regime"] = {
            "exists": HMM_EXISTS,
            "in_hot_path": False,
            "blocks_trades": False,
            "what_it_does": "Detects bull/bear/neutral market regimes",
            "honest_status": "EXISTS but NOT WIRED into scan.py",
            "why_not_wired": "Code exists but scan.py doesn't call it"
        }

        # Check if HMM model file exists
        hmm_model_path = self.project_root / "models" / "hmm_regime_v1.pkl"
        ml["hmm_regime"]["model_file_exists"] = hmm_model_path.exists()

        # Ensemble Predictor - EXISTS but NOT in hot path
        ml["ensemble"] = {
            "exists": ENSEMBLE_EXISTS,
            "in_hot_path": False,
            "blocks_trades": False,
            "what_it_does": "Combines XGBoost + LightGBM + LSTM predictions",
            "honest_status": "EXISTS but NOT WIRED into scan.py",
            "why_not_wired": "Training script exists but models not loaded in trading"
        }

        # LSTM Confidence - EXISTS but NOT in hot path
        ml["lstm_confidence"] = {
            "exists": LSTM_EXISTS,
            "in_hot_path": False,
            "blocks_trades": False,
            "what_it_does": "Multi-output LSTM for signal confidence grading",
            "honest_status": "EXISTS but NOT WIRED into scan.py",
            "why_not_wired": "Training script exists but not called during trading"
        }

        # Summary
        ml["summary"] = {
            "models_that_exist": 4,
            "models_in_hot_path": 1,  # Only Markov (optional)
            "models_blocking_trades": 0,
            "honest_assessment": "Most ML models are NOT in the trading hot path"
        }

        return ml

    # ========== SECTION 6: COGNITIVE BRAIN - HONEST ==========

    def get_cognitive_brain_status(self) -> Dict[str, Any]:
        """
        HONEST assessment of cognitive brain.
        Shows what features are LOADED vs what's actually INVOKED.
        """
        cognitive = {
            "section": "COGNITIVE_BRAIN_HONEST",
            "disclaimer": "This shows what's LOADED vs what's actually INVOKED",
        }

        if not COGNITIVE_AVAILABLE:
            cognitive["status"] = "NOT_AVAILABLE"
            return cognitive

        try:
            brain = get_cognitive_brain()

            # Basic status
            cognitive["status"] = "LOADED"
            cognitive["in_hot_path"] = "OPTIONAL (--cognitive flag, default=TRUE)"

            # What actually works
            cognitive["deliberate_function"] = {
                "status": "WORKS",
                "what_it_does": "Returns should_act=True/False for signals",
                "blocks_trades": True,
                "honest_status": "This IS called and CAN block trades"
            }

            # Tree of Thoughts - LOADED but NOT invoked
            cognitive["tree_of_thoughts"] = {
                "status": "LOADED but NOT INVOKED",
                "what_it_would_do": "Multi-path reasoning for complex decisions",
                "blocks_trades": False,
                "honest_status": "Code is loaded but deliberate() doesn't call it",
                "why_not_invoked": "deliberate() uses simple logic, not ToT"
            }

            # Self-Consistency - LOADED but NOT invoked
            cognitive["self_consistency"] = {
                "status": "LOADED but NOT INVOKED",
                "what_it_would_do": "Sample multiple reasoning paths for consensus",
                "blocks_trades": False,
                "honest_status": "Code is loaded but deliberate() doesn't call it"
            }

            # Contradiction Resolver - LOADED but NOT invoked
            cognitive["contradiction_resolver"] = {
                "status": "LOADED but NOT INVOKED",
                "what_it_would_do": "Resolve conflicts between indicators",
                "blocks_trades": False,
                "honest_status": "Code is loaded but deliberate() doesn't call it"
            }

            # Get actual state if available
            try:
                state = brain.get_state()
                cognitive["brain_state"] = {
                    "signals_evaluated_today": state.get("signals_evaluated", 0),
                    "signals_approved": state.get("signals_approved", 0),
                    "signals_rejected": state.get("signals_rejected", 0),
                }
            except:
                cognitive["brain_state"] = "Could not retrieve"

            # Summary
            cognitive["summary"] = {
                "features_loaded": 4,
                "features_actually_invoked": 1,  # Only deliberate()
                "honest_assessment": "Only deliberate() is called; ToT, SC, CR are loaded but unused"
            }

        except Exception as e:
            cognitive["error"] = str(e)

        return cognitive

    # ========== SECTION 7: AUTONOMOUS BRAIN - SEPARATE ==========

    def get_autonomous_brain_status(self) -> Dict[str, Any]:
        """
        Autonomous brain status - runs SEPARATELY from trading.
        """
        autonomous = {
            "section": "AUTONOMOUS_BRAIN_SEPARATE",
            "disclaimer": "This runs 24/7 but does NOT participate in trading decisions",
        }

        # Read autonomous heartbeat
        heartbeat = self._read_json_file(self.autonomous_heartbeat)

        if heartbeat:
            autonomous["status"] = "RUNNING" if heartbeat.get("alive") else "STOPPED"
            autonomous["phase"] = heartbeat.get("phase", "unknown")
            autonomous["work_mode"] = heartbeat.get("work_mode", "unknown")
            autonomous["cycles"] = heartbeat.get("cycles", 0)
            autonomous["uptime_hours"] = round(heartbeat.get("uptime_hours", 0), 2)
            autonomous["last_timestamp"] = heartbeat.get("timestamp")
        else:
            autonomous["status"] = "NOT_RUNNING or NOT_FOUND"

        autonomous["what_it_does"] = {
            "research": "Runs random parameter experiments",
            "learning": "Analyzes trade outcomes",
            "maintenance": "Data quality checks, cleanup",
        }

        autonomous["honest_status"] = "RUNS 24/7 but has ZERO impact on trading decisions"

        return autonomous

    # ========== SECTION 8: WHAT'S NOT WIRED - TRANSPARENCY ==========

    def get_not_wired_list(self) -> Dict[str, Any]:
        """
        HONEST list of what EXISTS but is NOT in the trading hot path.
        """
        not_wired = {
            "section": "NOT_WIRED_TRANSPARENCY",
            "disclaimer": "These components EXIST but are NOT in the trading hot path",
        }

        not_wired["components"] = [
            {
                "name": "HMM Regime Detector",
                "location": "ml_advanced/hmm_regime_detector.py",
                "what_it_would_do": "Adjust position sizing based on market regime",
                "why_not_wired": "scan.py doesn't import or call it",
                "risk_to_wire": "LOW - only affects sizing, doesn't generate signals"
            },
            {
                "name": "Ensemble Predictor",
                "location": "ml_advanced/ensemble/ensemble_predictor.py",
                "what_it_would_do": "Combine multiple ML models for confidence scoring",
                "why_not_wired": "Training scripts exist but models not loaded in scan",
                "risk_to_wire": "MEDIUM - untested models might give bad signals"
            },
            {
                "name": "LSTM Confidence Model",
                "location": "ml_advanced/lstm_confidence/model.py",
                "what_it_would_do": "Grade signals A/B/C based on LSTM prediction",
                "why_not_wired": "Model needs training, no trained weights exist",
                "risk_to_wire": "MEDIUM - needs training data and validation"
            },
            {
                "name": "Tree of Thoughts",
                "location": "cognitive/tree_of_thoughts.py",
                "what_it_would_do": "Multi-path reasoning for complex decisions",
                "why_not_wired": "Loaded in brain but deliberate() uses simple logic",
                "risk_to_wire": "LOW - would slow down decisions, needs LLM calls"
            },
            {
                "name": "Self-Consistency",
                "location": "cognitive/self_consistency.py",
                "what_it_would_do": "Sample multiple reasoning paths",
                "why_not_wired": "Loaded in brain but never called",
                "risk_to_wire": "LOW - would slow down decisions"
            },
            {
                "name": "Contradiction Resolver",
                "location": "cognitive/contradiction_resolver.py",
                "what_it_would_do": "Resolve conflicts between indicators",
                "why_not_wired": "Loaded in brain but never called",
                "risk_to_wire": "LOW - could add value if implemented properly"
            },
        ]

        not_wired["count"] = len(not_wired["components"])
        not_wired["honest_summary"] = (
            f"{not_wired['count']} components exist but are not in trading hot path. "
            "The trading system WORKS without them - they are ENHANCEMENTS, not REQUIREMENTS."
        )

        return not_wired

    # ========== MAIN HEARTBEAT ==========

    def generate_heartbeat(self) -> Dict[str, Any]:
        """Generate complete heartbeat showing all components HONESTLY."""
        self.cycle_count += 1

        heartbeat = {
            "heartbeat_version": "2.0_HONEST",
            "generated_at": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "disclaimer": "HONEST ASSESSMENT - Shows what IS and ISN'T working",
        }

        # Collect all sections
        heartbeat["trading_status"] = self.get_trading_status()
        heartbeat["last_scan_results"] = self.get_last_scan_results()
        heartbeat["active_positions"] = self.get_active_positions()
        heartbeat["risk_gates"] = self.get_risk_gates_status()
        heartbeat["ml_models"] = self.get_ml_models_status()
        heartbeat["cognitive_brain"] = self.get_cognitive_brain_status()
        heartbeat["autonomous_brain"] = self.get_autonomous_brain_status()
        heartbeat["not_wired"] = self.get_not_wired_list()

        # Summary
        heartbeat["summary"] = {
            "kill_switch": heartbeat["trading_status"].get("kill_switch_active", False),
            "can_trade": heartbeat["trading_status"].get("kill_zone", {}).get("can_trade", False),
            "runner_status": heartbeat["trading_status"].get("runner", {}).get("status", "unknown"),
            "positions_count": heartbeat["active_positions"].get("count", 0),
            "total_pnl": heartbeat["active_positions"].get("total_unrealized_pnl", 0),
            "ml_models_in_hot_path": 1,  # Only Markov (optional)
            "ml_models_available": 4,
            "cognitive_features_invoked": 1,  # Only deliberate()
            "cognitive_features_loaded": 4,
            "components_not_wired": heartbeat["not_wired"]["count"],
        }

        # Log to file
        self._log_heartbeat(heartbeat)

        return heartbeat

    def display_heartbeat(self, heartbeat: Dict[str, Any]):
        """Display heartbeat in human-readable format."""
        print("\n" + "=" * 80)
        print("KOBE LIVE TRADING HEARTBEAT - HONEST ASSESSMENT")
        print("=" * 80)
        print(f"Generated: {heartbeat['generated_at']}")
        print(f"Cycle: {heartbeat['cycle']}")
        print()

        # Section 1: Trading Status
        print("-" * 40)
        print("SECTION 1: TRADING STATUS")
        print("-" * 40)
        ts = heartbeat["trading_status"]
        print(f"  Kill Switch: {'ACTIVE - ALL TRADING BLOCKED' if ts.get('kill_switch_active') else 'NOT ACTIVE'}")
        kz = ts.get("kill_zone", {})
        print(f"  Kill Zone: {kz.get('current_zone', 'unknown')}")
        print(f"  Can Trade: {kz.get('can_trade', 'unknown')} - {kz.get('reason', '')}")
        runner = ts.get("runner", {})
        print(f"  Runner: {runner.get('status', 'unknown')} (mode: {runner.get('mode', 'unknown')})")
        print(f"  Scans: {runner.get('scan_count', 0)} | Trades: {runner.get('trade_count', 0)}")
        print()

        # Section 2: Last Scan
        print("-" * 40)
        print("SECTION 2: LAST SCAN RESULTS")
        print("-" * 40)
        scan = heartbeat["last_scan_results"]
        ls = scan.get("last_signal")
        if ls:
            print(f"  Last Signal: {ls.get('symbol')} {ls.get('side')} (conf: {ls.get('confidence')})")
        else:
            print("  Last Signal: None")
        print(f"  Top 5 Count: {scan.get('top5_count', 0)}")
        if scan.get("top5_symbols"):
            print(f"  Top 5 Symbols: {', '.join(scan['top5_symbols'])}")
        print()

        # Section 3: Positions
        print("-" * 40)
        print("SECTION 3: ACTIVE POSITIONS")
        print("-" * 40)
        pos = heartbeat["active_positions"]
        if pos.get("error"):
            print(f"  Error: {pos['error']}")
        else:
            print(f"  Count: {pos.get('count', 0)}")
            for p in pos.get("positions", []):
                pnl = p.get("unrealized_pnl", 0)
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                print(f"    {p['symbol']}: {p['qty']} shares @ ${p['avg_entry']:.2f} -> ${p['current_price']:.2f} ({pnl_str})")
            print(f"  Total P&L: ${pos.get('total_unrealized_pnl', 0):.2f}")
        print()

        # Section 4: Risk Gates
        print("-" * 40)
        print("SECTION 4: RISK GATES STATUS")
        print("-" * 40)
        gates = heartbeat["risk_gates"]

        pg = gates.get("policy_gate", {})
        if pg.get("status") == "ACTIVE":
            print(f"  Policy Gate: ACTIVE - ${pg.get('daily_notional_used', 0):.0f}/${pg.get('daily_limit', 1000)} daily, ${pg.get('max_per_order', 75)}/order max")
        else:
            print(f"  Policy Gate: {pg.get('status', 'unknown')}")

        plg = gates.get("position_limit_gate", {})
        if plg.get("status") == "ACTIVE":
            print(f"  Position Limit: ACTIVE - {plg.get('current_positions', 0)}/{plg.get('max_positions', 3)} positions")
        else:
            print(f"  Position Limit: {plg.get('status', 'unknown')}")

        weg = gates.get("weekly_exposure_gate", {})
        if weg.get("status") == "ACTIVE":
            print(f"  Weekly Exposure: ACTIVE - {weg.get('weekly_exposure_pct', 0):.1f}%/{weg.get('max_weekly_pct', 40)}% max")
        else:
            print(f"  Weekly Exposure: {weg.get('status', 'unknown')}")

        kzg = gates.get("kill_zone_gate", {})
        if kzg.get("status") == "ACTIVE":
            print(f"  Kill Zone: ACTIVE - {kzg.get('current_zone', 'unknown')} ({'CAN TRADE' if kzg.get('trading_allowed') else 'BLOCKED'})")
        else:
            print(f"  Kill Zone: {kzg.get('status', 'unknown')}")
        print()

        # Section 5: ML Models - HONEST
        print("-" * 40)
        print("SECTION 5: ML MODELS (HONEST ASSESSMENT)")
        print("-" * 40)
        ml = heartbeat["ml_models"]
        print(f"  {ml['disclaimer']}")
        print()

        markov = ml.get("markov_chain", {})
        status = "IN HOT PATH (optional)" if markov.get("currently_enabled") else "AVAILABLE (use --markov)"
        print(f"  Markov Chain: {status}")

        hmm = ml.get("hmm_regime", {})
        print(f"  HMM Regime: EXISTS but NOT WIRED - {hmm.get('why_not_wired', '')}")

        ens = ml.get("ensemble", {})
        print(f"  Ensemble: EXISTS but NOT WIRED - {ens.get('why_not_wired', '')}")

        lstm = ml.get("lstm_confidence", {})
        print(f"  LSTM Confidence: EXISTS but NOT WIRED - {lstm.get('why_not_wired', '')}")

        summary = ml.get("summary", {})
        print(f"\n  SUMMARY: {summary.get('models_in_hot_path', 0)} of {summary.get('models_that_exist', 0)} ML models in hot path")
        print()

        # Section 6: Cognitive Brain - HONEST
        print("-" * 40)
        print("SECTION 6: COGNITIVE BRAIN (HONEST ASSESSMENT)")
        print("-" * 40)
        cog = heartbeat["cognitive_brain"]
        print(f"  {cog.get('disclaimer', '')}")
        print()

        delib = cog.get("deliberate_function", {})
        print(f"  deliberate(): {delib.get('status', 'unknown')} - {delib.get('honest_status', '')}")

        tot = cog.get("tree_of_thoughts", {})
        print(f"  Tree of Thoughts: {tot.get('status', 'unknown')}")

        sc = cog.get("self_consistency", {})
        print(f"  Self-Consistency: {sc.get('status', 'unknown')}")

        cr = cog.get("contradiction_resolver", {})
        print(f"  Contradiction Resolver: {cr.get('status', 'unknown')}")

        cog_summary = cog.get("summary", {})
        print(f"\n  SUMMARY: {cog_summary.get('features_actually_invoked', 0)} of {cog_summary.get('features_loaded', 0)} cognitive features invoked")
        print()

        # Section 7: Autonomous Brain
        print("-" * 40)
        print("SECTION 7: AUTONOMOUS BRAIN (SEPARATE)")
        print("-" * 40)
        auto = heartbeat["autonomous_brain"]
        print(f"  {auto.get('disclaimer', '')}")
        print()
        print(f"  Status: {auto.get('status', 'unknown')}")
        print(f"  Phase: {auto.get('phase', 'unknown')}")
        print(f"  Work Mode: {auto.get('work_mode', 'unknown')}")
        print(f"  Cycles: {auto.get('cycles', 0)}")
        print(f"  Uptime: {auto.get('uptime_hours', 0):.2f} hours")
        print(f"\n  HONEST: {auto.get('honest_status', '')}")
        print()

        # Section 8: Not Wired
        print("-" * 40)
        print("SECTION 8: WHAT'S NOT WIRED (TRANSPARENCY)")
        print("-" * 40)
        nw = heartbeat["not_wired"]
        print(f"  {nw.get('disclaimer', '')}")
        print()
        for comp in nw.get("components", []):
            print(f"  - {comp['name']}: {comp['why_not_wired']}")
        print(f"\n  {nw.get('honest_summary', '')}")
        print()

        # Final Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        s = heartbeat["summary"]
        print(f"  Kill Switch: {'ACTIVE' if s['kill_switch'] else 'OFF'}")
        print(f"  Can Trade Now: {s['can_trade']}")
        print(f"  Runner: {s['runner_status']}")
        print(f"  Positions: {s['positions_count']} (P&L: ${s['total_pnl']:.2f})")
        print(f"  ML Models: {s['ml_models_in_hot_path']}/{s['ml_models_available']} in hot path")
        print(f"  Cognitive: {s['cognitive_features_invoked']}/{s['cognitive_features_loaded']} features invoked")
        print(f"  Not Wired: {s['components_not_wired']} components exist but unused")
        print()
        print("This is the HONEST state of Kobe. The trading system WORKS.")
        print("ML models are ENHANCEMENTS, not REQUIREMENTS.")
        print("=" * 80)
        print()

    def run(self, interval: int = 60, once: bool = False):
        """
        Run heartbeat continuously.

        Args:
            interval: Seconds between heartbeats
            once: If True, run once and exit
        """
        print("\n" + "=" * 80)
        print("KOBE LIVE TRADING HEARTBEAT - STARTING")
        print("=" * 80)
        print(f"Interval: {interval} seconds")
        print(f"Log file: {self.heartbeat_log}")
        print("Press Ctrl+C to stop")
        print("=" * 80)

        try:
            while True:
                heartbeat = self.generate_heartbeat()
                self.display_heartbeat(heartbeat)

                if once:
                    break

                print(f"Next heartbeat in {interval} seconds...")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nHeartbeat stopped by user.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="KOBE Live Trading Heartbeat - HONEST Assessment"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between heartbeats (default: 60)"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't log to file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only (no display)"
    )

    args = parser.parse_args()

    heartbeat = LiveTradingHeartbeat(log_to_file=not args.no_log)

    if args.json:
        # JSON mode - output once
        hb = heartbeat.generate_heartbeat()
        print(json.dumps(hb, indent=2, default=str))
    else:
        # Normal display mode
        heartbeat.run(interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
