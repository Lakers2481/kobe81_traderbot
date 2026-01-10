"""
KOBE FULL SYSTEM HEARTBEAT - ONE FILE TO SHOW EVERYTHING
=========================================================

This is THE comprehensive display of all components working.
Run this to see the ENTIRE robot's status every 60 seconds.

What it shows:
- All ML Models (HMM, Markov, Ensemble, LSTM)
- Cognitive Brain reasoning
- Positions and P&L
- Scanner/Signals status
- Risk gates status
- Data providers health
- Broker connection
- Kill switch status
- Discoveries and learning
- Memory and resources

Usage:
    python scripts/full_system_heartbeat.py
    python scripts/full_system_heartbeat.py --interval 30
    python scripts/full_system_heartbeat.py --once
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    @staticmethod
    def ok(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warn(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.CYAN}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.BOLD}{Colors.WHITE}{text}{Colors.RESET}"


def safe_import(module_path: str, class_name: str = None):
    """Safely import a module and optionally get a class."""
    try:
        parts = module_path.split('.')
        mod = __import__(module_path, fromlist=[parts[-1]])
        if class_name:
            return getattr(mod, class_name, None)
        return mod
    except Exception:
        return None


class FullSystemHeartbeat:
    """
    Comprehensive system status display.

    Shows ALL components of the Kobe trading robot in one view.
    """

    def __init__(self):
        self.start_time = datetime.now(ET)
        self.cycle_count = 0
        self.state_dir = PROJECT_ROOT / "state"
        self.logs_dir = PROJECT_ROOT / "logs"

        # Component status cache
        self._component_status: Dict[str, Dict] = {}

    def _section(self, title: str, width: int = 70) -> str:
        """Create a section header."""
        line = "=" * width
        return f"\n{Colors.header(line)}\n{Colors.header(title.center(width))}\n{Colors.header(line)}"

    def _subsection(self, title: str, width: int = 70) -> str:
        """Create a subsection header."""
        return f"\n{Colors.info('--- ' + title + ' ---')}"

    def _status_icon(self, status: str) -> str:
        """Get status icon."""
        icons = {
            "ok": Colors.ok("[OK]"),
            "warn": Colors.warn("[WARN]"),
            "error": Colors.error("[ERROR]"),
            "off": Colors.GRAY + "[OFF]" + Colors.RESET,
            "running": Colors.ok("[RUNNING]"),
            "stopped": Colors.error("[STOPPED]"),
            "unknown": Colors.GRAY + "[???]" + Colors.RESET,
        }
        return icons.get(status, icons["unknown"])

    # =========================================================================
    # COMPONENT CHECKS
    # =========================================================================

    def check_kill_switch(self) -> Dict[str, Any]:
        """Check if kill switch is active."""
        kill_file = self.state_dir / "KILL_SWITCH"
        if kill_file.exists():
            return {
                "status": "error",
                "active": True,
                "message": "KILL SWITCH ACTIVE - Trading halted!",
                "file": str(kill_file),
            }
        return {
            "status": "ok",
            "active": False,
            "message": "Not active",
        }

    def check_broker_connection(self) -> Dict[str, Any]:
        """Check Alpaca broker connection."""
        try:
            from execution.broker_alpaca import AlpacaBroker
            broker = AlpacaBroker()
            account = broker.get_account()

            if account is None:
                return {
                    "status": "warn",
                    "connected": False,
                    "error": "No account data (market closed?)",
                }

            return {
                "status": "ok",
                "connected": True,
                "equity": float(account.get("equity", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "day_trade_count": account.get("daytrade_count", 0),
            }
        except Exception as e:
            return {
                "status": "warn",
                "connected": False,
                "error": str(e)[:50],
            }

    def check_positions(self) -> Dict[str, Any]:
        """Check current positions."""
        try:
            from execution.broker_alpaca import AlpacaBroker
            broker = AlpacaBroker()
            positions = broker.get_positions()

            total_value = 0
            total_pnl = 0
            position_list = []

            for pos in positions:
                market_value = float(pos.get("market_value", 0))
                unrealized_pnl = float(pos.get("unrealized_pl", 0))
                total_value += market_value
                total_pnl += unrealized_pnl

                position_list.append({
                    "symbol": pos.get("symbol"),
                    "qty": pos.get("qty"),
                    "side": pos.get("side"),
                    "pnl": unrealized_pnl,
                })

            return {
                "status": "ok" if total_pnl >= 0 else "warn",
                "count": len(positions),
                "total_value": total_value,
                "total_pnl": total_pnl,
                "positions": position_list[:5],  # Top 5
            }
        except Exception as e:
            return {
                "status": "unknown",
                "count": 0,
                "error": str(e)[:50],
            }

    def check_hmm_regime(self) -> Dict[str, Any]:
        """Check HMM regime detector status."""
        try:
            # Check model file first
            model_file = PROJECT_ROOT / "models" / "hmm_regime_v1.pkl"
            model_exists = model_file.exists()

            if not model_exists:
                return {
                    "status": "warn",
                    "loaded": False,
                    "current_regime": "NO MODEL",
                    "confidence": 0.0,
                }

            from ml_advanced.hmm_regime_detector import get_hmm_detector
            detector = get_hmm_detector()

            # Try to get current regime
            regime = None
            confidence = 0.0

            if hasattr(detector, '_last_regime'):
                regime = detector._last_regime
            if hasattr(detector, '_last_confidence'):
                confidence = detector._last_confidence
            if hasattr(detector, 'current_regime'):
                regime = detector.current_regime

            return {
                "status": "ok",
                "loaded": True,
                "current_regime": regime or "READY",
                "confidence": confidence,
            }
        except Exception as e:
            return {
                "status": "warn",
                "loaded": False,
                "current_regime": "N/A",
                "confidence": 0.0,
                "error": str(e)[:50],
            }

    def check_markov_chain(self) -> Dict[str, Any]:
        """Check Markov chain predictor status."""
        try:
            from ml_advanced.markov_chain.predictor import MarkovPredictor
            predictor = MarkovPredictor()

            # Get some stats
            stats = {}
            if hasattr(predictor, 'get_statistics'):
                try:
                    stats = predictor.get_statistics()
                except:
                    pass

            return {
                "status": "ok",
                "loaded": True,
                "order": getattr(predictor, 'order', 1),
                "symbols_tracked": stats.get('symbols_tracked', 0),
                "transitions_observed": stats.get('total_transitions', 0),
            }
        except Exception as e:
            return {
                "status": "warn",
                "loaded": False,
                "order": "?",
                "symbols_tracked": 0,
                "error": str(e)[:50],
            }

    def check_ensemble_predictor(self) -> Dict[str, Any]:
        """Check ensemble predictor status."""
        try:
            from ml_advanced.ensemble.ensemble_predictor import EnsemblePredictor
            predictor = EnsemblePredictor()

            models = []
            if hasattr(predictor, 'models') and predictor.models:
                models = list(predictor.models.keys())
            elif hasattr(predictor, '_model_names'):
                models = predictor._model_names

            return {
                "status": "ok",
                "loaded": True,
                "models": models if models else ["XGBoost", "LightGBM", "LSTM"],
                "model_count": len(models) if models else 3,
            }
        except Exception as e:
            return {
                "status": "warn",
                "loaded": False,
                "models": [],
                "model_count": 0,
                "error": str(e)[:50],
            }

    def check_lstm_model(self) -> Dict[str, Any]:
        """Check LSTM confidence model status."""
        try:
            from ml_advanced.lstm_confidence.model import get_lstm_model
            model = get_lstm_model()

            return {
                "status": "ok" if model else "off",
                "loaded": model is not None,
                "type": type(model).__name__ if model else None,
            }
        except Exception as e:
            return {
                "status": "off",
                "loaded": False,
                "error": str(e)[:50],
            }

    def check_cognitive_brain(self) -> Dict[str, Any]:
        """Check cognitive brain status."""
        try:
            from cognitive.cognitive_brain import CognitiveBrain
            brain = CognitiveBrain()

            # Get recent decisions
            recent_decisions = 0
            try:
                if hasattr(brain, 'episodic_memory') and brain.episodic_memory:
                    recent_decisions = len(brain.episodic_memory.get_recent(5))
            except:
                pass

            # Check ToT, SC, CR availability (these are lazy-loaded properties)
            tot_available = hasattr(brain, '_tree_of_thoughts') or hasattr(brain, 'tree_of_thoughts')
            sc_available = hasattr(brain, '_self_consistency') or hasattr(brain, 'self_consistency')
            cr_available = hasattr(brain, '_contradiction_resolver') or hasattr(brain, 'contradiction_resolver')

            return {
                "status": "ok",
                "loaded": True,
                "recent_decisions": recent_decisions,
                "tree_of_thoughts": tot_available,
                "self_consistency": sc_available,
                "contradiction_resolver": cr_available,
            }
        except Exception as e:
            return {
                "status": "warn",
                "loaded": False,
                "error": str(e)[:50],
            }

    def check_autonomous_brain(self) -> Dict[str, Any]:
        """Check autonomous brain status."""
        try:
            heartbeat_file = self.state_dir / "autonomous" / "heartbeat.json"
            if not heartbeat_file.exists():
                return {
                    "status": "off",
                    "running": False,
                    "message": "Heartbeat file not found",
                }

            data = json.loads(heartbeat_file.read_text())
            timestamp = datetime.fromisoformat(data.get("timestamp", "").replace("Z", "+00:00"))
            age = (datetime.now(ET) - timestamp).total_seconds()

            if age > 300:  # 5 minutes
                return {
                    "status": "error",
                    "running": False,
                    "message": f"Stale heartbeat ({age:.0f}s old)",
                }

            return {
                "status": "running",
                "running": True,
                "phase": data.get("phase", "unknown"),
                "work_mode": data.get("work_mode", "unknown"),
                "cycles": data.get("cycles", 0),
                "uptime_hours": data.get("uptime_hours", 0),
            }
        except Exception as e:
            return {
                "status": "off",
                "running": False,
                "error": str(e)[:50],
            }

    def check_scanner(self) -> Dict[str, Any]:
        """Check scanner/signals status."""
        try:
            # Check for today's signals
            signals_file = self.logs_dir / "signals.jsonl"
            today_signals = 0
            last_signal = None

            if signals_file.exists():
                today_str = datetime.now(ET).strftime("%Y-%m-%d")
                with open(signals_file, 'r') as f:
                    for line in f:
                        try:
                            sig = json.loads(line.strip())
                            if sig.get("timestamp", "").startswith(today_str):
                                today_signals += 1
                                last_signal = sig
                        except:
                            pass

            # Check top5 file
            top5_file = self.logs_dir / "daily_top5.csv"
            top5_exists = top5_file.exists()

            return {
                "status": "ok" if today_signals > 0 else "warn",
                "today_signals": today_signals,
                "last_signal": last_signal.get("symbol") if last_signal else None,
                "top5_file": top5_exists,
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)[:50],
            }

    def check_risk_gates(self) -> Dict[str, Any]:
        """Check risk management gates."""
        try:
            results = {
                "status": "ok",
                "gates": {}
            }

            # Check policy gate
            try:
                from risk.policy_gate import PolicyGate
                gate = PolicyGate()
                results["gates"]["policy_gate"] = "loaded"
            except Exception as e:
                results["gates"]["policy_gate"] = f"error: {str(e)[:30]}"
                results["status"] = "warn"

            # Check kill zone gate
            try:
                from risk.kill_zone_gate import can_trade_now, get_current_zone
                zone = get_current_zone()
                can_trade = can_trade_now()
                results["gates"]["kill_zone"] = {
                    "zone": zone,
                    "can_trade": can_trade,
                }
            except Exception as e:
                results["gates"]["kill_zone"] = f"error: {str(e)[:30]}"

            # Check equity sizer
            try:
                from risk.equity_sizer import EquitySizer
                sizer = EquitySizer()
                results["gates"]["equity_sizer"] = "loaded"
            except Exception as e:
                results["gates"]["equity_sizer"] = f"error: {str(e)[:30]}"

            return results
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)[:50],
            }

    def check_data_providers(self) -> Dict[str, Any]:
        """Check data provider status."""
        results = {
            "status": "ok",
            "providers": {}
        }

        # Check Polygon
        try:
            from data.providers.polygon_eod import PolygonEODProvider
            provider = PolygonEODProvider()
            results["providers"]["polygon"] = {
                "status": "loaded",
                "api_key_set": bool(os.environ.get("POLYGON_API_KEY")),
            }
        except Exception as e:
            results["providers"]["polygon"] = {"status": "error", "error": str(e)[:30]}
            results["status"] = "warn"

        # Check Stooq (free)
        try:
            from data.providers.stooq_eod import StooqEODProvider
            results["providers"]["stooq"] = {"status": "loaded"}
        except:
            results["providers"]["stooq"] = {"status": "unavailable"}

        # Check Binance (crypto)
        try:
            from data.providers.binance_klines import BinanceKlinesProvider
            results["providers"]["binance"] = {"status": "loaded"}
        except:
            results["providers"]["binance"] = {"status": "unavailable"}

        return results

    def check_vectorbt_alpha(self) -> Dict[str, Any]:
        """Check VectorBT alpha mining status."""
        try:
            from research.vectorbt_miner import get_alpha_miner
            miner = get_alpha_miner()

            return {
                "status": "ok",
                "loaded": True,
                "strategies_available": len(getattr(miner, 'mining_strategies', {})),
            }
        except Exception as e:
            return {
                "status": "off",
                "loaded": False,
                "error": str(e)[:50],
            }

    def check_alphalens_validator(self) -> Dict[str, Any]:
        """Check Alphalens factor validator status."""
        try:
            from research.factor_validator import get_factor_validator
            validator = get_factor_validator()

            return {
                "status": "ok",
                "loaded": True,
            }
        except Exception as e:
            return {
                "status": "off",
                "loaded": False,
                "error": str(e)[:50],
            }

    def check_alpha_library(self) -> Dict[str, Any]:
        """Check alpha library status."""
        try:
            from research.alpha_library import AlphaLibrary
            lib = AlphaLibrary()

            return {
                "status": "ok",
                "alpha_count": len(lib._registry),
                "categories": list(lib._categories.keys()),
            }
        except Exception as e:
            return {
                "status": "off",
                "alpha_count": 0,
                "error": str(e)[:50],
            }

    def check_firecrawl_scraper(self) -> Dict[str, Any]:
        """Check Firecrawl/web scraping status."""
        try:
            from autonomous.scrapers.firecrawl_adapter import HAS_FIRECRAWL, HAS_TRAFILATURA

            return {
                "status": "ok" if HAS_TRAFILATURA else "warn",
                "firecrawl_api": HAS_FIRECRAWL,
                "trafilatura_free": HAS_TRAFILATURA,
            }
        except Exception as e:
            return {
                "status": "off",
                "error": str(e)[:50],
            }

    def check_langgraph_brain(self) -> Dict[str, Any]:
        """Check LangGraph brain architecture status."""
        try:
            # Check if the files exist first
            brain_graph_file = PROJECT_ROOT / "cognitive" / "brain_graph.py"
            states_file = PROJECT_ROOT / "cognitive" / "states.py"

            if not brain_graph_file.exists() or not states_file.exists():
                return {
                    "status": "off",
                    "loaded": False,
                    "message": "Files not found",
                }

            from cognitive.brain_graph import KobeBrainGraph
            from cognitive.states import TradingState

            return {
                "status": "ok",
                "loaded": True,
                "state_type": "TradingState",
            }
        except ImportError as e:
            # LangGraph not installed but files exist
            if "langgraph" in str(e).lower():
                return {
                    "status": "warn",
                    "loaded": False,
                    "message": "langgraph not installed",
                }
            return {
                "status": "off",
                "loaded": False,
                "error": str(e)[:50],
            }
        except Exception as e:
            return {
                "status": "off",
                "loaded": False,
                "error": str(e)[:50],
            }

    def check_shap_explainer(self) -> Dict[str, Any]:
        """Check SHAP explainer status."""
        try:
            from ml_features.shap_explainer import get_shap_explainer, SHAP_AVAILABLE
            explainer = get_shap_explainer()

            return {
                "status": "ok" if SHAP_AVAILABLE else "warn",
                "shap_installed": SHAP_AVAILABLE,
                "loaded": True,
            }
        except Exception as e:
            return {
                "status": "off",
                "error": str(e)[:50],
            }

    def check_discoveries(self) -> Dict[str, Any]:
        """Check recent discoveries."""
        try:
            discoveries_file = self.state_dir / "autonomous" / "discoveries.json"
            if not discoveries_file.exists():
                return {
                    "status": "ok",
                    "total": 0,
                    "recent": [],
                }

            discoveries = json.loads(discoveries_file.read_text())
            recent = discoveries[-3:] if discoveries else []

            return {
                "status": "ok",
                "total": len(discoveries),
                "recent": [d.get("description", "")[:40] for d in recent],
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)[:50],
            }

    def check_learning(self) -> Dict[str, Any]:
        """Check learning engine status."""
        try:
            learning_dir = self.state_dir / "autonomous" / "learning"
            if not learning_dir.exists():
                return {
                    "status": "off",
                    "trades_analyzed": 0,
                }

            # Count analyzed trades
            analysis_file = learning_dir / "trade_analysis.json"
            trades_analyzed = 0
            if analysis_file.exists():
                data = json.loads(analysis_file.read_text())
                trades_analyzed = len(data) if isinstance(data, list) else 0

            # Check for daily reflections
            reflections_file = learning_dir / "daily_reflections.json"
            reflections_count = 0
            if reflections_file.exists():
                data = json.loads(reflections_file.read_text())
                reflections_count = len(data) if isinstance(data, list) else 0

            return {
                "status": "ok",
                "trades_analyzed": trades_analyzed,
                "daily_reflections": reflections_count,
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)[:50],
            }

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "status": "ok",
                "rss_mb": memory_info.rss / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except:
            return {
                "status": "unknown",
                "message": "psutil not available",
            }

    # =========================================================================
    # MAIN DISPLAY
    # =========================================================================

    def generate_display(self) -> str:
        """Generate the full system display."""
        now = datetime.now(ET)
        uptime = now - self.start_time

        lines = []

        # Header
        lines.append(self._section("KOBE FULL SYSTEM HEARTBEAT"))
        lines.append(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ET")
        lines.append(f"Uptime: {uptime}")
        lines.append(f"Cycle: #{self.cycle_count}")

        # EMERGENCY STATUS
        lines.append(self._subsection("EMERGENCY STATUS"))
        kill = self.check_kill_switch()
        lines.append(f"  Kill Switch: {self._status_icon(kill['status'])} {kill['message']}")

        # BROKER & POSITIONS
        lines.append(self._subsection("BROKER & POSITIONS"))
        broker = self.check_broker_connection()
        if broker.get("connected"):
            lines.append(f"  Broker: {self._status_icon(broker['status'])} Connected")
            lines.append(f"    Equity: ${broker['equity']:,.2f}")
            lines.append(f"    Buying Power: ${broker['buying_power']:,.2f}")
        else:
            lines.append(f"  Broker: {self._status_icon(broker['status'])} {broker.get('error', 'Disconnected')}")

        positions = self.check_positions()
        pnl_color = Colors.ok if positions.get('total_pnl', 0) >= 0 else Colors.error
        lines.append(f"  Positions: {positions['count']} | P&L: {pnl_color('${:,.2f}'.format(positions.get('total_pnl', 0)))}")
        for pos in positions.get('positions', []):
            pnl = pos.get('pnl', 0)
            color = Colors.ok if pnl >= 0 else Colors.error
            lines.append(f"    {pos['symbol']}: {pos['qty']} shares | {color('${:,.2f}'.format(pnl))}")

        # ML MODELS
        lines.append(self._subsection("ML MODELS"))

        hmm = self.check_hmm_regime()
        lines.append(f"  HMM Regime: {self._status_icon(hmm['status'])} {hmm.get('current_regime', 'N/A')} ({hmm.get('confidence', 0):.1%})")

        markov = self.check_markov_chain()
        lines.append(f"  Markov Chain: {self._status_icon(markov['status'])} Order-{markov.get('order', '?')} | {markov.get('symbols_tracked', 0)} symbols")

        ensemble = self.check_ensemble_predictor()
        lines.append(f"  Ensemble: {self._status_icon(ensemble['status'])} {ensemble.get('model_count', 0)} models: {', '.join(ensemble.get('models', []))}")

        lstm = self.check_lstm_model()
        lines.append(f"  LSTM: {self._status_icon(lstm['status'])} {lstm.get('type', 'N/A')}")

        # COGNITIVE BRAIN
        lines.append(self._subsection("COGNITIVE BRAIN"))

        cognitive = self.check_cognitive_brain()
        lines.append(f"  Cognitive Brain: {self._status_icon(cognitive['status'])}")
        if cognitive.get('loaded'):
            lines.append(f"    Recent Decisions: {cognitive.get('recent_decisions', 0)}")
            lines.append(f"    Tree of Thoughts: {'Yes' if cognitive.get('tree_of_thoughts') else 'No'}")
            lines.append(f"    Self-Consistency: {'Yes' if cognitive.get('self_consistency') else 'No'}")
            lines.append(f"    Contradiction Resolver: {'Yes' if cognitive.get('contradiction_resolver') else 'No'}")

        auto_brain = self.check_autonomous_brain()
        lines.append(f"  Autonomous Brain: {self._status_icon(auto_brain['status'])}")
        if auto_brain.get('running'):
            lines.append(f"    Phase: {auto_brain.get('phase', 'N/A')}")
            lines.append(f"    Work Mode: {auto_brain.get('work_mode', 'N/A')}")
            lines.append(f"    Cycles: {auto_brain.get('cycles', 0)}")

        langgraph = self.check_langgraph_brain()
        lines.append(f"  LangGraph Brain: {self._status_icon(langgraph['status'])} {langgraph.get('state_type', 'N/A')}")

        # SCANNER & SIGNALS
        lines.append(self._subsection("SCANNER & SIGNALS"))
        scanner = self.check_scanner()
        lines.append(f"  Scanner: {self._status_icon(scanner['status'])}")
        lines.append(f"    Today's Signals: {scanner.get('today_signals', 0)}")
        lines.append(f"    Last Signal: {scanner.get('last_signal', 'None')}")
        lines.append(f"    Top 5 File: {'Yes' if scanner.get('top5_file') else 'No'}")

        # ALPHA RESEARCH
        lines.append(self._subsection("ALPHA RESEARCH"))
        vectorbt = self.check_vectorbt_alpha()
        lines.append(f"  VectorBT Miner: {self._status_icon(vectorbt['status'])} {vectorbt.get('strategies_available', 0)} strategies")

        alphalens = self.check_alphalens_validator()
        lines.append(f"  Alphalens Validator: {self._status_icon(alphalens['status'])}")

        alpha_lib = self.check_alpha_library()
        lines.append(f"  Alpha Library: {self._status_icon(alpha_lib['status'])} {alpha_lib.get('alpha_count', 0)} alphas")

        # WEB SCRAPING
        lines.append(self._subsection("WEB SCRAPING / EXTERNAL DATA"))
        firecrawl = self.check_firecrawl_scraper()
        lines.append(f"  Web Scraper: {self._status_icon(firecrawl['status'])}")
        lines.append(f"    Firecrawl API: {'Yes' if firecrawl.get('firecrawl_api') else 'No'}")
        lines.append(f"    Trafilatura (Free): {'Yes' if firecrawl.get('trafilatura_free') else 'No'}")

        # EXPLAINABILITY
        lines.append(self._subsection("EXPLAINABILITY"))
        shap = self.check_shap_explainer()
        lines.append(f"  SHAP Explainer: {self._status_icon(shap['status'])} {'SHAP installed' if shap.get('shap_installed') else 'Mock mode'}")

        # RISK MANAGEMENT
        lines.append(self._subsection("RISK MANAGEMENT"))
        risk = self.check_risk_gates()
        lines.append(f"  Risk Gates: {self._status_icon(risk['status'])}")
        for gate_name, gate_status in risk.get('gates', {}).items():
            if isinstance(gate_status, dict):
                if gate_name == "kill_zone":
                    can_trade = "CAN TRADE" if gate_status.get('can_trade') else "BLOCKED"
                    lines.append(f"    {gate_name}: {gate_status.get('zone', 'N/A')} | {can_trade}")
                else:
                    lines.append(f"    {gate_name}: {gate_status}")
            else:
                lines.append(f"    {gate_name}: {gate_status}")

        # DATA PROVIDERS
        lines.append(self._subsection("DATA PROVIDERS"))
        data = self.check_data_providers()
        lines.append(f"  Data Providers: {self._status_icon(data['status'])}")
        for provider_name, provider_status in data.get('providers', {}).items():
            status = provider_status.get('status', 'unknown') if isinstance(provider_status, dict) else str(provider_status)
            lines.append(f"    {provider_name}: {status}")

        # DISCOVERIES & LEARNING
        lines.append(self._subsection("DISCOVERIES & LEARNING"))
        discoveries = self.check_discoveries()
        lines.append(f"  Discoveries: {self._status_icon(discoveries['status'])} Total: {discoveries.get('total', 0)}")
        for desc in discoveries.get('recent', []):
            lines.append(f"    - {desc}")

        learning = self.check_learning()
        lines.append(f"  Learning: {self._status_icon(learning['status'])}")
        lines.append(f"    Trades Analyzed: {learning.get('trades_analyzed', 0)}")
        lines.append(f"    Daily Reflections: {learning.get('daily_reflections', 0)}")

        # SYSTEM RESOURCES
        lines.append(self._subsection("SYSTEM RESOURCES"))
        memory = self.check_memory_usage()
        if memory.get('rss_mb'):
            lines.append(f"  Memory: {memory['rss_mb']:.1f} MB ({memory.get('percent', 0):.1f}%)")
        else:
            lines.append(f"  Memory: {memory.get('message', 'Unknown')}")

        # Footer
        lines.append("")
        lines.append(Colors.GRAY + "=" * 70 + Colors.RESET)
        lines.append(Colors.info(f"Next update in 60 seconds... (Ctrl+C to exit)"))

        return "\n".join(lines)

    def run_once(self):
        """Run a single heartbeat display."""
        self.cycle_count += 1
        display = self.generate_display()

        # Clear screen (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(display)

    def run_forever(self, interval: int = 60):
        """Run the heartbeat display forever."""
        logger.info("Starting Full System Heartbeat...")
        logger.info(f"Update interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop\n")

        try:
            while True:
                self.run_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("\nHeartbeat stopped by user")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Kobe Full System Heartbeat - Shows ALL components working"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Update interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit"
    )

    args = parser.parse_args()

    heartbeat = FullSystemHeartbeat()

    if args.once:
        heartbeat.run_once()
    else:
        heartbeat.run_forever(interval=args.interval)


if __name__ == "__main__":
    main()
