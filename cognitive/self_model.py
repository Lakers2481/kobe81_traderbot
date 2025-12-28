"""
Self-Model - The AI's Sense of Self
=======================================

This module provides the cognitive architecture with a model of its own
capabilities, traits, and limitations. It is the foundation of the AI's
self-awareness, allowing it to answer questions like:
- "What am I good at?"
- "What are my weaknesses?"
- "How trustworthy is my own confidence score right now?"

Based on AI self-awareness research, this component enables the agent to take
an objective stance on itself, leading to more robust and safer decision-making.
It moves the AI from simply *making* decisions to *understanding its ability*
to make decisions.

Features:
- Tracks and categorizes its performance across different strategies and market regimes.
- Maintains a list of its own known limitations and weaknesses.
- Monitors its confidence calibration (i.e., when it says it's 80% confident,
  is it actually correct 80% of the time?).
- Generates a natural language "self-description" of its strengths and weaknesses.
- Persists this self-knowledge across sessions.

Usage:
    from cognitive.self_model import get_self_model

    self_model = get_self_model()

    # The ReflectionEngine updates the self-model after a trade.
    self_model.record_trade_outcome('ibs_rsi', 'BULL', won=True, pnl=150.0)

    # The MetacognitiveGovernor queries it before making a decision.
    should_i_act, reason = self_model.should_stand_down('ibs_rsi', 'CHOPPY')
    if should_i_act:
        print(f"Standing down because: {reason}")

    # The brain can introspect by reading its own self-description.
    print(self_model.get_self_description())
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class Capability(Enum):
    """A qualitative rating of the AI's performance in a specific context."""
    EXCELLENT = "excellent"      # > 65% win rate
    GOOD = "good"                # 55-65% win rate
    ADEQUATE = "adequate"        # 45-55% win rate
    WEAK = "weak"                # 35-45% win rate
    POOR = "poor"                # < 35% win rate
    UNKNOWN = "unknown"          # Not enough data to form an opinion.


@dataclass
class StrategyPerformance:
    """Tracks the AI's historical performance for a specific strategy/regime pair."""
    strategy: str
    regime: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    notes: List[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Calculates the win rate for this context."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def capability(self) -> Capability:
        """Assigns a qualitative capability rating based on the win rate."""
        if self.total_trades < 10:
            return Capability.UNKNOWN
        wr = self.win_rate
        if wr >= 0.65: return Capability.EXCELLENT
        elif wr >= 0.55: return Capability.GOOD
        elif wr >= 0.45: return Capability.ADEQUATE
        elif wr >= 0.35: return Capability.WEAK
        else: return Capability.POOR

    def to_dict(self) -> Dict:
        """Serializes the performance record to a dictionary."""
        d = asdict(self)
        # Add computed properties to the dictionary for easy access.
        d['win_rate'] = self.win_rate
        d['capability'] = self.capability.value
        d['last_updated'] = self.last_updated.isoformat()
        return d


@dataclass
class Limitation:
    """Represents a known weakness or failure mode of the AI."""
    context: str          # When this limitation typically occurs.
    description: str      # What the limitation is.
    severity: str         # 'minor', 'moderate', or 'severe'.
    occurrences: int = 1
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    mitigation: Optional[str] = None  # A suggested way to handle this limitation.

    def to_dict(self) -> Dict:
        """Serializes the limitation to a dictionary."""
        d = asdict(self)
        d['first_observed'] = self.first_observed.isoformat()
        d['last_observed'] = self.last_observed.isoformat()
        return d


@dataclass
class ConfidenceCalibration:
    """Tracks how well the AI's stated confidence matches its actual accuracy."""
    confidence_bucket: str  # e.g., "70-80%"
    predictions: int = 0
    correct: int = 0

    @property
    def actual_accuracy(self) -> float:
        """The actual success rate for this confidence bucket."""
        return self.correct / self.predictions if self.predictions > 0 else 0.0

    @property
    def calibration_error(self) -> float:
        """
        The difference between stated confidence and actual accuracy.
        A lower number is better.
        """
        # e.g., for bucket "70-80%", the expected accuracy is 75%.
        low, high = map(int, self.confidence_bucket.replace('%', '').split('-'))
        expected_accuracy = (low + high) / 200.0
        return abs(expected_accuracy - self.actual_accuracy)


class SelfModel:
    """
    Manages the AI's persistent model of its own capabilities, limits, and
    calibration. This is the core of the agent's self-awareness.
    """

    def __init__(self, state_dir: str = "state/cognitive", auto_persist: bool = True):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist

        # --- Core Self-Knowledge Stores ---
        self._performance: Dict[str, StrategyPerformance] = {}
        self._limitations: Dict[str, Limitation] = {}
        self._calibration: Dict[str, ConfidenceCalibration] = {}
        self._recent_errors: List[Dict] = []
        self._error_limit = 100

        self._self_description: str = "" # A cached natural language summary.

        self._load_state()
        logger.info(f"SelfModel initialized, loaded {len(self._performance)} performance records.")

    def _make_key(self, strategy: str, regime: str) -> str:
        """Creates a consistent dictionary key for a strategy-regime pair."""
        return f"{strategy.lower()}|{regime.lower()}"

    def record_trade_outcome(
        self,
        strategy: str,
        regime: str,
        won: bool,
        pnl: float,
        r_multiple: float = 0.0,
        notes: Optional[str] = None,
    ) -> None:
        """
        The primary method for updating the self-model. The ReflectionEngine
        calls this after every trade to provide feedback.
        """
        key = self._make_key(strategy, regime)
        if key not in self._performance:
            self._performance[key] = StrategyPerformance(strategy=strategy, regime=regime)

        perf = self._performance[key]
        perf.total_trades += 1
        if won: perf.winning_trades += 1
        else: perf.losing_trades += 1
        perf.total_pnl += pnl
        # Update rolling average for R-multiple
        perf.avg_r_multiple = ((perf.avg_r_multiple * (perf.total_trades - 1)) + r_multiple) / perf.total_trades
        perf.last_updated = datetime.now()
        if notes:
            perf.notes.append(notes)
            perf.notes = perf.notes[-10:] # Keep last 10 notes

        # After updating performance, regenerate the summary description.
        self._update_self_description()

        if self.auto_persist:
            self._save_state()

        logger.info(f"Self-model updated for {key}: Win Rate is now {perf.win_rate:.1%} over {perf.total_trades} trades.")

    def record_limitation(self, context: str, description: str, severity: str = "moderate") -> None:
        """Records a new weakness or updates an existing one."""
        key = f"{context.lower()}|{description.lower()}"
        if key in self._limitations:
            self._limitations[key].occurrences += 1
            self._limitations[key].last_observed = datetime.now()
        else:
            self._limitations[key] = Limitation(context=context, description=description, severity=severity)
        
        logger.info(f"Recorded new limitation: '{description}' in context '{context}'.")
        self._update_self_description()
        if self.auto_persist: self._save_state()

    def record_prediction(self, confidence: float, correct: bool) -> None:
        """Records a prediction to track confidence calibration."""
        bucket = f"{int(confidence*10) * 10}-{int(confidence*10)*10+10}%"
        
        if bucket not in self._calibration:
            self._calibration[bucket] = ConfidenceCalibration(confidence_bucket=bucket)
            
        cal = self._calibration[bucket]
        cal.predictions += 1
        if correct: cal.correct += 1
        
        if self.auto_persist: self._save_state()

    def get_performance(self, strategy: str, regime: str) -> Optional[StrategyPerformance]:
        """Gets the detailed performance record for a given context."""
        return self._performance.get(self._make_key(strategy, regime))

    def get_capability(self, strategy: str, regime: str) -> Capability:
        """Gets the capability rating for a given strategy/regime context."""
        perf = self.get_performance(strategy, regime)
        if perf is None:
            return Capability.UNKNOWN
        return perf.capability

    def get_strengths(self) -> List[str]:
        """Returns a list of contexts where the AI has EXCELLENT or GOOD capability."""
        return [
            f"{p.strategy} in {p.regime} ({p.win_rate:.1%})"
            for p in self._performance.values()
            if p.capability in [Capability.EXCELLENT, Capability.GOOD]
        ]

    def get_weaknesses(self) -> List[str]:
        """Returns a list of contexts where the AI has WEAK or POOR capability."""
        weaknesses = [
            f"{p.strategy} in {p.regime} ({p.win_rate:.1%})"
            for p in self._performance.values()
            if p.capability in [Capability.WEAK, Capability.POOR]
        ]
        weaknesses.extend([
            f"{lim.context}: {lim.description}"
            for lim in self._limitations.values()
            if lim.severity in ['moderate', 'severe']
        ])
        return weaknesses

    def known_limitations(self) -> List[Limitation]:
        """Returns the list of known limitations as Limitation objects."""
        return list(self._limitations.values())

    def is_well_calibrated(self) -> bool:
        """Checks if the AI's confidence is trustworthy (average error < 10%)."""
        if not self._calibration: return True # Assume calibrated if no data
        errors = [c.calibration_error for c in self._calibration.values() if c.predictions >= 10]
        return statistics.mean(errors) < 0.10 if errors else True

    def get_calibration_error(self) -> float:
        """Returns the average calibration error across all confidence buckets."""
        if not self._calibration:
            return 0.0
        errors = [c.calibration_error for c in self._calibration.values() if c.predictions >= 10]
        return statistics.mean(errors) if errors else 0.0

    def should_stand_down(self, strategy: str, regime: str) -> Tuple[bool, str]:
        """
        Advises if the AI should stand down in a given context due to self-assessed weakness.
        This is a critical input for the MetacognitiveGovernor.
        """
        # 1. Check if performance in this context is known to be POOR.
        perf = self.get_performance(strategy, regime)
        if perf and perf.capability == Capability.POOR:
            return True, f"Self-assessed capability is POOR for {strategy} in {regime}."

        # 2. Check for any severe limitations related to this context.
        for lim in self._limitations.values():
            if lim.severity == 'severe' and (strategy in lim.context or regime in lim.context):
                return True, f"Severe known limitation: {lim.description}."

        # 3. Check if confidence is known to be badly miscalibrated.
        if self.get_calibration_error() > 0.25:
             return True, f"Confidence calibration error is critically high ({self.get_calibration_error():.1%})."

        return False, ""

    def _update_self_description(self):
        """Forces a regeneration of the cached natural language summary."""
        self._self_description = self._generate_self_description()
        
    def get_self_description(self) -> str:
        """Returns a cached, human-readable summary of the AI's self-knowledge."""
        return self._self_description or self._generate_self_description()

    def _generate_self_description(self) -> str:
        """Generates a fresh natural language summary of the AI's self-knowledge."""
        lines = ["I am a trading agent. Based on my experience, I have the following characteristics:"]
        
        if strengths := self.get_strengths():
            lines.append("\nMy Strengths:")
            lines.extend([f"  - I am effective at {s}." for s in strengths[:3]])
        else:
            lines.append("\nI have not yet identified any statistically significant strengths.")

        if weaknesses := self.get_weaknesses():
            lines.append("\nMy Weaknesses:")
            lines.extend([f"  - I am not effective at {w}." for w in weaknesses[:3]])
        else:
            lines.append("\nI have not yet identified any statistically significant weaknesses.")

        cal_error = self.get_calibration_error()
        cal_status = "well-calibrated" if self.is_well_calibrated() else "in need of calibration"
        lines.append(f"\nMy confidence is currently {cal_status} (error: {cal_error:.1%}).")
        
        total_trades = sum(p.total_trades for p in self._performance.values())
        lines.append(f"\nI have analyzed a total of {total_trades} trades to form this self-assessment.")
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Returns a dictionary of the self-model's current statistics."""
        return {
            'performance_records': len(self._performance),
            'known_limitations': len(self._limitations),
            'calibration_error': self.get_calibration_error(),
            'is_well_calibrated': self.is_well_calibrated(),
        }

    def _save_state(self) -> None:
        """Persists the self-model's state to a JSON file."""
        state = {
            'performance': {k: v.to_dict() for k, v in self._performance.items()},
            'limitations': {k: v.to_dict() for k, v in self._limitations.items()},
            'calibration': {k: asdict(v) for k, v in self._calibration.items()},
            'self_description': self._self_description,
        }
        try:
            with open(self.state_dir / "self_model.json", 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save self-model state: {e}")

    def _load_state(self) -> None:
        """Loads the self-model's state from a JSON file on startup."""
        state_file = self.state_dir / "self_model.json"
        if not state_file.exists(): return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            for key, data in state.get('performance', {}).items():
                self._performance[key] = StrategyPerformance(**{k:v for k,v in data.items() if k not in ['win_rate', 'capability', 'last_updated']})
            
            for key, data in state.get('limitations', {}).items():
                self._limitations[key] = Limitation(**{k:v for k,v in data.items() if k not in ['first_observed', 'last_observed']})

            for key, data in state.get('calibration', {}).items():
                self._calibration[key] = ConfidenceCalibration(**data)

            self._self_description = state.get('self_description', '')
            logger.info("SelfModel state loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load self-model state from {state_file}: {e}. Starting with a blank slate.")
            self._performance, self._limitations, self._calibration = {}, {}, {}


# --- Singleton Implementation ---
_self_model: Optional[SelfModel] = None
_lock = threading.Lock()

def get_self_model() -> SelfModel:
    """Factory function to get the singleton instance of the SelfModel."""
    global _self_model
    if _self_model is None:
        with _lock:
            if _self_model is None:
                _self_model = SelfModel()
    return _self_model
