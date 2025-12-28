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


@dataclass
class CognitiveEfficiencyRecord:
    """
    Tracks the AI's cognitive efficiency for different decision modes.
    This enables meta-metacognitive learning: the AI learns which thinking
    style (fast/slow) works best in which context.
    """
    strategy: str
    regime: str
    decision_mode: str  # 'fast', 'slow', 'hybrid'
    total_decisions: int = 0
    positive_pnl_count: int = 0
    total_pnl: float = 0.0
    avg_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Percentage of decisions that resulted in positive P&L."""
        if self.total_decisions == 0:
            return 0.0
        return self.positive_pnl_count / self.total_decisions

    @property
    def avg_pnl_per_decision(self) -> float:
        """Average P&L per decision."""
        if self.total_decisions == 0:
            return 0.0
        return self.total_pnl / self.total_decisions

    def to_dict(self) -> Dict:
        """Serializes the efficiency record to a dictionary."""
        return {
            'strategy': self.strategy,
            'regime': self.regime,
            'decision_mode': self.decision_mode,
            'total_decisions': self.total_decisions,
            'positive_pnl_count': self.positive_pnl_count,
            'total_pnl': self.total_pnl,
            'avg_time_ms': self.avg_time_ms,
            'success_rate': self.success_rate,
            'avg_pnl_per_decision': self.avg_pnl_per_decision,
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class CognitiveAdjustmentRecord:
    """Records a proposed or applied cognitive parameter adjustment."""
    adjustment_id: str
    param_name: str
    current_value: float
    proposed_value: float
    rationale: str
    context: str  # strategy|regime this applies to
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    outcome_pnl: Optional[float] = None

    def to_dict(self) -> Dict:
        """Serializes the adjustment record."""
        return {
            'adjustment_id': self.adjustment_id,
            'param_name': self.param_name,
            'current_value': self.current_value,
            'proposed_value': self.proposed_value,
            'rationale': self.rationale,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'applied': self.applied,
            'outcome_pnl': self.outcome_pnl,
        }


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

        # --- Meta-Metacognitive Stores (Task B1) ---
        self._cognitive_efficiency: Dict[str, CognitiveEfficiencyRecord] = {}
        self._adjustment_history: List[CognitiveAdjustmentRecord] = []
        self._pending_adjustments: List[CognitiveAdjustmentRecord] = []
        self._min_samples_for_adjustment = 20  # Min decisions before proposing adjustments
        self._max_adjustment_per_param = 0.10  # Max 10% adjustment per parameter

        self._self_description: str = "" # A cached natural language summary.

        self._load_state()
        logger.info(f"SelfModel initialized, loaded {len(self._performance)} performance records, {len(self._cognitive_efficiency)} efficiency records.")

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

    # ==========================================================================
    # COGNITIVE EFFICIENCY TRACKING (Task B1: Meta-Metacognitive Self-Config)
    # ==========================================================================

    def _make_efficiency_key(self, strategy: str, regime: str, decision_mode: str) -> str:
        """Creates a consistent dictionary key for efficiency tracking."""
        return f"{strategy.lower()}|{regime.lower()}|{decision_mode.lower()}"

    def record_cognitive_efficiency_feedback(
        self,
        decision_id: str,
        decision_mode_used: str,
        strategy: str,
        regime: str,
        actual_pnl: float,
        llm_critique_summary: Optional[str] = None,
        time_taken_ms: float = 0.0,
    ) -> None:
        """
        Records feedback about a cognitive decision's efficiency.

        This is called by the ReflectionEngine after a trade completes to track
        which decision mode (fast/slow/hybrid) produces the best outcomes in
        which contexts. This enables meta-metacognitive learning.

        Args:
            decision_id: Unique identifier for the decision
            decision_mode_used: 'fast', 'slow', or 'hybrid'
            strategy: The strategy that was used
            regime: The market regime at time of decision
            actual_pnl: The realized P&L from this decision
            llm_critique_summary: Optional LLM feedback on the decision
            time_taken_ms: Time taken to make the decision
        """
        key = self._make_efficiency_key(strategy, regime, decision_mode_used)

        if key not in self._cognitive_efficiency:
            self._cognitive_efficiency[key] = CognitiveEfficiencyRecord(
                strategy=strategy,
                regime=regime,
                decision_mode=decision_mode_used,
            )

        record = self._cognitive_efficiency[key]
        record.total_decisions += 1
        record.total_pnl += actual_pnl
        if actual_pnl > 0:
            record.positive_pnl_count += 1

        # Update rolling average for time
        n = record.total_decisions
        record.avg_time_ms = ((record.avg_time_ms * (n - 1)) + time_taken_ms) / n
        record.last_updated = datetime.now()

        # If there's a critique, potentially record it as a limitation
        if llm_critique_summary and actual_pnl < 0:
            self.record_limitation(
                context=f"{strategy}|{regime}|{decision_mode_used}",
                description=llm_critique_summary[:200],  # Truncate
                severity="minor" if abs(actual_pnl) < 50 else "moderate",
            )

        if self.auto_persist:
            self._save_state()

        logger.debug(
            f"Cognitive efficiency recorded: {key} - "
            f"Success rate: {record.success_rate:.1%}, "
            f"Avg P&L: ${record.avg_pnl_per_decision:.2f}"
        )

    def get_cognitive_efficiency(
        self,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        decision_mode: Optional[str] = None,
    ) -> List[CognitiveEfficiencyRecord]:
        """
        Gets cognitive efficiency records, optionally filtered.

        Args:
            strategy: Filter by strategy name
            regime: Filter by regime
            decision_mode: Filter by decision mode ('fast', 'slow', 'hybrid')

        Returns:
            List of matching CognitiveEfficiencyRecord objects
        """
        records = list(self._cognitive_efficiency.values())

        if strategy:
            records = [r for r in records if r.strategy.lower() == strategy.lower()]
        if regime:
            records = [r for r in records if r.regime.lower() == regime.lower()]
        if decision_mode:
            records = [r for r in records if r.decision_mode.lower() == decision_mode.lower()]

        return records

    def get_best_decision_mode(self, strategy: str, regime: str) -> Tuple[str, float]:
        """
        Determines the best decision mode for a given context based on historical performance.

        Returns:
            Tuple of (best_mode, confidence) where confidence is 0-1
        """
        modes = ['fast', 'slow', 'hybrid']
        mode_performance = {}

        for mode in modes:
            key = self._make_efficiency_key(strategy, regime, mode)
            if key in self._cognitive_efficiency:
                record = self._cognitive_efficiency[key]
                if record.total_decisions >= 5:  # Need some data
                    # Score = success_rate * avg_pnl (positive bias)
                    mode_performance[mode] = (
                        record.success_rate * 0.7 +
                        min(1.0, max(0.0, (record.avg_pnl_per_decision + 100) / 200)) * 0.3
                    )

        if not mode_performance:
            return 'slow', 0.5  # Default to slow with low confidence

        best_mode = max(mode_performance, key=mode_performance.get)
        confidence = min(1.0, mode_performance[best_mode])

        return best_mode, confidence

    def propose_cognitive_param_adjustments(self) -> Dict[str, Any]:
        """
        Analyzes cognitive efficiency data and proposes parameter adjustments.

        This is the core of meta-metacognitive self-configuration: the AI
        learns to tune its own cognitive parameters based on performance.

        Returns:
            Dict with 'adjustments' list and 'summary' string
        """
        import hashlib

        adjustments = []

        # Only propose adjustments if we have enough data
        total_decisions = sum(r.total_decisions for r in self._cognitive_efficiency.values())
        if total_decisions < self._min_samples_for_adjustment:
            return {
                'adjustments': [],
                'summary': f"Insufficient data for adjustments (need {self._min_samples_for_adjustment}, have {total_decisions}).",
            }

        # --- ANALYSIS 1: Fast vs Slow Path Efficiency ---
        fast_records = [r for r in self._cognitive_efficiency.values() if r.decision_mode == 'fast']
        slow_records = [r for r in self._cognitive_efficiency.values() if r.decision_mode == 'slow']

        if fast_records and slow_records:
            fast_success = sum(r.positive_pnl_count for r in fast_records) / max(1, sum(r.total_decisions for r in fast_records))
            slow_success = sum(r.positive_pnl_count for r in slow_records) / max(1, sum(r.total_decisions for r in slow_records))

            # If fast path is significantly better, consider raising fast_confidence_threshold
            if fast_success > slow_success + 0.10:  # Fast is 10%+ better
                adj_id = hashlib.md5(f"fast_confidence_threshold_{datetime.now().date()}".encode()).hexdigest()[:8]
                adjustments.append(CognitiveAdjustmentRecord(
                    adjustment_id=adj_id,
                    param_name='fast_confidence_threshold',
                    current_value=0.75,  # Default
                    proposed_value=max(0.65, 0.75 - self._max_adjustment_per_param),
                    rationale=f"Fast path success rate ({fast_success:.1%}) exceeds slow path ({slow_success:.1%}) by 10%+. Lowering threshold to use fast path more often.",
                    context='global',
                ))
            # If slow path is significantly better, raise the threshold
            elif slow_success > fast_success + 0.10:
                adj_id = hashlib.md5(f"fast_confidence_threshold_{datetime.now().date()}".encode()).hexdigest()[:8]
                adjustments.append(CognitiveAdjustmentRecord(
                    adjustment_id=adj_id,
                    param_name='fast_confidence_threshold',
                    current_value=0.75,
                    proposed_value=min(0.85, 0.75 + self._max_adjustment_per_param),
                    rationale=f"Slow path success rate ({slow_success:.1%}) exceeds fast path ({fast_success:.1%}) by 10%+. Raising threshold to use slow path more often.",
                    context='global',
                ))

        # --- ANALYSIS 2: Context-Specific Adjustments ---
        # Look for contexts where we consistently fail
        for key, record in self._cognitive_efficiency.items():
            if record.total_decisions >= 10 and record.success_rate < 0.35:
                adj_id = hashlib.md5(f"stand_down_{key}_{datetime.now().date()}".encode()).hexdigest()[:8]
                adjustments.append(CognitiveAdjustmentRecord(
                    adjustment_id=adj_id,
                    param_name='stand_down_threshold',
                    current_value=0.30,
                    proposed_value=min(0.45, 0.30 + self._max_adjustment_per_param),
                    rationale=f"Context {key} has {record.success_rate:.1%} success rate over {record.total_decisions} decisions. Raising stand-down threshold.",
                    context=key,
                ))

        # --- ANALYSIS 3: Time Efficiency ---
        # If slow path takes much longer but doesn't improve outcomes
        if fast_records and slow_records:
            fast_avg_time = sum(r.avg_time_ms * r.total_decisions for r in fast_records) / max(1, sum(r.total_decisions for r in fast_records))
            slow_avg_time = sum(r.avg_time_ms * r.total_decisions for r in slow_records) / max(1, sum(r.total_decisions for r in slow_records))

            if slow_avg_time > fast_avg_time * 5 and slow_success <= fast_success:
                adj_id = hashlib.md5(f"slow_budget_{datetime.now().date()}".encode()).hexdigest()[:8]
                adjustments.append(CognitiveAdjustmentRecord(
                    adjustment_id=adj_id,
                    param_name='default_slow_budget_ms',
                    current_value=5000,
                    proposed_value=max(2000, 5000 - 1000),
                    rationale=f"Slow path takes {slow_avg_time:.0f}ms (5x+ fast path) but doesn't improve outcomes. Reducing time budget.",
                    context='global',
                ))

        # Store pending adjustments for possible application
        self._pending_adjustments = adjustments
        self._adjustment_history.extend(adjustments)

        if self.auto_persist:
            self._save_state()

        return {
            'adjustments': [adj.to_dict() for adj in adjustments],
            'summary': f"Proposed {len(adjustments)} cognitive parameter adjustments based on {total_decisions} decisions.",
            'pending_count': len(adjustments),
        }

    def get_pending_adjustments(self) -> List[CognitiveAdjustmentRecord]:
        """Returns pending adjustments that haven't been applied yet."""
        return [adj for adj in self._pending_adjustments if not adj.applied]

    def mark_adjustment_applied(self, adjustment_id: str) -> bool:
        """Marks an adjustment as applied."""
        for adj in self._pending_adjustments:
            if adj.adjustment_id == adjustment_id:
                adj.applied = True
                if self.auto_persist:
                    self._save_state()
                return True
        return False

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
        total_efficiency_decisions = sum(
            r.total_decisions for r in self._cognitive_efficiency.values()
        )
        return {
            'performance_records': len(self._performance),
            'known_limitations': len(self._limitations),
            'calibration_error': self.get_calibration_error(),
            'is_well_calibrated': self.is_well_calibrated(),
            # Meta-metacognitive stats (Task B1)
            'cognitive_efficiency_records': len(self._cognitive_efficiency),
            'total_efficiency_decisions': total_efficiency_decisions,
            'pending_adjustments': len(self.get_pending_adjustments()),
            'adjustment_history_count': len(self._adjustment_history),
        }

    def _save_state(self) -> None:
        """Persists the self-model's state to a JSON file."""
        state = {
            'performance': {k: v.to_dict() for k, v in self._performance.items()},
            'limitations': {k: v.to_dict() for k, v in self._limitations.items()},
            'calibration': {k: asdict(v) for k, v in self._calibration.items()},
            'self_description': self._self_description,
            # Meta-metacognitive state (Task B1)
            'cognitive_efficiency': {k: v.to_dict() for k, v in self._cognitive_efficiency.items()},
            'adjustment_history': [adj.to_dict() for adj in self._adjustment_history[-100:]],  # Keep last 100
            'pending_adjustments': [adj.to_dict() for adj in self._pending_adjustments],
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

            # Load meta-metacognitive state (Task B1)
            for key, data in state.get('cognitive_efficiency', {}).items():
                excluded = ['success_rate', 'avg_pnl_per_decision', 'last_updated']
                self._cognitive_efficiency[key] = CognitiveEfficiencyRecord(
                    **{k: v for k, v in data.items() if k not in excluded}
                )

            for adj_data in state.get('adjustment_history', []):
                excluded = ['timestamp']
                self._adjustment_history.append(CognitiveAdjustmentRecord(
                    **{k: v for k, v in adj_data.items() if k not in excluded}
                ))

            for adj_data in state.get('pending_adjustments', []):
                excluded = ['timestamp']
                self._pending_adjustments.append(CognitiveAdjustmentRecord(
                    **{k: v for k, v in adj_data.items() if k not in excluded}
                ))

            logger.info("SelfModel state loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load self-model state from {state_file}: {e}. Starting with a blank slate.")
            self._performance, self._limitations, self._calibration = {}, {}, {}
            self._cognitive_efficiency = {}
            self._adjustment_history, self._pending_adjustments = [], []


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
