"""
Learning Hub - The Central Nervous System of Kobe's Learning
=============================================================

This is the critical integration component that wires together all of Kobe's
cognitive and learning subsystems. Before this existed, components worked in
isolation - episodic memory stored episodes nobody read, semantic memory had
rules nobody applied, and the online learning manager was never fed trade outcomes.

The LearningHub fixes this by providing a single entry point that:
1. Routes trade outcomes to ALL relevant learning systems
2. Triggers reflection for significant trades
3. Extracts and stores semantic rules from insights
4. Enables online model updates
5. Detects concept drift and alerts

Usage:
    from integration.learning_hub import get_learning_hub

    hub = get_learning_hub()

    # After a trade closes, call this ONCE and everything gets wired:
    await hub.process_trade_outcome(trade_outcome)

    # The hub handles:
    # - Storing in episodic memory
    # - Feeding to online learning
    # - Triggering reflection (if significant)
    # - Extracting semantic rules
    # - Checking for concept drift

Author: Kobe Trading System
Created: 2026-01-04
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json
import numpy as np

from core.structured_log import get_logger

logger = get_logger(__name__)


@dataclass
class TradeOutcomeEvent:
    """
    Complete information about a closed trade, used to feed all learning systems.
    """
    # Core trade info
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    shares: int
    entry_time: datetime
    exit_time: datetime

    # Outcome
    pnl: float  # Absolute P&L
    pnl_pct: float  # Percentage P&L
    won: bool  # True if profitable

    # Context at entry
    signal_score: float = 0.0  # ML confidence score at entry
    pattern_type: str = ""  # e.g., 'ibs_rsi', 'turtle_soup'
    regime: str = "unknown"  # Market regime at entry

    # Features for ML (optional)
    entry_features: Optional[np.ndarray] = None

    # Exit reason
    exit_reason: str = ""  # 'stop_loss', 'take_profit', 'time_stop', 'manual'

    # Metadata
    trade_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def holding_period(self) -> int:
        """Returns holding period in bars/days."""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).days
        return 0

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'won': self.won,
            'signal_score': self.signal_score,
            'pattern_type': self.pattern_type,
            'regime': self.regime,
            'exit_reason': self.exit_reason,
            'trade_id': self.trade_id,
            'holding_period': self.holding_period,
            'metadata': self.metadata
        }


class LearningHub:
    """
    Central integration hub that routes trade outcomes to all learning systems.

    This is the "nervous system" that was missing - it connects:
    - Trade outcomes → Episodic Memory (experience storage)
    - Trade outcomes → Online Learning Manager (model updates)
    - Significant trades → Reflection Engine (self-critique)
    - Insights → Semantic Memory (rule extraction)

    Configuration via config:
    - reflection_threshold_pct: Trigger reflection for trades with |pnl%| > this
    - min_trades_for_model_update: Minimum trades before triggering model update
    - enable_auto_reflection: Whether to auto-trigger reflection
    - enable_concept_drift_alerts: Whether to alert on drift detection
    """

    STATE_FILE = Path("state/integration/learning_hub_state.json")

    def __init__(
        self,
        reflection_threshold_pct: float = 0.03,
        min_trades_for_model_update: int = 20,
        enable_auto_reflection: bool = True,
        enable_concept_drift_alerts: bool = True,
        alerter: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the Learning Hub.

        Args:
            reflection_threshold_pct: Trigger reflection for |pnl%| > this (default 3%)
            min_trades_for_model_update: Batch size for incremental training
            enable_auto_reflection: Auto-trigger reflection for significant trades
            enable_concept_drift_alerts: Send alerts on concept drift detection
            alerter: Callback function for sending alerts (e.g., Telegram)
        """
        self.reflection_threshold_pct = reflection_threshold_pct
        self.min_trades_for_model_update = min_trades_for_model_update
        self.enable_auto_reflection = enable_auto_reflection
        self.enable_concept_drift_alerts = enable_concept_drift_alerts
        self.alerter = alerter

        # Lazy-loaded components
        self._episodic_memory = None
        self._semantic_memory = None
        self._reflection_engine = None
        self._online_learning = None

        # Stats tracking
        self.trades_processed = 0
        self.reflections_triggered = 0
        self.rules_extracted = 0
        self.model_updates_triggered = 0
        self.drift_alerts_sent = 0
        self.started_at = datetime.now()

        # FIX (2026-01-07): Fix 1 - Track pending entries for matching with exits
        # FIX (2026-01-07): Fix 1a - Thread safety for concurrent access
        self._entries_lock = threading.RLock()
        self._pending_entries: Dict[str, Dict[str, Any]] = {}

        # FIX (2026-01-07): Fix 1b - Persistence path for crash recovery
        self._pending_entries_file = Path("state/integration/pending_entries.json")

        # Ensure state directory exists
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._pending_entries_file.parent.mkdir(parents=True, exist_ok=True)

        # Load any persisted pending entries from previous session
        self._load_pending_entries()

        logger.info(
            f"LearningHub initialized: reflection_threshold={reflection_threshold_pct}, "
            f"min_trades_for_update={min_trades_for_model_update}, auto_reflection={enable_auto_reflection}"
        )

    # --- Lazy-loaded properties for dependencies ---

    @property
    def episodic_memory(self):
        """Lazy-load episodic memory to avoid circular imports."""
        if self._episodic_memory is None:
            from cognitive.episodic_memory import get_episodic_memory
            self._episodic_memory = get_episodic_memory()
        return self._episodic_memory

    @property
    def semantic_memory(self):
        """Lazy-load semantic memory to avoid circular imports."""
        if self._semantic_memory is None:
            from cognitive.semantic_memory import get_semantic_memory
            self._semantic_memory = get_semantic_memory()
        return self._semantic_memory

    @property
    def reflection_engine(self):
        """Lazy-load reflection engine to avoid circular imports."""
        if self._reflection_engine is None:
            from cognitive.reflection_engine import ReflectionEngine
            self._reflection_engine = ReflectionEngine()
        return self._reflection_engine

    @property
    def online_learning(self):
        """Lazy-load online learning manager to avoid circular imports."""
        if self._online_learning is None:
            from ml_advanced.online_learning import OnlineLearningManager
            self._online_learning = OnlineLearningManager(
                update_frequency='trade',  # Update per trade batch
                auto_update=True
            )
        return self._online_learning

    # --- Entry Recording (Fix 1 - 2026-01-07) ---

    def record_trade_entry(self, entry_data: Dict[str, Any]) -> None:
        """
        Record a trade entry for later matching with exit.

        FIX (2026-01-07): Fix 1 - Wire trade outcomes to Learning Hub.
        This method is called when a trade fills to record entry data.
        When the position closes, the entry data is matched with exit data
        to create a complete TradeOutcomeEvent for learning.

        FIX (2026-01-07): Fix 1a - Thread-safe with RLock
        FIX (2026-01-07): Fix 1b - Persisted to disk for crash recovery

        Args:
            entry_data: Dict containing:
                - symbol: Stock symbol
                - side: 'long' or 'short'
                - entry_price: Fill price
                - shares: Number of shares
                - strategy: Strategy name
                - trade_id: Unique trade identifier
                - entry_time: ISO timestamp
                - stop_loss: Optional stop loss price
                - take_profit: Optional take profit price
        """
        symbol = entry_data.get('symbol')
        if not symbol:
            logger.warning("record_trade_entry called without symbol")
            return

        # Use symbol|side as key to handle multiple positions in same symbol
        side = entry_data.get('side', 'long')
        key = f"{symbol}|{side}"

        with self._entries_lock:
            self._pending_entries[key] = entry_data
            self._persist_pending_entries()

        logger.info(
            f"Recorded trade entry: {symbol} {side} "
            f"{entry_data.get('shares')} @ ${entry_data.get('entry_price', 0):.2f}"
        )

    def get_pending_entry(self, symbol: str, side: str = 'long') -> Optional[Dict[str, Any]]:
        """Get pending entry data for a symbol, if any."""
        key = f"{symbol}|{side}"
        with self._entries_lock:
            return self._pending_entries.get(key)

    def clear_pending_entry(self, symbol: str, side: str = 'long') -> None:
        """Clear pending entry after it's been processed."""
        key = f"{symbol}|{side}"
        with self._entries_lock:
            if key in self._pending_entries:
                del self._pending_entries[key]
                self._persist_pending_entries()

    def _persist_pending_entries(self) -> None:
        """Persist pending entries to disk for crash recovery."""
        try:
            with open(self._pending_entries_file, 'w') as f:
                json.dump(self._pending_entries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist pending entries: {e}")

    def _load_pending_entries(self) -> None:
        """Load pending entries from disk on startup."""
        try:
            if self._pending_entries_file.exists():
                with open(self._pending_entries_file, 'r') as f:
                    self._pending_entries = json.load(f)
                if self._pending_entries:
                    logger.info(f"Loaded {len(self._pending_entries)} pending entries from previous session")
        except Exception as e:
            logger.warning(f"Failed to load pending entries: {e}")
            self._pending_entries = {}

    async def record_exit_and_learn(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        side: str = 'long',
        exit_time: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method to record a trade exit and trigger learning.

        FIX (2026-01-07): Fix 1c - Single entry point for ALL exit paths.
        Call this from ANY exit path (time-stop, stop-loss, take-profit, manual, external).

        Args:
            symbol: Stock symbol
            exit_price: Price at which position was closed
            exit_reason: One of 'time_stop', 'stop_loss', 'take_profit', 'manual', 'external'
            side: 'long' or 'short' (default 'long')
            exit_time: Optional datetime, defaults to now

        Returns:
            Processing results dict, or None if no pending entry found
        """
        entry = self.get_pending_entry(symbol, side)
        if not entry:
            logger.warning(f"record_exit_and_learn: No pending entry for {symbol}|{side}")
            return None

        exit_time = exit_time or datetime.now()
        entry_price = entry.get('entry_price', 0)
        shares = entry.get('shares', 0)

        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        else:
            pnl = (entry_price - exit_price) * shares
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0

        # Parse entry time
        entry_time_str = entry.get('entry_time')
        if entry_time_str:
            try:
                entry_time = datetime.fromisoformat(entry_time_str)
            except (ValueError, TypeError):
                entry_time = datetime.now()
        else:
            entry_time = datetime.now()

        # Create TradeOutcomeEvent
        outcome = TradeOutcomeEvent(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            won=pnl > 0,
            signal_score=entry.get('signal_score', 0.0),
            pattern_type=entry.get('strategy', 'dual_strategy'),
            regime=entry.get('regime', 'unknown'),
            exit_reason=exit_reason,
            trade_id=entry.get('trade_id', ''),
            metadata=entry.get('metadata', {})
        )

        # Process through learning systems
        result = await self.process_trade_outcome(outcome)

        # Clear the pending entry
        self.clear_pending_entry(symbol, side)

        logger.info(
            f"Exit recorded and learning triggered: {symbol} {side} "
            f"exit_reason={exit_reason} pnl=${pnl:.2f} ({pnl_pct:.2%})"
        )

        return result

    # --- Main Integration Method ---

    async def process_trade_outcome(self, trade: TradeOutcomeEvent) -> Dict[str, Any]:
        """
        Process a completed trade through ALL learning systems.

        This is the SINGLE entry point that wires everything together.
        Call this ONCE after a trade closes and it handles everything.

        Returns:
            Dict with processing results from each system
        """
        results = {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'pnl_pct': trade.pnl_pct,
            'steps_completed': [],
            'errors': []
        }

        logger.info(
            f"Processing trade outcome: symbol={trade.symbol}, pnl_pct={trade.pnl_pct:.2%}, won={trade.won}"
        )

        # Step 1: Store in Episodic Memory
        try:
            episode_id = await self._store_in_episodic_memory(trade)
            results['episode_id'] = episode_id
            results['steps_completed'].append('episodic_memory')
        except Exception as e:
            logger.error(f"Failed to store in episodic memory: {e}")
            results['errors'].append(f"episodic_memory: {e}")

        # Step 2: Feed to Online Learning Manager
        try:
            self._feed_to_online_learning(trade)
            results['steps_completed'].append('online_learning')
        except Exception as e:
            logger.error(f"Failed to feed to online learning: {e}")
            results['errors'].append(f"online_learning: {e}")

        # Step 3: Trigger Reflection for Significant Trades
        if self.enable_auto_reflection and self._should_reflect(trade):
            try:
                reflection = await self._trigger_reflection(trade, results.get('episode_id'))
                results['reflection'] = reflection.to_dict() if reflection else None
                results['steps_completed'].append('reflection')
                self.reflections_triggered += 1

                # Step 4: Extract Semantic Rules from Reflection
                if reflection and reflection.lessons:
                    try:
                        rules_added = self._extract_semantic_rules(trade, reflection)
                        results['rules_added'] = rules_added
                        results['steps_completed'].append('semantic_rules')
                        self.rules_extracted += rules_added
                    except Exception as e:
                        logger.error(f"Failed to extract semantic rules: {e}")
                        results['errors'].append(f"semantic_rules: {e}")

            except Exception as e:
                logger.error(f"Failed to trigger reflection: {e}")
                results['errors'].append(f"reflection: {e}")

        # Step 5: Check for Model Updates
        try:
            update_triggered = self._check_and_trigger_model_update()
            results['model_update_triggered'] = update_triggered
            if update_triggered:
                results['steps_completed'].append('model_update')
                self.model_updates_triggered += 1
        except Exception as e:
            logger.error(f"Failed to check model update: {e}")
            results['errors'].append(f"model_update: {e}")

        # Step 6: Check for Concept Drift
        if self.enable_concept_drift_alerts:
            try:
                drift_detected = self._check_concept_drift(trade)
                results['drift_detected'] = drift_detected
                if drift_detected:
                    results['steps_completed'].append('drift_alert')
                    self.drift_alerts_sent += 1
            except Exception as e:
                logger.error(f"Failed to check concept drift: {e}")
                results['errors'].append(f"concept_drift: {e}")

        # Step 7: CRITICAL FIX (2026-01-08) - Close the learning feedback loop
        # Update semantic memory rules that were applied for this trade
        try:
            rules_updated = self._update_rule_outcomes(trade)
            results['rules_updated'] = rules_updated
            if rules_updated > 0:
                results['steps_completed'].append('rule_feedback_loop')
        except Exception as e:
            logger.error(f"Failed to update rule outcomes: {e}")
            results['errors'].append(f"rule_feedback_loop: {e}")

        self.trades_processed += 1

        # Save state periodically
        if self.trades_processed % 10 == 0:
            self._save_state()

        logger.info(
            f"Trade outcome processing complete: steps={results['steps_completed']}, errors={len(results['errors'])}"
        )

        return results

    # --- Internal Methods ---

    async def _store_in_episodic_memory(self, trade: TradeOutcomeEvent) -> str:
        """Store trade as complete episode in episodic memory."""
        # Build market context
        market_context = {
            'symbol': trade.symbol,
            'regime': trade.regime,
            'entry_price': trade.entry_price,
            'timestamp': trade.entry_time.isoformat() if trade.entry_time else None
        }

        # Build signal context
        signal_context = {
            'pattern_type': trade.pattern_type,
            'signal_score': trade.signal_score,
            'side': trade.side
        }

        # Start episode
        episode_id = self.episodic_memory.start_episode(
            market_context=market_context,
            signal_context=signal_context
        )

        # Add reasoning
        reasoning = f"Entered {trade.side} on {trade.symbol} based on {trade.pattern_type} pattern with {trade.signal_score:.2f} confidence in {trade.regime} regime."
        self.episodic_memory.add_reasoning(episode_id, reasoning)

        # Add action
        action = {
            'type': trade.side,
            'symbol': trade.symbol,
            'shares': trade.shares,
            'entry_price': trade.entry_price
        }
        self.episodic_memory.add_action(episode_id, action)

        # Complete with outcome
        outcome = {
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'won': trade.won,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'holding_period': trade.holding_period
        }
        self.episodic_memory.complete_episode(episode_id, outcome=outcome)

        logger.debug(f"Stored episode {episode_id} for {trade.symbol}")
        return episode_id

    def _feed_to_online_learning(self, trade: TradeOutcomeEvent):
        """Feed trade outcome to online learning manager."""
        # Use entry features if available, otherwise create minimal features
        features = trade.entry_features
        if features is None:
            # Create minimal feature vector from available data
            features = np.array([
                trade.signal_score,
                trade.entry_price,
                1.0 if trade.side == 'long' else 0.0,
                trade.holding_period
            ])

        self.online_learning.record_trade_outcome(
            symbol=trade.symbol,
            features=features,
            prediction=trade.signal_score,
            actual_pnl=trade.pnl_pct,
            holding_period=trade.holding_period
        )

        logger.debug(f"Fed trade to online learning: {trade.symbol}")

    def _should_reflect(self, trade: TradeOutcomeEvent) -> bool:
        """Determine if this trade warrants reflection."""
        # Reflect on significant wins or losses
        if abs(trade.pnl_pct) >= self.reflection_threshold_pct:
            return True

        # Always reflect on losses (to learn from mistakes)
        if not trade.won and trade.pnl_pct < -0.01:
            return True

        return False

    async def _trigger_reflection(self, trade: TradeOutcomeEvent, episode_id: Optional[str]) -> Optional[Any]:
        """Trigger reflection engine on this trade."""
        if not episode_id:
            logger.warning("Cannot reflect without episode_id")
            return None

        # Get the episode
        episode = self.episodic_memory.get_episode(episode_id)
        if not episode:
            logger.warning(f"Episode {episode_id} not found")
            return None

        # Run reflection
        reflection = self.reflection_engine.reflect_on_episode(episode)

        logger.info(
            f"Reflection triggered for {trade.symbol}: lessons={len(reflection.lessons) if reflection else 0}"
        )

        return reflection

    def _extract_semantic_rules(self, trade: TradeOutcomeEvent, reflection: Any) -> int:
        """Extract and store semantic rules from reflection insights."""
        rules_added = 0

        for lesson in reflection.lessons[:3]:  # Max 3 rules per reflection
            # Build condition from trade context
            condition = f"pattern_type = {trade.pattern_type} AND regime = {trade.regime}"

            # Determine action based on outcome
            if trade.won:
                action = "increase_confidence"
                confidence = 0.6
            else:
                action = "decrease_confidence"
                confidence = 0.5

            # Add rule
            self.semantic_memory.add_rule(
                condition=condition,
                action=action,
                parameters={
                    'lesson': lesson,
                    'source_trade': trade.trade_id,
                    'pnl_pct': trade.pnl_pct
                },
                confidence=confidence,
                source='learning_hub_auto'
            )
            rules_added += 1

        # Also process behavior changes from reflection
        for change in reflection.behavior_changes[:2]:  # Max 2 behavior changes
            if isinstance(change, dict) and 'condition' in change and 'action' in change:
                self.semantic_memory.add_rule(
                    condition=change.get('condition', ''),
                    action=change.get('action', ''),
                    parameters=change.get('parameters', {}),
                    confidence=change.get('confidence', 0.5),
                    source='reflection_behavior_change'
                )
                rules_added += 1

        if rules_added > 0:
            logger.info(f"Extracted {rules_added} semantic rules from reflection")

        return rules_added

    def _check_and_trigger_model_update(self) -> bool:
        """Check if we should trigger incremental model training."""
        if not self.online_learning.should_update():
            return False

        buffer_size = len(self.online_learning.replay_buffer)
        if buffer_size >= self.min_trades_for_model_update:
            logger.info(f"Model update triggered (buffer size: {buffer_size})")

            # FIX (2026-01-07): Actually perform model retraining
            try:
                # Trigger incremental update with batch size
                batch_size = min(buffer_size, 100)  # Max 100 samples per batch
                result = self.online_learning.trigger_incremental_update(batch_size)

                if result.get("status") == "completed":
                    logger.info(
                        f"Model retraining complete: {result.get('samples_trained', 0)} samples, "
                        f"WR={result.get('batch_win_rate', 0):.1%}"
                    )
                    return True
                else:
                    logger.warning(f"Model retraining skipped: {result.get('reason', 'unknown')}")
                    return False

            except Exception as e:
                logger.error(f"Model retraining failed: {e}")
                # Don't crash - learning is advisory, not critical
                return False

        return False

    def _check_concept_drift(self, trade: TradeOutcomeEvent) -> bool:
        """Check for concept drift and send alert if detected."""
        # Update drift detector
        drift_status = self.online_learning.drift_detector.update(
            prediction=trade.signal_score,
            actual=1 if trade.won else 0
        )

        if drift_status == 'drift':
            message = (
                f"CONCEPT DRIFT DETECTED\n"
                f"Recent accuracy: {self.online_learning.drift_detector.get_current_accuracy():.1%}\n"
                f"Baseline accuracy: {self.online_learning.drift_detector.baseline_accuracy:.1%}\n"
                f"Consider full model retrain."
            )
            logger.warning(message)

            if self.alerter:
                self.alerter(message)

            return True

        return False

    def _update_rule_outcomes(self, trade: TradeOutcomeEvent) -> int:
        """
        CRITICAL FIX (2026-01-08): Close the learning feedback loop.

        Updates semantic memory rules that were applied when this trade was entered.
        This is the KEY missing piece - rules were being created but never updated
        with their outcomes, so times_applied was always 0.

        Args:
            trade: The completed trade outcome

        Returns:
            Number of rules updated
        """
        rules_updated = 0

        # Get rules that were applied from trade metadata
        rules_applied = trade.metadata.get('rules_applied', [])

        # Also try to find matching rules by pattern type and regime
        # This catches rules even if they weren't explicitly tracked
        if not rules_applied:
            # Find rules that match this trade's context
            matching_rules = self._find_matching_rules(trade)
            rules_applied = [r.rule_id for r in matching_rules]

        # Update each rule with the trade outcome
        for rule_id in rules_applied:
            try:
                self.semantic_memory.record_rule_outcome(
                    rule_id=rule_id,
                    successful=trade.won
                )
                rules_updated += 1
                logger.debug(f"Updated rule outcome: {rule_id} -> won={trade.won}")
            except Exception as e:
                logger.warning(f"Failed to update rule {rule_id}: {e}")

        if rules_updated > 0:
            logger.info(
                f"Feedback loop closed: updated {rules_updated} rules "
                f"for {trade.symbol} (won={trade.won})"
            )

        return rules_updated

    def _find_matching_rules(self, trade: TradeOutcomeEvent) -> list:
        """
        Find semantic rules that match the trade's context.

        This allows feedback even for trades where rules weren't explicitly tracked.
        """
        matching = []

        try:
            # Build context for rule matching
            context = {
                'pattern_type': trade.pattern_type,
                'regime': trade.regime,
                'symbol': trade.symbol,
                'side': trade.side,
            }

            # Get all active rules and check which ones match
            all_rules = self.semantic_memory.get_all_rules()
            for rule in all_rules:
                if not rule.is_active:
                    continue

                # Simple condition matching
                condition_lower = rule.condition.lower()

                # Check if rule condition matches trade context
                if trade.pattern_type and trade.pattern_type.lower() in condition_lower:
                    matching.append(rule)
                elif trade.regime and trade.regime.lower() in condition_lower:
                    if not matching or rule not in matching:  # Avoid duplicates
                        matching.append(rule)

        except Exception as e:
            logger.warning(f"Error finding matching rules: {e}")

        return matching[:5]  # Limit to 5 most relevant rules

    # --- State Management ---

    def _save_state(self):
        """Save hub state to disk."""
        state = {
            'trades_processed': self.trades_processed,
            'reflections_triggered': self.reflections_triggered,
            'rules_extracted': self.rules_extracted,
            'model_updates_triggered': self.model_updates_triggered,
            'drift_alerts_sent': self.drift_alerts_sent,
            'started_at': self.started_at.isoformat(),
            'saved_at': datetime.now().isoformat()
        }

        with open(self.STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load hub state from disk."""
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)
            self.trades_processed = state.get('trades_processed', 0)
            self.reflections_triggered = state.get('reflections_triggered', 0)
            self.rules_extracted = state.get('rules_extracted', 0)
            self.model_updates_triggered = state.get('model_updates_triggered', 0)
            self.drift_alerts_sent = state.get('drift_alerts_sent', 0)
            logger.info("LearningHub state loaded from disk")

    def get_status(self) -> Dict:
        """Get current hub status."""
        return {
            'trades_processed': self.trades_processed,
            'reflections_triggered': self.reflections_triggered,
            'rules_extracted': self.rules_extracted,
            'model_updates_triggered': self.model_updates_triggered,
            'drift_alerts_sent': self.drift_alerts_sent,
            'online_learning_status': self.online_learning.get_status(),
            'uptime_hours': (datetime.now() - self.started_at).total_seconds() / 3600
        }


# --- Singleton Pattern ---

_learning_hub: Optional[LearningHub] = None


def get_learning_hub() -> LearningHub:
    """Get or create the singleton LearningHub instance."""
    global _learning_hub
    if _learning_hub is None:
        _learning_hub = LearningHub()
        _learning_hub.load_state()
    return _learning_hub


def reset_learning_hub():
    """Reset the singleton (mainly for testing)."""
    global _learning_hub
    _learning_hub = None
