"""
Autonomous Learning Module for Kobe.

This module enables Kobe to learn from its experiences:
- Analyze trade outcomes
- Update episodic memory
- Generate daily reflections
- Identify patterns in success/failure
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class TradeLesson:
    """A lesson learned from a trade."""
    id: str
    trade_date: datetime
    symbol: str
    outcome: str  # "win", "loss", "breakeven"
    pnl: float
    entry_reason: str
    exit_reason: str
    what_worked: List[str]
    what_failed: List[str]
    lesson: str
    confidence: float = 0.5


@dataclass
class DailyReflection:
    """Daily reflection on trading performance."""
    date: datetime
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    best_trade: Optional[str]
    worst_trade: Optional[str]
    lessons_learned: List[str]
    tomorrow_focus: List[str]
    mood: str  # "confident", "cautious", "learning", "struggling"


class LearningEngine:
    """
    The learning engine that helps Kobe improve from experience.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous/learning")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.lessons: List[TradeLesson] = []
        self.reflections: List[DailyReflection] = []

        self._load_state()

    def _load_state(self):
        """Load learning state."""
        state_file = self.state_dir / "learning_state.json"
        if state_file.exists():
            try:
                json.loads(state_file.read_text())
                # Lessons and reflections would be restored here
            except Exception as e:
                logger.warning(f"Could not load learning state: {e}")

    def save_state(self):
        """Save learning state."""
        state_file = self.state_dir / "learning_state.json"
        data = {
            "updated_at": datetime.now(ET).isoformat(),
            "lessons_count": len(self.lessons),
            "reflections_count": len(self.reflections),
            "recent_lessons": [
                {
                    "id": l.id,
                    "symbol": l.symbol,
                    "outcome": l.outcome,
                    "lesson": l.lesson,
                    "trade_date": l.trade_date.isoformat(),
                }
                for l in self.lessons[-20:]
            ],
        }
        state_file.write_text(json.dumps(data, indent=2))

    def analyze_trades(self) -> Dict[str, Any]:
        """Analyze recent trades and extract lessons."""
        logger.info("Analyzing recent trades...")

        try:
            # Load trade log
            trade_log = Path("logs/trades.jsonl")
            if not trade_log.exists():
                return {"status": "no_trades", "message": "No trade log found"}

            trades = []
            with open(trade_log) as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue

            if not trades:
                return {"status": "no_trades", "message": "Trade log is empty"}

            # Analyze last 7 days
            cutoff = datetime.now(ET) - timedelta(days=7)
            recent_trades = [
                t for t in trades
                if datetime.fromisoformat(t.get("timestamp", "2000-01-01")) > cutoff
            ]

            lessons_found = []

            for trade in recent_trades:
                lesson = self._analyze_single_trade(trade)
                if lesson:
                    self.lessons.append(lesson)
                    lessons_found.append({
                        "symbol": lesson.symbol,
                        "outcome": lesson.outcome,
                        "lesson": lesson.lesson,
                    })

            self.save_state()

            return {
                "status": "success",
                "trades_analyzed": len(recent_trades),
                "lessons_found": len(lessons_found),
                "lessons": lessons_found,
                # Include raw trades data for LearningHub integration
                "trades_data": recent_trades,
            }

        except Exception as e:
            logger.error(f"Trade analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_single_trade(self, trade: Dict[str, Any]) -> Optional[TradeLesson]:
        """Analyze a single trade for lessons."""
        try:
            pnl = trade.get("pnl", 0)
            symbol = trade.get("symbol", "UNKNOWN")

            outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"

            # Extract lessons based on outcome
            what_worked = []
            what_failed = []
            lesson = ""

            entry_reason = trade.get("entry_reason", "signal")
            exit_reason = trade.get("exit_reason", "unknown")

            if outcome == "win":
                what_worked.append("Entry signal was correct")
                if "target" in exit_reason.lower():
                    what_worked.append("Take profit level was appropriate")
                lesson = f"Profitable trade on {symbol} - signal quality was good"

            elif outcome == "loss":
                what_failed.append("Entry timing or signal quality")
                if "stop" in exit_reason.lower():
                    what_failed.append("Stop was hit - possible false signal")
                    lesson = f"Stop hit on {symbol} - review entry criteria"
                elif "time" in exit_reason.lower():
                    what_failed.append("Time stop - no momentum")
                    lesson = f"Time stop on {symbol} - may need stronger entry filter"

            return TradeLesson(
                id=f"lesson_{datetime.now(ET).strftime('%Y%m%d%H%M%S')}",
                trade_date=datetime.fromisoformat(
                    trade.get("timestamp", datetime.now(ET).isoformat())
                ),
                symbol=symbol,
                outcome=outcome,
                pnl=pnl,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                what_worked=what_worked,
                what_failed=what_failed,
                lesson=lesson,
            )

        except Exception as e:
            logger.warning(f"Could not analyze trade: {e}")
            return None

    def update_memory(self) -> Dict[str, Any]:
        """Update cognitive episodic memory with lessons."""
        logger.info("Updating episodic memory...")

        try:
            from cognitive.episodic_memory import EpisodicMemory

            memory = EpisodicMemory()

            # Store recent lessons as episodes
            stored = 0
            for lesson in self.lessons[-10:]:  # Last 10 lessons
                episode = {
                    "type": "trade_lesson",
                    "symbol": lesson.symbol,
                    "outcome": lesson.outcome,
                    "lesson": lesson.lesson,
                    "confidence": lesson.confidence,
                    "what_worked": lesson.what_worked,
                    "what_failed": lesson.what_failed,
                }
                memory.store(
                    context=f"Trade {lesson.symbol} {lesson.outcome}",
                    reasoning=lesson.lesson,
                    outcome=lesson.outcome,
                    metadata=episode,
                )
                stored += 1

            return {
                "status": "success",
                "episodes_stored": stored,
            }

        except ImportError:
            logger.warning("Cognitive memory not available")
            return {"status": "not_available", "message": "Cognitive memory module not loaded"}
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            return {"status": "error", "error": str(e)}

    def record_trade_experience(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        === TIER 2.3: Wire Episodic/Semantic Memory Updates ===

        Records a trade experience to both episodic and semantic memory systems.
        - Episodic memory: Stores the specific trade context and outcome
        - Semantic memory: Reinforces or weakens patterns based on outcomes

        Args:
            trade: Dict with symbol, pnl, reason/pattern_type, regime, etc.

        Returns:
            Dict with status and memory update results
        """
        from core.structured_log import jlog

        results = {
            "episodic_stored": False,
            "semantic_updated": False,
            "errors": []
        }

        symbol = trade.get('symbol', 'UNKNOWN')
        pnl = trade.get('pnl', 0)
        won = pnl > 0
        pattern = trade.get('reason', trade.get('pattern_type', trade.get('entry_reason', 'unknown')))
        regime = trade.get('regime', 'UNKNOWN')

        # === Store to Episodic Memory ===
        try:
            from cognitive.episodic_memory import EpisodicMemory

            episodic = EpisodicMemory()

            episode = {
                'timestamp': trade.get('timestamp', datetime.now(ET).isoformat()),
                'context': {
                    'symbol': symbol,
                    'regime': regime,
                    'pattern': pattern,
                    'entry_price': trade.get('entry_price', 0),
                    'stop_loss': trade.get('stop_loss', 0),
                    'target': trade.get('target', 0),
                },
                'outcome': {
                    'pnl': pnl,
                    'won': won,
                    'exit_reason': trade.get('exit_reason', 'unknown'),
                    'duration': trade.get('duration_bars', 0),
                }
            }

            # Store the episode
            episodic.store(
                context=f"Trade {symbol} using {pattern} in {regime} regime",
                reasoning=f"Entry based on {pattern} signal",
                outcome="win" if won else "loss",
                metadata=episode,
            )

            results["episodic_stored"] = True
            logger.info(f"Episodic memory stored for {symbol}")

        except ImportError:
            results["errors"].append("EpisodicMemory not available")
            logger.debug("EpisodicMemory not available for trade recording")
        except Exception as e:
            results["errors"].append(f"Episodic error: {str(e)}")
            logger.warning(f"Failed to store episodic memory: {e}")

        # === Update Semantic Memory (Pattern Reinforcement) ===
        try:
            from cognitive.semantic_memory import SemanticMemory

            semantic = SemanticMemory()

            if won:
                # Reinforce successful patterns
                semantic.reinforce_pattern(pattern)

                # Also reinforce regime-specific pattern
                regime_pattern = f"{pattern}_{regime}"
                semantic.reinforce_pattern(regime_pattern)

                logger.info(f"Semantic memory: Reinforced pattern '{pattern}' (win)")
            else:
                # Weaken failed patterns
                semantic.weaken_pattern(pattern)

                # Also weaken regime-specific pattern
                regime_pattern = f"{pattern}_{regime}"
                semantic.weaken_pattern(regime_pattern)

                logger.info(f"Semantic memory: Weakened pattern '{pattern}' (loss)")

            results["semantic_updated"] = True
            results["pattern_action"] = "reinforced" if won else "weakened"

        except ImportError:
            results["errors"].append("SemanticMemory not available")
            logger.debug("SemanticMemory not available for pattern update")
        except AttributeError as e:
            # Handle case where reinforce_pattern/weaken_pattern don't exist
            results["errors"].append(f"Semantic method missing: {str(e)}")
            logger.warning(f"SemanticMemory missing method: {e}")
        except Exception as e:
            results["errors"].append(f"Semantic error: {str(e)}")
            logger.warning(f"Failed to update semantic memory: {e}")

        # Log the result
        jlog('trade_experience_recorded',
             symbol=symbol,
             won=won,
             pattern=pattern,
             episodic_stored=results["episodic_stored"],
             semantic_updated=results["semantic_updated"])

        return results

    def batch_record_experiences(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Record multiple trade experiences in batch.

        Args:
            trades: List of trade dicts

        Returns:
            Summary of batch recording results
        """
        from core.structured_log import jlog

        total = len(trades)
        episodic_success = 0
        semantic_success = 0
        errors = []

        for trade in trades:
            result = self.record_trade_experience(trade)
            if result["episodic_stored"]:
                episodic_success += 1
            if result["semantic_updated"]:
                semantic_success += 1
            errors.extend(result.get("errors", []))

        summary = {
            "total_trades": total,
            "episodic_stored": episodic_success,
            "semantic_updated": semantic_success,
            "success_rate": episodic_success / total if total > 0 else 0,
            "errors": errors[:10],  # Cap errors
        }

        jlog('batch_experience_recorded',
             total=total,
             episodic=episodic_success,
             semantic=semantic_success)

        return summary

    def daily_reflection(self) -> Dict[str, Any]:
        """Generate daily performance reflection."""
        logger.info("Generating daily reflection...")

        today = datetime.now(ET).date()

        try:
            # Load today's trades
            trade_log = Path("logs/trades.jsonl")
            if not trade_log.exists():
                return self._generate_empty_reflection(today)

            trades = []
            with open(trade_log) as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        t_date = datetime.fromisoformat(t.get("timestamp", "2000-01-01")).date()
                        if t_date == today:
                            trades.append(t)
                    except (json.JSONDecodeError, ValueError):
                        continue

            if not trades:
                return self._generate_empty_reflection(today)

            # Calculate stats
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
            total_pnl = sum(t.get("pnl", 0) for t in trades)

            # Find best and worst
            sorted_trades = sorted(trades, key=lambda t: t.get("pnl", 0))
            worst_trade = sorted_trades[0] if sorted_trades else None
            best_trade = sorted_trades[-1] if sorted_trades else None

            # Determine mood
            win_rate = wins / len(trades) if trades else 0
            if win_rate >= 0.6 and total_pnl > 0:
                mood = "confident"
            elif win_rate >= 0.5:
                mood = "learning"
            elif win_rate >= 0.4:
                mood = "cautious"
            else:
                mood = "struggling"

            # Generate lessons
            lessons = []
            if wins > losses:
                lessons.append("Entry signals were generally good today")
            if wins < losses:
                lessons.append("Need to be more selective with entries")
            if total_pnl < 0:
                lessons.append("Cut losses faster, let winners run longer")

            # Tomorrow's focus
            focus = []
            if mood in ("struggling", "cautious"):
                focus.append("Be more selective - only take A+ setups")
                focus.append("Reduce position sizes until confidence returns")
            else:
                focus.append("Continue with current approach")
                focus.append("Look for opportunities in validated patterns")

            reflection = DailyReflection(
                date=datetime.now(ET),
                total_trades=len(trades),
                wins=wins,
                losses=losses,
                total_pnl=total_pnl,
                best_trade=best_trade.get("symbol") if best_trade else None,
                worst_trade=worst_trade.get("symbol") if worst_trade else None,
                lessons_learned=lessons,
                tomorrow_focus=focus,
                mood=mood,
            )

            self.reflections.append(reflection)
            self.save_state()

            # Save reflection to file
            reflection_file = self.state_dir / f"reflection_{today.isoformat()}.json"
            reflection_file.write_text(json.dumps({
                "date": today.isoformat(),
                "total_trades": reflection.total_trades,
                "wins": reflection.wins,
                "losses": reflection.losses,
                "win_rate": f"{win_rate:.1%}",
                "total_pnl": f"${reflection.total_pnl:.2f}",
                "best_trade": reflection.best_trade,
                "worst_trade": reflection.worst_trade,
                "lessons": reflection.lessons_learned,
                "tomorrow_focus": reflection.tomorrow_focus,
                "mood": reflection.mood,
            }, indent=2))

            return {
                "status": "success",
                "date": today.isoformat(),
                "trades": len(trades),
                "win_rate": win_rate,
                "pnl": total_pnl,
                "mood": mood,
                "lessons": lessons,
                "focus": focus,
            }

        except Exception as e:
            logger.error(f"Daily reflection failed: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_empty_reflection(self, date) -> Dict[str, Any]:
        """Generate reflection for days with no trades."""
        return {
            "status": "success",
            "date": date.isoformat(),
            "trades": 0,
            "message": "No trades today - capital preservation",
            "lessons": [
                "No signals met quality criteria",
                "Staying patient is part of the edge",
            ],
            "focus": [
                "Continue monitoring for quality setups",
                "Review watchlist for next day",
            ],
            "mood": "patient",
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        return {
            "total_lessons": len(self.lessons),
            "total_reflections": len(self.reflections),
            "recent_mood": self.reflections[-1].mood if self.reflections else "unknown",
            "win_lessons": sum(1 for l in self.lessons if l.outcome == "win"),
            "loss_lessons": sum(1 for l in self.lessons if l.outcome == "loss"),
        }


# Task handlers
def analyze_trades() -> Dict[str, Any]:
    """
    Task handler for trade analysis.

    === TIER 2.3: Now also records to Episodic/Semantic Memory ===
    """
    engine = LearningEngine()
    result = engine.analyze_trades()

    # Wire memory recording for analyzed trades
    if result.get("status") == "success" and result.get("trades_data"):
        trades_data = result.get("trades_data", [])
        if trades_data:
            memory_result = engine.batch_record_experiences(trades_data)
            result["memory_update"] = memory_result

    return result


def update_memory() -> Dict[str, Any]:
    """Task handler for memory update."""
    engine = LearningEngine()
    return engine.update_memory()


def daily_reflection() -> Dict[str, Any]:
    """Task handler for daily reflection."""
    engine = LearningEngine()
    return engine.daily_reflection()


def record_trade_experience(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task handler for recording a single trade to memory.

    === TIER 2.3: Episodic/Semantic Memory Updates ===

    Args:
        trade: Dict with symbol, pnl, reason, regime, etc.

    Returns:
        Dict with memory update results
    """
    engine = LearningEngine()
    return engine.record_trade_experience(trade)


def batch_record_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Task handler for batch recording trades to memory.

    Args:
        trades: List of trade dicts

    Returns:
        Summary of batch recording
    """
    engine = LearningEngine()
    return engine.batch_record_experiences(trades)


if __name__ == "__main__":
    # Demo
    engine = LearningEngine()

    print("Learning Engine Demo")
    print("=" * 50)

    result = engine.daily_reflection()
    print("\nDaily Reflection:")
    print(f"  Trades: {result.get('trades', 0)}")
    print(f"  Mood: {result.get('mood', 'unknown')}")
    print(f"  Lessons: {result.get('lessons', [])}")
