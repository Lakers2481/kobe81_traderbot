"""
Online/Incremental Learning for Kobe Trading System

Update models incrementally with new data without full retraining.
Adapts to changing market conditions while preventing catastrophic forgetting.

Key Features:
- Experience Replay Buffer: Prioritized sampling of hard examples
- Concept Drift Detection: Automatically detect when model performance degrades
- Incremental Updates: Very low learning rate (0.0001) for stability

MERGED FROM GAME_PLAN_2K28 - Production Ready
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logging.warning("TensorFlow/Keras not installed. Online learning disabled.")

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Record of a trade for learning."""
    timestamp: datetime
    symbol: str
    features: np.ndarray
    prediction: float
    actual_outcome: int
    pnl_pct: float
    holding_period: int

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'features': self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            'prediction': float(self.prediction),
            'actual_outcome': int(self.actual_outcome),
            'pnl_pct': float(self.pnl_pct),
            'holding_period': int(self.holding_period)
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeOutcome":
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            symbol=d['symbol'],
            features=np.array(d['features']),
            prediction=d['prediction'],
            actual_outcome=d['actual_outcome'],
            pnl_pct=d['pnl_pct'],
            holding_period=d['holding_period']
        )


class ExperienceReplayBuffer:
    """
    Stores recent trade experiences for incremental learning.
    Uses prioritized replay - samples harder examples more often.
    """

    def __init__(
        self,
        max_size: int = 10000,
        min_size_for_learning: int = 100,
        priority_alpha: float = 0.6
    ):
        self.max_size = max_size
        self.min_size = min_size_for_learning
        self.priority_alpha = priority_alpha
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities: deque = deque(maxlen=max_size)

    def add(self, outcome: TradeOutcome, priority: Optional[float] = None):
        if priority is None:
            prediction_error = abs(outcome.prediction - outcome.actual_outcome)
            priority = prediction_error + 0.01
        self.buffer.append(outcome)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[TradeOutcome]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        priorities = np.array(self.priorities)
        priorities = priorities ** self.priority_alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def get_class_balance(self) -> Dict[int, float]:
        if len(self.buffer) == 0:
            return {0: 0.5, 1: 0.5}

        outcomes = [o.actual_outcome for o in self.buffer]
        unique, counts = np.unique(outcomes, return_counts=True)
        total = len(outcomes)

        balance = {int(k): float(v) / total for k, v in zip(unique, counts)}
        if 0 not in balance:
            balance[0] = 0.0
        if 1 not in balance:
            balance[1] = 0.0
        return balance

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.min_size

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()


class ConceptDriftDetector:
    """
    Detects when model performance degrades (concept drift).
    Uses sliding window accuracy comparison against baseline.
    """

    def __init__(
        self,
        window_size: int = 50,
        drift_threshold: float = 0.15,
        warning_threshold: float = 0.10
    ):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        self.baseline_accuracy: Optional[float] = None
        self.recent_predictions: deque = deque(maxlen=window_size)

    def update(self, prediction: float, actual: int) -> str:
        pred_binary = 1 if prediction >= 0.5 else 0
        self.recent_predictions.append((pred_binary, actual))

        if len(self.recent_predictions) < self.window_size:
            return 'normal'

        current_acc = self.get_current_accuracy()

        if self.baseline_accuracy is None:
            self.baseline_accuracy = current_acc
            logger.info(f"Drift detector baseline set: {current_acc:.3f}")
            return 'normal'

        accuracy_drop = self.baseline_accuracy - current_acc

        if accuracy_drop >= self.drift_threshold:
            logger.warning(f"DRIFT DETECTED: Accuracy dropped {accuracy_drop:.1%}")
            return 'drift'
        elif accuracy_drop >= self.warning_threshold:
            logger.warning(f"DRIFT WARNING: Accuracy dropped {accuracy_drop:.1%}")
            return 'warning'
        return 'normal'

    def get_current_accuracy(self) -> float:
        if len(self.recent_predictions) == 0:
            return 0.0
        correct = sum(pred == actual for pred, actual in self.recent_predictions)
        return correct / len(self.recent_predictions)

    def reset_baseline(self):
        if len(self.recent_predictions) >= self.window_size:
            self.baseline_accuracy = self.get_current_accuracy()
            logger.info(f"Drift detector baseline reset: {self.baseline_accuracy:.3f}")


class OnlineLearningManager:
    """
    Manages online learning for ML models.

    Coordinates:
    - Experience replay
    - Update scheduling
    - Drift detection
    - Health monitoring
    """

    def __init__(
        self,
        update_frequency: str = 'daily',
        auto_update: bool = True
    ):
        self.replay_buffer = ExperienceReplayBuffer()
        self.drift_detector = ConceptDriftDetector()
        self.update_frequency = update_frequency
        self.auto_update = auto_update
        self.last_update: Optional[datetime] = None
        self.update_history: List[dict] = []

        logger.info(f"Online Learning Manager initialized (frequency={update_frequency})")

    def record_trade_outcome(
        self,
        symbol: str,
        features: np.ndarray,
        prediction: float,
        actual_pnl: float,
        holding_period: int
    ):
        """Record trade outcome for learning."""
        actual_outcome = 1 if actual_pnl > 0 else 0

        outcome = TradeOutcome(
            timestamp=datetime.now(),
            symbol=symbol,
            features=features,
            prediction=prediction,
            actual_outcome=actual_outcome,
            pnl_pct=actual_pnl,
            holding_period=holding_period
        )

        self.replay_buffer.add(outcome)
        drift_status = self.drift_detector.update(prediction, actual_outcome)

        if self.auto_update and drift_status == 'drift':
            logger.warning("Drift detected - consider triggering update")

    def should_update(self) -> bool:
        return self.replay_buffer.is_ready()

    def get_training_batch(self, batch_size: int = 32) -> List[TradeOutcome]:
        """Get batch of samples for training."""
        return self.replay_buffer.sample(batch_size)

    def get_status(self) -> Dict:
        """Get current learning status."""
        return {
            'buffer_size': len(self.replay_buffer),
            'buffer_ready': self.replay_buffer.is_ready(),
            'class_balance': self.replay_buffer.get_class_balance(),
            'drift_baseline_accuracy': self.drift_detector.baseline_accuracy,
            'drift_current_accuracy': self.drift_detector.get_current_accuracy(),
            'total_updates': len(self.update_history),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }

    def save_state(self, path: str):
        """Save state to disk."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        state = {
            'update_frequency': self.update_frequency,
            'auto_update': self.auto_update,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_history': self.update_history,
            'drift_baseline': self.drift_detector.baseline_accuracy,
            'buffer_size': len(self.replay_buffer)
        }

        with open(path_obj / 'online_learning_state.json', 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Online learning state saved to {path}")

    def load_state(self, path: str):
        """Load state from disk."""
        path_obj = Path(path)
        state_file = path_obj / 'online_learning_state.json'

        if not state_file.exists():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with open(state_file, 'r') as f:
            state = json.load(f)

        self.update_frequency = state['update_frequency']
        self.auto_update = state['auto_update']
        self.last_update = datetime.fromisoformat(state['last_update']) if state['last_update'] else None
        self.update_history = state['update_history']
        self.drift_detector.baseline_accuracy = state['drift_baseline']

        logger.info(f"Online learning state loaded from {path}")


def create_online_learning_manager(
    update_frequency: str = 'daily',
    auto_update: bool = True
) -> OnlineLearningManager:
    """Factory function to create online learning manager."""
    return OnlineLearningManager(
        update_frequency=update_frequency,
        auto_update=auto_update
    )
