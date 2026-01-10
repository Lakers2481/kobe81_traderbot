"""
Global Workspace - The AI's Shared Consciousness
===================================================

This module implements a "Global Workspace," a central information bus inspired
by the Global Workspace Theory from cognitive science (Bernard Baars). It acts
as a shared blackboard where all independent cognitive modules can publish
information, making it "globally available" to every other module.

Core Concepts:
- **Shared Blackboard:** Any module can write (`publish`) information to a named
  `topic` on the workspace.
- **Global Availability:** Other modules can read (`get`) the current value of
  any topic or `subscribe` to be notified of updates in real-time.
- **Working Memory:** The workspace maintains a small, limited-capacity
  "working memory" (inspired by Miller's 7+/-2 law) that holds the most
  important, high-priority, or recently accessed information.
- **Attention Mechanism:** Information enters working memory based on its
  `priority`. The "attention" on items in working memory decays over time,
  ensuring that only currently relevant information remains in focus.

This architecture allows for flexible and decoupled communication between the
various components of the cognitive system, enabling complex, emergent behavior.

Based on: Baars' Global Workspace Theory
Paper: A Cognitive Theory of Consciousness (Baars, 1988)

Usage:
    from cognitive.global_workspace import get_workspace

    # The workspace is a singleton, ensuring one shared space for the whole app.
    workspace = get_workspace()

    # A market regime module publishes the current market state.
    workspace.publish(
        topic='regime',
        data={'state': 'BULL', 'confidence': 0.85},
        source='RegimeDetector'
    )

    # A decision-making module can subscribe to regime changes.
    def on_regime_change(item):
        print(f"New regime detected: {item.data['state']}")

    workspace.subscribe('regime', on_regime_change)

    # Another module can simply get the current context at any time.
    current_context = workspace.get_context()
    print(current_context['regime'])
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Defines the importance of information published to the workspace."""
    CRITICAL = 1    # Immediate, system-critical attention (e.g., kill switch activated).
    HIGH = 2        # Important, requires timely processing (e.g., new trade signal).
    NORMAL = 3      # Standard informational updates (e.g., portfolio value).
    LOW = 4         # Background information (e.g., periodic self-check status).
    BACKGROUND = 5  # Passive data that doesn't require immediate attention.


@dataclass
class WorkspaceItem:
    """A structured piece of information stored in the global workspace."""
    topic: str
    data: Any
    priority: Priority
    source: str  # The name of the module that published this item.
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # Default time-to-live: 1 hour.

    def is_expired(self) -> bool:
        """Checks if the item has passed its time-to-live."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the item to a dictionary."""
        return {
            'topic': self.topic,
            'data': self.data,
            'priority': self.priority.name,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'ttl_seconds': self.ttl_seconds,
        }


@dataclass
class WorkingMemorySlot:
    """
    Represents an item currently held in the limited-capacity working memory.
    It includes an attention weight that determines its importance and lifespan.
    """
    item: WorkspaceItem
    attention_weight: float  # Score from 0.0 to 1.0, decays over time.
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class GlobalWorkspace:
    """
    A thread-safe, publish-subscribe message bus that implements the principles
    of Global Workspace Theory for decoupled inter-module communication.
    """

    # --- Standardized Topic Names ---
    # Using consistent topic names is good practice.
    TOPIC_REGIME = 'regime'
    TOPIC_SIGNAL = 'signal'
    TOPIC_RISK = 'risk'
    TOPIC_CONFIDENCE = 'confidence'
    TOPIC_INSIGHT = 'insight' # For newly discovered edges or patterns
    TOPIC_ERROR = 'error'
    TOPIC_ALERT = 'alert'
    TOPIC_SELF_STATE = 'self_state' # For metacognitive updates
    TOPIC_DECISION = 'decision'     # For final adjudicated decisions

    def __init__(
        self,
        working_memory_capacity: int = 7,  # Inspired by Miller's "7 +/- 2" law.
        attention_decay_rate: float = 0.1,  # Rate at which attention fades per minute.
    ):
        self.working_memory_capacity = working_memory_capacity
        self.attention_decay_rate = attention_decay_rate

        # --- Core Data Structures ---
        # The main blackboard: stores the latest item for every topic.
        self._workspace: Dict[str, WorkspaceItem] = {}
        # Limited-capacity "short-term" memory for high-priority items.
        self._working_memory: List[WorkingMemorySlot] = []
        # Subscriber registry: maps topics to callback functions.
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        # A rolling history of all published items for later review.
        self._history: List[WorkspaceItem] = []
        self._history_limit = 1000

        self._lock = threading.RLock()  # To ensure thread-safe operations.

        # --- Statistics ---
        self._publish_count = 0
        self._broadcast_count = 0

        logger.info("GlobalWorkspace initialized.")

    def publish(
        self,
        topic: str,
        data: Any,
        priority: Priority = Priority.NORMAL,
        source: str = "unknown",
        ttl_seconds: int = 3600,
    ) -> None:
        """
        Publishes a piece of information to the workspace, making it globally
        available to all other modules.

        Args:
            topic: The channel or "name" of the information (e.g., 'regime').
            data: The actual data payload.
            priority: The importance of this information.
            source: The name of the module publishing the data.
            ttl_seconds: How long this information should be considered valid.
        """
        with self._lock:
            item = WorkspaceItem(
                topic=topic,
                data=data,
                priority=priority,
                source=source,
                ttl_seconds=ttl_seconds,
            )

            # Update the main workspace with the latest item for this topic.
            self._workspace[topic] = item
            self._publish_count += 1

            # If the information is high priority, bring it into the "focus" of working memory.
            if priority.value <= Priority.HIGH.value:
                self._add_to_working_memory(item)

            # Record the event in the historical log.
            self._history.append(item)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

            # Notify all subscribers listening to this topic.
            self._broadcast(topic, item)

            logger.debug(f"Published to '{topic}' from {source}.")

    def subscribe(self, topic: str, callback: Callable[[WorkspaceItem], None]) -> None:
        """
        Subscribes a callback function to a topic. The callback will be executed
        whenever new information is published to that topic.

        Args:
            topic: The topic to listen to. Use '*' to subscribe to all topics.
            callback: The function to call with the `WorkspaceItem` as an argument.
        """
        with self._lock:
            self._subscribers[topic].append(callback)
            logger.debug(f"New subscriber added for topic '{topic}'.")

    def get(self, topic: str, default: Any = None) -> Any:
        """
        Retrieves the last published data for a given topic.

        Returns:
            The data payload, or a default value if the topic doesn't exist or
            the last item has expired.
        """
        with self._lock:
            item = self._workspace.get(topic)
            if item and not item.is_expired():
                return item.data
            return default

    def get_item(self, topic: str) -> Optional[WorkspaceItem]:
        """Retrieves the full WorkspaceItem for a topic, including metadata."""
        with self._lock:
            item = self._workspace.get(topic)
            if item and not item.is_expired():
                return item
            return None

    def get_context(self) -> Dict[str, Any]:
        """
        Returns a snapshot of the entire current state of the workspace.
        This provides a complete contextual picture for decision-making modules.
        """
        with self._lock:
            context = {}
            for topic, item in self._workspace.items():
                if not item.is_expired():
                    context[topic] = item.data
            return context

    def get_working_memory(self) -> List[Dict[str, Any]]:
        """
        Returns the contents of the limited-capacity working memory, representing
        the current "focus of attention" of the AI.
        """
        with self._lock:
            self._decay_attention()  # Apply decay before returning.
            return [
                {
                    'topic': slot.item.topic,
                    'data': slot.item.data,
                    'attention_weight': round(slot.attention_weight, 3),
                    'access_count': slot.access_count,
                }
                for slot in sorted(self._working_memory, key=lambda s: s.attention_weight, reverse=True)
            ]

    def focus_attention(self, topic: str) -> Optional[Any]:
        """
        Explicitly focuses attention on a topic, boosting its weight in working
        memory and preventing it from decaying.
        """
        with self._lock:
            # Find the item in working memory and boost its attention.
            for slot in self._working_memory:
                if slot.item.topic == topic:
                    slot.attention_weight = min(1.0, slot.attention_weight + 0.5) # Significant boost
                    slot.access_count += 1
                    slot.last_accessed = datetime.now()
                    return slot.item.data

            # If not in working memory, try to add it.
            if item := self.get_item(topic):
                self._add_to_working_memory(item)
                return item.data
            return None

    def _add_to_working_memory(self, item: WorkspaceItem) -> None:
        """Adds an item to working memory, evicting the least important item if full."""
        with self._lock:
            # If item is already present, just refresh its attention.
            for slot in self._working_memory:
                if slot.item.topic == item.topic:
                    slot.item = item
                    slot.attention_weight = 1.0  # Reset attention to max
                    slot.last_accessed = datetime.now()
                    return

            # If working memory is full, evict the item with the lowest attention weight.
            if len(self._working_memory) >= self.working_memory_capacity:
                self._working_memory.sort(key=lambda s: s.attention_weight)
                evicted = self._working_memory.pop(0)
                logger.debug(f"Evicted '{evicted.item.topic}' from working memory.")

            # Add the new item with full attention.
            new_slot = WorkingMemorySlot(item=item, attention_weight=1.0)
            self._working_memory.append(new_slot)

    def _decay_attention(self) -> None:
        """Periodically reduces the attention weight of items in working memory."""
        with self._lock:
            now = datetime.now()
            for slot in self._working_memory:
                elapsed_minutes = (now - slot.last_accessed).total_seconds() / 60.0
                decay = self.attention_decay_rate * elapsed_minutes
                slot.attention_weight = max(0, slot.attention_weight - decay)

            # Remove items that have completely lost attention.
            self._working_memory = [s for s in self._working_memory if s.attention_weight > 0.01]

    def _broadcast(self, topic: str, item: WorkspaceItem) -> None:
        """Notifies all relevant subscribers about a new item."""
        with self._lock:
            callbacks = self._subscribers.get(topic, []) + self._subscribers.get('*', [])
            for callback in callbacks:
                try:
                    # In a more robust system, this might be done in a separate thread
                    # to avoid blocking the publish call.
                    callback(item)
                except Exception as e:
                    logger.error(f"Subscriber callback for topic '{topic}' failed: {e}", exc_info=True)
            self._broadcast_count += 1

    def cleanup(self) -> int:
        """Removes all expired items from the main workspace."""
        with self._lock:
            expired_topics = [
                topic for topic, item in self._workspace.items() if item.is_expired()
            ]
            for topic in expired_topics:
                del self._workspace[topic]
            if expired_topics:
                logger.info(f"Cleaned up {len(expired_topics)} expired workspace items.")
            return len(expired_topics)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the workspace's current state and usage."""
        with self._lock:
            return {
                'active_topics': len(self._workspace),
                'working_memory_size': len(self._working_memory),
                'working_memory_capacity': self.working_memory_capacity,
                'total_publishes': self._publish_count,
                'total_broadcasts': self._broadcast_count,
                'total_subscribers': sum(len(s) for s in self._subscribers.values()),
                'history_log_size': len(self._history),
            }

    def dump_state(self) -> str:
        """Returns a JSON string representing the full current state of the workspace."""
        with self._lock:
            state = {
                'workspace': {k: v.to_dict() for k, v in self._workspace.items() if not v.is_expired()},
                'working_memory': self.get_working_memory(),
                'stats': self.get_stats(),
            }
            return json.dumps(state, indent=2)

# --- Singleton Implementation ---
# There should only ever be one Global Workspace for the entire application.
_global_workspace: Optional[GlobalWorkspace] = None
_lock = threading.Lock()

def get_workspace() -> GlobalWorkspace:
    """Factory function to get the singleton instance of the GlobalWorkspace."""
    global _global_workspace
    if _global_workspace is None:
        with _lock:
            if _global_workspace is None:
                _global_workspace = GlobalWorkspace()
    return _global_workspace
