"""
Global Workspace - Shared Blackboard Bus
==========================================

Brain-inspired Global Workspace Theory implementation.

All cognitive modules publish/subscribe to this shared space.
Information that enters the workspace becomes "globally available"
to all processors simultaneously.

Features:
- Publish/subscribe pattern for modules
- Priority-based attention mechanism
- Working memory with decay
- Context management
- Event broadcasting

Based on: Baars' Global Workspace Theory
Paper: Nature (PMC) - Brain-like integration architectures

Usage:
    from cognitive.global_workspace import get_workspace

    workspace = get_workspace()

    # Publish information
    workspace.publish('regime', {'state': 'BULL', 'confidence': 0.85})

    # Subscribe to updates
    workspace.subscribe('regime', callback_function)

    # Get current context
    context = workspace.get_context()
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for workspace items."""
    CRITICAL = 1    # Immediate attention (kill switch, risk breach)
    HIGH = 2        # Important (new signal, regime change)
    NORMAL = 3      # Standard updates
    LOW = 4         # Background info
    BACKGROUND = 5  # Passive updates


@dataclass
class WorkspaceItem:
    """Item stored in the global workspace."""
    topic: str
    data: Any
    priority: Priority
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # Time to live
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if item has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'data': self.data,
            'priority': self.priority.name,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata,
        }


@dataclass
class WorkingMemorySlot:
    """Slot in working memory with attention weight."""
    item: WorkspaceItem
    attention_weight: float  # 0-1, decays over time
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class GlobalWorkspace:
    """
    Shared blackboard for all cognitive modules.

    Implements Global Workspace Theory:
    - Information becomes globally available when published
    - Attention mechanism determines what gets processed
    - Working memory has limited capacity with decay
    """

    # Standard topics
    TOPIC_REGIME = 'regime'
    TOPIC_SIGNAL = 'signal'
    TOPIC_RISK = 'risk'
    TOPIC_POSITION = 'position'
    TOPIC_CONFIDENCE = 'confidence'
    TOPIC_REFLECTION = 'reflection'
    TOPIC_INSIGHT = 'insight'
    TOPIC_HYPOTHESIS = 'hypothesis'
    TOPIC_ERROR = 'error'
    TOPIC_ALERT = 'alert'
    TOPIC_SELF_STATE = 'self_state'
    TOPIC_DECISION = 'decision'

    def __init__(
        self,
        working_memory_capacity: int = 7,  # Miller's 7+/-2
        attention_decay_rate: float = 0.1,  # Per minute
        cleanup_interval: int = 60,  # Seconds
    ):
        self.working_memory_capacity = working_memory_capacity
        self.attention_decay_rate = attention_decay_rate
        self.cleanup_interval = cleanup_interval

        # Storage
        self._workspace: Dict[str, WorkspaceItem] = {}
        self._working_memory: List[WorkingMemorySlot] = []
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[WorkspaceItem] = []
        self._history_limit = 1000

        # Thread safety
        self._lock = threading.RLock()

        # Stats
        self._publish_count = 0
        self._broadcast_count = 0

        logger.info("GlobalWorkspace initialized")

    def publish(
        self,
        topic: str,
        data: Any,
        priority: Priority = Priority.NORMAL,
        source: str = "unknown",
        ttl_seconds: int = 3600,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Publish information to the global workspace.

        Args:
            topic: Topic/channel name
            data: The information to publish
            priority: Priority level
            source: Module that published this
            ttl_seconds: Time to live
            metadata: Additional metadata
        """
        with self._lock:
            item = WorkspaceItem(
                topic=topic,
                data=data,
                priority=priority,
                source=source,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
            )

            # Store in workspace
            self._workspace[topic] = item
            self._publish_count += 1

            # Add to working memory if high priority
            if priority.value <= Priority.HIGH.value:
                self._add_to_working_memory(item)

            # Add to history
            self._history.append(item)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit:]

            # Broadcast to subscribers
            self._broadcast(topic, item)

            logger.debug(f"Published to '{topic}' from {source}: {type(data).__name__}")

    def subscribe(self, topic: str, callback: Callable[[WorkspaceItem], None]) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Topic to subscribe to ('*' for all)
            callback: Function called when topic is updated
        """
        with self._lock:
            self._subscribers[topic].append(callback)
            logger.debug(f"Subscriber added for topic '{topic}'")

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe from a topic."""
        with self._lock:
            if callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)

    def get(self, topic: str, default: Any = None) -> Any:
        """
        Get current value for a topic.

        Args:
            topic: Topic to retrieve

        Returns:
            The data, or default if not found/expired
        """
        with self._lock:
            item = self._workspace.get(topic)
            if item is None or item.is_expired():
                return default
            return item.data

    def get_item(self, topic: str) -> Optional[WorkspaceItem]:
        """Get full workspace item for a topic."""
        with self._lock:
            item = self._workspace.get(topic)
            if item and not item.is_expired():
                return item
            return None

    def get_context(self) -> Dict[str, Any]:
        """
        Get current context from all workspace items.

        Returns:
            Dict of topic -> data for all non-expired items
        """
        with self._lock:
            context = {}
            for topic, item in self._workspace.items():
                if not item.is_expired():
                    context[topic] = {
                        'data': item.data,
                        'priority': item.priority.name,
                        'source': item.source,
                        'age_seconds': (datetime.now() - item.timestamp).total_seconds(),
                    }
            return context

    def get_working_memory(self) -> List[Dict[str, Any]]:
        """
        Get current working memory contents.

        Returns:
            List of items in working memory with attention weights
        """
        with self._lock:
            self._decay_attention()
            return [
                {
                    'topic': slot.item.topic,
                    'data': slot.item.data,
                    'attention_weight': slot.attention_weight,
                    'access_count': slot.access_count,
                }
                for slot in sorted(self._working_memory,
                                   key=lambda s: s.attention_weight,
                                   reverse=True)
            ]

    def focus_attention(self, topic: str) -> Optional[Any]:
        """
        Focus attention on a topic, boosting its working memory weight.

        Args:
            topic: Topic to focus on

        Returns:
            The data for that topic
        """
        with self._lock:
            # Boost attention weight
            for slot in self._working_memory:
                if slot.item.topic == topic:
                    slot.attention_weight = min(1.0, slot.attention_weight + 0.3)
                    slot.access_count += 1
                    slot.last_accessed = datetime.now()
                    return slot.item.data

            # Not in working memory - try to add it
            item = self._workspace.get(topic)
            if item and not item.is_expired():
                self._add_to_working_memory(item)
                return item.data

            return None

    def broadcast_alert(self, message: str, priority: Priority = Priority.HIGH) -> None:
        """Broadcast an alert to all modules."""
        self.publish(
            topic=self.TOPIC_ALERT,
            data={'message': message, 'timestamp': datetime.now().isoformat()},
            priority=priority,
            source='workspace',
        )

    def _add_to_working_memory(self, item: WorkspaceItem) -> None:
        """Add item to working memory, evicting if necessary."""
        # Check if already in working memory
        for slot in self._working_memory:
            if slot.item.topic == item.topic:
                slot.item = item
                slot.attention_weight = 1.0
                slot.last_accessed = datetime.now()
                return

        # Create new slot
        slot = WorkingMemorySlot(item=item, attention_weight=1.0)

        # Evict if at capacity (lowest attention weight)
        if len(self._working_memory) >= self.working_memory_capacity:
            self._working_memory.sort(key=lambda s: s.attention_weight)
            self._working_memory.pop(0)

        self._working_memory.append(slot)

    def _decay_attention(self) -> None:
        """Apply attention decay to working memory items."""
        now = datetime.now()
        for slot in self._working_memory:
            elapsed_minutes = (now - slot.last_accessed).total_seconds() / 60
            decay = self.attention_decay_rate * elapsed_minutes
            slot.attention_weight = max(0, slot.attention_weight - decay)

        # Remove items with zero attention
        self._working_memory = [s for s in self._working_memory if s.attention_weight > 0]

    def _broadcast(self, topic: str, item: WorkspaceItem) -> None:
        """Broadcast update to subscribers."""
        # Topic-specific subscribers
        for callback in self._subscribers.get(topic, []):
            try:
                callback(item)
            except Exception as e:
                logger.warning(f"Subscriber callback failed for {topic}: {e}")

        # Wildcard subscribers
        for callback in self._subscribers.get('*', []):
            try:
                callback(item)
            except Exception as e:
                logger.warning(f"Wildcard subscriber callback failed: {e}")

        self._broadcast_count += 1

    def cleanup(self) -> int:
        """Remove expired items from workspace."""
        with self._lock:
            expired_count = 0
            expired_topics = []

            for topic, item in self._workspace.items():
                if item.is_expired():
                    expired_topics.append(topic)
                    expired_count += 1

            for topic in expired_topics:
                del self._workspace[topic]

            return expired_count

    def get_history(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get history of published items.

        Args:
            topic: Filter by topic (None for all)
            limit: Maximum items to return
            since: Filter by timestamp

        Returns:
            List of historical items
        """
        with self._lock:
            history = self._history

            if topic:
                history = [h for h in history if h.topic == topic]

            if since:
                history = [h for h in history if h.timestamp >= since]

            return [h.to_dict() for h in history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        with self._lock:
            return {
                'topics_active': len(self._workspace),
                'working_memory_used': len(self._working_memory),
                'working_memory_capacity': self.working_memory_capacity,
                'total_publishes': self._publish_count,
                'total_broadcasts': self._broadcast_count,
                'subscriber_count': sum(len(subs) for subs in self._subscribers.values()),
                'history_size': len(self._history),
            }

    def dump_state(self) -> str:
        """Dump current workspace state as JSON."""
        with self._lock:
            state = {
                'workspace': {k: v.to_dict() for k, v in self._workspace.items()
                              if not v.is_expired()},
                'working_memory': self.get_working_memory(),
                'stats': self.get_stats(),
            }
            return json.dumps(state, indent=2, default=str)


# Singleton instance
_global_workspace: Optional[GlobalWorkspace] = None


def get_workspace() -> GlobalWorkspace:
    """Get or create the global workspace singleton."""
    global _global_workspace
    if _global_workspace is None:
        _global_workspace = GlobalWorkspace()
    return _global_workspace
