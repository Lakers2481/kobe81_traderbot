"""
Redis Streams Pub/Sub for Real-Time Event Propagation

Production-grade event bus using Redis Streams for:
- Low-latency message delivery (<1ms)
- Persistent message storage with replay capability
- Consumer groups for load balancing
- Automatic reconnection and failover

Usage:
    from messaging import get_event_bus, EventType, publish_event

    # Publish quote
    publish_event(EventType.QUOTE, {"symbol": "AAPL", "price": 150.00})

    # Subscribe to signals
    for event in subscribe_to_events(EventType.SIGNAL):
        process_signal(event)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Union

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for pub/sub messaging."""
    # Market data events
    QUOTE = "quote"              # Real-time price quote
    BAR = "bar"                  # OHLCV bar update
    TRADE = "trade"              # Market trade tick

    # Signal events
    SIGNAL = "signal"            # Trading signal generated
    SIGNAL_VALIDATED = "signal_validated"  # Signal passed validation
    SIGNAL_REJECTED = "signal_rejected"    # Signal rejected

    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"

    # System events
    HEARTBEAT = "heartbeat"
    ALERT = "alert"
    ERROR = "error"
    KILL_SWITCH = "kill_switch"

    # Learning events
    TRADE_OUTCOME = "trade_outcome"
    REFLECTION = "reflection"
    RULE_LEARNED = "rule_learned"


@dataclass
class Event:
    """Standard event structure."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "kobe"
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(**data)


class RedisEventBus:
    """
    Redis Streams-based event bus for real-time messaging.

    Features:
    - Automatic connection management
    - Consumer groups for scalable processing
    - Message persistence and replay
    - Graceful degradation when Redis unavailable
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        stream_prefix: str = "kobe:",
        max_stream_length: int = 10000,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.stream_prefix = stream_prefix
        self.max_stream_length = max_stream_length

        self._redis: Optional[Any] = None
        self._connected = False
        self._fallback_mode = False
        self._fallback_queue: List[Event] = []
        self._lock = threading.Lock()
        self._subscribers: Dict[EventType, List[Callable]] = {}

    def _get_stream_name(self, event_type: EventType) -> str:
        """Get Redis stream name for event type."""
        return f"{self.stream_prefix}{event_type.value}"

    def connect(self) -> bool:
        """Connect to Redis server."""
        try:
            import redis

            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            self._redis.ping()
            self._connected = True
            self._fallback_mode = False
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except ImportError:
            logger.warning("redis-py not installed. Using fallback mode (in-memory queue)")
            self._fallback_mode = True
            return False

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using fallback mode")
            self._fallback_mode = True
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        self._connected = False
        self._redis = None

    def publish(self, event_type: EventType, payload: Dict[str, Any], source: str = "kobe") -> Optional[str]:
        """
        Publish an event to the stream.

        Args:
            event_type: Type of event
            payload: Event data
            source: Source component name

        Returns:
            Event ID if published, None if in fallback mode
        """
        event = Event(
            event_type=event_type.value,
            payload=payload,
            source=source,
        )

        if self._fallback_mode or not self._connected:
            with self._lock:
                self._fallback_queue.append(event)
                # Keep queue bounded
                if len(self._fallback_queue) > self.max_stream_length:
                    self._fallback_queue = self._fallback_queue[-self.max_stream_length:]

            # Trigger local subscribers
            self._notify_subscribers(event_type, event)
            return None

        try:
            stream_name = self._get_stream_name(event_type)

            # Convert event to flat dict for Redis
            flat_data = {
                "event_type": event.event_type,
                "payload": json.dumps(payload),
                "timestamp": event.timestamp,
                "source": source,
            }

            # Add to stream with auto-generated ID
            event_id = self._redis.xadd(
                stream_name,
                flat_data,
                maxlen=self.max_stream_length,
                approximate=True,
            )

            event.event_id = event_id

            # Trigger local subscribers
            self._notify_subscribers(event_type, event)

            return event_id

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            # Fallback to local queue
            with self._lock:
                self._fallback_queue.append(event)
            return None

    def subscribe(
        self,
        event_types: Union[EventType, List[EventType]],
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        block_ms: int = 1000,
        count: int = 10,
    ) -> Generator[Event, None, None]:
        """
        Subscribe to events from streams.

        Args:
            event_types: Event type(s) to subscribe to
            consumer_group: Consumer group name for load balancing
            consumer_name: Consumer name within group
            block_ms: Block timeout in milliseconds
            count: Max events per read

        Yields:
            Event objects
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]

        # In fallback mode, yield from local queue
        if self._fallback_mode or not self._connected:
            yield from self._subscribe_fallback(event_types)
            return

        try:
            streams = {self._get_stream_name(et): ">" if consumer_group else "$"
                       for et in event_types}

            # Create consumer groups if specified
            if consumer_group:
                for stream_name in streams:
                    try:
                        self._redis.xgroup_create(stream_name, consumer_group, mkstream=True)
                    except Exception:
                        pass  # Group may already exist

            while True:
                try:
                    if consumer_group:
                        results = self._redis.xreadgroup(
                            consumer_group,
                            consumer_name or "default",
                            streams,
                            count=count,
                            block=block_ms,
                        )
                    else:
                        results = self._redis.xread(
                            streams,
                            count=count,
                            block=block_ms,
                        )

                    if results:
                        for stream_name, messages in results:
                            for msg_id, msg_data in messages:
                                try:
                                    event = Event(
                                        event_type=msg_data.get("event_type", "unknown"),
                                        payload=json.loads(msg_data.get("payload", "{}")),
                                        timestamp=msg_data.get("timestamp", ""),
                                        source=msg_data.get("source", "unknown"),
                                        event_id=msg_id,
                                    )
                                    yield event

                                    # Acknowledge message if using consumer group
                                    if consumer_group:
                                        self._redis.xack(stream_name, consumer_group, msg_id)

                                except Exception as e:
                                    logger.error(f"Error parsing event: {e}")

                except Exception as e:
                    logger.error(f"Error reading from stream: {e}")
                    time.sleep(1)

        except GeneratorExit:
            pass

    def _subscribe_fallback(self, event_types: List[EventType]) -> Generator[Event, None, None]:
        """Fallback subscription from in-memory queue."""
        type_values = {et.value for et in event_types}
        last_index = 0

        while True:
            with self._lock:
                queue_len = len(self._fallback_queue)

            if last_index < queue_len:
                with self._lock:
                    new_events = self._fallback_queue[last_index:queue_len]

                for event in new_events:
                    if event.event_type in type_values:
                        yield event

                last_index = queue_len
            else:
                time.sleep(0.1)

    def register_callback(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for synchronous event notification."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def _notify_subscribers(self, event_type: EventType, event: Event) -> None:
        """Notify registered callbacks."""
        callbacks = self._subscribers.get(event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_stream_info(self, event_type: EventType) -> Dict[str, Any]:
        """Get information about a stream."""
        if self._fallback_mode or not self._connected:
            return {"mode": "fallback", "queue_size": len(self._fallback_queue)}

        try:
            stream_name = self._get_stream_name(event_type)
            info = self._redis.xinfo_stream(stream_name)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recent_events(
        self,
        event_type: EventType,
        count: int = 10,
    ) -> List[Event]:
        """Get recent events from stream (newest first)."""
        if self._fallback_mode or not self._connected:
            with self._lock:
                type_value = event_type.value
                events = [e for e in self._fallback_queue if e.event_type == type_value]
                return events[-count:][::-1]

        try:
            stream_name = self._get_stream_name(event_type)
            messages = self._redis.xrevrange(stream_name, count=count)

            events = []
            for msg_id, msg_data in messages:
                try:
                    event = Event(
                        event_type=msg_data.get("event_type", "unknown"),
                        payload=json.loads(msg_data.get("payload", "{}")),
                        timestamp=msg_data.get("timestamp", ""),
                        source=msg_data.get("source", "unknown"),
                        event_id=msg_id,
                    )
                    events.append(event)
                except Exception:
                    pass

            return events

        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []


# Global event bus instance
_event_bus: Optional[RedisEventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> RedisEventBus:
    """Get or create the global event bus instance."""
    global _event_bus

    with _event_bus_lock:
        if _event_bus is None:
            _event_bus = RedisEventBus(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=os.getenv("REDIS_PASSWORD"),
            )
            _event_bus.connect()

        return _event_bus


def publish_event(event_type: EventType, payload: Dict[str, Any], source: str = "kobe") -> Optional[str]:
    """Convenience function to publish an event."""
    bus = get_event_bus()
    return bus.publish(event_type, payload, source)


def subscribe_to_events(
    event_types: Union[EventType, List[EventType]],
    consumer_group: Optional[str] = None,
) -> Generator[Event, None, None]:
    """Convenience function to subscribe to events."""
    bus = get_event_bus()
    yield from bus.subscribe(event_types, consumer_group)


# ============================================================================
# Quote Broadcaster Integration
# ============================================================================

class QuoteBroadcaster:
    """
    Broadcasts real-time quotes to all subscribers.

    Integrates with Alpaca WebSocket to distribute quotes
    to strategy, risk, and monitoring components.
    """

    def __init__(self, event_bus: Optional[RedisEventBus] = None):
        self.event_bus = event_bus or get_event_bus()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def broadcast_quote(self, symbol: str, price: float, bid: float, ask: float, timestamp: str) -> None:
        """Broadcast a quote update."""
        self.event_bus.publish(
            EventType.QUOTE,
            {
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "timestamp": timestamp,
            },
            source="quote_broadcaster",
        )

    def broadcast_bar(self, symbol: str, ohlcv: Dict[str, float], timestamp: str) -> None:
        """Broadcast a bar update."""
        self.event_bus.publish(
            EventType.BAR,
            {
                "symbol": symbol,
                **ohlcv,
                "timestamp": timestamp,
            },
            source="quote_broadcaster",
        )


class SignalDistributor:
    """
    Distributes trading signals to execution and risk components.

    Ensures all components receive signals for:
    - Order execution
    - Risk validation
    - Position tracking
    - Audit logging
    """

    def __init__(self, event_bus: Optional[RedisEventBus] = None):
        self.event_bus = event_bus or get_event_bus()

    def distribute_signal(self, signal: Dict[str, Any]) -> None:
        """Distribute a new trading signal."""
        self.event_bus.publish(
            EventType.SIGNAL,
            signal,
            source="signal_generator",
        )

    def distribute_validated_signal(self, signal: Dict[str, Any], validation: Dict[str, Any]) -> None:
        """Distribute a validated signal."""
        self.event_bus.publish(
            EventType.SIGNAL_VALIDATED,
            {"signal": signal, "validation": validation},
            source="signal_validator",
        )

    def distribute_rejected_signal(self, signal: Dict[str, Any], reason: str) -> None:
        """Distribute a rejected signal."""
        self.event_bus.publish(
            EventType.SIGNAL_REJECTED,
            {"signal": signal, "reason": reason},
            source="signal_validator",
        )


class LearningEventPublisher:
    """
    Publishes learning-related events for the cognitive system.

    Connects to the LearningHub for:
    - Trade outcome recording
    - Reflection triggers
    - Rule learning notifications
    """

    def __init__(self, event_bus: Optional[RedisEventBus] = None):
        self.event_bus = event_bus or get_event_bus()

    def publish_trade_outcome(self, trade: Dict[str, Any]) -> None:
        """Publish a trade outcome for learning."""
        self.event_bus.publish(
            EventType.TRADE_OUTCOME,
            trade,
            source="broker",
        )

    def publish_reflection(self, reflection: Dict[str, Any]) -> None:
        """Publish a reflection insight."""
        self.event_bus.publish(
            EventType.REFLECTION,
            reflection,
            source="reflection_engine",
        )

    def publish_rule_learned(self, rule: Dict[str, Any]) -> None:
        """Publish a newly learned rule."""
        self.event_bus.publish(
            EventType.RULE_LEARNED,
            rule,
            source="semantic_memory",
        )
