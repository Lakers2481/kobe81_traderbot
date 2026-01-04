"""
Kobe Trading System - Messaging Module

Real-time pub/sub messaging using Redis Streams for:
- Market quotes propagation
- Trade signal distribution
- Position updates
- System alerts
"""

from .redis_pubsub import (
    RedisEventBus,
    EventType,
    publish_event,
    subscribe_to_events,
    get_event_bus,
)

__all__ = [
    "RedisEventBus",
    "EventType",
    "publish_event",
    "subscribe_to_events",
    "get_event_bus",
]
