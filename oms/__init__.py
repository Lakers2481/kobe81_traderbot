"""
Order Management System (OMS)
=============================

Handles order state tracking and duplicate prevention.

Components:
- OrderRecord: Data structure for order tracking
- IdempotencyStore: Prevents duplicate order submissions
"""

from .order_state import OrderRecord, OrderStatus
from .idempotency_store import IdempotencyStore

__all__ = [
    'OrderRecord',
    'OrderStatus',
    'IdempotencyStore',
]
