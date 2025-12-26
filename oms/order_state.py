from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    VETOED = "VETOED"
    CLOSED = "CLOSED"


@dataclass
class OrderRecord:
    decision_id: str
    signal_id: str
    symbol: str
    side: str  # BUY/SELL
    qty: int
    limit_price: float
    tif: str  # IOC
    order_type: str  # IOC_LIMIT
    idempotency_key: str
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    last_update: Optional[datetime] = None
    notes: Optional[str] = None

