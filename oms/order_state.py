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
    FAILED = "FAILED" # Added for clarity in execution


@dataclass
class OrderRecord:
    decision_id: str
    signal_id: str
    symbol: str
    side: str  # BUY/SELL
    qty: int
    limit_price: float
    tif: str  # IOC (Time In Force)
    order_type: str  # IOC_LIMIT (Order Type)
    idempotency_key: str
    created_at: datetime

    # New fields for Intelligent Execution and TCA
    execution_id: Optional[str] = None # Link to intelligent execution attempt
    entry_price_decision: Optional[float] = None # Price from AI decision, for TCA benchmark
    strategy_used: Optional[str] = None # Strategy that generated the signal

    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: Optional[str] = None
    last_update: Optional[datetime] = None
    notes: Optional[str] = None
    fill_price: Optional[float] = None  # Actual fill price from broker
    filled_qty: Optional[int] = None    # Number of shares filled
    
    def update_status(self, new_status: OrderStatus, message: Optional[str] = None, filled_qty: Optional[int] = None, fill_price: Optional[float] = None):
        """Updates the order's status and relevant fields."""
        self.status = new_status
        self.last_update = datetime.now()
        if message:
            self.notes = message
        if filled_qty is not None:
            self.filled_qty = filled_qty
        if fill_price is not None:
            self.fill_price = fill_price
