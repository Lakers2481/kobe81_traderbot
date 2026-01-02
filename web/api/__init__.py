"""
Web API Module.

Provides REST API endpoints for external integrations:
- TradingView webhooks
- Signal ingestion queue
- External alerting services
"""

from web.api.webhooks import router as webhooks_router
from web.api.signal_queue import SignalQueue, get_signal_queue

__all__ = [
    "webhooks_router",
    "SignalQueue",
    "get_signal_queue",
]
