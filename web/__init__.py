"""
Kobe Trading System - Web Dashboard Module.

Provides real-time trading dashboard with:
- 5-second WebSocket refresh
- Kill switch monitoring
- Live positions and P&L
- Signal display and ranking
- Market context (VIX, indices)
"""

from .dashboard import app, start_dashboard
from .data_provider import DashboardDataProvider, get_data_provider

__all__ = [
    'app',
    'start_dashboard',
    'DashboardDataProvider',
    'get_data_provider',
]
