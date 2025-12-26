"""
Kobe Trading System - News Monitoring Module.

Provides real-time market news monitoring with:
- Polygon API integration
- Impact classification (CRITICAL/HIGH/MEDIUM/LOW)
- Auto-pause trading on critical events
- Telegram alerts for major news
"""

from .news_monitor import NewsMonitor, NewsItem, NewsImpact, get_news_monitor

__all__ = [
    'NewsMonitor',
    'NewsItem',
    'NewsImpact',
    'get_news_monitor',
]
