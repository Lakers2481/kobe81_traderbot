"""
News Monitoring System for Kobe Trading.

Monitors major market-moving news in real-time using Polygon API.
Integrates with trading system to pause/alert on major events.

Features:
- Real-time news monitoring via Polygon API
- Impact classification (CRITICAL/HIGH/MEDIUM/LOW)
- Auto-pause trading on major events
- Telegram alerts for critical news
- Caching to avoid API rate limits
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NewsImpact(Enum):
    """News impact classification."""
    CRITICAL = "CRITICAL"  # Fed announcements, geopolitical events
    HIGH = "HIGH"          # Earnings, economic data
    MEDIUM = "MEDIUM"      # Company news, analyst ratings
    LOW = "LOW"            # Minor updates


@dataclass
class NewsItem:
    """Structured news item."""
    id: str
    headline: str
    summary: str
    source: str
    published: datetime
    symbols: List[str]
    impact: NewsImpact
    url: Optional[str] = None


class NewsMonitor:
    """
    Monitor major market news using Polygon API.

    Institutional-grade news monitoring with automatic trading pauses
    for critical events.
    """

    POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"

    def __init__(self, trading_system: Any = None, api_key: Optional[str] = None):
        """
        Initialize news monitor.

        Args:
            trading_system: Reference to trading system for pausing
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
        """
        self.trading_system = trading_system
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")

        # Cache settings
        self._cache: List[NewsItem] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

        # News file for persistence
        self.news_file = Path("data/news/latest_news.json")
        self.news_file.parent.mkdir(parents=True, exist_ok=True)

        # Critical keywords that trigger trading pause
        self.critical_keywords = [
            'fed', 'fomc', 'powell', 'interest rate', 'rate hike', 'rate cut',
            'war', 'invasion', 'nuclear', 'attack', 'missile',
            'bank failure', 'default', 'bankruptcy', 'collapse',
            'circuit breaker', 'halt', 'crash', 'flash crash',
            'recession', 'depression', 'crisis'
        ]

        # High impact keywords
        self.high_keywords = [
            'earnings', 'gdp', 'jobs', 'unemployment', 'inflation', 'cpi', 'ppi',
            'retail sales', 'housing', 'manufacturing', 'ism',
            'sec', 'investigation', 'fraud', 'scandal'
        ]

        # Medium impact keywords
        self.medium_keywords = [
            'upgrade', 'downgrade', 'guidance', 'forecast', 'outlook',
            'buyback', 'dividend', 'split', 'merger', 'acquisition'
        ]

    def check_major_news(self, symbols: Optional[List[str]] = None, limit: int = 20) -> List[NewsItem]:
        """
        Check for major market-moving news.

        Args:
            symbols: Filter by specific symbols (optional)
            limit: Maximum number of news items

        Returns:
            List of NewsItem objects with impact classification
        """
        if self._is_cache_valid():
            logger.debug("Returning cached news")
            return self._filter_by_symbols(self._cache, symbols)

        news_items = self._fetch_news_from_polygon(symbols, limit)

        self._cache = news_items
        self._cache_time = datetime.now()

        self._save_news_to_file(news_items)

        critical_news = [n for n in news_items if n.impact == NewsImpact.CRITICAL]
        if critical_news:
            for item in critical_news:
                self.handle_critical_news(item)

        return news_items

    def has_critical_news(self) -> bool:
        """Check if there's any critical news that should pause trading."""
        news = self.check_major_news(limit=10)
        return any(n.impact == NewsImpact.CRITICAL for n in news)

    def _fetch_news_from_polygon(self, symbols: Optional[List[str]] = None, limit: int = 20) -> List[NewsItem]:
        """Fetch news from Polygon API."""
        if not self.api_key:
            logger.warning("No Polygon API key configured - news monitoring disabled")
            return []

        try:
            params = {
                "apiKey": self.api_key,
                "limit": limit,
                "order": "desc",
                "sort": "published_utc"
            }

            if symbols:
                params["ticker"] = ",".join(symbols[:10])

            response = requests.get(self.POLYGON_NEWS_URL, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            news_items = []
            for article in results:
                item = NewsItem(
                    id=article.get("id", ""),
                    headline=article.get("title", ""),
                    summary=article.get("description", "")[:500] if article.get("description") else "",
                    source=article.get("publisher", {}).get("name", "Unknown"),
                    published=self._parse_datetime(article.get("published_utc")),
                    symbols=article.get("tickers", []),
                    impact=self._classify_impact_from_article(article),
                    url=article.get("article_url")
                )
                news_items.append(item)

            logger.info(f"Fetched {len(news_items)} news items from Polygon")
            return news_items

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch news from Polygon: {e}")
            return self._load_news_from_file()
        except Exception as e:
            logger.error(f"Error processing news: {e}")
            return []

    def _classify_impact_from_article(self, article: Dict) -> NewsImpact:
        """Classify news impact from article data."""
        headline = article.get("title", "").lower()
        description = article.get("description", "").lower() if article.get("description") else ""
        combined = f"{headline} {description}"

        if any(kw in combined for kw in self.critical_keywords):
            return NewsImpact.CRITICAL

        if any(kw in combined for kw in self.high_keywords):
            return NewsImpact.HIGH

        if any(kw in combined for kw in self.medium_keywords):
            return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def classify_impact(self, news_item: Any) -> NewsImpact:
        """Classify news impact level."""
        if isinstance(news_item, NewsItem):
            return news_item.impact
        return self._classify_impact_from_article(news_item)

    def handle_critical_news(self, news_item: NewsItem):
        """Handle critical market news - pauses trading and sends alert."""
        logger.critical(f"CRITICAL NEWS DETECTED: {news_item.headline}")

        if self.trading_system and hasattr(self.trading_system, 'pause_trading'):
            self.trading_system.pause_trading(
                reason=f"Critical news: {news_item.headline[:100]}"
            )

        self._send_critical_alert(news_item)
        return True

    def _send_critical_alert(self, news_item: NewsItem):
        """Send Telegram alert for critical news."""
        try:
            from alerts.telegram_alerter import get_alerter

            alerter = get_alerter()
            if alerter.enabled:
                message = (
                    f"<b>[CRITICAL NEWS ALERT]</b>\n\n"
                    f"{news_item.headline}\n\n"
                    f"Source: {news_item.source}\n"
                    f"Symbols: {', '.join(news_item.symbols[:5])}\n"
                    f"Time: {news_item.published.strftime('%Y-%m-%d %H:%M')}\n\n"
                    f"Trading may be paused."
                )
                alerter.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send critical news alert: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache or not self._cache_time:
            return False
        return datetime.now() - self._cache_time < self._cache_ttl

    def _filter_by_symbols(self, news: List[NewsItem], symbols: Optional[List[str]]) -> List[NewsItem]:
        """Filter news by symbols."""
        if not symbols:
            return news
        symbols_set = set(s.upper() for s in symbols)
        return [n for n in news if any(s.upper() in symbols_set for s in n.symbols)]

    def _save_news_to_file(self, news_items: List[NewsItem]):
        """Save news to file for dashboard access."""
        try:
            data = [
                {
                    "id": n.id,
                    "headline": n.headline,
                    "summary": n.summary,
                    "source": n.source,
                    "published": n.published.isoformat() if n.published else None,
                    "symbols": n.symbols,
                    "impact": n.impact.value,
                    "url": n.url
                }
                for n in news_items
            ]
            with open(self.news_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save news to file: {e}")

    def _load_news_from_file(self) -> List[NewsItem]:
        """Load news from cached file."""
        try:
            if self.news_file.exists():
                with open(self.news_file) as f:
                    data = json.load(f)
                return [
                    NewsItem(
                        id=n.get("id", ""),
                        headline=n.get("headline", ""),
                        summary=n.get("summary", ""),
                        source=n.get("source", ""),
                        published=datetime.fromisoformat(n["published"]) if n.get("published") else datetime.now(),
                        symbols=n.get("symbols", []),
                        impact=NewsImpact(n.get("impact", "LOW")),
                        url=n.get("url")
                    )
                    for n in data
                ]
        except Exception as e:
            logger.error(f"Failed to load news from file: {e}")
        return []

    def _parse_datetime(self, dt_str: Optional[str]) -> datetime:
        """Parse datetime string from Polygon."""
        if not dt_str:
            return datetime.now()
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse datetime string '{dt_str}': {e}")
            return datetime.now()

    def get_news_for_dashboard(self, limit: int = 10) -> List[Dict]:
        """Get formatted news for dashboard display."""
        news = self.check_major_news(limit=limit)
        return [
            {
                "headline": n.headline,
                "summary": n.summary[:200] + "..." if len(n.summary) > 200 else n.summary,
                "source": n.source,
                "time": n.published.strftime("%Y-%m-%d %H:%M") if n.published else "",
                "symbols": n.symbols[:3],
                "impact": n.impact.value,
                "impact_color": self._get_impact_color(n.impact),
                "url": n.url
            }
            for n in news
        ]

    def _get_impact_color(self, impact: NewsImpact) -> str:
        """Get color code for impact level."""
        return {
            NewsImpact.CRITICAL: "red",
            NewsImpact.HIGH: "orange",
            NewsImpact.MEDIUM: "yellow",
            NewsImpact.LOW: "gray"
        }.get(impact, "gray")


# Singleton instance
_news_monitor: Optional[NewsMonitor] = None


def get_news_monitor() -> NewsMonitor:
    """Get or create the global news monitor instance."""
    global _news_monitor
    if _news_monitor is None:
        _news_monitor = NewsMonitor()
    return _news_monitor
