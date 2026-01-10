"""
Finnhub News API - FREE Alternative to Polygon News
===================================================

Finnhub offers 60 API calls/minute on their free tier.
Perfect for fetching recent news headlines and sentiment for stocks.

Free Tier Limits:
- 60 calls/minute
- Company news: last 7 days
- Market news: real-time

Get your free API key at: https://finnhub.io/

Usage:
    from altdata.finnhub_news import FinnhubNewsClient

    client = FinnhubNewsClient()
    news = client.fetch_company_news('AAPL', days_back=7)
    sentiment = client.get_sentiment_summary('AAPL')
"""

import logging
import os
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import time

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_RATE_LIMIT = 1.0  # 60 calls/min = 1 call per second (conservative)
_last_request_time: float = 0.0


@dataclass
class FinnhubArticle:
    """Represents a news article from Finnhub."""
    id: int
    headline: str
    summary: str
    source: str
    url: str
    datetime: datetime
    category: str
    related: str  # Symbol
    image: str
    sentiment_score: float = 0.0  # VADER compound score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "datetime": self.datetime.isoformat(),
            "category": self.category,
            "related": self.related,
            "image": self.image,
            "sentiment_score": self.sentiment_score,
        }


class FinnhubNewsClient:
    """
    Fetches company news from Finnhub (free tier).
    Performs VADER sentiment analysis on headlines/summaries.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub News client.

        Args:
            api_key: Finnhub API key (or uses FINNHUB_API_KEY from env)
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY', '')
        self.analyzer = SentimentIntensityAnalyzer()

        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not found. News fetching will fail.")
            logger.info("Get free key at: https://finnhub.io/register")

    def _rate_limit(self):
        """Enforce rate limit (60 calls/min = 1 per second)."""
        global _last_request_time
        elapsed = time.time() - _last_request_time
        if elapsed < FINNHUB_RATE_LIMIT:
            time.sleep(FINNHUB_RATE_LIMIT - elapsed)
        _last_request_time = time.time()

    def fetch_company_news(self, symbol: str, days_back: int = 7) -> List[FinnhubArticle]:
        """
        Fetch recent company news for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            days_back: Number of days to look back (default: 7)

        Returns:
            List of FinnhubArticle with sentiment scores
        """
        if not self.api_key:
            logger.error("Cannot fetch news without FINNHUB_API_KEY")
            return []

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Format dates as YYYY-MM-DD
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')

        # Rate limit
        self._rate_limit()

        # API request
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date,
            'to': to_date,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse articles
            articles = []
            for item in data[:20]:  # Limit to 20 most recent
                try:
                    article = FinnhubArticle(
                        id=item.get('id', 0),
                        headline=item.get('headline', ''),
                        summary=item.get('summary', ''),
                        source=item.get('source', ''),
                        url=item.get('url', ''),
                        datetime=datetime.fromtimestamp(item.get('datetime', 0)),
                        category=item.get('category', ''),
                        related=item.get('related', symbol),
                        image=item.get('image', ''),
                    )

                    # Perform sentiment analysis
                    text_to_analyze = f"{article.headline}. {article.summary}"
                    vader_scores = self.analyzer.polarity_scores(text_to_analyze)
                    article.sentiment_score = vader_scores['compound']

                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue

            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
            return []

    def get_sentiment_summary(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get aggregated sentiment summary for a symbol.

        Args:
            symbol: Stock ticker
            days_back: Number of days to analyze

        Returns:
            Dict with sentiment stats: {
                'symbol': str,
                'num_articles': int,
                'avg_sentiment': float,  # -1 to +1
                'sentiment_label': str,  # 'positive', 'neutral', 'negative'
                'positive_count': int,
                'negative_count': int,
                'neutral_count': int,
            }
        """
        articles = self.fetch_company_news(symbol, days_back)

        if not articles:
            return {
                'symbol': symbol,
                'num_articles': 0,
                'avg_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
            }

        # Calculate stats
        sentiments = [a.sentiment_score for a in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)

        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count

        # Classify overall sentiment
        if avg_sentiment > 0.15:
            sentiment_label = 'positive'
        elif avg_sentiment < -0.15:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        return {
            'symbol': symbol,
            'num_articles': len(articles),
            'avg_sentiment': avg_sentiment,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'days_analyzed': days_back,
        }


def get_finnhub_client() -> FinnhubNewsClient:
    """Singleton accessor for FinnhubNewsClient."""
    global _client_singleton
    if '_client_singleton' not in globals():
        _client_singleton = FinnhubNewsClient()
    return _client_singleton


if __name__ == "__main__":
    # Test script
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = 'AAPL'

    client = FinnhubNewsClient()
    summary = client.get_sentiment_summary(symbol, days_back=7)

    print(f"\nNews Sentiment for {symbol} (last 7 days):")
    print(f"  Articles: {summary['num_articles']}")
    print(f"  Avg Sentiment: {summary['avg_sentiment']:.3f}")
    print(f"  Label: {summary['sentiment_label']}")
    print(f"  Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")
