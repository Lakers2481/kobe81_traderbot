"""
Yahoo Finance News Scraper - NO API KEY NEEDED
==============================================

Scrapes news headlines and sentiment directly from Yahoo Finance.
100% free, no rate limits (just don't abuse it).

Usage:
    from altdata.yahoo_news_scraper import YahooNewsScraper

    scraper = YahooNewsScraper()
    news = scraper.get_news('AAPL', limit=10)
    sentiment = scraper.get_sentiment_summary('AAPL')
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
import time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article from Yahoo Finance."""
    title: str
    publisher: str
    link: str
    published_at: datetime
    sentiment_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'publisher': self.publisher,
            'link': self.link,
            'published_at': self.published_at.isoformat(),
            'sentiment_score': self.sentiment_score,
        }


class YahooNewsScraper:
    """
    Scrapes news from Yahoo Finance using yfinance library.
    NO API KEY NEEDED - completely free.
    """

    def __init__(self):
        """Initialize the scraper."""
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not installed. Run: pip install yfinance")

        if not VADER_AVAILABLE:
            logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")

        self.analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def get_news(self, symbol: str, limit: int = 20) -> List[NewsArticle]:
        """
        Get recent news for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            limit: Max number of articles (default 20)

        Returns:
            List of NewsArticle objects with sentiment scores
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not available - cannot fetch news")
            return []

        symbol = symbol.upper()

        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news  # This is FREE - no API key needed!

            articles = []
            for item in news_data[:limit]:
                try:
                    # Handle nested content structure (yfinance changed their format)
                    content = item.get('content', item)

                    # Parse timestamp - try multiple fields
                    pub_time = None
                    for time_field in ['providerPublishTime', 'pubDate', 'publishedAt']:
                        if time_field in item:
                            try:
                                pub_time = datetime.fromtimestamp(item[time_field])
                                break
                            except:
                                pass
                        if time_field in content:
                            try:
                                # Handle ISO format
                                pub_time = datetime.fromisoformat(content[time_field].replace('Z', '+00:00'))
                                break
                            except:
                                pass

                    if not pub_time:
                        pub_time = datetime.now()

                    # Extract title, publisher, link from nested structure
                    title = content.get('title', item.get('title', ''))
                    publisher_obj = content.get('provider', item.get('provider', {}))
                    publisher = publisher_obj.get('displayName', 'Unknown') if isinstance(publisher_obj, dict) else str(publisher_obj)

                    canonical_url = content.get('canonicalUrl', item.get('canonicalUrl', {}))
                    link = canonical_url.get('url', item.get('link', '')) if isinstance(canonical_url, dict) else str(canonical_url)

                    article = NewsArticle(
                        title=title,
                        publisher=publisher,
                        link=link,
                        published_at=pub_time,
                    )

                    # Add sentiment analysis
                    if self.analyzer:
                        vader_scores = self.analyzer.polarity_scores(article.title)
                        article.sentiment_score = vader_scores['compound']

                    articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to parse news item: {e}")
                    continue

            logger.info(f"Fetched {len(articles)} news articles for {symbol} from Yahoo Finance")
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []

    def get_sentiment_summary(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get aggregated sentiment summary for a symbol.

        Args:
            symbol: Stock ticker
            limit: Number of articles to analyze

        Returns:
            Dict with sentiment stats
        """
        articles = self.get_news(symbol, limit)

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
            'recent_headlines': [a.title for a in articles[:5]],
        }


def get_yahoo_news_client() -> YahooNewsScraper:
    """Singleton accessor for YahooNewsScraper."""
    global _client_singleton
    if '_client_singleton' not in globals():
        _client_singleton = YahooNewsScraper()
    return _client_singleton


if __name__ == "__main__":
    # Test script
    import sys
    logging.basicConfig(level=logging.INFO)

    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'

    scraper = YahooNewsScraper()
    summary = scraper.get_sentiment_summary(symbol)

    print(f"\nYahoo Finance News for {symbol}:")
    print(f"  Articles: {summary['num_articles']}")
    print(f"  Avg Sentiment: {summary['avg_sentiment']:.3f}")
    print(f"  Label: {summary['sentiment_label']}")
    print(f"  Positive: {summary['positive_count']}, Negative: {summary['negative_count']}, Neutral: {summary['neutral_count']}")

    if summary.get('recent_headlines'):
        print(f"\nRecent Headlines:")
        for i, headline in enumerate(summary['recent_headlines'], 1):
            print(f"  {i}. {headline}")
